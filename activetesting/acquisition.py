"""Implement acquisition functions."""

from copy import deepcopy
import logging
import warnings
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch._C import Value
from scipy.special import softmax
from scipy.stats import multivariate_normal

from activetesting.models import BayesQuadModel
from bayesquad.batch_selection import (
    select_batch, LOCAL_PENALISATION, KRIGING_BELIEVER, KRIGING_OPTIMIST)

from activetesting.risk_estimators import QuadratureRiskEstimator
from activetesting.models import (
    SVMClassifier, RandomForestClassifier, GPClassifier,
    GaussianProcessRegressor, RandomDirectionRandomForestClassifier,
    RadialBNN, make_efficient
    )
from activetesting.loss import CrossEntropyLoss
from activetesting.utils.utils import t2n, TwoComponentMixtureDistribution


class AcquisitionFunction:
    """Acquisition function is its own class.

    In the beginning this may seem like overkill, but remember that our
    acquisition function will likely have a powerfull substitute model.

    Implement get_next_point
    """
    def __init__(self, cfg_run, dataset):
        self.cfg, run = cfg_run
        logging.info(f'**Initialising acquisition {self.__class__}.')

        self.dataset = dataset
        # keep track of acquisition weights
        self.weights = np.array([])

        if self.cfg.animate and run < self.cfg.animate_until:
            self.all_pmfs = list()
        else:
            self.all_pmfs = None

        self.counter = 0

        if self.cfg.lazy_save:

            if L := self.cfg.get('lazy_save_schedule', False):
                L = list(L)
            else:
                L = list(range(1000))
                L += list(range(int(1e3), int(1e4), 500))
                L += list(range(int(1e4), int(10e4), int(1e3)))

            self.lazy_list = L

        # For model selection hot-patching.
        self.externally_controlled = False
        self.ext_test_idx = None
        self.ext_pmf_idx = None

    @staticmethod
    def acquire():
        raise NotImplementedError

    def check_save(self, off=0):
        if self.all_pmfs is None:
            return False
        if self.cfg.lazy_save and (self.counter - off in self.lazy_list):
            return True
        else:
            return False

        return True

    def sample_pmf(self, pmf):
        """Sample from pmf."""

        if isinstance(pmf, float) or (len(pmf) == 1):
            # Always choose last datum
            pmf = [1]

        if self.externally_controlled:
            idx = self.ext_pmf_idx
            test_idx = self.ext_test_idx

        else:
            # print(self.cfg['sample'])
            if self.cfg['sample']:

                try:
                    # we don't want to have normalised over pool for the real
                    # importance sampling.. normalise again internally
                    pmf = np.array(pmf)
                    pmf_sample = pmf/pmf.sum()
                    # this is one-hot over all remaining test data
                    sample = np.random.multinomial(1, pmf_sample)
                except Exception as e:
                    logging.info(e)
                    logging.info(f'This was pmf {pmf}')
                    raise ValueError

                # idx in test_remaining
                idx = np.where(sample)[0][0]
            else:
                idx = np.argmax(pmf)

            # get index of chosen test datum
            test_idx = self.dataset.test_remaining[idx]

        # get value of acquisition function at that index
        self.weights = np.append(
            self.weights, pmf[idx])

        if self.check_save():
            self.all_pmfs.append(dict(
                idx=idx,
                test_idx=test_idx,
                pmf=pmf,
                remaining=self.dataset.test_remaining,
                observed=self.dataset.test_observed))

        self.counter += 1
        return test_idx, idx

    @staticmethod
    def safe_normalise(pmf):
        """If loss is 0, we want to sample uniform and avoid nans."""

        if (Σ := pmf.sum()) != 0:
            pmf /= Σ
        else:
            pmf = np.ones(len(pmf))/len(pmf)

        return pmf


class RandomAcquisition(AcquisitionFunction):
    def __init__(self, cfg, dataset, *args, **kwargs):
        super().__init__(cfg, dataset)

    def acquire(self, *args, **kwargs):
        n_remaining = len(self.dataset.test_remaining)
        pmf = np.ones(n_remaining)/n_remaining
        return self.sample_pmf(pmf)


class TrueLossAcquisition(AcquisitionFunction):
    def __init__(self, cfg, dataset, true_loss_vals, *args, **kwargs):
        super().__init__(cfg, dataset)

        # make sure indexes are aligned
        if (dataset.N < dataset.y.size) and dataset.cfg.get('with_unseen'):
            N = dataset.y.size
        else:
            N = dataset.N

        self.true_loss = np.zeros(N)
        self.true_loss[self.dataset.test_idxs] = true_loss_vals

    def acquire(self, *args, **kwargs):
        """Sample according to true loss dist."""

        pmf = self.true_loss[self.dataset.test_remaining]

        pmf = self.safe_normalise(pmf)

        return self.sample_pmf(pmf)


class DistanceBasedAcquisition(AcquisitionFunction):
    def __init__(self, cfg, dataset, *args, **kwargs):
        super().__init__(cfg, dataset)

    def acquire(self, *args, **kwargs):
        """Sample according to distance to previously sampled points."""
        remaining_idx = self.dataset.test_remaining
        observed_idx = self.dataset.test_observed

        # First test index sampled at random
        if observed_idx.size == 0:
            N = len(self.dataset.test_idxs)
            pmf = np.ones(N) / N

        else:
            # For each point in remaining
            # calculate distance to all points in observed
            remaining = self.dataset.x[remaining_idx]
            observed = self.dataset.x[observed_idx]

            # broadcasting to get all paired differences
            d = remaining[:, np.newaxis, :] - observed
            d = d**2
            # sum over feature dimension
            d = d.sum(-1)
            # sqrt to get distance
            d = np.sqrt(d)
            # mean over other pairs
            distances = d.mean(1)

            # Constract PDF via softmax
            pmf = softmax(distances)

        return self.sample_pmf(pmf)


# --- Acquisition Functions Based on Expected Loss

class _LossAcquisitionBase(AcquisitionFunction):
    def __init__(self, cfg, dataset, model):

        super().__init__(cfg, dataset)

        # also save original model
        self.model = model

    def acquire(self, *args, **kwargs):
        # predict + std for both models on all remaining test points
        remaining_idxs = self.dataset.test_remaining
        remaining_data = self.dataset.x[remaining_idxs]

        # build expected loss
        expected_loss = self.expected_loss(remaining_data, remaining_idxs)

        if self.cfg['sample'] and (expected_loss < 0).sum() > 0:
            # Log-lik can be negative.
            # Make all values positive.
            # Alternatively could set <0 values to 0.
            expected_loss += np.abs(expected_loss.min())

        if np.any(np.isnan(expected_loss)):
            logging.info(
                'Found NaN values in expected loss, replacing with 0.')
            logging.info(f'{expected_loss}')
            expected_loss = np.nan_to_num(expected_loss, nan=0)

        if not (expected_loss.sum() == 0):
            expected_loss /= expected_loss.sum()

        if self.cfg.get('uniform_clip', False):
            # clip all values less than 10 percent of uniform propability
            p = self.cfg['uniform_clip_val']
            expected_loss = np.maximum(p * 1/expected_loss.size, expected_loss)
            expected_loss /= expected_loss.sum()

        if self.cfg.get('defensive', False):
            a = self.cfg['defensive_val']
            n = len(expected_loss)
            expected_loss = (
                a * np.ones(n)/n) + (1-a) * expected_loss
            # small numerical inaccuracies
            expected_loss /= expected_loss.sum()

        if np.all(np.isclose(expected_loss, 0)):
            expected_loss = np.ones_like(expected_loss)
            expected_loss /= len(expected_loss)

        return self.sample_pmf(expected_loss)

    def get_aleatoric(self):
        if getattr(self.dataset.cfg, 'expose_aleatoric', True):
            ret = getattr(self.dataset, 'aleatoric', 0)
        else:
            ret = 0

        return ret


class GPAcquisitionUncertainty(_LossAcquisitionBase):
    # warning = (
    #     'GPAcquisitionUncertainty is currently only appropriate if '
    #     'the aleatoric uncertainty is 0.')

    def __init__(self, cfg, dataset, model, **kwargs):

        super().__init__(cfg, dataset, model)

    def expected_loss(self, remaining_data, remaining_idxs):

        mu, std = self.model.predict(
            remaining_data, idxs=remaining_idxs, return_std=True)

        aleatoric = self.get_aleatoric()

        return std**2 + aleatoric**2


class SawadeAcquisition(_LossAcquisitionBase):
    # warning = (
    #     'GPAcquisitionUncertainty is currently only appropriate if '
    #     'the aleatoric uncertainty is 0.')

    def __init__(self, cfg, dataset, model, model_cfg, **kwargs):

        _cfg = OmegaConf.merge(
            OmegaConf.structured(cfg[0]),
            OmegaConf.structured(model_cfg.acquisition))
        cfg = [_cfg, cfg[1]]

        super().__init__(cfg, dataset, model)

        # acquisition function does not change for Sawade et al
        # can precompute entirely
        self.proxy_risk = self.get_proxy_risk()

    def expected_loss(self, remaining_data, remaining_idxs):
        """Precalculate acquisition function from Sawade et al."""

        # print(f'Calc. expected loss at {len(remaining_idxs)} entries.')

        # model predictions
        mu_m, std_m = self.model.predict(
            remaining_data, idxs=remaining_idxs, return_std=True)

        aleatoric = self.get_aleatoric()
        std_m = np.sqrt(std_m**2 + aleatoric**2)

        proxy_risk = self.proxy_risk

        # calculate acquisition function
        q = np.sqrt((3 * std_m**2 - 2 * proxy_risk) * std_m**2 + proxy_risk**2)

        # note that if proxy_risk = std_m**2
        # then q propto std_m**2 directly

        return q

    def get_proxy_risk(self):
        """Approximate expected model risk from model predictions.

        This is identical to the variance MSE = Bias^2 + Variance = Variance
        b/c posterior samples agree with mean model prediction on avg per def.
        """

        test_idxs = self.dataset.test_idxs
        test_data = self.dataset.x[test_idxs]

        # print(f'Calc. proxy risk at {len(test_idxs)} entries.')

        # model predictions
        mu_m, std_m = self.model.predict(
            test_data, idxs=test_idxs, return_std=True)

        if self.cfg.get('sample_proxy_risk', False):
            # (I actually need to get to the sklearn model, which I usually
            # don't. So this is a bit ugly)
            samples = self.model.model.model.sample_y(
                test_data, n_samples=10000)

            # calculate proxy risk (MC estimate of integral)
            proxy_risk = ((samples - mu_m[:, np.newaxis])**2).mean(1).mean(0)

        else:
            # (btw not quite sure why Sawade et al. write the integral here)
            # pretty sure the integral in the proxy risk is equal to std_m**2
            # making the proxy risk exactly equal to \sum_x std_m(x)**2
            proxy_risk = (std_m**2).mean()

        return proxy_risk


class SawadeOptimalAcquisition(_LossAcquisitionBase):
    # warning = (
    #     'GPAcquisitionUncertainty is currently only appropriate if '
    #     'the aleatoric uncertainty is 0.')

    def __init__(self, cfg, dataset, model, model_cfg, true_loss_vals,
                 **kwargs):

        _cfg = OmegaConf.merge(
            OmegaConf.structured(cfg[0]),
            OmegaConf.structured(model_cfg.acquisition))
        cfg = [_cfg, cfg[1]]

        super().__init__(cfg, dataset, model)

        # acquisition function does not change for Sawade et al
        # can precompute entirely
        self._expected_loss = self._expected_loss_precompute(true_loss_vals)

        if self.get_aleatoric() != 0:
            raise ValueError(
                'Derivation for Optimal Acq. does not hold for noisy data!')

    def _expected_loss_precompute(self, true_loss_vals):
        """Precalculate optimal acquisition function from Sawade et al."""

        test_idxs = self.dataset.test_idxs
        test_data = self.dataset.x[test_idxs]

        # model predictions
        mu_m = self.model.predict(
            test_data, idxs=test_idxs, return_std=False)

        # calculate optimal acquisition function
        R = true_loss_vals.mean()
        M = true_loss_vals.size

        q = 1/M * np.sqrt((true_loss_vals - R)**2)

        # align indices
        acquisition = np.zeros(self.dataset.N)
        acquisition[self.dataset.test_idxs] = q

        return acquisition

    def expected_loss(self, remaining_data, remaining_idxs):
        # print(f"expected_loss() called with {len(remaining_data)} rem data")

        return self._expected_loss[remaining_idxs]


class BNNClassifierAcquisitionMI(_LossAcquisitionBase):
    # warning = (
    #     'GPAcquisitionUncertainty is currently only appropriate if '
    #     'the aleatoric uncertainty is 0.')

    def __init__(self, cfg, dataset, model, **kwargs):

        super().__init__(cfg, dataset, model)

    def expected_loss(self, remaining_data, remaining_idxs):

        mutual_information = self.model.predict(
            remaining_data, idxs=remaining_idxs, mutual_info=True)

        return mutual_information


class BNNClassifierAcquisitionRiskCovariance(_LossAcquisitionBase):
    # warning = (
    #     'GPAcquisitionUncertainty is currently only appropriate if '
    #     'the aleatoric uncertainty is 0.')

    def __init__(self, cfg, dataset, model, **kwargs):

        super().__init__(cfg, dataset, model)

    def expected_loss(self, remaining_data, remaining_idxs):

        preds = self.model.model.predict(
            remaining_data, idxs=remaining_idxs,
            log_sum_exp=False, n_samples=100)

        mean_preds = preds.mean(1, keepdims=True)

        return covariance(preds, mean_preds, self.dataset, self.model)


class _SurrogateAcquisitionBase(_LossAcquisitionBase):
    def __init__(self, cfg_run, dataset, model, SurrModel, surr_cfg):
        logging.info(
            f'**Initialising {self.__class__} with name {model.cfg["name"]}.')
        logging.info(f'Config {surr_cfg}')

        if surr_cfg.get('acquisition', False):
            # the surrogate acquisition can specialise the
            # acquisition configs. this mostly affects clipping behaviour
            cfg = OmegaConf.merge(
                OmegaConf.structured(cfg_run[0]),
                OmegaConf.structured(surr_cfg.acquisition))
            cfg_run = [cfg, cfg_run[1]]

        super().__init__(cfg_run, dataset, model)

        if surr_cfg.get('copy_main', False):
            self.surr_cfg = deepcopy(model.cfg)
        else:
            self.surr_cfg = deepcopy(surr_cfg)

        self.surr_class = SurrModel

        self.surr_model = SurrModel(self.surr_cfg)
        if len(self.dataset.train_idxs) > 0:
            self.surr_model.fit(*self.dataset.total_observed)

        if self.surr_cfg.get('efficient', False):
            # make efficient predictions on remaining test data
            self.surr_model = make_efficient(self.surr_model, self.dataset)

        if surr_cfg.get('lazy', False):
            if (s := self.surr_cfg.get('lazy_schedule', False)) is not False:
                retrain = list(s)
            else:
                retrain = [5]
                retrain += list(range(10, 50, 10))
                retrain += [50]
                retrain += list(range(100, 1000, 150))
                retrain += list(range(1000, 10000, 2000))
                retrain += list(range(int(10e3), int(100e3), int(10e3)))

            # always remove 0, since we train at it 0
            # self.retrain = list(set(retrain) - {0})
            self.retrain = list(set(retrain))
            self.update_surrogate = self.lazy_update_surrogate
        else:
            self.update_surrogate = self.vanilla_update_surrogate

        if risk := self.surr_cfg.get('weights', False):
            assert self.dataset.cfg.test_proportion == 1, (
                'No weights for train data available.')
            assert not self.surr_cfg.get('on_train_only', False), (
                'No weights for train data available.')

            from activetesting.utils.maps import risk_estimator
            self.get_weights = risk_estimator[risk].get_weights

    def get_weight_kwargs(self):
        if self.surr_cfg.get('weights', False):
            kwargs = dict(sample_weight=self.get_weights(
                self.weights, self.dataset.N))
        else:
            kwargs = dict()
        return kwargs

    def vanilla_update_surrogate(self):
        # logging.info('calling vanilla_update_surrogate')
        is_fit = getattr(self.surr_model, 'is_fit', False)
        keep_params = self.surr_cfg.get('init_with_last_params', False)
        if keep_params and is_fit:
            params = self.surr_model.get_params()
        else:
            params = None

        self.surr_model = self.surr_class(self.surr_cfg, params=params)

        if self.surr_cfg.get('on_train_only', False):
            self.surr_model.fit(*self.dataset.train_data)
        else:
            # fit on all observed data
            self.surr_model.fit(
                *self.dataset.total_observed,
                **self.get_weight_kwargs())

        if self.surr_cfg.get('efficient', False):
            # make efficient predictions on remaining test data
            self.surr_model = make_efficient(self.surr_model, self.dataset)

    def lazy_update_surrogate(self):
        # logging.info('calling lazy_update_surrogate')
        if self.counter in self.retrain:
            self.vanilla_update_surrogate()
            # logging.info(
            #     f'>> Triggering lazy refit for {self.__class__}/'
            #     f'{self.surr_model.cfg["name"]} '
            #     f'of surrogate in it {self.counter}.')

            # logging.info(
                # f'>> Finish lazy refit of surrogate in it {self.counter}.')

    def acquire(self, update_surrogate=True):

        if update_surrogate:
            self.update_surrogate()

        return super().acquire()


class _SelfSurrogateAcquisitionBase(_SurrogateAcquisitionBase):

    def __init__(self, cfg, dataset, model, model_cfg):
        from activetesting.utils.maps import model as model_maps
        SurrModel = model_maps[model.cfg['name']]
        super().__init__(cfg, dataset, model, SurrModel, model_cfg)


class SelfSurrogateAcquisitionEntropy(
        _SelfSurrogateAcquisitionBase):

    def __init__(self, cfg, dataset, model, model_cfg, *args, **kwargs):
        super().__init__(cfg, dataset, model, model_cfg)

    def expected_loss(self, remaining_data, remaining_idxs):
        return entropy_loss(
            remaining_data, remaining_idxs, self.model, self.surr_model)


class SelfSurrogateAcquisitionSurrogateEntropy(
        _SelfSurrogateAcquisitionBase):

    def __init__(self, cfg, dataset, model, model_cfg, *args, **kwargs):
        super().__init__(cfg, dataset, model, model_cfg)

    def expected_loss(self, remaining_data, remaining_idxs):
        return entropy_loss(
            remaining_data, remaining_idxs, self.surr_model, self.surr_model)


class SelfSurrogateAcquisitionSurrogateEntropyPlusExpectedLoss(
        _SelfSurrogateAcquisitionBase):

    def __init__(self, cfg, dataset, model, model_cfg, *args, **kwargs):
        super().__init__(cfg, dataset, model, model_cfg)

    def expected_loss(self, remaining_data, remaining_idxs):
        predictive_entropy = entropy_loss(
            remaining_data, remaining_idxs, self.surr_model, self.surr_model)
        expected_loss = entropy_loss(
            remaining_data, remaining_idxs, self.model, self.surr_model)

        return 0.5 * (predictive_entropy + expected_loss)


class SelfSurrogateAcquisitionSurrogateMI(_SelfSurrogateAcquisitionBase):
    # warning = (
    #     'GPAcquisitionUncertainty is currently only appropriate if '
    #     'the aleatoric uncertainty is 0.')

    def __init__(self, cfg, dataset, model, model_cfg, *args, **kwargs):
        super().__init__(cfg, dataset, model, model_cfg)

    def expected_loss(self, remaining_data, remaining_idxs):

        mutual_information = self.surr_model.predict(
            remaining_data, idxs=remaining_idxs, mutual_info=True)

        return mutual_information


class SelfSurrogateBNNClassifierAcquisitionRiskCovariance(
        _SelfSurrogateAcquisitionBase):
    """Acquire by approximating MI(L_i, R) with cov(L_i, R) from samples.
    """

    def __init__(self, cfg, dataset, model, model_cfg, *args, **kwargs):
        super().__init__(cfg, dataset, model, model_cfg)

        # this needs to estimate risk. probably easiest to just give this
        # function its own risk estimator

    def expected_loss(self, remaining_data, remaining_idxs):

        # sample predictions from model
        # (this does not have/need efficient support)
        # these are correctly normalised probabilities
        # (n_data, n_samples, n_classes)
        # outcomes.sum(-1) == 1

        preds = self.model.model.predict(
            remaining_data, idxs=remaining_idxs,
            log_sum_exp=False, n_samples=100)

        # use surrogate to predict truth
        mean_preds = self.surr_model.predict(
                remaining_data, idxs=remaining_idxs)
        mean_preds = mean_preds[:, np.newaxis, :]
        # mean_preds = preds.mean(1, keepdims=True)

        return covariance(preds, mean_preds, self.dataset, self.model)


class SelfSurrogateAcquisitionAccuracy(
        _SelfSurrogateAcquisitionBase):

    def __init__(self, cfg, dataset, model, model_cfg, *args, **kwargs):
        super().__init__(cfg, dataset, model, model_cfg)

    def expected_loss(self, remaining_data, remaining_idxs):
        return accuracy_loss(
            remaining_data, remaining_idxs, self.model, self.surr_model)


class _AnySurrogateAcquisitionBase(_SurrogateAcquisitionBase):

    def __init__(self, cfg, dataset, model, model_cfg):
        from activetesting.utils.maps import model as model_maps
        SurrModel = model_maps[model_cfg.name]
        super().__init__(cfg, dataset, model, SurrModel, model_cfg)


class AnySurrogateRandomAcquisition(_AnySurrogateAcquisitionBase):
    """Train a surrogate but don't use it for acquisition.

    Instead acquire randomly.
    Surrogate may still be used, e.g. for calling predict in QuadratureRisk-
    Estimator.

    .. warning: Not sure if this will work with small data. (Because currently
        I am just re-using ensemble configs for this estimator. But there is no
        saving-loading for the ensembles anyways, right? So maybe this will
        work? (But it will be very inefficient!) Not sure.) Probably it would
        be best to restructure the code to have surrogates, acquisitions, and
        risk estimates separately. And specify all combinations of these
        explicitly. Oh well.
    """
    def __init__(self, cfg, dataset, model, model_cfg, *args, **kwargs):
        super().__init__(cfg, dataset, model, model_cfg)

    def acquire(self, update_surrogate=True):

        # keep updating surrogate
        if update_surrogate:
            self.update_surrogate()

        # but actually perform random acquisition
        n_remaining = len(self.dataset.test_remaining)
        pmf = np.ones(n_remaining)/n_remaining
        return self.sample_pmf(pmf)


class AnySurrogateDistanceBasedAcquisition(_AnySurrogateAcquisitionBase):
    def __init__(self, cfg, dataset, model, model_cfg, *args, **kwargs):
        super().__init__(cfg, dataset, model, model_cfg)

    def acquire(self, update_surrogate=True):
        """Sample according to distance to previously sampled points."""

        # keep updating surrogate
        if update_surrogate:
            self.update_surrogate()

        remaining_idx = self.dataset.test_remaining
        observed_idx = self.dataset.test_observed

        # First test index sampled at random
        if observed_idx.size == 0:
            N = len(self.dataset.test_idxs)
            pmf = np.ones(N) / N

        else:
            # For each point in remaining
            # calculate distance to all points in observed
            remaining = self.dataset.x[remaining_idx]
            observed = self.dataset.x[observed_idx]

            # broadcasting to get all paired differences
            d = remaining[:, np.newaxis, :] - observed
            d = d**2
            # sum over feature dimension
            d = d.sum(-1)
            # sqrt to get distance
            d = np.sqrt(d)
            # mean over other pairs
            distances = d.mean(1)

            # Constract PDF via softmax
            pmf = softmax(distances)

        return self.sample_pmf(pmf)


class AnySurrogateAcquisitionEntropy(
        _AnySurrogateAcquisitionBase):

    def __init__(self, cfg, dataset, model, model_cfg, *args, **kwargs):
        super().__init__(cfg, dataset, model, model_cfg)

    def expected_loss(self, remaining_data, remaining_idxs):
        return entropy_loss(
            remaining_data, remaining_idxs, self.model, self.surr_model,
            cfg=self.cfg)


class AnySurrogateBayesQuadAcquisition(_AnySurrogateAcquisitionBase):
    def __init__(self, cfg, dataset, model, model_cfg, *args, **kwargs):
        super().__init__(cfg, dataset, model, model_cfg)

    def expected_loss(self, remaining_data, remaining_idxs):

        if len(self.dataset.test_observed) == 0:
            return np.ones(len(remaining_data))/len(remaining_data)

        locs = select_batch(
            self.surr_model.model, batch_size=1,
            batch_method=LOCAL_PENALISATION
            # batch_method=KRIGING_BELIEVER
            # batch_method=KRIGING_OPTIMIST
            )

        diffs = self.find_match(locs)[:, 0]
        diffs = softmax(-diffs)

        return diffs

    def find_match(self, locs):
        rem_idx = self.dataset.test_remaining
        rem = self.dataset.x[rem_idx]
        locs = np.array(locs)
        diffs = ((rem[:, np.newaxis] - locs[np.newaxis, :])**2).sum(-1)
        diffs = np.sqrt(diffs)
        return diffs


class AnySurrogateBayesQuadSampleFromPDFDirectly(_AnySurrogateAcquisitionBase):

    def __init__(self, cfg, dataset, model, model_cfg, *args, **kwargs):
        super().__init__(cfg, dataset, model, model_cfg)

    def acquire(self, *args, update_surrogate=False, **kwargs):

        if update_surrogate:
            self.update_surrogate()

        test_idx = self.dataset.test_idxs[len(self.dataset.test_observed)]

        if len(self.dataset.test_observed) == 0:
            # just sample from p(x)
            x = self.dataset.p.rvs(1)

        else:
            locs = select_batch(
                self.surr_model.model, batch_size=1,
                batch_method=LOCAL_PENALISATION
                # batch_method=KRIGING_BELIEVER
                # batch_method=KRIGING_OPTIMIST
                )

            x = locs[0]

        self.dataset.x[test_idx] = x
        self.dataset.y[test_idx] = self.dataset.f.pdf(x)

        idx, pmf, weight = 1e19, 1e19, 1e19

        self.weights = np.append(
            self.weights, weight)

        if self.check_save():
            self.all_pmfs.append(dict(
                idx=idx,  # I don't think this is used anywhere
                test_idx=test_idx,
                pmf=pmf,
                remaining=self.dataset.test_remaining,
                observed=self.dataset.test_observed))

        self.counter += 1

        return test_idx, idx


class AnySurrogateAcquisitionValue(
        _AnySurrogateAcquisitionBase):

    def __init__(self, cfg, dataset, model, model_cfg, *args, **kwargs):
        super().__init__(cfg, dataset, model, model_cfg)

    def expected_loss(self, remaining_data, remaining_idxs):
        return self.surr_model.predict(remaining_data, idxs=remaining_idxs)


class AnySurrogateAcquisitionValuePDF(
        _AnySurrogateAcquisitionBase):
    """Importance sampling acquisition.

    Old Version that still relies on a pool.
    """

    def __init__(self, cfg, dataset, model, model_cfg, *args, **kwargs):
        super().__init__(cfg, dataset, model, model_cfg)

    def acquire(self, *args, **kwargs):
        # predict + std for both models on all remaining test points
        remaining_idxs = self.dataset.test_remaining
        remaining_data = self.dataset.x[remaining_idxs]

        fhat = self.surr_model.predict(
            remaining_data, idxs=remaining_idxs)

        p = self.dataset.p.pdf(remaining_data)

        if not self.surr_model.is_fit:
            return self.sample_pmf(p)

        dim = self.dataset.cfg['creation_args']['dim']
        fhat_cov = np.eye(dim) * (torch.exp(self.surr_model.model.log_stds)**2
            ).detach().cpu().numpy()

        fhat_mean = self.surr_model.model.means.detach().cpu().numpy()

        d = self.dataset
        norm = multivariate_normal(
            mean=d.p.mean, cov=d.p.cov+fhat_cov).pdf(fhat_mean)

        q = fhat * p / norm

        if self.cfg.get('defensive', False):
            a = self.cfg['defensive_val']
            q = a * p + (1-a) * q

        return self.sample_pmf(q)


class SamplePDFAcquisition(AcquisitionFunction):
    def __init__(self, cfg, dataset, *args, **kwargs):
        super().__init__(cfg, dataset, *args, **kwargs)

    def sample_pdf(self, p, q=None):

        # sample new point location directly from pdf
        if q is None:
            x = p.rvs(1)
            weight = 1
        else:
            x = q.rvs(1)
            weight = p.pdf(x) / q.pdf(x)
            # x = q.rvs(100)
            # a = p.pdf(x) / q.pdf(x)
            # a.min(), a.max(), a.mean(), a.std()
            # I've checked that maximum weight is 1/a


        # kind of cheaty: hotpatch the data by adding the new observations
        # on the fly --> this allows me to do non-pool-based
        test_idx = self.dataset.test_idxs[len(self.dataset.test_observed)]
        self.dataset.x[test_idx] = x
        self.dataset.y[test_idx] = self.dataset.y_at_any(x)

        idx, pmf = 1e19, 1e19

        self.weights = np.append(
            self.weights, weight)

        if self.check_save():
            self.all_pmfs.append(dict(
                idx=idx,  # I don't think this is used anywhere
                test_idx=test_idx,
                pmf=pmf,
                remaining=self.dataset.test_remaining,
                observed=self.dataset.test_observed))

        self.counter += 1

        return test_idx, idx


class RandomAcquisitionSampleFromPDFDirectly(
        SamplePDFAcquisition):
    """Importance sampling acquisition.

    New Version that is entirely pool free.
    """

    def __init__(self, cfg, dataset, *args, **kwargs):
        super().__init__(cfg, dataset)

    def acquire(self, *args, **kwargs):
        return self.sample_pdf(self.dataset.p)


class AnySurrogateAcquisitionValueSampleFromPDFDirectly(
        SamplePDFAcquisition, _AnySurrogateAcquisitionBase):
    """Importance sampling acquisition.

    New Version that is entirely pool free.
    """

    def __init__(self, cfg, dataset, model, model_cfg, *args, **kwargs):
        # inits SamplePDFAcquisition first then anysurrogate. is mixin class 
        super().__init__(cfg, dataset, model, model_cfg)

    def acquire(self, *args, update_surrogate=False, **kwargs):
        # predict + std for both models on all remaining test points
        # I've moved to updating the surrogate *after* acquisition
        # because this is fairer to the ASMC risk estimates
        # but for legacy, I'll include an update option here
        # we're no longer using the acquire from _AnySurrogateAcquisitionBase
        # because we redefine acquire here, so need to copy this
        if update_surrogate:
            self.update_surrogate()

        p = self.dataset.p

        if not self.surr_model.is_fit:
            return self.sample_pdf(p)

        if self.dataset.__class__.__name__ == 'DoubleGaussianDataset':

            # now need to build q distribution that is correctly normalised

            dim = self.dataset.cfg['creation_args']['dim']

            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            fhat_mean = self.surr_model.mean.to(device)
            fhat_cov = self.surr_model.covariance_matrix.to(device)

            p_mean = self.dataset.torch_p.mean.to(device)
            p_cov = self.dataset.torch_p.covariance_matrix.to(device)

            # cookbook 8.1.8
            pci = p_cov.inverse()
            fci = fhat_cov.inverse()
            new_cov = (pci + fci).inverse()
            new_loc = (new_cov) @ (pci @ p_mean + fci @ fhat_mean)
            q = multivariate_normal(
                mean=t2n(new_loc),
                cov=t2n(new_cov))

            if not self.cfg.get('defensive', False):
                return self.sample_pdf(p, q)

            a = self.cfg.defensive_val
            q_safe = TwoComponentMixtureDistribution(p, q, a)
            return self.sample_pdf(p, q_safe)

        elif self.dataset.__class__.__name__ == 'OnlineToyDataset':
            raise ValueError
            # p = self.dataset.p
            # mean = p.mean

            # q =  p(x) * 1/D \sum_j x_j  / (int p(x) * 1/D \sum_j x_j dx)
            # =  p(x) * 1/D \sum_j x_j  / mean x_j
            # =  p(x) * empirical mean (x_j) / actual mean (x_j)

            # this actually does not seem trivial to sample from
            # (note this is only a distribution if x_j always positive)


class AnySurrogateAcquisitionLogEntropy(
        _AnySurrogateAcquisitionBase):
    """Add extra log to actually target entropy of surrogate as per ASMC."""

    def __init__(self, cfg, dataset, model, model_cfg, *args, **kwargs):
        super().__init__(cfg, dataset, model, model_cfg)

    def expected_loss(self, remaining_data, remaining_idxs):
        return entropy_loss(
            remaining_data, remaining_idxs, self.model, self.surr_model,
            cfg=self.cfg)


class AnySurrogateSurrogateUncertaintyAcquisition(
        _AnySurrogateAcquisitionBase):

    def __init__(self, cfg, dataset, model, model_cfg, *args, **kwargs):
        super().__init__(cfg, dataset, model, model_cfg)

    def expected_loss(self, remaining_data, remaining_idxs):
        return entropy_loss(
            remaining_data, remaining_idxs, self.surr_model, self.surr_model,
            cfg=self.cfg)


class AnySurrogateAcquisitionAccuracy(
        _AnySurrogateAcquisitionBase):

    def __init__(self, cfg, dataset, model, model_cfg, *args, **kwargs):
        super().__init__(cfg, dataset, model, model_cfg)

    def expected_loss(self, remaining_data, remaining_idxs):
        return accuracy_loss(
            remaining_data, remaining_idxs, self.model, self.surr_model)


class _GPSurrogateAcquisitionBase(_SurrogateAcquisitionBase):
    def __init__(self, cfg, dataset, model, model_cfg):
        super().__init__(
            cfg, dataset, model, GaussianProcessRegressor, model_cfg)


class GPSurrogateAcquisitionLogLik(_GPSurrogateAcquisitionBase):
    warning = (
        'GPSurrogateAcquisitionLogLik only works if aleatoric noise 0.')
    # warnings.warn(warning)

    def __init__(self, cfg, dataset, model, model_cfg, **kwargs):

        super().__init__(cfg, dataset, model, model_cfg)

    def expected_loss(self, remaining_data, remaining_idxs, *args, **kwargs):

        std = dict(return_std=True)
        mu_s, std_s = self.surr_model.predict(
            remaining_data, idxs=remaining_idxs, **std)
        mu_m, std_m = self.model.predict(
            remaining_data, idxs=remaining_idxs, **std)

        aleatoric = self.get_aleatoric()

        expected_loss = (
            np.log(2*np.pi*std_m**2)
            + 1/(2*std_m**2) * (
                (mu_s - mu_m)**2 + std_s**2 + aleatoric**2)
        )

        return expected_loss


class GPSurrogateAcquisitionMSE(_GPSurrogateAcquisitionBase):
    warning = (
        'GPSurrogateAcquisitionMSE is currently only appropriate if '
        'the aleatoric uncertainty is 0.')
    # warnings.warn(warning)

    def __init__(self, cfg, dataset, model, model_cfg, **kwargs):

        super().__init__(cfg, dataset, model, model_cfg)

    def expected_loss(self, remaining_data, remaining_idxs):

        mu_s, std_s = self.surr_model.predict(
            remaining_data, idxs=remaining_idxs, return_std=True)
        mu_m = self.model.predict(remaining_data, idxs=remaining_idxs)

        # each model needs to have this set
        # TODO: should probs be a data property
        # (move reliance on main model out of GPDataset)
        aleatoric = self.get_aleatoric()

        expected_loss = (mu_s - mu_m)**2 + std_s**2 + aleatoric**2

        # a = (mu_s - mu_m)**2
        # b = std_s**2
        # c = aleatoric**2
        # abc = a + b + c
        # print(
        #     f'mse {(a/(abc)).mean():.2f} +- {(a/(abc)).std():.2f} \n'
        #     f'var {(b/(abc)).mean():.2f} +- {(b/(abc)).std():.2f} \n'
        #     f'alea {(c/(abc)).mean():.2f} +- {(c/(abc)).std():.2f} \n')

        if self.cfg.get('clip', False):
            clip_val = 0.05 * np.max(expected_loss)
            if clip_val < 1e-10:
                warnings.warn('All loss values small!')

            expected_loss = np.maximum(clip_val, expected_loss)

        return expected_loss


class GPSurrogateAcquisitionMSENoDis(_GPSurrogateAcquisitionBase):
    warning = (
        'GPSurrogateAcquisitionMSE is currently only appropriate if '
        'the aleatoric uncertainty is 0.')
    # warnings.warn(warning)

    def __init__(self, cfg, dataset, model, model_cfg, **kwargs):

        super().__init__(cfg, dataset, model, model_cfg)

    def expected_loss(self, remaining_data, remaining_idxs):

        mu_s, std_s = self.surr_model.predict(
            remaining_data, idxs=remaining_idxs, return_std=True)
        mu_m = self.model.predict(remaining_data, idxs=remaining_idxs)

        # each model needs to have this set
        # TODO: should probs be a data property
        # (move reliance on main model out of GPDataset)
        aleatoric = self.get_aleatoric()

        expected_loss = std_s**2 + aleatoric**2

        # a = (mu_s - mu_m)**2
        # b = std_s**2
        # c = aleatoric**2
        # abc = a + b + c
        # print(
        #     f'mse {(a/(abc)).mean():.2f} +- {(a/(abc)).std():.2f} \n'
        #     f'var {(b/(abc)).mean():.2f} +- {(b/(abc)).std():.2f} \n'
        #     f'alea {(c/(abc)).mean():.2f} +- {(c/(abc)).std():.2f} \n')

        if self.cfg.get('clip', False):
            clip_val = 0.05 * np.max(expected_loss)
            if clip_val < 1e-10:
                warnings.warn('All loss values small!')

            expected_loss = np.maximum(clip_val, expected_loss)

        return expected_loss


class GPSurrogateAcquisitionMSEDoublyUncertain(_GPSurrogateAcquisitionBase):
    warning = (
        'GPSurrogateAcquisitionMSEDoublyUncertain is currently only '
        'appropriate if the aleatoric uncertainty is 0.')
    # warnings.warn(warning)

    def __init__(self, cfg, dataset, model, model_cfg, **kwargs):

        super().__init__(cfg, dataset, model, model_cfg)
        self.model_cfg = model_cfg

    def expected_loss(self, remaining_data, remaining_idxs):

        mu_s, std_s = self.surr_model.predict(remaining_data, return_std=True)
        mu_m, std_m = self.model.predict(
            remaining_data, idxs=remaining_idxs, return_std=True)

        expected_loss = (mu_s - mu_m)**2 + std_s**2 + std_m**2

        return expected_loss


class ClassifierAcquisitionEntropy(_LossAcquisitionBase):

    def __init__(self, cfg, dataset, model, model_cfg, *args, **kwargs):

        if model_cfg is not None and model_cfg.get('acquisition', False):
            _cfg = OmegaConf.merge(
                OmegaConf.structured(cfg[0]),
                OmegaConf.structured(model_cfg.acquisition))
            cfg = [_cfg, cfg[1]]

        super().__init__(cfg, dataset, model)
        logging.info(f'Config {cfg}.')
        self.T = model_cfg.get('temperature', None)

    def expected_loss(self, remaining_data, remaining_idxs):
        return entropy_loss(
            remaining_data, remaining_idxs, self.model, None, T=self.T,
            cfg=self.cfg)


class ClassifierAcquisitionAccuracy(_LossAcquisitionBase):

    def __init__(self, cfg, dataset, model, model_cfg, *args, **kwargs):
        super().__init__(cfg, dataset, model)

    def expected_loss(self, remaining_data, remaining_idxs):
        return accuracy_loss(
            remaining_data, remaining_idxs, self.model, None)


class ClassifierAcquisitionValue(_LossAcquisitionBase):
    """Just acquire based on predictions of model.

    This is useful for the toy scenario where predictions of model
    are directly the f(x) we care about.
    """
    def __init__(self, cfg, dataset, model, model_cfg, *args, **kwargs):

        if model_cfg is not None and model_cfg.get('acquisition', False):
            _cfg = OmegaConf.merge(
                OmegaConf.structured(cfg[0]),
                OmegaConf.structured(model_cfg.acquisition))
            cfg = [_cfg, cfg[1]]

        super().__init__(cfg, dataset, model)
        logging.info(f'Config {cfg}.')

    def expected_loss(self, remaining_data, remaining_idxs):
        return self.model.predict(remaining_data, idxs=remaining_idxs)


class _RandomForestSurrogateAcquisitionBase(_SurrogateAcquisitionBase):

    def __init__(self, cfg, dataset, model, model_cfg):
        super().__init__(
            cfg, dataset, model, RandomForestClassifier, model_cfg)


class RandomForestClassifierSurrogateAcquisitionEntropy(
        _RandomForestSurrogateAcquisitionBase):

    def __init__(self, cfg, dataset, model, model_cfg, *args, **kwargs):
        super().__init__(cfg, dataset, model, model_cfg)

    def expected_loss(self, remaining_data, remaining_idxs):
        return entropy_loss(
            remaining_data, remaining_idxs, self.model, self.surr_model,
            cfg=self.cfg)


class _SVMClassifierSurrogateAcquisitionBase(_SurrogateAcquisitionBase):

    def __init__(self, cfg, dataset, model, model_cfg):
        super().__init__(cfg, dataset, model, SVMClassifier, model_cfg)


class SVMClassifierSurrogateAcquisitionEntropy(
        _SVMClassifierSurrogateAcquisitionBase):

    def __init__(self, cfg, dataset, model, model_cfg, *args, **kwargs):
        super().__init__(cfg, dataset, model, model_cfg)

    def expected_loss(self, remaining_data, remaining_idxs):
        return entropy_loss(
            remaining_data, remaining_idxs, self.model, self.surr_model,
            cfg=self.cfg)


class _GPClassifierSurrogateAcquisitionBase(_SurrogateAcquisitionBase):

    def __init__(self, cfg, dataset, model, model_cfg):
        super().__init__(cfg, dataset, model, GPClassifier, model_cfg)


class GPClassifierSurrogateAcquisitionEntropy(
        _GPClassifierSurrogateAcquisitionBase):

    def __init__(self, cfg, dataset, model, model_cfg, *args, **kwargs):
        super().__init__(cfg, dataset, model, model_cfg)

    def expected_loss(self, remaining_data, remaining_idxs):
        return entropy_loss(
            remaining_data, remaining_idxs, self.model, self.surr_model,
            cfg=self.cfg
            )


class _RandomRandomForestSurrogateAcquisitionBase(_LossAcquisitionBase):
    """Randomize Hypers each iteration."""

    def __init__(self, cfg, dataset, model, model_cfg):
        super().__init__(cfg, dataset, model)

        self.model_cfg = model_cfg
        self.surr_model = None
        self.random_init_model()

    def random_init_model(self):

        if self.model_cfg['params_from'] == 'main':
            if self.surr_model is not None:
                return True
            else:
                sk_args = self.model.model.get_params()
                cfg = OmegaConf.create(dict(sk_args=sk_args))

        elif self.model_cfg['params_from'] == 'random':
            # This may be highly dependent on the data!!
            sk_args = dict(
                max_features='sqrt',
                criterion=str(np.random.choice(["gini", "entropy"])),
                max_depth=int(np.random.choice([3, 5, 10, 20])),
                n_estimators=int(np.random.choice([10, 50, 100, 200])),
                # min_samples_split=int(np.random.choice([2, 5, 10]))
            )
            cfg = OmegaConf.create(dict(sk_args=sk_args))
        else:
            raise ValueError

        if self.model_cfg['rotated']:
            self.surr_model = RandomDirectionRandomForestClassifier(
                cfg, speedup=True, dim=self.dataset.D[0]
                )
        else:
            self.surr_model = RandomForestClassifier(cfg)

    def update_surrogate(self):

        self.random_init_model()

        self.surr_model.fit(*self.dataset.total_observed)

    def acquire(self, update_surrogate=True):
        if update_surrogate:
            self.update_surrogate()

        return super().acquire()


class RandomRandomForestClassifierSurrogateAcquisitionEntropy(
        _RandomRandomForestSurrogateAcquisitionBase):

    def __init__(self, cfg, dataset, model, model_cfg, *args, **kwargs):
        super().__init__(cfg, dataset, model, model_cfg)

    def expected_loss(self, remaining_data, remaining_idxs):
        return entropy_loss(
            remaining_data, remaining_idxs, self.model, self.surr_model,
            cfg=self.cfg)


def entropy_loss(
        remaining_data, remaining_idxs, model, surr_model=None,
        eps=1e-15, T=None, cfg=None, extra_log=False):

    model_pred = model.predict(remaining_data, idxs=remaining_idxs)

    if T is not None:
        model_pred = np.exp(np.log(model_pred)/T)

        model_pred = np.clip(model_pred, eps, 1/eps)
        model_pred[np.isnan(model_pred)] = 1/eps

        model_pred /= model_pred.sum(axis=1, keepdims=True)

        model_pred = np.clip(model_pred, eps, 1/eps)
        model_pred[np.isnan(model_pred)] = 1/eps

    if surr_model is not None:
        surr_model_pred = surr_model.predict(
            remaining_data, idxs=remaining_idxs)

        if T is not None:
            surr_model_pred = np.exp(np.log(surr_model_pred)/T)
            surr_model_pred = np.clip(surr_model_pred, eps, 1/eps)
            surr_model_pred[np.isnan(surr_model_pred)] = 1/eps

            surr_model_pred /= surr_model_pred.sum(axis=1, keepdims=True)
            surr_model_pred = np.clip(surr_model_pred, eps, 1/eps)
            surr_model_pred[np.isnan(surr_model_pred)] = 1/eps

    else:
        surr_model_pred = model_pred

    if T is None:
        model_pred = np.clip(model_pred, eps, 1 - eps)
        model_pred /= model_pred.sum(axis=1, keepdims=True)

    # Sum_{y=c} p_surr(y=c|x) log p_model(y=c|x)
    if not extra_log:
        res = -1 * (surr_model_pred * np.log(model_pred)).sum(-1)
    else:
        raise NotImplementedError('Not sure what this should look like')
        res = -1 * (surr_model_pred * np.log(model_pred)).sum(-1)

    if T is not None:
        res[np.isnan(res)] = np.nanmax(res)

    # Entropy may have zero support over some of the remaining items!
    # This is not good! Model is overconfident! Condition of estimator
    # do no longer hold!

    # clip at lowest 10 percentile of prediction (add safeguard for 0 preds)
    # clip_val = max(np.percentile(res, 10), 1e-3)
    # 1e-3 is a lot for large remaining_data, probably better as
    # 1/(100*len(remaining_data))

    if cfg is not None and not cfg.get('uniform_clip', False):
        clip_val = np.percentile(res, 10)
        res = np.clip(res, clip_val, 1/eps)

    # clipping has moved to after acquisition
    return res


def accuracy_loss(
        remaining_data, remaining_idxs, model, surr_model=None):
    # we need higher values = higher loss
    # so we will return 1 - accuracy

    model_pred = model.predict(remaining_data, idxs=remaining_idxs)

    if surr_model is not None:
        surr_model_pred = surr_model.predict(
            remaining_data, idxs=remaining_idxs)
    else:
        surr_model_pred = model_pred

    pred_classes = np.argmax(model_pred, axis=1)

    # instead of 0,1 loss we get p_surr(y|x) for accuracy

    res = 1 - surr_model_pred[np.arange(len(surr_model_pred)), pred_classes]

    res = np.maximum(res, np.max(res)*0.05)

    return res


def covariance(preds, mean_preds, dataset, model):

    loss = CrossEntropyLoss()

    eps = 1e-15
    preds = np.clip(preds, eps, 1 - eps)
    preds /= preds.sum(axis=1, keepdims=True)

    # for each remaining sample, for each datum, get a loss
    losses = (-mean_preds * np.log(preds)).sum(-1)

    # get the risk: has shape n_samples
    unobserved_risk = losses

    observed_idxs = dataset.test_observed

    if len(observed_idxs) > 0:
        observed_data = dataset.x[observed_idxs]
        observed_labels = dataset.y[observed_idxs]
        observed_predictions = model.predict(
            observed_data, idxs=observed_idxs)
        observed_risk = loss(
            observed_predictions,
            observed_labels)

        # tile according to number of samples
        observed_risk = observed_risk[:, np.newaxis]
        observed_risk = np.tile(observed_risk, unobserved_risk.shape[1])

        risk = np.concatenate([unobserved_risk, observed_risk], 0)
        risk = risk.mean(0)
    else:
        risk = unobserved_risk.mean(0)

    # we just get a single estimate of risk
    # average over samples to get covariance
    cov = ((losses - risk[np.newaxis])**2).mean(1)

    return cov


class BayesQuadAcquisition(AcquisitionFunction):
    def __init__(self, cfg_run, dataset, model_cfg, *args, **kwargs):

        _cfg = OmegaConf.merge(
            OmegaConf.structured(cfg_run[0]),
            OmegaConf.structured(model_cfg.acquisition))
        cfg_run = [_cfg, cfg_run[1]]

        super().__init__(cfg_run, dataset)

        self.surr_class = BayesQuadModel
        self.surr_cfg = model_cfg

        # self.surr_model.y_train_mean = dataset.y_train_mean
        # self.surr_model.y_train_std = dataset.y_train_std

    def update_surrogate(self):

        # train surrogate on train data + currently observed test
        self.surr_model = self.surr_class(self.surr_cfg)

        # fit on all observed data
        x, y = self.dataset.total_observed
        self.surr_model.fit(x, y)

        integral = self.surr_model.model.integral_mean()
        logging.info(f'Acquisition: BQ est for N, D={x.shape}, is {integral}.')

    def acquire(self, update_surrogate=True):

        if update_surrogate:
            self.update_surrogate()

        # this is the expensive part.
        locs = select_batch(
            self.surr_model.model, batch_size=1,
            batch_method=LOCAL_PENALISATION)

        diffs = self.find_match(locs)[:, 0]
        diffs = softmax(-diffs)

        return self.sample_pmf(diffs)

    def find_match(self, locs):
        rem_idx = self.dataset.test_remaining
        rem = self.dataset.x[rem_idx]
        locs = np.array(locs)
        diffs = ((rem[:, np.newaxis] - locs[np.newaxis, :])**2).sum(-1)
        diffs = np.sqrt(diffs)
        return diffs
