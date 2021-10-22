from activetesting.datasets import DoubleGaussianDataset
import logging
import numpy as np
from omegaconf import OmegaConf


DEBUG_WEIGHTS = False


class RiskEstimator:
    def __init__(self, loss, *args, risk_cfg=None, **kwargs):
        from activetesting.utils import maps
        self.loss = maps.loss[loss]()
        self.risks = np.array([[]])
        self.risk_cfg = risk_cfg

    def return_and_save(self, loss):
        self.risks = np.append(self.risks, loss)
        return loss


class TrueRiskEstimator(RiskEstimator):
    """Used for performance evaluation."""
    def __init__(self, loss, dataset, model, *args, **kwargs):
        super().__init__(loss)

        idxs = dataset.test_idxs
        y_true = dataset.y[idxs]
        y_pred = model.predict(dataset.x[idxs], idxs=idxs)
        self.true_loss_vals = self.loss(y_pred, y_true)
        self.true_loss = self.true_loss_vals.mean()

        if (dataset.N < dataset.y.size) and dataset.cfg.get('with_unseen'):
            N = dataset.y.size
        else:
            N = dataset.N

        self.true_loss_all_idxs = np.zeros(N)
        self.true_loss_all_idxs[idxs] = self.true_loss_vals
        # print('true loss debug', self.true_loss)

    def estimate(self, *args, **kwargs):
        return self.return_and_save(self.true_loss)


class ExactExpectedRiskEstimator(RiskEstimator):
    """Used for performance evaluation."""
    def __init__(self, loss, dataset, model, *args, **kwargs):
        super().__init__(loss)

        # We have E_x[f(x)] = E_x[\sum x_j]
        # with f(x) = \prod_j p(x_j)
        # \int p(x) f(x) dx 
        # = \int \prod_j p(x_j) \sum_j x_j
        # = \sum_j \int p(x_j) x_j
        # = \sum_j E[x_j]
        # = \sum_j \mu_j
        # = N * \mu

        # the individual values don't matter for this one..
        # I'm not sure why
        if dataset.__class__.__name__ == 'DoubleGaussianDataset':
            from scipy.stats import multivariate_normal
            d = dataset
            self.true_loss = multivariate_normal(
                mean=d.p.mean, cov=d.p.cov+dataset.f.cov).pdf(d.f.mean)
        elif dataset.__class__.__name__ == 'OnlineToyDataset':
            dargs = dataset.creation_args
            test_mean, n_pixels = dargs['test_mean'], dargs['n_pixels']

            if dargs['normalise']:
                self.true_loss = test_mean
            else:
                self.true_loss = n_pixels * test_mean

        else:
            raise ValueError

        logging.info(f'*******THE TRUE VALUE IS {self.true_loss}')

    def estimate(self, *args, **kwargs):
        return self.return_and_save(self.true_loss)


class TrueUnseenRiskEstimator(RiskEstimator):
    """Used for performance evaluation."""
    def __init__(self, loss, dataset, model, *args, **kwargs):
        super().__init__(loss)

        # not compatible with lazy prediction
        idxs = dataset.test_unseen_idxs
        y_true = dataset.y[idxs]
        y_pred = model.predict(dataset.x[idxs], idxs=idxs)
        self.true_loss_vals = self.loss(y_pred, y_true)
        self.true_loss = self.true_loss_vals.mean()

        if (dataset.N < dataset.y.size) and dataset.cfg.get('with_unseen'):
            N = dataset.y.size
        else:
            N = dataset.N

        self.true_loss_all_idxs = np.zeros(N)
        self.true_loss_all_idxs[idxs] = self.true_loss_vals
        # print('true loss debug', self.true_loss)

    def estimate(self, *args, **kwargs):
        return self.return_and_save(self.true_loss)


class BiasedRiskEstimator(RiskEstimator):
    def __init__(self, loss, *args, **kwarg):
        super().__init__(loss)

    def estimate(self, predictions, observed, *args, **kwargs):
        l_i = self.loss(predictions, observed).mean()
        # logging.info(f'debug biased risk estimator {l_i}')
        return self.return_and_save(l_i)


class ImportanceWeightedRiskEstimator(RiskEstimator):
    def __init__(self, loss, dataset, *args, **kwargs):
        super().__init__(loss)

    def estimate(self, predictions, observed, acq_weights, *args, **kwargs):

        l_i = self.loss(predictions, observed)
        M = len(predictions)

        # R = 1/M * (1/acq_weights * l_i).sum()
        # as per Scheffer / Landwehr paper
        R = (1/acq_weights * l_i).sum() / (1/acq_weights).sum()

        return self.return_and_save(R)


class ImportanceWeightedRiskEstimatorWithP(RiskEstimator):
    def __init__(self, loss, dataset, *args, **kwargs):
        super().__init__(loss)
        self.dataset = dataset

    def estimate(self, predictions, observed, acq_weights, *args, **kwargs):

        l_i = self.loss(predictions, observed)
        M = len(predictions)

        # acq_weights = fhat * p
        x = self.dataset.x[self.dataset.test_observed]
        p = self.dataset.p.pdf(x)

        R = 1/M * (p/acq_weights * l_i).sum()

        return self.return_and_save(R)


class ImportanceWeightedRiskEstimatorForPDFs(RiskEstimator):
    """Expects acquisition weights do be directly correct.

    Does not add p like the above.

    """
    def __init__(self, loss, dataset, *args, **kwargs):
        super().__init__(loss)
        self.dataset = dataset

    def estimate(self, predictions, observed, acq_weights, *args, **kwargs):

        l_i = self.loss(predictions, observed)
        M = len(predictions)

        # acq_weights = fhat * p
        R = 1/M * (acq_weights * l_i).sum()

        return self.return_and_save(R)


class NaiveUnbiasedRiskEstimator(RiskEstimator):
    def __init__(self, loss, dataset, *args, **kwargs):
        super().__init__(loss)
        self.N = len(dataset.test_idxs)

    def estimate(self, predictions, observed, acq_weights, *args, **kwargs):

        l_i = self.loss(predictions, observed)
        N = self.N
        M = len(predictions)
        m = np.arange(1, M+1)

        v = 1/(N * acq_weights) + (M-m) / N

        R = 1/M * (v * l_i).sum()

        return self.return_and_save(R)


class FancyUnbiasedRiskEstimator(RiskEstimator):
    def __init__(self, loss, dataset, *args, **kwargs):
        super().__init__(loss)
        self.N = len(dataset.test_idxs)

    @staticmethod
    def get_weights(acq_weights, N):
        M = len(acq_weights)
        if M < N:
            m = np.arange(1, M+1)
            v = (
                1
                + (N-M)/(N-m) * (
                        1 / ((N-m+1) * acq_weights)
                        - 1
                        )
                )
        else:
            v = 1

        return v

    def estimate(self, predictions, observed, acq_weights, *args, **kwargs):

        l_i = self.loss(predictions, observed)
        N = self.N
        M = len(predictions)

        v = self.get_weights(acq_weights, N)

        R = 1/M * (v * l_i).sum()

        if DEBUG_WEIGHTS:
            if isinstance(v, int):
                v = [v]
            with open('weights.csv', 'a') as f:
                data = str(list(v)).replace('[', '').replace(']', '')
                f.write(f'{len(v)}, {data}\n')

        return self.return_and_save(R)


class _QuadratureRiskEstimator(RiskEstimator):
    """Naively estimate risk by using

    * true outcomes for data in observed.
    * surrogate outcomes for data not observed.

    """
    def __init__(self, loss, dataset, model, *args, **kwargs):
        super().__init__(loss)

        self.dataset = dataset
        idxs = self.dataset.test_idxs
        y_pred = model.predict(self.dataset.x[idxs], idxs=idxs)
        self.task = self.dataset.cfg.task_type
        classes = self.dataset.cfg.n_classes

        if (dataset.N < dataset.y.size) and dataset.cfg.get('with_unseen'):
            N = dataset.y.size
        else:
            N = dataset.N

        if self.task == 'classification':
            self.y_pred = np.zeros((N, classes))
        else:
            self.y_pred = np.zeros(N)

        self.y_pred[idxs] = y_pred
        self.model = model

    # @profile
    def estimate(
            self, predictions, observed, weights, surrogate, acquisition_name,
            *args, **kwargs):

        if acquisition_name == 'RandomAcquisition':
            self.quadrature_loss = [None]
            self.remaining_idxs = None
            return self.return_and_save(-1e19)

        if (
                surrogate is None or
                getattr(surrogate, 'name', False) == 'BayesQuadModel'):
            # A bit hacky. Fail silently.
            # This estimator will only return sensible values for acquisition
            # strategies that have a surrogate model.
            # This is ugly like this in the code because
            # estimators are always applied to all acquisition strategies.
            surrogate = self.model

        # Risk for data observed from test pool
        # -------------------------------------
        R_observed = self.loss(predictions, observed)

        # Risk estimate using the surrogate model
        # ---------------------------------------
        # Remaining data
        remaining_idxs = self.dataset.test_remaining

        if len(remaining_idxs) == 0:
            return self.return_and_save(R_observed.mean())

        remaining_data = self.dataset.x[remaining_idxs]

        # Model predictions
        remaining_y_pred = self.y_pred[remaining_idxs]

        # Estimate of the surrogate model of true outcome
        if self.task == 'regression':
            y_true_surr, std_surr = surrogate.predict(
                remaining_data, idxs=remaining_idxs, return_std=True)
            # Estimate loss from disagreement between the two
            R_surrogate = self.loss(remaining_y_pred, y_true_surr)
            R_surrogate = R_surrogate + self.add_uncertainty(std_surr)

        else:
            eps = 1e-20
            y_true_surr = surrogate.predict(
                remaining_data, idxs=remaining_idxs)

            remaining_y_pred = np.clip(remaining_y_pred, eps, 1 - eps)
            remaining_y_pred /= remaining_y_pred.sum(axis=1, keepdims=True)

            R_surrogate = self.get_R_surrogate(
                remaining_y_pred, y_true_surr,
                surrogate, remaining_data, remaining_idxs)

        R = np.concatenate([R_observed, R_surrogate], 0)

        # export estimates at first iteration
        if len(observed) == 1:
            # just export way more stuff
            self.quadrature_loss = [
                R_surrogate,
                y_true_surr,
                remaining_y_pred,
                remaining_idxs]

            self.remaining_idxs = remaining_idxs

        R = R.mean()

        return self.return_and_save(R)


class QuadratureRiskEstimator(_QuadratureRiskEstimator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_uncertainty(self, *args):
        return 0

    def get_R_surrogate(self, remaining_y_pred, y_true_surr, *args, **kwargs):
        return -1 * (y_true_surr * np.log(remaining_y_pred)).sum(-1)


class QuadratureRiskEstimatorWithUncertainty(_QuadratureRiskEstimator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_uncertainty(self, std_surr):
        if getattr(self.dataset.cfg, 'expose_aleatoric', True):
            aleatoric = getattr(self.dataset, 'aleatoric', 0)
        else:
            aleatoric = 0

        return std_surr**2 + aleatoric**2

    def get_R_surrogate(self, remaining_y_pred, y_true_surr, *args, **kwargs):
        return -1 * (y_true_surr * np.log(remaining_y_pred)).sum(-1)


class QuadratureRiskEstimatorRemoveNoise(_QuadratureRiskEstimator):
    """
    - E_surr[log model]
    = E_surr[log surr/model] - E_surr[log surr]
    = KL[surr|model] + H[surr]

    here: remove all noise --> compute
    KL[surr|model] =  - E_surr[log model] + E_surr[log surr]

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_R_surrogate(
            self, remaining_y_pred, y_true_surr, *args, **kwargs):
        model_loss = -1 * (y_true_surr * np.log(remaining_y_pred)).sum(-1)
        neg_entropy = (y_true_surr * np.log(y_true_surr)).sum(-1)
        return model_loss + neg_entropy


class QuadratureRiskEstimatorNoDisagreement(_QuadratureRiskEstimator):
    """
    - E_surr[log model]
    = E_surr[log surr/model] - E_surr[log surr]
    = KL[surr|model] + H[surr]

    here: remove all all disagreement --> compute
    H[surr]

    Note: This is somewhat of a dumb one. If this works, then we know that
    a) aleatoric uncertainties between model and surr are the same
    b) all disagreement is noise

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_R_surrogate(self, remaining_y_pred, y_true_surr, *args, **kwargs):
        entropy = - (y_true_surr * np.log(y_true_surr)).sum(-1)
        return entropy


class QuadratureRiskEstimatorNoAleatoric(_QuadratureRiskEstimator):
    """
    -E _surr[log model]
    = E_surr[log surr/model] - E_surr[log surr]
    = KL[surr|model] + H[surr]

    here: remove all aleatoric uncertainty from H[surr] --> compute
    KL[surr|model] + H[surr] - E_surr[H[surr| th]]
    = KL[surr|model] + epistemic
    = KL[surr|model] - E_surr[log surr] + E_surr[E[log surr|th]]
    = -E_surr[log model] + E_surr[E[log surr|th]]

    (= NLL - aleatoric. --> exactly!)


    Note: can't do that one right now. also unclear how to draw samples from
    my temperature ensemble. I could just jacknife estimate. (i.e. leave one
    out) --> but this would have only a small impact 

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_R_surrogate(
            self, remaining_y_pred, y_true_surr, surrogate,
            remaining_data, remaining_idxs):

        loss = -1 * (y_true_surr * np.log(remaining_y_pred)).sum(-1)

        # annoying thing because of efficientmodel
        if hasattr(surrogate, 'joint_predict') and hasattr(surrogate.model, 'joint_predict'):
            joint_pred = surrogate.joint_predict(
                remaining_data, idxs=remaining_idxs)
            eps = 1e-20
            joint_pred = np.clip(joint_pred, eps, 1 - eps)
            joint_pred /= joint_pred.sum(axis=-1, keepdims=True)

            aleatoric = - ((joint_pred * np.log(joint_pred)).sum(-1)).mean(0)

        else:
            # I guess in this case
            # E_surr[E[log surr|th]] is the same as E_surr[log_surr]
            # and therefore I return
            aleatoric = - (y_true_surr * np.log(y_true_surr)).sum(-1)

        return loss - aleatoric


class QuadratureRiskEstimatorNoEpistemic(_QuadratureRiskEstimator):
    """
    - E_surr[log model]
    = E_surr[log surr/model] - E_surr[log surr]
    = KL[surr|model] + H[surr]

    here: remove all epistemic uncertainty from H[surr] -->
    KL[surr|model] + H[surr] - epistemic
    = KL[surr|model] + H[surr] - (H[surr] - E[H[sur|theta]])
    = -E_surr[log model] + E_surr[log surr] - E[E[log sur|theta]])

    (= NLL - ( H[surr]  - E[H[surr]]) = NLL - epistemic )

    Note: can't do that one right now. also unclear how to draw samples from
    my temperature ensemble. I could just jacknife estimate. (i.e. leave one
    out) --> Ah well. T=1 anyways for the ensemble. Just ignore it for
    this experiment.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_R_surrogate(
            self, remaining_y_pred, y_true_surr, surrogate,
            remaining_data, remaining_idxs):
        # raise for now
        loss = -1 * (y_true_surr * np.log(remaining_y_pred)).sum(-1)

        # annoying thing because of efficientmodel
        if hasattr(surrogate, 'joint_predict') and hasattr(surrogate.model, 'joint_predict'):
            joint_pred = surrogate.joint_predict(
                remaining_data, idxs=remaining_idxs)
            eps = 1e-20
            joint_pred = np.clip(joint_pred, eps, 1 - eps)
            joint_pred /= joint_pred.sum(axis=-1, keepdims=True)

            epistemic = (
                (- y_true_surr * np.log(y_true_surr)).sum(-1)
                - (-((joint_pred * np.log(joint_pred)).sum(-1)).mean(0))
            )

        else:
            # I guess in this case
            # E[H[sur|theta]]) is the same as H[sur]
            # and therefore I return 0
            epistemic = 0

        return loss - epistemic


class ConvexCombo(RiskEstimator):
    def __init__(self, *args, risk_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        if risk_cfg is None:
            raise ValueError

        from activetesting.utils.maps import risk_estimator
        risks = risk_cfg.get('risks', False)
        self.risk1 = risk_estimator[risks[0]](*args, **kwargs)
        self.risk2 = risk_estimator[risks[1]](*args, **kwargs)

        self.cfg = risk_cfg
        s = risk_cfg.get('schedule', False) if risk_cfg is not None else False
        if not s:
            self.alpha = lambda: 0.5
        elif s == 'alpha_from_1to2':
            self.alpha = self.alpha_from_1to2
        elif s == 'alpha_from_2to1':
            self.alpha = self.alpha_from_2to1
        else:
            raise ValueError

    def alpha_from_1to2(self):
        data = self.risk2.dataset
        lim = self.cfg.get('n_max', len(data.test_idxs))
        alpha = 1 - len(data.test_observed) / lim
        return max(min(alpha, 1), 0)

    def alpha_from_2to1(self):
        return 1 - self.alpha_from_1to2()

    def estimate(self, *args, **kwargs):
        risk = (
            self.alpha() * self.risk1.estimate(*args, **kwargs)
            + (1-self.alpha()) * self.risk2.estimate(*args, **kwargs)
        )
        return self.return_and_save(risk)


class ConvexComboWithUncertainty(ConvexCombo):
    def __init__(self, *args, risk_cfg=None, **kwargs):

        _risk_cfg = dict(risks=[
            'FancyUnbiasedRiskEstimator',
            'QuadratureRiskEstimatorWithUncertainty'])

        if risk_cfg is None:
            risk_cfg = _risk_cfg
        else:
            risk_cfg = OmegaConf.merge(
                OmegaConf.structured(risk_cfg), _risk_cfg)

        super().__init__(*args, risk_cfg=risk_cfg, **kwargs)


class ConvexComboWithOutUncertainty(ConvexCombo):
    def __init__(self, *args, risk_cfg=None, **kwargs):

        _risk_cfg = dict(risks=[
            'FancyUnbiasedRiskEstimator',
            'QuadratureRiskEstimator'])

        if risk_cfg is None:
            risk_cfg = _risk_cfg
        else:
            risk_cfg = OmegaConf.merge(
                OmegaConf.structured(risk_cfg), _risk_cfg)

        super().__init__(*args, risk_cfg=risk_cfg, **kwargs)


class _BayesQuadRiskEstimator(RiskEstimator):
    """Estimate Risk Using BayesQuad
    """
    def __init__(self, loss, dataset, model, *args, **kwargs):
        super().__init__(loss)
        self.dataset = dataset
        args = self.dataset.cfg.creation_args

        if D := args.get('n_pixels', False):
            self.prior_cfg = OmegaConf.structured(dict(model_cfg=dict(
                data_CHW=D,
                prior_mean=args['test_mean'],
                prior_std=args['test_std'],
                test_distribution=args['test_distribution'],
                test_cij=args.get('test_c_ij', None),
                lik_var=args['lik_var'],
                scale_val=args['scale_val'],
                )))

        else:
            dim = args['dim']
            self.prior_cfg = OmegaConf.structured(dict(model_cfg=dict(
                data_CHW=dim,
                prior_mean=args['p_mean'],
                prior_std=args['p_std'],
                test_distribution='gaussian',
                scale_val=args['scale_val'],
                lik_var=args['lik_var'],
                mean_mean_function=args.get('mean_mean_function', False),
                )))

    def valid(self, acquisition_name):
        # maybe don't want to estimate risks for all acquisition functions
        valid_list = [
            'AnySurrogateAcquisitionValue',
            'RandomAcquisition',
            'AnySurrogateBayesQuadSampleFromPDFDirectly']

        if acquisition_name in valid_list:
            # print('valid')
            return True
        else:
            # print('not valid')
            return False


class BayesQuadRiskEstimator(_BayesQuadRiskEstimator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def estimate(
            self, predictions, observed, weights, surrogate,
            *args, acquisition_name=None, **kwargs):

        if not self.valid(acquisition_name):
            return self.return_and_save(1e19)

        # TODO: check if BQ specifically
        if (
                (surrogate is not None) and
                str(surrogate.__class__) == 'BayesQuadModel'):

            # surrogate is not a
            # # DEBUGTOM
            c = surrogate._int_cfg.model_cfg.get('scale_val', 1)
            integral = surrogate.model.integral_mean() * c
            logging.info(f'Risk: {integral}.')
        else:
            from activetesting.models import BayesQuadModel
            bq = BayesQuadModel(self.prior_cfg)
            x, y = self.dataset.total_observed
            bq.fit(x, y)
            # DEBUGTOM
            c = bq._int_cfg.model_cfg.get('scale_val', 1)
            integral = bq.model.integral_mean() * c
            logging.info(f'Risk: {integral}.')

        return self.return_and_save(integral)


class BayesQuadPointWiseRiskEstimator(_BayesQuadRiskEstimator):
    """Estimate Risk Using BayesQuad
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def estimate(
            self, predictions, observed, weights, surrogate,
            *args, acquisition_name=None, **kwargs):

        if not self.valid(acquisition_name):
            return self.return_and_save(1e19)

        # TODO: check if BQ specifically
        if (
                (surrogate is not None) and
                str(surrogate.__class__) == 'BayesQuadModel'):
            bq = surrogate

        else:
            from activetesting.models import BayesQuadModel
            bq = BayesQuadModel(self.prior_cfg)
            x, y = self.dataset.total_observed
            bq.fit(x, y)
        
        # does not make sense when sampling from pdf directly
        x_test = self.dataset.x[self.dataset.test_idxs]

        mean = bq.predict(x_test)
        # if noise is zero, check if predictions converge to empirical obs
        # x_to = self.dataset.x[self.dataset.test_observed]
        # y_to = self.dataset.y[self.dataset.test_observed]
        # bq = BayesQuadModel(self.prior_cfg)
        # bq.fit(x_to, y_to)
        # mean_observed, _ = bq.model.warped_gp.posterior_mean_and_variance(x_to)

        # integral = bq.model.integral_mean()
        integral = mean.mean()
        logging.info(f'Risk: {integral} pointwise')
        # logging.info(f'BQ Pointwise {integral}')
        return self.return_and_save(integral)


class FullSurrogateASMC(RiskEstimator):
    """Rely entirely on surrogate for ASMC. Do not update with observations.
    """
    def __init__(self, loss, dataset, model, risk_cfg=None, *args, **kwargs):
        super().__init__(loss)

        self.dataset = dataset
        self.task = self.dataset.cfg.task_type

        self.cfg = risk_cfg

        if self.cfg is not None and (lim := self.cfg.get('limit', False)):
            n_sub = len(self.dataset.test_idxs)
            self.test_idxs = np.random.choice(
                self.dataset.test_idxs, size=round(n_sub * lim), replace=False)
        else:
            self.test_idxs = self.dataset.test_idxs

        self.x_test = self.dataset.x[self.test_idxs]

        self.model = model

    # @profile
    def estimate(
            self, predictions, observed, weights, surrogate, acquisition_name,
            *args, **kwargs):

        if acquisition_name == 'RandomAcquisition':
            return self.return_and_save(-1e19)

        # Risk estimate using the surrogate model over all test data
        # ---------------------------------------
        # Remaining data
        if self.cfg is not None and self.cfg.get('increase_pool', False):
            if (len(observed) > self.cfg.get('after_m_steps', 0)):
                N_test = self.cfg['N_test']
                x_test = self.dataset.p.rvs(N_test)
                test_idxs = None
            else:
                return self.return_and_save(-1e19)
        else:
            x_test = self.x_test
            test_idxs = self.test_idxs

        model_predictions = self.model.predict(x_test, idxs=test_idxs)

        if surrogate is None:

            # A bit hacky. Fail silently.
            # This estimator will only return sensible values for acquisition
            # strategies that have a surrogate model.
            # This is ugly like this in the code because
            # estimators are always applied to all acquisition strategies.
            surr_predictions = model_predictions
        else:
            surr_predictions = surrogate.predict(x_test, idxs=test_idxs)

        if self.task == 'regression':
            # will return predictions as truth, will raise if used for
            # loss estimation for now (because model does not predict)
            # over all datapoints)
            R = self.loss(model_predictions, surr_predictions).mean()
        else:
            eps = 1e-20

            model_predictions = np.clip(model_predictions, eps, 1 - eps)
            model_predictions /= model_predictions.sum(axis=1, keepdims=True)

            R = -1 * (surr_predictions * np.log(model_predictions)).sum(-1)
            R = R.mean()

        return self.return_and_save(R)
