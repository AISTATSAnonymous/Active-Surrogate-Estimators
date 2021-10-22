"""Datasets for active testing."""

from copy import deepcopy
import warnings
import logging
import pickle
from pathlib import Path
from numpy.lib.arraypad import _round_if_needed
from omegaconf import OmegaConf
import hydra

import numpy as np
from sklearn.model_selection import train_test_split as SKtrain_test_split
from scipy.stats import multivariate_normal

import torch
from torch._C import Value
from torch.distributions.multivariate_normal import MultivariateNormal

from activetesting.utils.data import (
    to_json, get_root, array_reduce)


class _Dataset:
    """Implement generic dataset.

    Load and preprocess data.
    Provide basic acess, train-test split.

    raise generic methods
    """
    def __init__(self, cfg):

        # Set task_type and global_std if not present.
        self.cfg = OmegaConf.merge(
                OmegaConf.structured(cfg),
                dict(
                    task_type=cfg.get('task_type', 'regression'),
                    global_std=cfg.get('global_std', False),
                    n_classes=cfg.get('n_classes', -1)))

        self.N = cfg.n_points
        self.x, self.y = self.generate_data()

        # For 1D data, ensure Nx1 shape
        if self.x.ndim == 1:
            self.x = self.x[:, np.newaxis]

        self.D = self.x.shape[1:]

        self.train_idxs, self.test_idxs = self.train_test_split(self.N)
        self.x_test = self.x[self.test_idxs]

        to_json(
            dict(
                train_idxs=self.train_idxs.tolist(),
                test_idxs=self.test_idxs.tolist()),
            Path('train_test_split.json'))

        if self.cfg.standardize:
            self.standardize()

    def train_test_split(self, N, test_size=None):
        if (N < self.y.size) and self.cfg.get('with_unseen'):
            all_indices = np.random.choice(
                np.arange(0, self.y.size),
                N,
                replace=False)
        else:
            all_indices = np.arange(0, N)

        if self.cfg.get('stratify', False):
            stratify = self.y[all_indices]
        else:
            stratify = None

        if test_size is None:
            test_size = self.cfg.test_proportion

        if test_size == 1:
            train = np.array([]).astype(np.int)
            test = all_indices
        else:
            train, test = SKtrain_test_split(
                    all_indices, test_size=test_size,
                    stratify=stratify)

        assert np.intersect1d(train, test).size == 0
        assert np.setdiff1d(
            np.union1d(train, test),
            all_indices).size == 0

        # this option splits the test into test and unseen
        if p := self.cfg.get('test_unseen_proportion', False):
            if (N < self.y.size) and self.cfg.get('with_unseen'):
                raise ValueError('Not compatible.')
            test, test_unseen_idxs = SKtrain_test_split(
                np.arange(0, len(test)), test_size=p)
            self.test_unseen_idxs = test_unseen_idxs

        # this option takes all test idxs as unseen
        if self.cfg.get('with_unseen', False):
            test_unseen_idxs = np.setdiff1d(
                np.arange(0, self.y.size), train)
            self.test_unseen_idxs = test_unseen_idxs

        if self.cfg.get('remove_initial_75_keep_train_size', False):
            # need to do this here, before unseen is applied
            train, test_unseen_idxs = self.remove_initial_75_keep_train_size(
                train, test, self.test_unseen_idxs, self.y)
            self.test_unseen_idxs = test_unseen_idxs

        if (freq := self.cfg.get('filter_nums_relative_frequency', 1)) != 1:
            test = self.reduce_filter_num_frequency(freq, test)

        assert np.intersect1d(train, test).size == 0

        if self.cfg.get('with_unseen', False):
            assert np.intersect1d(test_unseen_idxs, train).size == 0
            assert np.intersect1d(
                test_unseen_idxs, test).size == test.size
        if self.cfg['task_type'] == 'classification':
            logging.info(
                f'Final bincount for test {np.bincount(self.y[test])}.')

        return train, test

    def reduce_filter_num_frequency(self, freq, test):
        nums = self.cfg.filter_nums

        _7_bin = array_reduce(self.y, nums)
        _7 = np.where(_7_bin)
        test_7 = np.intersect1d(test, _7)
        assert np.all(array_reduce(self.y[test_7], nums))

        old_num_7 = len(test_7)
        new_num_7 = round(freq * old_num_7)

        # draw those indices to delete
        delete_7 = np.random.choice(
            test_7, size=old_num_7 - new_num_7, replace=False)

        new_test = np.setdiff1d(test, delete_7)

        # delete 7 no longer in teset
        assert np.intersect1d(new_test, delete_7).size == 0
        # test been reduced by delete 7 amount
        assert len(test) - len(new_test) == len(delete_7)
        # deleted as many as we wanted
        assert len(delete_7) == old_num_7 - new_num_7
        # new num count is reduced by frequency
        assert array_reduce(self.y[new_test], nums).sum() == new_num_7

        return new_test

    def remove_initial_75_keep_train_size(
            self, train, test, test_unseen_idxs, y):

        # need to have extra data, because we don't want to copy from test
        if not self.cfg.get('with_unseen', False):
            raise ValueError

        nums = self.cfg.filter_nums
        # find number of 5s and 7s in train
        # train_7_bin = (y[train] == 7) | (y[train] == 5)
        train_7_bin = array_reduce(y[train], nums)

        train_7 = np.where(train_7_bin)[0]
        n_replace = len(train_7)

        # assert np.all((y[train[train_7]] == 5) | (y[train[train_7]] == 7))
        assert np.all(array_reduce(y[train][train_7], nums))

        # find some *unseen* idxs that are not 5 or 7
        y = self.y
        unseen_not_test = np.setdiff1d(test_unseen_idxs, test)
        assert np.intersect1d(unseen_not_test, train).size == 0
        assert np.intersect1d(unseen_not_test, test).size == 0
        unseen_not_test_bin = np.zeros(y.size, dtype=np.bool)
        unseen_not_test_bin[unseen_not_test] = 1
        # unseen_not_75_bin = unseen_not_test_bin & (y != 7) & (y != 5)
        unseen_not_75_bin = unseen_not_test_bin & array_reduce(
            y, nums, compare='neq', combine='and')
        unseen_not_75 = np.where(unseen_not_75_bin)[0]

        # assert np.all((y[unseen_not_75] != 7) | (y[unseen_not_75] != 5))
        assert np.all(array_reduce(y[unseen_not_75], nums, compare='neq'))

        # chose as many as we want to swap
        chosen = np.random.choice(unseen_not_75, n_replace, replace=False)

        # remove the chosen ones from the unseen idx
        # I don't want to bias my true loss
        # (don't need to remove from test because they where never in test)
        pre_size = len(test_unseen_idxs)
        test_unseen_idxs = np.setdiff1d(test_unseen_idxs, chosen)
        post_size = len(test_unseen_idxs)
        assert pre_size - post_size == len(chosen)
        assert np.intersect1d(test_unseen_idxs, chosen).size == 0

        # now replace the train idxs with the chosen idxs
        train[train_7] = chosen

        # assert np.all((y[train[train_7]] != 5) | (y[train[train_7]] != 7))
        # assert np.all((y[train] != 5) | (y[train] != 7))
        assert np.all(array_reduce(y[train[train_7]], nums, compare='neq'))
        assert np.all(array_reduce(y[train], nums, compare='neq'))

        assert np.intersect1d(test_unseen_idxs, train).size == 0
        assert np.intersect1d(
            test_unseen_idxs, test).size == test.size

        return train, test_unseen_idxs

    @property
    def train_data(self):
        return self.x[self.train_idxs], self.y[self.train_idxs]

    @property
    def test_data(self):
        return self.x[self.test_idxs], self.y[self.test_idxs]

    def standardize(self):
        """Standardize to zero mean and unit variance using train_idxs."""

        ax = None if self.cfg['global_std'] else 0

        x_train, y_train = self.train_data

        x_std = self.cfg.get('x_std', True)
        if x_std:
            self.x_train_mean = x_train.mean(ax)
            self.x_train_std = x_train.std(ax)
            self.x = (self.x - self.x_train_mean) / self.x_train_std

        y_std = self.cfg.get('y_std', True)
        if (self.cfg['task_type'] == 'regression') and y_std:
            self.y_train_mean = y_train.mean(ax)
            self.y_train_std = y_train.std(ax)
            self.y = (self.y - self.y_train_mean) / self.y_train_std

    def export(self):
        package = dict(
            x=self.x,
            y=self.y,
            train_idxs=self.train_idxs,
            test_idxs=self.test_idxs
            )
        return package


class _ActiveTestingDataset(_Dataset):
    """Active Testing Dataset.

    Add functionality for active testing.

    Split test data into observed unobserved.

    Add Methods to keep track of unobserved/observed.
    Use an ordered set or sth to keep track of that.
    Also keep track of activation function values at time that
    sth was observed.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.start()

    def start(self):
        self.test_observed = np.array([], dtype=np.int)
        self.test_remaining = self.test_idxs

    def restart(self, *args):
        self.start()

    def observe(self, idx, with_replacement=True):
        """Observe data at idx and move from unobserved to observed.

        Note: For efficiency reasons idx is index in test
        """

        self.test_observed = np.append(self.test_observed, idx)
        if not with_replacement:
            self.test_remaining = self.test_remaining[
                self.test_remaining != idx]

        return self.x[[idx]], self.y[[idx]]

    @property
    def total_observed(self):
        """Return train and observed test data"""
        test = self.x[self.test_observed], self.y[self.test_observed]
        train = self.train_data
        # concatenate x and y separately
        total_observed = [
            np.concatenate([train[i], test[i]], 0)
            for i in range(2)]

        return total_observed


class QuadraticDatasetForLinReg(_ActiveTestingDataset):
    """Parabolic data for use with linear regression – proof of concept."""
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg)

    def generate_data(self):
        x = np.linspace(0, 1, self.N)
        y = x**2
        y -= np.mean(y)
        return x, y


class SinusoidalDatasetForLinReg(_ActiveTestingDataset):
    """Sinusoidal data for use with linear regression – proof of concept.

    This dataset has a high and a low-density region.
    A clever acquisition strategy is necessary to estimate the error correctly.

    """
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg)

    def generate_data(self):
        def regression_function(min_x, max_x, n_samples):
            x = np.linspace(min_x, max_x, n_samples)
            y = np.sin(x * 10) + x ** 3
            return (x, y)

        def split_dataset(n_total, min_x=0, max_x=2, center=1):
            """Split regression function into high and low density regions."""

            # if getattr(self.cfg, 'low_density_train_only', False):
            #     n_low = int(0.5 * n_total)
            #     n_high = int(0.5 * n_total)
            # else:
            n_low = int(0.1 * n_total)
            n_high = int(0.9 * n_total)

            low_density_data = regression_function(
                min_x, center - 0.01, n_low)
            high_density_data = regression_function(
                center, max_x, n_high)

            x = np.concatenate([
                low_density_data[0], high_density_data[0]], 0)
            y = np.concatenate([
                low_density_data[1], high_density_data[1]], 0)

            n_low = len(low_density_data[0])

            return x, y, n_low

        x, y, self.n_low = split_dataset(self.N)

        # TODO: add back!?
        # y = y - np.mean(y)

        return x, y

    def train_test_split(self, *args):
        """Need to overwrite train_test_split.
        Stratify across low and high_density regions.
        """
        n_low = self.n_low
        n_high = self.N - n_low

        if getattr(self.cfg, 'low_density_train_only', False):
            ts = int(0.5 * n_low)
        else:
            ts = 4

        low_train, low_test = super().train_test_split(n_low, test_size=ts)
        high_train, high_test = super().train_test_split(n_high)
        high_train += n_low
        high_test += n_low

        train = np.concatenate([low_train, high_train], 0)
        test = np.concatenate([low_test, high_test], 0)

        return train, test


class GPDatasetForGPReg(_ActiveTestingDataset):
    """Sample from GP prior."""
    def __init__(self, cfg, model_cfg, *args, **kwargs):
        self.model_cfg = model_cfg
        super().__init__(cfg)

    def generate_data(self):
        from activetesting.utils import maps
        self.model = maps.model[self.model_cfg.name](self.model_cfg)
        self.aleatoric = getattr(self.model_cfg, 'with_noise', 0)
        xmax = self.cfg.get('xmax', 1)
        x = np.linspace(0, xmax, self.N)[:, np.newaxis]
        y = self.model.sample_y(x, random_state=np.random.randint(0, 10000))
        return x, y


class MNISTDataset(_ActiveTestingDataset):
    """MNIST Data.

    TODO: Respect train/test split of MNIST.
    """
    def __init__(self, cfg, n_classes=10, *args, **kwargs):

        cfg = OmegaConf.merge(
            OmegaConf.structured(cfg),
            dict(task_type='classification', global_std=True,
                 n_classes=n_classes))

        super().__init__(cfg)

    # def generate_data(self):
    #     from tensorflow.keras import datasets

    #     # data_home = Path(hydra.utils.get_original_cwd()) / 'data/MNIST'

    #     # # from sklearn.datasets import fetch_openml
    #     # # x, y = fetch_openml(
    #     # #     'mnist_784', version=1, return_X_y=True, data_home=data_home,
    #     # #     cache=True)
    #     # data = datasets.mnist.load_data(
    #     #     path=data_home / 'mnist.npz'
    #     # )

    #     data = datasets.mnist.load_data()

    #     return self.preprocess(data)

    def generate_data(self):
        from torchvision.datasets import MNIST

        train = MNIST(get_root() / 'data/torch_mnist', download=True)
        x_train, y_train = train.data, train.targets
        test = MNIST(
            get_root() / 'data/torch_mnist', download=False,
            train=False)
        x_test, y_test = test.data, test.targets

        data = ((x_train, y_train), (x_test, y_test))

        return self.preprocess(data)

    def preprocess(self, data):

        (x_train, y_train), (x_test, y_test) = data
        x = np.concatenate([x_train, x_test], 0)
        x = x.astype(np.float32) / 255
        x = x.reshape(x.shape[0], -1)
        y = np.concatenate([y_train, y_test], 0)
        y = y.astype(np.int)

        N = self.N

        if (N < y.size) and not self.cfg.get('with_unseen', False):
            logging.info('Keeping only a subset of the input data.')
            # get a stratified subset
            # note that mnist does not have equal class count
            # want to keep full data for unseeen
            idxs, _ = SKtrain_test_split(
                np.arange(0, y.size), train_size=N, stratify=y)
            x = x[idxs]
            y = y[idxs]
        # no longer want this. keep indices always true to dataset!!
        # elif (N < y.size) and not self.cfg.get('respect_train_test', False):
        #     logging.info('Scrambling input data.')
        #     # still want to scramble because the first N entries are used
        #     # to assign test and train data
        #     idxs = np.random.permutation(np.arange(0, y.size))
        #     x = x[idxs]
        #     y = y[idxs]

        return x, y

    def train_test_split(self, N):

        if self.cfg.get('respect_train_test', False):
            # use full train set, subsample from original test set
            train_lim = self.cfg.get('train_limit', 50000)
            train = np.arange(0, train_lim)

            n_test = round(self.cfg.test_proportion * N)
            max_test = 60e3 - train_lim
            if n_test <= max_test:

                replace = self.cfg.get('test_with_replacement', False)

                test = np.random.choice(
                    np.arange(train_lim, 60000), n_test, replace=replace)
                test = np.sort(test)
            else:
                raise ValueError

            self.test_unseen_idxs = np.setdiff1d(
                np.arange(train_lim, 60000), test)

            return train, test

        else:
            train, test = super().train_test_split(N)

        # only keep the first n sevens in the train distribution
        if n7 := self.cfg.get('n_initial_7', False):
            # to get correct indices, need to first select from y
            old7 = np.where(self.y == 7)[0]
            # then filter to train indicees
            old_train7 = np.intersect1d(old7, train)
            # now only keep the first n7
            sevens_remove = old_train7[n7:]
            # and now remove those from the train set
            train = np.setdiff1d(train, sevens_remove)

        return train, test


class TwoMoonsDataset(_ActiveTestingDataset):
    """TwoMoons Data."""
    def __init__(self, cfg,
                 *args, **kwargs):

        cfg = OmegaConf.merge(
            OmegaConf.structured(cfg),
            dict(task_type='classification', global_std=False, n_classes=2))

        super().__init__(cfg)

    def generate_data(self):

        from sklearn.datasets import make_moons
        x, y = make_moons(n_samples=self.cfg.n_points, noise=self.cfg.noise)

        return x, y


class FashionMNISTDataset(MNISTDataset):
    """FashionMNIST Data.

    TODO: Respect train/test split of FashionMNIST.
    """
    def __init__(self, cfg,
                 *args, **kwargs):

        super().__init__(cfg)

    # def generate_data(self):
    #     from tensorflow.keras import datasets
    #     data = datasets.fashion_mnist.load_data()

    #     return self.preprocess(data)

    def generate_data(self):
        from torchvision.datasets import FashionMNIST

        train = FashionMNIST(get_root() / 'data/torch_fmnist', download=True)
        x_train, y_train = train.data, train.targets
        test = FashionMNIST(
            get_root() / 'data/torch_fmnist',
            train=False, download=False)
        x_test, y_test = test.data, test.targets

        data = ((x_train, y_train), (x_test, y_test))

        return self.preprocess(data)


class Cifar10Dataset(MNISTDataset):
    """CIFAR10 Data.
    """
    def __init__(self, cfg,
                 *args, **kwargs):

        super().__init__(cfg)

    def generate_data(self):
        from tensorflow.keras import datasets
        data = datasets.cifar10.load_data()

        x, y = self.preprocess(data)
        x = x.reshape(len(x), 32, 32, 3).transpose(0, 3, 1, 2)
        x = x.reshape(len(x), -1)
        return x, y[:, 0]


class Cifar100Dataset(MNISTDataset):
    """CIFAR100 Data.
    """
    def __init__(self, cfg,
                 *args, **kwargs):

        super().__init__(cfg, n_classes=100)

    def generate_data(self):
        from tensorflow.keras import datasets
        data = datasets.cifar100.load_data()

        x, y = self.preprocess(data)
        x = x.reshape(len(x), 32, 32, 3).transpose(0, 3, 1, 2)
        x = x.reshape(len(x), -1)
        return x, y[:, 0]


class ToyDataset(_ActiveTestingDataset):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg)

    def generate_data(self):

        path = Path(get_root()) / 'toy_data' / self.cfg.data_path
        # path = Path(get_root / 'toy_data' / self.cfg.data_path
        data_dict = pickle.load(open(path, 'rb'))

        data = data_dict['data']
        args = data_dict['args']
        self.creation_args = args

        logging.info(f'Loaded ToyDataset created from {path}.')
        logging.info(f'Dataset was created at {data_dict["time"]}.')
        logging.info(f'Args of creation were {args}.')

        x_train, y_train, x_test, y_test = [
            data[i] for i in ['x_train', 'y_train', 'x_test', 'y_test']]

        if noise := self.cfg.get('add_test_outcome_noise', False):
            y_test += noise * np.random.randn(*y_test.shape)

        if noise := self.cfg.get('add_train_outcome_noise', False):
            y_train += noise * np.random.randn(*y_train.shape)

        x = np.concatenate([x_train, x_test])
        y = np.concatenate([y_train, y_test])

        return x, y

    def train_test_split(self, N):

        if self.cfg.get('new_respect_train_test', False):
            # subsample from both train set and test set
            orig_train = self.creation_args['train_n']
            orig_test = self.creation_args['test_n']
            orig = orig_train + orig_test

            n = self.cfg['n_keep']
            train_n = round(n * self.cfg.train_proportion)
            test_n = round(n * self.cfg.test_proportion)

            assert test_n <= orig_test
            assert train_n <= orig_train

            train = np.random.choice(
                np.arange(0, orig_train), train_n, replace=False)

            test = np.random.choice(
                np.arange(orig_train, orig), test_n, replace=False)

            assert test.min() > train.max(), 'Overlap between test and train?'

            return train, test

        elif self.cfg.get('respect_train_test', False):
            # use full train set, subsample from original test set
            train_n = self.creation_args['train_n']
            test_n = self.creation_args['test_n']

            n = train_n + test_n

            train = np.arange(0, train_n)
            n_test = int(self.cfg.test_proportion * n)

            test = np.random.choice(
                np.arange(train_n, n), n_test, replace=False)

            assert test.min() > train.max()

            return train, test

        else:
            train, test = super().train_test_split(N)


class OnlineToyDataset(_ActiveTestingDataset):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg)

    def data_generator(self, add=''):

        args = self.creation_args

        dist = args[f'{add}distribution']
        std = args[f'{add}std']
        N, D = args[f'{add}n'], args[f'n_pixels']
        if dist == 'gaussian':
            mean = args[f'{add}mean']
            p = multivariate_normal(
                mean=mean * np.ones(D), 
                cov=std**2 * np.eye(D))
            x = p.rvs(N)

        elif dist == 'lognormal':
            mean = args[f'{add}mean']
            x = np.random.lognormal(mean=mean, sigma=std, size=(N, D))
            p = None
            warnings.warn('Missing explicit p for IS.')

        elif dist == 'correlated-gaussian':
            mean = args[f'{add}mean']
            c_ij = args[f'{add}c_ij']
            cov = std**2 * (np.eye(D) + (np.ones(D) - np.eye(D)) * c_ij)

            p = multivariate_normal(
                mean=mean * np.ones(D),
                cov=cov)

            x = p.rvs(N)
        else:
            p = None
            warnings.warn('Missing explicit p for IS.')

            # draw a mean for each image
            prior, dist = dist.split('-')

            if prior == 'unif':
                means = np.random.uniform(low=1, high=10, size=N)
            elif prior == 'expunif':
                means = np.exp(np.random.uniform(2, 10, size=N))

            x = []
            for mean in means:
                if dist == 'gaussian':
                    samples = np.random.normal(loc=mean, scale=std, size=D)
                elif dist == 'lognormal':
                    samples = np.random.lognormal(mean=mean, sigma=std, size=D)
                else:
                    raise ValueError

                x.append(samples)

            x = np.stack(x, 0)

        if args.get('abs', False):
            x = np.abs(x)

        y = self.y_at_any(x)

        if D == 1:
            x = x[:, np.newaxis]

        return x, y, p

    def generate_data(self):
        # from activetesting.utils.generate_toy_data import generate_data as gd
        # from activetesting.utils.generate_toy_data import get_parser

        self.creation_args = dict()
        p = self.cfg.test_proportion

        self.creation_args['test_n'] = self.cfg.n_points
        self.creation_args.update(self.cfg.creation_args)

        logging.info(self.creation_args)

        x, y, self.p = self.data_generator(add='test_')

        if noise := self.cfg.get('add_outcome_noise', False):
            y += noise * np.random.randn(*y.shape)

        return x, y

    def y_at_any(self, x):

        if len(x.shape) == 1:
            x = x[:, np.newaxis]

        if not self.creation_args['normalise']:
            y = x.sum(1)
        else:
            y = x.mean(1)

        return y


class DoubleGaussianDataset(_ActiveTestingDataset):
    """p(x) Gaussian, f(x) Gaussian."""
    def __init__(self, cfg, *args, **kwargs):
        self.C = cfg.creation_args
        super().__init__(cfg)

    def restart(self, acquisition):
        if 'SampleFromPDFDirectly' in acquisition:
            logging.info('Restarting DoubleGaussianDataset in PDF mode!')
            self.activate_pdf_mode()
            # arrays I need to keep working
            # self.dataset.test_remaining --> where is this used?!
            # --> maybe just keep a dummy around? I just care about exactexpectedriskesitmator
            # self.dataset.test_observed
            # --> this is easy, just do a second list --> self.dataset.y[self.dataset.test_observed] needs to work though
            # --> maybe append the observed ones at the end?!
            # self.dataset.total_observed --> will continue to work if I fix test_observed

        else:
            self.deactivate_pdf_mode()

        super().restart()
    
    def y_at_any(self, x):
        return self.f.pdf(x)

    def activate_pdf_mode(self):
        # this will set all remaining to 1e19! --> because they are unknown!
        self.data_copy = [deepcopy(self.x), deepcopy(self.y)]

        self.x[self.test_idxs] = self.x[self.test_idxs] * 0 + 1e19
        self.y[self.test_idxs] = self.y[self.test_idxs] * 0 + 1e19

        # self.observe = self.pdf_observe

    def deactivate_pdf_mode(self):
        # self.observe = super().observe
        if copy := getattr(self, 'data_copy', False):
            self.x, self.y = copy

    def generate_data(self):
        C = self.C
        D = C.dim

        cov = C.p_std**2 * np.eye(D)

        if c_ij := C.get('c_ij', False):
            cov = cov + (np.ones(D) - np.eye(D)) * c_ij

        self.p = multivariate_normal(
            mean=C.p_mean * np.ones(D),
            cov=cov)

        self.f = multivariate_normal(
            mean=C.f_mean * np.ones(D),
            cov=C.f_std**2 * np.eye(D))

        x = self.p.rvs(self.N)
        y = self.f.pdf(x)

        # if noise := self.cfg.get('add_outcome_noise', False):
        #     print('adding noise')
        #     # multiplicative noise sucks, because this changes expectation
        #     y *= np.random.lognormal(mean=1, sigma=noise, size=y.shape)
        #     # additive noise sucks, because this introduces error
        #     y += noise * np.random.normal(size=y.shape)
        #     y = np.maximum(y, 1e-20)
        return x, y

    @property
    def torch_p(self):
        return MultivariateNormal(
            loc=torch.from_numpy(self.p.mean),
            covariance_matrix=torch.from_numpy(self.p.cov))


def get_CIFAR10():
    """From pruning code. Only used for debugging purposes."""

    import torch
    from torchvision import transforms, datasets
    root = get_root()

    input_size = 32
    num_classes = 10
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train_dataset = datasets.CIFAR10(
        root + "/CIFAR10", train=True, transform=train_transform, download=True
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    test_dataset = datasets.CIFAR10(
        root + "/CIFAR10", train=False, transform=test_transform,
        download=False
    )

    kwargs = {"num_workers": 4, "pin_memory": True}
    batch_size = 128

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **kwargs
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1000, shuffle=False, **kwargs
    )

    return train_loader, test_loader
