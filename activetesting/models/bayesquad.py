import logging
import numpy as np
import GPy

from bayesquad.gps import WsabiLGP
from bayesquad.priors import Gaussian
from bayesquad.quadrature import IntegrandModel

from .skmodels import BaseModel


def true_function(x):
    x = np.atleast_2d(x)
    return np.atleast_2d(np.sum(x, -1))


class BayesQuadModel(BaseModel):
    def __init__(self, cfg, *args, **kwargs):
        self.name = 'BayesQuadModel'
        super().__init__(cfg, model=None)
        # y merge fucks the interpolation in BaseModel
        self._int_cfg = cfg
        self.is_fit = False

    def fit(self, x, y):
        # DEBUGTOM
        y = y / self._int_cfg.model_cfg.get('scale_val', 1)

        cfg = self._int_cfg
        dim = cfg.model_cfg.data_CHW
        mean = cfg.model_cfg.prior_mean
        var = cfg.model_cfg.prior_std**2

        if (dist := cfg.model_cfg.test_distribution) == 'gaussian':
            prior = Gaussian(
                mean=mean * np.ones(dim), covariance=var*np.eye(dim))
        elif dist == 'correlated-gaussian':
            c_ij = cfg.model_cfg.test_cij
            cov = var * (np.eye(dim) + (np.ones(dim) - np.eye(dim)) * c_ij)
            prior = Gaussian(
                    mean=mean * np.ones(dim), covariance=cov)
        else:
            raise ValueError

        # k = GPy.kern.RBF(dim, variance=1, lengthscale=1)
        # k = GPy.kern.RBF(dim, variance=1, lengthscale=0.001)
        k = GPy.kern.RBF(dim, variance=25, lengthscale=2)
        lik = GPy.likelihoods.Gaussian(
            variance=cfg.model_cfg.lik_var)

        y = np.atleast_2d(np.sqrt(2 * y))

        if cfg.model_cfg.get('mean_mean_function', False):

            mf = GPy.core.Mapping(dim, 1)
            # y is already sqrt transformed
            mf.f = lambda *args, **kwargs: np.mean(y)
            mf.update_gradients = lambda a, b: None
            self.gpy_gp = GPy.core.GP(
                x, y.T, kernel=k, likelihood=lik, mean_function=mf)
        else:
            self.gpy_gp = GPy.core.GP(x, y.T, kernel=k, likelihood=lik)

        self.warped_gp = WsabiLGP(self.gpy_gp)
        self.model = IntegrandModel(self.warped_gp, prior)
        import logging.config
        logging.config.dictConfig({
            'version': 1,
            'disable_existing_loggers': True,
        })

        # self.model.update(x, y)
        self.gpy_gp.optimize()
        self.is_fit = True

    def predict(self, x, idxs=None, return_std=False, **kwargs):
        # Sklearn expects predict to be pointwise
        # mean of this is our integral estimate
        # return np.array([self.model.integral_mean()] * len(x))

        if self.is_fit:
            y, _ = self.warped_gp.posterior_mean_and_variance(x)
            # if y.shape[1] == 1:
            #     y = y[:, 0]
            return y * self._int_cfg.model_cfg.get('scale_val', 1)
        else:
            return np.ones(len(x))/len(x)


# class BayesQuadModel(BaseModel):
#     def __init__(self, cfg, *args, **kwargs):
#         self.name = 'BayesQuadModel'
#         dim = cfg.model_cfg.data_CHW
#         mean = cfg.model_cfg.prior_mean
#         var = cfg.model_cfg.prior_std**2

#         if (dist := cfg.model_cfg.test_distribution) == 'gaussian':
#             prior = Gaussian(
#                 mean=mean * np.ones(dim), covariance=var*np.eye(dim))
#         elif dist == 'correlated-gaussian':
#             c_ij = cfg.model_cfg.test_cij
#             cov = var * (np.eye(dim) + (np.ones(dim) - np.eye(dim)) * c_ij)
#             prior = Gaussian(
#                     mean=mean * np.ones(dim), covariance=cov)
#         else:
#             raise ValueError

#         k = GPy.kern.RBF(dim, variance=var, lengthscale=2)
#         lik = GPy.likelihoods.Gaussian(
#             variance=cfg.model_cfg.get('lik_var', 1e-11))
#         initial_x = np.array([dim * [mean]])
#         initial_y = np.sqrt(2 * true_function(initial_x))

#         # mf = GPy.core.Mapping(dim, 1)
#         # mf.f = lambda *args, **kwargs: np.sqrt(2 * mean * dim)
#         # mf.update_gradients = lambda a,b: None

#         # self.gpy_gp = GPy.core.GP(
#         #     initial_x, initial_y.T, kernel=k, likelihood=lik,
#         #     mean_function=mf)

#         self.gpy_gp = GPy.core.GP(
#             initial_x, initial_y.T, kernel=k, likelihood=lik)

#         self.warped_gp = WsabiLGP(self.gpy_gp)
#         model = IntegrandModel(self.warped_gp, prior)
#         import logging.config
#         logging.config.dictConfig({
#             'version': 1,
#             'disable_existing_loggers': True,
#         })

#         super().__init__(cfg, model)
#         self.is_fit = False

#     def fit(self, x, y):
#         # y = np.atleast_2d(y)
#         self.model.update(x, y)
#         self.gpy_gp.optimize()
#         self.is_fit = True

#     def predict(self, x, idxs=None, return_std=False, **kwargs):
#         # Sklearn expects predict to be pointwise
#         # mean of this is our integral estimate
#         # return np.array([self.model.integral_mean()] * len(x))

#         if self.is_fit:
#             y, _ = self.warped_gp.posterior_mean_and_variance(x)
#             # if y.shape[1] == 1:
#             #     y = y[:, 0]
#             return y
#         else:
#             return np.ones(len(x))/len(x)
