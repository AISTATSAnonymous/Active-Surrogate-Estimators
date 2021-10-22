import os
from math import e
import numpy as np
import logging

from scipy.stats import multivariate_normal
from scipy.optimize import curve_fit
from sklearn.base import clone
from torch.distributions.multivariate_normal import MultivariateNormal
import torch
import torch.nn as nn
import torch.nn.functional as F

from activetesting.models import BaseModel


class FixedLinearModel(BaseModel):
    def __init__(self, cfg):
        model = np.ones(cfg.data_CHW)
        model += cfg.weight_std * np.random.randn(cfg.data_CHW)
        super().__init__(cfg, model)

    def fit(self, *args, **kwargs):
        pass

    def predict(self, x, idxs=None, return_std=False, *args, **kwargs):
        out = x @ self.model

        if return_std:
            out = out, None
        return out


class DummyModel(BaseModel):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, model=None)

    def fit(self, *args, **kwargs):
        pass

    def predict(self, x, idxs=None, *args, return_std=False, **kwargs):
        if return_std:
            return np.array(len(x) * [-1e19]), np.array([-1e19])
        else:
            return np.array(len(x) * [-1e19])


class GaussianCurveFit(BaseModel):
    def __init__(self, cfg, *args, **kwargs):

        def gauss(x, *p):
            mean = p[:self.dim]
            log_std = p[self.dim:]
            mvn = multivariate_normal(
                mean=mean,
                cov=np.exp(log_std)**2 * np.eye(cfg.dim))

            return mvn.logpdf(x)

        self.dim = cfg.dim
        model = gauss
        super().__init__(cfg, model)

    def fit(self, x, y, *args, **kwargs):

        p0 = [* self.dim * [1.], *self.dim * [0.]]
        try:
            coeff, var_matrix = curve_fit(
                self.model, x, np.log(y), p0=p0,
                method='dogbox',
                # method='lm',
                bounds=(
                    [* self.dim * [0.], * self.dim * [-1.]],
                    [* self.dim * [100.], * self.dim * [2.]])
                    )
            self.p = coeff
            self.is_fit = True

            self.mean = torch.tensor(coeff[:self.dim])
            self.covariance_matrix = (
                torch.exp(torch.tensor(coeff[self.dim:]))**2
                * torch.eye(self.dim))

            logging.info(f'mean = {self.mean.mean()}')
            logging.info(f'std = {np.exp(coeff[self.dim:]).mean()}')

        except Exception as e:
            logging.info(f'Fitting failed with error: {e}.')
            self.is_fit = False

    def predict(self, x, idxs=None, return_std=False, *args, **kwargs):

        if self.is_fit:
            pred = np.exp(self.model(x, *self.p))
        else:
            pred = np.ones(len(x))

        if not return_std:
            return pred
        else:
            return pred, 0


class TorchGaussianCurveFit(BaseModel):
    def __init__(self, cfg, params=None):

        self.dim = cfg.dim
        self.loss = F.mse_loss
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._cfg = cfg
        self.init_params = params
        # if params is not None:
            # logging.info(f'Init means: {params[0].mean().item()}')
            # logging.info(f'Init stds: {torch.exp(params[1]).mean().item()}')

        model = self.init_model()

        self.is_fit = False
        super().__init__(cfg, model)

    def init_model(self, use_init_params=True):
        if self._cfg.get('mean_only', True):
            std = self._cfg.std
        else:
            std = None

        if use_init_params and self.init_params is not None:
            params = [i.clone() for i in self.init_params]
        else:
            params = None

        model = MVNModel(
            self.dim,
            self.device,
            init=[self._cfg.mean, self._cfg.std],
            std=std,
            params=params,
            std_center_init=self._cfg.get('std_center_init', False)
            ).to(
                self.device).double()

        return model

    @property
    def lr(self):
        val = {
            1: 1,
            5: 1,
            10: 1,
            25: 1,
            50: 1,
            100: 0.01,
          }.get(self.dim, 1)
        # if self.init_params is not None:
            # val *= 10

        return val

    def fit(self, x, y, *args, **kwargs):
        if len(x) == 0:
            self.is_fit = False
            return

        lr = self.lr

        x = torch.tensor(x).to(self.device)
        y = torch.tensor(y).to(self.device)

        def eval():
            optimizer.zero_grad()
            loss = F.mse_loss(self.model(x), torch.log(y))
            loss.backward()
            return loss

        retry = 0
        max_iter = 1000
        while retry < 4:
            try:
                optimizer = torch.optim.LBFGS(
                    self.model.parameters(),
                    lr=lr,
                    max_iter=max_iter,
                    # tolerance_change=1e-9,
                    # tolerance_grad=1e-6)
                )
                out = optimizer.step(eval)
                loss = F.mse_loss(self.model(x), torch.log(y))

                # if loss > 1:
                if loss > 1e-5:
                    raise ValueError(f'Loss too large: {loss}.')
                elif (
                        torch.any(self.model.means.isnan())
                        or torch.any(self.model.covariance.isnan())):
                    raise ValueError(f'Nans in mean or cov!')
                else:
                    break

                # stds = torch.exp(self.model.log_stds).mean().item()
                # means = self.model.means.mean().item()
                # c_std = (stds < 100) and (stds > 0.001)
                # c_mean = (means > 10) and (means < 100)
                # if c_std and c_mean:
                #     break
                # else:
                #     raise ValueError(
                #         f'Stds or means too large or small: {stds}, {means}.')

            except Exception as e:
                logging.info('Optimisation failed')
                logging.info(f'{e}')
                self.model = self.init_model(use_init_params=False)
                if retry > 0:
                    lr *= 0.1

                if retry > 2:
                    max_iter += 1000

                retry += 1

        if retry == 4:
            logging.warning('Continuing despite failed optimisation!!')
            os.system(f'echo warning >> warnings.txt')
            self.failed_optimisation = True
        else:
            self.failed_optimisation = False

        means = self.model.means.mean().item()
        stds = torch.exp(self.model.log_stds).mean().item()
        loss = F.mse_loss(self.model(x), torch.log(y))

        # if len(x) % 5 == 0:
        logging.info(
            f'Len (x) {len(x)}, '
            f'Loss {loss}, mean {means:.2f}, '
            f'std {stds:.2f}.'
                )
        self.is_fit = True

    def predict(
            self, x, idxs=None, return_std=False, check_fit=True,
            *args, **kwargs):

        if check_fit or self.is_fit:
            lim = int(1e5)
            x = torch.tensor(x)
            splits = torch.split(x, lim)
            preds = []
            for split in splits:
                split = split.to(self.device).double()
                with torch.no_grad():
                    pred = torch.exp(self.model(split))
                preds += [pred.cpu().numpy()]
            pred = np.concatenate(preds, 0)
        else:
            pred = np.ones(len(x))

        if not return_std:
            return pred
        else:
            return pred, 0

    def get_params(self):
        return (
            self.model.means.detach(),
            self.model.log_stds.detach())

    @property
    def mean(self):
        return self.model.f.mean

    @property
    def covariance_matrix(self):
        return self.model.f.covariance_matrix


class MVNModel(nn.Module):
    def __init__(
            self, dim, device, init, std=None, params=None,
            std_center_init=False):
        super().__init__()
        self.dim = dim
        self.device = device

        if params is not None:
            self.means = nn.Parameter(params[0])
            self.log_stds = nn.Parameter(params[1])

        else:
            if std is None:
                a = 2 if std_center_init else 0
                self.log_stds = nn.Parameter(
                    torch.log(torch.DoubleTensor(self.dim).uniform_(
                        init[1] - a, init[1]+2)))
            else:
                self.log_stds = torch.log(std * torch.ones(self.dim)).to(
                    self.device).double()

            # init around correct solution
            self.means = nn.Parameter(
                init[0] + torch.randn(self.dim),
                requires_grad=True)

    def get_params(self):
        return self.log_stds, self.means

    @property
    def covariance(self):
        return torch.exp(
                self.log_stds)**2 * torch.eye(self.dim, device=self.device)

    @property
    def f(self):
        return MultivariateNormal(
            loc=self.means,
            covariance_matrix=self.covariance)

    def forward(self, x):
        return self.f.log_prob(x)

    # def fit(self, x, y, *args, **kwargs):
    #     if len(x) == 0:
    #         return

    #     optimizer = torch.optim.SGD(
    #         self.model.parameters(),
    #         lr=0.1/self.dim, momentum=0.1, weight_decay=0)

    #     x = torch.tensor(x).to(self.device)
    #     y = torch.tensor(y).to(self.device)

    #     i, patience, old_means, old_stds = 0, 0, 1e10, 1e10

    #     while True:
    #         i += 1
    #         optimizer.zero_grad()
    #         pred = self.model(x)

    #         loss = F.mse_loss(pred, torch.log(y))
    #         loss.backward()
    #         optimizer.step()
    #         loss = loss.item()

    #         if i % 200 == 0:
    #             means = self.model.means.mean().item()
    #             stds = torch.exp(self.model.log_stds).mean().item()
    #             logging.info(
    #                 f'It {i}, loss {loss}, mean {means:.2f}, '
    #                 f'std {stds:.2f}.'
    #                 )
    #             c1 = np.abs(means - old_means) < 0.001
    #             c2 = np.abs(stds - old_stds) < 0.001

    #             if c1 and c2:
    #                 logging.info(f'Done at {i}')
    #                 break

    #             old_means = means
    #             old_stds = stds
