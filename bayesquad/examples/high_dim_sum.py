"""Plot bayesian quadrature on a simple 2D test function."""

from typing import Dict, Any

import GPy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import AxesImage

import bayesquad.plotting as plotting
from bayesquad.batch_selection import select_batch, LOCAL_PENALISATION
from bayesquad.gps import WsabiLGP
from bayesquad.priors import Gaussian
from bayesquad.quadrature import IntegrandModel


# Set up test function and WSABI-L model.

DIM = 3
MEAN = 5
VAR = 2
TRUE = DIM * MEAN
print(f'True Integral Value: {TRUE}')


def true_function(x):
    x = np.atleast_2d(x)
    return np.atleast_2d(np.sum(x, -1))


prior = Gaussian(mean=np.array(DIM * [MEAN]), covariance=VAR*np.eye(DIM))
k = GPy.kern.RBF(DIM, variance=VAR, lengthscale=2)
lik = GPy.likelihoods.Gaussian(variance=1e-10)

initial_x = np.array([DIM * [MEAN]])
# initial_x = np.atleast_2d(prior.sample(20))
# initial_y = true_function(initial_x)
initial_y = np.sqrt(2 * true_function(initial_x))
gpy_gp = GPy.core.GP(initial_x, initial_y.T, kernel=k, likelihood=lik)
warped_gp = WsabiLGP(gpy_gp)
model = IntegrandModel(warped_gp, prior)

X = np.atleast_2d(prior.sample(500))
Y = true_function(X)
model.update(X, Y)
gpy_gp.optimize()
print("Initial estimate: {}".format(model.integral_mean()))


def true_integrand(x):
    return true_function(x) * prior(x)


# Set up plotting.
if DIM == 2:

    LOWER_LIMIT = MEAN - 5 * np.sqrt(VAR)
    UPPER_LIMIT = MEAN + 5 * np.sqrt(VAR)
    print(f'Plotting limits {LOWER_LIMIT}, {UPPER_LIMIT}.')
    PLOTTING_RESOLUTION = 200
    COLOUR_MAP = 'summer'

    def get_plotting_domain(lower_limit, upper_limit, resolution):
        x = []
        for _ in range(DIM):
            x += [np.linspace(lower_limit, upper_limit, resolution)]
        x_grids = np.meshgrid(*x)
        return np.concatenate(np.dstack(x_grids))

    figure = plt.figure(figsize=(18, 6))
    images: Dict[Any, AxesImage] = {}

    PLOTTING_DOMAIN = get_plotting_domain(LOWER_LIMIT, UPPER_LIMIT, PLOTTING_RESOLUTION)

    def plot_data(data, subplot, title=""):
        figure = plt.figure(figsize=(18, 6))
        images: Dict[Any, AxesImage] = {}
        PLOTTING_DOMAIN = get_plotting_domain(LOWER_LIMIT, UPPER_LIMIT, PLOTTING_RESOLUTION)

        data = data.reshape(PLOTTING_RESOLUTION, PLOTTING_RESOLUTION)

        if subplot in images:
            image = images[subplot]
            image.set_data(data)
            image.set_clim(vmin=data.min(), vmax=data.max())
        else:
            axis = figure.add_subplot(subplot)
            image = axis.imshow(data, cmap=plt.get_cmap(COLOUR_MAP), vmin=data.min(), vmax=data.max(),
                                extent=[LOWER_LIMIT, UPPER_LIMIT, LOWER_LIMIT, UPPER_LIMIT],
                                interpolation='nearest', origin='lower')
            images[subplot] = image

            axis.set_title(title)

        plt.show()

    def plot_true_function():
        z = true_integrand(PLOTTING_DOMAIN)
        plot_data(z, 133, "True Integrand")

    def plot_integrand_posterior(integrand_model: IntegrandModel):
        z = integrand_model.posterior_mean_and_variance(PLOTTING_DOMAIN)[0]
        plot_data(z, 132, "Posterior Mean")

    def plotting_callback(func):
        z = np.exp(func(PLOTTING_DOMAIN)[0])
        plot_data(z, 131, "Acquisition Function")

    plotting.add_callback("Soft penalised log acquisition function", plotting_callback)
    plot_true_function()


# Run algorithm.

BATCHES = 25
BATCH_SIZE = 4
BATCH_METHOD = LOCAL_PENALISATION
# print("Initial estimate: {}".format(model.integral_mean()))

for i in range(BATCHES):
    if DIM == 2:
        plot_integrand_posterior(model)

    batch = select_batch(model, BATCH_SIZE, BATCH_METHOD)

    X = np.array(batch)
    # I can just replace this with whatever point is closest

    Y = true_function(X)
    print(X.shape, Y.shape, X)
    model.update(X, Y)

    gpy_gp.optimize()

    print("Integral: {}".format(model.integral_mean()))

if DIM == 2:
    plot_integrand_posterior(model)
    plt.show()
