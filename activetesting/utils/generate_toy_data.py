# Contains script to generate toy data.
import sys
import logging
import warnings
import argparse
import pickle
from pathlib import Path
from datetime import datetime
from scipy.stats import multivariate_normal

import numpy as np

from activetesting.utils.data import get_root


def main(args):
    logging.info('Generating training data.')
    x_train, y_train = generate_data(args, add='train_')
    logging.info('Generating test data.')
    x_test, y_test = generate_data(args, add='test_')

    logging.info('Saving data to file.')

    data = dict(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test)

    out = dict(
        data=data,
        args=args,
        time=datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))

    root = Path(get_root()) / 'toy_data'
    root.mkdir(exist_ok=True, parents=True)
    name = get_name(args)

    with open(root / name, 'wb') as f:
        pickle.dump(out, f)

    logging.info(f'Wrote to {root / name}.')
    logging.info('Finished successfully.')


def generate_data(args, add=''):
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

    if not args['normalise']:
        y = x.sum(1)
    else:
        y = x.mean(1)

    return x, y, p


def get_name(args):
    name = [f'{i}-{j}' for i, j in args.items()]
    name = ('_').join(name)
    name = 'TOY_DATA__' + name + '.pkl'
    return name


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--debug', dest='debug', action='store_true',
        help='Enable debug options.')

    parser.add_argument(
        "--n_pixels", type=int, default=3072,
        help=f'Number of pixels in toy image.')

    parser.add_argument(
        "--train_mean", type=float, default=1,
        help=f'Mean of p(x) at training time.')

    parser.add_argument(
        "--train_std", type=float, default=1,
        help=f'Std of p(x) at training time.')

    parser.add_argument(
        "--test_mean", type=float, default=5,
        help=f'Mean of p(x) at test time.')

    parser.add_argument(
        "--test_std", type=float, default=1,
        help=f'Std of p(x) at test time.')

    parser.add_argument(
        "--train_n", type=int, default=50000,
        help=f'Number of training points.')

    parser.add_argument(
        "--test_n", type=int, default=50000,
        help=f'Number of test points')

    parser.add_argument(
        "--train_distribution", type=str, default='unif-gaussian',
        help=f'Dist at test time. Options are unif-gaussian, unif-lognormal'
    )

    parser.add_argument(
        "--test_distribution", type=str, default='gaussian',
        help=f'Dist at test time. Options are gaussian, expunif-gmm, exp-lognormal'
    )

    return parser


if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format='%(levelname)s %(asctime)s %(message)s',)

    main(args.__dict__)
