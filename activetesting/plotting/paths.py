from pathlib import Path

def plus_base(base, end):
    if isinstance(end, list):
        return [plus_base(base, i) for i in end]
    else:
        return Path(base) / end


def iter_dir(base):
    try:
        return [str(i) for i in  list(Path(base).iterdir())]
    except Exception as e:
        return []


class ReproduciblePaths:
    """Store paths for experiment results."""

    synthetic_gaussian_names = [5, 15, 25]
    synthetic_gaussian = [
        iter_dir('outputs/SyntheticGaussian_dim_5'),
        iter_dir('outputs/SyntheticGaussian_dim_15'),
        iter_dir('outputs/SyntheticGaussian_dim_25'),]
    synthetic_gaussian_bq = [
        iter_dir('outputs/SyntheticGaussianBQOnly_dim_5'),
        iter_dir('outputs/SyntheticGaussianBQOnly_dim_15'),
        iter_dir('outputs/SyntheticGaussianBQOnly_dim_25'),]

    radialbnn_missing_7 = iter_dir('outputs/RadialBNNMNISTMissing7')

    resnets_names = ['Fashion-MNIST', 'CIFAR-10', 'CIFAR-100']
    resnets = [
        iter_dir('outputs/ResNetFMNIST/'),
        iter_dir('outputs/ResNetCifar10/'),
        iter_dir('outputs/ResNetCifar100/'),]

    synthetic_linear_names = [1, 10, 100, 1000]
    synthetic_linear = [
        iter_dir('outputs/SyntheticLinear_dim_1'),
        iter_dir('outputs/SyntheticLinear_dim_10'),
        iter_dir('outputs/SyntheticLinear_dim_100'),
        iter_dir('outputs/SyntheticLinear_dim_1000'),]
