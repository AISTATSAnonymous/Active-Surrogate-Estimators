"""Map strings to classes."""
from activetesting.models import (
    LinearRegressionModel, GaussianProcessRegressor, RandomForestClassifier,
    SVMClassifier, GPClassifier, RadialBNN, TinyRadialBNN, ResNet18,
    WideResNet, ResNet18Ensemble, ResNet34Ensemble, ResNet50Ensemble,
    ResNet101Ensemble, WideResNetEnsemble, SimpleFF,
    FixedLinearModel)
from activetesting.datasets import (
    QuadraticDatasetForLinReg, SinusoidalDatasetForLinReg,
    GPDatasetForGPReg, MNISTDataset, TwoMoonsDataset, FashionMNISTDataset,
    Cifar10Dataset, Cifar100Dataset, ToyDataset)
from activetesting.acquisition import (
    RandomAcquisition, TrueLossAcquisition, DistanceBasedAcquisition,
    GPAcquisitionUncertainty,
    GPSurrogateAcquisitionLogLik, GPSurrogateAcquisitionMSE,
    ClassifierAcquisitionEntropy,
    RandomForestClassifierSurrogateAcquisitionEntropy,
    SVMClassifierSurrogateAcquisitionEntropy,
    GPClassifierSurrogateAcquisitionEntropy,
    RandomRandomForestClassifierSurrogateAcquisitionEntropy,
    GPSurrogateAcquisitionMSEDoublyUncertain,
    SelfSurrogateAcquisitionEntropy,
    BNNClassifierAcquisitionMI,
    AnySurrogateAcquisitionEntropy,
    ClassifierAcquisitionAccuracy,
    SelfSurrogateAcquisitionAccuracy,
    AnySurrogateAcquisitionAccuracy,
    SawadeAcquisition,
    SawadeOptimalAcquisition,
    GPSurrogateAcquisitionMSENoDis,
    AnySurrogateRandomAcquisition,
    SelfSurrogateBNNClassifierAcquisitionRiskCovariance,
    BNNClassifierAcquisitionRiskCovariance,
    AnySurrogateSurrogateUncertaintyAcquisition,
    ClassifierAcquisitionValue,
    BayesQuadAcquisition
    )
from activetesting.loss import (
    SELoss, MSELoss, RMSELoss, CrossEntropyLoss, AccuracyLoss, YIsLoss)

from activetesting.risk_estimators import (
    BiasedRiskEstimator, NaiveUnbiasedRiskEstimator,
    FancyUnbiasedRiskEstimator, TrueRiskEstimator,
    ImportanceWeightedRiskEstimator, TrueUnseenRiskEstimator,
    QuadratureRiskEstimator, QuadratureRiskEstimatorWithUncertainty,
    ConvexComboWithUncertainty, ConvexComboWithOutUncertainty,
    QuadratureRiskEstimatorRemoveNoise,
    QuadratureRiskEstimatorNoDisagreement,
    QuadratureRiskEstimatorNoAleatoric,
    QuadratureRiskEstimatorNoEpistemic,
    ExactExpectedRiskEstimator,
    BayesQuadRiskEstimator,
    BayesQuadPointWiseRiskEstimator
    )

from activetesting.selector import (
    IteratingSelector,
    ThompsonSamplingSelector)

model = dict(
    LinearRegressionModel=LinearRegressionModel,
    GaussianProcessRegressor=GaussianProcessRegressor,
    RandomForestClassifier=RandomForestClassifier,
    SVMClassifier=SVMClassifier,
    GPClassifier=GPClassifier,
    RadialBNN=RadialBNN,
    TinyRadialBNN=TinyRadialBNN,
    ResNet18=ResNet18,
    WideResNet=WideResNet,
    ResNet18Ensemble=ResNet18Ensemble,
    ResNet34Ensemble=ResNet34Ensemble,
    ResNet50Ensemble=ResNet50Ensemble,
    ResNet101Ensemble=ResNet101Ensemble,
    WideResNetEnsemble=WideResNetEnsemble,
    SimpleFF=SimpleFF,
    FixedLinearModel=FixedLinearModel
)

dataset = dict(
    QuadraticDatasetForLinReg=QuadraticDatasetForLinReg,
    SinusoidalDatasetForLinReg=SinusoidalDatasetForLinReg,
    GPDatasetForGPReg=GPDatasetForGPReg,
    MNISTDataset=MNISTDataset,
    TwoMoonsDataset=TwoMoonsDataset,
    FashionMNISTDataset=FashionMNISTDataset,
    Cifar10Dataset=Cifar10Dataset,
    Cifar100Dataset=Cifar100Dataset,
    ToyDataset=ToyDataset,
)

acquisition = dict(
    RandomAcquisition=RandomAcquisition,
    TrueLossAcquisition=TrueLossAcquisition,
    DistanceBasedAcquisition=DistanceBasedAcquisition,
    GPAcquisitionUncertainty=GPAcquisitionUncertainty,
    GPSurrogateAcquisitionLogLik=GPSurrogateAcquisitionLogLik,
    GPSurrogateAcquisitionMSE=GPSurrogateAcquisitionMSE,
    ClassifierAcquisitionEntropy=ClassifierAcquisitionEntropy,
    RandomForestClassifierSurrogateAcquisitionEntropy=(
        RandomForestClassifierSurrogateAcquisitionEntropy),
    SVMClassifierSurrogateAcquisitionEntropy=(
        SVMClassifierSurrogateAcquisitionEntropy),
    GPClassifierSurrogateAcquisitionEntropy=(
        GPClassifierSurrogateAcquisitionEntropy),
    RandomRandomForestClassifierSurrogateAcquisitionEntropy=(
        RandomRandomForestClassifierSurrogateAcquisitionEntropy),
    GPSurrogateAcquisitionMSEDoublyUncertain=(
        GPSurrogateAcquisitionMSEDoublyUncertain),
    SelfSurrogateAcquisitionEntropy=SelfSurrogateAcquisitionEntropy,
    BNNClassifierAcquisitionMI=BNNClassifierAcquisitionMI,
    AnySurrogateAcquisitionEntropy=AnySurrogateAcquisitionEntropy,
    ClassifierAcquisitionAccuracy=ClassifierAcquisitionAccuracy,
    SelfSurrogateAcquisitionAccuracy=SelfSurrogateAcquisitionAccuracy,
    AnySurrogateAcquisitionAccuracy=AnySurrogateAcquisitionAccuracy,
    SawadeAcquisition=SawadeAcquisition,
    SawadeOptimalAcquisition=SawadeOptimalAcquisition,
    GPSurrogateAcquisitionMSENoDis=GPSurrogateAcquisitionMSENoDis,
    AnySurrogateRandomAcquisition=AnySurrogateRandomAcquisition,
    SelfSurrogateBNNClassifierAcquisitionRiskCovariance=(
        SelfSurrogateBNNClassifierAcquisitionRiskCovariance),
    BNNClassifierAcquisitionRiskCovariance=(
        BNNClassifierAcquisitionRiskCovariance),
    AnySurrogateSurrogateUncertaintyAcquisition=(
        AnySurrogateSurrogateUncertaintyAcquisition),
    ClassifierAcquisitionValue=ClassifierAcquisitionValue,
    BayesQuadAcquisition=BayesQuadAcquisition,  
)

loss = dict(
    SELoss=SELoss,
    MSELoss=MSELoss,
    RMSELoss=RMSELoss,
    CrossEntropyLoss=CrossEntropyLoss,
    AccuracyLoss=AccuracyLoss,
    YIsLoss=YIsLoss,
)

risk_estimator = dict(
    TrueRiskEstimator=TrueRiskEstimator,
    BiasedRiskEstimator=BiasedRiskEstimator,
    NaiveUnbiasedRiskEstimator=NaiveUnbiasedRiskEstimator,
    FancyUnbiasedRiskEstimator=FancyUnbiasedRiskEstimator,
    ImportanceWeightedRiskEstimator=ImportanceWeightedRiskEstimator,
    TrueUnseenRiskEstimator=TrueUnseenRiskEstimator,
    QuadratureRiskEstimator=QuadratureRiskEstimator,
    QuadratureRiskEstimatorWithUncertainty=(
        QuadratureRiskEstimatorWithUncertainty),
    ConvexComboWithUncertainty=ConvexComboWithUncertainty,
    ConvexComboWithOutUncertainty=ConvexComboWithOutUncertainty,
    QuadratureRiskEstimatorRemoveNoise=QuadratureRiskEstimatorRemoveNoise,
    QuadratureRiskEstimatorNoDisagreement=(
        QuadratureRiskEstimatorNoDisagreement),
    QuadratureRiskEstimatorNoAleatoric=QuadratureRiskEstimatorNoAleatoric,
    QuadratureRiskEstimatorNoEpistemic=QuadratureRiskEstimatorNoEpistemic,
    ExactExpectedRiskEstimator=ExactExpectedRiskEstimator,
    BayesQuadRiskEstimator=BayesQuadRiskEstimator,
    BayesQuadPointWiseRiskEstimator=BayesQuadPointWiseRiskEstimator,
)


selector = dict(
    IteratingSelector=IteratingSelector,
    ThompsonSamplingSelector=ThompsonSamplingSelector
)