"""Map strings to classes."""
import importlib


class Item:
    def __init__(self, target):
        self.target = target

    # def __getitem__(self, item):
        # module = importlib.import_module(f'activetesting.{self.target}')
        # return module.__dict__[item]
    def __getitem__(self, item):
        module = __import__(f'activetesting.{self.target}', fromlist=[item])
        return getattr(module, item)


model = Item('models')
dataset = Item('datasets')
acquisition = Item('acquisition')
loss = Item('loss')
risk_estimator = Item('risk_estimators')
selector = Item('selector')
