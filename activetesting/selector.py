import numpy as np

class _Selector:
    def __init__(self, cfg, models, experiment):
        self.cfg = cfg
        self.models = models
        self.model_names = list(self.models.keys())
        self.model_idxs = list(range(len(models.keys())))
        self.model_set = set(self.model_idxs)
        self.experiment = experiment
        self.logs = []

    def get_next(self, i):
        raise NotImplementedError

    def get_others(self, other_idxs):
        return [self.model_names[i] for i in other_idxs]

    def get_current_and_others(self, curr_idx):
        curr_model = self.model_names[curr_idx]
        other_idxs = sorted(list(self.model_set - {curr_idx}))
        others = self.get_others(other_idxs)

        return curr_model, others

class IteratingSelector(_Selector):
    """Iterate through competing models in order."""

    def __init__(self, cfg, models, experiment):
        super().__init__(cfg, models, experiment)
    
    def get_next(self, i):
        curr_idx = i % len(self.model_idxs)

        return self.get_current_and_others(curr_idx)


class ThompsonSamplingSelector(_Selector):
    """Sample from belief over which model is currently best."""

    def __init__(self, cfg, models, experiment):
        super().__init__(cfg, models, experiment)
    
    def get_next(self, i):

        if i == 1:
            # just choose first model in the beginning
            return self.get_current_and_others(i)

        if i == 2:
            # get true risks of models for logging purposes
            self.true_risks = [
                    self.experiment[
                        name].risk_estimators['TrueRiskEstimator'].risks[-1]
                    for name in self.model_names]

            self.true_order = np.argsort(self.true_risks)
            self.true_best = self.true_order[0]

        # get current risk estimates from all models
        risks = self.get_curr_risks()

        """Constructing a sampling distribution:
            We want to sample from posterior model probability.
            W/o prior, this is model evidence.
            Model evidence is approximately equal to CV with respect to log
            likelihood.
            Our risk values are mse/cross-entropy which is ~= NLL.
            Therefore exp(-risk) values ~= posterior model probability.
        """

        #TODO: this may not be the best way to do it
        #TODO: already these probs are looking quite close...
        #TODO: maybe there is some way to include variance over runs into this
        #TODO: alternatively could make more precise by including tempering

        probs = np.exp(-risks)
        probs /= probs.sum()

        # log distributions over risks, ranking, as well as which model is
        self.log_iter(i, risks, probs)

        # sample from distribution
        if self.cfg.sample:
            sample = np.random.multinomial(1, probs)
            curr_idx = np.where(sample)[0][0]
        else:
            curr_idx = np.argmax(probs)

        curr_model, others = self.get_current_and_others(curr_idx)

        #TODO: use the model that was currently sampled as surrogate acquisition

        return curr_model, others

    def get_curr_risks(self):
        risks = []
        for name in self.model_names:
            exp = self.experiment[name]
            curr_risk = exp.risk_estimators[self.cfg.risk].risks[-1]
            risks.append(curr_risk)
        return np.array(risks)

    def log_iter(self, step, risks, probs):

        best = np.argmin(risks)
        order = np.argsort(risks)

        for i, model in enumerate(self.model_names):
            pos = np.where(order == i)[0][0]
            true_pos = np.where(self.true_order == i)[0][0]

            self.logs.append([
                step, model, risks[i], probs[i],
                pos, # how is model i ranked currently?
                float(best == i), # is model i best model according to curr ranking?
                float(pos == true_pos), # is model i ranked correctly wrt true loss?
                float(self.true_best == i)]) # is model i the true best model?

        a = 1
