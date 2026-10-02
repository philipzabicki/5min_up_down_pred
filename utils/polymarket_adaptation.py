"""Small past-only settlement correction over the unchanged BTC OOF log odds."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from utils.polymarket_policy import logits

CONTEXT_COLUMNS = ['past_volatility_30m', 'last_return_1m']
REGULARIZATION = (.1, 1., 10.)


def decision_context(oof):
    """Closed candles only; neither settlement nor future quotes are inputs."""
    ordered = oof.sort_values('Opened').copy()
    ordered['Opened'] = pd.to_datetime(ordered.Opened, utc=True)
    indexed = ordered.set_index('Opened').Close
    # Timestamp reindexing prevents a missing minute becoming a one-minute return.
    previous = indexed.reindex(indexed.index-pd.Timedelta(minutes=1)).to_numpy()
    returns = pd.Series(np.log(indexed.to_numpy()/previous), index=indexed.index)
    volatility = returns.rolling('30min', min_periods=30).std()
    return pd.DataFrame({'Opened': indexed.index, 'last_return_1m': returns.to_numpy(),
                         'past_volatility_30m': volatility.to_numpy(),
                         'context_available_at': indexed.index+pd.Timedelta(minutes=1)})


def available_labels(data, cutoff):
    available = data[['resolved_at_utc', 'market_end_utc']].max(axis=1)
    return data[(data.decision_available_at < cutoff) & (available < cutoff)]


class SettlementCorrection:
    """Penalized residual correction with mandatory original logit offset.

    Four coefficients: intercept, logit scale correction, logit × volatility,
    and last closed-minute return. Polymarket prices never enter this layer.
    """
    def __init__(self, strength):
        self.strength = strength

    def design(self, data):
        z = (data[CONTEXT_COLUMNS].to_numpy()-self.mean)/self.scale
        base = logits(data.p_model_up).ravel()
        return base, np.column_stack([np.ones(len(data)), base, base*z[:, 0], z[:, 1]])

    def fit(self, past, cutoff):
        if len(available_labels(past, cutoff)) != len(past):
            raise ValueError('Settlement correction includes unavailable labels')
        if (past.context_available_at > past.decision_available_at).any():
            raise ValueError('Context not available at decision')
        x = past[CONTEXT_COLUMNS].to_numpy()
        if not np.isfinite(x).all():
            raise ValueError('Missing historical context')
        self.mean, self.scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-12)
        base, design = self.design(past)
        y = past.target_polymarket_up.to_numpy()

        def objective(beta):
            score = base+design@beta
            penalty = 1/(len(past)*self.strength)
            loss = np.mean(np.logaddexp(0, score)-y*score)+penalty*np.dot(beta, beta)/2
            grad = design.T@(expit(score)-y)/len(past)+penalty*beta
            return loss, grad

        result = minimize(objective, np.zeros(4), jac=True, method='L-BFGS-B')
        if not result.success:
            raise ValueError('Settlement correction optimization failed: '+result.message)
        self.coefficients = result.x
        return self

    def predict(self, data):
        if (data.context_available_at > data.decision_available_at).any():
            raise ValueError('Context not available at decision')
        base, design = self.design(data)
        if not np.isfinite(design).all():
            raise ValueError('Missing historical context')
        return expit(base+design@self.coefficients)
