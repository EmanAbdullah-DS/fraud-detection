import numpy as np

from config import REVIEW_COST, MISSED_FRAUD_RATE


def total_cost(y, amount, flagged, review_cost=REVIEW_COST, missed_rate=MISSED_FRAUD_RATE):
    """Review cost for every flagged transaction + lost amount for every missed fraud."""
    y, amount, flagged = np.asarray(y), np.asarray(amount), np.asarray(flagged).astype(bool)
    review = flagged.sum() * review_cost
    missed = (amount[(y == 1) & ~flagged]).sum() * missed_rate
    return float(review + missed)


def best_threshold(y, amount, proba, review_cost=REVIEW_COST, missed_rate=MISSED_FRAUD_RATE):
    grid = np.unique(np.round(np.quantile(proba, np.linspace(0.5, 0.9995, 400)), 6))
    costs = [total_cost(y, amount, proba >= t, review_cost, missed_rate) for t in grid]
    i = int(np.argmin(costs))
    return float(grid[i]), float(costs[i])
