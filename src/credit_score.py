import numpy as np


def pd_to_credit_score(pd_value):
    """
    Convert probability of default to credit score (300–850).
    """

    score = 850 - (pd_value * 550)
    score = np.clip(score, 300, 850)

    return int(round(score))


def get_score_band(score):
    """
    Return credit score risk band.
    """

    if score >= 750:
        return "Excellent"
    elif score >= 700:
        return "Good"
    elif score >= 650:
        return "Fair"
    elif score >= 600:
        return "Poor"
    else:
        return "Very Poor"