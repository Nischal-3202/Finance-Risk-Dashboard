

"""
Policy Engine Module

This module contains business logic rules:
- Risk categorization
- Loan decision policy
- Premium calculation policy

This layer is intentionally separated from ML models
to keep business rules configurable and modular.
"""

DEFAULT_APPROVAL_THRESHOLD = 0.35
MANUAL_REVIEW_THRESHOLD = 0.50

BASE_PREMIUM_FEE = 500


def categorize_risk(pd_value: float) -> str:
    """
    Categorize applicant risk level based on probability of default.
    """

    if pd_value < 0.20:
        return "Low Risk"
    elif pd_value < 0.40:
        return "Medium Risk"
    else:
        return "High Risk"


def make_loan_decision(pd_value: float) -> str:
    """
    Determine loan approval decision.
    """

    if pd_value < DEFAULT_APPROVAL_THRESHOLD:
        return "Approve"
    elif pd_value < MANUAL_REVIEW_THRESHOLD:
        return "Manual Review"
    else:
        return "Reject"


def calculate_premium(pd_value: float, predicted_loss: float) -> float:
    """
    Compute recommended premium using expected loss logic.
    """

    premium = BASE_PREMIUM_FEE + (pd_value * predicted_loss)

    return round(premium, 2)