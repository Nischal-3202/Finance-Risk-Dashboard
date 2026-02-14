

import numpy as np
from credit_score import pd_to_credit_score
from policy_engine import (
    categorize_risk,
    make_loan_decision,
    calculate_premium,
)


MODEL_PATH = "models/credit_risk_model.npz"


class CreditRiskEngine:

    def __init__(self, model_path=MODEL_PATH):
        data = np.load(model_path, allow_pickle=True)

        self.w_log = data["w_logistic"]
        self.b_log = data["b_logistic"]
        self.mean_log = data["mean_logistic"]
        self.std_log = data["std_logistic"]
        self.logistic_features = data["logistic_features"]
        self.threshold = float(data["threshold"])

        self.w_lin = data["w_linear"]
        self.b_lin = data["b_linear"]
        self.mean_lin = data["mean_linear"]
        self.std_lin = data["std_linear"]
        self.linear_features = data["linear_features"]

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _scale(self, x, mean, std):
        std_safe = np.where(std == 0, 1, std)
        return (x - mean) / std_safe

    def predict_default_probability(self, user_input: dict):
        """
        user_input: dictionary containing required logistic features
        """

        x = np.array([user_input[feat] for feat in self.logistic_features], dtype=float)
        x_scaled = self._scale(x, self.mean_log, self.std_log)

        z = np.dot(x_scaled, self.w_log) + self.b_log
        pd = self._sigmoid(z)

        return float(pd)

    def predict_outstanding_loss(self, user_input: dict):
        """
        Predict expected outstanding loss using linear regression model
        """

        x = np.array([user_input[feat] for feat in self.linear_features], dtype=float)
        x_scaled = self._scale(x, self.mean_lin, self.std_lin)

        loss = np.dot(x_scaled, self.w_lin) + self.b_lin

        return max(0.0, float(loss))

    def evaluate_applicant(self, user_input: dict):
        """
        Full evaluation pipeline:
        - Probability of Default
        - Credit Score
        - Risk Band
        - Predicted Loss
        - Insurance Premium
        """

        pd = self.predict_default_probability(user_input)
        score = pd_to_credit_score(pd)
        risk_category = categorize_risk(pd)
        loss = self.predict_outstanding_loss(user_input)

        premium = calculate_premium(pd, loss)
        decision = make_loan_decision(pd)

        return {
            "probability_of_default": round(pd, 4),
            "credit_score": score,
            "risk_category": risk_category,
            "predicted_loss": round(loss, 2),
            "recommended_premium": premium,
            "decision": decision
        }