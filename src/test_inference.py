from inference_engine import CreditRiskEngine

engine = CreditRiskEngine()

sample_input = {
    "LIMIT_BAL": 90000,
    "AGE": 30,
    "PAY_0": 1,
    "PAY_2": 0,
    "PAY_3": 0,
    "BILL_AMT1": 25000,
    "PAY_AMT1": 5000
}

result = engine.evaluate_applicant(sample_input)

for key, value in result.items():
    print(f"{key}: {value}")