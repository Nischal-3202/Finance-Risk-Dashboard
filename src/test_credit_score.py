from credit_score import pd_to_credit_score, get_score_band

test_pd = 0.35

score = pd_to_credit_score(test_pd)
band = get_score_band(score)

print("PD:", test_pd)
print("Credit Score:", score)
print("Risk Band:", band)