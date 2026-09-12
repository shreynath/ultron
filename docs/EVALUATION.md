# Evaluation

FoundationEvaluator scores a complete local checkpoint against a versioned, hash-pinned held-out token-loss suite. It writes immutable reports and compares baseline/candidate loss. PromotionPolicy can reject regressions, but a passing loss check still returns INSUFFICIENT_EVIDENCE because broader capability, safety, and regression suites are required.

For the Phase E development run, checkpoint 26 regressed held-out loss versus checkpoint 25 by 0.00684 and was rejected.
