# E2 — Confidence score vs expert judgement, M4 as expert proxy

Matched triples: **19** (expert records 29, unmatched 38).

> **EXPLORATORY.** With n < 30 all interval estimates are wide; report effect sizes with their CIs and never quote a bare correlation coefficient.

## (a) Does confidence track expert judgement?
- **confidence** (n=19): tau-b = -0.2435, permutation p = 0.235, AUC(Y vs P/N) = 0.3631. Y: 0.2767 [0.197, 0.3703] (n=7); P: 0.3334 [0.2335, 0.4291] (n=10); N: 0.4654 [0.4307, 0.5] (n=2)
- **support_papers** (n=19): tau-b = -0.26, permutation p = 0.222, AUC(Y vs P/N) = 0.3452. Y: 2.2857 [1.4286, 3.5714] (n=7); P: 3.7 [2.2, 5.2] (n=10); N: 4.5 [3.0, 6.0] (n=2)
- **support_chunks** (n=19): tau-b = 0.1014, permutation p = 0.6293, AUC(Y vs P/N) = 0.625. Y: 17.0 [8.0, 25.7143] (n=7); P: 12.0 [3.7, 24.8] (n=10); N: 34.0 [3.0, 65.0] (n=2)
- **tier_inverted** (n=19): tau-b = -0.1712, permutation p = 0.5121, AUC(Y vs P/N) = 0.4524. Y: -1.4286 [-1.8571, -1.1429] (n=7); P: -1.4 [-1.7, -1.1] (n=10); N: -1.0 [-1.0, -1.0] (n=2)

Interpretation rule agreed in advance: a positive tau with a monotone mean-confidence ordering Y > P > N supports the score as an external-validated reliability signal; a null or non-monotone result is reported as a limitation motivating human validation, not hidden.

## (b) Is the M4 panel a proxy for an expert?
- n = 19, exact agreement = 0.4211, kappa unweighted = -0.0097, kappa linear-weighted = -0.0693.
- Human inter-expert reference (Section 4.4): [0.3, 0.37].
- Reading: if machine-human kappa is comparable to human-human kappa, the panel is as consistent with an expert as experts are with each other — the agreement ceiling is a property of the task, not of the judges.