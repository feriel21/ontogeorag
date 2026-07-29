# E2 — Confidence score vs expert judgement, M4 as expert proxy

Matched triples: **29** (expert records 29, unmatched 0).

> **EXPLORATORY.** With n < 30 all interval estimates are wide; report effect sizes with their CIs and never quote a bare correlation coefficient.

## (a) Does confidence track expert judgement?
- **confidence** (n=29): tau-b = -0.0313, permutation p = 0.8626, AUC(Y vs P/N) = 0.4789. Y: 0.5533 [0.4259, 0.6842] (n=19); P: 0.5382 [0.4061, 0.6771] (n=10)
- **support_papers** (n=29): tau-b = 0.2559, permutation p = 0.1364, AUC(Y vs P/N) = 0.6605. Y: 3.6316 [2.4211, 4.9474] (n=19); P: 1.8 [1.1, 2.8] (n=10)
- **support_chunks** (n=29): tau-b = 0.0539, permutation p = 0.7516, AUC(Y vs P/N) = 0.5368. Y: 9.5789 [5.2632, 14.2105] (n=19); P: 8.4 [1.8, 16.9] (n=10)
- **tier_inverted** (n=29): tau-b = -0.2855, permutation p = 0.2068, AUC(Y vs P/N) = 0.3658. Y: -1.3684 [-1.5789, -1.1579] (n=19); P: -1.1 [-1.3, -1.0] (n=10)

Interpretation rule agreed in advance: a positive tau with a monotone mean-confidence ordering Y > P > N supports the score as an external-validated reliability signal; a null or non-monotone result is reported as a limitation motivating human validation, not hidden.

## (a-bis) Independent-unit check
The pooled statistics above double-count each triple (one row per annotator, identical confidence). Independent-unit results:
- **P** (n=10): no channel with sufficient variance
- **Y** (n=19): no channel with sufficient variance
- **consensus** (18 triples, 17 with full annotator agreement): confidence__min tau=0.1096 (p=0.6325); confidence__mean tau=0.0382 (p=0.8778); support_papers__min tau=0.3572 (p=0.1102); support_papers__mean tau=0.3132 (p=0.1456); support_chunks__min tau=0.2675 (p=0.2282); support_chunks__mean tau=0.2015 (p=0.3509); tier_inverted__min tau=-0.1 (p=1.0); tier_inverted__mean tau=-0.2078 (p=0.5451)

**Range restriction.** Confidence in this sample spans [0.1809, 1.0] (sd=0.2612), tiers present: [1, 2]. The Section 4.4 sample was drawn from Tier-1 only, so w_tier is constant and the confidence range is truncated. Any attenuation of the correlation must be read in that light: this tests the score WITHIN the top stratum, not across the full reliability range.

## (b) Is the M4 panel a proxy for an expert?
- n = 29, exact agreement = 0.3793, kappa unweighted = -0.2548, kappa linear-weighted = -0.2548.
- Human inter-expert reference (Section 4.4): [0.3, 0.37].
- Reading: if machine-human kappa is comparable to human-human kappa, the panel is as consistent with an expert as experts are with each other — the agreement ceiling is a property of the task, not of the judges.