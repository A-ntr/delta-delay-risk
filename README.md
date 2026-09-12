# Flight Delay Risk Tiering — Night-Before Ops Planning

## Problem
Give airline ops planners a **risk tier (low/medium/high)** for every flight on
tomorrow's schedule, computed the night before, so they can pre-position ground
crew, deicing teams, and spare aircraft ahead of likely disruptions.

## Key design decisions (and why)

1. **Target = probability, not tier.** The model outputs `P(delay > 15 min)`.
   Tiers are produced by thresholding that probability *after* the fact. This
   decouples the model (an honest risk estimate) from policy (how aggressive
   ops wants to be), so thresholds can be tuned per season/capacity without
   retraining.

2. **Night-before constraint applies to every feature AND every row.**
   Nothing that's only known at/after departure time can be used (e.g. actual
   departure time, delay cause codes, same-day weather at departure).
   Historical averages must only use data *before* each row's flight date —
   otherwise the model leaks future information into training and looks great
   in validation but fails in production.

3. **Historical delay-rate features use a specificity fallback hierarchy:**
   `aircraft tail → route → carrier → time-of-day → global average`
   When a more specific grouping has too little history (new aircraft, new
   route), fall back to a broader grouping instead of dropping the flight —
   ops needs a risk tier for every flight on tomorrow's list, not just the
   ones with rich history.

4. **Train/test split is chronological, not random**, and spans full seasonal
   cycles in both sets (e.g. train on year 1, test on year 2) to avoid leaking
   future information through shuffling and to avoid a mid-season cutoff bias.
   Caveat to state up front in any pitch: a chronological split can still be
   confounded by structural shifts between years (fuel prices, fleet changes,
   major disruptions) — performance should be monitored and the model
   periodically retrained.

5. **Model = gradient-boosted trees (XGBoost/LightGBM), not a neural net.**
   Tabular data, moderate volume, high-cardinality categoricals (route, tail
   number) that trees split on natively without an encoding step, and SHAP
   values give per-flight, plain-language explanations ("flagged mainly due
   to low historical performance on this route + aircraft age") — which
   matters because an ops planner won't act on a black box.

6. **Metric = F-beta (beta > 1), not accuracy.** Only ~8% of flights are
   delayed, so a model that predicts "never delayed" scores ~92% accuracy
   while catching zero real delays. Missing a real delay (false negative) is
   more operationally costly than a false alarm (false positive), so recall
   is weighted more heavily than precision.

## Features (all night-before-knowable)
- Carrier, origin/destination, scheduled departure time, day of week, distance
- Weather forecast for the flight's departure day (not same-day actuals)
- Historical delay rate: tail / route / carrier / time-of-day, with fallback
- Aircraft age (flight date − manufacture date, via FAA Registry lookup on tail number)

## Data
Bureau of Transportation Statistics, Airline On-Time Performance Data
(https://www.transtats.bts.gov/) — download instructions in `data/raw/README.md`.
FAA Aircraft Registry (https://registry.faa.gov/aircraftinquiry/) for aircraft age.

## Status
- [ ] Pull real BTS data (manual download — see data/raw/README.md)
- [ ] Implement `src/features.py::historical_rate_with_fallback` (in progress — see TODOs)
- [ ] Join FAA registry for aircraft age
- [ ] Train baseline XGBoost model
- [ ] SHAP explainability pass
- [ ] Precision-recall curve → pick F2-optimal threshold → define tiers
