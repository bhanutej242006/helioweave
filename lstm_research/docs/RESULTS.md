# Measured results

Environment: Python 3.14.7, NumPy 2.5.3, SciPy 1.18.1.

## Solar forecasting

621 held-out forecast windows; 16 horizons per window. Windows overlap.

LSTM normalized RMSE: 12.898% of training power scale.
Previous-day normalized RMSE: 12.886%.

The LSTM did not outperform the previous-day baseline on this test block. No accuracy superiority is claimed. The test block must not be reused for tuning.

Mean empirical interval coverage over horizons: 86.2%. Calibration target was 90%; this is not a guaranteed interval.

## Paired scheduling replay

Six held-out days, identical ten-EV requests (seed 42), 60 kW scenario PV scale, 25 kW grid cap. Building demand and tariffs are simulated.

| Controller | Mean daily cost (INR) | Mean unmet (kWh) | Mean daily peak (kW) | Grid violations |
|---|---:|---:|---:|---:|
| immediate | 1189.83 | 0.0000 | 25.00 | 0 |
| learned | 894.06 | 0.0000 | 17.01 | 0 |
| rolling | 841.04 | 0.0000 | 21.05 | 0 |
| oracle | 796.65 | 0.0000 | 19.08 | 0 |

The learned controller reduced mean scenario cost by 24.86% relative to immediate charging, with zero unmet demand in this default scenario. Rolling optimization performed better than the learned policy. These are simulated outcomes, not field savings.

The oracle is solved optimally by the LP solver under the simplified scenario assumptions. Cost gaps are meaningful only alongside matched unmet energy.

## Verification

- Six core unit tests passed on Python 3.14.7; see reports/tests.txt.
- LSTM analytical gradients checked against numerical finite differences.
- Future-data perturbation leaves the causal forecast unchanged.
- Known LP case and impossible request checked.
- Projection tested across randomized proposals and all 96 slots.
- Low-solar/high-demand stress scenario reports unmet demand and no grid violations; see compatibility_and_stress.json.
- Complete controlled-error study: six days × five seeds × four nominal error levels.

## Limitations

Approximately one month of data, no actual EV sessions, assumed 100% charging efficiency, source-local timestamps, incomplete intervals forward-filled and flagged, short forecast horizon, and a simplified per-EV learned policy. The solar model has no location-based nighttime mask; future work must add geometry using a verified site location. The learned policy uses previous-day forecast inputs while learning and LSTM forecast inputs in replay. This distribution shift is explicitly retained in evaluation.

- Browser verification: desktop and 390 px mobile layouts rendered; zero JavaScript errors; no mobile horizontal overflow; controller switching and CSV export passed.
