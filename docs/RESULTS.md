# Measured results and verification

Training environment: CPython 3.14.7, Linux x86_64. Exact dependency versions are pinned in requirements-tested.txt.

## Five-model weather-to-power study

Source: 4213 rows, 4213 retained. 2527 train, 843 validation, 843 test.

Contiguous source-row blocks, 60/20/20. Source CSV has no timestamp; chronological forecasting validation cannot be claimed.

| Model | Test MAE (kW) | Test RMSE (kW) | Test R² |
|---|---:|---:|---:|
| LightGBM | 510.055 | 675.338 | 0.4472 |
| XGBoost | 502.638 | 658.357 | 0.4747 |
| CatBoost | 500.331 | 638.972 | 0.5052 |
| KNN | 530.091 | 686.876 | 0.4282 |
| Stacking Model | 492.121 | 639.465 | 0.5044 |

Default chosen on validation RMSE: **Stacking Model**. CatBoost has a slightly better final test RMSE in this run; the test block was not used to change the selected model.

R² is not an accuracy percentage. These scores do not establish a verified future forecasting horizon because source timestamps are absent.

## Verification

13 core tests passed on Python 3.14.7, covering model presence, exact schema, finite inference, target exclusion, invalid input, stacking split boundaries, batch prediction, charger/grid/energy constraints, infeasible requests, retained LSTM artifacts and local assets. See reports/tests.txt.

Browser checks exercised all six screens, preset prediction, schedule optimization, timeline interaction, dataset pagination and LSTM replay. The browser created a WebGL canvas, reported no JavaScript errors and showed no horizontal document overflow at 390 px width. Screenshots and browser_tests.json are included.

## LSTM study

The original solar LSTM did not beat previous-day persistence in its held-out run. The complete original metrics and controller comparisons are in lstm_research/reports; its explanatory report is in lstm_research/docs/RESULTS.md. Results are not pooled with the new 20-feature study.

## Limits

All new EV dispatch outputs are estimated outcomes under an illustrative hourly weather profile, user EV requests, assumed building load, charging efficiency and tariffs. The scene is illustrative and does not establish a physical digital twin. No live-data connection, field savings, patent novelty or universal model superiority is claimed.
