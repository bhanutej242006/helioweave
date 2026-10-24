# Understand and present HelioWeave 3D

## What the project solves
Solar power changes with weather and the sun's position. EVs need energy before their drivers leave. The application combines solar estimation, time-series forecasting and charging optimization so you can study those decisions together.

It now has two complementary research tracks:
1. **Your requested 20-feature track:** LightGBM, XGBoost, CatBoost, KNN and a stacking ensemble estimate generated power from a weather observation.
2. **Your original paper's LSTM track:** a time-series LSTM forecasts future solar power, and another LSTM imitates optimizer-generated charging actions.

These tracks use different datasets for a reason: the 20-feature CSV has no timestamps. It cannot honestly establish a future time horizon by itself. The LSTM module uses timestamped data and is evaluated separately.

## How the application is organized
- **Frontend:** HTML gives the page structure; CSS supplies the layout and appearance; JavaScript handles interaction, charts and API calls.
- **Flask backend:** validates submitted data, loads models, performs inference and solves charging plans.
- **Three.js:** renders the solar campus using real 3D meshes, a camera, lighting, shadows and animation. It is bundled locally.
- **NumPy, pandas and scikit-learn:** numeric arrays, CSV processing, preprocessing, metrics, KNN and the stacking meta-model.
- **LightGBM, XGBoost, CatBoost:** actual installed machine-learning libraries; their names are not just display labels.
- **SciPy / HiGHS:** solves the charging linear program.

No OpenAI API key is used. The campus is an illustrative model, not a measured digital twin of a specific site.

## The five requested algorithms
| Model | Plain explanation | Implementation |
|---|---|---|
| LightGBM | Learns successive decision trees that correct earlier errors; grows trees leaf-wise | LGBMRegressor, 240 trees |
| XGBoost | Boosted decision trees with regularization and sampled rows/features | XGBRegressor, 240 trees |
| CatBoost | Gradient boosting with symmetric trees in this configuration | CatBoostRegressor, 240 iterations |
| KNN | Predicts using nearby examples in a standardized feature space | Nine neighbors, distance-weighted |
| Stacking Model | Learns how to combine the four base models' predictions | Four expanding out-of-fold blocks, then Ridge regression |

The feature set is entirely numeric; CatBoost's support for categorical data is not required here. Each model receives all 20 features. `generated_power_kw` is the training target, never an input.

## Why the stack uses out-of-fold predictions
If the meta-model learned from base-model predictions on their own training rows, its inputs could look unrealistically accurate. Instead, the training block is split into four expanding folds. Each base learner predicts a held-out portion after fitting only the preceding portion. The Ridge meta-model learns from those held-out predictions. The base models are then refit on the full training block.

Preprocessing is inside each model pipeline. Training medians fill missing inputs; KNN scaling is fitted only on the relevant training fold. The validation block selects the default model. The final test block is reserved for reporting. Because timestamps are absent, source-row ordering is not described as a verified chronological evaluation.

## How EV dispatch works
You enter battery capacity, starting SOC, target SOC, arrival hour, departure hour and charger power. For example, a 50 kWh battery going from 30% to 80% needs:

`50 × (80 − 30) / 100 = 25 kWh delivered to the battery`

At 92% efficiency it requires approximately `25 / 0.92 = 27.17 kWh` from the charger.

The optimizer first minimizes total unmet battery energy. It then minimizes estimated grid-import cost while preserving that minimum shortfall. It enforces:
- no charging outside the connection window;
- charger power limits;
- requested battery energy limits;
- grid-import limits;
- the declared charging efficiency.

The planner assumes hourly intervals, same-day sessions, constant building demand, no export, no discharge and illustrative time-of-use prices. Its solar input is generated from an illustrative weather profile and the selected model, scaled to the chosen site size. It is not tomorrow's verified weather forecast.

An immediate-charging baseline uses the same scenario. Always read cost together with unmet energy: a controller that leaves cars undercharged may appear artificially cheap. If a request is impossible, the app reports unmet energy. If the building itself cannot be served within the grid limit during low-solar hours, the app rejects the scenario with an explanation.

## What the 3D animation means
- Solar arrays show an architectural panel layout with individual cells and supports.
- The power-conversion enclosure represents inverter/transformer equipment.
- Charging bays show vehicles and charging infrastructure.
- Energy particles indicate solar, EV and grid-flow activity.
- The EV schedule slider changes the displayed hour and the corresponding flow intensities.
- The sun marker responds to geometry inputs, but the campus orientation is illustrative rather than geolocated.
- Orbit, zoom and camera presets let you inspect the system. Pause motion freezes animation while leaving forms usable.

The model uses four display vehicles as an illustration; the numerical planner can handle up to twelve EV requests. The tables and CSVs are the authoritative numerical outputs.

## Where each dataset lives
- `data/spg.csv`: exact 20-column weather/geometry schema plus power target, 4,213 rows.
- `lstm_research/data/raw/`: original Plant 1 and Plant 2 generation/weather CSVs.
- `lstm_research/data/processed_solar.csv`: cleaned timestamped data for the LSTM module.
- `data/presets.json`: held-out observations for the prediction form.

The full input names and source attribution are in `data/DATASET_CARD.md` and in the Dataset Explorer screen.

## A seven-minute demonstration
1. Start Flask and show the campus. Orbit, select the Solar array camera and pause/resume motion.
2. Open Prediction studio. Load High generation, run all models and explain why they disagree.
3. Open the Wind field and Solar geometry tabs to show all 20 parameters.
4. Upload a CSV or show the original dataset download.
5. Open EV dispatch. Explain one vehicle's battery/SOC requirement and run the optimizer.
6. Move the hour slider and relate energy-flow animation to the numerical chart.
7. Increase an EV's requirement and shorten its connection window to show an honest shortfall.
8. Open Model observatory and discuss the actual test metrics and stacking method.
9. Open LSTM research and run a historical date; explain why this has a future horizon while the new CSV track does not.
10. Export a schedule and show the reports folder.

## Viva answers
**What is your contribution?** Integrating multi-model solar estimation, the original LSTM research workflow, a service-first EV planner, transparent evaluation and a project-specific 3D explanation interface. This is an implementation contribution, not a proven patent novelty claim.

**Why do you still use LSTM?** The original title includes temporal forecasting and learned scheduling. Those remain implemented in the timestamped module; the five new regressors add a separate weather-to-power comparison.

**Is stacking always best?** No. The bundled run selects stacking on validation RMSE, while CatBoost has a slightly lower final test RMSE. This is why validation selection and final test reporting are separate.

**What does R² mean?** It compares squared prediction error to predicting the evaluation set's mean. It is not a classification accuracy percentage and must not be presented as one.

**Is the application real time?** The interface computes results on demand and the LSTM module runs causal historical replay. There is no live sensor or charger connection in this delivery.

**Are the EV sessions measured?** The new planner uses your entered scenario requests. The original replay uses explicitly simulated sessions. Neither is misrepresented as measured EV data.

**Why Flask?** It provides a lightweight Python backend that directly calls the models and optimizer while serving an ordinary HTML/CSS/JS frontend.

**What is the main limitation?** The requested dataset lacks timestamps and complete collection metadata; EV conditions are scenarios; and there is no field validation. Additional time-aligned data and hardware testing are needed for deployment claims.

## Extending the project responsibly
Collect timestamped weather forecasts with issue times and actual PV measurements, add real EV sessions, evaluate over seasons, add deployment-grade authentication and monitoring, and test communication delays and charger constraints. Treat these as future work, not completed features.
