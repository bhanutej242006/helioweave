# Dataset card

## Original measured data
- Title: Solar Power Generation Data.
- Publisher: Ani Kannal, Kaggle.
- Source: https://www.kaggle.com/datasets/anikannal/solar-power-generation-data
- Download endpoint: https://www.kaggle.com/api/v1/datasets/download/anikannal/solar-power-generation-data
- Retrieved for this project: 29 September 2026.
- Original source: two solar plants in India, approximately 34 days at 15-minute intervals.
- Files: Plant_1_Generation_Data.csv, Plant_1_Weather_Sensor_Data.csv, Plant_2_Generation_Data.csv, Plant_2_Weather_Sensor_Data.csv.
- Kaggle metadata returned `licenseName: Data files © Original Authors`; this is not a blanket open-source license. Attribution and original-author rights remain applicable. No ownership or relicensing of these data is claimed. Check source terms before publishing or commercial redistribution.
- A snapshot of source metadata is stored in `kaggle_metadata.json`.
- Actual original-file SHA-256 hashes are in `reports/data_quality.json`.

Generation contains DATE_TIME, PLANT_ID, SOURCE_KEY (inverter), DC_POWER, AC_POWER, DAILY_YIELD and TOTAL_YIELD. Weather contains DATE_TIME, PLANT_ID, SOURCE_KEY (sensor), AMBIENT_TEMPERATURE, MODULE_TEMPERATURE and IRRADIATION.

## Data actually used
Default training uses Plant 1 AC power, irradiation, ambient temperature, time of day and a valid-observation flag. DC power and cumulative yield are not targets. Averages are taken across observed inverters, avoiding misleading summation of incomplete inverter sets. At least 90% of the plant's distinct inverters and a matching weather row are required for a valid target. Duplicate inverter/timestamp records use the last valid row.

A regular 15-minute grid is formed. Missing or incomplete intervals are marked invalid and forward-filled from past observations for history only. A training/evaluation forecast window is excluded if any of its 16 future target intervals is invalid. No future interpolation is used. The complete cleaned CSV is `processed_solar.csv`.

Power is normalized using the maximum mean inverter AC power in the training block. This is an empirical training scale, **not independently verified nameplate capacity**. The dashboard multiplies the normalized series by a configurable 60 kW default to form a simulated site-scale PV profile. Values can reach 1.2 times that scale. This is a measured-shape, scaled scenario rather than an assertion that the original plant had 60 kW installed capacity.

Source timestamps contain no timezone offset. They remain source-local, with Asia/Kolkata assumed for interpretation. The project does not invent a precise plant location or compute unsupported solar geometry. It does not retrieve future observed weather as a forecast.

## Chronological split
First 60%: training. Next 10%: checkpoint selection. Next 10%: uncertainty calibration. Last 20%: final forecasting test. Forecast target windows stay entirely within each block; past history may cross a boundary because it was already observed. Exact dates, rows, duplicates, coverage and normalization are in `reports/data_quality.json`. Test windows overlap; they are not independent trials.

Scheduler teacher trajectories use only training-period days with at least 90% valid intervals. The final portion of teacher days is reserved for scheduler validation. The policy input uses previous-day forecasts, not future measured PV; offline teacher labels may use the full realized day. Missing replay intervals remain forward-filled and are identified in the dashboard coverage. Test days below 90% observation coverage are excluded; remaining imputations are a limitation.

## Simulated, not Kaggle measurements
Ten EV requests are seeded synthetic scenarios. Arrival: slots 26–42; departure: slots 60–84; energy requests: 9–23 kWh before stress scaling; charger limit: 7.2 kW. Building demand varies approximately 7–13.2 kW. Tariffs are illustrative INR 4/6/10 per kWh, not a named utility tariff. Charging efficiency is assumed 100%, and no SOC, battery capacity, degradation, export revenue or vehicle-to-grid behavior is modeled.

These assumptions are implemented in `helioweave/scheduler.py:scenario`. Simulated EV data are not described as measured ACN-Data. See each saved replay JSON's `vehicles` and `settings` for the exact requests and parameters used.
