# Data sources and treatment

## Requested 20-feature data
- Name: Solar energy power generation dataset.
- Publisher: `stucom`, Kaggle.
- URL: https://www.kaggle.com/datasets/stucom/solar-energy-power-generation-dataset
- Original file: `spg.csv`, downloaded directly from Kaggle for this project on 29 September 2026.
- Source rows: 4,213; original schema has 20 numeric inputs plus `generated_power_kw`.
- Kaggle source metadata declares **Community Data License Agreement – Sharing – Version 1.0**. License: https://cdla.dev/sharing-1-0/ . Data are provided under that source license; original rights and attribution are retained. The dataset is not relicensed as project code. `kaggle_metadata.json` preserves the source metadata. An offline copy of the license page is included as `CDLA_Sharing_1.0.html`.
- SHA-256, versions, split boundaries, missing counts and measured metrics are in `reports/metrics.json`.

This CSV has no timestamps. We preserve source-row order, remove invalid targets/negative targets and identical duplicate rows, and split 60/20/20 into train/validation/test blocks. The supplied file retained all 4,213 rows. Do not call this a verified chronological forecast evaluation. Weather units not fully documented in the source are kept in their original numeric scale; no unsupported unit conversion is applied. The target is labeled kW by the source.

Training-median imputation is learned inside each training fold. KNN also uses fold-specific standardization. The target is never passed as an input. LightGBM validation permutation importance is predictive, not causal. The source may contain feature-timing or collection limitations that cannot be resolved without original collection metadata.

## Original LSTM data
Timestamped solar generation/weather data from Ani Kannal:
https://www.kaggle.com/datasets/anikannal/solar-power-generation-data

These four original CSVs are kept under `lstm_research/data/raw/`. Their source metadata and original-author rights are described in `lstm_research/data/DATASET_CARD.md`. They are not the source of the new 20-column schema and are not substituted for it. The LSTM module retains its own chronological splits and metrics.

## Explicit scenarios
The new EV planner's 24-hour weather profile is illustrative, generated from a dataset observation plus a smooth daylight/radiation profile, configured cloud cover and declared sunrise/sunset assumptions. Model predictions are normalized by a training-only target scale and clipped to the configured scenario PV size. This does not establish a real plant location or real installed capacity.

EV requests are user inputs. Default requests are examples. Building load and tariffs are assumptions, not Kaggle measurements. Tariffs are INR 4/6/10 per kWh in declared time bands, without export or demand charges. Charging efficiency is 92%, with no discharge, battery degradation or overnight sessions. Do not describe these modeled outcomes as measured field savings.

## Exact inputs
1. `temperature_2_m_above_gnd`
2. `relative_humidity_2_m_above_gnd`
3. `mean_sea_level_pressure_MSL`
4. `total_precipitation_sfc`
5. `snowfall_amount_sfc`
6. `total_cloud_cover_sfc`
7. `high_cloud_cover_high_cld_lay`
8. `medium_cloud_cover_mid_cld_lay`
9. `low_cloud_cover_low_cld_lay`
10. `shortwave_radiation_backwards_sfc`
11. `wind_speed_10_m_above_gnd`
12. `wind_direction_10_m_above_gnd`
13. `wind_speed_80_m_above_gnd`
14. `wind_direction_80_m_above_gnd`
15. `wind_speed_900_mb`
16. `wind_direction_900_mb`
17. `wind_gust_10_m_above_gnd`
18. `angle_of_incidence`
19. `zenith`
20. `azimuth`

Target: `generated_power_kw`.
