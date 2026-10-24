# Understand your project

## 1. The problem in simple words
Imagine your college has solar panels and ten EV charging points. Sunlight changes throughout the day. Cars arrive at different times and must receive enough energy before leaving. Charging everything immediately can buy unnecessary electricity from the grid. Waiting too long can leave a car undercharged.

Your project asks: **Can we predict sunlight and schedule charging together while respecting physical limits?**

## 2. The complete flow
1. Read actual solar generation and weather records from Kaggle.
2. Clean timestamps, flag missing readings and form 15-minute intervals.
3. Train an LSTM to forecast the next four hours of solar output.
4. Use an optimizer on historical training days to create good charging examples.
5. Train a second LSTM to imitate those charging actions.
6. During replay, predict solar power, ask the charging LSTM for a proposal, and correct it using current constraints.
7. Update each car's remaining energy and repeat for the next interval.
8. Compare cost and charging service against other controllers.

This is a historical simulation. The same control structure could later connect to a real system, but this version does not send commands to chargers.

## 3. Why LSTM?
LSTM means Long Short-Term Memory. It is a recurrent neural network with a memory cell. Solar power is a sequence: what happened during the previous hours helps describe what may happen next. The input gate chooses new information, the forget gate controls retained memory, and the output gate controls the exposed hidden state.

Unlike a model that sees only one row, the solar LSTM sees 24 consecutive 15-minute observations (six hours). Each observation has normalized AC power, irradiation, temperature, sine and cosine of time of day, and a validity flag. It predicts 16 intervals (four hours).

The model learns a correction to yesterday's same-time solar profile. Its final forecast is the previous-day baseline plus that learned correction, clipped to a plausible normalized range. This is a genuine trainable LSTM: all gate and output weights are optimized using backpropagation through time. `lstm.py` implements the mathematics directly in NumPy to avoid large deep-learning dependencies.

## 4. What the second LSTM does
The charging LSTM receives, for each EV: current time, time until departure, remaining energy, connection state, current solar power, mean upcoming solar forecast, current building load, tariff and grid limit. It processes eight consecutive states (two hours); initial histories are padded with the first state.

It proposes a fraction of the 7.2 kW charger limit. A shared model is applied to each of ten EVs. Because individual proposals could collectively exceed site capacity, a separate layer checks them together.

Training labels come from the optimizer. This is called **imitation learning**: the LSTM learns examples of the optimizer's decisions. The optimizer is the teacher; the LSTM is the student. The student can be fast without always matching the teacher.

## 5. How the optimizer works
Power is in kW. Energy is in kWh. For one 15-minute slot:

`delivered_energy_kWh = charging_power_kW × 0.25`

The optimizer first minimizes the total energy cars fail to receive. Keeping that minimum fixed, it then minimizes electricity import cost. This avoids apparent savings produced merely by refusing to charge cars.

It enforces nonnegative charging, charger limits, connection windows, energy requests and a site grid limit. Solar surplus can be curtailed. It does not sell electricity, discharge cars, or model battery degradation. Therefore continuous linear programming is sufficient; a mixed-integer solver is not needed for this specific simplified model.

The offline oracle can see the full historical day and establishes an optimistic benchmark. The rolling optimizer only sees forecasts/current measurements and replans every interval. The deployed learned controller must never receive actual future weather or solar power as an input.

## 6. Why a constraint layer is necessary
Neural networks produce approximate numbers. They can ask for charging after a car leaves, exceed a power limit or overcharge a requested energy amount. The projection layer clips impossible requests and shares site capacity. It gives priority to imminent departures when individual remaining charging time becomes tight.

It logs intervention counts and energy shortfalls. It cannot guarantee all future deadlines when vehicles compete for limited power, and it cannot make an impossible request possible. Real equipment still requires electrical safety protection.

## 7. What makes this project distinctive
- Two separate, explainable LSTM roles rather than a single prediction chart.
- Optimization-generated teaching data for the charging model.
- Visible action corrections and remaining violations.
- A service-first benchmark that exposes undercharging.
- A forecast reserve setting to explore uncertainty.
- A demand stress setting to demonstrate infeasible requests.
- Same-day comparisons on identical requests and tariffs.
- An included dataset, trained models and reproducible evidence.

These are the project's design choices and integration features. They do not prove patentability or research novelty.

## 8. What is real and what is simulated?
The solar generation and weather files are real Kaggle records. The site-size scaling is a scenario transformation. EV arrivals, departures, energy requests, building demand and prices are generated assumptions. The dataset contains no measured EV sessions. The dashboard labels the mode accordingly.

The original measured mean inverter power is normalized by a training-only scale, then multiplied by the dashboard solar scale. Do not call that scale the verified original plant capacity. The six-hour sequence is causal: the solar model never sees future observed weather.

## 9. Understanding evaluation
MAE measures typical absolute forecast error. RMSE penalizes larger mistakes more strongly. Both are reported in normalized units in `training.json`; multiply by a chosen scenario kW scale for scenario-unit errors. Daylight metrics use observed normalized PV > 0.01 as an evaluation mask, not an online feature.

The first 60% of timestamps trains the model. The next 10% chooses the best checkpoint. The following 10% calibrates prediction intervals. The final 20% is held out for testing. A 90% empirical error band means its widths were estimated from calibration errors; actual test coverage can differ.

Cost must always be read beside unmet energy. A controller that buys less electricity but leaves cars undercharged is not automatically better. The comparison also records peak grid power, served sessions, interventions and computation time.

Read `RESULTS.md` for this package's actual measured outcomes. The supplied model did not beat the previous-day forecast on this short held-out record; reporting that is a valid scientific result. Do not claim universal accuracy or guaranteed savings.

The controlled-error study deliberately creates artificial forecasts from future actual PV plus noise. This is a sensitivity experiment, not an online forecast or operational result. Five seeds make random-noise variation visible. The 0% case gives perfect solar information only.

## 10. Five-minute demonstration
1. Run `app.py` and open the dashboard.
2. Point out the historical replay label and dataset source.
3. Select a test date and run the experiment.
4. Explain solar, building, EV and grid curves.
5. Switch between immediate, learned and rolling controllers.
6. Compare cost **and** unmet energy in the table.
7. Explain one EV's heatmap and received/requested energy.
8. Increase demand stress and lower grid capacity; observe shortfalls rather than hiding them.
9. Turn on a forecast reserve and discuss conservative decisions.
10. Export the action CSV and show the training/evaluation reports.

## 11. Viva questions
**Why two LSTMs?** One predicts an external variable (solar generation); the other proposes a control action (charging power).

**Why not use only the optimizer?** The learned policy explores a speed/performance trade-off. The optimizer remains an important comparator and can perform better.

**How do you avoid data leakage?** Chronological target splits, training-only normalization, historical/current inputs, and training-period teacher episodes. Teacher labels may use future data offline, but policy features may not.

**Is this really real time?** It is a causal historical replay at 15-minute control intervals. A real deployment needs sensor/charger communication, timestamp freshness checks and hardware testing.

**Does lower forecast error always lower charging cost?** Not necessarily. Error timing, tariffs, flexibility and constraints matter. This motivates the controlled-error experiment.

**Why NumPy instead of TensorFlow?** The project implements genuine LSTM training directly in NumPy, keeping installation small and compatible with the tested Python 3.14.7 environment. It is educational and CPU-friendly, not a replacement for industrial deep-learning frameworks at scale.

**What is the main limitation?** About one month of solar data and simulated EV/building inputs; results cannot establish seasonal or field performance.

**What comes next?** More seasonal data, real charging sessions, stronger joint-deadline constraints, additional baselines and hardware-in-the-loop validation.
