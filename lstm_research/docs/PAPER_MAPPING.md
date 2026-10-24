# From your paper to this implementation

Your attached file is titled "Research Proposal and System Design Report" and explicitly says trained models and experimental results were not supplied. This project turns its main pipeline into a working, evaluated prototype; it is not an exact implementation of every proposed experiment.

| Paper component | Delivered implementation | Difference or remaining work |
|---|---|---|
| Solar LSTM | Trainable NumPy LSTM with input/forget/output gates, BPTT, Adam and checkpoint selection | One 20-unit layer, six-hour history, four-hour direct forecast, previous-day residual connection; not proposed stacked 64/32 units and 24-hour horizon |
| Weather and PV inputs | Actual Kaggle Plant 1 inverter/weather files | No live sensors, issued weather archive, plant geolocation or astronomical night mask |
| Separate learned scheduler | Shared per-EV LSTM, 16 hidden units, eight-step state history | Ten ports, per-EV outputs instead of a monolithic port vector |
| Offline optimization teacher | Perfect-information HiGHS linear program | LP replaces MILP because charging-only/no-export assumptions require no binary variables |
| Service before cost | Two LP solves: minimize unmet kWh, then minimize cost with minimum unmet locked | No degradation or monthly demand charges |
| Rolling optimization | Re-solves each 15-minute interval, executes first action | Uses four-hour LSTM prediction then previous-day continuation to departure/day end |
| Feasibility projection | Connection, charger, remaining-energy, current grid and deadline-urgency checks | Greedy earliest-departure projection does not certify joint future feasibility |
| Audit trail | Proposed-to-executed aggregate correction, causes, grid violations, energy, cost and timing | No hardware communication or field protection system |
| Information separation | Online inputs use past/current measured data and known scenario information | Teacher training uses causal previous-day forecasts; deployment LSTM forecasts introduce distribution shift |
| Baselines | Previous-day solar; immediate charging; rolling LP; oracle | Gradient boosting and forecast-free learned policy are not implemented |
| Uncertainty | Held-out per-horizon empirical error intervals and configurable conservative reserve | No statistical coverage guarantee under time dependence or distribution shift |
| Error study | 0/5/10/20% nominal Gaussian error, five seeds, six eligible held-out days, achieved nRMSE logged | Controlled forecasts start from actual PV and are explicitly artificial; no bias/time-shift study yet |
| Real time | Local interactive dashboard and causal 15-minute historical replay | No actual live operation or charger integration |
| Data sources | Included original Kaggle solar CSVs | EV sessions, tariffs and building load are declared simulations, not ACN/BDG2 measurements |
| Delivery | Source, trained weights, exact data, metrics, test suite, documentation | Not patent novelty certification or finished peer-reviewed validation |

Distinctive project features are the integration of two explicit LSTMs with an optimizer, a visible constraint-correction trail, service-first comparisons, and an interactive uncertainty/stress workbench. These are implementation differentiators, not a claim of novelty across the scientific literature.

Further work: real EV sessions; better chronological out-of-fold solar forecasts for scheduler training; additional baselines; geometry-based night constraints when the true location is known; larger seasonal datasets; scheduling efficiency and SOC; stronger joint-deadline feasibility; hardware-in-the-loop testing. Do not represent these future items as delivered features.
