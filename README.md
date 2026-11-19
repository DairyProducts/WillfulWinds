# Willful Winds

The very official project repository for the Willful Winds BN project!!!!!

## Six-hour persistence experiment

From PowerShell in the project directory, run the validation or final comparison.

```powershell
.\.venv\Scripts\python.exe evaluate.py
.\.venv\Scripts\python.exe evaluate.py --split test
```

Both runs build hourly features from the two raw station files. To refresh all 12
processed CSVs, run `process_data.py` with the same Python interpreter. The CSVs
retain missing values so each analysis can select its own required columns.

### Stations and forecast cases

Changping and Nongzhanguan were fixed in the pilot and retained for the final
comparison. Their coordinates are 40.217° N, 116.230° E and 39.937° N, 116.461° E,
respectively. Their great-circle separation is about 36.8 km. During the
pre-validation training period, each station had usable predictors and outcomes
for over 96% of possible starts for each pollutant. The pair was selected in
advance of the final test. This design supports conclusions about these two sites.
[Station coordinates, Table 1](https://www.mdpi.com/2071-1050/14/9/5104)
· [UCI Beijing Multi-Site Air Quality data](https://doi.org/10.24432/C5RK5G)

The unit of analysis is one station-hour. The PM-only network uses both current
PM readings and their changes since three hours earlier. The weather-only network
uses wind direction, wind speed and rain measured at forecast time. The combined
network uses both sets of inputs. For each target, all three networks train and
score on identical complete starts when that pollutant is elevated. An outcome
is positive if the same pollutant is elevated at hour six. Pollution can clear
and recur in the intervening hours.

### Training and scoring

Validation covers March 2015 through February 2016. The final test covers March
2016 through February 2017. Each fit uses all earlier forecast starts whose
six-hour outcomes precede the scoring period. Both stations share the cut points.
Each concentration must strictly exceed its training 90th percentile to be
elevated. Trends and wind speeds use training terciles. Each threshold uses the
available training measurements for that variable, before the target-specific
missing-value filter.

Each forecast input is a root with an edge to the target. This graph assumes
marginal independence among the inputs. With all parents observed, exact
inference selects the matching target CPT entry. K2 smoothing adds one count per
outcome, so the chance of persistence is `(persisted + 1) / (training_hours + 2)`.
Brier `gain` is the PM-only mean squared probability error minus the combined
model's error. `weather_only_gain` compares the training persistence rate as a
constant forecast against the weather-only network. A positive value in either
comparison means that weather improved prediction for that comparison. These
are measures of predictive information. The paired 95% intervals resample
seven-day blocks of forecast times 5,000 times with both stations together
(seed 797).

Each run writes to `outputs/validation/` or `outputs/test/`.

| File | Contents |
| --- | --- |
| `metrics.csv` | Pooled and station-level thresholds, elevated starts, persistence rates, Brier scores, gains and intervals |
| `predictions.csv` | Station, issue time, outcome time, observed event, training-rate baseline and all three model probabilities |
| `training_coverage.csv` | Possible and usable training starts for each station and target |
| `target_cpts.csv` | Fitted probabilities and observed training counts for elevated starts, including unseen combinations |
| `models.md` | Fitted DAGs, cut points and matched dry/rainy CPT examples chosen using training counts |
