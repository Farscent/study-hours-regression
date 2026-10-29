[![Tests](https://github.com/Farscent/study-hours-regression/actions/workflows/tests.yml/badge.svg)](https://github.com/Farscent/study-hours-regression/actions/workflows/tests.yml)

# Study Hours and Exam Scores: A Regression Exercise

A small machine learning project that explains linear regression from data preparation to evaluation. It uses one input variable and a simple baseline so the important ideas stay visible.

**Research question:** In a fictional dataset, can weekly study hours predict an exam score more accurately than always predicting the average training score?

## Example results

![Fitted line and held-out residuals for the synthetic dataset](docs/overview.png)

| Model | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|
| Linear regression | 5.89 | 6.87 | 0.768 |
| Training-mean baseline | 13.72 | 17.39 | -0.489 |

The model improves on the baseline on this fixed test split. MAE is measured in score points, not percent accuracy. These results describe a deliberately linear synthetic dataset. [Read the generated example report](docs/example-report.md).

## Learning objectives

- Separate features from a target.
- Split observations into training and test sets.
- Fill missing inputs without leaking test information.
- Fit and interpret a straight-line regression model.
- Compare a model with a meaningful baseline using MAE, RMSE, and R².

## Dataset and scope

The included [students.csv](data/students.csv) contains **120 synthetic observations**. Student IDs are fictional, and two study-hour values are deliberately missing.

| Column | Meaning |
|---|---|
| student_id | Unique fictional ID, excluded from model inputs |
| study_hours | Weekly study hours; the only model feature |
| exam_score | Score out of 100; the prediction target |

[make_demo_data.py](make_demo_data.py) creates scores from <code>35 + 5 × study_hours + deterministic noise</code>. This known relationship makes the mechanics easy to inspect. It also means good performance is expected by design: it does not show that study hours alone can predict real academic performance.

## CRISP-DM workflow

| Phase | Application |
|---|---|
| Business understanding | Test whether the model improves on a training-mean baseline. |
| Data understanding | Inspect ranges, missing inputs, and the study-hours/score relationship. |
| Data preparation | Validate records; split 80/20; learn the missing-value median from training rows only. |
| Modeling | Fit LinearRegression and DummyRegressor with the same training data. |
| Evaluation | Compare both models on the same held-out test set and inspect residuals. |
| Deployment | Provide a reproducible script, a report, and an optional example prediction. |

The split uses <code>random_state=42</code>: 96 training rows and 24 test rows. The imputer and model are combined in a scikit-learn pipeline. Fitting the pipeline on training rows prevents the median from learning anything about the test set.

## Run the project

Use **Python 3.11, 3.12, or 3.13**. Clone the project and create an environment:

~~~bash
git clone https://github.com/Farscent/study-hours-regression.git
cd study-hours-regression
python -m venv .venv
~~~

~~~powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
~~~

~~~bash
# macOS / Linux
source .venv/bin/activate
~~~

~~~bash
python -m pip install -r requirements.txt
python model.py
python model.py --hours 4
python -m unittest discover -s tests -v
~~~

If PowerShell blocks activation, call <code>.\.venv\Scripts\python.exe</code> directly. The included data runs offline after installation.

## Understanding the model

~~~text
predicted score = intercept + slope × study hours
residual = actual score − predicted score
~~~

The intercept is the fitted score at zero hours. The slope is the fitted change in score associated with one more hour. Association is not causation.

The baseline always predicts the **training set's mean score**. A model that cannot beat this baseline on held-out rows has not demonstrated useful prediction for this exercise.

| Metric | Plain-language interpretation |
|---|---|
| MAE | Average absolute prediction error, measured in score points; lower is better. |
| RMSE | Error in score points that penalizes larger mistakes more strongly; lower is better. |
| R² | Improvement relative to a test-mean reference; 1 is perfect, 0 matches that reference, and negative values are possible. |

The DummyRegressor uses the **training** mean, while the R² formula uses the **test** mean as its reference. Therefore, the baseline's test R² need not be exactly zero. R² is not a percentage of predictions that are correct.

## Outputs

| File | Contents |
|---|---|
| outputs/report.md | Actual test metrics and a baseline comparison |
| outputs/metrics.json | Metrics, split sizes, imputation value, slope, and intercept |
| outputs/test_predictions.csv | Held-out actual scores, predictions, and residuals |
| outputs/regression.png | Fitted line and held-out residual plot |

Run the script to obtain the results rather than relying on a claimed accuracy percentage. The report states whether regression beats the baseline on the fixed test split.

The <code>--hours</code> option uses the model fitted on training rows. It rejects inputs outside the observed training range to discourage extrapolation.

## Limitations

This dataset is synthetic, small, and deliberately linear. The evaluation demonstrates a workflow, not educational validity. A single split has sampling uncertainty, and the model does not express prediction intervals. Real outcomes depend on many factors not represented here. Predictions are not automatically constrained to 0–100.

The test set should be used for final evaluation, not repeated parameter selection. If you extend the project, use cross-validation within the training data before evaluating a final model on untouched test data.

## Suggested study order

Open [walkthrough.ipynb](walkthrough.ipynb), then follow <code>load_students()</code>, <code>build_model()</code>, and <code>train_and_evaluate()</code> in [model.py](model.py).

Try these exercises:

1. Calculate MAE manually for five held-out rows.
2. Explain why student_id is not a useful feature.
3. Increase the synthetic noise and observe how the metrics change.
4. Use cross-validation on the training set and report the variability.

**Questions to be able to answer:** What is data leakage? Why use a baseline? Why does a small training error not prove good generalization?

## References

- [scikit-learn: common pitfalls and data leakage](https://scikit-learn.org/stable/common_pitfalls.html)
- [IBM: CRISP-DM overview](https://www.ibm.com/docs/en/spss-modeler/saas?topic=dm-crisp-help-overview)

This is a student learning exercise, not a tool for evaluating actual students.


## Optional notebook

From this project folder, install the notebook tools and open the walkthrough:

~~~bash
python -m pip install -r requirements-notebook.txt
python -m notebook walkthrough.ipynb
~~~

Choose the project's Python environment and use **Run All**. The notebook imports the same tested functions as the script and adds explanations and intermediate tables.

## Validation and license

The project has been tested locally on Windows with Python 3.13. GitHub Actions checks Python 3.11 and 3.12; see the status badge at the top for the current result. The analysis notebooks include saved example outputs and can be rerun from top to bottom.

Code and the synthetic demonstration data are available under the [MIT License](LICENSE).
