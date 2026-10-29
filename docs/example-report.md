# Regression findings

This is a synthetic learning exercise, not evidence about real students.

Training rows: 96. Test rows: 24. Random seed: 42.

| Model | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|
| linear_regression | 5.89 | 6.87 | 0.768 |
| mean_baseline | 13.72 | 17.39 | -0.489 |

Regression improves on the training-mean baseline on this fixed test split. MAE is the average absolute error in score points. The generator deliberately creates a linear relationship, so this is not a claim of real-world prediction quality.
