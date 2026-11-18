# Cross-validated performance (mean ± SD over 5 leave-trips-out folds)

## Heater on/off (classification)

| Model | Features | ROC AUC | Average precision | Balanced accuracy | F1 | MCC | Brier |
|---|---|---|---|---|---|---|---|
| Logistic regression | exogenous | 0.933 ± 0.090 | 0.973 ± 0.029 | 0.923 ± 0.071 | 0.942 ± 0.064 | 0.806 ± 0.179 | 0.084 ± 0.084 |
| Decision tree (depth 3) | exogenous | 0.912 ± 0.087 | 0.952 ± 0.043 | 0.904 ± 0.088 | 0.951 ± 0.042 | 0.793 ± 0.185 | 0.064 ± 0.060 |
| Random forest | exogenous | 0.973 ± 0.043 | 0.992 ± 0.013 | 0.953 ± 0.058 | 0.963 ± 0.064 | 0.885 ± 0.166 | 0.042 ± 0.046 |
| Gradient boosting | exogenous | 0.968 ± 0.052 | 0.989 ± 0.017 | 0.951 ± 0.061 | 0.963 ± 0.063 | 0.884 ± 0.168 | 0.049 ± 0.080 |
| Logistic regression | exogenous-electrical | 0.944 ± 0.068 | 0.978 ± 0.022 | 0.932 ± 0.052 | 0.943 ± 0.059 | 0.818 ± 0.148 | 0.080 ± 0.075 |
| Decision tree (depth 3) | exogenous-electrical | 0.891 ± 0.096 | 0.938 ± 0.058 | 0.882 ± 0.093 | 0.943 ± 0.041 | 0.761 ± 0.173 | 0.075 ± 0.060 |
| Random forest | exogenous-electrical | 0.967 ± 0.045 | 0.990 ± 0.013 | 0.891 ± 0.098 | 0.956 ± 0.044 | 0.809 ± 0.188 | 0.052 ± 0.046 |
| Gradient boosting | exogenous-electrical | 0.960 ± 0.050 | 0.987 ± 0.017 | 0.905 ± 0.091 | 0.960 ± 0.040 | 0.822 ± 0.183 | 0.053 ± 0.059 |
| Logistic regression | exogenous+thermal | 0.955 ± 0.069 | 0.985 ± 0.022 | 0.934 ± 0.069 | 0.951 ± 0.065 | 0.837 ± 0.178 | 0.065 ± 0.086 |
| Decision tree (depth 3) | exogenous+thermal | 0.961 ± 0.041 | 0.981 ± 0.018 | 0.931 ± 0.051 | 0.946 ± 0.052 | 0.816 ± 0.144 | 0.063 ± 0.061 |
| Random forest | exogenous+thermal | 0.978 ± 0.041 | 0.994 ± 0.011 | 0.931 ± 0.094 | 0.972 ± 0.041 | 0.868 ± 0.189 | 0.036 ± 0.039 |
| Gradient boosting | exogenous+thermal | 0.975 ± 0.044 | 0.993 ± 0.012 | 0.954 ± 0.057 | 0.968 ± 0.052 | 0.891 ± 0.158 | 0.043 ± 0.071 |

## Heating power (regression)

| Model | Features | R² | RMSE (W) | MAE (W) |
|---|---|---|---|---|
| Ridge regression | exogenous | 0.597 ± 0.192 | 785 ± 188 | 487 ± 133 |
| Decision tree (depth 3) | exogenous | 0.551 ± 0.172 | 840 ± 218 | 510 ± 138 |
| Random forest | exogenous | 0.716 ± 0.165 | 650 ± 204 | 413 ± 97 |
| Gradient boosting | exogenous | 0.720 ± 0.213 | 622 ± 257 | 383 ± 137 |
| Ridge regression | exogenous-electrical | 0.604 ± 0.174 | 781 ± 174 | 484 ± 126 |
| Decision tree (depth 3) | exogenous-electrical | 0.549 ± 0.170 | 842 ± 220 | 515 ± 141 |
| Random forest | exogenous-electrical | 0.671 ± 0.176 | 708 ± 227 | 434 ± 112 |
| Gradient boosting | exogenous-electrical | 0.639 ± 0.221 | 732 ± 269 | 432 ± 142 |
| Ridge regression | exogenous+thermal | 0.647 ± 0.187 | 738 ± 213 | 482 ± 139 |
| Decision tree (depth 3) | exogenous+thermal | 0.567 ± 0.190 | 826 ± 225 | 457 ± 104 |
| Random forest | exogenous+thermal | 0.646 ± 0.224 | 715 ± 248 | 458 ± 142 |
| Gradient boosting | exogenous+thermal | 0.691 ± 0.248 | 655 ± 279 | 393 ± 161 |
