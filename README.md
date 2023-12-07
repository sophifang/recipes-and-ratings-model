# Food.com Total Time to Make Recipe Prediction Model

## Framing the Problem
### Model Description
For our baseline model, we trained a regression model using two numerical features:
- `n_steps`: a quantitative, numeric attribute
- `n_ingredients`: a quantitative, numeric attribute

Our `Pipeline`
1. log-scales `n_steps` and `n_ingredients`, then
2. predicts `log minutes` (the natural logarithm of `minutes`) using a linear regression model (using the transformed `n_steps` and `n_ingredients`.

### Model Performance
Our baseline model has a R^2 score of 0.2265, meaning that 22.65% of the variance of `log minutes` is explained by the varaince of log-scaled `n_steps` and `n_ingredients`.

## Baseline Model

## Final Model

## Fairness Analysis
