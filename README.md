# Food.com Total Time to Make Recipe Prediction Model

## Framing the Problem
### Model Description
For our baseline model, we trained a regression model using two numerical features:
- `n_steps`: a quantitative, numeric attribute
- `n_ingredients`: a quantitative, numeric attribute

Our `Pipeline`
1. log-scales `n_steps` and `n_ingredients`, then
2. predicts the natural logarithm of `minutes` using a linear regression model (using the transformed `n_steps` and `n_ingredients`.

### Model Performance

## Baseline Model

## Final Model

## Fairness Analysis
