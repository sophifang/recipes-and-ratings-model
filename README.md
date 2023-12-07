# Food.com Total Time to Make Recipe Prediction Model

## Framing the Problem
For our baseline model, we trained a regression model using two log-scaled numerical features:
- `n_steps`: a quantitative, numeric attribute
- `n_ingredients`: a quantitative, numeric attribute

Our `Pipeline`
1. log-scales `n_steps` and `n_ingredients`, then
2. predicts `minutes` using a linear regression model (using the transformed `n_steps` and `n_ingredients`.


## Baseline Model

## Final Model

## Fairness Analysis
