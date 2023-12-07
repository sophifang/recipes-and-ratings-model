# Food.com Total Time to Make Recipe Prediction Model

## Framing the Problem

## Baseline Model
### Model Description
For our baseline model, we trained a regression model using two numerical features:
- `n_steps`: a quantitative, numeric attribute
- `n_ingredients`: a quantitative, numeric attribute

Our `Pipeline`
1. log-scales `n_steps` and `n_ingredients`, then
2. predicts `log minutes` (the natural logarithm of `minutes`) using a linear regression model (using the transformed `n_steps` and `n_ingredients`).

### Model Explanation: Scaling Transformations
Knowing that non-linear growth is commmon in real world data, we wanted to visualize the relationship between 1) `n_steps` and `minutes` and 2) `n_ingredients` and `minutes` respectively since we wanted to train a linear regression model on `n_steps` and `n_ingredients` to predict `minutes`.
<div><span><iframe src="assets/scatter-nsteps-minutes.html" width=400 height=300 frameBorder=0></iframe></span><span>
<iframe src="assets/scatter-ningredients-minutes.html" width=400 height=300 frameBorder=0></iframe></span></div>
<div><span><iframe src="assets/residual-nsteps-minutes.html" width=400 height=300 frameBorder=0></iframe></span><span>
<iframe src="assets/residual-ningredients-minutes.html" width=400 height=300 frameBorder=0></iframe></span></div>
As evident by the scatter and regression plots above, both `n_steps` and `minutes` and `n_ingredients` and `minutes` don't have linear relationships, meaning that these columns need to be transformed before any `Pipeline` can be created.

In order to be able to use a linear regression model on `n_steps` and `n_
ingredients` to predict `minutes`, we needed to transform their complicated, non-linear relationship into a linear relationship because linear relationships are easy for models like linear regression to use.

### Model Performance
Our baseline model has a R^2 score of 0.2265, meaning that 22.65% of the variance of `log minutes` is explained by the variables of log-scaled `n_steps` and `n_ingredients`. Additionally, our RMSE value is 0.9713, which represents the average distance between the observed `log minutes` values and the predicted `log minutes` values. This RMSE value will be helpful when creating our final model, as it is particularly useful for comparing the fit of different regression models (the lower the RMSE, the better the model fits the dataset).

Based on these metrics, our baseline model is not good since only 22.65% of the variance of our response variable `log minutes` is explained by the predictor variables `n_steps` and `n_ingredients`. Our goal for the our final model was to imrprove our regression model so that more of the variance of `log minutes` is explained (higher R^2) and to generate a lower RMSE to indicate a better model fit.

## Final Model

## Fairness Analysis
