# Food.com Total Time to Make Recipe Prediction Model

## Framing the Problem
The amount of time it takes to finish an activity or for an event to happen can vary among the respective categories because there are different internal and external factors that affect their duration of time. For this project, our prediction problem is to see how different recipes are prepared and whether factors such as the amount of ingredients each recipe requires or the difficulty rating of each recipe is, would have an affect on how long the recipes take to prepare.

**Response Variable**: The response variable is minutes, which is quantitative data. We chose minutes as it is used to determine what the preparation time is per recipe. In the real world, the duration of time it takes to cook a meal is important to each person because there are people who can only afford to spend short periods of their day preparing meals versus others who have more time and they do not mind spending longer when it comes to cooking.

**Evaluation Metric**: Because we are implementing a regression model, we would evaluate metrics such as $R^2$ and Root Mean Squared Error (RMSE). Both metrics are useful in their own way as $R^2$ measures the quality of a linear fit and RMSE measures the quality of a regression model's predictions. The higher $R^2$ is, it shows it does a better job at modeling data. The smaller RMSE is, it shows it does a better job at modeling data. It is also possible for the RMSE to increase on unseen data when features are added to the model. Taking this into consideration, we prioritized improving the $R^2$ metric based on its value increasing and the RMSE metric based on its value decreasing, while keeping track of the features we add to our model.

**Information We Know**: At the time of prediction, we would know `n_steps`, `n_ingredients`, `name`, `description`, and `review`. Columns like `n_steps`, `n_ingredients`, `name`, and `description` are all provided for people who would want to know what the recipe is like and what ingredients they must obtain before they might attempt the recipe for themselves. The `review` column confirms that people have actually seen each feature of the recipe and made an effort to prepare the meal. This tells us that each feature mentioned earlier, and the `review` column, are all known at the time of prediction.

For our prior analysis, which can be found [here](https://sophifang.github.io/recipes-and-ratings-analysis/), we cleaned a dataset where we excluded the review column because the column did not have much use. For our model, we decided to use a dataset where the review column is included. Our reasoning for this decision is so we would be able to figure how difficult each recipe was based on what the reviewees thought of the recipe. We were able to do this by querying each review to see if it contained text that had similar meanings to the words "easy" or "hard". We then assigned each review its own difficulty based on which word it had contained. If the review did not contain either word, we considered the recipe's difficulty to be neutral.

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
As evident by the scatter and regression plots above, both `n_steps` and `minutes` and `n_ingredients` and `minutes` don't have linear relationships, meaning that these columns need to be transformed before any `Pipeline` can be created. In order to be able to use a linear regression model on `n_steps` and `n_
ingredients` to predict `minutes`, we needed to transform their complicated, non-linear relationship into a linear relationship because linear relationships are easy for models like linear regression to use.

If we take a look at the individual columns themselves, we can see that the data for all three columns is skewed. Based on this knowledge, we can conduct log transformations in order to spread out clumps of data and bring together spread-out data.
<div><span><iframe src="assets/hist-nsteps.html" width=266 height=300 frameBorder=0></iframe></span><span>
<iframe src="assets/hist-ningredients.html" width=266 height=300 frameBorder=0></iframe></span><span>
<iframe src="assets/hist-minutes.html" width=266 height=300 frameBorder=0></iframe></span></div>
<div><span><iframe src="assets/hist-log-nsteps.html" width=266 height=300 frameBorder=0></iframe></span><span>
<iframe src="assets/hist-log-ningredients.html" width=266 height=300 frameBorder=0></iframe></span><span>
<iframe src="assets/hist-log-minutes.html" width=266 height=300 frameBorder=0></iframe></span></div>
Now that we have log-transformed our variables, we have successfully had non-linear relationships _more_ linear. While there is still evidence of heteroscedasticity in the residual plot, the residuals are more uniformly scattered and has no pattern compared to previously.\
\
[insert scatter plots (2) and residual plots (2)]

### Model Performance
Our baseline model has a $R^2$ score of 0.2265, meaning that 22.65% of the variance of `log minutes` is explained by the variables of log-scaled `n_steps` and `n_ingredients`. Additionally, our RMSE value is 0.9713, which represents the average distance between the observed `log minutes` values and the predicted `log minutes` values. This RMSE value will be helpful when creating our final model, as it is particularly useful for comparing the fit of different regression models (the lower the RMSE, the better the model fits the dataset).

Based on these metrics, our baseline model is not good since only 22.65% of the variance of our response variable `log minutes` is explained by the predictor variables `n_steps` and `n_ingredients`. Our goal for the our final model was to imrprove our regression model so that more of the variance of `log minutes` is explained (higher $R^2$) and to generate a lower RMSE to indicate a better model fit.

## Final Model

### Hyperparameters
For the baseline model, we used the default hyperparameters for linear regression, which are as follows:
- `fit_intercept`: True
- `copy_X`: True
- `n_jobs`: None
- `positive`: False

For our final model, we wanted to see if changing the values of any of these hyperparameters will optimize our model further. However, there are a couple of hyperparameters that won't be relevant to test for this model:
- `copy_X = True` makes sure that a copy of `X` is passed in. Setting `copy_X = False` would mean that `X` could possibly be overwritten during `fit`, so we will **not** be testing this hyperparameter.
- `n_jobs` specifies the number of jobs for the computation (CPU usage), something that we will **not** be testing for this model. And lastly,
- `positive = True` is only supported for dense arrays, which is not applicable to our data. Therefore, we will be sticking to the default `positive = False`.

This means that the only relevant hyperparameter for us to test was `fit-intercept` (whether or not to calculate the intercept for the linear regression model), which we did using `GridSearchCV`. Below is a `DataFrame` displaying the mean validation accuaries of different combinations of hyperparameter settings.

[insert dataframe]

The result of `GridSearchCV` is that the best hyperparameters for our model would be `{'fit_intercept': True}`, the default setting. Therefore, based on our results, we will be using all the default hyperparameter settings for the final model.

### K-fold Cross-Validation
Given that linear regression is an equation and does not have as many meaningful hyperparameters to tweak compared to other models, we decided to use K-fold cross-validation to further improve our model by figuring out the best combination of features to your for our final model.

In addition to the two features from our baseline model (log-scaled `n_steps` and `n_ingredients`), we wanted to explore the following features to improve our model:
- `difficulty`:
- `name`:
- `description`:

## Fairness Analysis
