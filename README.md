# Machine learning for cross-sectional stock selection

Master's thesis, MSc Finance, Frankfurt School of Finance & Management (October 2019).
Supervisors: Prof. Dr. Wolfgang Schmidt and Prof. Dr. Jan Nagler.
Full text: [thesis.pdf](thesis.pdf) ("Dissecting Characteristics via Machine Learning for Stock Selection").

## Question

Can machine learning methods that allow for nonlinearities and interactions between firm characteristics forecast the cross-section of monthly stock returns better than linear models? Do better forecasts translate into better portfolios?

## Data

| Item | Detail |
|---|---|
| Universe | S&P Developed BMI (USD), small, mid and large caps, about 4,200 stocks a month |
| Period | October 1989 to December 2018, monthly |
| Out of sample | October 1999 to December 2018 |
| Predictors | 21 firm characteristics: value (book, earnings, forward earnings, sales, dividend and net payout yields), momentum, residual momentum, short- and long-term reversal, seasonality, quality (ROA, ROE, leverage, distance to default), investment and issuance (asset growth, liability change, share issuance), risk (beta, volatility) and size |
| Fundamentals | Compustat and Worldscope, lagged three and six months to avoid look-ahead |
| Target | next-month total return in USD, demeaned across stocks each month |
| Preprocessing | characteristics winsorised at the 1st and 99th percentiles each month; standardised only where the model requires it |

The data is licensed and is not included in this repository.

## Method

The thesis follows the framework of Gu, Kelly and Xiu (2020) and models the expected return of each stock as a function of its characteristics, $E_t(r_{i,t+1}) = g(x_{i,t})$, focusing on the cross-section only.

| Family | Models |
|---|---|
| Benchmarks | Fama-MacBeth regressions with 10-year rolling slopes; pooled OLS |
| Penalised linear | ridge, lasso, elastic net |
| Generalised linear | spline expansion of each characteristic with a group lasso penalty |
| Trees | random forest, gradient boosted regression trees |
| Neural networks | feed-forward networks with one to three hidden layers (NN1 to NN3) |

- **Out-of-sample design.** Each model is estimated on a 10-year window, tuned on 12-month validation folds that lie ahead of the training data, refitted with the chosen hyperparameters, and used to forecast the next 12 months. The procedure then rolls forward one year.
- **Evaluation 1: predictive slope.** Each month, regress realised returns on the forecasts across stocks (Lewellen, 2015). A slope of 1 means unbiased forecasts; below 1 means the forecasts exaggerate dispersion. The t-statistics use Newey-West standard errors with 4 lags.
- **Evaluation 2: portfolios.** Each month, sort stocks into deciles of the forecast and form an equal-weighted long-short portfolio, top decile minus bottom decile.

## Results

Out of sample, October 1999 to December 2018 (thesis Tables 9 and 10).

| Model | Predictive slope | t-stat | Long-short return (% per month) | Std (% per month) | Information ratio | Max drawdown (%) | t-stat |
|---|---|---|---|---|---|---|---|
| Fama-MacBeth | 0.61 | 5.82 | 1.37 | 3.78 | 1.26 | 22 | 5.32 |
| Pooled OLS | 0.51 | 5.26 | 1.34 | 3.80 | 1.22 | 20 | 4.92 |
| Ridge | 0.51 | 5.26 | 1.34 | 3.80 | 1.22 | 20 | 4.92 |
| Lasso | 0.54 | 5.28 | 1.35 | 3.89 | 1.21 | 20 | 4.91 |
| Elastic net | 0.54 | 5.29 | 1.35 | 3.87 | 1.21 | 20 | 4.90 |
| GLM (splines) | 0.77 | 6.35 | 1.42 | 3.70 | 1.33 | 21 | 5.74 |
| Random forest | 0.88 | 8.10 | 1.94 | 3.56 | 1.89 | 26 | 7.43 |
| Boosted trees | 0.69 | 7.19 | 2.00 | 3.56 | 1.95 | 15 | 6.97 |
| NN1 | 0.51 | 8.17 | 1.58 | 2.90 | 1.89 | 15 | 7.78 |
| NN2 | 0.51 | 7.95 | 1.68 | 3.11 | 1.86 | 10 | 8.31 |
| NN3 | 0.60 | 8.68 | 1.56 | 2.94 | 1.84 | 7 | 7.92 |

**Main findings**

- **Every model has significant out-of-sample predictive power.** But all predictive slopes are below 1, so every model exaggerates the cross-sectional dispersion of expected returns to some degree.
- **Regularisation alone does not help.** Ridge, lasso and elastic net are no better than the linear benchmarks, and the spline-based GLM improves only slightly.
- **Interactions are what matters.** Tree ensembles and neural networks, which allow interactions between characteristics, raise the long-short information ratio from about 1.2 to about 1.9. Neural networks do so with the lowest volatility and drawdowns.
- **Robustness (thesis section 6 and appendix H).**
  - On the FTSE Developed Index (large and mid caps, about 1,400 stocks a month), predictive performance is somewhat weaker. Models with nonlinear interactions still outperform the linear ones.
  - Value-weighted portfolios give similar results.
  - The ranking is robust to changes in the tuning grids.

## Limitations

- **Gross returns.** Portfolio returns are before trading costs. Monthly turnover is similar across models, so costs would lower all spreads by a similar amount, but the absolute numbers are gross.
- **Small caps.** The main results use equal weights and include small caps, where anomalies are typically stronger and harder to trade.
- **Few characteristics.** The predictor set is small (21 characteristics), and macroeconomic variables are deliberately excluded.
- **Single sample.** Hyperparameter grids were tuned on validation folds, but the comparison across models uses a single out-of-sample period.

## Code

The code is the thesis code from 2019. The data is not included, so it cannot be run from this repository.

| Path | Contents |
|---|---|
| `src/data_module.py` | Loading the monthly characteristic panels, cleaning, winsorising and scaling |
| `src/linear_module.py` | Fama-MacBeth, pooled OLS, ridge, lasso and elastic net with time-series validation splits |
| `src/non_linear_module.py` | Spline GLM with group lasso, random forest, boosted trees and neural networks |
| `src/evaluation_module.py` | Predictive slopes, decile portfolios, information and Sortino ratios, drawdowns, model comparison |
| `notebooks/` | Main notebooks for each model family and the comparison, with their saved outputs |
| `requirements.txt` | Python packages used (2019 versions not recorded) |
