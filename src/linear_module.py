#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# Import numpy
import numpy as np

# Import pandas
import pandas as pd

# Import statsmodel
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Import tools from sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# # Create clases

# ### Walk-forward validation

class TrainValSplit():
        def __init__(self, n_splits, val_size):
            self.n_splits = n_splits
            self.val_size = val_size
    
        def get_n_splits(self, X, y, groups):
            return self.n_splits
    
        def get_val_size(self, X, y, groups):
            return self.val_size
    
        def split(self, X, y=None, groups=None):
            n_samples = len(X.index.get_level_values(0).unique()) # Here we collect the number of months in X
        
            k_fold_size = n_samples // self.n_splits
            indices = np.arange(len(X)) # Note that we need the length of the entire X here
        
            # We need the nr. of cross-sectional units each month:
            index = X.index.get_level_values(0).unique() # These contain the dates available
            N_cross = np.ones(n_samples)*np.nan # This will save the nr of cross sectional units at all dates
            for date in range(len(index)):
                N_cross[date] = len(X.loc[index[date]])
               
            for i in range(1, (self.n_splits+1)):
                total_obs = int(N_cross[:(k_fold_size*(i))].sum()) # total nr of observations for this fold
                start = 0
                stop = int(total_obs)
                mid = total_obs - int(N_cross[(k_fold_size*(i) - self.val_size*12):(k_fold_size*(i))].sum()) 
                yield indices[start: mid], indices[mid : stop]


# ### Blocked walk-forward validation

class BlockingTrainValSplit(): 
        def __init__(self, n_splits, val_size):
            self.n_splits = n_splits
            self.val_size = val_size
    
        def get_n_splits(self, X, y, groups):
            return self.n_splits
    
        def get_val_size(self, X, y, groups):
            return self.val_size
    
        def split(self, X, y=None, groups=None):
            n_samples = len(X.index.get_level_values(0).unique()) # Here we collect the number of months in X
        
            k_fold_size = n_samples // self.n_splits
            indices = np.arange(len(X)) # Note that we need the length of the entire X here
        
            # We need the nr. of cross-sectional units each month:
            index = X.index.get_level_values(0).unique() # These contain the dates available
            N_cross = np.ones(n_samples)*np.nan # This will save the nr of cross sectional units at all dates
            for date in range(len(index)):
                N_cross[date] = len(X.loc[index[date]])
            
            margin = 0
            for i in range(1, (self.n_splits+1)):
                start =  int(N_cross[:(k_fold_size*(i-1))].sum())
                stop =  int(N_cross[:(k_fold_size*(i))].sum())
                mid = stop - int(N_cross[(k_fold_size*(i) - self.val_size*12):(k_fold_size*(i))].sum()) 
                yield indices[start: mid], indices[mid : stop]


# # Create function

# ### FMB Regressions

def fmb_reg(Data, 
            Features, 
            Response, 
            Include_inter, 
            Method, 
            Years, 
            Refit, 
            Stats, 
            Data_stats, 
            Rounding=2, 
            maxlags=4):
    
    ''' 
    Performs Cross-sectional OLS and computes statistics.
  
    Parameters:
    ----------
    Data - A pandas MultiIndex containing the feature, response, return, and size variables.
    Features - A list of strings specifying the feature variables. 
    Response - A string specifying the response variable.
    Include_inter - A boolean to decide whether to include an intercept.
    Method - A string specifying the method (rolling, or expanding).
    Years - An integer specifying the years of rolling window/min years of expanding method.
    Refit - An integer specifying the months after which the model is refitted.
    Stats - A boolean to decide whether to return statistics.
    Data_stats - A pandas DataFrame containing summary statistics.
    Rounding - An integer specifying the rounding.
    maxlags - An integer specifying the Newey-West adjustment. 
    
    Returns: 
    ----------
    Coef_mean - A pandas DataFrame containing the means of coefficients.
    Inter_mean - A pandas DataFrame containing the mean of intercepts.
    Years - An integer specifying the years of rolling window/min years of expanding method.
    Reg_summary - A pandas DataFrame containing summary statistics.
    '''
       
    if Include_inter==True:
        # Fit an intercept       
        Regressor = LinearRegression(fit_intercept=True, 
                                     normalize=False, 
                                     n_jobs=-1)
    
    elif Include_inter==False:
        # Do not fit an intercept       
        Regressor = LinearRegression(fit_intercept=False, 
                                     normalize=False, 
                                     n_jobs=-1)
    
    # Create empty list for coefficients
    Coef =  []
    
    # Create empty list for intercepts
    Inter = []

    # Start of Loop
    start_date = Data.index[0][0]

    # End of Loop
    end_date = Data.index[-1][0]

    # Keep track of time
    Timestamp = start_date

    # Save index and shift it to avoid data leakage
    Index = Data.index.get_level_values(0).unique() + pd.offsets.BMonthEnd(1)

    # Add one month 
    TimeDelta = 0

    # Loop through months
    while Timestamp < end_date: 
    
        # Keep track of time
        Timestamp = start_date + pd.offsets.BMonthEnd(TimeDelta)
    
        # Perform Cross-sectional OLS regression for month i
        Regressor = Regressor.fit(X=Data.loc[Timestamp, Features],
                                  y=Data.loc[Timestamp, Response])

        # Add coefficient for month i to list
        Coef.append(pd.DataFrame(Regressor.coef_, columns=Features))
        
        # Add intercept for month i to list
        Inter.append(Regressor.intercept_)
    
        # Add one month 
        TimeDelta += 1 

    # Concat coefficients 
    Coef = pd.concat(Coef, axis=0).set_index(Index)
    
    # Create DataFrame for intercepts 
    Inter = pd.DataFrame(Inter, index=Index, columns=["Intercept"])

    if Method == "rolling":
        
        # Compute rolling average of Betas and intercepts
        Coef_mean = Coef.rolling(window=Years*12).mean().dropna(how="all")[::Refit]
        Inter_mean = Inter.rolling(window=Years*12).mean().dropna(how="all")[::Refit]
        
        # Forward fill missing values        
        idx = pd.date_range(Coef.index[0], Coef.index[-1], freq='BM')
        Coef_mean = Coef_mean.reindex(idx).fillna(method="ffill").dropna()
        Inter_mean = Inter_mean.reindex(idx).fillna(method="ffill").dropna()

        # Drop last line to avoid data leakage
        Coef_mean = Coef_mean.drop(Coef_mean.index[len(Coef_mean)-1])
        Inter_mean = Inter_mean.drop(Inter_mean.index[len(Inter_mean)-1])
        
    if Method == "expanding":
            
        if Years != 0:
            # Compute expanding average of Betas and intercepts
            Coef_mean = Coef.expanding(1).mean().iloc[Years*12-1:][::Refit]
            Inter_mean = Inter.expanding(1).mean().iloc[Years*12-1:][::Refit]
            
            # Forward fill missing values        
            idx = pd.date_range(Coef.index[0], Coef.index[-1], freq='BM')
            Coef_mean = Coef_mean.reindex(idx).fillna(method="ffill").dropna()
            Inter_mean = Inter_mean.reindex(idx).fillna(method="ffill").dropna()
            
        if Years == 0:
            # Compute expanding average of Betas and intercepts
            Coef_mean = Coef.expanding(1).mean()[::Refit]
            Inter_mean = Inter.expanding(1).mean()[::Refit]

            # Forward fill missing values        
            idx = pd.date_range(Coef.index[0], Coef.index[-1], freq='BM')
            Coef_mean = Coef_mean.reindex(idx).fillna(method="ffill").dropna()
            Inter_mean = Inter_mean.reindex(idx).fillna(method="ffill").dropna()
            
        # Drop last line to avoid data leakage
        Coef_mean = Coef_mean.drop(Coef_mean.index[len(Coef_mean)-1])
        Inter_mean = Inter_mean.drop(Inter_mean.index[len(Inter_mean)-1]) 
        
    ## Summary Statistic

    if Stats == True:
        
        # Create empty list for summary statistic
        Reg_summary = []

        # Add averages to summary statistic
        Reg_summary.append(pd.DataFrame(Coef.apply(np.mean, axis=0).values, columns=["Slope"]))

        # Create an empty list for t-values
        t_values = []

        # Loop through features
        for i in Features:
    
            # To obtain the Newey-West standard error correction for the sample mean of slope a series,
            # we fit an intercept-only model to the series
            reg = smf.ols(i+"~ 1", data=Coef).fit().get_robustcov_results(cov_type='HAC', maxlags=maxlags)
    
            # Add t-value for to summary statistic
            t_values.append(reg.tvalues)
    
        # Add t-values to summary statistic
        Reg_summary.append(pd.DataFrame(t_values, columns=["t-stat"]))

        # Add sample size to summary statistic
        Reg_summary.append(pd.DataFrame(Data_stats.iloc[:-1,-1].values, columns=["Stocks"]))

        # Concat summary statistic
        Reg_summary = pd.concat(Reg_summary, axis=1).set_index([Features]).round(Rounding)
 
        # return average of coef, average of inter, years of average and summary 
        return Coef_mean, Inter_mean, Years, Reg_summary
    
    else:
        # return average of coef, average of inter, and years of average
        return Coef_mean, Inter_mean, Years


# ### FMB Predictions

def fmb_pred(Data, 
             Coef, 
             Inter, 
             Features, 
             Stats=True, 
             Rounding=2):   
    
    ''' 
    Makes predictions based on slopes/intercepts and computes statistics.
  
    Parameters:
    ----------
    Data - A pandas MultiIndex containing the feature, response, return, and size variables.
    Coef - A pandas DataFrame containing the coefficients.
    Inter - A pandas DataFrame containing the intercepts.
    Features - A list of strings specifying the feature variables. 
    Stats - A boolean to decide whether to return statistics.
    Rounding - An integer specifying the rounding.

    Returns: 
    ----------
    Pre - A pandas MultiIndex containing the predictions.
    Pre_stats - A pandas DataFrame containing summary statistics.
    '''
    
    # Start of Loop
    start_date = Coef.index[0]

    # End of Loop
    end_date = Coef.index[-1]

    # Save ID
    ID = Data.loc[slice(start_date, end_date)].index.get_level_values(1)

    # Save Index
    Index = Data.loc[slice(start_date, end_date)].index.get_level_values(0)

    # Create empty list for predictions
    Pre = []

    # Keep track of time
    Timestamp = start_date

    # Add one month 
    TimeDelta = 0

    # Loop through months
    while Timestamp < end_date: 
    
        # Keep track of time
        Timestamp = start_date + pd.offsets.BMonthEnd(TimeDelta)
    
        # Add prediction for month i to list
        Pre.append(pd.DataFrame(np.add(np.dot(Data.loc[Timestamp, Features].values,
                   Coef.loc[Timestamp].values),
                   Inter.loc[Timestamp].values)))
        
        # Add one month
        TimeDelta += 1 
    
    # Concat prediction for all months
    Pre = pd.concat(Pre, axis=0).set_index([Index + pd.offsets.BMonthEnd(1), ID]).dropna()

    ## Summary Statistic
    if Stats==True:
    
        # Start Loop
        start_date = Pre.index[0][0]

        # End Loop
        end_date = Pre.index[-1][0]

        # Keep track of time
        Timestamp = start_date
    
        # Add one month 
        TimeDelta = 0

        # Create empty list for averages
        Pre_Avg = []

        # Create empty list for standard deviations
        Pre_Std = []

        # Create empty list for 10th Percentile
        Pre_10th = []

        # Create empty list for 90th Percentile
        Pre_90th = []

        # Loop through months
        while Timestamp < end_date: 
    
            # Keep track of time
            Timestamp = start_date + pd.offsets.BMonthEnd(TimeDelta)
    
            # Append average of month i to list
            Pre_Avg.append(Pre.loc[Timestamp].mean())
            
            # Append standard deviation of month i to list
            Pre_Std.append(Pre.loc[Timestamp].std())
    
            # Append 10th percentile of month i to list
            Pre_10th.append(Pre.loc[Timestamp].quantile(0.1)[0]) 
    
            # Append 90th percentile of month i to list
            Pre_90th.append(Pre.loc[Timestamp].quantile(0.9)[0]) 
    
            # Add one month
            TimeDelta += 1 
    
        # Concat statistics 
        Pre_stats = pd.DataFrame([Pre_Avg, Pre_Std, Pre_10th, Pre_90th], index=["Avg", "Std", "p10", "p90"]).T

        # Compute time-series average 
        Pre_stats = pd.DataFrame(Pre_stats.apply(np.mean, axis=0), columns=["Prediction"]).round(Rounding)      
     
        # Return predictions and statistics
        return Pre, Pre_stats
    
    else:
        
        # Return predictions 
        return Pre


# ### Panel Regressions + Predictions

def panel_reg(Data, 
              Features, 
              Response, 
              Include_inter, 
              Method, 
              Years, 
              Refit, 
              Stats,
              Rounding=2):
    
    ''' 
    Performs Pooled (Pool) or time fixed effect (TFE) regressions and computes statistics.
  
    Parameters:
    ----------
    Data - A pandas MultiIndex containing the feature, response, return, and size variables.
    Features - A list of strings specifying the feature variables. 
    Response - A string specifying the response variable.
    Include_inter - A boolean to decide whether to include an intercept.
    Method - A string specifying the method (rolling, or expanding).
    Years - An integer specifying the years of rolling window/min years of expanding method.
    Refit - An integer specifying the months after which the model is refitted.
    Stats - A boolean to decide whether to return statistics.
    Rounding - An integer specifying the rounding.

    Returns: 
    ----------
    Pre -  A pandas MultiIndex containing the predictions.
    Coef - A pandas DataFrame containing the coefficients.
    Pre_stats - A pandas DataFrame containing summary statistics.
    Years - An integer specifying the years of rolling window/min years of expanding method.
    '''
    
    if Include_inter==True:
        # Fit an intercept       
        Regressor = LinearRegression(fit_intercept=True, 
                                     normalize=False, 
                                     n_jobs=-1)
        
    elif Include_inter==False:
        # Do not fit an intercept       
        Regressor = LinearRegression(fit_intercept=False, 
                                     normalize=False, 
                                     n_jobs=-1)
    
    # Create empty list for coefficients
    Coef = []

    # Create empty list for predictions
    Pre = []

    # Start of slice
    Start = Data.index[0][0] + pd.offsets.BMonthEnd(Years*12)

    # End of slice
    End =  Data.index[-1][0]

    # Select index for coefficients
    Coef_Ind = Data[Start:End].index.get_level_values(0).unique()[::Refit]

    # Select IDs for predictions
    Pre_ID = Data.loc[slice(Start, End)].index.get_level_values(1)

    # Select index for predictions and shift it to avoid data leakage
    Pre_Ind = Data.loc[slice(Start, End)].index.get_level_values(0) + pd.offsets.BMonthEnd(1)

    # Start of Loop
    start_date = Data.index[0][0]      

    # End of Loop
    end_date = Data.index[-1][0] 

    # Keep track of time
    Timestamp = start_date 

    # Add one month 
    TimeDelta = 0

    # Loop thorugh months
    while Timestamp < end_date: 

        # Keep track of time
        Timestamp = start_date + pd.offsets.BMonthEnd(TimeDelta)
    
        # Check whether another regression is possible
        if Timestamp + pd.offsets.BMonthEnd(Years*12-1) < end_date:
        
            # Rolling methodti
            if Method == "rolling":
    
                # Select feature variables for training data
                X_train = Data.loc[Timestamp:Timestamp+pd.offsets.BMonthEnd(Years*12-1), 
                                   Features]
              
                # Select response variable for training data
                y_train = Data.loc[Timestamp:Timestamp+pd.offsets.BMonthEnd(Years*12-1), 
                                   Response]
            
            # Expanding method
            elif Method == "expanding":
        
                # Select feature variables for training data
                X_train = Data.loc[start_date:Timestamp+pd.offsets.BMonthEnd(Years*12-1), 
                                   Features]
                  
                # Select response variable for training data
                y_train = Data.loc[start_date:Timestamp+pd.offsets.BMonthEnd(Years*12-1), 
                                   Response]
    
            # Perform regression
            Regressor = Regressor.fit(X=X_train, y=y_train)
            
            # Add coefficient to list
            Coef.append(pd.DataFrame(Regressor.coef_, columns=Features))
                
           # print(len(np.unique(y_train.index.get_level_values(0))))
    
    
        # Check whether test over full period is possible
        if Timestamp+pd.offsets.BMonthEnd(Years*12+Refit-1) <= End:
        
            # Select features for test data
            X_test = Data.loc[Timestamp+pd.offsets.BMonthEnd(Years*12):Timestamp+pd.offsets.BMonthEnd(Years*12+Refit-1), 
                              Features]
        
            # Perform prediction and add it to the list
            Pre.append(pd.DataFrame(Regressor.predict(X_test), columns=["Prediction"]))
    
            #print(len(np.unique(X_test.index.get_level_values(0))))
        
        # Test over remaining period
        elif Timestamp+pd.offsets.BMonthEnd(Years*12) < End: 
        
            # Select features for test data
            X_test = Data.loc[Timestamp+pd.offsets.BMonthEnd(Years*12):, 
                              Features]

            # Perform prediction and add it to the list
            Pre.append(pd.DataFrame(Regressor.predict(X_test), columns=["Prediction"]))
        
           # print(len(np.unique(X_test.index.get_level_values(0))))

        # Add one month 
        TimeDelta += Refit
    
    # Concat predictions
    Pre = pd.concat(Pre, axis=0).set_index([Pre_Ind, Pre_ID])

    # Concat coefficients
    Coef = pd.concat(Coef, axis=0).set_index(Coef_Ind)

    ## Summary Statistic
    if Stats==True:
    
        # Start Loop
        start_date = Pre.index[0][0]

        # End Loop
        end_date = Pre.index[-1][0]

        # Keep track of time
        Timestamp = start_date
    
        # Add one month 
        TimeDelta = 0

        # Create empty list for averages
        Pre_Avg = []

        # Create empty list for standard deviations
        Pre_Std = []

        # Create empty list for 10th Percentile
        Pre_10th = []

        # Create empty list for 90th Percentile
        Pre_90th = []

        # Loop through months
        while Timestamp < end_date: 
    
            # Keep track of time
            Timestamp = start_date + pd.offsets.BMonthEnd(TimeDelta)
    
            # Append average of month i to list
            Pre_Avg.append(Pre.loc[Timestamp].mean())
            
            # Append standard deviation of month i to list
            Pre_Std.append(Pre.loc[Timestamp].std())
    
            # Append 10th percentile of month i to list
            Pre_10th.append(Pre.loc[Timestamp].quantile(0.1)[0]) 
    
            # Append 90th percentile of month i to list
            Pre_90th.append(Pre.loc[Timestamp].quantile(0.9)[0]) 
    
            # Add one month
            TimeDelta += 1 
    
        # Concat statistics 
        Pre_stats = pd.DataFrame([Pre_Avg, Pre_Std, Pre_10th, Pre_90th], index=["Avg", "Std", "p10", "p90"]).T

        # Compute time-series average 
        Pre_stats = pd.DataFrame(Pre_stats.apply(np.mean, axis=0), columns=["Prediction"]).round(Rounding)      
    
        # Return predictions and statistics
        return Pre, Coef, Pre_stats, Years
    
    else:
        
        # Return predictions 
        return Pre, Coef, Years


# ### Penalized Linear

def pl(Data, 
       Features, 
       Response, 
       Include_inter, 
       Method, 
       Years, 
       Refit, 
       Validation,
       Val_size, 
       N_splits,
       Penalty,
       param_grid, 
       Stats,
       Rounding=2):

    ''' 
    Performs Elastic net/Lasso/Ridge regressions and computes statistics.
  
    Parameters:
    ----------
    Data - A pandas MultiIndex containing the feature, response, return, and size variables.
    Features - A list of strings specifying the feature variables. 
    Response - A string specifying the response variable.
    Include_inter - A boolean to decide whether to include an intercept.
    Method - A string specifying the method (rolling, or expanding).
    Years - An integer specifying the years of rolling window/min years of expanding method.
    Refit - An integer specifying the months after which the model is refitted.
    Validation - A string specifying the validation method.
    Val_size - An integer specifying the years used for validation.
    N_splits - An integer specifying the number of splits used for validation.
    Penalty - A string specifying the penalty.
    param_grid - A list specifying the parameters to validate.
    Stats - A boolean to decide whether to return statistics.
    Rounding - An integer specifying the rounding.

    
    Returns: 
    ----------
    Pre -  A pandas MultiIndex containing the predictions.
    Summary - A pandas DataFrame containing coefs, alphas, and Ns.
    Pre_stats - A pandas DataFrame containing summary statistics.
    Years - An integer specifying the years of rolling window/min years of expanding method.
    '''
   
        
    if Include_inter==True:
        
        if Penalty=="elnet":
            # Fit an intercept       
            Regressor = ElasticNet(fit_intercept=True, 
                                   normalize=False, 
                                   random_state=1)
            
        elif Penalty=="lasso":
            # Fit an intercept       
            Regressor = Lasso(fit_intercept=True, 
                              normalize=False, 
                              random_state=1)
            
        elif Penalty=="ridge":
            # Fit an intercept       
            Regressor = Ridge(fit_intercept=True, 
                              normalize=False, 
                              random_state=1)
            
        
    elif Include_inter==False:
        
        if Penalty=="elnet":
            # Do not fit an intercept
            Regressor = ElasticNet(fit_intercept=False, 
                                   normalize=False, 
                                   random_state=1) 
        
        elif Penalty=="lasso":
            # Do not fit an intercept      
            Regressor = Lasso(fit_intercept=False, 
                              normalize=False, 
                              random_state=1)
            
        elif Penalty=="ridge":
            # Do not fit an intercept      
            Regressor = Ridge(fit_intercept=False, 
                              normalize=False, 
                              random_state=1)
    

    # Walk-forward validation
    if Validation == "wf":
        tvs = TrainValSplit(n_splits=N_splits, val_size=Val_size) 
        
    # Blocked walk-forward validation
    elif Validation == "bwf":
        tvs = BlockingTrainValSplit(n_splits=N_splits, val_size=Val_size) 
        
        
    # Set up pipeline
    steps=[('scaler', StandardScaler(copy=True, with_mean=True, with_std=True)), 
           ('linear', Regressor)] 
    pipeline=Pipeline(steps)
        
    
    # Set up girdsearch
    gs = GridSearchCV(pipeline, 
                      param_grid=param_grid, 
                      cv=tvs, 
                      verbose=0, 
                      n_jobs=-1, 
                      scoring="neg_mean_squared_error", 
                      refit=True, 
                      return_train_score=False,
                      iid=False)
    
    # Create empty list for parameters
    Par = []
    
    # Create empty list for number of selected features
    N = []
    
    # Create empty list for coefficients
    Coef = []
    
    # Create empty list for predictions
    Pre = []

    # Start of slice
    Start = Data.index[0][0] + pd.offsets.BMonthEnd(Years*12)

    # end of slice
    End =  Data.index[-1][0]

    # Select index for summary
    Ind = Data[Start:End].index.get_level_values(0).unique()[::Refit]

    # Select IDs for predictions
    Pre_ID = Data.loc[slice(Start, End)].index.get_level_values(1)

    # Select index for predictions and shift it to avoid data leakage
    Pre_Ind = Data.loc[slice(Start, End)].index.get_level_values(0) + pd.offsets.BMonthEnd(1)

    # Start of Loop
    start_date = Data.index[0][0]      

    # End of Loop
    end_date = Data.index[-1][0] 

    # Keep track of time
    Timestamp = start_date 

    # Add one month 
    TimeDelta = 0
        
      # Loop thorugh months
    while Timestamp < end_date: 

        # Keep track of time
        Timestamp = start_date + pd.offsets.BMonthEnd(TimeDelta)
    
        # Check whether another regression is possible
        if Timestamp + pd.offsets.BMonthEnd(Years*12-1) < end_date:
            
            # Set up scaler
            #scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        
            # Rolling method
            if Method == "rolling":
                
                
                # Select feature variables for training data
                X_train = Data.loc[Timestamp:Timestamp+pd.offsets.BMonthEnd(Years*12-1), 
                                       Features]
        
                # Select response variable for training data
                y_train = Data.loc[Timestamp:Timestamp+pd.offsets.BMonthEnd(Years*12-1), 
                                   Response]
                                      
            # Expanding method
            elif Method == "expanding":
                
                
                # Select feature variables for training data
                X_train= Data.loc[start_date:Timestamp+pd.offsets.BMonthEnd(Years*12-1), 
                                      Features]
            
                # Select response variable for training data
                y_train = Data.loc[start_date:Timestamp+pd.offsets.BMonthEnd(Years*12-1), 
                                   Response]
    
            # Perform cross validation
            gs.fit(X_train, y_train)
            
            # Add parameters to list
            Par.append(gs.best_params_)
            
            # Add number of selected features to list 
            N.append(np.count_nonzero(gs.best_estimator_.steps[1][1].coef_))
            
            # Add coefficient to list
            Coef.append(pd.DataFrame(gs.best_estimator_.steps[1][1].coef_.reshape(1,len(Features)), columns=Features))
        
            #print("train length", len(np.unique(y_train.index.get_level_values(0))))
            #print("train start", y_train.index[1])
            #print("train end", y_train.index[-1])
          
                
        # Check whether test over full period is possible
        if Timestamp+pd.offsets.BMonthEnd(Years*12+Refit-1) <= End:
        
            # Select features for test data
            X_test = Data.loc[Timestamp+pd.offsets.BMonthEnd(Years*12):Timestamp+pd.offsets.BMonthEnd(Years*12+Refit-1), 
                              Features]
        
            # Perform prediction and add it to the list
            Pre.append(pd.DataFrame(gs.predict(X_test), columns=["Prediction"]))

            #print("test length", len(np.unique(X_test.index.get_level_values(0))))
            #print("test start", X_test.index[1])
            #print("test end", X_test.index[-1])
        
        # Test over remaining period
        elif Timestamp+pd.offsets.BMonthEnd(Years*12) < End: 
            
            # Select features for test data
            X_test = Data.loc[Timestamp+pd.offsets.BMonthEnd(Years*12):, 
                              Features]

            # Perform prediction and add it to the list
            Pre.append(pd.DataFrame(gs.predict(X_test), columns=["Prediction"]))
        
            #print("test length", len(np.unique(X_test.index.get_level_values(0))))
            #print("test start", X_test.index[1])
            #print("test end", X_test.index[-1])

        # Add one month 
        TimeDelta += Refit
    
    # Concat predictions
    Pre = pd.concat(Pre, axis=0).set_index([Pre_Ind, Pre_ID])
  
    # Concat coefficients, alphas, and number of selected features
    Summary = pd.concat([pd.concat(Coef, axis=0).set_index(Ind.date), 
                         pd.DataFrame(N, columns=["N"]).set_index(Ind.date),
                         pd.DataFrame(Par).set_index(Ind.date)], 
                         axis=1)
    
    
    ## Summary Statistic
    if Stats==True:
    
        # Start Loop
        start_date = Pre.index[0][0]

        # End Loop
        end_date = Pre.index[-1][0]

        # Keep track of time
        Timestamp = start_date
    
        # Add one month 
        TimeDelta = 0

        # Create empty list for averages
        Pre_Avg = []

        # Create empty list for standard deviations
        Pre_Std = []

        # Create empty list for 10th Percentile
        Pre_10th = []

        # Create empty list for 90th Percentile
        Pre_90th = []

        # Loop through months
        while Timestamp < end_date: 
    
            # Keep track of time
            Timestamp = start_date + pd.offsets.BMonthEnd(TimeDelta)
    
            # Append average of month i to list
            Pre_Avg.append(Pre.loc[Timestamp].mean())
            
            # Append standard deviation of month i to list
            Pre_Std.append(Pre.loc[Timestamp].std())
    
            # Append 10th percentile of month i to list
            Pre_10th.append(Pre.loc[Timestamp].quantile(0.1)[0]) 
    
            # Append 90th percentile of month i to list
            Pre_90th.append(Pre.loc[Timestamp].quantile(0.9)[0]) 
    
            # Add one month
            TimeDelta += 1 
    
        # Concat statistics 
        Pre_stats = pd.DataFrame([Pre_Avg, Pre_Std, Pre_10th, Pre_90th], index=["Avg", "Std", "p10", "p90"]).T

        # Compute time-series average 
        Pre_stats = pd.DataFrame(Pre_stats.apply(np.mean, axis=0), columns=["Prediction"]).round(Rounding)      
       
        # Return predictions and statistics
        return Pre,  Summary, Pre_stats, Years
    
    else:
        
        # Return predictions 
        return Pre, Summary, Years


# ### Plot coef

def plot_coef(Coef, Method, Years):
    
    ''' 
    Plots coefficients.
  
    Parameters:
    ----------
    Coef - A pandas DataFrame containing the coefficients.
    Method - A string specifying the method (rolling, or expanding).
    Years - An integer specifying the years of rolling window/min years of expanding method.
    '''

    if Method == "rolling":
        # Plot average slopes
        Coef.plot.line(figsize=[15,8], fontsize=13).set_title(
        "%.0f-year rolling slope estimates, %.0f to %.0f" 
        % (Years, Coef.index[0].year, Coef.index[-1].year), size=20)
    
    if Method == "expanding":
        # Plot average slopes
        Coef.plot.line(figsize=[15,8], fontsize=13).set_title(
        "expanding slope estimates starting after %.0f years, %.0f to %.0f" 
        % (Years, Coef.index[0].year, Coef.index[-1].year), size=20)

