#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# Import statsmodels
import statsmodels.formula.api as smf

# Import pandas
import pandas as pd

# Import numpy
import numpy as np

# Import linear regression from sklearn
from sklearn.linear_model import LinearRegression

# Import Plotly to plot comparison
import plotly as py
import plotly.graph_objs as go
py.offline.init_notebook_mode(connected=True)


# # Create function

# ### Evaluation table

def eval_data(Data, Predictions, Return, Size):
    ''' 
    Creates an evaluation table containing prediction, returns, and market caps t-1.
  
    Parameters:
    Data - A pandas MultiIndex containing the feature, response, return, and size variables.
    Predictions - A pandas MultiIndex containing the predictions.
    Return - A string specifying the return variable.
    Size - A string specifying the size variable.
    
    Returns: 
    Eval_Data - A pandas MultiIndex that constains predictions, returns, and market caps t-1.
    '''
    # Select returns
    Actual = Data.loc[Predictions.index[0]:Predictions.index[-1], Return]

    # Reset Index of actual returns
    Actual = Actual.reset_index()

    # Rename colums
    Actual.columns = ["Index", "ID", "Return"]

    # Select size
    Size = Data.loc[Predictions.index[0]:Predictions.index[-1], Size]

    # Reset index 
    Size = Size.reset_index()

    # Shift size to obtain size of previous month
    Size.iloc[:,0] = Size.iloc[:,0] + pd.offsets.BMonthEnd(1) 

    # Rename columns
    Size.columns = ["Index", "ID", "Mcap_BMI_USD t-1"]

    # Reset index of predictions
    Prediction = Predictions.reset_index()
    
    # Rename columns
    Prediction.columns = ["Index", "ID", "Prediction"]


    # Merge prediction, return, and size variable
    Eval_Data = pd.merge(pd.merge(Prediction, Actual, how="left", left_on=["Index","ID"], right_on = ["Index","ID"]),
                           Size, how="left", left_on=["Index","ID"], right_on = ["Index","ID"])

    # Set Index
    Eval_Data = Eval_Data.set_index(["Index","ID"]).dropna()

    # return table
    return Eval_Data


# ### Evaluation Regression

def linear_eval_reg(Eval_Data, Rounding=2):
    ''' 
    Regresses actual return on predicted return and computes statistics.
    
    Parameters:
    Eval_Data - A pandas MultiIndex that constains predictions, returns, and market caps t-1 (Eval_Data).
    Rounding - An integer specifying the rounding.
    
    Returns: 
    Pred_summary - A pandas DataFrame that constains summary statistics.
    '''

    # Start of Regression Loop
    start_date = Eval_Data.index[0][0]

    # End of Regression Loop
    end_date = Eval_Data.index[-1][0]

    # Keep track of time
    Timestamp = start_date

    # Save Index
    Index = Eval_Data.index.get_level_values(0).unique() 

    # Add one month 
    TimeDelta = 0

    # Create empty list for coefficients
    Pred_coef =  []

    # Loop thorugh months
    while Timestamp < end_date: 
    
        # Keep track of time
        Timestamp = start_date + pd.offsets.BMonthEnd(TimeDelta)
    
        # Perform Cross-sectional OLS regression for month i
        regressor = LinearRegression(fit_intercept=True).fit(
                                     X=pd.DataFrame(Eval_Data.loc[Timestamp, "Prediction"]), 
                                     y=pd.DataFrame(Eval_Data.loc[Timestamp, "Return"]))
    
        # Add coefficient for month i to list
        Pred_coef.append(pd.DataFrame(regressor.coef_, columns=["Beta"]))
    
        # Add one month 
        TimeDelta += 1 

    # Concat coefficients
    eval_coef = pd.concat(Pred_coef, axis=0).set_index(Index)
    
    ## Summary Statistic
    
    # Create empty list for summary statistic
    Pred_summary = []

    # Add average slopes to summary statistic
    Pred_summary.append(pd.DataFrame(eval_coef.apply(np.mean, axis=0).values))

    # To obtain the Newey-West standard error correction for the sample mean of slope series, 
    # we fit an intercept-only model to the series
    reg = smf.ols("Beta ~ 1", data=eval_coef).fit().get_robustcov_results(cov_type='HAC', maxlags=4)
        
    # Add standard error to summary statistic
    Pred_summary.append(pd.DataFrame(reg.bse))
    
    # Add t-values to summary statistic
    Pred_summary.append(pd.DataFrame(reg.tvalues))

    # Concat summary statistic
    Pred_summary = pd.concat(Pred_summary, axis=0).set_axis(
                            ["Prediction"], axis='columns', inplace=False).round(Rounding).set_axis(
                            ["Slope", "S.E.", "t-stat"], axis='index', inplace=False).round(Rounding)
    
    # Return statistics
    return Pred_summary


# ### Evaluation Portfolios

def portfolios(Eval_Data, N=10):
    ''' 
    Creates a summary statistic for decile portfolios.
    
    Parameters:
    Eval_Data - A pandas MultiIndex that constains predictions, returns, and market caps t-1 (Eval_Data).
    N - An integer specifying the number of desiered portfolios.
    
    Returns: 
    Portfolios_summary - A pandas DataFrame containing summary statistic.
    Portfolios_cumReturn - A pandas DataFrame containing cummulative returns.
    '''
    
    def information_ratio(Returns, Periods=12):
        '''
        Computes the Information ratio for the portfolio.

        Parameters:
        Returns - A pandas Series representing period returns.
        Periods - An integer specifying the period, e.g. Monthly (12), Daily (252), etc.
        
        Returns: 
        information_ratio - The information ratio of the portfolio.
        '''
        # Compute the information ratio
        information_ratio = np.sqrt(Periods) * (np.mean(Returns)) / np.std(Returns)
        
        return information_ratio
    
    
    def sortino_ratio(Returns, Periods=12):
        '''
        Computes the Sortino ratio for the portfolio, based on a 
        benchmark of zero (i.e. no risk-free rate information).

        Parameters:
        Returns - A pandas Series representing period returns.
        Periods - An integer specifying the period, e.g. Monthly (12), Daily (252), etc.
        
        Returns: 
        sortino_ratio - The sortino ratio of the portfolio.
        '''
        # Compute the Sortino ratio
        semi_var = Returns[Returns<0]**2
        semi_var = semi_var.sum()/len(Returns)
        sortino = np.sqrt(semi_var)
        sortino_ratio = np.sqrt(Periods) * Returns.mean() / sortino
        
        return sortino_ratio
    
    
    def drawdowns(Returns, Drawdown=False, Duration=False): 
        '''
        Calculates the largest peak-to-trough drawdown of the portfolio
        as well as the duration of the drawdown. 

        Parameters:
        Returns - A pandas Series representing period returns.
        Drawdown - A boolean to decide whether to return drawdown.
        Duration - A boolean to decide whether to return drawdown duration.

        Returns:
        drawdown - Highest peak-to-trough drawdown.
        drawdown duration - Duration of drawdown.
        '''

        # Calculate the running maximum
        cumsum = Returns.cumsum()

        # Ensure the value never drops below 1
        end = (np.maximum.accumulate(cumsum) - cumsum).idxmax() # end of the period
        low = cumsum[end]

        # Calculate the percentage drawdown
        start = (cumsum[:end]).idxmax() # start of period
        high = cumsum[start]

        # Calculate max drawdown and max drawdown duration
        maxDD = np.absolute(low-high).round(0)
        maxDD_Durarion = np.absolute(end-start).days

        # Plot max Drawdown 
        #plt.plot(cumsum)
        #plt.plot([end, start], [cumsum[end], cumsum[start]], 'o', color='Red', markersize=10)

        # Return Drawdown if True
        if Drawdown==True:
            return maxDD

        # Return Drawdown duration if True
        if Duration==True:
            return maxDD_Durarion
        
        
    # Define number of portfolios
    N = N
    
    # Define portfolio names
    names = []
    for i in range(1,N+1):
        name = "Portfolio_%.0f" % i
        names.append(name)
    
    
    # Sort portfolios by predictions
    Eval_Data = Eval_Data.sort_values('Prediction', ascending=False).sort_index(level='Index', sort_remaining=False)

    # Save index
    Index = Eval_Data.index.get_level_values(0).unique()

    # Start of Loop
    start_date = Eval_Data.index[0][0]

    # End of Loop
    end_date = Eval_Data.index[-1][0]

    # Create empty list for summary of portfolios  
    Portfolios_summary = []

    # Create empty list for monthly portfolio returns 
    Portfolios_r_m = []

    # Create empty list for monthly portfolio predictions 
    Portfolios_p_m = []

    # Create empty list for monthly turnovers
    Portfolios_Turnover_m = []
    
    # Keep track of time
    Timestamp = start_date

    # Add one month 
    TimeDelta = 0
                
    # Loop thorugh months
    while Timestamp < end_date: 
            
        # Keep track of time
        Timestamp = start_date + pd.offsets.BMonthEnd(TimeDelta)
    
        # Create empty list for monthly predictions 
        Prediction_monthly = []
        
        # Create empty list for monthly returns 
        Return_monthly = []

        # Create empty list for monthly turnovers
        Turnover_monthly = []
    
        # Create empty lists for H-L weights
        Weights_HL = []
        Weights_new_HL = []

        # Loop over N portfolios
        for i in range(N):
        
            # Define start of slicing
            Start = int(Eval_Data.loc[Timestamp].shape[0]/N)*i
        
            # Define end of slicing
            End = int(Eval_Data.loc[Timestamp].shape[0]/N)*(i+1)
            
            # If it is not the last iteration select equal size of stocks
            if i != N-1:
            
                # Compute weights for portfolio i in month t
                Weights = Eval_Data.loc[Timestamp, "Mcap_BMI_USD t-1"].iloc[Start:End].div(
                          Eval_Data.loc[Timestamp, "Mcap_BMI_USD t-1"].iloc[Start:End].sum())
            
                # Append weights of H portfolio in month t
                if i == 0:
                    Weights_HL.append(Weights)
                
                # Compute prediction for portfolio i in month t
                Prediction = Eval_Data.loc[Timestamp, ["Prediction"]].iloc[Start:End].values.mean()
                
                # Compute returns for portfolio i in month t
                Return = Eval_Data.loc[Timestamp, ["Return"]].iloc[Start:End].values.mean()
                
                # There is no turnover for the first month
                if Timestamp < end_date:
                    
                    # Compute weights for portfolio i in month t+1
                    Weightsnew = Eval_Data.loc[Timestamp+pd.offsets.BMonthEnd(1), "Mcap_BMI_USD t-1"].iloc[Start:End].div(
                                 Eval_Data.loc[Timestamp+pd.offsets.BMonthEnd(1), "Mcap_BMI_USD t-1"].iloc[Start:End].sum())
                
                    # Append new weights of H portfolio in t+1
                    if i == 0:
                        Weights_new_HL.append(Weightsnew)
                    
            # If it is the last iteration select remaining stocks
            else:
            
                # Compute weights for portfolio i in month t
                Weights = Eval_Data.loc[Timestamp, "Mcap_BMI_USD t-1"].iloc[Start:].div(
                          Eval_Data.loc[Timestamp, "Mcap_BMI_USD t-1"].iloc[Start:].sum())
            
                # Append weights of L portfolio in t
                Weights_HL.append(Weights)
                            
                # Compute prediction for portfolio i in month t
                Prediction = Eval_Data.loc[Timestamp, ["Prediction"]].iloc[Start:].values.mean()
                
                # Compute returns for portfolio i in month t
                Return = Eval_Data.loc[Timestamp, ["Return"]].iloc[Start:].values.mean()
                
                # There is no turnover for the first month
                if Timestamp < end_date:
                    
                    # Compute weights for month t+1
                    Weightsnew = Eval_Data.loc[Timestamp+pd.offsets.BMonthEnd(1), "Mcap_BMI_USD t-1"].iloc[Start:].div(
                                 Eval_Data.loc[Timestamp+pd.offsets.BMonthEnd(1), "Mcap_BMI_USD t-1"].iloc[Start:].sum())
                
                    # Append new weights of L portfolio in t+1
                    Weights_new_HL.append(Weightsnew)
                    
            if Timestamp < end_date:        
                # Compute Turnover for portfolio i in month t
                Turnover = Weightsnew.subtract(Weights, fill_value=0).abs().sum()

            # Add prediction for portfolio i in month t
            Prediction_monthly.append(Prediction)
            
            # Add return for portfolio i in month t
            Return_monthly.append(Return)
            
            # Add turnover for portfolio i in month t
            Turnover_monthly.append(Turnover)
        
        # Compute turnover for H-L portfolio in month t 
        if Timestamp < end_date:
            Turnover_HL = pd.concat(Weights_new_HL).subtract(pd.concat(Weights_HL), fill_value=0).abs().sum()/2

        # Append turnover for H-L portfolio in month t     
        Turnover_monthly.append(Turnover_HL)
          
        # Add one month 
        TimeDelta += 1   
            
        # Add monthly prediction of portfolio i
        Portfolios_p_m.append(pd.DataFrame(Prediction_monthly))
    
        # Add monthly returns of portfolio i
        Portfolios_r_m.append(pd.DataFrame(Return_monthly))
            
        # Add monthly turnover of portfolio i
        Portfolios_Turnover_m.append(pd.DataFrame(Turnover_monthly))
        
    # Concat all portfolios 
    Portfolios_Turnover_m = pd.concat(Portfolios_Turnover_m, axis=1).T  
    
    # Drop last line since there is no turnover for the first month
    Portfolios_Turnover_m = Portfolios_Turnover_m[:-1]

    # Concat all portfolios
    Portfolios_r_m = pd.concat(Portfolios_r_m, axis=1).T.set_index(Index).set_axis(
                          [names], axis='columns', inplace=False)  

    # Compute returns for H-L portfolios
    Portfolios_r_m["HL"] = Portfolios_r_m.loc[:,"Portfolio_1"].values - Portfolios_r_m.loc[:,"Portfolio_10"].values


    # Concat all portfolios 
    Portfolios_p_m = pd.concat(Portfolios_p_m, axis=1).T.set_index(Index).set_axis(
                                  [names], axis='columns', inplace=False) 

    # Compute predictions for H-L portfolios
    Portfolios_p_m["HL"] = Portfolios_p_m.loc[:,"Portfolio_1"].values - Portfolios_p_m.loc[:,"Portfolio_10"].values

    # Compute cum Returns for all portfolios
    Portfolios_cumReturn = pd.DataFrame(Portfolios_r_m.apply(np.cumsum, axis=0))


    # Compute mean of predictions for all portfolios
    Portfolios_Pre = pd.DataFrame(Portfolios_p_m.apply(np.mean, axis=0).values.reshape(N+1,1), 
                                  columns=["Prediction"]).round(2)

    # Add to portfolio summary
    Portfolios_summary.append(Portfolios_Pre)


    # Compute mean of returns for all portfolios
    Portfolios_Return = pd.DataFrame(Portfolios_r_m.apply(np.mean, axis=0).values.reshape(N+1,1),
                                    columns=["Return"]).round(2)

    # Add means to portfolio summary
    Portfolios_summary.append(Portfolios_Return)


    # Compute standard deviations for all portfolios
    Portfolios_Std = pd.DataFrame(Portfolios_r_m.apply(np.std, axis=0).values.reshape(N+1,1),
                                 columns=["Std"]).round(2)
        
    # Add standard deviations to portfolio summary
    Portfolios_summary.append(Portfolios_Std)

    # Compute annualized Information Ratios for all portfolios
    Portfolios_IR = pd.DataFrame(Portfolios_r_m.apply(information_ratio, axis=0).values.reshape(N+1,1),
                                columns=["IR"]).round(2)
    
    # Add annualized Information Ratios to portfolio summary
    Portfolios_summary.append(Portfolios_IR)

    # Compute sortino ratios for all portfolios
    Portfolios_sortino = pd.DataFrame(Portfolios_r_m.apply(sortino_ratio, axis=0).values.reshape(N+1,1),
                                     columns=["Sortino"]).round(2)
               
    # Add to portfolio summary
    Portfolios_summary.append(Portfolios_sortino)
    
    # Compute Max 1M Loss for all portfolios
    Portfolios_Max_Loss = pd.DataFrame(np.absolute(Portfolios_r_m.apply(np.min, axis=0).values.reshape(N+1,1)),
                                      columns=["Max_1M_Loss"]).round()

    # Add to portfolio summary
    Portfolios_summary.append(Portfolios_Max_Loss)
    
    # Compute maximum drawdown for all portfolios
    Portfolios_Max_draw= pd.DataFrame(Portfolios_r_m.apply(drawdowns, Drawdown=True, axis=0).values.reshape(N+1,1),
                                     columns=["Max_Draw"])
                  
    # Add to portfolio summary
    Portfolios_summary.append(Portfolios_Max_draw)
    
    # Compute maximum drawdown duration for all portfolios
    Portfolios_Max_draw_dur= pd.DataFrame(Portfolios_r_m.apply(drawdowns, Duration=True, axis=0).values.reshape(N+1,1),
                                         columns=["Max_Draw_dur"])

    # Add to portfolio summary
    Portfolios_summary.append(Portfolios_Max_draw_dur)
    
    # Compute tunrover for all portfolios
    Portfolios_TO = pd.DataFrame(Portfolios_Turnover_m.apply(np.mean, axis=0).values.reshape(N+1,1).round(2),
                                columns=["Turnover"])

    # Add to portfolio summary
    Portfolios_summary.append(Portfolios_TO)

    # Create an empty list for t-values
    t_values = []

    # Loop thorugh Portfolio returns
    for i in names + ["HL"]:
    
        # To obtain the Newey-West standard error correction fit an intercept-only model to the series
        reg = smf.ols(i+"~ 1",data=Portfolios_r_m).fit().get_robustcov_results(cov_type='HAC', maxlags=4)
    
        # Add t-values to summary statistic
        t_values.append(reg.tvalues.round(2))

    # Gather t-values in DataFrame
    t_values = pd.DataFrame(np.array(t_values).reshape(N+1,1),
                            columns=["t-stat"])
    
    # Add t-values for to portfolio summary
    Portfolios_summary.append(t_values)

    # Concat Portfolio summary
    Portfolios_summary = pd.concat(Portfolios_summary, axis=1).set_index([names + ["H-L"]])
    
    # Return Portfolios_summary and Portfolios_cumReturn
    return Portfolios_summary, Portfolios_cumReturn


# ### Evaluation Portfolios (value weighted)

def portfolios_vw(Eval_Data, N=10):
    ''' 
    Creates a summary statistic for decile portfolios.
    
    Parameters:
    Eval_Data - A pandas MultiIndex that constains predictions, returns, and market caps t-1 (Eval_Data).
    N - An integer specifying the number of desiered portfolios.
    
    Returns: 
    Portfolios_summary - A pandas DataFrame containing summary statistic.
    Portfolios_cumReturn - A pandas DataFrame containing cummulative returns.
    '''
    
    def information_ratio(Returns, Periods=12):
        '''
        Computes the Information ratio for the portfolio.

        Parameters:
        Returns - A pandas Series representing period returns.
        Periods - An integer specifying the period, e.g. Monthly (12), Daily (252), etc.
        
        Returns: 
        information_ratio - The information ratio of the portfolio.
        '''
        # Compute the information ratio
        information_ratio = np.sqrt(Periods) * (np.mean(Returns)) / np.std(Returns)
        
        return information_ratio
    
    
    def sortino_ratio(Returns, Periods=12):
        '''
        Computes the Sortino ratio for the portfolio, based on a 
        benchmark of zero (i.e. no risk-free rate information).

        Parameters:
        Returns - A pandas Series representing period returns.
        Periods - An integer specifying the period, e.g. Monthly (12), Daily (252), etc.
        
        Returns: 
        sortino_ratio - The sortino ratio of the portfolio.
        '''
        # Compute the Sortino ratio
        semi_var = Returns[Returns<0]**2
        semi_var = semi_var.sum()/len(Returns)
        sortino = np.sqrt(semi_var)
        sortino_ratio = np.sqrt(Periods) * Returns.mean() / sortino
        
        return sortino_ratio
    
    
    def drawdowns(Returns, Drawdown=False, Duration=False): 
        '''
        Calculates the largest peak-to-trough drawdown of the portfolio
        as well as the duration of the drawdown. 

        Parameters:
        Returns - A pandas Series representing period returns.
        Drawdown - A boolean to decide whether to return drawdown.
        Duration - A boolean to decide whether to return drawdown duration.

        Returns:
        drawdown - Highest peak-to-trough drawdown.
        drawdown duration - Duration of drawdown.
        '''

        # Calculate the running maximum
        cumsum = Returns.cumsum()

        # Ensure the value never drops below 1
        end = (np.maximum.accumulate(cumsum) - cumsum).idxmax() # end of the period
        low = cumsum[end]

        # Calculate the percentage drawdown
        start = (cumsum[:end]).idxmax() # start of period
        high = cumsum[start]

        # Calculate max drawdown and max drawdown duration
        maxDD = np.absolute(low-high).round(0)
        maxDD_Durarion = np.absolute(end-start).days

        # Plot max Drawdown 
        #plt.plot(cumsum)
        #plt.plot([end, start], [cumsum[end], cumsum[start]], 'o', color='Red', markersize=10)

        # Return Drawdown if True
        if Drawdown==True:
            return maxDD

        # Return Drawdown duration if True
        if Duration==True:
            return maxDD_Durarion
        
        
    
    # Define number of portfolios
    N = N
    
        # Define portfolio names    
    names_w = []   
    for i in range(1,N+1):
        name = "Portfolio_w_%.0f" % i
        names_w.append(name)

    # Sort portfolios by predictions
    Eval_Data = Eval_Data.sort_values('Prediction', ascending=False).sort_index(level='Index', sort_remaining=False)

    # Save index
    Index = Eval_Data.index.get_level_values(0).unique()

    # Start of Loop
    start_date = Eval_Data.index[0][0]

    # End of Loop
    end_date = Eval_Data.index[-1][0]

    # Create empty list for summary of portfolios  
    Portfolios_summary = []

    # Create empty list for monthly portfolio returns 
    Portfolios_r_m_w = []

    # Create empty list for monthly portfolio predictions     
    Portfolios_p_m_w = []

    # Create empty list for monthly turnovers
    Portfolios_Turnover_m = []

    # Keep track of time
    Timestamp = start_date

    # Add one month 
    TimeDelta = 0

    # Loop thorugh months
    while Timestamp < end_date: 

        # Keep track of time
        Timestamp = start_date + pd.offsets.BMonthEnd(TimeDelta)

        # Create empty list for monthly weighted predictions 
        wPrediction_monthly = []

        # Create empty list for monthly weighted returns 
        wReturn_monthly = []

        # Create empty list for monthly turnovers
        Turnover_monthly = []

        # Create empty lists for H-L weights
        Weights_HL = []
        Weights_new_HL = []

        # Loop over N portfolios
        for i in range(N):

            # Define start of slicing
            Start = int(Eval_Data.loc[Timestamp].shape[0]/N)*i

            # Define end of slicing
            End = int(Eval_Data.loc[Timestamp].shape[0]/N)*(i+1)

            # If it is not the last iteration select equal size of stocks
            if i != N-1:

                # Compute weights for portfolio i in month t
                Weights = Eval_Data.loc[Timestamp, "Mcap_BMI_USD t-1"].iloc[Start:End].div(
                              Eval_Data.loc[Timestamp, "Mcap_BMI_USD t-1"].iloc[Start:End].sum())

                # Append weights of H portfolio in month t
                if i == 0:
                    Weights_HL.append(Weights)

                # Compute weighted prediction portfolio i in month t
                wPrediction = Eval_Data.loc[Timestamp, ["Prediction"]].iloc[Start:End].multiply(
                                  Weights, axis=0).values.sum()

                # Compute weighted returns for portfolio i in month t
                wReturn = Eval_Data.loc[Timestamp, ["Return"]].iloc[Start:End].multiply(
                              Weights, axis=0).values.sum()

                # There is no turnover for the first month
                if Timestamp < end_date:

                    # Compute weights for portfolio i in month t+1
                    Weightsnew = Eval_Data.loc[Timestamp+pd.offsets.BMonthEnd(1), "Mcap_BMI_USD t-1"].iloc[Start:End].div(
                                     Eval_Data.loc[Timestamp+pd.offsets.BMonthEnd(1), "Mcap_BMI_USD t-1"].iloc[Start:End].sum())

                    # Append new weights of H portfolio in t+1
                    if i == 0:
                        Weights_new_HL.append(Weightsnew)

            # If it is the last iteration select remaining stocks
            else:

                # Compute weights for portfolio i in month t
                Weights = Eval_Data.loc[Timestamp, "Mcap_BMI_USD t-1"].iloc[Start:].div(
                          Eval_Data.loc[Timestamp, "Mcap_BMI_USD t-1"].iloc[Start:].sum())

                # Append weights of L portfolio in t
                Weights_HL.append(Weights)

                # Compute prediction for portfolio i in month t
                wPrediction = Eval_Data.loc[Timestamp, ["Prediction"]].iloc[Start:].multiply(Weights, axis=0).values.sum()

                # Compute returns for portfolio i in month t
                wReturn =  Eval_Data.loc[Timestamp, ["Return"]].iloc[Start:].multiply(Weights, axis=0).values.sum()

                # There is no turnover for the first month
                if Timestamp < end_date:

                    # Compute weights for month t+1
                    Weightsnew = Eval_Data.loc[Timestamp+pd.offsets.BMonthEnd(1), "Mcap_BMI_USD t-1"].iloc[Start:].div(
                                 Eval_Data.loc[Timestamp+pd.offsets.BMonthEnd(1), "Mcap_BMI_USD t-1"].iloc[Start:].sum())

                    # Append new weights of L portfolio in t+1
                    Weights_new_HL.append(Weightsnew)    

            if Timestamp < end_date:        
                # Compute Turnover for portfolio i in month t
                Turnover = Weightsnew.subtract(Weights, fill_value=0).abs().sum()

            # Add prediction for portfolio i in month t
            wPrediction_monthly.append(wPrediction)

            # Add return for portfolio i in month t
            wReturn_monthly.append(wReturn)

            # Add turnover for portfolio i in month t
            Turnover_monthly.append(Turnover)

        # Compute turnover for H-L portfolio in month t 
        if Timestamp < end_date:
            Turnover_HL = pd.concat(Weights_new_HL).subtract(pd.concat(Weights_HL), fill_value=0).abs().sum()/2

        # Append turnover for H-L portfolio in month t     
        Turnover_monthly.append(Turnover_HL)

        # Add one month 
        TimeDelta += 1   

        # Add monthly weighted predictions of portfolio i
        Portfolios_p_m_w.append(pd.DataFrame(wPrediction_monthly))

        # Add monthly weighted returns of portfolio i
        Portfolios_r_m_w.append(pd.DataFrame(wReturn_monthly))

        # Add monthly turnover of portfolio i
        Portfolios_Turnover_m.append(pd.DataFrame(Turnover_monthly))

    # Concat all portfolios 
    Portfolios_Turnover_m = pd.concat(Portfolios_Turnover_m, axis=1).T  

    # Drop last line since there is no turnover for the first month
    Portfolios_Turnover_m = Portfolios_Turnover_m[:-1]

    # Concat all portfolios 
    Portfolios_r_m_w = pd.concat(Portfolios_r_m_w, axis=1).T.set_index(Index).set_axis(
                                  [names_w], axis='columns', inplace=False) 

    # Compute returns for H-L portfolios
    Portfolios_r_m_w["w_HL"] = Portfolios_r_m_w.loc[:,"Portfolio_w_1"].values - Portfolios_r_m_w.loc[:,"Portfolio_w_10"].values

    # Concat all portfolios 
    Portfolios_p_m_w = pd.concat(Portfolios_p_m_w, axis=1).T.set_index(Index).set_axis(
                                  [names_w], axis='columns', inplace=False) 

    # Compute predictions for H-L portfolios
    Portfolios_p_m_w["w_HL"] = Portfolios_p_m_w.loc[:,"Portfolio_w_1"].values - Portfolios_p_m_w.loc[:,"Portfolio_w_10"].values


    # Compute cum Returns for all portfolios
    Portfolios_cumReturn = pd.DataFrame(Portfolios_r_m_w.apply(np.cumsum, axis=0))


    # Compute mean of predictions for all portfolios
    Portfolios_Pre = pd.DataFrame(Portfolios_p_m_w.apply(np.mean, axis=0).values.reshape(N+1,1), 
                                      columns=["Prediction_w"]).round(2)

    # Add to portfolio summary
    Portfolios_summary.append(Portfolios_Pre)


    # Compute mean of returns for all portfolios
    Portfolios_Return = pd.DataFrame(Portfolios_r_m_w.apply(np.mean, axis=0).values.reshape(N+1,1),
                                        columns=["w_Return"]).round(2)

    # Add means to portfolio summary
    Portfolios_summary.append(Portfolios_Return)

    # Compute standard deviations for all portfolios
    Portfolios_Std = pd.DataFrame(Portfolios_r_m_w.apply(np.std, axis=0).values.reshape(N+1,1),
                                     columns=["w_Std"]).round(2)

    # Add standard deviations to portfolio summary
    Portfolios_summary.append(Portfolios_Std)

    # Compute annualized Information Ratios for all portfolios
    Portfolios_IR = pd.DataFrame(Portfolios_r_m_w.apply(information_ratio, axis=0).values.reshape(N+1,1),
                                    columns=["w_IR"]).round(2)

    # Add annualized Information Ratios to portfolio summary
    Portfolios_summary.append(Portfolios_IR)

    # Compute sortino ratios for all portfolios
    Portfolios_sortino = pd.DataFrame(Portfolios_r_m_w.apply(sortino_ratio, axis=0).values.reshape(N+1,1),
                                     columns=["w_Sortino"]).round(2)

    # Add to portfolio summary
    Portfolios_summary.append(Portfolios_sortino)

    # Compute Max 1M Loss for all portfolios
    Portfolios_Max_Loss = pd.DataFrame(np.absolute(Portfolios_r_m_w.apply(np.min, axis=0).values.reshape(N+1,1)),
                                      columns=["w_Max_1M_Loss"]).round()

    # Add to portfolio summary
    Portfolios_summary.append(Portfolios_Max_Loss)

    # Compute maximum drawdown for all portfolios
    Portfolios_Max_draw= pd.DataFrame(Portfolios_r_m_w.apply(drawdowns, Drawdown=True, axis=0).values.reshape(N+1,1),
                                     columns=["w_Max_Draw"])

    # Add to portfolio summary
    Portfolios_summary.append(Portfolios_Max_draw)

    # Compute maximum drawdown duration for all portfolios
    Portfolios_Max_draw_dur= pd.DataFrame(Portfolios_r_m_w.apply(drawdowns, Duration=True, axis=0).values.reshape(N+1,1),
                                         columns=["w_Max_Draw_dur"])

    # Add to portfolio summary
    Portfolios_summary.append(Portfolios_Max_draw_dur)

    # Compute tunrover for all portfolios
    Portfolios_TO = pd.DataFrame(Portfolios_Turnover_m.apply(np.mean, axis=0).values.reshape(N+1,1).round(2),
                                columns=["Turnover"])

    # Add to portfolio summary
    Portfolios_summary.append(Portfolios_TO)

    # Create an empty list for t-values
    t_values = []

    # Loop thorugh Portfolio returns
    for i in names_w + ["w_HL"]:

    # To obtain the Newey-West standard error correction fit an intercept-only model to the series
        reg = smf.ols(i+"~ 1",data=Portfolios_r_m_w).fit().get_robustcov_results(cov_type='HAC', maxlags=4)

        # Add t-values to summary statistic
        t_values.append(reg.tvalues.round(2))

    # Gather t-values in DataFrame
    t_values = pd.DataFrame(np.array(t_values).reshape(N+1,1),
                            columns=["w_t-stat"])

    # Add t-values for to portfolio summary
    Portfolios_summary.append(t_values)

    # Concat Portfolio summary
    Portfolios_summary = pd.concat(Portfolios_summary, axis=1).set_index([names_w + ["H-L"]])

    
    # Return Portfolios_summary and Portfolios_cumReturn
    return Portfolios_summary, Portfolios_cumReturn


# ### Compare models

def compare(Models, Globals, Plot):
    ''' 
    Creates a summary statistic for specified models.
    
    Parameters:
    Models - A list of strings specifying the models.
    
    Returns: 
    Summary - A pandas DataFrame containing summary statistic for the specified models.
    '''
    
    def plot(Comp):
        ''' 
        Plots comparison.
  
        Parameters:
        ----------
        Comp - A pandas DataFrame containing the comparison.
        '''
        # Create empty list for plot elements
        data = []
        
        # Loop through the comparison
        for i in  Comp.columns:
            trace1 = go.Bar(x=list(Comp.index),
            y=Comp[i],
            name=i)
            
            # Append plot element
            data.append(trace1)

        # Create a bar plot   
        layout = go.Layout(barmode='group')
        fig = go.Figure(data=data, layout=layout)
        py.offline.iplot(fig, filename='grouped-bar')
    

    # Create an empty list for summary statistics
    Summary = []
    
    # Loop through models
    for i in Models:
    
        # Create Empty DataFrame for one specific model
        Model = pd.DataFrame()
 
        # Add statistic to the DataFrame
        Model=Model.append(pd.Series(Globals[i+"_Eval_Reg"].loc["Slope"].values), ignore_index=True)
        Model=Model.append(pd.Series(Globals[i+"_Eval_Reg"].loc["t-stat"].values), ignore_index=True)
        Model=Model.append(pd.Series(Globals[i+"_Eval_Port"].loc["H-L", "Return"]), ignore_index=True)
        Model=Model.append(pd.Series(Globals[i+"_Eval_Port"].loc["H-L", "t-stat"]), ignore_index=True)
        Model=Model.append(pd.Series(Globals[i+"_Eval_Port"].loc["H-L", "IR"]), ignore_index=True)
        Model=Model.append(pd.Series(Globals[i+"_Eval_Port"].loc["H-L", "Sortino"]), ignore_index=True)
        Model=Model.append(pd.Series(Globals[i+"_Eval_Port"].loc["H-L", "Turnover"]), ignore_index=True)

    
        # Add summary statistics of specific model to list
        Summary.append(Model)
    
    # Concat Summary
    Summary = pd.concat(Summary, axis=1)

    # Set column names of models
    Summary.columns = Models
    
    # Set index of models
    Summary.index = ["Pre Slope", "Pre Slope t-stat", "H-L Return", "H-L Return t-stat", "H-L IR", "H-L Sortino", "Turnover"]
    
    # If true plot the summary
    if Plot==True:
        plot(Comp=Summary)
    
    # Return summary
    return Summary


def compar_cum_return(Models, Globals, Plot):
    ''' 
    Creates a summary statistic for specified models.
    
    Parameters:
    Models - A list of strings specifying the models.
    
    Returns: 
    Summary - A pandas DataFrame containing summary statistic for the specified models.
    '''
    
    def plot(Comp):
        ''' 
        Plots comparison.
  
        Parameters:
        ----------
        Comp - A pandas DataFrame containing the comparison.
        '''
        # Create empty list for plot elements
        data = []
        
        # Loop through the comparison
        for i in  Comp.columns:
            trace1 = go.Scatter(x=list(Comp.index),
            y=Comp[i],
            name=i)
            
            # Append plot element
            data.append(trace1)

        # Create a bar plot   
        layout = go.Layout(barmode='group')
        fig = go.Figure(data=data, layout=layout)
        py.offline.iplot(fig, filename='grouped-bar')
    

    # Create an empty list for summary statistics
    Summary = []
    
    # Loop through models
    for i in Models:
    
        # Create Empty DataFrame for one specific model
        Model = pd.DataFrame()
 
        # Add statistic to the DataFrame
        Model=Model.append(pd.Series(Globals[i+"_Eval_Port_cRet"].iloc[:,-1].values), ignore_index=True)
    
        # Add summary statistics of specific model to list
        Summary.append(Model)
    
    # Concat Summary
    Summary = pd.concat(Summary, axis=0).T

    # Set column names of models
    Summary.columns = Models
    
    # Set index of models
    Summary.index = Globals[i+"_Eval_Port_cRet"].iloc[:,-1].index
    
    # If true plot the summary
    if Plot==True:
        plot(Comp=Summary)
    
    # Return summary
    return Summary


def compare_max_DD(Models, Globals, Plot):
    ''' 
    Creates a summary statistic for specified models.
    
    Parameters:
    Models - A list of strings specifying the models.
    
    Returns: 
    Summary - A pandas DataFrame containing summary statistic for the specified models.
    '''
    
    def plot(Comp):
        ''' 
        Plots comparison.
  
        Parameters:
        ----------
        Comp - A pandas DataFrame containing the comparison.
        '''
        # Create empty list for plot elements
        data = []
        
        # Loop through the comparison
        for i in  Comp.columns:
            trace1 = go.Bar(x=list(Comp.index),
            y=Comp[i],
            name=i)
            
            # Append plot element
            data.append(trace1)

        # Create a bar plot   
        layout = go.Layout(barmode='group')
        fig = go.Figure(data=data, layout=layout)
        py.offline.iplot(fig, filename='grouped-bar')
    

    # Create an empty list for summary statistics
    Summary = []
    
    # Loop through models
    for i in Models:
    
        # Create Empty DataFrame for one specific model
        Model = pd.DataFrame()
 
        # Add statistic to the DataFrame
        Model=Model.append(pd.Series(Globals[i+"_Eval_Port"].loc["H-L", "Max_Draw"]), ignore_index=True)

        # Add summary statistics of specific model to list
        Summary.append(Model)
    
    # Concat Summary
    Summary = pd.concat(Summary, axis=1)

    # Set column names of models
    Summary.columns = Models
    
    # Set index of models
    Summary.index = ["Max_Draw"]
    
    # If true plot the summary
    if Plot==True:
        plot(Comp=Summary)
    
    # Return summary
    return Summary


