#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# Import scipy.io and hdf5storage to load matlab files
import scipy.io as sio
import hdf5storage

# Import pandas
import pandas as pd

# Import datetime
import datetime as dt

# Import winsorize to handle outliers
from scipy.stats.mstats import winsorize

# Import numpy
import numpy as np

# Import StandardScaler and RobustScaler
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer


# # Create functions

# ### Flatten

def flatten(x):
    ''' 
    Creates a generator object that loops through a nested array.
  
    Parameters:
    ----------
    x - A nested array.

    Returns: 
    ----------
    j - A flattened array.
    '''
    
    # Check if the array is iterable
    try:
        narray = iter(x)
    # If it is not iterable return the array
    except TypeError:
        yield x
    # If it is iterable, loop through the array recursively
    else:
        for i in narray:
            for j in flatten(i):
                yield j


# ### Import data

def import_data(Features, Return, Size, Start, End):
    ''' 
    Imports data from Matlab and puts it in the needed format.
  
    Parameters:
    ----------
    Features - A list of strings specifying the feature variables. 
    Return - A string specifying the return variable.
    Size - A string specifying the size variable.
    Start - A string specifying the start date.
    End - A string specifying the end date.
    
    Returns:
    ----------
    features - A pandas MultiIndex containing the feature variables.
    actual_return - A pandas MultiIndex containing the return variable.
    response - A pandas MultiIndex containing the response variable.
    size - A pandas MultiIndex containing the size variable.
    '''

    # Import fixed data
    #FixedData = hdf5storage.loadmat("FixedData.mat")

    # Import IDs and flat nested array
    #ID = FixedData["FixedData"]["ID"]
    #ID = list(flatten(ID))
    
    # Import dates and set datetime
    Dates = sio.loadmat("Mdates.mat", squeeze_me=True)
    Dates = Dates["Mdates"]
    Dates = [dt.datetime.strptime(date, "%m/%d/%Y") for date in Dates]
    
    
    def import_variable(Variable, Start, End, Response=False, Shift=1, Dates=Dates):      
        ''' 
        Imports Variables from Matlab and puts them in the needed format.
  
        Parameters:
        ----------
        Variable - A string specifying the variable.
        Start - A string specifying the start date.
        End - A string specifying the end date.
        Response - A boolean to flag the response variable.
        Shift - An integer specifying the shift of the response variable. 
        Dates - A list containing the dates. 
        
        Returns: 
        ----------
        df - A pandas MultiIndex containing the specified variable.

        '''
        
        # Set datetime
        start_date = dt.datetime.strptime(Start, '%m/%d/%Y')
        end_date = dt.datetime.strptime(End, '%m/%d/%Y')
    
        # Load data
        df = hdf5storage.loadmat(Variable + ".mat")
        df = pd.DataFrame.from_dict(df["variable"])
    
        # Define index and columns
        df = df.set_index([0]).rename_axis('ID').set_axis(Dates, axis='columns', inplace=False)
    
        # Select specified time
        df = df.loc[:,start_date:end_date]
    
        # Check if response
        if Response==True:
        
            # Shift reponse forward by specified shift and then stack data 
            df = df.T.shift(-Shift).stack(dropna=False)
        
            # Name DataFrame
            df.name = Variable+" t+1"
        
        # If not response
        else:
            
            # Stack data 
            df = df.T.stack(dropna=False)
    
            # Name DataFrame
            df.name = Variable
    
        # Return the DataFrame
        return df    
    
    
    # Import feature variables and gather them in a DataFrame
    features = []
    for i in Features:
        feature = import_variable(Variable=i, Start=Start, End=End, Dates=Dates)
        features.append(feature)
    features = pd.concat(features, axis=1)

    # Import actual return variable and save it in a DataFrame
    actual_return = pd.DataFrame(import_variable(Variable=Return, Start=Start, End=End,  Dates=Dates))

    # Import response (actual return shifted) variable and save it in a DataFrame
    response = pd.DataFrame(import_variable(Variable=Return, Start=Start, End=End, Response=True, Shift=1,  Dates=Dates))

    # Import market size variable and save it in a DataFrame
    size = pd.DataFrame(import_variable(Variable=Size, Start=Start, End=End, Dates=Dates))
    
    # Return DataFrames
    return features, actual_return, response, size


# ### Gather data

def gather_data(Data, Features):
    ''' 
    Gathers specific variables and saves them in a MultiIndex.
  
    Parameters:
    ----------
    Data - A pandas MultiIndex containing the data.
    Features - A list of strings specifying the feature variables. 
          
    Returns: 
    ----------
    df - A pandas MultiIndex containing the feature, response, return, and size variables.
    '''
    
    # Create empty List for features
    features = []

    # Loop through data
    for i in Features:

        # Select features from data
        feature = Data[i]
    
        # Add feature to feature list
        features.append(feature)
    
    # Concat features
    features = pd.concat(features, axis=1)

    # Create DataFrame
    df = pd.concat([features, Data.drop(Features, axis=1)], axis=1)
    
    # Return the DataFrame
    return df


# ### Winsorize panel data

def winsorize_panel(Panel_Data, Features, Upper, Lower):  
    ''' 
    Winsorizes features in panal data.
  
    Parameters:
    ----------
    Panel_Data - A pandas MultiIndex containing the data. 
    Features - A list of strings specifying the feature variables. 
    Upper - An integer specifying the upper limit.
    Lower - An integer specifying the lower limit.
          
    Returns: 
    ----------
    Panel_Data - A pandas MultiIndex containing the winsorized data. 
    '''

    # Define upper limit
    upper = Upper

    # Define lower Limit 
    lower = Lower

    # Start of Loop
    start_date = Panel_Data.index[0][0]

    # End of Loop
    end_date = Panel_Data.index[-1][0]

    # Loop through all features 
    for i in Features:
    
        # Keep track of time
        Timestamp = start_date

        # Add one month 
        TimeDelta = 0
    
        # Loop thorugh months
        while Timestamp < end_date: 
    
            # Keep track of time
            Timestamp = start_date + pd.offsets.BMonthEnd(TimeDelta)
    
            # Winsorize features for month i
            Panel_Data.loc[Timestamp, i] = winsorize(Panel_Data.loc[Timestamp, i], limits=[lower, upper])
    
            # Add one month
            TimeDelta += 1 
            
    # Return winsorized DataFrame
    return Panel_Data 


# ### Statistics for panel data

def stats_panel(Panel_Data, Variables, Rounding=2):   
    ''' 
    Descriptive statistics for panel data.
  
    Parameters:
    ----------
    Panel_Data - A pandas MultiIndex containing the data. 
    Variables - A list of strings specifying the variables. 
          
    Returns: 
    ----------
    df - A pandas DataFrame containing summary statistics.
    '''
    
    # Start Loop
    start_date = Panel_Data.index[0][0]

    # End Loop
    end_date = Panel_Data.index[-1][0]
    
    # Create empty list for averages 
    Avg = []

    # Create empty list for standard deviations 
    Std = []

    # Create empty list for sample sizes 
    N = []
    
    # Loop through variables
    for i in Variables:
    
        # Keep track of time
        Timestamp = start_date

        # Add one month 
        TimeDelta = 0

        # Create empty list for monthly averages for feature i
        Average =  []
    
        # Create empty list for monthly standard deviations for feature i
        Deviation =  []
    
        # Create empty list for monthly sample sizes for feature i
        Sample =  []
    
        # Loop thorugh months
        while Timestamp < end_date: 
    
            # Keep track of time
            Timestamp = start_date + pd.offsets.BMonthEnd(TimeDelta)
    
            # average for month i
            average = Panel_Data.loc[Timestamp, i].mean()
        
            # standard deviation for month i
            std = Panel_Data.loc[Timestamp, i].std()
        
            # sample size for month i
            sample = Panel_Data.loc[Timestamp, i].count()
    
            # Add average of month i to list
            Average.append(average)
        
            # Add standard deviation of month i to list
            Deviation.append(std)
        
            # Add sample size of month i to list
            Sample.append(sample)
        
            # Add one month
            TimeDelta += 1 
        
        # Append monthly averages for variable i to list
        Avg.append(pd.DataFrame(Average))
    
        # Append monthly standard deviations for variable i to list
        Std.append(pd.DataFrame(Deviation))
    
        # Append monthly sample sizes for variable i to list
        N.append(pd.DataFrame(Sample))
    
    # Concat averages of all variables   
    Avg = pd.concat(Avg, axis=1)

    # Concat standard deviations of all variables    
    Std = pd.concat(Std, axis=1)

    # Concat sample sizes of all variables 
    N = pd.concat(N, axis=1)

    # Compute time-series averages of cross-sectional averages
    Avg = Avg.apply(np.mean, axis=0).round(Rounding)

    # Compute time-series averages of cross-sectional standard deviations
    Std = Std.apply(np.mean, axis=0).round(Rounding)

    # Compute time-series averages of cross-sectional sample sizes
    N = N.apply(np.mean, axis=0).astype(int)

    # Name DataFrams
    Avg.name = "Avg"
    Std.name = "Std"
    N.name = "N"

    # Concat statistics
    df = pd.concat([Avg, Std, N], axis=1)

    # Set index
    df.index = Variables

    return df


# ### Standard Scaler

def scaler(Data, Variable):        
        ''' 
        Scales variables in panel data cross-secionally.
  
        Parameters:
        ----------
        Data - A pandas MultiIndex containing the feature, response, return, and size variables.
        Variable - A list of strings specifying the variables. 
        
        Returns: 
        ----------
        A pandas MultiIndex containing the demeaned variables.
        '''

        # Specify scaler 
        scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        #StandardScaler(copy=True, with_mean=False, with_std=True)
        #MinMaxScaler(feature_range=(0, 1), copy=True)
        
        # Copy panel data
        Data = Data.copy()

        # Start of Loop
        start_date = Data.index[0][0]

        # End of Loop
        end_date = Data.index[-1][0]

        # Loop through all variables 
        for i in Variable:
    
            # Keep track of time
            Timestamp = start_date

            # Add one month 
            TimeDelta = 0
    
            # Loop thorugh months
            while Timestamp < end_date: 
    
                # Keep track of time
                Timestamp = start_date + pd.offsets.BMonthEnd(TimeDelta)
    
                # Demean feature for month i
                Data.loc[Timestamp, i] = scaler.fit_transform(Data.loc[Timestamp, i].values.reshape(-1, 1))
            
                # Add one month
                TimeDelta += 1 
            
        # Return demeaned data 
        return Data 


# ### Demean Data

def demean(Data, Variable):        
     ''' 
     Demeans variables (dependet and independent) in panel data cross-secionally.
  
     Parameters:
     ----------
     Data - A pandas MultiIndex containing the feature, response, return, and size variables.
     Variable - A list of strings specifying the variables. 
     
     Returns: 
     ----------
     A pandas MultiIndex containing the demeaned variables.
     '''
 
     # Copy panel data
     Data = Data.copy()

     # Specify scaler 
     scaler = StandardScaler(copy=True, with_mean=True, with_std=False)

     # Start of Loop
     start_date = Data.index[0][0]

     # End of Loop
     end_date = Data.index[-1][0]

     # Loop through all variables 
     for i in Variable:
 
         # Keep track of time
         Timestamp = start_date

         # Add one month 
         TimeDelta = 0
 
         # Loop thorugh months
         while Timestamp < end_date: 
 
             # Keep track of time
             Timestamp = start_date + pd.offsets.BMonthEnd(TimeDelta)
 
             # Demean feature for month i
             Data.loc[Timestamp, i] = scaler.fit_transform(Data.loc[Timestamp, i].values.reshape(-1, 1))
         
             # Add one month
             TimeDelta += 1 
         
     # Return demeaned data 
     return Data 


# ### Scale to unit variance

def unit_variance(Data, Variable):        
        ''' 
        Scales variables in panel data cross-secionally.
  
        Parameters:
        ----------
        Data - A pandas MultiIndex containing the feature, response, return, and size variables.
        Variable - A list of strings specifying the variables. 
        
        Returns: 
        ----------
        Data - A pandas MultiIndex containing the demeaned variables.
        '''

        # Specify scaler 
        scaler = StandardScaler(copy=True, with_mean=False, with_std=True)
        #StandardScaler(copy=True, with_mean=False, with_std=True)
        #MinMaxScaler(feature_range=(0, 1), copy=True)
        
        # Copy panel data
        Data = Data.copy()

        # Start of Loop
        start_date = Data.index[0][0]

        # End of Loop
        end_date = Data.index[-1][0]

        # Loop through all variables 
        for i in Variable:
    
            # Keep track of time
            Timestamp = start_date

            # Add one month 
            TimeDelta = 0
    
            # Loop thorugh months
            while Timestamp < end_date: 
    
                # Keep track of time
                Timestamp = start_date + pd.offsets.BMonthEnd(TimeDelta)
    
                # Demean feature for month i
                Data.loc[Timestamp, i] = scaler.fit_transform(Data.loc[Timestamp, i].values.reshape(-1, 1))
            
                # Add one month
                TimeDelta += 1 
            
        # Return demeaned data 
        return Data 


# ### Robust Scaler

def robust_scaler(Data, Variable):        
        ''' 
        Scales variables in panel data cross-secionally.
  
        Parameters:
        ----------
        Data - A pandas MultiIndex containing the feature, response, return, and size variables.
        Variable - A list of strings specifying the variables. 
        
        Returns: 
        ----------
        A pandas MultiIndex containing the demeaned variables.
        '''

        # Specify scaler 
        scaler = RobustScaler(copy=True, quantile_range=(25.0, 75.0), with_centering=True, with_scaling=True)
    
        # Copy panel data
        Data = Data.copy()

        # Start of Loop
        start_date = Data.index[0][0]

        # End of Loop
        end_date = Data.index[-1][0]

        # Loop through all variables 
        for i in Variable:
    
            # Keep track of time
            Timestamp = start_date

            # Add one month 
            TimeDelta = 0
    
            # Loop thorugh months
            while Timestamp < end_date: 
    
                # Keep track of time
                Timestamp = start_date + pd.offsets.BMonthEnd(TimeDelta)
    
                # Demean feature for month i
                Data.loc[Timestamp, i] = scaler.fit_transform(Data.loc[Timestamp, i].values.reshape(-1, 1))
            
                # Add one month
                TimeDelta += 1 
            
        # Return demeaned data 
        return Data 


# ### robust_variance

def robust_variance(Data, Variable):        
        ''' 
        Scales variables in panel data cross-secionally.
  
        Parameters:
        ----------
        Data - A pandas MultiIndex containing the feature, response, return, and size variables.
        Variable - A list of strings specifying the variables. 
        
        Returns: 
        ----------
        A pandas MultiIndex containing the demeaned variables.
        '''

        # Specify scaler 
        scaler = RobustScaler(copy=True, quantile_range=(25.0, 75.0), with_centering=False, with_scaling=True)
    
        # Copy panel data
        Data = Data.copy()

        # Start of Loop
        start_date = Data.index[0][0]

        # End of Loop
        end_date = Data.index[-1][0]

        # Loop through all variables 
        for i in Variable:
    
            # Keep track of time
            Timestamp = start_date

            # Add one month 
            TimeDelta = 0
    
            # Loop thorugh months
            while Timestamp < end_date: 
    
                # Keep track of time
                Timestamp = start_date + pd.offsets.BMonthEnd(TimeDelta)
    
                # Demean feature for month i
                Data.loc[Timestamp, i] = scaler.fit_transform(Data.loc[Timestamp, i].values.reshape(-1, 1))
            
                # Add one month
                TimeDelta += 1 
            
        # Return demeaned data 
        return Data 


# ### QuantileTransformer

def quantile_transformer(Data, Variable):        
        ''' 
        Scales variables in panel data cross-secionally.
  
        Parameters:
        ----------
        Data - A pandas MultiIndex containing the feature, response, return, and size variables.
        Variable - A list of strings specifying the variables. 
        
        Returns: 
        ----------
        A pandas MultiIndex containing the demeaned variables.
        '''

        # Specify scaler 
        scaler = QuantileTransformer(copy=True)
    
        # Copy panel data
        Data = Data.copy()

        # Start of Loop
        start_date = Data.index[0][0]

        # End of Loop
        end_date = Data.index[-1][0]

        # Loop through all variables 
        for i in Variable:
    
            # Keep track of time
            Timestamp = start_date

            # Add one month 
            TimeDelta = 0
    
            # Loop thorugh months
            while Timestamp < end_date: 
    
                # Keep track of time
                Timestamp = start_date + pd.offsets.BMonthEnd(TimeDelta)
    
                # Demean feature for month i
                Data.loc[Timestamp, i] = scaler.fit_transform(Data.loc[Timestamp, i].values.reshape(-1, 1))
            
                # Add one month
                TimeDelta += 1 
            
        # Return demeaned data 
        return Data 


