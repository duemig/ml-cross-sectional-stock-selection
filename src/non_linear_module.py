#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# Import pandas
import pandas as pd

# Import numpy
import numpy as np

# Import patsy to transform data
from patsy import dmatrix

# Import grouplasso 
from grouplasso import GroupLassoRegressor

# Import libraries from sklearn
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Dropout, Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.regularizers import l1, l2, l1_l2

# Import Random Forest and Boosted Regression 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Running time
import time


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


# ### Spline (Group-Lasso) Regression

def glm(Data, 
        Features, 
        Response, 
        Spline, 
        DF, 
        Method, 
        Years, 
        Refit, 
        Validation,
        Val_size,
        N_splits,
        param_grid, 
        Stats=True, 
        Verbose=False,
        Rounding=2):
    
    ''' 
    Performs spline (Group-Lasso) regressions and computes statistics.
  
    Parameters:
    ---------- 
    Data - A pandas MultiIndex containing the feature, response, return, and size variables.
    Features - A list of strings specifying the feature variables. 
    Response - A string specifying the response variable.
    Spline - A string specifying the spline.
    DF - An integer specifying the degree of freedom. 
    Method - A string specifying the method (rolling, or expanding).
    Years - An integer specifying the years of rolling window/min years of expanding method.
    Refit - An integer specifying the months after which the model is refitted.
    Validation - A string specifying the validation method.
    Val_size - A float specifying the validation size
    N_splits - An integer specifying the number of splits used for validation.
    param_grid - A list specifying the parameters to validate
    Stats - A boolean to decide whether to return statistics.
    Verbose -  A boolean to decide whether to print status.
    Rounding - An integer specifying the rounding.

    Returns: 
    ---------- 
    Pre -  A pandas MultiIndex containing the predictions.
    Summary - A pandas DataFrame containing coefs, alphas, and Ns.
    Pre_stats - A pandas DataFrame containing summary statistics.
    Years - An integer specifying the years of rolling window/min years of expanding method.
    '''
    
    # Set timer 
    tic = time.time()

    # If Spline = B-spline (bs)
    if Spline == "bs":
        
        # Create empty list for transformed features
        Data_trans = []
        
        # Loop through features
        for i in Features:
            
            # Transform feature i
            transformed_x = dmatrix("bs(Data[i], df=DF, degree=3, include_intercept=False)-1",
                                {"Data[i]": Data[i]}, return_type='dataframe')
        
            # Append transformed feature to list 
            Data_trans.append(transformed_x)
        
        # Concat transformed features
        Data_trans = pd.concat(Data_trans, axis=1)
        
        # Print status 
        if Verbose == True:
            print("Transformation completed")
         
    # If Spline = cubic regression spline (cr)
    if Spline == "cr":
        
        # Create empty list for transformed features
        Data_trans = []
        
        # Loop through features
        for i in Features:
            
            # Transform feature i
            transformed_x = dmatrix("cr(Data[i], df=DF, constraints='center')-1",
                                {"Data[i]": Data[i]}, return_type='dataframe')
        
            # Append transformed feature to list 
            Data_trans.append(transformed_x)
        
        # Concat transformed features
        Data_trans = pd.concat(Data_trans, axis=1)
            
        # Print status 
        if Verbose == True:
            print("Transformation completed")
    
    # Set up Feature groups 
    Group_ids = np.repeat(np.arange(0, len(Features), 1), DF)
    
    # Set up GroupLasso regressor 
    Regressor = GroupLassoRegressor(group_ids=Group_ids, 
                                    random_state=45, 
                                    verbose=False,  
                                    max_iter=10000, # Default = 1000
                                    eta=0.05,        # Default = 0.1
                                    tol=0.01)       # Default = 0.001
    
    # Walk-forward validation
    if Validation == "wf":
        tvs = TrainValSplit(n_splits=N_splits, val_size=Val_size) 
        
    # Blocked walk-forward validation
    elif Validation == "bwf":
        tvs = BlockingTrainValSplit(n_splits=N_splits, val_size=Val_size) 

    # Set up girdsearch
    gs = GridSearchCV(Regressor, 
                      param_grid=param_grid, 
                      cv=tvs, 
                      verbose=0, 
                      n_jobs=-1, 
                      scoring="neg_mean_squared_error", 
                      refit=True, 
                      return_train_score=False,
                      iid=False)
    
    # Create empty list for predictions
    Pre = []
    
    # Create empty list for coefficients
    Coef = []
    
    # Create empty list for number of selected features
    N = []
    
    # Create empty list for alphas
    Par = []
    
    # Start of slice
    Start = Data.index[0][0] + pd.offsets.BMonthEnd(Years*12)
    
    # End of slice
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
    
    # Set up counter for status 
    Counter = 1
    
    # Loop through months
    while Timestamp < end_date: 

        # Keep track of time
        Timestamp = start_date + pd.offsets.BMonthEnd(TimeDelta)
    
        # Check whether another regression is possible
        if Timestamp + pd.offsets.BMonthEnd(Years*12-1) < end_date:
        
            # Rolling method
            if Method == "rolling":
        
                # Select feature variables for training data
                X_train = Data_trans.loc[Timestamp:Timestamp+pd.offsets.BMonthEnd(Years*12-1)]
        
                # Select response variable for training data
                y_train = Data.loc[Timestamp:Timestamp+pd.offsets.BMonthEnd(Years*12-1), 
                                   Response]
                   
            # Expanding method
            elif Method == "expanding":       
                    
                # Select feature variables for training data
                X_train = Data_trans.loc[start_date:Timestamp+pd.offsets.BMonthEnd(Years*12-1)]
        
                # Select response variable for training data
                y_train = Data.loc[start_date:Timestamp+pd.offsets.BMonthEnd(Years*12-1), 
                                   Response]
                    
            # Perform cross validation
            gs.fit(X_train, np.ravel(y_train))
            
            # Add parameters to list
            Par.append(gs.best_params_)
        
            # Add coefficient to list
            Coef.append(pd.DataFrame(gs.best_estimator_.coef_.reshape(1, len(Features)*DF), 
                        columns=list(np.repeat(Features, DF))))
            
            # Add number of selected features to list 
            N.append(np.count_nonzero(gs.best_estimator_.coef_)/DF)
            
        # Check whether test over full period is possible
        if Timestamp+pd.offsets.BMonthEnd(Years*12+Refit-1) <= End:
        
            # Select features for test data
            X_test = Data_trans.loc[Timestamp+pd.offsets.BMonthEnd(Years*12):Timestamp+pd.offsets.BMonthEnd(Years*12+Refit-1)]
            
            # Perform prediction and add it to the list
            Pre.append(pd.DataFrame(gs.predict(X_test), columns=["Prediction"]))
            
            # Print status 
            if Verbose == True:
                print(Counter,"." " Iteration completed", sep='')  

        # Test over remaining period
        elif Timestamp+pd.offsets.BMonthEnd(Years*12) < End: 
            
            # Select features for test data
            X_test = Data_trans.loc[Timestamp+pd.offsets.BMonthEnd(Years*12):,]
            
            # Perform prediction and add it to the list
            Pre.append(pd.DataFrame(gs.predict(X_test), columns=["Prediction"]))
            
            # Print status 
            if Verbose == True:
                print(Counter,"." " Iteration completed", sep='')  
        
        # Add one month 
        TimeDelta += Refit

        # Add 1 iteration to counter 
        Counter += 1

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
           
        # Set timer
        toc = time.time()
        
        # Print running time
        print("Running time = %.2f hours" % ((toc-tic)/3600))
        
        # Return predictions and statistics
        return Pre, Summary, Pre_stats, Years
    
    else:
        
        # Set timer
        toc = time.time()
        
        # Print running time
        print("Running time = %.2f hours" % ((toc-tic)/3600))
        
        # Return predictions 
        return Pre, Summary, Years


# ### Neural Network

def neural_net(Data,
               Features,
               Response,
               Method,
               Years,
               Refit,
               Validation,
               Val_size,
               N_splits,
               Name,
               Hidden_layers, 
               Optimizer,
               Batch_size,
               Epochs,
               ES,
               Patience,
               Param_grid,
               Stats,
               Verbose,
               Penalty=None, 
               Rounding=2):
    
    """
    Runs neural networks. 
    
    Parameters
    ----------    
    Data - A pandas MultiIndex containing the feature, response, return, and size variables.
    Features - A list of strings specifying the feature variables. 
    Response - A string specifying the response variable.
    Method - A string specifying the method (rolling, or expanding).
    Years - An integer specifying the years of rolling window/min years of expanding method.
    Refit - An integer specifying the months after which the model is refitted.
    Validation - A string specifying the validation method.
    Val_size - A float specifying the validation size.
    N_splits - An integer specifying the number of splits used for validation.
    Name - A string specifying the name of the neural network.
    Hidden_layers - A list specifying the number of hidden layers, and the size of each hidden layer.
    Optimizer - A string specifying the optimizer to train the network.
    Batch_size - An integer specifying the batch size.
    Epochs - An integer specifying the epochs.
    ES - A boolean to decide whether to apply early stopping.
    Patience - An integer specifying the patience of early stopping.
    Param_grid - A list specifying the parameters to validate.
    Stats - A boolean to decide whether to return statistics.
    Verbose -  A boolean to decide whether to print status.
    Penalty - A string specifying the penalty.
    Rounding - An integer specifying the rounding.

    Returns: 
    -------
    Pre -  A pandas MultiIndex containing the predictions.
    Pre_stats - A pandas DataFrame containing summary statistics.
    Years - An integer specifying the years of rolling window/min years of expanding method.
    """
    
    # import plot_model to plot the architecture of the neural network 
    from keras.utils import plot_model

    
    def model(Penalty=Penalty, 
              Hidden_layers=Hidden_layers, 
              dropout_rate=0, 
              Optimizer=Optimizer,
              loss='mean_squared_error',
              N_input=len(Features),
              l2_penalty=0,
              l1_penalty=0):
        
        """
        Creates a Keras Multi-layer neural network. 
    
        Parameters
        ----------    
        Penalty - A string specifying the penalty.
        Hidden_layers - A list specifying the number of hidden layers, and the size of each hidden layer.
        Dropout_rate - A float 0 ~ 1 (if bigger than 0, there will be a dropout layer).
        Optimizer - A string specifying the optimizer to train the network.
        loss -  A string specifying the loss function.
        N_input - An integer specifying the number of features. 
        l2_penalty - A float (so called l2 regularization).
        l1_penalty - A float (so called l1 regularization).

        Returns
        -------
        model - A keras model
        """   
    
        # Set up keras model
        model = Sequential()   
    
        if Penalty == "l1":
    
            for index, layers in enumerate(Hidden_layers):  
        
                if not index:
                    # specify the input_dim to be the number of features for the first layer
                    model.add(Dense(layers, 
                    input_dim = N_input, 
                    activation='relu', 
                    kernel_regularizer = l1(l1_penalty)))
            
                else:
                    # Add hidden layer 
                    model.add(Dense(layers, 
                    activation='relu', 
                    kernel_regularizer = l1(l1_penalty)))
        
                if dropout_rate:
                    # Add dropout
                    model.add(Dropout(rate = dropout_rate))   
                    
            # Add output layer
            model.add(Dense(1, activation='linear'))  
            
    
        elif Penalty == "l2":
    
            for index, layers in enumerate(Hidden_layers):  
        
                if not index:
                    # specify the input_dim to be the number of features for the first layer
                    model.add(Dense(layers, 
                    input_dim = N_input, 
                    activation='relu', 
                    kernel_regularizer = l2(l2_penalty)))
            
                else:
                    # Add hidden layer
                    model.add(Dense(layers, 
                    activation='relu', 
                    kernel_regularizer = l2(l2_penalty)))
                    
                if dropout_rate:
                    # Add dropout
                    model.add(Dropout(rate = dropout_rate))
                    
            # Add output layer
            model.add(Dense(1, activation='linear'))   
            
    
        elif Penalty == "l1_l2":
    
            for index, layers in enumerate(Hidden_layers):  
        
                if not index:
                    # specify the input_dim to be the number of features for the first layer
                    model.add(Dense(layers, 
                    input_dim = N_input, 
                    activation='relu', 
                    kernel_regularizer = l1_l2(l1=l1_penalty, l2=l2_penalty)))
            
                else:
                    # Add hidden layer
                    model.add(Dense(layers, 
                    activation='relu', 
                    kernel_regularizer = l1_l2(l1=l1_penalty, l2=l2_penalty)))
        
                if dropout_rate:
                    # Add dropout
                    model.add(Dropout(rate = dropout_rate))
                    
            # Add output layer
            model.add(Dense(1, activation='linear'))   
                    
       
        # If none of l1, l2 or l1_l2 
        else:
            
            for index, layers in enumerate(Hidden_layers):  
        
                if not index:
                    # specify the input_dim to be the number of features for the first layer
                    model.add(Dense(layers, 
                    input_dim = N_input, 
                    activation='relu'))
            
                else:
                    # Add hidden layer
                    model.add(Dense(layers, 
                    activation='relu'))
        
                if dropout_rate:
                    # Add dropout
                    model.add(Dropout(rate = dropout_rate))                        

            # Add output layer
            model.add(Dense(1, activation='linear')) 
            
        # Compile model
        model.compile(loss=loss, optimizer=Optimizer, metrics=[loss])
        
        # save model architecture 
        plot_model(model, show_shapes=True, 
                   show_layer_names=True, 
                   to_file=Name+".png", 
                   rankdir="TB")
    
        # Return model 
        return model
    
    
    # Set timer 
    tic = time.time()
    
    
    # Create regressor
    Regressor = KerasRegressor(
                build_fn=model,
                epochs=Epochs,
                batch_size=Batch_size, 
                verbose=0)
    
    # Walk-forward validation
    if Validation == "wf":
        tvs = TrainValSplit(n_splits=N_splits, val_size=Val_size) 
        
    # Blocked walk-forward validation
    elif Validation == "bwf":
        tvs = BlockingTrainValSplit(n_splits=N_splits, val_size=Val_size) 

    # Set up girdsearch
    gs = GridSearchCV(Regressor, 
                      param_grid=Param_grid,
                      cv=tvs, 
                      verbose=0, 
                      n_jobs=-1, 
                      scoring="neg_mean_squared_error", 
                      refit=True, 
                      return_train_score=False,
                      iid=False)
    
    # Create empty list for predictions
    Pre = []
    
    # Create empty list for parameters 
    Par = []

    # Start of slice
    Start = Data.index[0][0] + pd.offsets.BMonthEnd(Years*12)

    # End of slice
    End =  Data.index[-1][0]
    
    # Select index for summary
    Par_Ind = Data[Start:End].index.get_level_values(0).unique()[::Refit]

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

    # Set up counter for status 
    Counter = 1
    
    # Loop through months
    while Timestamp < end_date: 

        # Keep track of time
        Timestamp = start_date + pd.offsets.BMonthEnd(TimeDelta)
    
        # Check whether another regression is possible
        if Timestamp + pd.offsets.BMonthEnd(Years*12-1) < end_date:
            
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
                X_train = Data.loc[start_date:Timestamp+pd.offsets.BMonthEnd(Years*12-1), 
                                   Features]
        
                # Select response variable for training data
                y_train = Data.loc[start_date:Timestamp+pd.offsets.BMonthEnd(Years*12-1), 
                                   Response]   
                 
            # If early stopping = True
            if ES==True:
                
                # Set up early stopping
                early_stopping = EarlyStopping(
                monitor="mean_squared_error", 
                patience=Patience,
                mode="auto",
                verbose=1)
                
                # Perform cross validation
                gs.fit(X_train,  
                y_train, 
                callbacks=[early_stopping]) 
             
            # If early stopping = False
            elif ES==False:
                
                # Perform cross validation
                gs.fit(X_train, 
                y_train)  
                
            # Add parameters to list     
            Par.append(gs.best_params_)
           
        #print("train length", len(np.unique(y_train.index.get_level_values(0))))
        #print("train start", y_train.index[1])
        #print("train end", y_train.index[-1])
                    
        #print("val length", len(np.unique(y_val.index.get_level_values(0))))
        #print("val start", y_val.index[1])
        #print("val end", y_val.index[-1])
        
          
        # Check whether test over full period is possible
        if Timestamp+pd.offsets.BMonthEnd(Years*12+Refit-1) <= End:
        
            # Select features for test data
            X_test = Data.loc[Timestamp+pd.offsets.BMonthEnd(Years*12):Timestamp+pd.offsets.BMonthEnd(Years*12+Refit-1), 
                              Features]
        
            # Perform prediction and add it to the list
            Pre.append(pd.DataFrame(gs.predict(X_test), columns=["Prediction"]))
            
            # Print status 
            if Verbose == True:
                print(Counter,"." " Iteration completed", sep='')  

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
            
            # Print status 
            if Verbose == True:
                print(Counter,"." " Iteration completed", sep='')  
        
            #print("test length", len(np.unique(X_test.index.get_level_values(0))))
            #print("test start", X_test.index[1])
            #print("test end", X_test.index[-1])
            
        # Add one month 
        TimeDelta += Refit

        # Add 1 iteration to counter 
        Counter += 1

    # Concat predictions
    Pre = pd.concat(Pre, axis=0).set_index([Pre_Ind, Pre_ID])
    
    # Create DataFrame for parameters
    Par = pd.DataFrame(Par).set_index(Par_Ind)

    
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
        
        # Set timer
        toc = time.time()
        
        # Print running time
        print("Running time = %.2f hours" % ((toc-tic)/3600))   
        
        # Return predictions and statistics
        return Pre, Pre_stats, Par, Years
    
    else:
        
        # Set timer
        toc = time.time()
        
        # Print running time
        print("Running time = %.2f hours" % ((toc-tic)/3600))
        
        # Return predictions 
        return Pre, Years


# ### Random Forest 

def rf(Data, 
       Features, 
       Response, 
       Method, 
       Years, 
       Refit, 
       Validation,
       Val_size,
       N_splits,
       bootstrap,
       param_grid, 
       Stats,
       Verbose,
       Rounding=2):

    ''' 
    Runs Random Forests and computes statistics.
  
    Parameters:
    ----------
    Data - A pandas MultiIndex containing the feature, response, return, and size variables.
    Features - A list of strings specifying the feature variables. 
    Response - A string specifying the response variable.
    Method - A string specifying the method (rolling, or expanding).
    Years - An integer specifying the years of rolling window/min years of expanding method.
    Refit - An integer specifying the months after which the model is refitted.
    Validation - A string specifying the validation method.
    Val_size - A float specifying the validation size.
    N_splits - An integer specifying the number of splits used for validation.
    bootstrap - A boolean to decide whether bootstrap samples are used when building trees.
    param_grid - A list specifying the parameters to validate.
    Stats - A boolean to decide whether to return statistics.
    Verbose - A boolean to decide whether to print status.
    Rounding - An integer specifying the rounding.
    
    Returns: 
    ----------
    Pre -  A pandas MultiIndex containing the predictions.
    Pre_stats - A pandas DataFrame containing summary statistics.
    Fi - A pandas DataFrame containing feature importance. 
    Fi_avg - A pandas DataFrame containing average feature importance. 
    Years - An integer specifying the years of rolling window/min years of expanding method.
    '''
    
    # Set timer 
    tic = time.time()

    # Set up Regressor
    Regressor = RandomForestRegressor(random_state=1, 
                                      criterion="mse", 
                                      bootstrap=bootstrap,
                                      verbose=0,
                                      n_jobs=-1)
            
    # Walk-forward validation
    if Validation == "wf":
        tvs = TrainValSplit(n_splits=N_splits, val_size=Val_size) 
        
    # Blocked walk-forward validation
    elif Validation == "bwf":
        tvs = BlockingTrainValSplit(n_splits=N_splits, val_size=Val_size) 
    
    # Set up girdsearch
    gs = GridSearchCV(Regressor, 
                      param_grid=param_grid, 
                      cv=tvs, 
                      verbose=0, 
                      n_jobs=-1, 
                      scoring="neg_mean_squared_error", 
                      refit=True, 
                      return_train_score=False,
                      iid=False)
    
    # Create empty list for predictions
    Pre = []

    # Create empty list for feature importance
    Fi = []

    # Start of slice
    Start = Data.index[0][0] + pd.offsets.BMonthEnd(Years*12)

    # End of slice
    End =  Data.index[-1][0]
    
    # Select index for features importance
    Fi_Ind = Data[Start:End].index.get_level_values(0).unique()[::Refit]
        
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
    
    # Set up counter for status 
    Counter = 1
        
    # Loop through months
    while Timestamp < end_date: 

        # Keep track of time
        Timestamp = start_date + pd.offsets.BMonthEnd(TimeDelta)
    
        # Check whether another regression is possible
        if Timestamp + pd.offsets.BMonthEnd(Years*12-1) < end_date:
        
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
                X_train = Data.loc[start_date:Timestamp+pd.offsets.BMonthEnd(Years*12-1), 
                                   Features]
                  
                # Select response variable for training data
                y_train = Data.loc[start_date:Timestamp+pd.offsets.BMonthEnd(Years*12-1), 
                                   Response]
    
            # Perform cross validation
            gs.fit(X_train, np.ravel(y_train))
            
            # Add feature importance to list
            Fi.append(pd.DataFrame((gs.best_estimator_.feature_importances_*100).reshape(1, len(Features)), 
                                    columns=Features))
                      
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
            
            # Print status 
            if Verbose == True:
                print(Counter,"." " Iteration completed", sep='')  
                
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
                  
            # Print status 
            if Verbose == True:
                print(Counter,"." " Iteration completed", sep='')  
                
            #print("test length", len(np.unique(X_test.index.get_level_values(0))))
            #print("test start", X_test.index[1])
            #print("test end", X_test.index[-1])

        # Add one month 
        TimeDelta += Refit
        
        # Add 1 iteration to counter 
        Counter += 1
    
    # Concat feature importance
    Fi = pd.concat(Fi, axis=0).set_index(Fi_Ind)
    
    # Compute average feature importance
    Fi_avg = Fi.apply(np.mean, axis=0).sort_values(axis=0, ascending=True)
    
    # Concat predictions
    Pre = pd.concat(Pre, axis=0).set_index([Pre_Ind, Pre_ID])

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
        
        # Set timer
        toc = time.time()
        
        # Print running time
        print("Running time = %.2f mins" % ((toc-tic)/60))
        
        # Return predictions and statistics
        return Pre, Pre_stats, Fi, Fi_avg, Years
    
    else:
        
        # Set timer
        toc = time.time()
        
        # Print running time
        print("Running time = %.2f mins" % ((toc-tic)/60))
        
        # Return predictions 
        return Pre, Fi, Fi_avg, Years


# ### Boosted Regression

def brt(Data, 
        Features, 
        Response, 
        Method, 
        Years, 
        Refit, 
        Validation,
        Val_size,
        N_splits,
        param_grid, 
        Stats,
        Verbose,
        Rounding=2):

    ''' 
    Performs boosted regressions and computes statistics.
  
    Parameters:
    ----------
    Data - A pandas MultiIndex containing the feature, response, return, and size variables.
    Features - A list of strings specifying the feature variables. 
    Response - A string specifying the response variable.
    Method - A string specifying the method (rolling, or expanding).
    Years - An integer specifying the years of rolling window/min years of expanding method.
    Refit - An integer specifying the months after which the model is refitted.
    Validation - A string specifying the validation method.
    Val_size - A float specifying the validation size.
    N_splits - An integer specifying the number of splits used for validation.
    param_grid - A list specifying the parameters to validate.
    Stats - A boolean to decide whether to return statistics.
    Verbose - A boolean to decide whether to print status.
    Rounding - An integer specifying the rounding.

    Returns: 
    ----------
    Pre -  A pandas MultiIndex containing the predictions.
    Pre_stats - A pandas DataFrame containing summary statistics.
    Fi - A pandas DataFrame containing feature importance. 
    Fi_avg - A pandas DataFrame containing average feature importance. 
    Years - An integer specifying the years of rolling window/min years of expanding method.
    '''
    
    # Set timer 
    tic = time.time()

    # Set up Regressor
    Regressor = GradientBoostingRegressor(random_state=1, 
                                          loss="ls",
                                          criterion="mse",
                                          verbose=0)   
    
    # Walk-forward validation
    if Validation == "wf":
        tvs = TrainValSplit(n_splits=N_splits, val_size=Val_size) 
        
    # Blocked walk-forward validation
    elif Validation == "bwf":
        tvs = BlockingTrainValSplit(n_splits=N_splits, val_size=Val_size) 
     
    # Set up girdsearch
    gs = GridSearchCV(Regressor, 
                      param_grid=param_grid, 
                      cv=tvs, 
                      verbose=0, 
                      n_jobs=-1, 
                      scoring="neg_mean_squared_error", 
                      refit=True, 
                      return_train_score=False,
                      iid=False)
    
    # Create empty list for predictions
    Pre = []

    # Create empty list for features importance
    Fi = []

    # Start of slice
    Start = Data.index[0][0] + pd.offsets.BMonthEnd(Years*12)

    # End of slice
    End =  Data.index[-1][0]
    
    # Select index for features importance
    Fi_Ind = Data[Start:End].index.get_level_values(0).unique()[::Refit]
        
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
    
    # Set up counter for status 
    Counter = 1
        
    # Loop through months
    while Timestamp < end_date: 

        # Keep track of time
        Timestamp = start_date + pd.offsets.BMonthEnd(TimeDelta)
    
        # Check whether another regression is possible
        if Timestamp + pd.offsets.BMonthEnd(Years*12-1) < end_date:
        
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
                X_train = Data.loc[start_date:Timestamp+pd.offsets.BMonthEnd(Years*12-1), 
                                   Features]
                  
                # Select response variable for training data
                y_train = Data.loc[start_date:Timestamp+pd.offsets.BMonthEnd(Years*12-1), 
                                   Response]

            # Perform cross validation
            gs.fit(X_train, np.ravel(y_train))
            
            # Add feature importance to list
            Fi.append(pd.DataFrame((gs.best_estimator_.feature_importances_*100).reshape(1, len(Features)), 
                                    columns=Features))
            
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
            
            # Print status 
            if Verbose == True:
                print(Counter,"." " Iteration completed", sep='')  
                
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
            
            # Print status 
            if Verbose == True:
                print(Counter,"." " Iteration completed", sep='')  
                        
            #print("test length", len(np.unique(X_test.index.get_level_values(0))))
            #print("test start", X_test.index[1])
            #print("test end", X_test.index[-1])

        # Add one month 
        TimeDelta += Refit
        
        # Add 1 iteration to counter 
        Counter += 1
    
    # Concat feature importance
    Fi = pd.concat(Fi, axis=0).set_index(Fi_Ind)
    
    # Compute average feature importance
    Fi_avg = Fi.apply(np.mean, axis=0).sort_values(axis=0, ascending=True)
    
    # Concat predictions
    Pre = pd.concat(Pre, axis=0).set_index([Pre_Ind, Pre_ID])

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
        
        # Set timer
        toc = time.time()
        
        # Print running time
        print("Running time = %.2f mins" % ((toc-tic)/60))
        
        # Return predictions and statistics
        return Pre, Pre_stats, Fi, Fi_avg, Years
    
    else:
        
        # Set timer
        toc = time.time()
        
        # Print running time
        print("Running time = %.2f mins" % ((toc-tic)/60))
        
        # Return predictions 
        return Pre, Fi, Fi_avg, Years

