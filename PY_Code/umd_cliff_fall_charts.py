# -*- coding: utf-8 -*-
"""
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_lower_triangle_returns_matrix(rtns, \
                                      num_securities_long_leg, \
                                      num_securities_short_leg, \
                                      max_lags, \
                                      total_N ):
    
    # Initialize output:
    nans = np.empty( ( rtns.shape[0] , max_lags ) )
    nans[:] = np.nan
    triangle_mtrx = pd.DataFrame(nans)
    triangle_mtrx = triangle_mtrx.set_index(rtns.index)
    triangle_mtrx.columns = list(range(0+1,max_lags+1))
    
    # Populate ranks for each row:
    ranks = initialize_output(rtns)
    for rr in range(0,ranks.shape[0]):
        # Lowest = 1; largest = n ... when "ascending=True"
        temp_rank = rtns.iloc[rr,:].rank(method='average',ascending=False) 
        # Put transpose in output variable:
        ranks.iloc[rr,:] = temp_rank.transpose()
        
    # Based on ranks, populate weights:
    historical_all_weights      = initialize_output(rtns)
    historical_long_weights     = initialize_output(rtns)
    historical_short_weights    = initialize_output(rtns)
    
    # Populate Long weights:
    long_thresh = num_securities_long_leg + 0.99 # threshold rank for long leg inclusion
    for tt in range( 0 , rtns.shape[0] ):
        # Lowest = 1 (1=most desirable: lowest vol, lowest rho, highest momo); 
        # largest = n
        
        # Initialze entire row to zeroes:
        historical_long_weights.iloc[tt,:] = np.zeros([1,rtns.shape[1]]) 
        
        # Check if there are ties (happens more than you'd think, 
        # especially for (1,0) or (x,x-1) formations)):
        ranks_tt = ranks.iloc[tt,:]
        num_secs_tt = ranks_tt[ranks_tt < long_thresh].count()
        wt = 1 / num_secs_tt
        
        # Assign the non-zero weights:
        for nn in range( 0 , rtns.shape[1] ):
            if(ranks_tt[nn] < long_thresh):
                historical_long_weights.iloc[tt,nn] = wt
                
    # Populate SHORT weights:
    short_thresh = total_N - num_securities_short_leg + 0.01 # threshold rank for short leg inclusion
    for tt in range( 0 , rtns.shape[0] ):
        # Lowest = 1 (1=most desirable: lowest vol, lowest rho, highest momo); 
        # largest = n
        
        # Initialze entire row to zeroes:
        historical_short_weights.iloc[tt,:] = np.zeros([1,rtns.shape[1]]) 
        
        if (num_securities_short_leg > 0) :
            # Check if there are ties (happens more than you'd think):
            ranks_tt = ranks.iloc[tt,:]
            num_secs_tt = ranks_tt[ranks_tt > short_thresh].count()
            wt = 1 / num_secs_tt
            
            # Assign the non-zero weights:
            for nn in range( 0 , rtns.shape[1] ):
                if(ranks_tt[nn] > short_thresh ):
                    historical_short_weights.iloc[tt,nn] = -wt
                
    # All weights:
    historical_all_weights = historical_long_weights + historical_short_weights
    
    
    # Get MOMO returns for each of {(1,0), (2,1), (3,2), ..., (max_lags,max_lags-1) } :
    for tt in range( 0 , rtns.shape[0] ):
        
        # Print for visibility to console:
        if (tt % 100 == 0):
            print(tt)
        
        for lags in range( 0 , max_lags):
            
            if (tt >= (lags+1) ):
                wts = historical_all_weights.iloc[tt-(lags+1),:].values
                pos_rtns = rtns.iloc[tt,:].values
                
                triangle_mtrx.iloc[tt,lags] = sum(wts * pos_rtns)
                
    # Check that weights sum to zero:
    weight_test_1 = 1 # initialize to TRUE
    
    # Check that max weight is not > 1/num_securities_long_leg (within machine precision)
    # &
    # Check that min weight is not < -1/num_securities_short_leg:
    weight_test_2 = 1 # initialize to TRUE
    
    
    return  triangle_mtrx, historical_all_weights, ranks, \
            weight_test_1, weight_test_2 


def initialize_output(rtns):
    
    # Function to initialize output & keep row headings (dates) and column headings (tickers):
    nans = np.empty( ( rtns.shape[0] , rtns.shape[1] ) )
    nans[:] = np.nan
    output_df = pd.DataFrame(nans)
    output_df = output_df.set_index(rtns.index)
    output_df.columns = rtns.columns
    
    return output_df


def get_arith_means_chart(triangle_mtrx):
    
    # Determine "max_lag_tested" from structure of matrix itself:
    max_lags_tested = triangle_mtrx.shape[1]
    
    # Initialize outputs:
    nans = np.empty( ( 1 , max_lags_tested ) )
    nans[:] = np.nan
    row_of_means_same_history = pd.DataFrame(nans)
    row_of_means_same_history.columns = triangle_mtrx.columns
    
    # Initialize again because of how python handles pointers:
    nans2 = np.empty( ( 1 , max_lags_tested ) )
    nans2[:] = np.nan
    row_of_means_all_rows = pd.DataFrame(nans2)
    row_of_means_all_rows.columns = triangle_mtrx.columns
    
    # Aritmetic means of all columns for common subperiod of no NANs:
    temp_row_of_means_same_history = \
        pd.DataFrame.mean(triangle_mtrx.iloc[(max_lags_tested-1):,:], axis=0)
    # Convert from series to DFrame:    
    row_of_means_same_history.iloc[0,:] = temp_row_of_means_same_history
    
    # Aritmetic means of all columns without regard to NANs:
    temp_row_of_means_all_rows = \
        pd.DataFrame.mean(triangle_mtrx, axis=0)    
    # Convert from series to DFrame: 
    row_of_means_all_rows.iloc[0,:] = temp_row_of_means_all_rows
    
    # Unit test: check that these two entries match:
    # (1) row_of_means_same_history.iloc[0,max_lag_tested-1]
    # (2) row_of_means_all_rows.iloc[0,max_lag_tested-1]
    test_result = 1 # initialize to TRUE
    
    return row_of_means_same_history, row_of_means_all_rows, test_result


def get_geometric_means_chart(triangle_mtrx):
    
    # Determine "max_lag_tested" from structure of matrix itself:
    max_lag_tested = triangle_mtrx.shape[1]
    
    # Geometric means of 
    
    return row_of_geoms_same_history, row_of_geoms_all_rows
    
    
def get_volatility_chart(triangle_mtrx):
    
    # Determine "max_lag_tested" from structure of matrix itself:
    max_lags_tested = triangle_mtrx.shape[1]
    
    # Initialize outputs:
    nans = np.empty( ( 1 , max_lags_tested ) )
    nans[:] = np.nan
    row_of_vols_same_history = pd.DataFrame(nans)
    row_of_vols_same_history.columns = triangle_mtrx.columns
    
    # Initialize again because of how python handles pointers:
    nans2 = np.empty( ( 1 , max_lags_tested ) )
    nans2[:] = np.nan
    row_of_vols_all_rows = pd.DataFrame(nans2)
    row_of_vols_all_rows.columns = triangle_mtrx.columns
    
    # Sample Std Devs of all columns for common subperiod of no NANs:
    temp_row_same_history = \
        pd.DataFrame.std(triangle_mtrx.iloc[(max_lags_tested-1):,:], axis=0)
    # Convert from series to DFrame:    
    row_of_vols_same_history.iloc[0,:] = temp_row_same_history
    
    # Sample Std Devs of all columns without regard to NANs:
    temp_row_all_rows = \
        pd.DataFrame.std(triangle_mtrx, axis=0)    
    # Convert from series to DFrame: 
    row_of_vols_all_rows.iloc[0,:] = temp_row_all_rows
    
    # Unit test: check that these two entries match:
    # (1) row_of_means_same_history.iloc[0,max_lag_tested-1]
    # (2) row_of_means_all_rows.iloc[0,max_lag_tested-1]
    test_result = 1 # initialize to TRUE
    
    return row_of_vols_same_history, row_of_vols_all_rows, test_result
    
    
def get_significance_chart(triangle_mtrx):
    
    
    return row_of_tstats_same_history, row_of_tstats_all_rows
    
    
def plot_arith_means_chart(row_of_means):
    
    plt.style.use('ggplot')
    
    x = row_of_means.columns
    vals = row_of_means.values
    
    x_pos = [i for i, _ in enumerate(x)]
    
    plt.bar(x_pos, vals, color='green')
    plt.xlabel("Lag (X, X-1)")
    plt.ylabel("Average Monthly Return (%)")
    plt.title("Panel A: xxx")
    
    plt.xticks(x_pos, x)
    
    plt.show()

    return

def plot_arith_means_chart_test(row_of_means):
    
    plt.style.use('ggplot')
    
    x = ['Nuclear', 'Hydro', 'Gas', 'Oil', 'Coal', 'Biofuel']
    energy = [5, 6, 15, 22, 24, 8]
    
    x_pos = [i for i, _ in enumerate(x)]
    
    plt.bar(x_pos, energy, color='green')
    plt.xlabel("Energy Source")
    plt.ylabel("Energy Output (GJ)")
    plt.title("Energy output from various fuel sources")
    
    plt.xticks(x_pos, x)
    
    plt.show()

    return
    