# -*- coding: utf-8 -*-
"""
"""

import pandas as pd
import numpy as np
import datetime as dt
import calendar
import umd as sumd # sumd = "sector umd"
import backtest_stats as bt
import umd_cliff_fall_charts as cliff


def define_input_parameters():
    
    # Driver function to use as inputs for the following function:
    # sumd.main()
    
    ##############################################
    # LOAD AS DATAFRAME FROM CSV:
    ##############################################
#    csv_daily = r'C:\Users\Kyle\Desktop\_FOLDERS_\__DATASETS\Fama French CSVs\12_Industry_Portfolios_Daily.CSV'
    csv_monthly = r'C:\Users\Kyle\Desktop\_FOLDERS_\__DATASETS\Fama French CSVs\12_Industry_Portfolios.CSV'
#    df_daily = pd.read_csv(csv_daily,skiprows=9,nrows=24966-9,index_col=0)
    df_monthly = pd.read_csv(csv_monthly,skiprows=11,nrows=1148-11,index_col=0)
  
    # Date conversion from "YYYYMM" to an actual fucking date:
    dates_fama_m = df_monthly.index
    dates_good_m = convert_fama_monthly_to_dates(dates_fama_m)
    df_monthly = df_monthly.set_index(dates_good_m.index)
    
    # Date conversion from "YYYYMMDD" to an actual fucking date:
#    dates_fama_d = df_daily.index
#    dates_good_d = convert_fama_daily_to_dates(dates_fama_d)
    
    df_to_use = df_monthly
    
    # Convert from Fama French format (1% return = 1.00) to more common format (1% return = 0.01)
    df_to_use = df_to_use / 100

    momentum_window_1 = 12 # note: for classic "(12,1) UMD", this is the "12".
    momentum_window_2 = 0 # note: for classic "(12,1) UMD", this is the "1".      
        
    num_securities_long_leg = 3
    num_securities_short_leg = 3
    
    total_N = df_to_use.shape[1]
    
    data_frequency = 'M'
    
    return df_to_use, momentum_window_1, momentum_window_2, \
           num_securities_long_leg, num_securities_short_leg, total_N, \
           data_frequency, dates_good_m
          
            
def last_business_day_in_month(year: int, month: int) -> int:
    
    return max(calendar.monthcalendar(year, month)[-1:][0][:5])


def convert_fama_monthly_to_dates(fama_dates):
    
    nans = np.empty( ( len(fama_dates) ) )
    nans[:] = np.nan
    good_dates = pd.DataFrame(nans)
    
    for tt in range(0, len(fama_dates)):
        yyyy = int(str(fama_dates[tt])[:4])
        mm = int(str(fama_dates[tt])[4:])
        dd = last_business_day_in_month(yyyy, mm)
        good_dates.iloc[tt,0] = dt.datetime(yyyy, mm, dd) 
 
    good_dates = good_dates.set_index(0)
    return good_dates


def main():
    
    # Get input parameters:
    rtns_df, momentum_window_1, momentum_window_2, \
        num_securities_long_leg, num_securities_short_leg, total_N, \
        data_frequency, dates_good_m = \
        define_input_parameters()
    
    # Call main function from "umd.py":
    rtns_all, historical_weights, \
        historical_momo_signals, historical_momo_ranks, \
        umd_weight_test_1, umd_weight_test_2 = \
        sumd.main(rtns_df, momentum_window_1, momentum_window_2, \
                  num_securities_long_leg, num_securities_short_leg, total_N)
    
    # Collect position level returns from backtest:
    position_rtns = historical_weights.iloc[0:-1,:].values * rtns_all.iloc[1:,:].values
    
    # NOTE: can't do the above in DataFrame space...need to extract raw values 
    # if you don't want your dates/indexes mashed/joined incorrectly...
    position_rtns_df = pd.DataFrame(position_rtns, 
                                    columns = rtns_all.columns, 
                                    index = rtns_all.index[1:])
    backtest_rtns_series = np.sum(position_rtns_df,axis=1)
    
    # Convert from Series to dataframe:
    backtest_rtns = pd.DataFrame(backtest_rtns_series, 
                                 columns = ['UMD_('+str(momentum_window_1)+\
                                                   str(',')+\
                                                   str(momentum_window_2)+\
                                                   str(')')])
        
    # The first "momentum_window_1" rows will be blank, by construction:
    backtest_rtns = backtest_rtns.iloc[momentum_window_1:,:]
    
    # Get yearly frequency returns:
    if (data_frequency == 'D'):
        backtest_m = bt.convert_daily_rtns_to_monthly_rtns(backtest_rtns)
        backtest_y = bt.convert_daily_rtns_to_yearly_rtns(backtest_rtns)
        backtest_cal = bt.calendar_dataframe( backtest_m , backtest_y )
    elif (data_frequency == 'M'):
        backtest_y = bt.convert_monthly_rtns_to_yearly_rtns(backtest_rtns)
        backtest_cal = bt.calendar_dataframe( backtest_rtns , backtest_y )
    
    # Current wts = most recent row of historicals:
    current_weights = \
        bt.get_current_weights(historical_weights)
        
    # Get "fell off cliff" charts:
    max_lags_to_test = 20
    triangle_rtns_mtrx, weights_for_triangle, ranks_for_triangle, \
            triangle_weight_test_1, triangle_weight_test_2 = \
        cliff.get_lower_triangle_returns_matrix(rtns_all, \
                                                num_securities_long_leg, \
                                                num_securities_short_leg, \
                                                max_lags_to_test, \
                                                total_N)
        
    row_of_means_same_history, row_of_means_all_rows, test_result_arith = \
        cliff.get_arith_means_chart(triangle_rtns_mtrx)
        
    row_of_vols_same_history, row_of_vols_all_rows, test_result_vols = \
        cliff.get_volatility_chart(triangle_rtns_mtrx)
         
        
    return  backtest_rtns, backtest_y, backtest_cal, \
            rtns_all, \
            current_weights, historical_weights, \
            historical_momo_signals, historical_momo_ranks, \
            umd_weight_test_1, umd_weight_test_2, \
            triangle_rtns_mtrx, weights_for_triangle, ranks_for_triangle, \
            triangle_weight_test_1, triangle_weight_test_2, \
            row_of_means_same_history, row_of_means_all_rows, test_result_arith, \
            row_of_vols_same_history, row_of_vols_all_rows, test_result_vols
            

backtest_rtns, backtest_y, backtest_cal, \
    rtns_all, \
    current_weights, historical_weights, \
    historical_momo_signals, historical_momo_ranks, \
    umd_weight_test_1, umd_weight_test_2, \
    triangle_rtns_mtrx, weights_for_triangle, ranks_for_triangle, \
    triangle_weight_test_1, triangle_weight_test_2, \
    row_of_means_same_history, row_of_means_all_rows, test_result_arith, \
    row_of_vols_same_history, row_of_vols_all_rows, test_result_vols \
    = main()
