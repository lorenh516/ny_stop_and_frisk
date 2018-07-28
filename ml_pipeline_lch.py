import pandas as pd
import numpy as np
import re
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler, normalize

def retrieve_data(filename, headers = False, set_ind = None):
    '''
    Read in data from CSV to a pandas dataframe

    Inputs:
        filename: (string) filename of CSV
        headers: (boolean) whether or not CSV includes headers
        ind: (integer) CSV column number of values to be used as indices in 
            data frame

    Output: pandas data frame
    '''
    if headers and isinstance(set_ind, int):
        data_df = pd.read_csv(filename, header = 0, index_col = set_ind)
    elif headers and not set_ind:
        data_df = pd.read_csv(filename, header = 0)
    else:
        data_df = pd.read_csv(filename)
    return data_df



def print_null_freq(df, blanks_only = False):
    '''
    For all columns in a given dataframe, calculate and print number of null and non-null values

    Attribution: Adapted from https://github.com/yhat/DataGotham2013/blob/master/analysis/main.py
    '''
    df_lng = pd.melt(df)
    null_variables = df_lng.value.isnull()
    all_rows = pd.crosstab(df_lng.variable, null_variables)
        
    if blanks_only:
        try:
            return all_rows[all_rows[True] > 0]
        except:
            return False
    else: 
        return all_rows


def still_blank(train_test_tuples):
    '''
    Check for remaining null values after dummy variable creation is complete.
    '''
    to_impute = []
    for train, test in train_test_tuples:
        with_blanks = print_null_freq(train, blanks_only = True)
        print(with_blanks)
        print()
        to_impute.append(list(with_blanks.index))
    return to_impute


def create_col_ref(df):
    '''
    Develop quick check of column position via dictionary
    '''
    col_list = df.columns
    col_dict = {}
    for list_position, col_name in enumerate(col_list):
        col_dict[col_name] = list_position
    return col_dict


def abs_diff(col, factor, col_median, MAD):
    '''
    Calculate modified z-score of value in pandas data frame column, using 
    sys.float_info.min to avoid dividing by zero

    Inputs:
        col: column name in pandas data frame
        factor: factor for calculating modified z-score (0.6745)
        col_median: median value of pandas data frame column
        MAD: mean absolute difference calculated from pandas dataframe column
    
    Output: (float) absolute difference between column value and column meaan 
        absolute difference

    Attribution: workaround for MAD = 0 adapted from https://stats.stackexchange.com/questions/339932/iglewicz-and-hoaglin-outlier-test-with-modified-z-scores-what-should-i-do-if-t
    '''
    if MAD == 0:
        MAD = 2.2250738585072014e-308 
    return (x - y)/ MAD



def outliers_modified_z_score(df, col):
    '''
    Identify outliers (values falling outside 3.5 times modified z-score of 
    median) in a column of a given data frame

    Output: (pandas series) outlier values in designated column

    Attribution: Modified z-score method for identifying outliers adapted from 
    http://colingorrie.github.io/outlier-detection.html
    '''
    threshold = 3.5
    zscore_factor = 0.6745
    col_median = df[col].astype(float).median()
    median_absolute_deviation = abs(df[col] - col_median).mean()
    
    modified_zscore = df[col].apply(lambda x: abs_diff(x, zscore_factor, 
                                    col_median, median_absolute_deviation))
    return modified_zscore[modified_zscore > threshold]


def convert_dates(date_series):
    '''
    Faster approach to datetime parsing for large datasets leveraging repated dates.

    Attribution: https://github.com/sanand0/benchmarks/commit/0baf65b290b10016e6c5118f6c4055b0c45be2b0
    '''
    dates = {date:pd.to_datetime(date) for date in date_series.unique()}
    return date_series.map(dates)


def make_boolean(df, cols, value_1s, ints = True):
    if ints:
        true_val = 1
        neg_val = 0
    else:
        true_val = True
        neg_val = False
        
    for col in cols:
        df.loc[df[col] != value_1s, col] = neg_val
        df.loc[df[col] == value_1s, col] = true_val





def view_max_mins(df, max = True):
    '''
    View top and bottom 10% of values in each column of a given data frame

    Inputs: 
        df: pandas dataframe
        max: (boolean) indicator of whether to return to or bottom values

    Output: (dataframe) values at each 100th of a percentile for top or bottom 
        values dataframe column
    '''
    if max:
        return df.quantile(q=np.arange(0.99, 1.001, 0.001))
    else: 
        return df.quantile(q=np.arange(0.0, 0.011, 0.001))



def view_likely_outliers(df, max = True):
    '''
    View percent change between percentiles in top or bottom 10% of values in  
    each column of a given data frame 

    Inputs: 
        df: pandas dataframe
        max: (boolean) indicator of whether to return to or bottom values

    Output: (dataframe) percent changes between values at each 100th of a 
        percentile for top or bottom values in given dataframe column
    '''
    if max:
        return df.quantile(q=np.arange(0.9, 1.001, 0.001)).pct_change()
    else: 
        return df.quantile(q=np.arange(0.0, 0.011, 0.001)).pct_change()



def remove_over_under_threshold(df, col, min_val = False, max_val = False, lwr_threshold = None, upr_threshold = False):
    '''
    Remove values over given percentile or value in a column of a given data 
    frame
    '''
    if max_val:
        df.loc[df[col] > max_val, col] = None
    if min_val:
        df.loc[df[col] < min_val, col] = None
    if upr_threshold:
        maxes = view_max_mins(df, max = True)
        df.loc[df[col] > maxes.loc[upr_threshold, col], col] = None
    if lwr_threshold:
        mins = view_max_mins(df, max = False)
        df.loc[df[col] < mins.loc[lwr_threshold, col], col] = None
    

def remove_dramatic_outliers(df, col, threshold, max = True):
    '''
    Remove values over certain level of percent change in a column of a given 
    data frame
    '''
    if max:
        maxes = view_max_mins(df, max = True)
        likely_outliers_upper = view_likely_outliers(df, max = True)
        outlier_values = list(maxes.loc[likely_outliers_upper[likely_outliers_upper[col] > threshold][col].index, col])
    else: 
        mins = view_max_mins(df, max = False)
        likely_outliers_lower = view_likely_outliers(df, max = False)
        outlier_values = list(mins.loc[likely_outliers_lower[likely_outliers_lower[col] > threshold][col].index, col])
    
    df = df[~df[col].isin(outlier_values)]



def basic_fill_vals(df, col_name, test_df = None, method = None, replace_with = None):
    '''
    For columns with more easily predicatable null values, fill with mean, median, or zero

    Inputs:
        df: pandas data frame
        col_name: (string) column of interest
        method: (string) desired method for filling null values in data frame. 
            Inputs can be "zeros", "median", or "mean"
    '''
    if method == "zeros":
        df[col_name].fillna(0, inplace = True)
    elif method == "replace":
        replacement_val = replace_with
        df[col_name].fillna(replacement_val, inplace = True)
    elif method == "median":
        replacement_val = df[col_name].median()
        df[col_name].fillna(replacement_val, inplace = True)
    elif method == "mean":
        replacement_val = df[col_name].mean()
        df[col_name].fillna(replacement_val, inplace = True)

    # if imputing train-test set, fill test data frame with same values
    if test_df is not None:
        test_df[col_name].fillna(replacement_val, inplace = True)



def check_col_types(df):
    return pd.DataFrame(df.dtypes, df.columns).rename({0: 'data_type'}, axis = 1)




def is_category(col_name, flag = None, geos = True):
    '''
    Utility function to determine whether a given column name includes key words or
    phrases indicating it is categorical.

    Inputs:
        col_name: (string) name of a column
        geos: (boolean) whether or not to include geographical words or phrases
            in column name search
    '''
    search_for = ["_bin","_was_null"]

    if flag:
        search_for += [flag]

    if geos:
        search_for += ["city", "state", "county", "country", "zip", "zipcode", "latitude", "longitude"]

    search_for = "|".join(search_for)

    return re.search(search_for, col_name)


def summarize_df(df):
    type_dict = defaultdict(list)
    geos = ["city", "state", "county", "country", "zip", "zipcode", "latitude", "longitude"]
    geos = "|".join(geos)
    summary = pd.DataFrame(columns = ["col_name", "num_values", "num_nulls", "unique_values",  "data_type", "col_type", "most_common", "prevalence"])
    
    for col in df.columns:
        num_values = df[col].value_counts().sum()
        uniques = len(df[col].unique())
        nulls = df[col].isnull().sum()
        most_common = list(df[col].mode())[0]
        mode_count = (df[col].value_counts().max() / num_values) * 100
        dtype = df[col].dtype


        if re.search(geos, col):
            col_type = "geo"
            type_dict["geo"].append(col)
        elif re.search("id|_id", col):
            col_type = "ID"
            type_dict["ID"].append(col)
        elif df[col].dtype.str[1] == 'M':
            col_type = "datetime"
            type_dict["datetime"].append(col)
        elif df[col].dtype.kind in 'uifc':
            col_type = "numeric"
            type_dict["numeric"].append(col)
        elif uniques == 1 or uniques == 2:
            col_type = "binary"
            type_dict["binary"].append(col)
        elif uniques <= 6:
            col_type = "multi"
            type_dict["multi"].append(col)
        elif uniques > 6:
            col_type = "tops"
            type_dict["tops"].append(col)
        summary.loc[col] = [col, num_values, nulls, uniques, dtype, col_type, most_common, mode_count]
    
    summary.set_index("col_name", inplace = True)
    return summary, type_dict


def recateogrize_col(col, new_category, col_dict):
    for category, cols_list in col_dict.items():
        if col in cols_list:
            col_dict[category] = [column for column in cols_list if column != col]
    col_dict[new_category].append(col)
    return col_dict


def replace_dummies(df, cols_to_dummy):
    return pd.get_dummies(df, columns = cols_to_dummy , dummy_na=True)



def isolate_categoricals(df, categoricals_fcn, ret_categoricals = False, keyword = None, geos_indicator = True):
    '''
    Retrieve list of cateogrical or non-categorical columns from a given dataframe

    Inputs:
        df: pandas dataframe
        categoricals_fcn: (function) Function to parse column name and return boolean
            indicating whether or not column is categorical
        ret_categoricals: (boolean) True when output should be list of  
            categorical colmn names, False when output should be list of 
            non-categorical column names

    Outputs: list of column names from data frame
    '''
    categorical = [col for col in df.columns if categoricals_fcn(col, flag = keyword, geos = geos_indicator)]
    non_categorical = [col for col in df.columns if not categoricals_fcn(col, flag = keyword, geos = geos_indicator)]
    
    if ret_categoricals:
        return categorical
    else:
        return non_categorical



def change_col_name(df, current_name, new_name):
    '''
    Change name of a single column in a given data frame
    '''
    df.columns = [new_name if col == current_name else col for col in df.columns]



def drop_unwanted(df, drop_list):
    df.drop(drop_list, axis = 1, inplace = True)




def time_series_split(df, date_col, train_size, test_size, increment = 'month', specify_start = None):
    
    if specify_start:
        min_date = datetime.strptime(specify_start, '%Y-%m-%d')
    else:
        min_date = df[date_col].min()

        if min_date.day > 25:
            min_date += datetime.timedelta(days = 7)
            min_date = min_date.replace(day=1, hour=0, minute=0, second=0)

        else:
            min_date = min_date.replace(day=1, hour=0, minute=0, second=0)
    
    if increment == 'month':
        train_max = min_date + relativedelta(months = train_size) - timedelta(days = 1)
        test_min = train_max + timedelta(days = 1)
        test_max = min(test_min + relativedelta(months = test_size), df[date_col].max())
        
    if increment == 'day':
        train_max = min_date + relativedelta(days = train_size)
        test_min = train_max + timedelta(days = 1)
        test_max = min((test_min + relativedelta(days = test_size)), df[date_col].max())
    
    if increment == 'year':
        train_max = timedelta(months = train_size) - timedelta(days = 1)
        test_min = train_max + relativedelta(years = train_size)
        test_max = min(test_min + relativedelta(years = test_size), df[date_col].max())
    
    new_df = df[df.columns]
    train_df = new_df[(new_df[date_col] >= min_date) & (new_df[date_col] <= train_max)]
    test_df = new_df[(new_df[date_col] >= test_min) & (new_df[date_col] <= test_max)]
    
    date_refs = (increment, min_date, train_size, test_min, test_size)

    return train_df, test_df, date_refs



def create_expanding_splits(df, total_periods, dates, train_period_base, test_period_size, period = 'month', defined_start = None):
    num_months = total_periods / test_period_size
    months_used = train_period_base
    
    tt_sets = []
    set_dates = pd.DataFrame(columns = ("period", "training_start", "training_period", "test_period_start", "test_period"))
    
    while months_used < total_periods:
        
        print("original train period lenth: {}".format(train_period_base))
        train, test, date_ref = time_series_split(df, date_col = dates, train_size = train_period_base, test_size = test_period_size, increment = period, specify_start = defined_start)
        
        print("train: {}, test: {}".format(train.shape, test.shape))
        tt_sets.append((train, test))
        train_period_base += test_period_size
        months_used += test_period_size
        set_dates.loc[len(set_dates)] = list(date_ref)

    return (tt_sets, set_dates)




def train_top_dummies(train_df, tops_list, threshold, max_options = 10):
    set_distro_dummies = []
    counter = 1
    dummies_dict = {}

    for col in tops_list:
        col_sum = train_df[col].value_counts().sum()
        top = train_df[col].value_counts().nlargest(max_options)
        
        top_value = 0
        num_dummies = 0

        while ((top_value / col_sum) < threshold) & (num_dummies < max_options):
            top_value += top[num_dummies]
            num_dummies += 1

        keep_dummies = list(top.index)[:num_dummies]
        dummies_dict[col] = keep_dummies
        
    counter += 1
    set_distro_dummies.append(dummies_dict)

    return set_distro_dummies



def apply_tops(set_specific_dummies, var_dict, train_df, test_df):
    counter = 0
    for set_dict in set_specific_dummies:
        counter += 1
        for col, vals in set_dict.items():
            train_df.loc[~train_df[col].isin(vals), col] = 'Other'
            test_df.loc[~test_df[col].isin(vals), col] = 'Other'




def iza_process(train_df, test_df, var_dict, tops_threshold = 0.5, binary = None, geos = False):
    # for i, (train_df, test_df) in enumerate(dfs):
    #     print("Starting set {}...".format(i))
        
    drop_unwanted(train_df, var_dict['datetime'])
    drop_unwanted(test_df, var_dict['datetime'])
    
    if binary is not None:
        make_boolean(train_df, var_dict['binary'], value_1s = binary)
        make_boolean(test_df, var_dict['binary'], value_1s = binary)
        print("Binary columns successfully converted.")

    
    train_df = replace_dummies(train_df, var_dict['multi'])
    test_df = replace_dummies(test_df, var_dict['multi'])
    
    # print("Values in columns {} successfully converted to dummies".format(var_dict['multi']))

    tops = train_top_dummies(train_df, var_dict['tops'], threshold = tops_threshold, max_options = 10)
    apply_tops(tops, var_dict, train_df, test_df)
    

    train_df = pd.get_dummies(train_df, columns = var_dict['tops'], dummy_na = True)
    test_df = pd.get_dummies(test_df, columns = var_dict['tops'], dummy_na = True)


    # print("Top values in columns {} successfully converted to dummies".format(var_dict['tops']))


    if geos:
        geo_tops = train_top_dummies(train_df, var_dict['geo'], threshold = tops_threshold, max_options = 5)
        apply_tops(geo_tops, var_dict, train_df, test_df)

        train_df = pd.get_dummies(train_df, columns = var_dict['geo'], dummy_na = True)
        test_df = pd.get_dummies(test_df, columns = var_dict['geo'], dummy_na = True)
        
        # print("Values in columns {} successfully converted to dummies".format(var_dict['geo']))
    print("Converted nonbinary, non-numeric columns to dummies.")

    
    for col in var_dict['numeric']:
        basic_fill_vals(train_df, col_name = col, test_df = test_df, method = 'mean')
        train_df.loc[:, col] = normalize(pd.DataFrame(train_df[col]), axis = 0)
        test_df.loc[:, col] = normalize(pd.DataFrame(test_df[col]), axis = 0)
    print("Filled missing values and normalizied values in numeric columns.")

    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)

    extra_train = train_cols - test_cols
    extra_test = test_cols - train_cols

    if len(extra_train) > 0:
        for col in extra_train:
            test_df[col] = 0

    if len(extra_test) > 0:
        for col in extra_test:
            train_df[col] = 0

    print("Moving to next set!")
    return (train_df, test_df)







