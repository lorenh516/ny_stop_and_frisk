import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import math
import re
from ml_pipeline_lch import isolate_categoricals, is_category


def view_dist(df, geo_columns = True, fig_size=(20,15), labels = None):
    '''
    Plot distributions of non-categorical columns in a given dataframe

    Inputs:
        df: pandas dataframe
        geo_columns: list of column names corresponding to columns with numeric geographical information (ex: zipcodes)
        labels: list of labels to apply to plot: title, xlable, ylabel, respectively
    '''
    non_categoricals = isolate_categoricals(df, categoricals_fcn = is_category, ret_categoricals = False, geos_indicator = geo_columns)
    # non_categoricals = isolate_noncategoricals(df, ret_categoricals = False, 
    #                                             geo_cols = geo_columns)
    df[non_categoricals].hist(bins = 10, figsize=fig_size, color = 'blue')
    if labels:
        plt.title(labels[0])
        plt.xlabel(labels[1])
        plt.ylabel(labels[2])
    plt.show()


def plot_value_counts(df, type_dict, category, norm = False, plot_kind = 'bar'):
    for col in type_dict[category]:
        plot_title = col + " distribution"
        df[col].value_counts(normalize = norm).plot(kind=plot_kind)
        plt.title(plot_title)
        plt.xlabel(col)
        plt.ylabel("frequency")
        plt.show()


def check_corr(df, geo_columns = True, cat_cols = None):
    '''
   Display heatmap of linear correlation between non-categorical columns in a 
   given dataframe

    Inputs:
        df: pandas dataframe
        geo_columns: list of column names corresponding to columns with numeric 
            geographical information (ex: zipcodes)

    Attribution: Colormap Attribution: adapted from gradiated dataframe at 
    https://www.datascience.com/blog/introduction-to-correlation-learn-data-science-tutorials and correlation heatmap at https://stackoverflow.com/questions/29432629/correlation-matrix-using-pandas
    '''
    try:
        non_categoricals = isolate_categoricals(df, categoricals_fcn = is_category, 
            ret_categoricals = False, geos_indicator = geo_columns)

        fig, ax = plt.subplots(figsize=(12, 12))
        corr = df[non_categoricals].corr(method="pearson")
        sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), 
                    cmap=plt.get_cmap("coolwarm"), square=True, ax=ax, annot=True)
        
        ax.set_xticks(range(len(non_categoricals)))
        ax.set_yticks(range(len(non_categoricals)))

        ax.tick_params(direction='inout')
        ax.set_xticklabels(non_categoricals, rotation=45, ha='right')
        ax.set_yticklabels(non_categoricals, rotation=45, va='top')
        plt.title('Feature Correlation')
        plt.show()
    
    except:
        if cat_cols:
            cat_df = df[df.columns]
            
            for col in cat_cols:
                cat_df[col] = cat_df[col].astype('categorical')

            fig, ax = plt.subplots(figsize=(12, 12))
            corr = cat_df.corr(method="pearson")
            sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), 
                        cmap=plt.get_cmap("coolwarm"), square=True, ax=ax, annot=True)
            
            ax.set_xticks(range(len(cat_df.columns)))
            ax.set_yticks(range(len(cat_df.columns)))

            ax.tick_params(direction='inout')
            ax.set_xticklabels(cat_df.columns, rotation=45, ha='right')
            ax.set_yticklabels(cat_df.columns, rotation=45, va='top')
            plt.title('Feature Correlation')
            plt.show()



def discretize_cols(df, num_bins, geo_columns=True, specific_cols = False, split = False):
    '''
    Add columns to discretize and classify non-categorical columns in a given 
    data frame

    Inputs:
        df: pandas dataframe
        geo_columns:  list of column names corresponding to columns with 
            numeric geographical information (ex: zipcodes)
        num_bins: number of groups into which column values should be 
            discretized
    '''
    if specific_cols:
        non_categoricals = specific_cols
    else:
        non_categoricals = isolate_categoricals(df, 
            categoricals_fcn = is_category, ret_categoricals = False, 
            geos_indicator = geo_columns)
   
    for col in non_categoricals:
        bin_col = col + "_bin"
        if col == "age":
            age_bins = math.ceil((df[col].max() - df[col].min()) / 10)

            if split:
                df[bin_col], train_bins = pd.cut(df[col], bins = age_bins, 
                    right = False, precision=0, retbins=split)
            else:
                df[bin_col] = pd.cut(df[col], bins = age_bins, right = False, 
                    precision=0, retbins=split)
        else:
            if split:
                df[bin_col], train_bins = pd.cut(df[col], bins = num_bins, 
                    precision=0, retbins=split)
            else:
                df[bin_col] = pd.cut(df[col], bins = num_bins, precision=0, 
                    retbins=split)
    if split:
        return train_bins





def discretize_train_test(train_test_tuples, still_blanks):
    for i, (train, test) in enumerate(train_test_tuples):
        fill_cols = still_blanks[i]
        for col in fill_cols:
            grouped = col + '_bin'
            train[grouped], train_bins = pd.cut(train[col], bins = 4, precision = 0, retbins = True)
            test[grouped] = pd.cut(test[col], bins = train_bins, precision = 0)


def confirm_train_test_discretization(train_test_tuples, still_blanks):
    for i, (train, test) in enumerate(train_test_tuples):
        for col in still_blanks[i]:
            grouped = col
            grouped = col + '_bin'
            print("set {} {} train: {}.".format(i, col, train[grouped].unique()))
            print()
            print("set {} {} test: {}.".format(i, col, test[grouped].unique()))
            print()
        

def drop_tt_binned(train_test_tuples, to_drop):
    '''
    Drop columns from both train and test sets.
    
    Inputs:
        train_test_tuples: list of tupled dataframes
        to_drop: list of columns to drop
    '''
    for train, test in train_test_tuples:
        train.drop(to_drop, axis = 1, inplace = True)
        test.drop(to_drop, axis = 1, inplace = True)


def create_binary_vars(df, cols_to_dummy, keyword_list):
    '''
    Create columns of binary values corresponding to values above zero for 
    selected columns in a given dataframe based on common keywords

    Inputs:
        df: pandas dataframe
        cols_to_dummy: (list of strings) columns in data frame to be evaluated 
            into dummy variables
        keyword_list: (list of strings) words or phrases included in columns 
            to be evaluated indicating a dummy variable should be created based 
            on its values 
    '''
    keyword_string = ("|").join(keyword_list)
    for col in cols_to_dummy:
        colname_trunc = re.sub(keyword_string, '', col)
        binary_col_name = 'tf_' + colname_trunc
        df[binary_col_name] = df[col].apply(lambda x: x > 0)



def plot_corr(df, color_category, geo_columns=True):
    '''
    Observe distributions and correlations of features for non-categorical 

    Inputs:
        df: pandas dataframe
        categoricals_list: list of strings corresponding to categorical columns 
            (ex: zip codes)
    '''
    non_categoricals = isolate_categoricals(df, categoricals_fcn = is_category, 
        ret_categoricals = False, geos_indicator = geo_columns)
   
    plot_list = non_categoricals + [color_category]
    corr = sns.pairplot(df[plot_list], hue = color_category, palette = "Set2")



def plot_relationship(df, feature_x, xlabel,feature_y, ylabel, xlimit = None, 
                        ylimit = None, color_cat = None, filter_col = None, 
                        filter_criteria = None, filter_param = None, 
                        filter_param2 = None):
    '''
    Plot two features in a given data frame against each other to view 
    relationship and outliers. 
    
    Attribution: adapted from https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Python_Seaborn_Cheat_Sheet.pdf
    '''
    if filter_col and filter_criteria and filter_param:
        if filter_criteria == 'geq':
            use_df = df[df[filter_col] >= filter_param]
        elif filter_criteria == 'gt':
            use_df = df[df[filter_col] > filter_param]
        elif filter_criteria == 'leq':
            use_df = df[df[filter_col] <= filter_param]
        elif filter_criteria == 'lt':
            use_df = df[df[filter_col] < filter_param]
        elif filter_criteria == 'eq':
            use_df = df[df[filter_col] == filter_param]
        elif filter_criteria == 'neq':
            use_df = df[df[filter_col] != filter_param]
        elif filter_criteria == 'between':
            use_df = df[(df[filter_col] > filter_param) and (df[filter_col] < filter_param2)]

        g = sns.lmplot(x = feature_x, y = feature_y, data = use_df, aspect = 3, 
                        hue = color_cat)
        g = (g.set_axis_labels(xlabel,ylabel)).set(xlim = xlimit , ylim = ylimit)
        plot_title = ylabel + " by " + xlabel
        plt.title(plot_title)
        plt.show(g)
    
    else:
        g = sns.lmplot(x = feature_x, y = feature_y, data = df, aspect = 3, 
                        hue = color_cat)
        g = (g.set_axis_labels(xlabel,ylabel)).set(xlim = xlimit , ylim = ylimit)
        plot_title = ylabel + " by " + xlabel
        plt.title(plot_title)
        plt.show(g)







def eval_ratios(df, include_cols, category_cols, method = "count", 
                pct = False):
    '''
    Evaluate specific features via grouping on one or more category 
    
    Inputs:
        df: (dataframe) pandas dataframe
        include_cols: (list of strings) column names to be aggregated or 
            grouped 
        category_cols: (list of strings) column name(s) for variable(s) used 
            for grouping
        method: (string) groupby aggregation method for column values

    Output:
        ratio_df: pandas data frame of grouped data
    '''
    if method == "count":
        ratio_df = df[include_cols].groupby(category_cols).count()
        if pct:
            single_col = include_cols[-1] + " Percentage"
            ratio_df[single_col] = ((df[include_cols].groupby(category_cols).count() / 
                df[include_cols].groupby(category_cols).count().sum()) * 100)

    elif method == "sum":
        ratio_df = df[include_cols].groupby(category_cols).sum()
        if pct:
            single_col = include_cols[-1] + " Percentage"
            ratio_df[single_col] = ((df[include_cols].groupby(category_cols).sum() / 
                df[include_cols].groupby(category_cols).sum().sum()) * 100)
    return ratio_df
    


def feature_by_geo(df, geo, expl_var, num_var, method = "median"):
    '''
    Evaluate specific features by geography (ex: zip code)
    
    Inputs:
        df: (dataframe) pandas dataframe
        geo: (string) column name corresponding to geography used for grouping
        expl_var: (string) column name for exploratory variable used for 
            grouping
        num_var: (string) column name for numeric variable/ feature to be 
            aggregated
        method: (string) groupby aggregation method for column values

    Output:
        geo_features: pandas data frame of grouped data
    '''
    df_geo = df[(df[geo] != 0)]
    groupby_list = [geo] + expl_var
    if method == "median":
        geo_features = df_geo.groupby(groupby_list)[num_var].median().unstack(level = 1)
    if method == "count":
        geo_features = df_geo.groupby(groupby_list)[num_var].count().unstack(level = 1)
    geo_features.fillna(value = "", inplace = True)
    return geo_features



def plot_top_distros(train_test_tuples, var_dict, set_num):
    for i, col in enumerate(var_dict['tops']):
        train, test = train_test_tuples[set_num]
        plot_title = "Projects by {} for Training Set {}".format(col, set_num)
        train[col].value_counts().sort_index().plot(kind='bar', title = plot_title)
        plt.show()

