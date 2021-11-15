# imports
import sys
import os
import collections
import itertools
import numpy as np
import pandas as pd
from pipeline import TRAIT_AFFECT_GROUPS, EPOCHS

def match_value(df, col1, x, col2):
    """ 
    Match value x from col1 row to value in col2.
    (StackOverflow)
    
    Args:
        df (DataFrame): A pandas DataFrame object
        col1 (str): The column to search
        x: The value to match in col1
        col2 (str): 
        
    Returns:
        The value from col2, from the first matched item
    
    """
    filtered = df[df[col1] == x]
    if filtered.empty:
        return np.nan
    else:
        return filtered[col2].values[0]
    

def find_week_by_timestamp(timestamps_df, pid, timestamp):
    """ 
    Find the week of the study (i.e. "Week 1") associated with a given timestamp, 
    for a single participant
    
    Args:
        timestamps_df (DataFrame): A pandas DataFrame object; 
            contains one row per participant and timestamp, with the week of study denoted
            
        pid (obj): The participant id for whom we need a labeled week
        timestamp: The timestamp to search for
        
    Returns:
        int: The week number
    """
    filtered = timestamps_df[(timestamps_df['pid'] == pid) & 
                  ((timestamp >= timestamps_df['date']) & (timestamp < (timestamps_df['date'] + pd.offsets.Day(7))))]
    if filtered.empty:
        return np.nan
    else:
        return filtered['week'].values[0]

    
# --------- LEGACY FUNCTION --------- 
def add_scores(df):
    """
    Include columns for baseline trait affect scores (anxiety and depression) in a df
    To be used in correlation plots
    
    Args:
        df (DataFrame): A pandas DataFrame object
    
    """
    for score in pipeline.bl_scores:
        df[score] = df['pid'].apply(
            lambda x: pipeline.match_value(master_featureset['all_all'], 'pid', x, score))

    # df2.drop('app_launches', axis=1,inplace=True)
    
    df.to_csv('features/app_all_epoch_ind_applevel.csv')
    
# ------------------------------------

def reset_index(df):
    """
    An excellent workaround for resetting categorical indices
    
    Credit to @akielbowicz
    https://github.com/pandas-dev/pandas/issues/19136#issuecomment-380908428
    
    Args:
        df (DataFrame): A pandas DataFrame object
    
    Returns: 
        A pandas DataFrame with index as columns
    """
    index_df = df.index.to_frame(index=False)
    df = df.reset_index(drop=True)
    # In merge is important the order in which you pass the dataframes
    # if the index contains a Categorical. 
    # pd.merge(df, index_df, left_index=True, right_index=True) does not work
    return pd.merge(index_df, df, left_index=True, right_index=True)

# --------- LEGACY FUNCTION --------- 
def weekly_epoch_breakdown(df, metric, merge_cols):
    
    """
    Create long-form (?) version of weekly feature vectors
    Useful for graphing
        
    Args:
        df (DataFrame): A pandas DataFrame object
        metric (str): The name of the metric (e.g. 'duration' or 'frequency')
        merge_cols ([str]): A list of common columns used for merging in future operations
            (e.g. the participant id column, or the week of study column); 
            these should never be renamed!
    Returns:
        df (Dataframe): The pandas dataframe, in long-form (?)
    
    """
    
    # Expand using reset_index workaround above
    df = df.unstack()
    df = reset_index(df)
   
     # Custom reordering of epochs to be consistent with times of the day
    df = df[merge_cols + EPOCHS['labels'][1:] + [EPOCHS['labels'][0]]]
    
    # Append metric label to each epoch column 
    #  (e.g. "morning_frequency")
    df.rename(columns=lambda x: x.lower().replace(" ", ""), inplace=True)
    df.rename(columns=lambda x: x +'_' + metric if x not in merge_cols else x, inplace=True)
    
    return df

# ------------------------------------

def calc_duration_noepoch(app_launch, groupbycols):
    """
    Calculate measurements specific to duration (sum, mean, std, etc) for a dataframe
    that has not been divided into epochs. 
        
    Args:
        app_launch (DataFrame): A pandas DataFrame object; contains one entry per participant
            per timestamped app launch
        groupbycols ([str]): The names of the columns by which to group, for each calculation
            (e.g. the participant id column)
    
    """
    
    to_merge = []

    df = app_launch.groupby(groupbycols)['duration'].sum().reset_index(name='duration')

    df2 = app_launch.groupby(groupbycols)['duration'].mean().reset_index(name='duration_mean')
    to_merge.append(df2)

    df2 = app_launch.groupby(groupbycols)['duration'].std().reset_index(name='duration_std')
    to_merge.append(df2)

    df2 = app_launch.groupby(groupbycols)['duration'].min().reset_index(name='duration_min')
    to_merge.append(df2)

    df2 = app_launch.groupby(groupbycols)['duration'].max().reset_index(name='duration_max')
    to_merge.append(df2)
    
    df2 = app_launch.sort_values('date').groupby(groupbycols)['date'].apply(
        lambda x: x.diff().mean().total_seconds()
    ).reset_index(name="betweenlaunch_duration_mean")
    to_merge.append(df2)
    
    df2 = app_launch.sort_values('date').groupby(groupbycols)['date'].apply(
        lambda x: x.diff().std().total_seconds()
    ).reset_index(name="betweenlaunch_duration_std")
    to_merge.append(df2)

    for df_to_merge in to_merge:
        df = pd.merge(df, df_to_merge, on=list(df_to_merge.columns[:-1]), how="outer")
        
    return df

def calc_duration_has_epoch(app_launch, groupbycols):
    
    merge_cols = groupbycols[:-1]
 
    df = app_launch.groupby(groupbycols)['duration'].sum()
    res = weekly_epoch_breakdown(df, 'duration', merge_cols)

    df = app_launch.groupby(groupbycols)['duration'].mean()
    df = weekly_epoch_breakdown(df, 'duration_mean', merge_cols)

    res = pd.merge(
        res, 
        df, 
        on = merge_cols,
        how="outer"
    )

    df = app_launch.groupby(groupbycols)['duration'].std()
    df = weekly_epoch_breakdown(df, 'duration_std', merge_cols)
    res = pd.merge(
        res, 
        df, 
        on = merge_cols,
        how="outer"
    )

    df = app_launch.groupby(groupbycols)['duration'].min()
    df = weekly_epoch_breakdown(df, 'duration_min', merge_cols)
    res = pd.merge(
        res, 
        df, 
        on = merge_cols,
        how="outer"
    )

    df = app_launch.groupby(groupbycols)['duration'].max()
    df = weekly_epoch_breakdown(df, 'duration_max', merge_cols)
    res = pd.merge(
        res, 
        df, 
        on = merge_cols,
        how="outer"
    )
    
    df = app_launch.groupby(groupbycols)['date'].apply(
            lambda x: x.diff().mean().total_seconds()
    )
    df = df.round(0)
    df = weekly_epoch_breakdown(df, 'betweenlaunch_duration_mean', merge_cols) 
    res = pd.merge(
        res, 
        df, 
        on = merge_cols,
        how="outer"
    )

    df = app_launch.sort_values('date').groupby(groupbycols)['date'].apply(
            lambda x: x.diff().std().total_seconds()
    )
    df = weekly_epoch_breakdown(df, 'betweenlaunch_duration_std', merge_cols)   
    res = pd.merge(
        res, 
        df, 
        on = merge_cols,
        how="outer"
    )
    
    return res

def construct_feature_vectors(app_launch, wklysurvey, timediv):
    vectors = None
    timediv_col = None
    
    # Get a list of apps over which to iterate
    apps = list(app_launch['package'].unique())
    filepath = 'features/app_users_only/'
    
    if timediv == 'daily':
        # Let our daily individual aggregate features be our starting dataframe
        vectors = pd.read_csv('features/app_dly_ind_agg.csv')  
        
        # Specify the column for our time division (e.g., daily or weekly)
        timediv_col = 'dayofstudy'
        
        # Specify the files containing applevel features
        # NOTE: Had problem last time with "for each" with only one list entry...hmm
        applevel_featurefiles = [filepath + 'app_dly_ind_applevel.csv']
        
    elif timediv == 'wkly':
        # Let our weekly individual aggregate features be our starting dataframe
        vectors = pd.read_csv(filepath + 'wkly_agg.csv')
        timediv_col = 'weekofstudy'
        applevel_featurefiles = [filepath + 'wkly_applevel.csv', filepath + 'wkly_epoch_applevel.csv']
        
    # for pid in list(vectors['pid'].unique()):
#     for week in range(2, 8):
#         series = pd.Series([pid, week], index = vectors.columns)
#         vectors = vectors.append(series, ignore_index=True)

    # Implement a sorting scheme, to more easily visualize participants' progression
    #   through the study
    vectors = vectors.sort_values(by=['pid', timediv_col])
    vectors.drop(columns=['Unnamed: 0'], inplace=True)
    
    # Obtain the day and week each app launch occured
    df = app_launch[['pid', 'dayofstudy','weekofstudy']]

    # epochO: Fix this merge - introduces duplicates :( which need to be manually removed in Excel
    vectors = pd.merge(vectors, df, on=['pid', timediv_col], how="left")
    
    # Remove any entries that fall outside the study window
    vectors = vectors[vectors['dayofstudy'] > 0]
    
    # Create new column for each measure, for each app
    # For instance, we want to add a column for the "Worryknot" app's total frequency of use
    #   We also want to add a column for the "Worryknot" app's frequency of use for each epoch
    # Note that we also include the file with epoch- and app-specific metrics.
    #   For instance, we want to add a column for the "worryknot" app's total frequency for the morning
    to_merge = []
    for f in applevel_featurefiles:
        df = pd.read_csv(f)
        df.drop(columns=['Unnamed: 0'], inplace=True)

        for app in apps:
            filtered = df[df['package'] == app].drop(columns=['package'])
            metric_cols = [col for col in filtered.columns if col not in ['pid', timediv_col]]
            for col in metric_cols:
                filtered.rename(index=str, columns={col: col + '_' +  app}, inplace=True)
            to_merge.append(filtered)

    for df_to_merge in to_merge:
        vectors = pd.merge(vectors, df_to_merge, on=['pid', timediv_col], how="left")
       
    # Calculate total number of apps used during time division (e.g., per week), per user
    df = app_launch.groupby(['pid', timediv_col])['package'].nunique().reset_index(name='num_apps_used')
    vectors = pd.merge(vectors, df, on=['pid', timediv_col], how="left")

    # Find the name(s) of the most used app(s) each day, per user
    df = app_launch.groupby(['pid', timediv_col])['package'].agg(pd.Series.mode).reset_index(name='most_used_app')

    # Create one column per most-used app (descending order)
    df = pd.concat([df, df['most_used_app'].apply(pd.Series).add_prefix('most_used_app_')], axis = 1)
    
    # No longer need the column we started with
    df.drop(columns=['most_used_app'], inplace=True)
    vectors = pd.merge(vectors, df, on=['pid', timediv_col], how="left")
    
    # Finally, let's add survey features
    # For daily features, daily survey features such as mood scores will be associated with the single weekly feature
    df = wklysurvey[['pid', 'weekofstudy', 'cope_alcohol_tob', 'physical_pain', 'connected', 'receive_support', 'support_others', 'active', 'healthy_food']]
    
    vectors = pd.merge(vectors, df, on=['pid', 'weekofstudy'], how="left")
    
    return vectors
        
    