#!/usr/bin/env python
# coding: utf-8

# # Imports and Consts

# In[ ]:


"""
@author: abaglione and lihuacai

Credit to Tyler Spears and Sonia Baee, who developed the precursor
to this preprocessing script
"""

# imports
import sys
import os
import functools
import pathlib
import glob
import collections
import itertools
import re
import random
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

import numpy as np
import pandas as pd
import copy
import pipeline

from sklearn import impute
from sklearn import datasets
from sklearn import svm, linear_model, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import scipy
from scipy.spatial.distance import cdist

# visualization libraries
import matplotlib as mpl
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'figure.facecolor': [1.0, 1.0, 1.0, 1.0]})


# In[ ]:


ID_COL = 'pid'
TIMEDIV_COL = 'weekofstudy'
OUTCOME_COLS = ['anx', 'dep', 'anx_cat', 'dep_cat']
APPS = ['aspire', 'boostme', 'dailyfeats', 'icope', 'mantra', 'messages',
        'moveme', 'relax', 'slumbertime', 'socialforce', 'thoughtchallenger', 'worryknot']
ENGAGEMENT_METRIC_PREFIXES = ['frequency', 'daysofuse', 'duration', 'betweenlaunch']
TIMES_OF_DAY = ['morning', 'afternoon', 'evening', 'late_night']


# # Load Data

# In[ ]:


# Read in the weekly feature vectors 
all_feats = pd.read_csv('features/' + pipeline.APP_USERS_DIR + 'all_ind_wkly.csv')
all_feats.drop(axis=1,columns=['Unnamed: 0'] +                [col for col in all_feats.columns if 'trait' in col and 'group' in col],
               inplace=True)
all_feats['pid'] = all_feats['pid'].astype(str)

# Store the feature names
featnames = list(all_feats.columns)
all_feats.head(15)


# # Tidy Up DataFrame

# In[ ]:


# Drop columns of all NaNs
all_feats.dropna(axis=1, how='all', inplace=True)

'''Replace missing values in engagement metrics. 
   For metrics like frequency, days of use, and duration, we can fill with 0, since 
   no usage means 0 minutes / days / etc. 
   
   For time between launches, we will fill with some absurdly high number (for example,
   the number of seconds in a full day, 60*60*24).
   This helps to distinguish from situations such as this: 
       Between launch duration is 0, indicating rapidly opening one app right after another.''' 

zerofill_cols = [col for col in all_feats.columns 
                 if any([prefix in col for prefix in ENGAGEMENT_METRIC_PREFIXES])
                 and 'betweenlaunch' not in col
                ]
all_feats[zerofill_cols] = all_feats[zerofill_cols].fillna(0)
print(all_feats[zerofill_cols])

highfill_cols = [col for col in all_feats.columns if 'betweenlaunch' in col]
all_feats[highfill_cols] = all_feats[highfill_cols].fillna(60*60*24)
print(all_feats[highfill_cols])


# # Add Classification Outcomes

# In[ ]:


''' Note to self: made these both adhere to the same threshold, since they had the same scale
    Not sure why depression was >=4 but anxiety was >=3, originally.'''

all_feats['dep_cat'] = np.where(all_feats['dep'] >= 4, 1, 0)
all_feats['anx_cat'] = np.where(all_feats['anx'] >= 4, 1, 0)


# In[ ]:


all_feats['dep_cat'].value_counts()


# In[ ]:


all_feats['anx_cat'].value_counts()


# # Select Features

# In[ ]:


# Survey Features Only
survey_fs_cols = ['cope_alcohol_tob', 'physical_pain', 'connected', 'receive_support', 'active',
                  'support_others', 'healthy_food']

# Get a list of columns indicating which app(s) were used the most often
mua_dummies = [col for col in all_feats.columns if 'most_used_app' in col]

# App Features - Aggregate, Across All Apps
app_overall_fs_cols = ['frequency', 'daysofuse', 'duration', 'duration_mean',
                       'duration_std', 'duration_min', 'duration_max', 'betweenlaunch_duration_mean',
                       'betweenlaunch_duration_std']

# App Features - From Individual Apps
app_ind_fs_cols = [col for col in all_feats.columns
     if any([app in col for app in APPS])
     and any([prefix in col for prefix in ENGAGEMENT_METRIC_PREFIXES])
     and not any([tod in col for tod in TIMES_OF_DAY])
     ]

app_ind_fs_cols[0:5]


# In[ ]:


# Create a subset with survey features + app features from only the most used apps
mua_dfs = []
MUA_PREFIX = 'most_used_app_' # One-hot encoded/dummitized columns indicating most used app(s)
mua_dummies = [col for col in all_feats.columns if MUA_PREFIX in col]

# First, drop the aggregate engagement feature columns
survey_app_mua_feats = all_feats.drop(columns=app_overall_fs_cols)    

# Now, iterate through the dataframe. For each observation (row):
for i in range(all_feats.shape[0]):

    # Get the current row as a dataframe
    df = survey_app_mua_feats.iloc[[i]]

    # Find the most used apps - retain only the first one
    df2 = df[mua_dummies]
    most_used_app_cols = list(df2.columns[(df2 == 1).any(axis=0)])
    mua = [col.replace(MUA_PREFIX, '') for col in most_used_app_cols][0]
     
    '''Eliminate individual app columns that aren't for the most used app
       However, retain the "most_used_app" dummitized columns! ''' 
    df2 = df.drop(columns = [col for col in survey_app_mua_feats.columns 
                             if mua not in col
                             and MUA_PREFIX not in col
                             and any([app in col for app in APPS])])

    ''' Remove the name of the most used app from all columns EXCEPT
        the dummitized "most_used_app" columns. This enables a clean pd.concat later on.'''
    df2.rename(mapper=lambda x: x.replace('_' + mua, '') if MUA_PREFIX not in x else x,
               axis=1, inplace=True)

    ''' Finally, set all other dummitized "most_used_app" columns to 0, since we are 
        creating separate dfs for each "most used app"
    '''
    mua_dummies_subset = list(set(mua_dummies) - set([MUA_PREFIX + mua]))
    df2[mua_dummies_subset] = 0
    mua_dfs.append(df2)

# Replace the temp dataframe with a concat of all the individual row dfs
# This is our final df
survey_app_mua_feats = pd.concat(mua_dfs, sort=False)
survey_app_mua_feats


# In[ ]:


# Other mods / additions
app_mua_fs_cols = app_overall_fs_cols.copy()
app_overall_fs_cols += ['num_apps_used']
app_overall_fs_cols


# In[ ]:


# Create dictionary of featuresets
featuresets = {
    'app_overall_fs': app_overall_fs_cols,
    'app_mua_fs': app_mua_fs_cols,
    'survey_app_overall_fs': survey_fs_cols+app_overall_fs_cols, 
    'survey_app_mua_fs': survey_fs_cols+app_mua_fs_cols
}


# In[ ]:


######regression tasks on 1-5 scale (cut off on both 1 (floor) and 5 (ceiling)) using lasso linear mixed effect model;
# TODO - change so not passing in whole df every time
alpha_list = np.arange(0.1, 0.81, 0.1)
lmm_res = []

for alpha in alpha_list:
    print('alpha: {0}'.format(alpha))
    for fs_name, fs_cols in featuresets.items():
        print(fs_name)
        exp_feats = [TIMEDIV_COL, 'intercept'] + fs_cols
        
        ''' Handle the special case in which we need to reference the dataframe with
        data from only the most used app(s) for each observation '''
        if 'mua' in fs_name:
            df = survey_app_mua_feats
        else:
            df = all_feats
        
        # Add the intercept column
        df['intercept'] = 1
        
        # Save a copy in case we need to reference this later
        df.to_csv('features/%s.csv' % fs_name)
        
        # Make predictions for each target
        for target_col in ['anx', 'dep']:
            
            print(target_col)
            # Subset the data so we only impute what we need
            df2 = df[[ID_COL] + exp_feats + [target_col]].copy()
            print(df2)
            res = pipeline.genMixedLM(df=df2, outvar=target_col, 
                                      expfeats=exp_feats,
                                      gpvar=ID_COL, fsLabel=fs_name, alpha=alpha)
            res.to_csv('results/lmm_res.csv', mode='a', index=False)
            lmm_res.append(res)
            
lmm_res = pd.concat(lmm_res, copy=True, ignore_index=True, sort=False)
lmm_res.to_csv('results/lmm_res.csv', index=False)


# In[ ]:


# Create updated dictionary of featuresets
featuresets = {
    'survey_app_overall_fs': survey_fs_cols+app_overall_fs_cols, 
    'survey_app_ind_fs': survey_fs_cols+app_ind_fs_cols,
    'survey_app_mua_fs': survey_fs_cols+app_mua_fs_cols
}


# In[ ]:


# Prediction
targets = {
    'depression': 'dep_cat'
}

for fs_name, fs_cols in featuresets.items():
    print(fs_name)
    if 'app' in fs_name:
        if 'mua' not in fs_name:
            df = all_feats
        else:
            # Handle special cases in which we want data only from the most used app
            df = survey_app_mua_feats

        for target_name, target_col in targets.items():  
            
            # Drop rows where target is NaN - should never impute these!
            df2 = df.dropna(subset=[target_col], how='any')
            X = df2[[ID_COL, TIMEDIV_COL] + fs_cols]
            
            # Ensure not including any duplicate columns
            X = X.loc[:,~X.columns.duplicated()]
            print(X)
            y = df2[target_col]

            ''' If this is a featureset with app features 
                Get a list of one-hot-encoded columns from the most_used_app feature.'''

            # Get categorical feature indices - will be used with SMOTENC later
            nominal_idx = [X.columns.get_loc(ID_COL)]

            for method in ['LogisticR', 'RF', 'XGB']:
                pipeline.classifyMood(X=X, y=y, id_col=ID_COL, target=target_name,
                                     nominal_idx = nominal_idx, fs=fs_name, method=method,
                                     optimize=False)
                pipeline.classifyMood(X=X, y=y, id_col=ID_COL, target=target_name,
                                      nominal_idx = nominal_idx, fs=fs_name, method=method,
                                      optimize=True)


# In[ ]:


from sklearn import metrics
sorted(metrics.SCORERS.keys())

