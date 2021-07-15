#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 20:14:41 2020

@author: lihuacai
"""

#https://www.statsmodels.org/dev/generated/statsmodels.regression.mixed_linear_model.MixedLM.html
#https://www.statsmodels.org/stable/generated/statsmodels.regression.mixed_linear_model.MixedLM.fit_regularized.html
#https://www.statsmodels.org/stable/generated/statsmodels.regression.mixed_linear_model.MixedLMResults.html#statsmodels.regression.mixed_linear_model.MixedLMResults



import pandas as pd
import numpy as np
import copy
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from statsmodels.regression.mixed_linear_model import MixedLM
from sklearn import linear_model
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import SMOTE
import xgboost
import shap
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


small_size = 8
medium_size = 14
bigger_size = 15

plt.rc('font', size=medium_size)          # controls default text sizes
plt.rc('axes', titlesize=medium_size)     # fontsize of the axes title
plt.rc('axes', labelsize=medium_size)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=medium_size)    # fontsize of the tick labels
plt.rc('ytick', labelsize=medium_size)    # fontsize of the tick labels
plt.rc('legend', fontsize=medium_size)    # legend fontsize
plt.rc('figure', titlesize=bigger_size)  # fontsize of the figure title


pd.options.display.float_format = "{:.2f}".format
pd.set_option('display.max_columns', 50)

# weekdf = pd.read_csv('/Users/lihuacai/Desktop/all_ind_wkly.csv')
# daydf = pd.read_csv('/Users/lihuacai/Desktop/all_ind_dly.csv')
# visdf = pd.read_csv('/Users/lihuacai/Desktop/app_launch_processed.csv')

weekdf = pd.read_csv('/mnt/c/Users/anbag/code/breast_cancer/features/all_ind_wkly.csv')
# daydf = pd.read_csv('/mnt/c/Users/anbag/code/breast_cancer/features/all_ind_dly.csv')
visdf = pd.read_csv('/mnt/c/Users/anbag/code/breast_cancer/data/processed/app_launch_processed.csv')

weekdf.drop(axis=1,columns=['Unnamed: 0'],inplace=True)
# daydf.drop(axis=1,columns=['Unnamed: 0'],inplace=True)

weekdf['pid'] = weekdf['pid'].astype(str)
# daydf['pid'] = daydf['pid'].astype(str)
visdf['pid'] = visdf['pid'].astype(str)

visdf['date'] = pd.to_datetime(visdf['date'])
visdf['duration'] = [pd.Timedelta(seconds=i) for i in visdf['duration']]
visdf['edate'] = visdf['date'] + visdf['duration']
visdf = visdf.loc[visdf['duration']>pd.Timedelta(seconds=0)].copy()


# dvarnames = list(daydf.columns)
wvarnames = list(weekdf.columns)


#MixedLM.fit_regularized(start_params=None, method='l1', alpha=0, ceps=0.0001, ptol=1e-06, maxit=200, **fit_kwargs)

#for i,col in enumerate(weekdf.columns):
#    if i >= 600 and i < 700:
#        print("'{0}',".format(col))
        
#frequency -- number of app launches
#daysofuse -- number of time unit (e.g., day, time bucket) app(s) being launched
#duration -- total
        

#class imbalance, missing data (especially response var)
#https://scikit-learn.org/stable/modules/impute.html#:~:text=Missing%20values%20can%20be%20imputed,for%20different%20missing%20values%20encodings.&text=%3E%3E%3E%20import%20numpy%20as%20np%20%3E%3E%3E%20from%20sklearn.
#https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html

def fillmissing(df,cols,val,fillval):
    for c in cols:
        df[c].loc[df[c] == val] = fillval

#imputing missing survey data -- weekly data
impweekvars = ['cope_alcohol_tob','physical_pain','connected','receive_support','anx',
               'dep','active','support_others','healthy_food']
wvarnames_app = [i for i in wvarnames if i not in impweekvars]
weekdf_app = weekdf.drop(axis=1,columns=impweekvars)
weekdf_napp = weekdf[impweekvars].copy()
fillmissing(weekdf_napp,impweekvars,-1,np.nan)


imputor = IterativeImputer(max_iter=50,random_state=1008,add_indicator=True)
imputor.fit(weekdf_napp)
impweekvars_ind = [i+'_ind' for i in impweekvars]
impweekvars_c = copy.deepcopy(impweekvars)
impweekvars_c.extend(impweekvars_ind)
weekdf_napp = pd.DataFrame(np.round(imputor.transform(weekdf_napp)),columns=impweekvars_c) 

weekdf_imp = pd.concat([weekdf_app,weekdf_napp],copy=True,axis=1)

# #imputing missing survey data -- daily data
# impdayvars = ['pid','dayofstudy','weekofstudy',
#               'cope_alcohol_tob','physical_pain','connected','receive_support',
#               'active','support_others','healthy_food','anx','dep']
# dvarnames_app = [i for i in dvarnames if i not in impdayvars]
# daydf_app = daydf.drop(axis=1,columns=impdayvars)
# daydf_napp = daydf[['pid','dayofstudy','weekofstudy']].copy()

weekdf_imp_c = weekdf_imp[['pid','weekofstudy']+impweekvars_c].copy()
# daydf_napp_imp = daydf_napp.merge(weekdf_imp_c,on=['pid','weekofstudy'])
#daydf_napp_imp.drop(axis=1,columns=['anx_x','dep_x'],inplace=True)
#daydf_napp_imp.rename(columns={'anx_y':'anx','dep_y':'dep'},inplace=True)

# daydf_imp = pd.concat([daydf_napp_imp,daydf_app],copy=True,axis=1)

#handling missing values in app features
#apps = ['aspire','boostme','dailyfeats','icope','mantra','messages','moveme','relax',
#        'slumbertime','socialforce','thoughtchallenger','worryknot']
#timebuckets = ['afternoon','evening','latenight','morning']
#featuretypes = ['frequency','daysofuse','duration','betweenlaunch_duration']

# frequency_dvars = [n for n in dvarnames if 'frequency' in n]
# reg_dvars = [n for n in dvarnames if 'daysofuse' in n]
# dur_dvars = [n for n in dvarnames if 'duration' in n and 'betweenlaunch' not in n]
# lau_dur_dvars = [n for n in dvarnames if 'betweenlaunch' in n]

# fillmissing(daydf_imp,frequency_dvars,-1,0)
# fillmissing(daydf_imp,reg_dvars,-1,0)
# fillmissing(daydf_imp,dur_dvars,-1,0)
# fillmissing(daydf_imp,lau_dur_dvars,-1,3600*24)

frequency_wvars = [n for n in wvarnames if 'frequency' in n]
reg_wvars = [n for n in wvarnames if 'daysofuse' in n]
dur_wvars = [n for n in wvarnames if 'duration' in n and 'betweenlaunch' not in n]
lau_dur_wvars = [n for n in wvarnames if 'betweenlaunch' in n]

fillmissing(weekdf_imp,frequency_wvars,-1,0)
fillmissing(weekdf_imp,reg_wvars,-1,0)
fillmissing(weekdf_imp,dur_wvars,-1,0)
fillmissing(weekdf_imp,lau_dur_wvars,-1,3600*24*7)

#add the intercept columns for the linear mixed model
weekdf_imp['intercept'] = 1
# daydf_imp['intercept'] = 1

#outcomes transformation -- anx, dep
#week to week change as outcome
#change to baseline level as outcome
#instead of difference, consider ratio between the weekly value and the baseline
#global average being subtracted

# daydf_imp['anx'].hist()
# daydf_imp['dep'].hist()

weekdf_imp['anx'].hist()
weekdf_imp['dep'].hist()

#add classification outcomes
# daydf_imp['dep_cat'] = np.where(daydf_imp['dep']>=4,1,0)
# daydf_imp['anx_cat'] = np.where(daydf_imp['anx']>=3,1,0)
weekdf_imp['dep_cat'] = np.where(weekdf_imp['dep']>=4,1,0)
weekdf_imp['anx_cat'] = np.where(weekdf_imp['anx']>=3,1,0)

outcomes = ['anx','dep','anx_cat','dep_cat']


#feature sets
#overall app usage; break down by apps; break down by apps and time windows; only most used app (each person will be different); all features
sr_fs = ['cope_alcohol_tob','physical_pain','connected','receive_support','active','support_others','healthy_food',
          'cope_alcohol_tob_ind','physical_pain_ind','connected_ind','receive_support_ind','active_ind','support_others_ind','healthy_food_ind']
app_overall_fs = ['intercept','weekofstudy','frequency','daysofuse','duration','duration_mean',
                  'duration_std','duration_min','duration_max','betweenlaunch_duration_mean','betweenlaunch_duration_std',
                  'num_apps_used'] #,'aspire', 'boostme', 'dailyfeats', 'icope', 'mantra', 'messages',
                  #'moveme', 'relax', 'slumbertime', 'thoughtchallenger', 'worryknot']
app_ind_fs = ['intercept','weekofstudy','frequency_aspire','daysofuse_aspire','duration_aspire','duration_mean_aspire',
              'duration_std_aspire','duration_min_aspire','duration_max_aspire','betweenlaunch_duration_mean_aspire',
              'betweenlaunch_duration_std_aspire','frequency_boostme','daysofuse_boostme','duration_boostme',
              'duration_mean_boostme','duration_std_boostme','duration_min_boostme','duration_max_boostme',
              'betweenlaunch_duration_mean_boostme','betweenlaunch_duration_std_boostme','frequency_dailyfeats',
              'daysofuse_dailyfeats','duration_dailyfeats','duration_mean_dailyfeats','duration_std_dailyfeats',
              'duration_min_dailyfeats','duration_max_dailyfeats','betweenlaunch_duration_mean_dailyfeats',
              'betweenlaunch_duration_std_dailyfeats','frequency_icope','daysofuse_icope','duration_icope',
              'duration_mean_icope','duration_std_icope','duration_min_icope','duration_max_icope',
              'betweenlaunch_duration_mean_icope','betweenlaunch_duration_std_icope','frequency_mantra','daysofuse_mantra',
              'duration_mantra','duration_mean_mantra','duration_std_mantra','duration_min_mantra','duration_max_mantra',
              'betweenlaunch_duration_mean_mantra','betweenlaunch_duration_std_mantra','frequency_messages',
              'daysofuse_messages','duration_messages','duration_mean_messages','duration_std_messages',
              'duration_min_messages','duration_max_messages','betweenlaunch_duration_mean_messages',
              'betweenlaunch_duration_std_messages','frequency_moveme','daysofuse_moveme','duration_moveme',
              'duration_mean_moveme','duration_std_moveme','duration_min_moveme','duration_max_moveme',
              'betweenlaunch_duration_mean_moveme','betweenlaunch_duration_std_moveme','frequency_relax',
              'daysofuse_relax','duration_relax','duration_mean_relax','duration_std_relax','duration_min_relax',
              'duration_max_relax','betweenlaunch_duration_mean_relax','betweenlaunch_duration_std_relax',
              'frequency_slumbertime','daysofuse_slumbertime','duration_slumbertime','duration_mean_slumbertime',
              'duration_std_slumbertime','duration_min_slumbertime','duration_max_slumbertime',
              'betweenlaunch_duration_mean_slumbertime','betweenlaunch_duration_std_slumbertime','frequency_socialforce',
              'daysofuse_socialforce','duration_socialforce','duration_mean_socialforce','duration_std_socialforce',
              'duration_min_socialforce','duration_max_socialforce','betweenlaunch_duration_mean_socialforce',
              'betweenlaunch_duration_std_socialforce','frequency_thoughtchallenger','daysofuse_thoughtchallenger',
              'duration_thoughtchallenger','duration_mean_thoughtchallenger','duration_std_thoughtchallenger',
              'duration_min_thoughtchallenger','duration_max_thoughtchallenger','betweenlaunch_duration_mean_thoughtchallenger',
              'betweenlaunch_duration_std_thoughtchallenger','frequency_worryknot','daysofuse_worryknot',
              'duration_worryknot','duration_mean_worryknot','duration_std_worryknot','duration_min_worryknot',
              'duration_max_worryknot','betweenlaunch_duration_mean_worryknot','betweenlaunch_duration_std_worryknot']
app_ind_tw_fs = ['intercept','weekofstudy','afternoon_frequency_aspire','evening_frequency_aspire','latenight_frequency_aspire',
                 'morning_frequency_aspire','latenight_daysofuse_aspire','morning_daysofuse_aspire',
                 'afternoon_daysofuse_aspire','evening_daysofuse_aspire','latenight_duration_aspire',
                 'morning_duration_aspire','afternoon_duration_aspire','evening_duration_aspire',
                 'latenight_duration_mean_aspire','morning_duration_mean_aspire','afternoon_duration_mean_aspire',
                 'evening_duration_mean_aspire','latenight_duration_std_aspire','morning_duration_std_aspire',
                 'afternoon_duration_std_aspire','evening_duration_std_aspire','latenight_duration_min_aspire',
                 'morning_duration_min_aspire','afternoon_duration_min_aspire','evening_duration_min_aspire',
                 'latenight_duration_max_aspire','morning_duration_max_aspire','afternoon_duration_max_aspire',
                 'evening_duration_max_aspire','latenight_betweenlaunch_duration_mean_aspire',
                 'morning_betweenlaunch_duration_mean_aspire','afternoon_betweenlaunch_duration_mean_aspire',
                 'evening_betweenlaunch_duration_mean_aspire','latenight_betweenlaunch_duration_std_aspire',
                 'morning_betweenlaunch_duration_std_aspire','afternoon_betweenlaunch_duration_std_aspire',
                 'evening_betweenlaunch_duration_std_aspire','afternoon_frequency_boostme','evening_frequency_boostme',
                 'latenight_frequency_boostme','morning_frequency_boostme','latenight_daysofuse_boostme',
                 'morning_daysofuse_boostme','afternoon_daysofuse_boostme','evening_daysofuse_boostme',
                 'latenight_duration_boostme','morning_duration_boostme','afternoon_duration_boostme',
                 'evening_duration_boostme','latenight_duration_mean_boostme','morning_duration_mean_boostme',
                 'afternoon_duration_mean_boostme','evening_duration_mean_boostme','latenight_duration_std_boostme',
                 'morning_duration_std_boostme','afternoon_duration_std_boostme','evening_duration_std_boostme',
                 'latenight_duration_min_boostme','morning_duration_min_boostme','afternoon_duration_min_boostme',
                 'evening_duration_min_boostme','latenight_duration_max_boostme','morning_duration_max_boostme',
                 'afternoon_duration_max_boostme','evening_duration_max_boostme','latenight_betweenlaunch_duration_mean_boostme',
                 'morning_betweenlaunch_duration_mean_boostme','afternoon_betweenlaunch_duration_mean_boostme',
                 'evening_betweenlaunch_duration_mean_boostme','latenight_betweenlaunch_duration_std_boostme',
                 'morning_betweenlaunch_duration_std_boostme','afternoon_betweenlaunch_duration_std_boostme',
                 'evening_betweenlaunch_duration_std_boostme','afternoon_frequency_dailyfeats','evening_frequency_dailyfeats',
                 'latenight_frequency_dailyfeats','morning_frequency_dailyfeats','latenight_daysofuse_dailyfeats',
                 'morning_daysofuse_dailyfeats','afternoon_daysofuse_dailyfeats','evening_daysofuse_dailyfeats',
                 'latenight_duration_dailyfeats','morning_duration_dailyfeats','afternoon_duration_dailyfeats',
                 'evening_duration_dailyfeats','latenight_duration_mean_dailyfeats','morning_duration_mean_dailyfeats',
                 'afternoon_duration_mean_dailyfeats','evening_duration_mean_dailyfeats','latenight_duration_std_dailyfeats',
                 'morning_duration_std_dailyfeats','afternoon_duration_std_dailyfeats','evening_duration_std_dailyfeats',
                 'latenight_duration_min_dailyfeats','morning_duration_min_dailyfeats','afternoon_duration_min_dailyfeats',
                 'evening_duration_min_dailyfeats','latenight_duration_max_dailyfeats','morning_duration_max_dailyfeats',
                 'afternoon_duration_max_dailyfeats','evening_duration_max_dailyfeats','latenight_betweenlaunch_duration_mean_dailyfeats',
                 'morning_betweenlaunch_duration_mean_dailyfeats','afternoon_betweenlaunch_duration_mean_dailyfeats',
                 'evening_betweenlaunch_duration_mean_dailyfeats','latenight_betweenlaunch_duration_std_dailyfeats',
                 'morning_betweenlaunch_duration_std_dailyfeats','afternoon_betweenlaunch_duration_std_dailyfeats',
                 'evening_betweenlaunch_duration_std_dailyfeats','afternoon_frequency_icope','evening_frequency_icope',
                 'latenight_frequency_icope','morning_frequency_icope','latenight_daysofuse_icope','morning_daysofuse_icope',
                 'afternoon_daysofuse_icope','evening_daysofuse_icope','latenight_duration_icope','morning_duration_icope',
                 'afternoon_duration_icope','evening_duration_icope','latenight_duration_mean_icope',
                 'morning_duration_mean_icope','afternoon_duration_mean_icope','evening_duration_mean_icope','latenight_duration_std_icope',
                 'morning_duration_std_icope','afternoon_duration_std_icope','evening_duration_std_icope',
                 'latenight_duration_min_icope','morning_duration_min_icope','afternoon_duration_min_icope',
                 'evening_duration_min_icope','latenight_duration_max_icope','morning_duration_max_icope',
                 'afternoon_duration_max_icope','evening_duration_max_icope','latenight_betweenlaunch_duration_mean_icope',
                 'morning_betweenlaunch_duration_mean_icope','afternoon_betweenlaunch_duration_mean_icope',
                 'evening_betweenlaunch_duration_mean_icope','latenight_betweenlaunch_duration_std_icope',
                 'morning_betweenlaunch_duration_std_icope','afternoon_betweenlaunch_duration_std_icope',
                 'evening_betweenlaunch_duration_std_icope','afternoon_frequency_mantra','evening_frequency_mantra',
                 'latenight_frequency_mantra','morning_frequency_mantra','latenight_daysofuse_mantra','morning_daysofuse_mantra',
                 'afternoon_daysofuse_mantra','evening_daysofuse_mantra','latenight_duration_mantra',
                 'morning_duration_mantra','afternoon_duration_mantra','evening_duration_mantra','latenight_duration_mean_mantra',
                 'morning_duration_mean_mantra','afternoon_duration_mean_mantra','evening_duration_mean_mantra',
                 'latenight_duration_std_mantra','morning_duration_std_mantra','afternoon_duration_std_mantra',
                 'evening_duration_std_mantra','latenight_duration_min_mantra','morning_duration_min_mantra',
                 'afternoon_duration_min_mantra','evening_duration_min_mantra','latenight_duration_max_mantra',
                 'morning_duration_max_mantra','afternoon_duration_max_mantra','evening_duration_max_mantra',
                 'latenight_betweenlaunch_duration_mean_mantra','morning_betweenlaunch_duration_mean_mantra',
                 'afternoon_betweenlaunch_duration_mean_mantra','evening_betweenlaunch_duration_mean_mantra',
                 'latenight_betweenlaunch_duration_std_mantra','morning_betweenlaunch_duration_std_mantra',
                 'afternoon_betweenlaunch_duration_std_mantra','evening_betweenlaunch_duration_std_mantra',
                 'afternoon_frequency_messages','evening_frequency_messages','latenight_frequency_messages','morning_frequency_messages',
                 'latenight_daysofuse_messages','morning_daysofuse_messages','afternoon_daysofuse_messages','evening_daysofuse_messages',
                 'latenight_duration_messages','morning_duration_messages','afternoon_duration_messages','evening_duration_messages',
                 'latenight_duration_mean_messages','morning_duration_mean_messages','afternoon_duration_mean_messages','evening_duration_mean_messages',
                 'latenight_duration_std_messages','morning_duration_std_messages','afternoon_duration_std_messages','evening_duration_std_messages',
                 'latenight_duration_min_messages','morning_duration_min_messages','afternoon_duration_min_messages','evening_duration_min_messages',
                 'latenight_duration_max_messages','morning_duration_max_messages','afternoon_duration_max_messages',
                 'evening_duration_max_messages','latenight_betweenlaunch_duration_mean_messages','morning_betweenlaunch_duration_mean_messages',
                 'afternoon_betweenlaunch_duration_mean_messages','evening_betweenlaunch_duration_mean_messages',
                 'latenight_betweenlaunch_duration_std_messages','morning_betweenlaunch_duration_std_messages','afternoon_betweenlaunch_duration_std_messages',
                 'evening_betweenlaunch_duration_std_messages','afternoon_frequency_moveme','evening_frequency_moveme',
                 'latenight_frequency_moveme','morning_frequency_moveme','latenight_daysofuse_moveme',
                 'morning_daysofuse_moveme','afternoon_daysofuse_moveme','evening_daysofuse_moveme','latenight_duration_moveme',
                 'morning_duration_moveme','afternoon_duration_moveme','evening_duration_moveme','latenight_duration_mean_moveme',
                 'morning_duration_mean_moveme','afternoon_duration_mean_moveme','evening_duration_mean_moveme','latenight_duration_std_moveme',
                 'morning_duration_std_moveme','afternoon_duration_std_moveme','evening_duration_std_moveme','latenight_duration_min_moveme',
                 'morning_duration_min_moveme','afternoon_duration_min_moveme','evening_duration_min_moveme','latenight_duration_max_moveme',
                 'morning_duration_max_moveme','afternoon_duration_max_moveme','evening_duration_max_moveme','latenight_betweenlaunch_duration_mean_moveme',
                 'morning_betweenlaunch_duration_mean_moveme','afternoon_betweenlaunch_duration_mean_moveme','evening_betweenlaunch_duration_mean_moveme',
                 'latenight_betweenlaunch_duration_std_moveme','morning_betweenlaunch_duration_std_moveme','afternoon_betweenlaunch_duration_std_moveme',
                 'evening_betweenlaunch_duration_std_moveme','afternoon_frequency_relax','evening_frequency_relax','latenight_frequency_relax',
                 'morning_frequency_relax','latenight_daysofuse_relax','morning_daysofuse_relax','afternoon_daysofuse_relax',
                 'evening_daysofuse_relax','latenight_duration_relax','morning_duration_relax','afternoon_duration_relax',
                 'evening_duration_relax','latenight_duration_mean_relax','morning_duration_mean_relax','afternoon_duration_mean_relax',
                 'evening_duration_mean_relax','latenight_duration_std_relax','morning_duration_std_relax','afternoon_duration_std_relax',
                 'evening_duration_std_relax','latenight_duration_min_relax','morning_duration_min_relax','afternoon_duration_min_relax',
                 'evening_duration_min_relax','latenight_duration_max_relax','morning_duration_max_relax','afternoon_duration_max_relax',
                 'evening_duration_max_relax','latenight_betweenlaunch_duration_mean_relax','morning_betweenlaunch_duration_mean_relax',
                 'afternoon_betweenlaunch_duration_mean_relax','evening_betweenlaunch_duration_mean_relax','latenight_betweenlaunch_duration_std_relax',
                 'morning_betweenlaunch_duration_std_relax','afternoon_betweenlaunch_duration_std_relax','evening_betweenlaunch_duration_std_relax',
                 'afternoon_frequency_slumbertime','evening_frequency_slumbertime','latenight_frequency_slumbertime','morning_frequency_slumbertime',
                 'latenight_daysofuse_slumbertime','morning_daysofuse_slumbertime','afternoon_daysofuse_slumbertime',
                 'evening_daysofuse_slumbertime','latenight_duration_slumbertime','morning_duration_slumbertime',
                 'afternoon_duration_slumbertime','evening_duration_slumbertime','latenight_duration_mean_slumbertime','morning_duration_mean_slumbertime',
                 'afternoon_duration_mean_slumbertime','evening_duration_mean_slumbertime','latenight_duration_std_slumbertime',
                 'morning_duration_std_slumbertime','afternoon_duration_std_slumbertime','evening_duration_std_slumbertime',
                 'latenight_duration_min_slumbertime','morning_duration_min_slumbertime','afternoon_duration_min_slumbertime',
                 'evening_duration_min_slumbertime','latenight_duration_max_slumbertime','morning_duration_max_slumbertime',
                 'afternoon_duration_max_slumbertime','evening_duration_max_slumbertime','latenight_betweenlaunch_duration_mean_slumbertime',
                 'morning_betweenlaunch_duration_mean_slumbertime','afternoon_betweenlaunch_duration_mean_slumbertime','evening_betweenlaunch_duration_mean_slumbertime',
                 'latenight_betweenlaunch_duration_std_slumbertime','morning_betweenlaunch_duration_std_slumbertime','afternoon_betweenlaunch_duration_std_slumbertime',
                 'evening_betweenlaunch_duration_std_slumbertime','afternoon_frequency_socialforce','evening_frequency_socialforce',
                 'latenight_frequency_socialforce','morning_frequency_socialforce','latenight_daysofuse_socialforce',
                 'morning_daysofuse_socialforce','afternoon_daysofuse_socialforce','evening_daysofuse_socialforce',
                 'latenight_duration_socialforce','morning_duration_socialforce','afternoon_duration_socialforce',
                 'evening_duration_socialforce','latenight_duration_mean_socialforce','morning_duration_mean_socialforce',
                 'afternoon_duration_mean_socialforce','evening_duration_mean_socialforce','latenight_duration_std_socialforce',
                 'morning_duration_std_socialforce','afternoon_duration_std_socialforce','evening_duration_std_socialforce',
                 'latenight_duration_min_socialforce','morning_duration_min_socialforce','afternoon_duration_min_socialforce',
                 'evening_duration_min_socialforce','latenight_duration_max_socialforce','morning_duration_max_socialforce',
                 'afternoon_duration_max_socialforce','evening_duration_max_socialforce','latenight_betweenlaunch_duration_mean_socialforce',
                 'morning_betweenlaunch_duration_mean_socialforce','afternoon_betweenlaunch_duration_mean_socialforce','evening_betweenlaunch_duration_mean_socialforce',
                 'latenight_betweenlaunch_duration_std_socialforce','morning_betweenlaunch_duration_std_socialforce',
                 'afternoon_betweenlaunch_duration_std_socialforce','evening_betweenlaunch_duration_std_socialforce','afternoon_frequency_thoughtchallenger',
                 'evening_frequency_thoughtchallenger','latenight_frequency_thoughtchallenger','morning_frequency_thoughtchallenger',
                 'latenight_daysofuse_thoughtchallenger','morning_daysofuse_thoughtchallenger','afternoon_daysofuse_thoughtchallenger',
                 'evening_daysofuse_thoughtchallenger','latenight_duration_thoughtchallenger','morning_duration_thoughtchallenger',
                 'afternoon_duration_thoughtchallenger','evening_duration_thoughtchallenger','latenight_duration_mean_thoughtchallenger',
                 'morning_duration_mean_thoughtchallenger','afternoon_duration_mean_thoughtchallenger','evening_duration_mean_thoughtchallenger',
                 'latenight_duration_std_thoughtchallenger','morning_duration_std_thoughtchallenger','afternoon_duration_std_thoughtchallenger',
                 'evening_duration_std_thoughtchallenger','latenight_duration_min_thoughtchallenger','morning_duration_min_thoughtchallenger',
                 'afternoon_duration_min_thoughtchallenger','evening_duration_min_thoughtchallenger','latenight_duration_max_thoughtchallenger',
                 'morning_duration_max_thoughtchallenger','afternoon_duration_max_thoughtchallenger','evening_duration_max_thoughtchallenger',
                 'latenight_betweenlaunch_duration_mean_thoughtchallenger','morning_betweenlaunch_duration_mean_thoughtchallenger',
                 'afternoon_betweenlaunch_duration_mean_thoughtchallenger','evening_betweenlaunch_duration_mean_thoughtchallenger',
                 'latenight_betweenlaunch_duration_std_thoughtchallenger','morning_betweenlaunch_duration_std_thoughtchallenger',
                 'afternoon_betweenlaunch_duration_std_thoughtchallenger','evening_betweenlaunch_duration_std_thoughtchallenger',
                 'afternoon_frequency_worryknot','evening_frequency_worryknot','latenight_frequency_worryknot','morning_frequency_worryknot',
                 'latenight_daysofuse_worryknot','morning_daysofuse_worryknot','afternoon_daysofuse_worryknot','evening_daysofuse_worryknot',
                 'latenight_duration_worryknot','morning_duration_worryknot','afternoon_duration_worryknot','evening_duration_worryknot',
                 'latenight_duration_mean_worryknot','morning_duration_mean_worryknot','afternoon_duration_mean_worryknot',
                 'evening_duration_mean_worryknot','latenight_duration_std_worryknot','morning_duration_std_worryknot',
                 'afternoon_duration_std_worryknot','evening_duration_std_worryknot','latenight_duration_min_worryknot',
                 'morning_duration_min_worryknot','afternoon_duration_min_worryknot','evening_duration_min_worryknot','latenight_duration_max_worryknot',
                 'morning_duration_max_worryknot','afternoon_duration_max_worryknot','evening_duration_max_worryknot',
                 'latenight_betweenlaunch_duration_mean_worryknot','morning_betweenlaunch_duration_mean_worryknot','afternoon_betweenlaunch_duration_mean_worryknot',
                 'evening_betweenlaunch_duration_mean_worryknot','latenight_betweenlaunch_duration_std_worryknot','morning_betweenlaunch_duration_std_worryknot',
                 'afternoon_betweenlaunch_duration_std_worryknot','evening_betweenlaunch_duration_std_worryknot']

frequency_fs = [i for i in app_ind_fs if 'frequency' in i]
weekdf_imp_l = weekdf_imp[frequency_fs].copy()
#weekdf_imp_l.max(axis=1)
wk_most_used_app = [i[1] for i in weekdf_imp_l.idxmax(axis=1).str.split('_')]
weekdf_imp['most_used_app'] = [i[1] for i in weekdf_imp_l.idxmax(axis=1).str.split('_')]

def rename_mapper(colname):
    new_colname = colname.replace('_'+wk_most_used_app[i],'')
    return(new_colname)

#mua = most used apps
mostused_app_df_list = []
for i in range(weekdf_imp.shape[0]):
    tempdf = weekdf_imp[[e for e in app_ind_fs if wk_most_used_app[i] in e]].iloc[[i]].copy()
    tempdf.rename(mapper=rename_mapper,axis=1,inplace=True)
    mostused_app_df_list.append(tempdf)
app_mua_weekdf_imp = pd.concat(mostused_app_df_list,sort=False)
app_mua_weekdf_imp = pd.concat([weekdf_imp[['pid','weekofstudy','anx','dep','anx_cat','dep_cat','most_used_app']],app_mua_weekdf_imp],axis=1,copy=True)
app_mua_weekdf_imp = pd.concat([app_mua_weekdf_imp,weekdf_imp[sr_fs]],axis=1,copy=True)
app_mua_weekdf_imp['intercept'] = 1 

#dummitize the most_used_app column
#weekdf_imp -- 'most_used_app'
wapp_mua_dummy = pd.get_dummies(weekdf_imp['most_used_app'])
weekdf_imp = pd.concat([weekdf_imp,wapp_mua_dummy],axis=1)

app_mua_fs = ['intercept','weekofstudy', 'frequency', 'daysofuse', 'duration','duration_mean', 'duration_std', 
          'duration_min', 'duration_max','betweenlaunch_duration_mean', 'betweenlaunch_duration_std'] # + list(wapp_mua_dummy.columns)
app_mua_weekdf_imp = pd.concat([app_mua_weekdf_imp,wapp_mua_dummy],axis=1,copy=True)


#daydf_imp -- 'most_used_app_0'
# dapp_mua_dummy = pd.get_dummies(daydf_imp['most_used_app_0'])
# daydf_imp = pd.concat([daydf_imp,dapp_mua_dummy],axis=1)



######regression tasks on 1-5 scale (cut off on both 1 (floor) and 5 (ceiling)) using lasso linear mixed effect model; 
m1_anx = MixedLM(weekdf_imp['anx'].astype(float),weekdf_imp[app_overall_fs].astype(float),weekdf_imp['pid'],weekdf_imp['intercept'])
r1_anx = m1_anx.fit_regularized(method='l1',alpha=0.2)
r1_anx.params

pred_df = pd.DataFrame(r1_anx.predict(weekdf_imp[app_overall_fs].astype(float)),columns=['pred_anx'])
pred_df['anx'] = weekdf_imp['anx']
pred_df['diff'] = pred_df['pred_anx'] - pred_df['anx']
rmse = np.sqrt(np.sum(pred_df['diff']**2)/pred_df.shape[0])


def genMixedLM(df,outvar,expvars,gpvar,fsLabel,alpha=0.5):
    mixedmodel =MixedLM(endog=df[outvar].astype(float),exog=df[expvars].astype(float),groups=df[gpvar],exog_re=df['intercept'])
    modelres = mixedmodel.fit_regularized(method='l1',alpha=alpha)
    rdf = pd.DataFrame({'expvar':modelres.params.index,'coef':modelres.params,'tvalue':modelres.tvalues,'pvalues':modelres.pvalues}).reset_index()
    rdf.drop(columns=['index'],inplace=True)
    rdf['feature_set'] = fsLabel
    rdf['alpha'] = alpha
    rdf['outvar'] = outvar
    
    pred_df = pd.DataFrame(modelres.predict(df[expvars].astype(float)),columns=['pred_'+outvar])
    pred_df[outvar] = df[outvar]
    pred_df['diff'] = pred_df['pred_'+outvar] - pred_df[outvar]
    rmse = np.sqrt(np.sum(pred_df['diff']**2)/pred_df.shape[0])
    rdf['rmse'] = rmse
    
    return(rdf)


#r1_anx.predict(weekdf_imp[app_overall_fs].astype(float)).hist()
#np.round(r1_anx.predict(weekdf_imp[app_overall_fs].astype(float)))

#r1_anx = m1_anx.fit()

alpha_list = np.arange(0.1,0.81,0.1)
#outcomes
#sr_fs
#app_overall_fs
#app_ind_fs
#app_mua_fs 
#app_ind_tw_fs_new = [f for f in app_ind_tw_fs if f not in exc_fs]

predoutcomes_list = []
for alpha in alpha_list:
    print('alpha: {0}'.format(alpha))
    anx_app_overall_fs_results = genMixedLM(weekdf_imp,'anx',app_overall_fs,'pid','app_overall_fs',alpha=alpha)
    predoutcomes_list.append(anx_app_overall_fs_results.copy())
    anx_sr_fs_results = genMixedLM(weekdf_imp,'anx',['intercept']+sr_fs,'pid','sr_fs',alpha=alpha)
    predoutcomes_list.append(anx_sr_fs_results.copy())
    anx_sr_overall_comb_results = genMixedLM(weekdf_imp,'anx',sr_fs+app_overall_fs,'pid','sr_app_overall_fs',alpha=alpha)
    predoutcomes_list.append(anx_sr_overall_comb_results.copy())
    anx_app_ind_fs_results = genMixedLM(weekdf_imp,'anx',app_ind_fs,'pid','app_ind_fs',alpha=alpha)
    predoutcomes_list.append(anx_app_ind_fs_results.copy())
    anx_sr_app_ind_fs_results = genMixedLM(weekdf_imp,'anx',sr_fs+app_ind_fs,'pid','sr_app_ind_fs',alpha=alpha)
    predoutcomes_list.append(anx_sr_app_ind_fs_results.copy())
    
    anx_app_mua_fs_results = genMixedLM(app_mua_weekdf_imp,'anx',app_mua_fs,'pid','app_mua_fs',alpha=alpha)
    predoutcomes_list.append(anx_app_mua_fs_results.copy())
    anx_sr_app_mua_fs_results = genMixedLM(app_mua_weekdf_imp,'anx',sr_fs+app_mua_fs,'pid','sr_app_mua_fs',alpha=alpha)
    predoutcomes_list.append(anx_sr_app_mua_fs_results.copy())
    
    dep_app_overall_fs_results = genMixedLM(weekdf_imp,'dep',app_overall_fs,'pid','app_overall_fs',alpha=alpha)
    predoutcomes_list.append(dep_app_overall_fs_results.copy())
    dep_sr_fs_results = genMixedLM(weekdf_imp,'dep',['intercept']+sr_fs,'pid','sr_fs',alpha=alpha)
    predoutcomes_list.append(dep_sr_fs_results.copy())
    dep_sr_overall_comb_results = genMixedLM(weekdf_imp,'dep',sr_fs+app_overall_fs,'pid','sr_app_overall_fs',alpha=alpha)
    predoutcomes_list.append(dep_sr_overall_comb_results.copy())
    dep_app_ind_fs_results = genMixedLM(weekdf_imp,'dep',app_ind_fs,'pid','app_ind_fs',alpha=alpha)
    predoutcomes_list.append(dep_app_ind_fs_results.copy())
    dep_sr_app_ind_fs_results = genMixedLM(weekdf_imp,'dep',sr_fs+app_ind_fs,'pid','sr_app_ind_fs',alpha=alpha)
    predoutcomes_list.append(dep_sr_app_ind_fs_results.copy())
    
    dep_app_mua_fs_results = genMixedLM(app_mua_weekdf_imp,'dep',app_mua_fs,'pid','app_mua_fs',alpha=alpha)
    predoutcomes_list.append(dep_app_mua_fs_results.copy())
    dep_sr_app_mua_fs_results = genMixedLM(app_mua_weekdf_imp,'dep',sr_fs+app_mua_fs,'pid','sr_app_mua_fs',alpha=alpha)
    predoutcomes_list.append(dep_sr_app_mua_fs_results.copy())
    

#for alpha in alpha_list:
#    print('alpha: {0}'.format(alpha))    
#    anx_app_mua_fs_results = genMixedLM(app_mua_weekdf_imp,'anx',app_mua_fs,'pid','app_mua_fs',alpha=alpha)
#    predoutcomes_list.append(anx_app_mua_fs_results.copy())
#    anx_sr_app_mua_fs_results = genMixedLM(app_mua_weekdf_imp,'anx',sr_fs+app_mua_fs,'pid','sr_app_mua_fs',alpha=alpha)
#    predoutcomes_list.append(anx_sr_app_mua_fs_results.copy())
#        
#    dep_app_mua_fs_results = genMixedLM(app_mua_weekdf_imp,'dep',app_mua_fs,'pid','app_mua_fs',alpha=alpha)
#    predoutcomes_list.append(dep_app_mua_fs_results.copy())
#    dep_sr_app_mua_fs_results = genMixedLM(app_mua_weekdf_imp,'dep',sr_fs+app_mua_fs,'pid','sr_app_mua_fs',alpha=alpha)
#    predoutcomes_list.append(dep_sr_app_mua_fs_results.copy())

predoutcomes = pd.concat(predoutcomes_list,copy=True,ignore_index=True,sort=False)
predoutcomes.to_csv('lmm_outcomes.csv',index=False)

# app_ind_tw_fs_results = genMixedLM(weekdf_imp,'anx',app_ind_tw_fs_new[200:300],'pid','app_ind_tw_fs',alpha=0.5)


#exc_fs = []
#for col in weekdf_imp.columns:
#    if len(weekdf_imp[col].unique()) == 1:
#        print(weekdf_imp[col].unique())
#        exc_fs.append(col)
#        
#dup_weekdf_imp = weekdf_imp.duplicated(subset=app_ind_tw_fs_new,keep='first')


#####classification tasks on low (<3) and high (>=3) binary classification using lasso generalized logistic regression model
#RF
#fix imbalance in classification
#leave one subject out CV

sr_app_overall_fs = ['pid'] + sr_fs + [i for i in app_overall_fs if i != 'intercept']
sr_app_ind_fs = ['pid'] + sr_fs + [i for i in app_ind_fs if i != 'intercept']
sr_app_mua_fs = ['pid'] + sr_fs + ['weekofstudy', 'frequency', 'daysofuse', 'duration','duration_mean', 'duration_std', 
          'duration_min', 'duration_max','betweenlaunch_duration_mean', 'betweenlaunch_duration_std','most_used_app']


featureset_sizes = []
for featureset in ['app_overall_fs', 'app_ind_fs', 'app_mua_fs', 'sr_app_overall_fs', 'sr_app_ind_fs', 'sr_app_mua_fs']:
    featureset_sizes.append({'featureset': featureset, 'size': len(featureset)})
pd.DataFrame(featureset_sizes).to_csv('featureset_sizes.csv', index=False)

X1 = weekdf_imp[sr_app_overall_fs].copy()
X2 = weekdf_imp[sr_app_ind_fs].copy()
X3 = app_mua_weekdf_imp[sr_app_mua_fs].copy()
print(X3)

y_anx = weekdf_imp['anx_cat'].copy()
y_dep = weekdf_imp['dep_cat'].copy()

from collections import Counter
sm1 = SMOTENC(random_state=1008,categorical_features=[0])
X1_anx_res, y1_anx_res = sm1.fit_resample(X1,y_anx)
X1_anx_res = pd.DataFrame(X1_anx_res,columns=X1.columns,dtype=float)
X1_anx_res['pid'] = X1_anx_res['pid'].astype(str)
y1_anx_res = pd.Series(y1_anx_res)
X1_anx_res_train = X1_anx_res[[c for c in X1_anx_res.columns if c != 'pid']]

X1_dep_res, y1_dep_res = sm1.fit_resample(X1,y_dep)
X1_dep_res = pd.DataFrame(X1_dep_res,columns=X1.columns,dtype=float)
X1_dep_res['pid'] = X1_dep_res['pid'].astype(str)
y1_dep_res = pd.Series(y1_dep_res)
X1_dep_res_train = X1_dep_res[[c for c in X1_dep_res.columns if c != 'pid']]

sm2 = SMOTENC(random_state=1008,categorical_features=[0])
X2_anx_res, y2_anx_res = sm2.fit_resample(X2,y_anx)
X2_anx_res = pd.DataFrame(X2_anx_res,columns=X2.columns,dtype=float)
X2_anx_res['pid'] = X2_anx_res['pid'].astype(str)
y2_anx_res = pd.Series(y2_anx_res)
X2_anx_res_train = X2_anx_res[[c for c in X2_anx_res.columns if c != 'pid']]

X2_dep_res, y2_dep_res = sm2.fit_resample(X2,y_dep)
X2_dep_res = pd.DataFrame(X2_dep_res,columns=X2.columns,dtype=float)
X2_dep_res['pid'] = X2_dep_res['pid'].astype(str)
y2_dep_res = pd.Series(y2_dep_res)
X2_dep_res_train = X2_dep_res[[c for c in X2_dep_res.columns if c != 'pid']]


sm3 = SMOTENC(random_state=1008,categorical_features=[0,25])
X3_anx_res, y3_anx_res = sm3.fit_resample(X3,y_anx)
X3_anx_res = pd.DataFrame(X3_anx_res,columns=X3.columns)
for col in X3_anx_res.columns:
    if col not in ['pid','most_used_app']:
        X3_anx_res[col] = X3_anx_res[col].astype(float)
y3_anx_res = pd.Series(y3_anx_res)
X3_anx_res_train = X3_anx_res[[c for c in X3_anx_res.columns if c not in ['pid','most_used_app']]]
X3_anx_res_app_mua_dummy = pd.get_dummies(X3_anx_res['most_used_app'])
X3_anx_res_train = pd.concat([X3_anx_res_train,X3_anx_res_app_mua_dummy],axis=1,copy=True)


X3_dep_res, y3_dep_res = sm3.fit_resample(X3,y_dep)
X3_dep_res = pd.DataFrame(X3_dep_res,columns=X3.columns)
for col in X3_dep_res.columns:
    if col not in ['pid','most_used_app']:
        X3_dep_res[col] = X3_dep_res[col].astype(float)
y3_dep_res = pd.Series(y3_dep_res)
X3_dep_res_train = X3_dep_res[[c for c in X3_dep_res.columns if c not in ['pid','most_used_app']]]
X3_dep_res_app_mua_dummy = pd.get_dummies(X3_dep_res['most_used_app'])
X3_dep_res_train = pd.concat([X3_dep_res_train,X3_dep_res_app_mua_dummy],axis=1,copy=True)


#transform the data
#label_encoder = LabelEncoder()
#onehot_encoder = OneHotEncoder()
#X3_anx_ohe_mua = onehot_encoder.fit_transform(X3_anx_res[['most_used_app']])
#X3_anx_ohe_mua = pd.DataFrame(X3_anx_ohe_mua)
#X3_anx_res_train = pd.concat([X3_anx_res_train,X3_anx_ohe_mua],axis=1,copy=True

#parameter tuning
def classifyMood(data,X,y,outvar,expvars_label,method='RF'):
    if method == 'XGB':
        n_estimators = [50,100,200,500]
        max_depth = [3,6,9]
        min_child_weight = [1,3,6]
        learning_rate = [0.05,0.1,0.3,0.5]
        param_grid = dict(n_estimators=n_estimators,max_depth=max_depth,min_child_weight=min_child_weight,
                          learning_rate=learning_rate)        
        model = xgboost.XGBClassifier()
        grid = GridSearchCV(estimator=model,param_grid=param_grid, cv= 5, scoring='accuracy', n_jobs=4)   
        grid_result = grid.fit(X,y)
        best_params = grid_result.best_params_        
        model_final= xgboost.XGBClassifier(**best_params)
    

    if method == 'RF':
        n_estimators = [50,100,200,500]
        max_depth = [2,5,10]
        max_features = [10,12,15]
        param_grid = dict(n_estimators=n_estimators,max_depth=max_depth,max_features=max_features)
        model = RandomForestClassifier(oob_score=True,random_state=1008)
        grid = GridSearchCV(estimator=model,param_grid=param_grid, cv= 5, scoring='accuracy', n_jobs=4)
        grid_result = grid.fit(X,y)
        best_params = grid_result.best_params_        
        model_final= RandomForestClassifier(**best_params)

    #leave one subject out cv
    #get the unique pids
    pids = data['pid'].unique()
    res = []
    list_shap_values = list()
    list_test_sets = list()

    for p in pids:
        X_train = X.loc[data['pid']!=p]
        X_test = X.loc[data['pid']==p]
        X_test_indices = list(X.index[data['pid']==p])
        y_train = y.loc[data['pid']!=p]
        y_test = y.loc[data['pid']==p]
        model_final.fit(X_train,y_train)
        
        pred  = model_final.predict(X_test)
        # print(pred)
        df = pd.DataFrame({'pred':pred, 'actual':y_test})
        res.append(df)

        shap_values = shap.TreeExplainer(model_final).shap_values(X_test)
        # print(shap_values)

        list_shap_values.append(shap_values)
        list_test_sets.append(X_test_indices)

    # https://lucasramos-34338.medium.com/visualizing-variable-importance-using-shap-and-cross-validation-bd5075e9063a

    #combining results from all iterations
    test_set = list_test_sets[0]
    shap_values = np.array(list_shap_values[0])
    # print(shap_values)
    # print(list_shap_values[1])
    for i in range(1,len(list_test_sets)):
        test_set = np.concatenate((test_set,list_test_sets[i]),axis=0)
        if method == 'RF':
            shap_values = np.concatenate((shap_values,np.array(list_shap_values[i])),axis=1)
        else:
            shap_values = np.concatenate((shap_values,list_shap_values[i]),axis=0)
    
    #bringing back variable names    
    X_test = pd.DataFrame(X.iloc[test_set],columns=X.columns)

    filename = method + '_' + outvar + '_' + expvars_label
    with open('X_test_' + filename + '.ob', 'wb') as fp:
        pickle.dump(X_test, fp)

    with open('shap_' + filename + '.ob', 'wb') as fp:
        pickle.dump(shap_values, fp)

    res_df = pd.concat(res,copy=True)
    
    res_df['acc'] = accuracy_score(y_true = res_df['actual'], y_pred=res_df['pred'])
    acc = res_df['acc'].sum()/res_df.shape[0]

    res_df['f1'] = f1_score(y_true = res_df['actual'], y_pred=res_df['pred'])
    f1 = res_df['f1'].sum()/res_df.shape[0]

    res_df['precision'] = precision_score(y_true = res_df['actual'], y_pred=res_df['pred'])
    precision = res_df['precision'].sum()/res_df.shape[0]

    res_df['recall'] = recall_score(y_true = res_df['actual'], y_pred=res_df['pred'])
    recall = res_df['recall'].sum()/res_df.shape[0]
    
    return(
        {'method':method,'outcome':outvar,'feature_set':expvars_label, 
        'accuracy':acc, 'f1_score': f1, 'precision': precision, 'recall': recall}
    )

class_outputs = []
class_outputs.append(classifyMood(X1_anx_res,X1_anx_res_train,y1_anx_res,'anxiety','sr_app_overall_fs',method='RF'))
class_outputs.append(classifyMood(X1_dep_res,X1_dep_res_train,y1_dep_res,'depression','sr_app_overall_fs',method='RF'))
class_outputs.append(classifyMood(X2_anx_res,X2_anx_res_train,y2_anx_res,'anxiety','sr_app_ind_fs',method='RF'))
class_outputs.append(classifyMood(X2_dep_res,X2_dep_res_train,y2_dep_res,'depression','sr_app_ind_fs',method='RF'))
class_outputs.append(classifyMood(X3_anx_res,X3_anx_res_train,y3_anx_res,'anxiety','sr_app_mua_fs',method='RF'))
class_outputs.append(classifyMood(X3_dep_res,X3_dep_res_train,y3_dep_res,'depression','sr_app_mua_fs',method='RF'))

class_outputs.append(classifyMood(X1_anx_res,X1_anx_res_train,y1_anx_res,'anxiety','sr_app_overall_fs',method='XGB'))
class_outputs.append(classifyMood(X1_dep_res,X1_dep_res_train,y1_dep_res,'depression','sr_app_overall_fs',method='XGB'))
class_outputs.append(classifyMood(X2_anx_res,X2_anx_res_train,y2_anx_res,'anxiety','sr_app_ind_fs',method='XGB'))
class_outputs.append(classifyMood(X2_dep_res,X2_dep_res_train,y2_dep_res,'depression','sr_app_ind_fs',method='XGB'))
class_outputs.append(classifyMood(X3_anx_res,X3_anx_res_train,y3_anx_res,'anxiety','sr_app_mua_fs',method='XGB'))
class_outputs.append(classifyMood(X3_dep_res,X3_dep_res_train,y3_dep_res,'depression','sr_app_mua_fs',method='XGB'))

class_outputs_df = pd.DataFrame(class_outputs)
class_outputs_df.to_csv('class_outcomes.csv',index=False)


#outcomes methods
#daily score prediction, and aggregate to get a weekly score
#baseline: straightforward prediction









dvarnames = ['pid',
'dayofstudy',
'weekofstudy',
'frequency',
'daysofuse',
'duration',
'duration_mean',
'duration_std',
'duration_min',
'duration_max',
'betweenlaunch_duration_mean',
'betweenlaunch_duration_std',
'frequency_aspire',
'daysofuse_aspire',
'duration_aspire',
'duration_mean_aspire',
'duration_std_aspire',
'duration_min_aspire',
'duration_max_aspire',
'betweenlaunch_duration_mean_aspire',
'betweenlaunch_duration_std_aspire',
'frequency_boostme',
'daysofuse_boostme',
'duration_boostme',
'duration_mean_boostme',
'duration_std_boostme',
'duration_min_boostme',
'duration_max_boostme',
'betweenlaunch_duration_mean_boostme',
'betweenlaunch_duration_std_boostme',
'frequency_dailyfeats',
'daysofuse_dailyfeats',
'duration_dailyfeats',
'duration_mean_dailyfeats',
'duration_std_dailyfeats',
'duration_min_dailyfeats',
'duration_max_dailyfeats',
'betweenlaunch_duration_mean_dailyfeats',
'betweenlaunch_duration_std_dailyfeats',
'frequency_icope',
'daysofuse_icope',
'duration_icope',
'duration_mean_icope',
'duration_std_icope',
'duration_min_icope',
'duration_max_icope',
'betweenlaunch_duration_mean_icope',
'betweenlaunch_duration_std_icope',
'frequency_mantra',
'daysofuse_mantra',
'duration_mantra',
'duration_mean_mantra',
'duration_std_mantra',
'duration_min_mantra',
'duration_max_mantra',
'betweenlaunch_duration_mean_mantra',
'betweenlaunch_duration_std_mantra',
'frequency_messages',
'daysofuse_messages',
'duration_messages',
'duration_mean_messages',
'duration_std_messages',
'duration_min_messages',
'duration_max_messages',
'betweenlaunch_duration_mean_messages',
'betweenlaunch_duration_std_messages',
'frequency_moveme',
'daysofuse_moveme',
'duration_moveme',
'duration_mean_moveme',
'duration_std_moveme',
'duration_min_moveme',
'duration_max_moveme',
'betweenlaunch_duration_mean_moveme',
'betweenlaunch_duration_std_moveme',
'frequency_relax',
'daysofuse_relax',
'duration_relax',
'duration_mean_relax',
'duration_std_relax',
'duration_min_relax',
'duration_max_relax',
'betweenlaunch_duration_mean_relax',
'betweenlaunch_duration_std_relax',
'frequency_slumbertime',
'daysofuse_slumbertime',
'duration_slumbertime',
'duration_mean_slumbertime',
'duration_std_slumbertime',
'duration_min_slumbertime',
'duration_max_slumbertime',
'betweenlaunch_duration_mean_slumbertime',
'betweenlaunch_duration_std_slumbertime',
'frequency_socialforce',
'daysofuse_socialforce',
'duration_socialforce',
'duration_mean_socialforce',
'duration_std_socialforce',
'duration_min_socialforce',
'duration_max_socialforce',
'betweenlaunch_duration_mean_socialforce',
'betweenlaunch_duration_std_socialforce',
'frequency_thoughtchallenger',
'daysofuse_thoughtchallenger',
'duration_thoughtchallenger',
'duration_mean_thoughtchallenger',
'duration_std_thoughtchallenger',
'duration_min_thoughtchallenger',
'duration_max_thoughtchallenger',
'betweenlaunch_duration_mean_thoughtchallenger',
'betweenlaunch_duration_std_thoughtchallenger',
'frequency_worryknot',
'daysofuse_worryknot',
'duration_worryknot',
'duration_mean_worryknot',
'duration_std_worryknot',
'duration_min_worryknot',
'duration_max_worryknot',
'betweenlaunch_duration_mean_worryknot',
'betweenlaunch_duration_std_worryknot',
'num_apps_used',
'most_used_app_0',
'most_used_app_1',
'most_used_app_2',
'anx',
'dep']


       
wvarnames = ['pid',
'weekofstudy',
'frequency',
'daysofuse',
'duration',
'duration_mean',
'duration_std',
'duration_min',
'duration_max',
'betweenlaunch_duration_mean',
'betweenlaunch_duration_std',
'frequency_aspire',
'daysofuse_aspire',
'duration_aspire',
'duration_mean_aspire',
'duration_std_aspire',
'duration_min_aspire',
'duration_max_aspire',
'betweenlaunch_duration_mean_aspire',
'betweenlaunch_duration_std_aspire',
'frequency_boostme',
'daysofuse_boostme',
'duration_boostme',
'duration_mean_boostme',
'duration_std_boostme',
'duration_min_boostme',
'duration_max_boostme',
'betweenlaunch_duration_mean_boostme',
'betweenlaunch_duration_std_boostme',
'frequency_dailyfeats',
'daysofuse_dailyfeats',
'duration_dailyfeats',
'duration_mean_dailyfeats',
'duration_std_dailyfeats',
'duration_min_dailyfeats',
'duration_max_dailyfeats',
'betweenlaunch_duration_mean_dailyfeats',
'betweenlaunch_duration_std_dailyfeats',
'frequency_icope',
'daysofuse_icope',
'duration_icope',
'duration_mean_icope',
'duration_std_icope',
'duration_min_icope',
'duration_max_icope',
'betweenlaunch_duration_mean_icope',
'betweenlaunch_duration_std_icope',
'frequency_mantra',
'daysofuse_mantra',
'duration_mantra',
'duration_mean_mantra',
'duration_std_mantra',
'duration_min_mantra',
'duration_max_mantra',
'betweenlaunch_duration_mean_mantra',
'betweenlaunch_duration_std_mantra',
'frequency_messages',
'daysofuse_messages',
'duration_messages',
'duration_mean_messages',
'duration_std_messages',
'duration_min_messages',
'duration_max_messages',
'betweenlaunch_duration_mean_messages',
'betweenlaunch_duration_std_messages',
'frequency_moveme',
'daysofuse_moveme',
'duration_moveme',
'duration_mean_moveme',
'duration_std_moveme',
'duration_min_moveme',
'duration_max_moveme',
'betweenlaunch_duration_mean_moveme',
'betweenlaunch_duration_std_moveme',
'frequency_relax',
'daysofuse_relax',
'duration_relax',
'duration_mean_relax',
'duration_std_relax',
'duration_min_relax',
'duration_max_relax',
'betweenlaunch_duration_mean_relax',
'betweenlaunch_duration_std_relax',
'frequency_slumbertime',
'daysofuse_slumbertime',
'duration_slumbertime',
'duration_mean_slumbertime',
'duration_std_slumbertime',
'duration_min_slumbertime',
'duration_max_slumbertime',
'betweenlaunch_duration_mean_slumbertime',
'betweenlaunch_duration_std_slumbertime',
'frequency_socialforce',
'daysofuse_socialforce',
'duration_socialforce',
'duration_mean_socialforce',
'duration_std_socialforce',
'duration_min_socialforce',
'duration_max_socialforce',
'betweenlaunch_duration_mean_socialforce',
'betweenlaunch_duration_std_socialforce',
'frequency_thoughtchallenger',
'daysofuse_thoughtchallenger',
'duration_thoughtchallenger',
'duration_mean_thoughtchallenger',
'duration_std_thoughtchallenger',
'duration_min_thoughtchallenger',
'duration_max_thoughtchallenger',
'betweenlaunch_duration_mean_thoughtchallenger',
'betweenlaunch_duration_std_thoughtchallenger',
'frequency_worryknot',
'daysofuse_worryknot',
'duration_worryknot',
'duration_mean_worryknot',
'duration_std_worryknot',
'duration_min_worryknot',
'duration_max_worryknot',
'betweenlaunch_duration_mean_worryknot',
'betweenlaunch_duration_std_worryknot',
'afternoon_frequency_aspire',
'evening_frequency_aspire',
'latenight_frequency_aspire',
'morning_frequency_aspire',
'latenight_daysofuse_aspire',
'morning_daysofuse_aspire',
'afternoon_daysofuse_aspire',
'evening_daysofuse_aspire',
'latenight_duration_aspire',
'morning_duration_aspire',
'afternoon_duration_aspire',
'evening_duration_aspire',
'latenight_duration_mean_aspire',
'morning_duration_mean_aspire',
'afternoon_duration_mean_aspire',
'evening_duration_mean_aspire',
'latenight_duration_std_aspire',
'morning_duration_std_aspire',
'afternoon_duration_std_aspire',
'evening_duration_std_aspire',
'latenight_duration_min_aspire',
'morning_duration_min_aspire',
'afternoon_duration_min_aspire',
'evening_duration_min_aspire',
'latenight_duration_max_aspire',
'morning_duration_max_aspire',
'afternoon_duration_max_aspire',
'evening_duration_max_aspire',
'latenight_betweenlaunch_duration_mean_aspire',
'morning_betweenlaunch_duration_mean_aspire',
'afternoon_betweenlaunch_duration_mean_aspire',
'evening_betweenlaunch_duration_mean_aspire',
'latenight_betweenlaunch_duration_std_aspire',
'morning_betweenlaunch_duration_std_aspire',
'afternoon_betweenlaunch_duration_std_aspire',
'evening_betweenlaunch_duration_std_aspire',
'afternoon_frequency_boostme',
'evening_frequency_boostme',
'latenight_frequency_boostme',
'morning_frequency_boostme',
'latenight_daysofuse_boostme',
'morning_daysofuse_boostme',
'afternoon_daysofuse_boostme',
'evening_daysofuse_boostme',
'latenight_duration_boostme',
'morning_duration_boostme',
'afternoon_duration_boostme',
'evening_duration_boostme',
'latenight_duration_mean_boostme',
'morning_duration_mean_boostme',
'afternoon_duration_mean_boostme',
'evening_duration_mean_boostme',
'latenight_duration_std_boostme',
'morning_duration_std_boostme',
'afternoon_duration_std_boostme',
'evening_duration_std_boostme',
'latenight_duration_min_boostme',
'morning_duration_min_boostme',
'afternoon_duration_min_boostme',
'evening_duration_min_boostme',
'latenight_duration_max_boostme',
'morning_duration_max_boostme',
'afternoon_duration_max_boostme',
'evening_duration_max_boostme',
'latenight_betweenlaunch_duration_mean_boostme',
'morning_betweenlaunch_duration_mean_boostme',
'afternoon_betweenlaunch_duration_mean_boostme',
'evening_betweenlaunch_duration_mean_boostme',
'latenight_betweenlaunch_duration_std_boostme',
'morning_betweenlaunch_duration_std_boostme',
'afternoon_betweenlaunch_duration_std_boostme',
'evening_betweenlaunch_duration_std_boostme',
'afternoon_frequency_dailyfeats',
'evening_frequency_dailyfeats',
'latenight_frequency_dailyfeats',
'morning_frequency_dailyfeats',
'latenight_daysofuse_dailyfeats',
'morning_daysofuse_dailyfeats',
'afternoon_daysofuse_dailyfeats',
'evening_daysofuse_dailyfeats',
'latenight_duration_dailyfeats',
'morning_duration_dailyfeats',
'afternoon_duration_dailyfeats',
'evening_duration_dailyfeats',
'latenight_duration_mean_dailyfeats',
'morning_duration_mean_dailyfeats',
'afternoon_duration_mean_dailyfeats',
'evening_duration_mean_dailyfeats',
'latenight_duration_std_dailyfeats',
'morning_duration_std_dailyfeats',
'afternoon_duration_std_dailyfeats',
'evening_duration_std_dailyfeats',
'latenight_duration_min_dailyfeats',
'morning_duration_min_dailyfeats',
'afternoon_duration_min_dailyfeats',
'evening_duration_min_dailyfeats',
'latenight_duration_max_dailyfeats',
'morning_duration_max_dailyfeats',
'afternoon_duration_max_dailyfeats',
'evening_duration_max_dailyfeats',
'latenight_betweenlaunch_duration_mean_dailyfeats',
'morning_betweenlaunch_duration_mean_dailyfeats',
'afternoon_betweenlaunch_duration_mean_dailyfeats',
'evening_betweenlaunch_duration_mean_dailyfeats',
'latenight_betweenlaunch_duration_std_dailyfeats',
'morning_betweenlaunch_duration_std_dailyfeats',
'afternoon_betweenlaunch_duration_std_dailyfeats',
'evening_betweenlaunch_duration_std_dailyfeats',
'afternoon_frequency_icope',
'evening_frequency_icope',
'latenight_frequency_icope',
'morning_frequency_icope',
'latenight_daysofuse_icope',
'morning_daysofuse_icope',
'afternoon_daysofuse_icope',
'evening_daysofuse_icope',
'latenight_duration_icope',
'morning_duration_icope',
'afternoon_duration_icope',
'evening_duration_icope',
'latenight_duration_mean_icope',
'morning_duration_mean_icope',
'afternoon_duration_mean_icope',
'evening_duration_mean_icope',
'latenight_duration_std_icope',
'morning_duration_std_icope',
'afternoon_duration_std_icope',
'evening_duration_std_icope',
'latenight_duration_min_icope',
'morning_duration_min_icope',
'afternoon_duration_min_icope',
'evening_duration_min_icope',
'latenight_duration_max_icope',
'morning_duration_max_icope',
'afternoon_duration_max_icope',
'evening_duration_max_icope',
'latenight_betweenlaunch_duration_mean_icope',
'morning_betweenlaunch_duration_mean_icope',
'afternoon_betweenlaunch_duration_mean_icope',
'evening_betweenlaunch_duration_mean_icope',
'latenight_betweenlaunch_duration_std_icope',
'morning_betweenlaunch_duration_std_icope',
'afternoon_betweenlaunch_duration_std_icope',
'evening_betweenlaunch_duration_std_icope',
'afternoon_frequency_mantra',
'evening_frequency_mantra',
'latenight_frequency_mantra',
'morning_frequency_mantra',
'latenight_daysofuse_mantra',
'morning_daysofuse_mantra',
'afternoon_daysofuse_mantra',
'evening_daysofuse_mantra',
'latenight_duration_mantra',
'morning_duration_mantra',
'afternoon_duration_mantra',
'evening_duration_mantra',
'latenight_duration_mean_mantra',
'morning_duration_mean_mantra',
'afternoon_duration_mean_mantra',
'evening_duration_mean_mantra',
'latenight_duration_std_mantra',
'morning_duration_std_mantra',
'afternoon_duration_std_mantra',
'evening_duration_std_mantra',
'latenight_duration_min_mantra',
'morning_duration_min_mantra',
'afternoon_duration_min_mantra',
'evening_duration_min_mantra',
'latenight_duration_max_mantra',
'morning_duration_max_mantra',
'afternoon_duration_max_mantra',
'evening_duration_max_mantra',
'latenight_betweenlaunch_duration_mean_mantra',
'morning_betweenlaunch_duration_mean_mantra',
'afternoon_betweenlaunch_duration_mean_mantra',
'evening_betweenlaunch_duration_mean_mantra',
'latenight_betweenlaunch_duration_std_mantra',
'morning_betweenlaunch_duration_std_mantra',
'afternoon_betweenlaunch_duration_std_mantra',
'evening_betweenlaunch_duration_std_mantra',
'afternoon_frequency_messages',
'evening_frequency_messages',
'latenight_frequency_messages',
'morning_frequency_messages',
'latenight_daysofuse_messages',
'morning_daysofuse_messages',
'afternoon_daysofuse_messages',
'evening_daysofuse_messages',
'latenight_duration_messages',
'morning_duration_messages',
'afternoon_duration_messages',
'evening_duration_messages',
'latenight_duration_mean_messages',
'morning_duration_mean_messages',
'afternoon_duration_mean_messages',
'evening_duration_mean_messages',
'latenight_duration_std_messages',
'morning_duration_std_messages',
'afternoon_duration_std_messages',
'evening_duration_std_messages',
'latenight_duration_min_messages',
'morning_duration_min_messages',
'afternoon_duration_min_messages',
'evening_duration_min_messages',
'latenight_duration_max_messages',
'morning_duration_max_messages',
'afternoon_duration_max_messages',
'evening_duration_max_messages',
'latenight_betweenlaunch_duration_mean_messages',
'morning_betweenlaunch_duration_mean_messages',
'afternoon_betweenlaunch_duration_mean_messages',
'evening_betweenlaunch_duration_mean_messages',
'latenight_betweenlaunch_duration_std_messages',
'morning_betweenlaunch_duration_std_messages',
'afternoon_betweenlaunch_duration_std_messages',
'evening_betweenlaunch_duration_std_messages',
'afternoon_frequency_moveme',
'evening_frequency_moveme',
'latenight_frequency_moveme',
'morning_frequency_moveme',
'latenight_daysofuse_moveme',
'morning_daysofuse_moveme',
'afternoon_daysofuse_moveme',
'evening_daysofuse_moveme',
'latenight_duration_moveme',
'morning_duration_moveme',
'afternoon_duration_moveme',
'evening_duration_moveme',
'latenight_duration_mean_moveme',
'morning_duration_mean_moveme',
'afternoon_duration_mean_moveme',
'evening_duration_mean_moveme',
'latenight_duration_std_moveme',
'morning_duration_std_moveme',
'afternoon_duration_std_moveme',
'evening_duration_std_moveme',
'latenight_duration_min_moveme',
'morning_duration_min_moveme',
'afternoon_duration_min_moveme',
'evening_duration_min_moveme',
'latenight_duration_max_moveme',
'morning_duration_max_moveme',
'afternoon_duration_max_moveme',
'evening_duration_max_moveme',
'latenight_betweenlaunch_duration_mean_moveme',
'morning_betweenlaunch_duration_mean_moveme',
'afternoon_betweenlaunch_duration_mean_moveme',
'evening_betweenlaunch_duration_mean_moveme',
'latenight_betweenlaunch_duration_std_moveme',
'morning_betweenlaunch_duration_std_moveme',
'afternoon_betweenlaunch_duration_std_moveme',
'evening_betweenlaunch_duration_std_moveme',
'afternoon_frequency_relax',
'evening_frequency_relax',
'latenight_frequency_relax',
'morning_frequency_relax',
'latenight_daysofuse_relax',
'morning_daysofuse_relax',
'afternoon_daysofuse_relax',
'evening_daysofuse_relax',
'latenight_duration_relax',
'morning_duration_relax',
'afternoon_duration_relax',
'evening_duration_relax',
'latenight_duration_mean_relax',
'morning_duration_mean_relax',
'afternoon_duration_mean_relax',
'evening_duration_mean_relax',
'latenight_duration_std_relax',
'morning_duration_std_relax',
'afternoon_duration_std_relax',
'evening_duration_std_relax',
'latenight_duration_min_relax',
'morning_duration_min_relax',
'afternoon_duration_min_relax',
'evening_duration_min_relax',
'latenight_duration_max_relax',
'morning_duration_max_relax',
'afternoon_duration_max_relax',
'evening_duration_max_relax',
'latenight_betweenlaunch_duration_mean_relax',
'morning_betweenlaunch_duration_mean_relax',
'afternoon_betweenlaunch_duration_mean_relax',
'evening_betweenlaunch_duration_mean_relax',
'latenight_betweenlaunch_duration_std_relax',
'morning_betweenlaunch_duration_std_relax',
'afternoon_betweenlaunch_duration_std_relax',
'evening_betweenlaunch_duration_std_relax',
'afternoon_frequency_slumbertime',
'evening_frequency_slumbertime',
'latenight_frequency_slumbertime',
'morning_frequency_slumbertime',
'latenight_daysofuse_slumbertime',
'morning_daysofuse_slumbertime',
'afternoon_daysofuse_slumbertime',
'evening_daysofuse_slumbertime',
'latenight_duration_slumbertime',
'morning_duration_slumbertime',
'afternoon_duration_slumbertime',
'evening_duration_slumbertime',
'latenight_duration_mean_slumbertime',
'morning_duration_mean_slumbertime',
'afternoon_duration_mean_slumbertime',
'evening_duration_mean_slumbertime',
'latenight_duration_std_slumbertime',
'morning_duration_std_slumbertime',
'afternoon_duration_std_slumbertime',
'evening_duration_std_slumbertime',
'latenight_duration_min_slumbertime',
'morning_duration_min_slumbertime',
'afternoon_duration_min_slumbertime',
'evening_duration_min_slumbertime',
'latenight_duration_max_slumbertime',
'morning_duration_max_slumbertime',
'afternoon_duration_max_slumbertime',
'evening_duration_max_slumbertime',
'latenight_betweenlaunch_duration_mean_slumbertime',
'morning_betweenlaunch_duration_mean_slumbertime',
'afternoon_betweenlaunch_duration_mean_slumbertime',
'evening_betweenlaunch_duration_mean_slumbertime',
'latenight_betweenlaunch_duration_std_slumbertime',
'morning_betweenlaunch_duration_std_slumbertime',
'afternoon_betweenlaunch_duration_std_slumbertime',
'evening_betweenlaunch_duration_std_slumbertime',
'afternoon_frequency_socialforce',
'evening_frequency_socialforce',
'latenight_frequency_socialforce',
'morning_frequency_socialforce',
'latenight_daysofuse_socialforce',
'morning_daysofuse_socialforce',
'afternoon_daysofuse_socialforce',
'evening_daysofuse_socialforce',
'latenight_duration_socialforce',
'morning_duration_socialforce',
'afternoon_duration_socialforce',
'evening_duration_socialforce',
'latenight_duration_mean_socialforce',
'morning_duration_mean_socialforce',
'afternoon_duration_mean_socialforce',
'evening_duration_mean_socialforce',
'latenight_duration_std_socialforce',
'morning_duration_std_socialforce',
'afternoon_duration_std_socialforce',
'evening_duration_std_socialforce',
'latenight_duration_min_socialforce',
'morning_duration_min_socialforce',
'afternoon_duration_min_socialforce',
'evening_duration_min_socialforce',
'latenight_duration_max_socialforce',
'morning_duration_max_socialforce',
'afternoon_duration_max_socialforce',
'evening_duration_max_socialforce',
'latenight_betweenlaunch_duration_mean_socialforce',
'morning_betweenlaunch_duration_mean_socialforce',
'afternoon_betweenlaunch_duration_mean_socialforce',
'evening_betweenlaunch_duration_mean_socialforce',
'latenight_betweenlaunch_duration_std_socialforce',
'morning_betweenlaunch_duration_std_socialforce',
'afternoon_betweenlaunch_duration_std_socialforce',
'evening_betweenlaunch_duration_std_socialforce',
'afternoon_frequency_thoughtchallenger',
'evening_frequency_thoughtchallenger',
'latenight_frequency_thoughtchallenger',
'morning_frequency_thoughtchallenger',
'latenight_daysofuse_thoughtchallenger',
'morning_daysofuse_thoughtchallenger',
'afternoon_daysofuse_thoughtchallenger',
'evening_daysofuse_thoughtchallenger',
'latenight_duration_thoughtchallenger',
'morning_duration_thoughtchallenger',
'afternoon_duration_thoughtchallenger',
'evening_duration_thoughtchallenger',
'latenight_duration_mean_thoughtchallenger',
'morning_duration_mean_thoughtchallenger',
'afternoon_duration_mean_thoughtchallenger',
'evening_duration_mean_thoughtchallenger',
'latenight_duration_std_thoughtchallenger',
'morning_duration_std_thoughtchallenger',
'afternoon_duration_std_thoughtchallenger',
'evening_duration_std_thoughtchallenger',
'latenight_duration_min_thoughtchallenger',
'morning_duration_min_thoughtchallenger',
'afternoon_duration_min_thoughtchallenger',
'evening_duration_min_thoughtchallenger',
'latenight_duration_max_thoughtchallenger',
'morning_duration_max_thoughtchallenger',
'afternoon_duration_max_thoughtchallenger',
'evening_duration_max_thoughtchallenger',
'latenight_betweenlaunch_duration_mean_thoughtchallenger',
'morning_betweenlaunch_duration_mean_thoughtchallenger',
'afternoon_betweenlaunch_duration_mean_thoughtchallenger',
'evening_betweenlaunch_duration_mean_thoughtchallenger',
'latenight_betweenlaunch_duration_std_thoughtchallenger',
'morning_betweenlaunch_duration_std_thoughtchallenger',
'afternoon_betweenlaunch_duration_std_thoughtchallenger',
'evening_betweenlaunch_duration_std_thoughtchallenger',
'afternoon_frequency_worryknot',
'evening_frequency_worryknot',
'latenight_frequency_worryknot',
'morning_frequency_worryknot',
'latenight_daysofuse_worryknot',
'morning_daysofuse_worryknot',
'afternoon_daysofuse_worryknot',
'evening_daysofuse_worryknot',
'latenight_duration_worryknot',
'morning_duration_worryknot',
'afternoon_duration_worryknot',
'evening_duration_worryknot',
'latenight_duration_mean_worryknot',
'morning_duration_mean_worryknot',
'afternoon_duration_mean_worryknot',
'evening_duration_mean_worryknot',
'latenight_duration_std_worryknot',
'morning_duration_std_worryknot',
'afternoon_duration_std_worryknot',
'evening_duration_std_worryknot',
'latenight_duration_min_worryknot',
'morning_duration_min_worryknot',
'afternoon_duration_min_worryknot',
'evening_duration_min_worryknot',
'latenight_duration_max_worryknot',
'morning_duration_max_worryknot',
'afternoon_duration_max_worryknot',
'evening_duration_max_worryknot',
'latenight_betweenlaunch_duration_mean_worryknot',
'morning_betweenlaunch_duration_mean_worryknot',
'afternoon_betweenlaunch_duration_mean_worryknot',
'evening_betweenlaunch_duration_mean_worryknot',
'latenight_betweenlaunch_duration_std_worryknot',
'morning_betweenlaunch_duration_std_worryknot',
'afternoon_betweenlaunch_duration_std_worryknot',
'evening_betweenlaunch_duration_std_worryknot',
'cope_alcohol_tob',
'physical_pain',
'connected',
'receive_support',
'anx',
'dep',
'active',
'support_others',
'healthy_food',
'manage_neg',
'num_apps_used']