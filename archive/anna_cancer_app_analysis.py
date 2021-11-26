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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
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

weekdf = pd.read_csv('/Users/lihuacai/Desktop/all_ind_wkly.csv')
daydf = pd.read_csv('/Users/lihuacai/Desktop/all_ind_dly.csv')
visdf = pd.read_csv('/Users/lihuacai/Desktop/app_launch_processed.csv')

weekdf.drop(axis=1,columns=['Unnamed: 0'],inplace=True)
daydf.drop(axis=1,columns=['Unnamed: 0'],inplace=True)

weekdf['pid'] = weekdf['pid'].astype(str)
daydf['pid'] = daydf['pid'].astype(str)
visdf['pid'] = visdf['pid'].astype(str)

visdf['date'] = pd.to_datetime(visdf['date'])
visdf['duration'] = [pd.Timedelta(seconds=i) for i in visdf['duration']]
visdf['edate'] = visdf['date'] + visdf['duration']
visdf = visdf.loc[visdf['duration']>pd.Timedelta(seconds=0)].copy()


dvarnames = list(daydf.columns)
wvarnames = list(weekdf.columns)


#MixedLM.fit_regularized(start_params=None, method='l1', alpha=0, ceps=0.0001, ptol=1e-06, maxit=200, **fit_kwargs)

#for i,col in enumerate(weekdf.columns):
#    if i >= 600 and i < 700:
#        print("'{0}',".format(col))
        
#loyalty -- number of app launches
#regularity -- number of time unit (e.g., day, time bucket) app(s) being launched
#duration -- total
        

#class imbalance, missing data (especially response var)
#https://scikit-learn.org/stable/modules/impute.html#:~:text=Missing%20values%20can%20be%20imputed,for%20different%20missing%20values%20encodings.&text=%3E%3E%3E%20import%20numpy%20as%20np%20%3E%3E%3E%20from%20sklearn.
#https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html

def fillmissing(df,cols,val,fillval):
    for c in cols:
        df[c].loc[df[c] == val] = fillval

#imputing missing survey data -- weekly data
impweekvars = ['cope_alcohol_tob','physical_pain','connected','receive_support','anx_mood',
               'dep_mood','active','support_others','healthy_food']
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

#imputing missing survey data -- daily data
impdayvars = ['pid','dayofstudy','weekofstudy',
              'cope_alcohol_tob','physical_pain','connected','receive_support',
              'active','support_others','healthy_food','anx_mood','dep_mood']
dvarnames_app = [i for i in dvarnames if i not in impdayvars]
daydf_app = daydf.drop(axis=1,columns=impdayvars)
daydf_napp = daydf[['pid','dayofstudy','weekofstudy']].copy()

weekdf_imp_c = weekdf_imp[['pid','weekofstudy']+impweekvars_c].copy()
daydf_napp_imp = daydf_napp.merge(weekdf_imp_c,on=['pid','weekofstudy'])
#daydf_napp_imp.drop(axis=1,columns=['anx_mood_x','dep_mood_x'],inplace=True)
#daydf_napp_imp.rename(columns={'anx_mood_y':'anx_mood','dep_mood_y':'dep_mood'},inplace=True)

daydf_imp = pd.concat([daydf_napp_imp,daydf_app],copy=True,axis=1)

#handling missing values in app features
#apps = ['aspire','boostme','dailyfeats','icope','mantra','messages','moveme','relax',
#        'slumbertime','socialforce','thoughtchallenger','worryknot']
#timebuckets = ['afternoon','evening','latenight','morning']
#featuretypes = ['loyalty','regularity','duration','betweenlaunch_duration']

loyalty_dvars = [n for n in dvarnames if 'loyalty' in n]
reg_dvars = [n for n in dvarnames if 'regularity' in n]
dur_dvars = [n for n in dvarnames if 'duration' in n and 'betweenlaunch' not in n]
lau_dur_dvars = [n for n in dvarnames if 'betweenlaunch' in n]

fillmissing(daydf_imp,loyalty_dvars,-1,0)
fillmissing(daydf_imp,reg_dvars,-1,0)
fillmissing(daydf_imp,dur_dvars,-1,0)
fillmissing(daydf_imp,lau_dur_dvars,-1,3600*24)

loyalty_wvars = [n for n in wvarnames if 'loyalty' in n]
reg_wvars = [n for n in wvarnames if 'regularity' in n]
dur_wvars = [n for n in wvarnames if 'duration' in n and 'betweenlaunch' not in n]
lau_dur_wvars = [n for n in wvarnames if 'betweenlaunch' in n]

fillmissing(weekdf_imp,loyalty_wvars,-1,0)
fillmissing(weekdf_imp,reg_wvars,-1,0)
fillmissing(weekdf_imp,dur_wvars,-1,0)
fillmissing(weekdf_imp,lau_dur_wvars,-1,3600*24*7)

#add the intercept columns for the linear mixed model
weekdf_imp['intercept'] = 1
daydf_imp['intercept'] = 1

#outcomes transformation -- anx_mood, dep_mood
#week to week change as outcome
#change to baseline level as outcome
#instead of difference, consider ratio between the weekly value and the baseline
#global average being subtracted

daydf_imp['anx_mood'].hist()
daydf_imp['dep_mood'].hist()

weekdf_imp['anx_mood'].hist()
weekdf_imp['dep_mood'].hist()

#convert the depression mood to its reverse scale to be consistent with anxiety mood
#the higher the anxiety and depression mood scores are, the worse they are (e.g., more anxious or depressed)
daydf_imp['dep_mood'] = daydf_imp['dep_mood'].map({1:5,2:4,3:3,4:2,5:1})
weekdf_imp['dep_mood'] = weekdf_imp['dep_mood'].map({1:5,2:4,3:3,4:2,5:1})

#add classification outcomes
daydf_imp['dep_mood_cat'] = np.where(daydf_imp['dep_mood']>=4,1,0)
daydf_imp['anx_mood_cat'] = np.where(daydf_imp['anx_mood']>=3,1,0)
weekdf_imp['dep_mood_cat'] = np.where(weekdf_imp['dep_mood']>=4,1,0)
weekdf_imp['anx_mood_cat'] = np.where(weekdf_imp['anx_mood']>=3,1,0)

outcomes = ['anx_mood','dep_mood','anx_mood_cat','dep_mood_cat']


#feature sets
#overall app usage; break down by apps; break down by apps and time windows; only most used app (each person will be different); all features
sur_fs = ['cope_alcohol_tob','physical_pain','connected','receive_support','active','support_others','healthy_food',
          'cope_alcohol_tob_ind','physical_pain_ind','connected_ind','receive_support_ind','active_ind','support_others_ind','healthy_food_ind']
overall_app_fs = ['intercept','weekofstudy','loyalty','regularity','duration','duration_mean',
                  'duration_std','duration_min','duration_max','betweenlaunch_duration_mean','betweenlaunch_duration_std',
                  'num_apps_used'] #,'aspire', 'boostme', 'dailyfeats', 'icope', 'mantra', 'messages',
                  #'moveme', 'relax', 'slumbertime', 'thoughtchallenger', 'worryknot']
ind_app_fs = ['intercept','weekofstudy','loyalty_aspire','regularity_aspire','duration_aspire','duration_mean_aspire',
              'duration_std_aspire','duration_min_aspire','duration_max_aspire','betweenlaunch_duration_mean_aspire',
              'betweenlaunch_duration_std_aspire','loyalty_boostme','regularity_boostme','duration_boostme',
              'duration_mean_boostme','duration_std_boostme','duration_min_boostme','duration_max_boostme',
              'betweenlaunch_duration_mean_boostme','betweenlaunch_duration_std_boostme','loyalty_dailyfeats',
              'regularity_dailyfeats','duration_dailyfeats','duration_mean_dailyfeats','duration_std_dailyfeats',
              'duration_min_dailyfeats','duration_max_dailyfeats','betweenlaunch_duration_mean_dailyfeats',
              'betweenlaunch_duration_std_dailyfeats','loyalty_icope','regularity_icope','duration_icope',
              'duration_mean_icope','duration_std_icope','duration_min_icope','duration_max_icope',
              'betweenlaunch_duration_mean_icope','betweenlaunch_duration_std_icope','loyalty_mantra','regularity_mantra',
              'duration_mantra','duration_mean_mantra','duration_std_mantra','duration_min_mantra','duration_max_mantra',
              'betweenlaunch_duration_mean_mantra','betweenlaunch_duration_std_mantra','loyalty_messages',
              'regularity_messages','duration_messages','duration_mean_messages','duration_std_messages',
              'duration_min_messages','duration_max_messages','betweenlaunch_duration_mean_messages',
              'betweenlaunch_duration_std_messages','loyalty_moveme','regularity_moveme','duration_moveme',
              'duration_mean_moveme','duration_std_moveme','duration_min_moveme','duration_max_moveme',
              'betweenlaunch_duration_mean_moveme','betweenlaunch_duration_std_moveme','loyalty_relax',
              'regularity_relax','duration_relax','duration_mean_relax','duration_std_relax','duration_min_relax',
              'duration_max_relax','betweenlaunch_duration_mean_relax','betweenlaunch_duration_std_relax',
              'loyalty_slumbertime','regularity_slumbertime','duration_slumbertime','duration_mean_slumbertime',
              'duration_std_slumbertime','duration_min_slumbertime','duration_max_slumbertime',
              'betweenlaunch_duration_mean_slumbertime','betweenlaunch_duration_std_slumbertime','loyalty_socialforce',
              'regularity_socialforce','duration_socialforce','duration_mean_socialforce','duration_std_socialforce',
              'duration_min_socialforce','duration_max_socialforce','betweenlaunch_duration_mean_socialforce',
              'betweenlaunch_duration_std_socialforce','loyalty_thoughtchallenger','regularity_thoughtchallenger',
              'duration_thoughtchallenger','duration_mean_thoughtchallenger','duration_std_thoughtchallenger',
              'duration_min_thoughtchallenger','duration_max_thoughtchallenger','betweenlaunch_duration_mean_thoughtchallenger',
              'betweenlaunch_duration_std_thoughtchallenger','loyalty_worryknot','regularity_worryknot',
              'duration_worryknot','duration_mean_worryknot','duration_std_worryknot','duration_min_worryknot',
              'duration_max_worryknot','betweenlaunch_duration_mean_worryknot','betweenlaunch_duration_std_worryknot']
ind_app_tw_fs = ['intercept','weekofstudy','afternoon_loyalty_aspire','evening_loyalty_aspire','latenight_loyalty_aspire',
                 'morning_loyalty_aspire','latenight_regularity_aspire','morning_regularity_aspire',
                 'afternoon_regularity_aspire','evening_regularity_aspire','latenight_duration_aspire',
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
                 'evening_betweenlaunch_duration_std_aspire','afternoon_loyalty_boostme','evening_loyalty_boostme',
                 'latenight_loyalty_boostme','morning_loyalty_boostme','latenight_regularity_boostme',
                 'morning_regularity_boostme','afternoon_regularity_boostme','evening_regularity_boostme',
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
                 'evening_betweenlaunch_duration_std_boostme','afternoon_loyalty_dailyfeats','evening_loyalty_dailyfeats',
                 'latenight_loyalty_dailyfeats','morning_loyalty_dailyfeats','latenight_regularity_dailyfeats',
                 'morning_regularity_dailyfeats','afternoon_regularity_dailyfeats','evening_regularity_dailyfeats',
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
                 'evening_betweenlaunch_duration_std_dailyfeats','afternoon_loyalty_icope','evening_loyalty_icope',
                 'latenight_loyalty_icope','morning_loyalty_icope','latenight_regularity_icope','morning_regularity_icope',
                 'afternoon_regularity_icope','evening_regularity_icope','latenight_duration_icope','morning_duration_icope',
                 'afternoon_duration_icope','evening_duration_icope','latenight_duration_mean_icope',
                 'morning_duration_mean_icope','afternoon_duration_mean_icope','evening_duration_mean_icope','latenight_duration_std_icope',
                 'morning_duration_std_icope','afternoon_duration_std_icope','evening_duration_std_icope',
                 'latenight_duration_min_icope','morning_duration_min_icope','afternoon_duration_min_icope',
                 'evening_duration_min_icope','latenight_duration_max_icope','morning_duration_max_icope',
                 'afternoon_duration_max_icope','evening_duration_max_icope','latenight_betweenlaunch_duration_mean_icope',
                 'morning_betweenlaunch_duration_mean_icope','afternoon_betweenlaunch_duration_mean_icope',
                 'evening_betweenlaunch_duration_mean_icope','latenight_betweenlaunch_duration_std_icope',
                 'morning_betweenlaunch_duration_std_icope','afternoon_betweenlaunch_duration_std_icope',
                 'evening_betweenlaunch_duration_std_icope','afternoon_loyalty_mantra','evening_loyalty_mantra',
                 'latenight_loyalty_mantra','morning_loyalty_mantra','latenight_regularity_mantra','morning_regularity_mantra',
                 'afternoon_regularity_mantra','evening_regularity_mantra','latenight_duration_mantra',
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
                 'afternoon_loyalty_messages','evening_loyalty_messages','latenight_loyalty_messages','morning_loyalty_messages',
                 'latenight_regularity_messages','morning_regularity_messages','afternoon_regularity_messages','evening_regularity_messages',
                 'latenight_duration_messages','morning_duration_messages','afternoon_duration_messages','evening_duration_messages',
                 'latenight_duration_mean_messages','morning_duration_mean_messages','afternoon_duration_mean_messages','evening_duration_mean_messages',
                 'latenight_duration_std_messages','morning_duration_std_messages','afternoon_duration_std_messages','evening_duration_std_messages',
                 'latenight_duration_min_messages','morning_duration_min_messages','afternoon_duration_min_messages','evening_duration_min_messages',
                 'latenight_duration_max_messages','morning_duration_max_messages','afternoon_duration_max_messages',
                 'evening_duration_max_messages','latenight_betweenlaunch_duration_mean_messages','morning_betweenlaunch_duration_mean_messages',
                 'afternoon_betweenlaunch_duration_mean_messages','evening_betweenlaunch_duration_mean_messages',
                 'latenight_betweenlaunch_duration_std_messages','morning_betweenlaunch_duration_std_messages','afternoon_betweenlaunch_duration_std_messages',
                 'evening_betweenlaunch_duration_std_messages','afternoon_loyalty_moveme','evening_loyalty_moveme',
                 'latenight_loyalty_moveme','morning_loyalty_moveme','latenight_regularity_moveme',
                 'morning_regularity_moveme','afternoon_regularity_moveme','evening_regularity_moveme','latenight_duration_moveme',
                 'morning_duration_moveme','afternoon_duration_moveme','evening_duration_moveme','latenight_duration_mean_moveme',
                 'morning_duration_mean_moveme','afternoon_duration_mean_moveme','evening_duration_mean_moveme','latenight_duration_std_moveme',
                 'morning_duration_std_moveme','afternoon_duration_std_moveme','evening_duration_std_moveme','latenight_duration_min_moveme',
                 'morning_duration_min_moveme','afternoon_duration_min_moveme','evening_duration_min_moveme','latenight_duration_max_moveme',
                 'morning_duration_max_moveme','afternoon_duration_max_moveme','evening_duration_max_moveme','latenight_betweenlaunch_duration_mean_moveme',
                 'morning_betweenlaunch_duration_mean_moveme','afternoon_betweenlaunch_duration_mean_moveme','evening_betweenlaunch_duration_mean_moveme',
                 'latenight_betweenlaunch_duration_std_moveme','morning_betweenlaunch_duration_std_moveme','afternoon_betweenlaunch_duration_std_moveme',
                 'evening_betweenlaunch_duration_std_moveme','afternoon_loyalty_relax','evening_loyalty_relax','latenight_loyalty_relax',
                 'morning_loyalty_relax','latenight_regularity_relax','morning_regularity_relax','afternoon_regularity_relax',
                 'evening_regularity_relax','latenight_duration_relax','morning_duration_relax','afternoon_duration_relax',
                 'evening_duration_relax','latenight_duration_mean_relax','morning_duration_mean_relax','afternoon_duration_mean_relax',
                 'evening_duration_mean_relax','latenight_duration_std_relax','morning_duration_std_relax','afternoon_duration_std_relax',
                 'evening_duration_std_relax','latenight_duration_min_relax','morning_duration_min_relax','afternoon_duration_min_relax',
                 'evening_duration_min_relax','latenight_duration_max_relax','morning_duration_max_relax','afternoon_duration_max_relax',
                 'evening_duration_max_relax','latenight_betweenlaunch_duration_mean_relax','morning_betweenlaunch_duration_mean_relax',
                 'afternoon_betweenlaunch_duration_mean_relax','evening_betweenlaunch_duration_mean_relax','latenight_betweenlaunch_duration_std_relax',
                 'morning_betweenlaunch_duration_std_relax','afternoon_betweenlaunch_duration_std_relax','evening_betweenlaunch_duration_std_relax',
                 'afternoon_loyalty_slumbertime','evening_loyalty_slumbertime','latenight_loyalty_slumbertime','morning_loyalty_slumbertime',
                 'latenight_regularity_slumbertime','morning_regularity_slumbertime','afternoon_regularity_slumbertime',
                 'evening_regularity_slumbertime','latenight_duration_slumbertime','morning_duration_slumbertime',
                 'afternoon_duration_slumbertime','evening_duration_slumbertime','latenight_duration_mean_slumbertime','morning_duration_mean_slumbertime',
                 'afternoon_duration_mean_slumbertime','evening_duration_mean_slumbertime','latenight_duration_std_slumbertime',
                 'morning_duration_std_slumbertime','afternoon_duration_std_slumbertime','evening_duration_std_slumbertime',
                 'latenight_duration_min_slumbertime','morning_duration_min_slumbertime','afternoon_duration_min_slumbertime',
                 'evening_duration_min_slumbertime','latenight_duration_max_slumbertime','morning_duration_max_slumbertime',
                 'afternoon_duration_max_slumbertime','evening_duration_max_slumbertime','latenight_betweenlaunch_duration_mean_slumbertime',
                 'morning_betweenlaunch_duration_mean_slumbertime','afternoon_betweenlaunch_duration_mean_slumbertime','evening_betweenlaunch_duration_mean_slumbertime',
                 'latenight_betweenlaunch_duration_std_slumbertime','morning_betweenlaunch_duration_std_slumbertime','afternoon_betweenlaunch_duration_std_slumbertime',
                 'evening_betweenlaunch_duration_std_slumbertime','afternoon_loyalty_socialforce','evening_loyalty_socialforce',
                 'latenight_loyalty_socialforce','morning_loyalty_socialforce','latenight_regularity_socialforce',
                 'morning_regularity_socialforce','afternoon_regularity_socialforce','evening_regularity_socialforce',
                 'latenight_duration_socialforce','morning_duration_socialforce','afternoon_duration_socialforce',
                 'evening_duration_socialforce','latenight_duration_mean_socialforce','morning_duration_mean_socialforce',
                 'afternoon_duration_mean_socialforce','evening_duration_mean_socialforce','latenight_duration_std_socialforce',
                 'morning_duration_std_socialforce','afternoon_duration_std_socialforce','evening_duration_std_socialforce',
                 'latenight_duration_min_socialforce','morning_duration_min_socialforce','afternoon_duration_min_socialforce',
                 'evening_duration_min_socialforce','latenight_duration_max_socialforce','morning_duration_max_socialforce',
                 'afternoon_duration_max_socialforce','evening_duration_max_socialforce','latenight_betweenlaunch_duration_mean_socialforce',
                 'morning_betweenlaunch_duration_mean_socialforce','afternoon_betweenlaunch_duration_mean_socialforce','evening_betweenlaunch_duration_mean_socialforce',
                 'latenight_betweenlaunch_duration_std_socialforce','morning_betweenlaunch_duration_std_socialforce',
                 'afternoon_betweenlaunch_duration_std_socialforce','evening_betweenlaunch_duration_std_socialforce','afternoon_loyalty_thoughtchallenger',
                 'evening_loyalty_thoughtchallenger','latenight_loyalty_thoughtchallenger','morning_loyalty_thoughtchallenger',
                 'latenight_regularity_thoughtchallenger','morning_regularity_thoughtchallenger','afternoon_regularity_thoughtchallenger',
                 'evening_regularity_thoughtchallenger','latenight_duration_thoughtchallenger','morning_duration_thoughtchallenger',
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
                 'afternoon_loyalty_worryknot','evening_loyalty_worryknot','latenight_loyalty_worryknot','morning_loyalty_worryknot',
                 'latenight_regularity_worryknot','morning_regularity_worryknot','afternoon_regularity_worryknot','evening_regularity_worryknot',
                 'latenight_duration_worryknot','morning_duration_worryknot','afternoon_duration_worryknot','evening_duration_worryknot',
                 'latenight_duration_mean_worryknot','morning_duration_mean_worryknot','afternoon_duration_mean_worryknot',
                 'evening_duration_mean_worryknot','latenight_duration_std_worryknot','morning_duration_std_worryknot',
                 'afternoon_duration_std_worryknot','evening_duration_std_worryknot','latenight_duration_min_worryknot',
                 'morning_duration_min_worryknot','afternoon_duration_min_worryknot','evening_duration_min_worryknot','latenight_duration_max_worryknot',
                 'morning_duration_max_worryknot','afternoon_duration_max_worryknot','evening_duration_max_worryknot',
                 'latenight_betweenlaunch_duration_mean_worryknot','morning_betweenlaunch_duration_mean_worryknot','afternoon_betweenlaunch_duration_mean_worryknot',
                 'evening_betweenlaunch_duration_mean_worryknot','latenight_betweenlaunch_duration_std_worryknot','morning_betweenlaunch_duration_std_worryknot',
                 'afternoon_betweenlaunch_duration_std_worryknot','evening_betweenlaunch_duration_std_worryknot']

loyalty_fs = [i for i in ind_app_fs if 'loyalty' in i]
weekdf_imp_l = weekdf_imp[loyalty_fs].copy()
#weekdf_imp_l.max(axis=1)
wk_most_used_app = [i[1] for i in weekdf_imp_l.idxmax(axis=1).str.split('_')]
weekdf_imp['most_used_app'] = [i[1] for i in weekdf_imp_l.idxmax(axis=1).str.split('_')]

def rename_mapper(colname):
    new_colname = colname.replace('_'+wk_most_used_app[i],'')
    return(new_colname)

#mua = most used apps
mostused_app_df_list = []
for i in range(weekdf_imp.shape[0]):
    tempdf = weekdf_imp[[e for e in ind_app_fs if wk_most_used_app[i] in e]].iloc[[i]].copy()
    tempdf.rename(mapper=rename_mapper,axis=1,inplace=True)
    mostused_app_df_list.append(tempdf)
mua_weekdf_imp = pd.concat(mostused_app_df_list,sort=False)
mua_weekdf_imp = pd.concat([weekdf_imp[['pid','weekofstudy','anx_mood','dep_mood','anx_mood_cat','dep_mood_cat','most_used_app']],mua_weekdf_imp],axis=1,copy=True)
mua_weekdf_imp = pd.concat([mua_weekdf_imp,weekdf_imp[sur_fs]],axis=1,copy=True)
mua_weekdf_imp['intercept'] = 1 

#dummitize the most_used_app column
#weekdf_imp -- 'most_used_app'
wmua_dummy = pd.get_dummies(weekdf_imp['most_used_app'])
weekdf_imp = pd.concat([weekdf_imp,wmua_dummy],axis=1)

mua_fs = ['intercept','weekofstudy', 'loyalty', 'regularity', 'duration','duration_mean', 'duration_std', 
          'duration_min', 'duration_max','betweenlaunch_duration_mean', 'betweenlaunch_duration_std'] # + list(wmua_dummy.columns)
mua_weekdf_imp = pd.concat([mua_weekdf_imp,wmua_dummy],axis=1,copy=True)


#daydf_imp -- 'most_used_app_0'
dmua_dummy = pd.get_dummies(daydf_imp['most_used_app_0'])
daydf_imp = pd.concat([daydf_imp,dmua_dummy],axis=1)



######regression tasks on 1-5 scale (cut off on both 1 (floor) and 5 (ceiling)) using lasso linear mixed effect model; 
m1_anx = MixedLM(weekdf_imp['anx_mood'].astype(float),weekdf_imp[overall_app_fs].astype(float),weekdf_imp['pid'],weekdf_imp['intercept'])
r1_anx = m1_anx.fit_regularized(method='l1',alpha=0.2)
r1_anx.params

pred_df = pd.DataFrame(r1_anx.predict(weekdf_imp[overall_app_fs].astype(float)),columns=['pred_anx'])
pred_df['anx_mood'] = weekdf_imp['anx_mood']
pred_df['diff'] = pred_df['pred_anx'] - pred_df['anx_mood']
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


#r1_anx.predict(weekdf_imp[overall_app_fs].astype(float)).hist()
#np.round(r1_anx.predict(weekdf_imp[overall_app_fs].astype(float)))

#r1_anx = m1_anx.fit()

alpha_list = np.arange(0.1,0.81,0.1)
#outcomes
#sur_fs
#overall_app_fs
#ind_app_fs
#mua_fs 
#ind_app_tw_fs_new = [f for f in ind_app_tw_fs if f not in exc_fs]

predoutcomes_list = []
for alpha in alpha_list:
    print('alpha: {0}'.format(alpha))
    anx_overall_app_fs_results = genMixedLM(weekdf_imp,'anx_mood',overall_app_fs,'pid','overall_app_fs',alpha=alpha)
    predoutcomes_list.append(anx_overall_app_fs_results.copy())
    anx_sur_fs_results = genMixedLM(weekdf_imp,'anx_mood',['intercept']+sur_fs,'pid','sur_fs',alpha=alpha)
    predoutcomes_list.append(anx_sur_fs_results.copy())
    anx_sur_overall_comb_results = genMixedLM(weekdf_imp,'anx_mood',sur_fs+overall_app_fs,'pid','sur_overall_fs',alpha=alpha)
    predoutcomes_list.append(anx_sur_overall_comb_results.copy())
    anx_ind_app_fs_results = genMixedLM(weekdf_imp,'anx_mood',ind_app_fs,'pid','ind_app_fs',alpha=alpha)
    predoutcomes_list.append(anx_ind_app_fs_results.copy())
    anx_sur_ind_app_fs_results = genMixedLM(weekdf_imp,'anx_mood',sur_fs+ind_app_fs,'pid','sur_ind_app_fs',alpha=alpha)
    predoutcomes_list.append(anx_sur_ind_app_fs_results.copy())
    
    anx_mua_fs_results = genMixedLM(mua_weekdf_imp,'anx_mood',mua_fs,'pid','mua_fs',alpha=alpha)
    predoutcomes_list.append(anx_mua_fs_results.copy())
    anx_sur_mua_fs_results = genMixedLM(mua_weekdf_imp,'anx_mood',sur_fs+mua_fs,'pid','sur_mua_fs',alpha=alpha)
    predoutcomes_list.append(anx_sur_mua_fs_results.copy())
    
    dep_overall_app_fs_results = genMixedLM(weekdf_imp,'dep_mood',overall_app_fs,'pid','overall_app_fs',alpha=alpha)
    predoutcomes_list.append(dep_overall_app_fs_results.copy())
    dep_sur_fs_results = genMixedLM(weekdf_imp,'dep_mood',['intercept']+sur_fs,'pid','sur_fs',alpha=alpha)
    predoutcomes_list.append(dep_sur_fs_results.copy())
    dep_sur_overall_comb_results = genMixedLM(weekdf_imp,'dep_mood',sur_fs+overall_app_fs,'pid','sur_overall_fs',alpha=alpha)
    predoutcomes_list.append(dep_sur_overall_comb_results.copy())
    dep_ind_app_fs_results = genMixedLM(weekdf_imp,'dep_mood',ind_app_fs,'pid','ind_app_fs',alpha=alpha)
    predoutcomes_list.append(dep_ind_app_fs_results.copy())
    dep_sur_ind_app_fs_results = genMixedLM(weekdf_imp,'dep_mood',sur_fs+ind_app_fs,'pid','sur_ind_app_fs',alpha=alpha)
    predoutcomes_list.append(dep_sur_ind_app_fs_results.copy())
    
    dep_mua_fs_results = genMixedLM(mua_weekdf_imp,'dep_mood',mua_fs,'pid','mua_fs',alpha=alpha)
    predoutcomes_list.append(dep_mua_fs_results.copy())
    dep_sur_mua_fs_results = genMixedLM(mua_weekdf_imp,'dep_mood',sur_fs+mua_fs,'pid','sur_mua_fs',alpha=alpha)
    predoutcomes_list.append(dep_sur_mua_fs_results.copy())
    

#for alpha in alpha_list:
#    print('alpha: {0}'.format(alpha))    
#    anx_mua_fs_results = genMixedLM(mua_weekdf_imp,'anx_mood',mua_fs,'pid','mua_fs',alpha=alpha)
#    predoutcomes_list.append(anx_mua_fs_results.copy())
#    anx_sur_mua_fs_results = genMixedLM(mua_weekdf_imp,'anx_mood',sur_fs+mua_fs,'pid','sur_mua_fs',alpha=alpha)
#    predoutcomes_list.append(anx_sur_mua_fs_results.copy())
#        
#    dep_mua_fs_results = genMixedLM(mua_weekdf_imp,'dep_mood',mua_fs,'pid','mua_fs',alpha=alpha)
#    predoutcomes_list.append(dep_mua_fs_results.copy())
#    dep_sur_mua_fs_results = genMixedLM(mua_weekdf_imp,'dep_mood',sur_fs+mua_fs,'pid','sur_mua_fs',alpha=alpha)
#    predoutcomes_list.append(dep_sur_mua_fs_results.copy())

predoutcomes = pd.concat(predoutcomes_list,copy=True,ignore_index=True,sort=False)
predoutcomes.to_csv('/Users/lihuacai/Desktop/anna_lmm_outcomes.csv',index=False)





#ind_app_tw_fs_results = genMixedLM(weekdf_imp,'anx_mood',ind_app_tw_fs_new[200:300],'pid','ind_app_tw_fs',alpha=0.5)


#exc_fs = []
#for col in weekdf_imp.columns:
#    if len(weekdf_imp[col].unique()) == 1:
#        print(weekdf_imp[col].unique())
#        exc_fs.append(col)
#        
#dup_weekdf_imp = weekdf_imp.duplicated(subset=ind_app_tw_fs_new,keep='first')


#####classification tasks on low (<3) and high (>=3) binary classification using lasso generalized logistic regression model
#RF
#fix imbalance in classification
#leave one subject out CV

fs_set1 = ['pid'] + sur_fs + [i for i in overall_app_fs if i != 'intercept']
fs_set2 = ['pid'] + sur_fs + [i for i in ind_app_fs if i != 'intercept']
fs_set3 = ['pid'] + sur_fs + ['weekofstudy', 'loyalty', 'regularity', 'duration','duration_mean', 'duration_std', 
          'duration_min', 'duration_max','betweenlaunch_duration_mean', 'betweenlaunch_duration_std','most_used_app']

X1 = weekdf_imp[fs_set1].copy()
X2 = weekdf_imp[fs_set2].copy()
X3 = mua_weekdf_imp[fs_set3].copy()

y_anx = weekdf_imp['anx_mood_cat'].copy()
y_dep = weekdf_imp['dep_mood_cat'].copy()

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
X3_anx_res_mua_dummy = pd.get_dummies(X3_anx_res['most_used_app'])
X3_anx_res_train = pd.concat([X3_anx_res_train,X3_anx_res_mua_dummy],axis=1,copy=True)


X3_dep_res, y3_dep_res = sm3.fit_resample(X3,y_dep)
X3_dep_res = pd.DataFrame(X3_dep_res,columns=X3.columns)
for col in X3_dep_res.columns:
    if col not in ['pid','most_used_app']:
        X3_dep_res[col] = X3_dep_res[col].astype(float)
y3_dep_res = pd.Series(y3_dep_res)
X3_dep_res_train = X3_dep_res[[c for c in X3_dep_res.columns if c not in ['pid','most_used_app']]]
X3_dep_res_mua_dummy = pd.get_dummies(X3_dep_res['most_used_app'])
X3_dep_res_train = pd.concat([X3_dep_res_train,X3_dep_res_mua_dummy],axis=1,copy=True)


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
    accuracies = []
    for p in pids:
        X_train = X.loc[data['pid']!=p]
        X_test = X.loc[data['pid']==p]
        y_train = y.loc[data['pid']!=p]
        y_test = y.loc[data['pid']==p]
        model_final.fit(X_train,y_train)
        
        out_pred  = model_final.predict(X_test)
        outdf = pd.DataFrame({'pred':out_pred, 'actual':y_test})
        accuracies.append(outdf)
    
    acc_df = pd.concat(accuracies,copy=True)
    acc_df['acc_ind'] = np.where(acc_df['pred'] == acc_df['actual'],1,0)
    overall_acc = acc_df['acc_ind'].sum()/acc_df.shape[0]
    
    return({'method':method,'accuracy':overall_acc,'outcome':outvar,'feature_set':expvars_label})

class_outputs = []
class_outputs.append(classifyMood(X1_anx_res,X1_anx_res_train,y1_anx_res,'anxiety','fs_set1',method='RF'))
class_outputs.append(classifyMood(X1_dep_res,X1_dep_res_train,y1_dep_res,'depression','fs_set1',method='RF'))
class_outputs.append(classifyMood(X2_anx_res,X2_anx_res_train,y2_anx_res,'anxiety','fs_set2',method='RF'))
class_outputs.append(classifyMood(X2_dep_res,X2_dep_res_train,y2_dep_res,'depression','fs_set2',method='RF'))
class_outputs.append(classifyMood(X3_anx_res,X3_anx_res_train,y3_anx_res,'anxiety','fs_set3',method='RF'))
class_outputs.append(classifyMood(X3_dep_res,X3_dep_res_train,y3_dep_res,'depression','fs_set3',method='RF'))

class_outputs.append(classifyMood(X1_anx_res,X1_anx_res_train,y1_anx_res,'anxiety','fs_set1',method='XGB'))
class_outputs.append(classifyMood(X1_dep_res,X1_dep_res_train,y1_dep_res,'depression','fs_set1',method='XGB'))
class_outputs.append(classifyMood(X2_anx_res,X2_anx_res_train,y2_anx_res,'anxiety','fs_set2',method='XGB'))
class_outputs.append(classifyMood(X2_dep_res,X2_dep_res_train,y2_dep_res,'depression','fs_set2',method='XGB'))
class_outputs.append(classifyMood(X3_anx_res,X3_anx_res_train,y3_anx_res,'anxiety','fs_set3',method='XGB'))
class_outputs.append(classifyMood(X3_dep_res,X3_dep_res_train,y3_dep_res,'depression','fs_set3',method='XGB'))

class_outputs_df = pd.DataFrame(class_outputs)
class_outputs_df.to_csv('/Users/lihuacai/Desktop/anna_class_outcomes.csv',index=False)


#outcomes methods
#daily score prediction, and aggregate to get a weekly score
#baseline: straightforward prediction









dvarnames = ['pid',
'dayofstudy',
'weekofstudy',
'loyalty',
'regularity',
'duration',
'duration_mean',
'duration_std',
'duration_min',
'duration_max',
'betweenlaunch_duration_mean',
'betweenlaunch_duration_std',
'loyalty_aspire',
'regularity_aspire',
'duration_aspire',
'duration_mean_aspire',
'duration_std_aspire',
'duration_min_aspire',
'duration_max_aspire',
'betweenlaunch_duration_mean_aspire',
'betweenlaunch_duration_std_aspire',
'loyalty_boostme',
'regularity_boostme',
'duration_boostme',
'duration_mean_boostme',
'duration_std_boostme',
'duration_min_boostme',
'duration_max_boostme',
'betweenlaunch_duration_mean_boostme',
'betweenlaunch_duration_std_boostme',
'loyalty_dailyfeats',
'regularity_dailyfeats',
'duration_dailyfeats',
'duration_mean_dailyfeats',
'duration_std_dailyfeats',
'duration_min_dailyfeats',
'duration_max_dailyfeats',
'betweenlaunch_duration_mean_dailyfeats',
'betweenlaunch_duration_std_dailyfeats',
'loyalty_icope',
'regularity_icope',
'duration_icope',
'duration_mean_icope',
'duration_std_icope',
'duration_min_icope',
'duration_max_icope',
'betweenlaunch_duration_mean_icope',
'betweenlaunch_duration_std_icope',
'loyalty_mantra',
'regularity_mantra',
'duration_mantra',
'duration_mean_mantra',
'duration_std_mantra',
'duration_min_mantra',
'duration_max_mantra',
'betweenlaunch_duration_mean_mantra',
'betweenlaunch_duration_std_mantra',
'loyalty_messages',
'regularity_messages',
'duration_messages',
'duration_mean_messages',
'duration_std_messages',
'duration_min_messages',
'duration_max_messages',
'betweenlaunch_duration_mean_messages',
'betweenlaunch_duration_std_messages',
'loyalty_moveme',
'regularity_moveme',
'duration_moveme',
'duration_mean_moveme',
'duration_std_moveme',
'duration_min_moveme',
'duration_max_moveme',
'betweenlaunch_duration_mean_moveme',
'betweenlaunch_duration_std_moveme',
'loyalty_relax',
'regularity_relax',
'duration_relax',
'duration_mean_relax',
'duration_std_relax',
'duration_min_relax',
'duration_max_relax',
'betweenlaunch_duration_mean_relax',
'betweenlaunch_duration_std_relax',
'loyalty_slumbertime',
'regularity_slumbertime',
'duration_slumbertime',
'duration_mean_slumbertime',
'duration_std_slumbertime',
'duration_min_slumbertime',
'duration_max_slumbertime',
'betweenlaunch_duration_mean_slumbertime',
'betweenlaunch_duration_std_slumbertime',
'loyalty_socialforce',
'regularity_socialforce',
'duration_socialforce',
'duration_mean_socialforce',
'duration_std_socialforce',
'duration_min_socialforce',
'duration_max_socialforce',
'betweenlaunch_duration_mean_socialforce',
'betweenlaunch_duration_std_socialforce',
'loyalty_thoughtchallenger',
'regularity_thoughtchallenger',
'duration_thoughtchallenger',
'duration_mean_thoughtchallenger',
'duration_std_thoughtchallenger',
'duration_min_thoughtchallenger',
'duration_max_thoughtchallenger',
'betweenlaunch_duration_mean_thoughtchallenger',
'betweenlaunch_duration_std_thoughtchallenger',
'loyalty_worryknot',
'regularity_worryknot',
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
'anx_mood',
'dep_mood']


       
wvarnames = ['pid',
'weekofstudy',
'loyalty',
'regularity',
'duration',
'duration_mean',
'duration_std',
'duration_min',
'duration_max',
'betweenlaunch_duration_mean',
'betweenlaunch_duration_std',
'loyalty_aspire',
'regularity_aspire',
'duration_aspire',
'duration_mean_aspire',
'duration_std_aspire',
'duration_min_aspire',
'duration_max_aspire',
'betweenlaunch_duration_mean_aspire',
'betweenlaunch_duration_std_aspire',
'loyalty_boostme',
'regularity_boostme',
'duration_boostme',
'duration_mean_boostme',
'duration_std_boostme',
'duration_min_boostme',
'duration_max_boostme',
'betweenlaunch_duration_mean_boostme',
'betweenlaunch_duration_std_boostme',
'loyalty_dailyfeats',
'regularity_dailyfeats',
'duration_dailyfeats',
'duration_mean_dailyfeats',
'duration_std_dailyfeats',
'duration_min_dailyfeats',
'duration_max_dailyfeats',
'betweenlaunch_duration_mean_dailyfeats',
'betweenlaunch_duration_std_dailyfeats',
'loyalty_icope',
'regularity_icope',
'duration_icope',
'duration_mean_icope',
'duration_std_icope',
'duration_min_icope',
'duration_max_icope',
'betweenlaunch_duration_mean_icope',
'betweenlaunch_duration_std_icope',
'loyalty_mantra',
'regularity_mantra',
'duration_mantra',
'duration_mean_mantra',
'duration_std_mantra',
'duration_min_mantra',
'duration_max_mantra',
'betweenlaunch_duration_mean_mantra',
'betweenlaunch_duration_std_mantra',
'loyalty_messages',
'regularity_messages',
'duration_messages',
'duration_mean_messages',
'duration_std_messages',
'duration_min_messages',
'duration_max_messages',
'betweenlaunch_duration_mean_messages',
'betweenlaunch_duration_std_messages',
'loyalty_moveme',
'regularity_moveme',
'duration_moveme',
'duration_mean_moveme',
'duration_std_moveme',
'duration_min_moveme',
'duration_max_moveme',
'betweenlaunch_duration_mean_moveme',
'betweenlaunch_duration_std_moveme',
'loyalty_relax',
'regularity_relax',
'duration_relax',
'duration_mean_relax',
'duration_std_relax',
'duration_min_relax',
'duration_max_relax',
'betweenlaunch_duration_mean_relax',
'betweenlaunch_duration_std_relax',
'loyalty_slumbertime',
'regularity_slumbertime',
'duration_slumbertime',
'duration_mean_slumbertime',
'duration_std_slumbertime',
'duration_min_slumbertime',
'duration_max_slumbertime',
'betweenlaunch_duration_mean_slumbertime',
'betweenlaunch_duration_std_slumbertime',
'loyalty_socialforce',
'regularity_socialforce',
'duration_socialforce',
'duration_mean_socialforce',
'duration_std_socialforce',
'duration_min_socialforce',
'duration_max_socialforce',
'betweenlaunch_duration_mean_socialforce',
'betweenlaunch_duration_std_socialforce',
'loyalty_thoughtchallenger',
'regularity_thoughtchallenger',
'duration_thoughtchallenger',
'duration_mean_thoughtchallenger',
'duration_std_thoughtchallenger',
'duration_min_thoughtchallenger',
'duration_max_thoughtchallenger',
'betweenlaunch_duration_mean_thoughtchallenger',
'betweenlaunch_duration_std_thoughtchallenger',
'loyalty_worryknot',
'regularity_worryknot',
'duration_worryknot',
'duration_mean_worryknot',
'duration_std_worryknot',
'duration_min_worryknot',
'duration_max_worryknot',
'betweenlaunch_duration_mean_worryknot',
'betweenlaunch_duration_std_worryknot',
'afternoon_loyalty_aspire',
'evening_loyalty_aspire',
'latenight_loyalty_aspire',
'morning_loyalty_aspire',
'latenight_regularity_aspire',
'morning_regularity_aspire',
'afternoon_regularity_aspire',
'evening_regularity_aspire',
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
'afternoon_loyalty_boostme',
'evening_loyalty_boostme',
'latenight_loyalty_boostme',
'morning_loyalty_boostme',
'latenight_regularity_boostme',
'morning_regularity_boostme',
'afternoon_regularity_boostme',
'evening_regularity_boostme',
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
'afternoon_loyalty_dailyfeats',
'evening_loyalty_dailyfeats',
'latenight_loyalty_dailyfeats',
'morning_loyalty_dailyfeats',
'latenight_regularity_dailyfeats',
'morning_regularity_dailyfeats',
'afternoon_regularity_dailyfeats',
'evening_regularity_dailyfeats',
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
'afternoon_loyalty_icope',
'evening_loyalty_icope',
'latenight_loyalty_icope',
'morning_loyalty_icope',
'latenight_regularity_icope',
'morning_regularity_icope',
'afternoon_regularity_icope',
'evening_regularity_icope',
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
'afternoon_loyalty_mantra',
'evening_loyalty_mantra',
'latenight_loyalty_mantra',
'morning_loyalty_mantra',
'latenight_regularity_mantra',
'morning_regularity_mantra',
'afternoon_regularity_mantra',
'evening_regularity_mantra',
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
'afternoon_loyalty_messages',
'evening_loyalty_messages',
'latenight_loyalty_messages',
'morning_loyalty_messages',
'latenight_regularity_messages',
'morning_regularity_messages',
'afternoon_regularity_messages',
'evening_regularity_messages',
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
'afternoon_loyalty_moveme',
'evening_loyalty_moveme',
'latenight_loyalty_moveme',
'morning_loyalty_moveme',
'latenight_regularity_moveme',
'morning_regularity_moveme',
'afternoon_regularity_moveme',
'evening_regularity_moveme',
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
'afternoon_loyalty_relax',
'evening_loyalty_relax',
'latenight_loyalty_relax',
'morning_loyalty_relax',
'latenight_regularity_relax',
'morning_regularity_relax',
'afternoon_regularity_relax',
'evening_regularity_relax',
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
'afternoon_loyalty_slumbertime',
'evening_loyalty_slumbertime',
'latenight_loyalty_slumbertime',
'morning_loyalty_slumbertime',
'latenight_regularity_slumbertime',
'morning_regularity_slumbertime',
'afternoon_regularity_slumbertime',
'evening_regularity_slumbertime',
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
'afternoon_loyalty_socialforce',
'evening_loyalty_socialforce',
'latenight_loyalty_socialforce',
'morning_loyalty_socialforce',
'latenight_regularity_socialforce',
'morning_regularity_socialforce',
'afternoon_regularity_socialforce',
'evening_regularity_socialforce',
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
'afternoon_loyalty_thoughtchallenger',
'evening_loyalty_thoughtchallenger',
'latenight_loyalty_thoughtchallenger',
'morning_loyalty_thoughtchallenger',
'latenight_regularity_thoughtchallenger',
'morning_regularity_thoughtchallenger',
'afternoon_regularity_thoughtchallenger',
'evening_regularity_thoughtchallenger',
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
'afternoon_loyalty_worryknot',
'evening_loyalty_worryknot',
'latenight_loyalty_worryknot',
'morning_loyalty_worryknot',
'latenight_regularity_worryknot',
'morning_regularity_worryknot',
'afternoon_regularity_worryknot',
'evening_regularity_worryknot',
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
'anx_mood',
'dep_mood',
'active',
'support_others',
'healthy_food',
'manage_neg',
'num_apps_used']






'''Mehdi's paper
Using the extracted mobility features, we investigated whether the mobility features could
 predict students’ SIAS scores. We studied the results of (1) the classification task that 
 classified participants as low or high socially anxious using SIAS score=34 as a cutoff and 
 (2) the regression task by predicting the actual SIAS score of a participant (between 0 and 80).
 
 For each day, we extracted 220 mobility features as follows: (1) cumulative staying time at each 
 different location and during each different time window: 8 am-4 pm, 4 pm-12 am, and 12 am-8 am; 
 (2) the distribution of visits over time (time series of locations visited during that day); (3) 
 location entropy; (4) the frequency of each different transition; (5) the type of day (weekday or weekend); 
 and (6) the day of the week (eg, Monday, Tuesday). 
 
Theoretically, daily mobility features would all be included in the same model as predictors of 
 trait social anxiety. However, in practice, this is not feasible because of the large number of 
 dimensions for a small number of participants. In other words, for each day, each different place, 
 each transition type, and each time window (morning, evening, etc) would be a different feature. 
 Thus, the total number of features will increase with the number of days in the study. 
 This phenomenon is known as the curse of dimensionality where the volume of the space increases so 
 fast that the available data become sparse [29]. There are two traditional solutions to tackle this problem.
 The first is to aggregate the features on daily basis, which means, instead of having more than 3000 features for 
 228 participants, we determine only 220 aggregated features (average time spent at home, at university, 
 etc during the study period). Thus, 220 is the number of distinct mobility features that we may have in a 
 given day. We utilized this approach as one of our baseline measures (BM1). The second solution is to apply 
 a dimensionality reduction technique such as principal component analysis (PCA) or autoencoders to select the
 most important features. We also applied this method as one of our baseline measure comparisons (BM2), 
 reducing our feature space to between 50 and 200.

Finally, we proposed a new method to predict a daily SIAS score, which uses a neural network with all 220 
mobility features and compare it to the two traditional solutions (BM1 and BM2). Figure 6 describes the design 
of our method. We started by predicting candidate SIAS scores for each participant for each day of the study 
(ie, if a participant had 15 days, we predicted 15 candidate SIAS scores); then, in the second layer, a global 
predicted SIAS score was calculated by aggregating the daily candidate SIAS scores. For the regression task, the 
aggregation function used a 7% trimmed mean of the predicted daily SIAS scores; the trimmed mean helps eliminate 
the influence of predictions on the tails that may unfairly affect the traditional mean. However, 
for the classification task, the dominant class was chosen. If there was no dominant class (number of days 
predicted as low and high were the same), the aggregation function chose one class randomly (See aggregation 
in Figure 6).
'''


'''Phil's paper
In all statistical models, state affect served as the independent variable and time spent at home was the 
dependent variable. Social anxiety and depression symptoms (using the SIAS and the depression subscale of 
the DASS-21) were entered as the moderators. We examined 4 different models. The first model examined 
associations between change in state affect (we examined positive and negative affect separately) across a 
time window lasting up to 4 hours and time spent at home during that same time window. Change in state affect
 was computed as the difference between self-reported affect from the start to the end of a time window 
 (computed separately for positive and negative affect). We based the decision to use a window length of 
 up to 4 hours on any 2 random time prompts being timed to go off at a maximum of 4 hours from one another 
 (note that only about 12%, 329/2741, of the total random time surveys were rendered unanalyzable because 
 they were too far apart in timing from another survey). The next 3 models examined the associations between 
 mean-level positive and negative state affect and ratio of time spent at home over the course of a 
 typical workday (10:00 AM to 6:00 PM). This was done for models in which state affect was associated with 
 (1) time spent at home the same day, (2) time spent at home the following day, and (3) time spent at home 
 the previous day (models testing prior day homestay have a predictor that follows the outcome, although we 
 wish to stress that all models tested are correlational).

Due to skewed distribution of the time-spent-at-home variable, we computed 2 sets of analyses using generalized 
mixed-effects models. In 1 set of analyses, using mixed-effects regression, we entered time spent at home as a 
continuous variable. Time spent at home scores were log transformed to address right skew. For these analyses, 
we computed time spent at home as a ratio of the percentage of time an individual spent at home during a set 
window (ie, within a 4-hour period or over the course of a day), relative to the average percentage of time 
that individual spent at home over the entire study period. Thus, the ratio provides an indicator of whether 
an individual spent more or less time at home over a predefined window relative to the amount of time they 
typically spent at home during the study. Another set of analyses, using mixed-effects logistic regression, 
examined the likelihood that an individual was at home at some point during that window (in this case, we 
treated time spent at home as binary, where 1=spent some time at home during predetermined window, and 0=did 
not spend any time at home during that window).

We conducted all analyses using generalized mixed-effects models and fitted them using the lme4 package in 
R 3.3.2 (R Foundation) [22]. We computed effect sizes representing the amount of variance explained by the 
fixed effects in our generalized mixed-effects models using the MuMIn package in R [23]. For analyses with 
mixed-effects logistic regression, in which time spent at home was a dichotomous variable, we report the 
unstandardized betas from our models [24]. We used generalized mixed-effects models because they can account 
for changes over time and missing data more effectively than repeated-measures analyses of variance. 
In all models, we entered subject and day as separate random intercepts to account for differences in mean 
responses between participants and between days. For analyses examining time windows within a day, to control 
for differences in the length of time windows, we entered time window length as a random slope in the analyses.

An example of the model with time spent at home as the criterion variable, state negative affect and depression 
as predictor variables, and subject as the random intercept is the equation 
Tsi = β0 + β1(NA) + β2(Dep) + β3(NA × Dep) + S0s + esi, where S0s~ N (0, τ200) and esi~ N (0, σ2), and 
where T is time spent at home, NA is state negative affect, Dep is trait depression, and S0s is the random intercept.

Due to the amount of time spent at home during a given time window being evaluated relative to each participants’ 
personal average time spent at home (ie, a within-subject ratio) for the continuous measure, and the difficulties 
with interpreting prediction of prior versus same versus next day affect when the predictor was only measured 
at baseline, we computed a separate set of between-subjects analyses to examine the main effects of depression 
and social anxiety predicting both time spent at home (without accounting for each participants’ personal average 
time spent at home) and likelihood of time spent at home. This was done using mixed-effects regression models 
in which time spent at home (as a continuous or dichotomous variable) was the response variable, depression and 
social anxiety were predictor variables, and subject and day were random intercepts. We computed separate models 
for 4-hour time windows and between 10:00 AM and 6:00 PM.
'''






