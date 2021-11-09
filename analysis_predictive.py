#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 20:14:41 2020
Last Updated Thurs Nov 11 2020

@author: Lihua Cai (lihuacai)
@author: Anna Baglione (abaglione)

Some functions included from the following project: https://www.github.com/abaglione/bcpn-mems
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
from imblearn.over_sampling import SMOTENC
import xgboost
import shap
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# --------- Display Options ------------
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

# --------- Lee Notes ------------
#class imbalance, missing data (especially response var)
#https://scikit-learn.org/stable/modules/impute.html#:~:text=Missing%20values%20can%20be%20imputed,for%20different%20missing%20values%20encodings.&text=%3E%3E%3E%20import%20numpy%20as%20np%20%3E%3E%3E%20from%20sklearn.
#https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html

# --------- Functions ------------

def fillmissing(df, cols, val, fillval):
    for c in cols:
        df[c].loc[df[c] == val] = fillval

def rename_mapper(colname):
    new_colname = colname.replace('_'+all_feats['most_used_app'][i], '')
    return(new_colname)

def get_performance_metrics(df, actual='actual', pred='pred'):
    stats = {}

    stats['accuracy'] = accuracy_score(y_true=df[actual], y_pred=df[pred])
    
    precision, recall, f1_score, support = precision_recall_fscore_support(
        y_true=df[actual], y_pred=df[pred], average='macro'
    )
    
    stats.update({'precision': precision, 'recall': recall, 
                  'f1_score': f1_score, 'support': support
                 })

    return stats

def calc_shap(X_train, X_test, model, method):
    shap_values = None
    
    if method == 'LogisticR':
        shap_values = shap.LinearExplainer(model, X_train).shap_values(X_test)
    elif method == 'RF' or method == 'XGB':
        shap_values = shap.TreeExplainer(model).shap_values(X_test)
    elif method == 'SVM':
        X_train_sampled = shap.sample(X_train, 5)
        shap_values = shap.KernelExplainer(model.predict_proba, X_train_sampled).shap_values(X_test)

    return shap_values

def gather_shap(X, method, shap_values, test_indices):
    print('Gathering SHAP stats.')

    # https://lucasramos-34338.medium.com/visualizing-variable-importance-using-shap-and-cross-validation-bd5075e9063a

    # Combine results from all iterations
    test_indices_all = test_indices[0]
    shap_values_all = np.array(shap_values[0])

    for i in range(1, len(test_indices)):
        test_indices_all = np.concatenate((test_indices_all, test_indices[i]), axis=0)
        
        if method == 'RF' or method == 'SVM': # classifiers with multiple outputs
            shap_values_all = np.concatenate(
                (shap_values_all, np.array(shap_values[i])), axis=1)
        else:
            shap_values_all = np.concatenate((shap_values_all, shap_values[i]), axis=0)

    # Bring back variable names
    X_test = pd.DataFrame(X.iloc[test_indices_all], columns=X.columns)

    return X_test, shap_values_all

def genMixedLM(df, outvar, expfeats, gpvar, fsLabel, alpha=0.5):
    mixedmodel = MixedLM(endog=df[outvar].astype(float), exog=df[expfeats].astype(
        float), groups=df[gpvar], exog_re=df['intercept'])
    modelres = mixedmodel.fit_regularized(method='l1', alpha=alpha)
    rdf = pd.DataFrame({'expvar': modelres.params.index, 'coef': modelres.params,
                        'tvalue': modelres.tvalues, 'pvalues': modelres.pvalues}).reset_index()
    rdf.drop(columns=['index'], inplace=True)
    rdf['feature_set'] = fsLabel
    rdf['alpha'] = alpha
    rdf['outvar'] = outvar

    pred_df = pd.DataFrame(modelres.predict(
        df[expfeats].astype(float)), columns=['pred_'+outvar])
    pred_df[outvar] = df[outvar]
    pred_df['diff'] = pred_df['pred_'+outvar] - pred_df[outvar]
    rmse = np.sqrt(np.sum(pred_df['diff']**2)/pred_df.shape[0])
    rdf['rmse'] = rmse

    return(rdf)

def classifyMood(X, y, id, target, nominal_idx, fs, method, random_state=1008):
    # Set up outer CV
    outer_cv = StratifiedGroupKFold(n_splits=5, shuffle=True,
                                    random_state=random_state)

    inner_cv = StratifiedGroupKFold(n_splits=5, shuffle=True,
                                    random_state=random_state)

    test_res_all = []
    shap_values_all = list() 
    test_indices_all = list()

    # Do prediction task
    for train_index, test_index in outer_cv.split(X=X, y=y, groups=X[id]):
        X_train, y_train = X.loc[train_index, :], y[train_index]
        X_test, y_test = X.loc[test_index, :], y[test_index]

        # Perform upsampling to handle class imbalance
        print('Conducting upsampling with SMOTE.')
        cols = X_train.columns

        try:
            smote = SMOTENC(random_state=random_state, categorical_features=nominal_idx)
            X_train_upsampled, y_train_upsampled = smote.fit_resample(X_train, y_train)
        except ValueError:       
            # Set n_neighbors = n_samples
            # Not great if we have a really small sample size. Hmm.
            k_neighbors = (y_train == 1).sum() - 1
            print('%d neighbors for SMOTE' % k_neighbors)
            smote = SMOTENC(random_state=random_state, categorical_features=nominal_idx,
                            k_neighbors=k_neighbors)
            print(smote)
            X_train_upsampled, y_train_upsampled = smote.fit_resample(X_train, y_train)

        X_train = pd.DataFrame(X_train_upsampled, columns=cols, dtype=float)

        # Save the upsampled groups array
        upsampled_groups = X_train[id]

        # Drop this column from the Xs - IMPORTANT!
        X_train.drop(columns=[id], inplace=True)
        X_test.drop(columns=[id], inplace=True)

        # Format y
        y_train = pd.Series(y_train_upsampled)

        ''' Perform Scaling
            Thank you for your guidance, @Miriam Farber
            https://stackoverflow.com/questions/45188319/sklearn-standardscaler-can-effect-test-matrix-result
        '''
        print('Performing MinMax scaling.')
        scaler = MinMaxScaler(feature_range=(0, 1))

        X_train_scaled = scaler.fit_transform(X_train)
        index = X_train.index
        cols = X_train.columns
        X_train = pd.DataFrame(X_train_scaled, index=index, columns=cols)

        X_test_scaled = scaler.fit_transform(X_test)
        index = X_test.index
        cols = X_test.columns
        X_test = pd.DataFrame(X_test_scaled, index=index, columns=cols)

        # Do gridsearch
        if method == 'XGB':
            param_grid = {
                'n_estimators': [50, 100, 200, 500],
                'max_depth': [3, 6, 9],
                'min_child_weight': [1, 3, 6],
                'learning_rate': [0.05, 0.1, 0.3, 0.5]
            }
            model = xgboost.XGBClassifier(random_state=random_state)

        elif method == 'RF':
            param_grid = {
                'n_estimators': [50, 100, 200, 500],
                'max_depth': [1, 2, 5, 10],
                'max_features': [8, 10, 12],
            }
            model = RandomForestClassifier(oob_score=True, random_state=random_state)
        
        print('Getting optimized classifier using gridsearch.')
        grid = GridSearchCV(estimator=model, param_grid=param_grid,
                            cv=inner_cv, scoring='accuracy', n_jobs=2,
                        )
        clf = grid.fit(X_train.values, y_train.values, groups=upsampled_groups)
        
        print('Making predictions.')
        pred = clf.predict(X_test.values)
        test_res_all.append(pd.DataFrame({'pred': pred, 'actual': y_test}))

        print('Calculating feature importance for this fold.')
        shap_values = calc_shap(X_train=X_train, X_test=X_test,
                                model=clf.best_estimator_, method=method)

        shap_values_all.append(shap_values)
        test_indices_all.append(test_index)

    all_res = pd.concat(test_res_all, copy=True)
    all_res = get_performance_metrics(all_res)
    
    all_res.update({'method': method, 'target': target, 'feature_set': fs, 
                    'n_observations': X.shape[0], 'n_feats': X.shape[1]})

    pd.DataFrame([all_res]).to_csv('results/pred_res.csv', mode='a', index=False)

# ---------------------------------------------------------------------------

# Read in the weekly feature vectors 
wkly_df = pd.read_csv('features/all_ind_wkly.csv')

# Store the feature names
featnames = list(wkly_df.columns)

#imputing missing survey data -- weekly data
impfeats = ['cope_alcohol_tob', 'physical_pain', 'connected', 'receive_support', 'anx',
            'dep', 'active', 'support_others', 'healthy_food']
featnames_app = [i for i in featnames if i not in impfeats]
feats_app = wkly_df.drop(axis=1, columns=impfeats)
feats_napp = wkly_df[impfeats].copy()
fillmissing(feats_napp, impfeats, -1, np.nan)

imputer = IterativeImputer(max_iter=50, random_state=1008, add_indicator=True)
imputer.fit(feats_napp)
impfeats_ind = [i+'_ind' for i in impfeats]
impfeats_c = copy.deepcopy(impfeats)
impfeats_c.extend(impfeats_ind)
feats_napp = pd.DataFrame(
    np.round(imputer.transform(feats_napp)), columns=impfeats_c)
all_feats = pd.concat([feats_app, feats_napp], copy=True, axis=1)

frequency_feats = [n for n in featnames if 'frequency' in n]
reg_feats = [n for n in featnames if 'daysofuse' in n]
dur_feats = [n for n in featnames if 'duration' in n and 'betweenlaunch' not in n]
lau_dur_feats = [n for n in featnames if 'betweenlaunch' in n]

fillmissing(all_feats, frequency_feats, -1, 0)
fillmissing(all_feats, reg_feats, -1, 0)
fillmissing(all_feats, dur_feats, -1, 0)
fillmissing(all_feats, lau_dur_feats, -1, 3600*24*7)

#add the intercept columns for the linear mixed model
all_feats['intercept'] = 1

#outcomes transformation -- anx, dep
#week to week change as outcome
#change to baseline level as outcome
#instead of difference, consider ratio between the weekly value and the baseline
#global average being subtracted

all_feats['anx'].hist()
all_feats['dep'].hist()

#add classification outcomes
all_feats['dep_cat'] = np.where(all_feats['dep'] >= 4, 1, 0)
all_feats['anx_cat'] = np.where(all_feats['anx'] >= 3, 1, 0)

# ------ Feature Set Spec -----------

APPS = ['aspire', 'boostme', 'dailyfeats', 'icope', 'mantra', 'messages',
        'moveme', 'relax', 'slumbertime', 'thoughtchallenger', 'worryknot']
ENGAGEMENT_METRICS = ['frequency', 'duration',
                      'betweenlaunch_duration', 'days_of_use']
TIMES_OF_DAY = ['morning', 'afternoon', 'evening', 'latenight']

# Survey Features Only
survey_fs_cols = ['cope_alcohol_tob', 'physical_pain', 'connected', 'receive_support', 'active',
                  'support_others', 'healthy_food', 'cope_alcohol_tob_ind', 'physical_pain_ind',
                  'connected_ind', 'receive_support_ind', 'active_ind', 'support_others_ind',
                  'healthy_food_ind']

# App Features - All Apps
app_overall_fs_cols = ['weekofstudy', 'frequency', 'daysofuse', 'duration', 'duration_mean',
                       'duration_std', 'duration_min', 'duration_max', 'betweenlaunch_duration_mean',
                       'betweenlaunch_duration_std', 'num_apps_used']

# App Features - Individual Apps
app_ind_fs_cols = ['weekofstudy'] + \
    [col for col in all_feats.columns
     if any([app in col for app in APPS])
     and any([metric in col for metric in ENGAGEMENT_METRICS])
     and not any([tod in col for tod in TIMES_OF_DAY])]

# Add one last feature - an indicator of which app was used most often
df = all_feats[[i for i in app_ind_fs_cols if 'frequency' in i]].copy()
all_feats['most_used_app'] = [i[1] for i in df.idxmax(axis=1).str.split('_')]

#dummitize the most_used_app column
mua_dummy_df = pd.get_dummies(all_feats['most_used_app'])
mua_dummy_cols = list(mua_dummy_df.columns)
all_feats = pd.concat([all_feats,mua_dummy_df],axis=1)

# Create a subset with survey features + only features from the most used app
mua_dfs = []
for i in range(all_feats.shape[0]):
    df = all_feats[[
        e for e in app_ind_fs_cols if all_feats['most_used_app'][i] in e]].iloc[[i]].copy()
    df.rename(mapper=rename_mapper, axis=1, inplace=True)
    mua_dfs.append(df)

app_mua_feats = pd.concat(mua_dfs, sort=False)
survey_app_mua_feats = pd.concat(
    [all_feats[['pid', 'weekofstudy', 'anx', 'dep','anx_cat', 'dep_cat', 'most_used_app']],
     all_feats[survey_fs_cols],
     app_mua_feats], 
    axis=1, copy=True
)

# Create last featureset
# App Features - Only Features from the Most Used App for a Given Observation (Row)
app_mua_fs_cols = ['weekofstudy', 'frequency', 'daysofuse', 
                   'duration', 'duration_mean', 'duration_std',
                   'duration_min', 'duration_max', 'betweenlaunch_duration_mean', 
                   'betweenlaunch_duration_std'] + mua_dummy_cols

# Add new dummy columns to other featuresets
app_overall_fs_cols += mua_dummy_cols
app_ind_fs_cols += mua_dummy_cols


######regression tasks on 1-5 scale (cut off on both 1 (floor) and 5 (ceiling)) using lasso linear mixed effect model;
m1_anx = MixedLM(all_feats['anx'].astype(float), all_feats[app_overall_fs_cols].astype(
    float), all_feats['pid'], all_feats['intercept'])
r1_anx = m1_anx.fit_regularized(method='l1', alpha=0.2)
r1_anx.params

pred_df = pd.DataFrame(r1_anx.predict(
    all_feats[app_overall_fs_cols].astype(float)), columns=['pred_anx'])
pred_df['anx'] = all_feats['anx']
pred_df['diff'] = pred_df['pred_anx'] - pred_df['anx']
rmse = np.sqrt(np.sum(pred_df['diff']**2)/pred_df.shape[0])

alpha_list = np.arange(0.1, 0.81, 0.1)

lmm_res = []

featuresets = {
    'survey_fs': survey_fs_cols,
    'app_overall_fs': app_overall_fs_cols,
    'app_ind_fs': app_ind_fs_cols,
    'app_mua_fs': app_mua_fs_cols,
    'survey_app_overall_fs': survey_fs_cols+app_overall_fs_cols, 
    'survey_app_ind_fs': survey_fs_cols+app_ind_fs_cols,
    'survey_app_mua_fs': survey_fs_cols+app_mua_fs_cols
}

for alpha in alpha_list:
   print('alpha: {0}'.format(alpha))
   for fs_name, fs_cols in featuresets.items():
       if 'mua' in fs_name:
           df = survey_app_mua_feats
       else:
           df = all_feats

       for target in ['anx', 'dep']:
           res = genMixedLM(df, target, ['intercept'] + fs_cols,
                            'pid', fs_name, alpha=alpha)
           lmm_res.append(res.copy())

lmm_res = pd.concat(lmm_res, copy=True, ignore_index=True, sort=False)
lmm_res.to_csv('results/lmm_res.csv', index=False)

id_col = 'pid'
target_cols = ['anx_cat', 'dep_cat']

for fs_name, fs_cols in featuresets.items():

    if 'mua' not in fs_name:
        df = all_feats
    else:
        # Handle special cases in which we want data only from the most used app
        df = survey_app_mua_feats

    X = df[[id_col] + fs_cols].copy()
    
    ''' If this is a featureset with app features 
        Get a list of one-hot-encoded columns from the most_used_app feature.'''
    mua_onehots = [col for col in X.columns if 'most_used_app' in col]
    
    print(X.columns)
    # Get categorical feature indices - will be used with SMOTENC later
    nominal_idx = sorted([X.columns.get_loc(c) for c in ['pid'] + mua_onehots])

    # y
    targets = {
        'anxiety': df['anx_cat'].copy(),
        'depression': df['dep_cat'].copy()
    }

    for target_name, target_col in targets.items():
        for method in ['RF', 'XGB']:
            res = classifyMood(X=X, y=target_col, id=id_col, target=target_name,
                              nominal_idx = nominal_idx, fs=fs_name, method=method)
