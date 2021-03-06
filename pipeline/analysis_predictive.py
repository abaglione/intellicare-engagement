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
from imblearn.over_sampling import SMOTENC
import xgboost
import shap
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from . import transform
from . import metrics

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
def genMixedLM(df, outvar, expfeats, gpvar, fsLabel, alpha=0.5, random_state=1008):

    # Drop rows where target is NaN - should never impute these!
    df.dropna(subset=[outvar], how='any', inplace=True)

    # Do imputation  
    print('Imputing missing vars')
    imputer = IterativeImputer(max_iter=50, random_state=random_state)
    df_imp = imputer.fit_transform(df)
    index = df.index
    cols = df.columns
    df = pd.DataFrame(np.round(df_imp), index=index, columns = cols)
    
    #add the intercept columns for the linear mixed model
    df['intercept'] = 1
    np.random.seed(random_state)

    print('Fitting Mixed Linear Model.')
    mixedmodel = MixedLM(endog=df[outvar].astype(float), exog=df[expfeats].astype(
        float), groups=df[gpvar], exog_re=df['intercept'])
    modelres = mixedmodel.fit_regularized(method='l1', alpha=alpha, disp=1)
    rdf = pd.DataFrame({'expvar': modelres.params.index, 'coef': modelres.params,
                        'tvalue': modelres.tvalues, 'pvalues': modelres.pvalues}).reset_index()
    rdf.drop(columns=['index'], inplace=True)
    rdf['feature_set'] = fsLabel
    rdf['alpha'] = alpha
    rdf['outvar'] = outvar

    print('Making predictions.')
    pred_df = pd.DataFrame(modelres.predict(
        df[expfeats].astype(float)), columns=['pred_'+outvar])
    pred_df[outvar] = df[outvar]
    pred_df['diff'] = pred_df['pred_'+outvar] - pred_df[outvar]
    rmse = np.sqrt(np.sum(pred_df['diff']**2)/pred_df.shape[0])
    rdf['rmse'] = rmse

    return(rdf)
def tune_hyperparams(X, y, groups, method, random_state):
    cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=random_state+1)

    # Do gridsearch
    if method == 'LogisticR':
        param_grid = {
            'C': np.logspace(-4, 4, 20),
            'penalty': ['l1'],  # Use LASSO for feature selection
            'solver': ['liblinear']
        }
        model = LogisticRegression(random_state=random_state)

    if method == 'XGB':
        param_grid = {
            'n_estimators': [1, 2, 3],
            'max_depth': [1, 2],
            'min_child_weight': [1, 3],
            'learning_rate': [0.05, 0.1, 0.3]
        }
        model = xgboost.XGBClassifier(random_state=random_state)

    elif method == 'RF':
        param_grid = {
            'n_estimators': [1, 2, 3],
            'max_depth': [1, 2],
            'max_features': [4, 5, 6, 7, 8]
        }
        model = RandomForestClassifier(oob_score=True, random_state=random_state)
    
    print('Getting optimized classifier using gridsearch.')
    grid = GridSearchCV(estimator=model, param_grid=param_grid,
                        cv=cv, scoring='accuracy', n_jobs=3,
                        verbose=3
                        )
    clf = grid.fit(X.values, y.values, groups=groups)
    return clf
    
def classifyMood(X, y, id_col, target, nominal_idx, fs, method, 
                 optimize=False, random_state=1008):
    # Set up outer CV
    outer_cv = LeaveOneGroupOut()

    train_res_all = []
    test_res_all = []
    shap_values_all = list() 
    test_indices_all = list()
    
    # Do prediction task
    for train_indices, test_indices in outer_cv.split(X=X, y=y, groups=X[id_col]):
        X_train, y_train = X.loc[train_indices, :], y[train_indices]
        X_test, y_test = X.loc[test_indices, :], y[test_indices]

        # Do imputation
        imputer = IterativeImputer(random_state=5)
        imputer.fit(X_train)
        X_train = transform.impute(X_train, id_col, imputer)
        X_test = transform.impute(X_test, id_col, imputer)
        
        # Perform upsampling to handle class imbalance
        smote = SMOTENC(random_state=random_state, categorical_features=nominal_idx)
        X_train, y_train, upsampled_groups = transform.upsample(X_train, y_train, id_col, smote)
        
        # Drop the id column from the Xs - IMPORTANT!
        X_train.drop(columns=[id_col], inplace=True)
        X_test.drop(columns=[id_col], inplace=True)

        # Format y
        y_train = pd.Series(y_train)
        y_test = pd.Series(y_test)

        print('Making predictions.')
        if optimize:
            clf = tune_hyperparams(X=X_train, y=y_train, groups=upsampled_groups, method=method, random_state=random_state)
        else:
            if method == 'LogisticR':
                clf = LogisticRegression(solver='liblinear', random_state=random_state)
            elif method == 'RF':
                clf = RandomForestClassifier(max_depth=1, random_state=random_state)
            elif method == 'XGB':
                clf = xgboost.XGBClassifier(random_state=random_state)

        train_pred = clf.predict(X_train.values)
        test_pred = clf.predict(X_test.values)
        train_res_all.append(pd.DataFrame({'pred': train_pred, 'actual': y_train}))
        test_res_all.append(pd.DataFrame({'pred': test_pred, 'actual': y_test}))

        print('Calculating feature importance for this fold.')
        shap_values = metrics.calc_shap(X_train=X_train, X_test=X_test,
                                        model=clf.best_estimator_, method=method)

        shap_values_all.append(shap_values)
        test_indices_all.append(test_indices)

    # Get train and test results as separate dictionaries
    train_perf_metrics = metrics.calc_performance_metrics(pd.concat(train_res_all, copy=True))
    test_perf_metrics = metrics.calc_performance_metrics(pd.concat(test_res_all, copy=True))

    # Save the train accuracy alongside all test metrics - this will help us check for overfitting
    all_res = {
        **{'train_accuracy': train_perf_metrics['accuracy']},
        **{'test_' + str(k): v for k, v in test_perf_metrics.items()}
    }
    
    all_res.update({'method': method, 'target': target, 'feature_set': fs, 
                    'n_observations': X.shape[0], 'n_feats': X.shape[1],
                    'optimized': optimize})

    pd.DataFrame([all_res]).to_csv('results/pred_res.csv', mode='a', index=False)

    ''' Don't forget to drop the groups (id) col.
        Otherwise, we'll have issues with alignment.'''
    X_test, shap_values = metrics.gather_shap(
        X=X.drop(columns=[id_col]), method=method, 
        shap_values=shap_values_all, test_indices=test_indices_all)

    filename = fs + '_' + method + 'tuned.ob'
    
    with open('feature_importance/X_test_' + filename, 'wb') as fp:
        pickle.dump(X_test, fp)

    with open('feature_importance/shap_' + filename, 'wb') as fp:
        pickle.dump(shap_values, fp)
        
