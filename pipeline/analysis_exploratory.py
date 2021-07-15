import numpy as np
import pandas as pd
from scipy import stats

from pipeline import XLIM, YLIM1

# visualization libraries
import matplotlib as mpl
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import seaborn as sns
sns.set_style("whitegrid")

import matplotlib.pyplot as plt
plt.rcParams.update({'figure.autolayout': True})
# plt.rcParams.update({'figure.facecolor': [1.0, 1.0, 1.0, 1.0]})

def add_group_labels(df, ref_feats):
    """ Helper function for adding trait affect group (anxiety / depression)
    and subgroup (low / high) columns to an existing dataframe, from a reference dataframe.

    Args:
        df (DataFrame): A pandas DataFrame object
        ref_feats (DataFrame): A pandas DataFrame object; holds a list of pids and their corresponding
            trait affect groups and subgroups as recorded at baseline, among other features

    Returns:
        res (DataFrame): A pandas DataFrame object, with the contents of df and two new columns:
          - trait_affect_group
          - trait_affect_subgroup
    """
    
    
    res = pd.DataFrame()
    for group, group_label in pipeline.SYMPTOMGROUP_COL_DISPLAYNAMES.items():

        df2 = df.copy()
        # Record baseline group category (i.e. anxiety or depression)
        df2['trait_affect_group'] = group_label

        # Record subgroup - i.e., whether user was designated as high or low for that group (e.g. high anxious)
        df2['trait_affect_subgroup'] = df2['pid'].apply(
            lambda x: pipeline.match_value(ref_feats, 'pid', x, group))

        if res.empty:
            res = df2
        else:
            res = res.append(df2, ignore_index=True)

    return res


def corr_sig(df=None):
    """ Function for obtaining a matrix of significant p values. Useful if we want a Seaborn
    heatmap with only significant regions.
    
    Credit to Björn B:
    https://stackoverflow.com/questions/57226054/seaborn-correlation-matrix-with-p-values-with-python

    Args:
        df (optional): A pandas DataFrame object

    Returns:
        p_matrix: TODO: Add description here Anna
    """
    
    
    p_matrix = np.zeros(shape=(df.shape[1],df.shape[1]))
    for col in df.columns:
        for col2 in df.drop(col,axis=1).columns:
            _ , p = stats.pearsonr(df[col],df[col2])
            p_matrix[df.columns.to_list().index(col),df.columns.to_list().index(col2)] = p
    return p_matrix


# ----------- OLD FUNCTIONS - Leaving in case I ever need them again ---------------------

# Blahhh I just adapted this then realized we no longer need it - already implemented during feature gen 
def get_moodscore_byweek(features): 
    
    res_full = pd.DataFrame()
    
    for mood in ['anx', 'dep']:
    
        # Grab existing columns from our featureset and convert wide-form columns to long-form
        cols = ['pid'] + [col for col in features.columns if 'state_' + mood in col]
        df = features[cols]
        
        # Reset the cols var to only include mood cols, not pid
        cols = df.columns[1:]

        # First we'll rename the mood columns to numbers (representing weeks 1,2 etc)
        # This lets us use a 1-based index for graphing, beginning at the proper week (Week 1)
        renamed_cols = [str(i + 1) for i, item in enumerate(cols)]
        renaming_rules = dict(zip(cols, renamed_cols))
        df.rename(columns=renaming_rules, inplace=True)

        # Now we'll use the stack function to make things easier on ourselves
        res = df.set_index('pid').stack().reset_index()

        # Have to rename the levels after a stack
        res.rename(columns={'level_1': 'weekofstudy', 0: 'score'}, inplace=True)
        
        res['subscale'] = mood

        # Ensure we're formatting "weekofstudy" correctly
        res['weekofstudy'] = res['weekofstudy'].astype(int)
        
        # Combine the dfs
        if res_full.empty:
            res_full = res
        else:
            res_full.append(res, ignore_index=True)

    # Voila! A dataframe we can easily graph
    return res_full


def plot_lines_ind(scores, features, ax, metric, color_palette, title, legend):
    """ Function for plotting a single engagement metric (e.g. 'duration' or 'frequency'), 
    stratified by app, against mood score. Designed to plot lines for a single participant, in case
    we want to examine individual variations in engagement.
    
    Args:
        scores (DataFrame): A pandas DataFrame object; holds 1 mood score per week of the study
        features (DataFrame): A pandas DataFrame object; holds engagement metrics (e.g. 'duration' or 'frequency')
          for each week of the study
        ax: A matplotlib Axis object; the axis along which to plot
        metric (str): the name of the metric to plot
        color_palette: TODO - add description
        title (str): the title to display on the plit
        legend (bool): whether to display a legend on the plot
        
    """
    
    
    ylab = metric.capitalize()
    
    if metric == "duration":
        features[metric] = features[metric] / 60.0
        ylab = ylab + ' (Minutes)'
    
    ylim2 = [0, max(features[metric]) + 5]
    
    sns.lineplot(x='weekofstudy', 
                 y='mood_score', 
                 data=scores, 
                 ax=ax)
    
    line = ax.lines[0]
    line.set_color('red')
    line.set_linewidth(4)

    ax.set(adjustable='datalim',
           xlim=XLIM, ylim=YLIM1,
           yticks=np.arange(YLIM1[0], YLIM1[1]+1.0, 1.0),
           xlabel="Week of Study", ylabel="Mood Score")
 
    if title:
        
#         # https://stackoverflow.com/questions/52914441/seaborn-title-and-subtitle-placement
        ax.text(x=0.5, y=1.1, 
                s=metric.capitalize(), 
                fontsize=30, 
                ha='center', 
                va='bottom', 
                transform=ax.transAxes)
    
    ax2 = ax.twinx()

    # https://stackoverflow.com/questions/47417286/different-markers-for-each-hue-in-lmplot-seaborn
#     marker = ['o', 'x', '^', '+', '*', '8', 's', 'p', 'D', 'V', '.', '1']

    g = sns.lineplot(x='weekofstudy', 
                     y=metric, 
                     data=features, 
                     ax=ax2, 
                     hue='package',
                     style='package',
                     palette = color_palette                
#                  markers=[marker[i] for i in range(len(features['package'].unique()))],
#                  markersize=10
                    )

#     ax2.legend(title="App", ncol=1).texts[0].set_text('')
#     plt.setp(ax2.get_legend().get_texts(), fontsize='28') # for legend text
#     plt.setp(ax2.get_legend().get_title(), fontsize='30') # for legend title
    
#     box = g.get_position()
#     g.set_position([box.x0, box.y0, box.width * 0.85, box.height]) # resize position

    if legend:
        g.legend(title="App", loc='center right', bbox_to_anchor=(1.75, 0.5), ncol=1).texts[0].set_text('')
        legend = g.get_legend()
        plt.setp(legend.get_texts(), fontsize='20') # for legend text
        plt.setp(legend.get_title(), fontsize='24') # for legend title
        plt.setp(legend.get_lines()[1:], linewidth='4')
    else: 
        g.legend_.remove()
    
    
    ax2.set(adjustable='datalim',
            ylim=ylim2,
            ylabel=ylab)

    for line in ax2.lines:
#         line.set_linestyle('dotted')
        plt.setp(ax2.lines,linewidth=4) 