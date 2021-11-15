import pathlib

# global paths and variables
APP_USERS_DIR = 'app_users_only/'
ALL_USERS_DIR = 'all_users/'

DATA_CLEANED_DIR = pathlib.Path("data/cleaned")
DATA_PROCESSED_DIR = pathlib.Path("data/processed")

# Columns to be renamed during processing
CODEBOOK_MAPPINGS = {
    'q3_1_num': 'cope_alcohol_tob',
    'q3_2_num': 'physical_pain',
    'q3_3_num': 'connected',
    'q3_4_num': 'receive_support',
    'q3_5_num': 'anx',
    'q3_6_num': 'dep',
    'q3_7_num': 'active',
    'q3_8_num': 'support_others',
    'q3_9_num': 'healthy_food',
    'q3_10_num': 'manage_neg',
    'q9': 'feel_tomorrow'
}

# Trait affect score thresholds
PHQ4_THRESH = 3.0
PROMIS_THRESH = 60.0

# Graphing limits
MAX_WKLY_AFFECT = 4
MAX_WEEKS = 7

XLIM = [1, MAX_WEEKS]
YLIM1 = [0, MAX_WKLY_AFFECT + 1]

# Labels for subplots on graphs
SUBPLOT_LABELS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j','k','l']

PHQ4_CODES = {
    'Not at all': 0,
    'Several days': 1,
    'More than half the days': 2,
    'Nearly every day': 3
}
 
"""
Scoring guidelines for the PHQ4, including the main scale and the anxiety
and depression subscales
"""
PHQ4_SCORING = {
    'main_scale': PHQ4_CODES,
    'subscales': {
        'anxiety': {
            'cols': ['q1_1', 'q1_2'],
            'codes': PHQ4_CODES
        },
        'depression': {
            'cols': ['q1_3', 'q1_4'],
            'codes': PHQ4_CODES
        }
    }
}

"""
Names for the PHQ4 and PROMIS anxiety and depression subscales; 
used for labeling exploratory analysis graphs
"""
TRAIT_AFFECT_SUBSCALES = {
    'phq4_anxiety_bl': "PHQ4-A",
    'phq4_depression_bl': "PHQ4-D",
    'promis_anx_bl': "PROMIS-A",
    'promis_dep_bl': "PROMIS-D"
}

""" 
Names for trait affect groups; used for labeling exploratory 
analysis graphs
"""
TRAIT_AFFECT_GROUPS = {
    'trait_anx_group': 'Trait Anxiety Group',
    'trait_dep_group': 'Trait Depression Group'
}  

""" 
Names for the types of affect (mood); used for labeling exploratory 
analysis graphs
"""
WKLY_AFFECT = {
    'anx': 'Weekly Anxiety',
    'dep': 'Weekly Depression'
}  
    
"""
Dictionary used to subdivide data into epochs (time windows) based on
hour of the day (bins)

This was not used in the final analysis due to the small sample size, but 
could be used for larger samples.

"""
EPOCHS = {
    'bins': [-1, 6, 12, 18, 24],
    'labels': ['late_night','morning', 'afternoon', 'evening']
}

# Hacky way of imposing a common file naming scheme
FILENAME_MAPPINGS = {
    'timediv': {
        'time_window': 'tod',
        'weekofstudy': 'wkly'
    },
    'group': dict(zip(list(TRAIT_AFFECT_GROUPS.keys()), ['anx', 'dep'])),
    'metric': {
        'frequency': 'freq',
        'daysofuse': 'daysuse',
        'duration': 'dur'
    },
    'appdiv': {
        'applevel': 'applevel',
        'aggregate': 'agg'
    }
}