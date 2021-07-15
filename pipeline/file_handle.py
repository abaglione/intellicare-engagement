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
import numpy as np
import pandas as pd
from .consts import *

def create_file_dictionary(rootdir):
    """
    Creates a nested dictionary that represents the folder structure of rootdir
    """
    dir = {}
    rootdir = rootdir.rstrip(os.sep)
    start = rootdir.rfind(os.sep) + 1
    for path, dirs, files in os.walk(rootdir):
        rootdir = rootdir.rstrip(os.sep)
        start = rootdir.rfind(os.sep) + 1
        for path, dirs, files in os.walk(rootdir):
            folders = path[start:].split(os.sep)
            subdir = {}
            for filename in files:
                try:
                    df = pd.read_csv(path + '/' + filename, parse_dates=True)
                    df.drop(columns=['Unnamed: 0'], inplace=True, errors="ignore") 
                    corrected_filename = filename.replace(".csv", "")
                    subdir[corrected_filename] = df
                except:
                    continue
            parent = functools.reduce(dict.get, folders[:-1], dir)
            parent[folders[-1]] = subdir
    return dir

def gather_files(data_dir):
    files = list(data_dir.glob('*'))

    print("==== All data files: ====\n", *files, sep="\n")

    datafiles = {}

    for f in files:
        fname = re.sub(r'_clean.*', r'', f.name)
        fname = re.sub(r'_processed.*', r'', fname)
        if 'raw' in fname:
            continue

        datafiles[f] = fname

    print(
        "\n\n==== Selected data files: ====\n", 
        *[x for x in zip(list(datafiles.values()), list(datafiles.keys()))], 
        sep='\n')
    
    return datafiles

def import_processed_files():
    datafiles = gather_files(DATA_PROCESSED_DIR)
    items = list(datafiles.items())
    master_dataset = {}

    for f, fname in items:
        # TODO: Look into why we're getting this unnamed column
        try:
            df = pd.read_csv(f)
            df.drop(columns=[col for col in ['unnamed: 0', 'Unnamed: 0'] if col in df],inplace=True) 
            master_dataset[fname] = df
        except:
            continue
        
    return master_dataset