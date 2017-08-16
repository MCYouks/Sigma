# -*- coding: utf-8 -*-
'''

The objective is to build all the functions to store and extract frames into excel files.

@author: Andréas

@date: Tuesday, August the 15th of 2017
'''

from path import MyPath2

import pandas as pd

import toolbox
import os

#-----------------------------------
# READ EXCEL
#-----------------------------------

def read_dict_df(io, name=''):
    ''' Opens each tab of an excel file and returns a dict of dataframes. '''
    xl = pd.ExcelFile(io)
    dict_df = {}
    for sheet_name in xl.sheet_names:
        df = xl.parse(sheet_name)
        df = df.set_index([df.columns[0]])
        dict_df[sheet_name] = df
    return dict_df

def read_df(io, name=''):
    ''' Opens an excel file and returns a dataframe. '''
    df = pd.read_excel(io)
    df = df.set_index([df.columns[0]])
    if name:
        df.index.name = name
    return df

def read_series(io, name=''):
    ''' Opens an excel file and returns a series. '''
    df = read_df(io, name)
    df = pd.read_excel(io)
    series = df[df.columns[0]]
    if name:
        series.name = name
    return series

def read_csv(io, name=''):
    ''' Opens an csv file and returns a dataframe. '''
    df = pd.read_csv(io)
    df = df.set_index([df.columns[0]])
    if name:
        df.index.name = name
    return df

#-----------------------------------
# STORE EXCEL
#-----------------------------------
def valid_sheetname(str):
    ''' Returns a valid string for excel sheet names. '''
    invalid_excel_args = ['\\', '/', '#', '*', '?', '!', '[', ']']
    validstr = ''.join(e for e in str if not e in invalid_excel_args)
    max_length = 31
    validstr = validstr[:max_length]
    return validstr

def store_df(dataframe, filename='', sheet_name='Sheet1'):
    ''' Stores a dataframe in an excel file. '''
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    sheet_name = valid_sheetname(sheet_name)
    dataframe.to_excel(writer, sheet_name=sheet_name)

def store_series(series, filename='', sheet_name='Sheet1'):
    ''' Stores a series in an excel file. '''
    dataframe = series.to_frame()
    store_df(dataframe, filename, sheet_name)

def store(df_or_series, filename='', sheet_name='Sheet1'):
    ''' Stores a dataframe or series in an excel file. '''
    if isinstance(df_or_series, pd.Series):
        store_series(df_or_series, filename, sheet_name)
    elif isinstance(df_or_series, pd.DataFrame):
        store_df(df_or_series, filename, sheet_name)

def store_dict(dict, filename):
    ''' Stores a dict of dataframes in an excel file. '''
    for key, df_or_series in dict.items():
        store(df_or_series, filename, key)

def store_dict_dict(dict_dict, filepath):
    ''' Stores a dict of dict of dataframes in an excel file. '''
    for key, dict in dict_dict.items():
        filename = os.path.join(filepath, key+'.xlsx')
        store_dict(dict, filename)

def summary_store(df_or_series, filename='', sheet_name='Sheet1',
                dropList=['count'], percentiles=[.25,.5,.75]):
    ''' Stores a summary of the dataframe in an excel file. '''
    summary = toolbox.describe(df_or_series, dropList, percentiles)
    store(summary, filename, sheet_name)

def summary_store_dict(dict, filename,
                    dropList=['count'], percentiles=[.25,.5,.75]):
    ''' Stores a summary of each dataframe of the dict of dataframe in an excel file. '''
    summary_dict = {key: toolbox.describe(df_or_series, dropList, percentiles)
                       for key, df_or_series in dict.items()}
    store_dict(summary_dict, filename)

def summary_store_dict_dict(dict_dict, filepath,
                    dropList=['count'], percentiles=[.25,.5,.75]):
    ''' Stores a summary of each dataframe of the dict of dict of dataframe in an excel file. '''
    for key, dict in dict_dict.items():
        filename = os.path.join(filepath, key+'.xlsx')
        summary_store_dict(dict, filename, dropList, percentiles)

def mode_store_dict(dict, directory, mode, category, name):
    ''' 
    Store a dict of dataframes in an excel file in a mode/category subdirectory.
    
    Parameters:
    -----------
    directory: path
    mode: string, {'full', 'summary'}
    '''
    filename = os.path.join(directory, mode, category, name+'.xlsx')
    if mode == 'full':
        store_dict(dict, filename)
    elif mode == 'summary':
        summary_store_dict(dict, filename)


def mode_store_dict_dict(dict_dict, directory, mode, category):
    ''' 
    Store a dict of dict of dataframes in an excel file in a mode/category subdirectory.
    
    Parameters:
    -----------
    directory: path
    mode: string, {'full', 'summary'}
    '''
    filepath = os.join.path(directory, mode, category)
    if mode == 'full':
        store_dict_dict(dict, filepath)
    elif mode == 'summary':
        summary_store_dict_dict(dict, filepath)

def virgin_store_dict(dict, directory, mode, category, name):
    ''' 
    Store a full dict of dataframes in an excel file in a mode/category subdirectory.
    
    Parameters:
    -----------
    directory: path
    mode: string, {'full', 'summary'}
    '''
    filename = os.path.join(directory, mode, category, name+'.xlsx')
    store_dict(dict, filename)

def virgin_store_dict_dict(dict_dict, directory, mode, category):
    ''' 
    Store a full dict of dict of dataframes in an excel file in a mode/category subdirectory.
    
    Parameters:
    -----------
    directory: path
    mode: string, {'full', 'summary'}
    '''
    filepath = os.join.path(directory, mode, category)
    store_dict_dict(dict_dict, filepath)

def multi_store_dict(dict, directory, modes, category, name, virgin=False):
    ''' 
    Store a dict of dataframes in an excel file in a mode/category subdirectory.
    
    Parameters:
    -----------
    directory: path
    modes: list, ['full', 'summary']
    '''
    for mode in modes:
        mode_store_dict(dict, directory, mode, category, name)
        if virgin:
            virgin_store_dict(dict, directory, mode, category, name)

def multi_store_dict_dict(dict_dict, directory, modes, category, virgin=False):
    ''' 
    Store a dict of dataframes in an excel file in a mode/category subdirectory.
    
    Parameters:
    -----------
    directory: path
    modes: list, ['full', 'summary']
    '''
    for mode in modes:
        mode_store_dict_dict(dict, directory, mode, category)
        if virgin:
            virgin_store_dict_dict(dict, directory, mode, category)



class excel(object):
    """
    Manages all the excel storage.
    """
    def __init__(self):
        self.path = MyPath2()

    











