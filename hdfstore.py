# -*- coding: utf-8 -*-
'''

The objective is to build a hdf storage methods to optimize the memory.

@author: Andréas

@date: Monday, August the 14th
'''

from path import MyPath2

import pandas as pd
import os


class sexyPandas(object):
    """ 
    Manages all the HDF files storage.

    Parameters:
    -----------
    singleSeries: (explained below, in the superPandas class)

    resetMode: if False, the data stored won't be remove. It allows the explore the data of the simulations outside from the 
                risk valorisation module.

    """
    def __init__(self, name, singleSeries=False, resetMode=True):
        self.name = name
        self.path = MyPath2()
        self.filename = os.path.join(self.path._hdfstore_path, self.name+'.hdf5')
        self.singleSeries = singleSeries
        self.resetMode = resetMode
        # Initialization
        self.init()

    def init(self):
        if self.resetMode:
            self.check_if_dir_exists()
            self.remove_all_keys()

    def put(self, key, df):
        key = self.set_validstr(key)
        with pd.HDFStore(self.filename) as store:
            store.put(key, df)

    def select(self, key):
        key = self.set_validstr(key)
        with pd.HDFStore(self.filename) as store:
            return store.select(key)

    def keys(self):
        ''' Returns the list of the keys in store. '''
        with pd.HDFStore(self.filename) as store:
            return store.keys()
            
    def check_if_dir_exists(self):
        '''Creates the directory if this one does not exist already.'''
        if not os.path.exists(self.path._hdfstore_path):
            os.makedirs(self.path._hdfstore_path)

    def remove_all_keys(self):
        with pd.HDFStore(self.filename) as store:
            for key in store.keys():
                store.remove(key)

    def set_validstr(self, key):
        dict_accent = {u'á':'a', u'é':'e', u'í':'i', u'ó':'o', u'ú':'u', u'ñ':'n'}
        validstr = ''
        for l in key:
            if l in dict_accent:
                l = dict_accent[l]
            validstr += l
        return validstr

class simulationSeriesPandas(sexyPandas):
    """ 
    Manages the loading of the current simulation series. 

    """
    def __init__(self, *args, **kwargs):
        super(simulationSeriesPandas, self).__init__(*args, **kwargs)

    def refresh(self, n_series, simulations):
        '''
        Stores the series number and the simulation numbers.
        
        Parameters:
        -----------
        n_series: int
        simulations: list of int, for example [0, 1,..., 999]
        '''
        key = 'series{number}'.format(number=n_series)
        df =  pd.DataFrame(columns=simulations)
        self.put(key, df)

    def get_last_series_number(self):
        ''' Returns the last series number. '''
        if self.keys():
            return max([key.split('/')[1] for key in self.keys()])
        else:
            return 'series0'

class superPandas(sexyPandas):
    """ 
    Manages the storage with subgroups to optimize memory. 

    Parameters:
    -----------
    singleSeries: boolean, default False

            False --> the data in the dataframe are supposed to be different for each simulation series.
                    (example: It perfectly fits for SPOT_price, Random_Walk and Gross_Profit because they are different for each simulation)

            True --> the data in the dataframe are independant from the simulation series, they are
                    always the same over each series.
                    (example: It perfectly fits for the modellings (price, demand, generation) and for the summary table such
                    as Insolvency_rate because they are independant from the simulations, or are the result of an analysis
                    of the simulations.)

    """
    def __init__(self, *args, **kwargs):
        super(superPandas, self).__init__(*args, **kwargs)
        self.simulationSeriesPandas = simulationSeriesPandas('simulation_series')

    def get_current_simulation_series(self):
        if not self.singleSeries:
            return self.simulationSeriesPandas.get_last_series_number()
        else:
            return 'series0'

    def get(self, key):
        series = self.get_current_simulation_series()
        subkey = self.subkey(key, series)
        return self.select(subkey)

    def set(self, key, df):
        series = self.get_current_simulation_series()
        subkey = self.subkey(key, series)
        self.put(subkey, df)

    def items(self):
        keys = self.groups()
        return [(key, self.get(key)) for key in keys]

    def to_dict(self):
        keys = self.groups()
        return {key: self.getAll(key) for key in keys}

    def values(self):
        keys = self.groups()
        return [self.get(key) for key in keys]

    def getAll(self, key, axis=1):
        ''' Returns  dataframe that gathers all the simulations series data in one single frame. '''
        subkeys = self.group(key)
        return self.concat(subkeys, axis=axis)

    def concat(self, subkeys, axis=1):
        frames = [self.select(subkey) for subkey in subkeys]
        df = pd.concat(frames, axis=axis).sort_index(axis=axis)
        return df

    def groups(self):
        ''' Returns the main keys. For example, '/key1/series2' will be transformed in 'key1'. '''
        return list(set([self.get_key(subkey) for subkey in self.keys()]))

    def group(self, key):
        ''' Returns a list of subkeys corresponding with the key. '''
        return [subkey for subkey in self.keys() if self.get_key(subkey) == key]

    def subkeys(self):
        ''' Returns a list of subkeys corresponding with the key. '''
        series = self.get_current_simulation_series()
        return [subkey for subkey in self.keys() if self.get_series(subkey) == series]

    def subkey(self, key, series):
        ''' Returns the subkey corresponding with the HDF subkey format. '''
        return u'/{key}/{series}'.format(key=key, series=series)

    def get_key(self, subkey):
        return subkey.split('/')[1]
        
    def get_series(self, subkey):
        return subkey.split('/')[2]

class superMultiPandas(superPandas):
    """
    Stores a dict of dataframes (or series).
    """
    def __init__(self, *args, **kwargs):
        ''' The class inherits the methods from superPandas. '''
        super(superMultiPandas, self).__init__(*args, **kwargs)

    def __getitem__(self, key):
        ''' Returns the value of the dataframe corresponding to the key for the current simulation series. '''
        return self.get(key)

    def __setitem__(self, key, df):
        ''' Sets the value of the dataframe corresponding to the key for the current simulation series. '''
        self.set(key, df)

class superSinglePandas(superPandas):
    """
    Stores only one single dataframe (or series).
    """
    def __init__(self, *args, **kwargs):
        ''' The class inherits the methods from superPandas. '''
        super(superSinglePandas, self).__init__(*args, **kwargs)
        self.key = self.name

    @property
    def value(self):
        ''' Returns the value of the dataframe for the current simulation series. '''
        return self.get(self.key)

    @value.setter
    def value(self, df):
        ''' Sets the value of the dataframe for the current simulation series. '''
        self.set(self.key, df)

    def Get(self):
        ''' Returns  dataframe that gathers all the simulations series data in one single frame. '''
        return self.getAll(self.key)

    def to_frame(self, axis=1):
        ''' Returns  dataframe that gathers all the simulations series data in one single frame. '''
        return self.getAll(self.key, axis=axis)


if __name__ == '__main__':

    Gross_Profit = superSinglePandas('Gross_Profit', resetMode=False)
    Energy_Generation_P90 = superMultiPandas('Energy_Generation_P90', resetMode=False)
    Generation_depreciation = superMultiPandas('Generation_depreciation', resetMode=False)

    dict_df = Energy_Generation_P90.to_dict()
    dict_df[dict_df.keys()[0]].loc[dict_df[dict_df.keys()[0]].index[0]].hist()
    print dict_df.keys()[0]