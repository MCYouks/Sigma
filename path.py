# *- coding: utf-8 -*-
"""
Created on August, the 4th of 2017

This .py file serves to organize all the path access.

@author: Andréas
"""

#import DBSnippets as db
import os


class MyPath2:
    """Manages all the directories manipulations."""
    def __init__(self, funds_name='', localtime=''):
        '''
        LOCAL PATHS
        ---------------------------
        '''
        self._current_path = os.path.dirname(os.path.abspath(__file__))

        # DATA BASE
        self._local_database_path = os.path.join(self._current_path, 'DataBase_Local')
        self._price_path = os.path.join(self._local_database_path, 'Data', 'Precios')

        # DATA STORE
        self._data_store_path = os.path.join(self._current_path, 'Data_Store')
        self._hdfstore_path = os.path.join(self._data_store_path, '_hdfstore')
        self.report_path = os.path.join(self._data_store_path, 'Reports')

        # FUNDS DIRECTORY > LOCAL TIME DIRECTORY
        self.funds_path = os.path.join(self.report_path, funds_name)
        self.localtime_dir_path = os.path.join(self.funds_path, localtime)
        self._excel_path = os.path.join(self.localtime_dir_path, 'Excel Assessment')
        self._figure_path = os.path.join(self.localtime_dir_path, 'Fig_Report')

        # RESSOURCES
        self._ressources_path = os.path.join(self._data_store_path, '_ressources')
        self._css_path = os.path.join(self._ressources_path, 'css')
        self._scripts_path = os.path.join(self._ressources_path, 'scripts')
        self.images_path = os.path.join(self._ressources_path, 'images')

        '''
        DROPBOX PATHS
        ---------------------------
        self._dropbox_home_path = self._get_dropbox_home_path() #db.dropbox_home()

        # DATA BASE
        self._dropbox_database_path = os.path.join(self._dropbox_home_path, 'DataBase')
        self._dropbox_reports_path = os.path.join(self._dropbox_database_path, 'Reports')
        self._dropbox_valorisation_path = os.path.join(self._dropbox_reports_path, 'Valoracion')
        '''

    #-----------------
    # TOOLBOX
    #------------------
    def _get_dropbox_home_path(self):
        try:
            _dropbox_home_path = db.dropbox_home()
        except:
            print('''\nImpossible to find dropbox home path. The path considered 
                  will be the local one: {path}'''.format(path=self._local_database_path))
            _dropbox_home_path = self._local_database_path
        return _dropbox_home_path

    def _set_figure_category_path(self, category=''):
        _category_path = os.path.join(self._figure_path, category)
        self._check_if_path_exists(_category_path)
        return _category_path

    def _set_excel_category_path(self, mode='', category=''):
        ''' Returns the path of a category for a certain mode.
                Mode: Full or Summary
                Category: Demand, Generation, Price or Finance'''
        _category_path = os.path.join(self._excel_path, mode, category)
        self._check_if_path_exists(_category_path)
        return _category_path

    def _set_figure_category_path(self, category=''):
        ''' Returns the path of a category for a certain mode.
                Mode: Full or Summary
                Category: Demand, Generation, Price or Finance'''
        _category_path = os.path.join(self._figure_path, category)
        self._check_if_path_exists(_category_path)
        return _category_path

    def _check_if_path_exists(self, _path):
        '''Creates the path if this one does not exist already.'''
        if not os.path.exists(_path):
            os.makedirs(_path)
            print('A new directory has been created: {}'.format(_path))