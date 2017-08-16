# -*- coding: utf-8 -*-
"""
Created on Thu April 20, 2017

Useful Classes for Arkenstone

@author: Andréas
"""

from collections import defaultdict, OrderedDict
import multiprocessing as mp
import pandas as pd
import numpy as np
import time
import copy
import os

from zbaaar import MyHTML, MyColor, MyTable, MyColumn, MyRow, MyPlot, MyReport, MyToolBox
from path import MyPath2

from hdfstore import simulationSeriesPandas, superMultiPandas, superSinglePandas

from debug import debug, decorate_classmethods

__dict_months__ = {
    'ene':1, 'enero':1, 'ENE':1, 'ENERO':1, 'Ene':1,
    'feb':2, 'febrero':2, 'FEB':2, 'FEBRERO':2, 'Feb':2,
    'mar':3, 'marzo':3, 'MAR':3, 'MARZO':3, 'Mar':3,
    'abr':4, 'abril':4, 'ABR':4, 'ABRIL':4, 'Abr':4,
    'may':5, 'mayo':5, 'MAY':5, 'MAYO':5, 'May':5,
    'jun':6, 'junio':6, 'JUN':6, 'JUNIO':6, 'Jun':6,
    'jul':7, 'julio':7, 'JUL':7, 'JULIO':7, 'Jul':7,
    'ago':8, 'agosto':8, 'AGO':8, 'AGOSTO':8, 'Ago':8,
    'sep':9, 'septiembre':9, 'SEP':9, 'SEPTIEMBRE':9, 'Sep':9,
    'oct':10, 'octubre':10, 'OCT':10, 'OCTUBRE':10, 'Oct':10,
    'nov':11, 'noviembre':11, 'NOV':11, 'NOVIEMBRE':11, 'Nov':11,
    'dic':12, 'diciembre':12, 'DIC':12, 'DICIEMBRE':12, 'Dic':12
}

__Annual_Start__ = {
    1:'AS-JAN', 2:'AS-FEB', 3:'AS-MAR', 4:'AS-APR', 
    5:'AS-MAY', 6:'AS-JUN', 7:'AS-JUL', 8:'AS-AUG',
    9:'AS-SEP', 10:'AS-OCT', 11:'AS-NOV', 12:'AS-DEC'
}

class MyTime(object):
    """
    This class contains all the time informations that we be used during the computaion
    
    """
    def __init__(self):
        self.datetime = time.strftime('%Y%m%d',time.localtime())
        self.localtime = time.strftime('%Y-%m-%d - %Hh%M',time.localtime())
        self.timing = dict()

    def __start__(self, label='** GLOBAL **'):
        self.timing[label] = time.time()

    def __timer__(self, label='** GLOBAL **'):
        if label in self.timing:
            t0 = self.timing[label]
            delta = time.time() - t0
            timer = self.divmod(delta)
            print('\nTime to {}: {}.\n'.format(label, timer))
        else:
            print('\nTimer for label \'{}\' has never been initialized.'.format(label))

    def divmod(self, delta):
        m, s = divmod(delta, 60)
        h, m = divmod(m, 60)
        hour = 'hours' if h > 1 else 'hour'
        minute = 'minutes' if m > 1 else 'minute'
        second = 'seconds'
        if h:
            timer = '{:.0f} {}, {:.0f} {} and {:.1f} {}'.format(h, hour, m, minute, s, second)
        elif m:
            timer = '{:.0f} {} and {:.1f} {}'.format(m, minute, s, second)
        else:
            timer = '{:.1f} {}'.format(s, second)
        return timer


class MyParameters(object):
    """
    """
    def __init__(self, Param={}):
        if Param:
            ''' Parametric values '''
            self.random_seed = Param['Random Seed']
            self.n_simulation = int(Param['n_simulation'])
            self.discount_rate = float(Param['Tipo Descuento'])
            ''' Options '''
            self.Calc_Precios_Reset = Param['Calc_Precios_Reset']
            self.Price_Calculation_Fast_Mode = Param['Calc_Precios_Fast']
            self.InterfaceFile = Param['InterfaceFile']
            self.OilAdjust = Param['OilAdjust']
            self.Hydro = Param['Hydro']
            self.Valuation_date = Param['FechaValo']
            self.Override_Uniform_Generation = Param['ForceUniform']
            ''' Multiprocessing sequencing '''
            self.n_cpu = mp.cpu_count()
            self.n_calcultation_per_cpu = 100
            self.simulation_series = self.get_simulation_series()

    def get_simulation_series(self):
        #n_simulation_per_series = self.n_cpu * self.n_calcultation_per_cpu
        n_simulation_per_series = 10
        simulation_series = defaultdict(list)
        count = 0
        n_series = 1
        for simulation in range(self.n_simulation):
            if not count < n_simulation_per_series:
                n_series += 1
                count = 0
            simulation_series[n_series].append(simulation)
            count += 1
        return dict(simulation_series)


class MyExcel:
    """Manages all the excel storage of information."""
    def __init__(self, funds_name='', localtime=''):
        self.path = MyPath2(funds_name, localtime)

    #-----------------------------------
    #STORE DICT OF DATAFRAME
    #-----------------------------------
    def store_dict_df(self, dict_df, category='', summary_mode=['full', 'summary'], virgin=False):
        for key, dataframe in dict_df.items():
            if isinstance(dataframe, pd.Series):
                dataframe = dataframe.to_frame()
            for mode in summary_mode:
                category_path = self.path._set_excel_category_path(mode, category)
                filename = self.set_filename(category_path, key)
                writer = pd.ExcelWriter(filename, engine='xlsxwriter')
                sheet_name = self._get_validstr(key)
                if mode == 'full' or mode == '':
                    dataframe.to_excel(writer, sheet_name=sheet_name)
                elif mode == 'summary':
                    if virgin:
                        dataframe.to_excel(writer, sheet_name=sheet_name)
                    else:
                        summary_df = get_summary_df(dataframe, percentiles=[.05,.1,.25,.5,.75,.9,.95], dropList=['count'])
                        summary_df.to_excel(writer, sheet_name=sheet_name)
            print('>>> ... Excel storage for {} ---> OK'.format(key))

    def store_dict_series(self, dict_series, category='', summary_mode=['full', 'summary']):
        '''Note: (Andréas) In my view, storing a dict of series is USELESS because we would better convert a dict of series into a dataframe, using the function dict_series_to_df (more convenient). Then we would insert this dataframe into a dict (exactly this way: {key: dataframe}). Thus, it would be equivalent to store a dict of dataframe (see function store_dict_df above).'''
        pass

    #-----------------------------------
    #STORE DICT OF DICT OF DATAFRAME
    #-----------------------------------
    def store_dict_dict_df(self, dict_dict_df, category='', summary_mode=['full', 'summary'], virgin=False):
        for key, dict_df in dict_dict_df.items():
            for mode in summary_mode:
                category_path = self.path._set_excel_category_path(mode, category)
                filename = self.set_filename(category_path, key)
                writer = pd.ExcelWriter(filename, engine='xlsxwriter')
                for key_key in dict_df: #We do not use dict.items() because it can create trouble with HDF store
                    sheet_name = self._get_validstr(key_key)
                    dataframe = dict_df[key_key]
                    if isinstance(dataframe, pd.Series):
                        dataframe = dataframe.to_frame()
                    if mode == 'full':
                        dataframe.to_excel(writer, sheet_name=sheet_name)
                    elif mode == 'summary':
                        if virgin:
                            dataframe.to_excel(writer, sheet_name=sheet_name)
                        else:
                            summary_df = get_summary_df(dataframe, percentiles=[.25,.5,.75], dropList=['count'])
                            summary_df.to_excel(writer, sheet_name=sheet_name)
            print('>>> ... Excel storage for {} ---> OK'.format(key))

    def store_dict_dict_series(self, dict_dict_series, category='', summary_mode=['full', 'summary']):
        '''Note: (Andréas) In my view, storing a dict of dict of series is USELESS because we would better convert a dict of series into a dataframe, using the function dict_series_to_df (more convenient). Thus, it would be equivalent to store a dict of dataframe (see function store_dict_df above).'''
        pass

    #-------------------------------
    #TOOLBOX
    #-------------------------------
    def set_filename(self, filepath, name):
        return os.path.join(filepath, name + '.xlsx')

    def _get_validstr(self, str):
        '''Returns a valid string for excel sheet names.'''
        #Make sure all invalid arguments are excluded
        invalid_excel_args = ['\\', '/', '#', '*', '?', '!', '[', ']']
        validstr = ''.join(e for e in str if not e in invalid_excel_args)
        #Make sure string length stay smaller than the valid maximum length
        max_length = 31
        validstr = validstr[:max_length]
        #Return
        return validstr

debugOption = True
@decorate_classmethods(debug(debugOption))
class MVRA(object):
    """ 
    MODULE of VALUATION and RISK ASSESSMENT
    """
    def __init__(self, Price=None, Funds=None, Consumers=None, Generators=None, Param=None, gif=None):
        if Price and Funds and Consumers and Generators and Param:
            ''' Inputs '''
            self.Price = Price
            self.Funds = Funds
            self.Generators = Generators
            self.Consumers = Consumers
            self.parameters = MyParameters(Param)
            ''' Random Seed initialisation. '''
            self.init_random_seed()
            ''' We internalize the paths and parameters information '''
            self.simulation_series = simulationSeriesPandas('simulation_series')
            self.n_series = 0
            self.simulations = []
            ''' Extract useful informations '''
            self.Generators_node_name = self.get_Generators_node_name()
            self.Generators_nodes = self.get_Generators_nodes()
            self.Consumers_node_name = self.get_Consumers_node_name()
            self.Consumers_nodes = self.get_Consumers_nodes()
            self.all_nodes = self.get_all_nodes()
            self.start_date = self.get_start_date()
            self.end_date = self.get_end_date()
            ''' PPA Demand '''
            self.PPA_Demand_series = superMultiPandas('PPA_Demand_series', singleSeries=True)
            self.PPA_Demand = superMultiPandas('PPA_Demand')
            self.PFI_Demand = superMultiPandas('PFI_Demand') #Projection of Indexation Formula
            ''' Other useful informations '''
            self.SPOT_price = superMultiPandas('SPOT_price')
            self.SPOT_price_monthly_modelling = superMultiPandas('SPOT_price_monthly_modelling', singleSeries=True)
            self.SPOT_price_hourly_modelling = superMultiPandas('SPOT_price_hourly_modelling', singleSeries=True)
            self.SPOT_price_modelling = superMultiPandas('SPOT_price_modelling', singleSeries=True)
            self.Hydrological_scenarios = superSinglePandas('Hydrological_scenarios')
            ''' Brent Random Walk '''
            self.Random_Walk = superSinglePandas('Random_Walk')
            ''' Load the demands '''
            self.Energy_Demand = superMultiPandas('Energy_Demand')
            self.Energy_Demand_modelling = superMultiPandas('Energy_Demand_modelling', singleSeries=True)
            self.demand_distribution = superMultiPandas('demand_distribution')
            # (useless) to be deleted
            self.Total_Energy_Demand = superSinglePandas('Total_Energy_Demand')
            ''' Load the generations '''
            self.Energy_Generation = superMultiPandas('Energy_Generation')
            self.Generation_scenarios = superMultiPandas('Generation_scenarios')
            self.Energy_Generation_probabilities = superMultiPandas('Energy_Generation_probabilities', singleSeries=True)
            self.Energy_Generation_probabilities_interpolation = superMultiPandas('Energy_Generation_probabilities_interpolation', singleSeries=True)
            self.Energy_Generation_Threshold = superMultiPandas('Energy_Generation_Threshold', singleSeries=True)
            self.Generation_depreciation = superMultiPandas('Generation_depreciation', singleSeries=True)
            self.PPAbase = superMultiPandas('PPAbase')
            self.PPAalt = superMultiPandas('PPAalt')
            self.Monthly_Cumulative_Generation = superMultiPandas('Monthly_Cumulative_Generation')
            ''' Sales and Purchase (Generation side) '''
            self.Energy_Generation_P50 = superMultiPandas('Energy_Generation_P50')
            self.Energy_Generation_P90 = superMultiPandas('Energy_Generation_P90')
            self.Energy_Generation_P90_P50 = superMultiPandas('Energy_Generation_P90_P50')
            self.Purchased_Energy_Generation_Cost = superMultiPandas('Purchased_Energy_Generation_Cost')
            self.Purchased_Energy_Generation_P90_Cost = superMultiPandas('Purchased_Energy_Generation_P90_Cost')
            self.Purchased_Energy_Generation_P50_Cost = superMultiPandas('Purchased_Energy_Generation_P50_Cost')
            self.Sold_Energy_Generation_Incomes = superMultiPandas('Sold_Energy_Generation_Incomes')
            self.Sold_Energy_Generation_P90_Incomes = superMultiPandas('Sold_Energy_Generation_P90_Incomes')
            self.Sold_Energy_Generation_P50_Incomes = superMultiPandas('Sold_Energy_Generation_P50_Incomes')
            self.Sold_Energy_Generation_Profit = superMultiPandas('Sold_Energy_Generation_Profit')
            self.Sold_Energy_Generation_Price = superMultiPandas('Sold_Energy_Generation_Price')
            # (Useless) To be deleted
            self.Total_Energy_Generation_P90_P50 = superSinglePandas('Total_Energy_Generation_P90_P50')
            ''' Sales and Purchase (Demand side) '''
            self.Purchased_Energy_Demand_Cost = superMultiPandas('Purchased_Energy_Demand_Cost')
            self.Sold_Energy_Demand_Incomes = superMultiPandas('Sold_Energy_Demand_Incomes')
            self.Sold_Energy_Demand_Profit = superMultiPandas('Sold_Energy_Demand_Profit')
            self.Purchased_Energy_Demand_Price = superMultiPandas('Purchased_Energy_Demand_Price')
            ''' Global Profit '''
            self.Gross_Profit = superSinglePandas('Gross_Profit')
            self.Operating_Margin = superSinglePandas('Operating_Margin')
            self.Extra_Operating_Margin = superSinglePandas('Extra_Operating_Margin')
            self.Operating_Margin_Taxes = superSinglePandas('Operating_Margin_Taxes')
            self.Net_Incomes = superSinglePandas('Net_Incomes')
            self.Net_Liquid_Assets = superSinglePandas('Net_Liquid_Assets')
            self.Dividend = superSinglePandas('Dividend')
            self.Benchmark = superSinglePandas('Benchmark')
            self.Success_Fee = superSinglePandas('Success_Fee')
            self.Shareholder_Cash_Flow = superSinglePandas('Shareholder_Cash_Flow')
            self.IRR_Shareholder_Cash_Flow = superSinglePandas('IRR_Shareholder_Cash_Flow')
            self.NPV_Shareholder_Cash_Flow = superSinglePandas('NPV_Shareholder_Cash_Flow')
            ''' Insolvency Analysis '''
            self.Insolvency = superSinglePandas('Insolvency', singleSeries=True)
            self.Insolvency_rate = superSinglePandas('Insolvency_rate', singleSeries=True)
            self.Insolvency_Coverage_Cost = superSinglePandas('Insolvency_Coverage_Cost', singleSeries=True)
            self.Theoretical_Capital = superSinglePandas('Theoretical_Capital', singleSeries=True)
            self.Theoretical_Shareholder_Cash_Flow = superMultiPandas('Theoretical_Shareholder_Cash_Flow', singleSeries=True)
            self.Theoretical_IRR = superSinglePandas('Theoretical_IRR', singleSeries=True)
            self.Theoretical_NPV = superSinglePandas('Theoretical_NPV', singleSeries=True)
            ''' Tables of Parameters '''
            self.Computation_Parameters = pd.Series()
            self.Portofolio_Features = pd.Series()
            self.Generation_Assets = pd.DataFrame()
            self.Withdrawal_PPAs = pd.DataFrame()
            self.Portofolio_Summary = pd.Series()
            self.Success_Fee_Parameters = pd.Series()
            self.Energy_Source = pd.Series()
            self.Shareholder_Cash_Flow_Statistics = pd.DataFrame()
            self.Generation_and_Demand = pd.DataFrame()
            self.Generation_series = pd.Series()
            self.Demand_series = pd.Series()
            self.Total_Energy = pd.DataFrame()
            ''' To construct the report '''
            self.time = MyTime()
            self.excel = MyExcel(self.Funds.name, self.time.localtime)
            self.report = MyReport(self.Funds.name, self.time.localtime)
        ''' (Andréas): I don't really understand what gif is '''
        self.gif = gif

    '''
    TABLE OF FUNCTIONS (BEGINNING)
    -------------------------------------------------------------------------
    -------------------------------------------------------------------------
    '''

    #----------------
    #INITIALIZATION
    #----------------
    def init_random_seed(self):
        ''' Initializes the random seed just once at the start.'''
        random_seed = self.parameters.random_seed
        np.random.seed(random_seed)

    #------------------
    #SIMULATION SERIES
    #------------------
    def refresh_simulation_series(self, n_series, simulations):
        ''' Updates the simulation series number and stocks inside a hdf file. '''
        self.simulation_series.refresh(n_series, simulations)
        self.n_series = n_series
        self.simulations = simulations
        self.print_n_cycle(n_series)

    def print_n_cycle(self, n_series):
        max_series = max(self.parameters.simulation_series.keys())
        print('''
        ****************************************************************
                      Cycle {n_series} out of {max_series}
        ****************************************************************
        '''.format(n_series=n_series, max_series=max_series))

    def print_out_cycle(self):
        max_series = max(self.parameters.simulation_series.keys())
        print('''
        ****************************************************************
          Series of simulations terminated ! Total number of cycles: {max_series}
        ****************************************************************
        '''.format(max_series=max_series))

    #-----------------------
    #LOAD INPUT MODELLINGS
    #-----------------------
    def load_SPOT_price_modelling(self):
        self.check_nodes_price_projection()
        self.set_SPOT_price_monthly_modelling()
        self.set_SPOT_price_hourly_modelling()
        self.set_SPOT_price_modelling()
        self.check_SPOT_price_modelling_date_range()

    def load_Energy_Demand_modelling(self):
        self.set_Energy_Demand_modelling()
        
    def load_Energy_Generation_modelling(self):
        self.set_Energy_Generation_probabilities()
        self.set_Energy_Generation_probabilities_interpolation()

    def load_PPA(self):
        self.set_PPA_Demand_series()

    #----------------------
    #SCENARIOS GENERATION
    #----------------------
    def generate_SPOT_price_scenarios(self):
        self.set_Hydrological_scenarios()
        self.set_SPOT_price3()

    def generate_Energy_Demand_scenarios(self):
        self.check_trapezoidal_demand_distribution_parameters()
        self.set_trapezoidal_distribution()
        self.set_Energy_Demand()

    def generate_Energy_Generation_scenarios(self):
        self.set_Generation_scenarios()
        self.set_Generation_depreciation()
        self.set_Energy_Generation()

    #----------------------
    #GENERATE RANDOM WALK
    #----------------------
    def set_Random_Walk_modeling(self):
        self.check_availability_Random_Walk()
        self.set_Random_Walk()

    #--------------------------
    #VARIABILITY INTRODUCTION
    #--------------------------
    def introduce_SPOT_Price_Variability(self):
        self.check_Random_Walk_SPOT_Price_parameters()
        self.apply_SPOT_Price_Variability()

    #-------------------------------------------------------------------
    #ADJUST DATA IF VARIABLE COST OR BRENT INDEXATION OPTION ACTIVATED
    #-------------------------------------------------------------------
    def introduce_PPA_Demand_Variability(self):
        self.set_PFI_Demand()
        self.set_PPA_Demand()
    
    def introduce_PPA_Generation_Variability(self):
        self.check_PPA_Generation_parameters()
        self.apply_PPA_Generation_Variability()

    def introduce_Energy_Generation_Variability(self):
        self.adjust_Energy_Generation()

    #------------------
    #CALCULATE PROFIT
    #------------------
    #results in monthly frequency
    def create_Energy_Demand_Profit(self):
        self.set_Purchased_Energy_Demand_Cost()
        self.resample_Energy_Demand_to_monthly_frequency()
        self.set_Sold_Energy_Demand_Incomes()
        self.set_Sold_Energy_Demand_Profit()
        self.set_Purchased_Energy_Demand_Price()
        #-- Total energy consumed
        self.set_Total_Energy_Demand()

    #results in monthly frequency
    def create_Energy_Generation_Profit(self):
        self.set_Energy_Generation_Threshold()
        self.set_Monthly_Cumulative_Generation()
        self.set_Energy_Generation_P90()
        self.set_Energy_Generation_P50()
        self.set_Sold_Energy_Generation_P90_Incomes()
        self.set_Sold_Energy_Generation_P50_Incomes()
        self.set_Sold_Energy_Generation_Incomes()
        self.resample_Energy_Generation_to_monthly_frequency()
        self.resample_Energy_Generation_P90_to_monthly_frequency()
        self.resample_Energy_Generation_P50_to_monthly_frequency()
        self.set_Energy_Generation_P90_P50()
        self.set_Purchased_Energy_Generation_P90_Cost()
        self.set_Purchased_Energy_Generation_P50_Cost()
        self.set_Purchased_Energy_Generation_Cost()
        self.set_Sold_Energy_Generation_Profit()
        self.set_Sold_Energy_Generation_Price()
        #-- Total energy generated
        self.set_Total_Energy_Generation_P90_P50()

    #results in monthly frequency
    def resample_data_before_Assessment(self):
        self.resample_SPOT_price_to_monthly_frequency()
        self.resample_SPOT_price_modelling_to_monthly_frequency()
        self.resample_Energy_Generation_probabilities_to_monthly_frequency()
        #self.resample_Energy_Generation_probabilities_interpolation_to_monthly_frequency()

    #-----------------------------
    #CREATE FINANCIAL ASSESSMENT
    #-----------------------------
    def create_Financial_Assessment(self):
        self.set_Gross_Profit()
        self.set_Operating_Margin()
        self.set_Extra_Operating_Margin()
        self.set_Operating_Margin_Taxes()
        self.set_Net_Incomes()
        self.set_Net_Liquid_Assets_and_Dividend()
        self.set_Benchmark()
        self.set_Success_Fee()
        self.set_Shareholder_Cash_Flow()
        self.set_IRR_Shareholder_Cash_Flow()
        self.set_NPV_Shareholder_Cash_Flow()

    def create_Insolvency_Analysis(self):
        self.set_Insolvency()
        self.set_Insolvency_rate()
        self.set_Insolvency_Coverage_Cost()
        self.set_Theoretical_Capital()
        self.set_Theoretical_Shareholder_Cash_Flow()
        self.set_Theoretical_IRR()
        self.set_Theoretical_NPV()

    #----------------------------------
    #CREATE TABLES FOR THE REPORTING
    #----------------------------------
    def set_Parameter_Tables(self):
        self.set_Computation_Parameters()
        self.set_Portofolio_Features()
        self.set_Portofolio_Summary()
        self.set_Success_Fee_Parameters()

    def set_Summary_Tables(self):
        self.set_Energy_Source()
        self.set_Withdrawal_PPAs()
        self.set_Generation_Assets()
        self.set_Shareholder_Cash_Flow_Statistics()
        self.set_Generation_and_Demand()
        self.set_Generation_series()
        self.set_Demand_series()

    #-----------------------------
    #STORE VALUES IN EXCEL FILES
    #-----------------------------
    def set_Excel_Assessment(self): 
        self.set_SPOT_price_Excel_Assessment()
        self.set_Hydrological_scenarios_Excel_Assessment()
        self.set_Energy_Demand_Excel_Assessment()
        self.set_Energy_Generation_Excel_Assessment()
        self.set_Energy_Generation_Probabilities_Excel_Assessment()
        self.set_Financial_Excel_Assessment()
        self.set_IRR_NPV_Excel_Assessment()
        self.set_Parameter_Tables_Excel_Assessment()
        self.set_Summary_Tables_Excel_Assessment()

    #-----------------------------
    #CREATE HTML REPORT
    #-----------------------------
    def create_HTML_Report(self):
        self.create_Summary_html_section()
        self.create_Characteristics_and_Results_html_section()
        self.create_Net_Liquid_Assets_and_Dividend_html_section()
        self.create_Profit_Analysis_html_section()
        self.create_Shareholder_Cashflow_html_section()
        self.create_Energy_Generation_P90_P50_html_section()
        self.create_Energy_Demand_html_section()
        self.create_SPOT_price_html_section()
        self.compile_HTML_Report()

    '''
    TABLE OF FUNCTIONS (END)
    -------------------------------------------------------------------------
    -------------------------------------------------------------------------
    '''

    #---------------
    #Error handler
    #---------------
    def __exit__(self, gif):
        if not self.gif:
            pass
        else: 
            gif.PlayGif(False)
        return False, False, False


    #-----------------
    #Initialization
    #-----------------
    def get_List_PG(self):
        List_PG = defaultdict(list)
        for obj in self.o_F:
            obj_Fon = db.LoadFromDB(-1, [obj], 'List_Fondo')
            obj_Gen_SE = db.LoadFromDB(-1,obj_Fon[0].List_PG_ID,'List_Proy_Gen',['subestacion'])
            for obj_Gen in obj_Gen_SE:
                List_PG[obj_Gen.subestacion].append(obj_Gen.ID)
        return dict(List_PG)
        
    def Get_List_CC(self):
        List_CC = defaultdict(list)
        for obj in self.o_F:
            obj_Fon = db.LoadFromDB(-1, [obj], 'List_Fondo')
            obj_Dem_SE = db.LoadFromDB(-1,obj_Fon[0].List_CC_ID,'List_Con_Consu',['subestacion'])
            for obj_Dem in obj_Dem_SE:
                List_CC[obj_Dem.subestacion].append(obj_Dem.ID)
        return dict(List_CC)

    def get_obj_Gen(self):
        obj_Gen = {substation: db.LoadFromDB(-1, ID, 'List_Proy_Gen') 
                    for substation, ID in self.List_PG.items()}
        return obj_Gen

    def get_obj_Dem(self):
        obj_Dem = {substation: db.LoadFromDB(-1,ID,'List_Con_Consu') 
                   for substation, ID in self.List_CC.items()}
        return obj_Dem

    def get_Funds(self):
        ''' Funds inherits from the class Fondo (see DBSnippets.py) '''
        Funds = copy.deepcopy(self.obj_Fon[0])
        del self.obj_Fon #Optimizing Memory
        return Funds

    def get_Price(self):
        ''' Price inherits from the class Proy_Precios (see DBSnippets.py) '''
        Price = copy.deepcopy(self.obj_PP[0])
        del self.obj_PP #Optimizing Memory
        return Price

    def get_Consumers(self):
        ''' Each consumer inherits from the class Con_Consu (see DBSnippets.py) '''
        Consumers = list()
        for node, consumers in self.obj_Dem.items():
            for consumer in consumers:
                Consumers.append(copy.deepcopy(consumer))
        del self.obj_Dem #Optimizing Memory
        return Consumers

    def get_Generators(self):
        ''' Each generator inherits from the class Proy_Gen (see DBSnippets.py) '''
        Generators = list()
        for node, generators in self.obj_Gen.items():
            for generator in generators:
                Generators.append(copy.deepcopy(generator))
        del self.obj_Gen #Optimizing Memory
        return Generators

    def get_Consumers_node_name(self):
        Consumers_node_name = defaultdict(list)
        for consumer in self.Consumers:
            Consumers_node_name[consumer.subestacion].append(consumer.name)
        return Consumers_node_name

    def get_Generators_node_name(self):
        Generators_node_name = defaultdict(list)
        for generator in self.Generators:
            Generators_node_name[generator.subestacion].append(generator.name)
        return Generators_node_name

    def get_Consumers_nodes(self):
        Consumers_nodes = self.Consumers_node_name.keys()
        return Consumers_nodes

    def get_Generators_nodes(self):
        Generators_nodes = self.Generators_node_name.keys()
        return Generators_nodes

    def get_all_nodes(self):
        all_nodes = list(set(self.Consumers_nodes + self.Generators_nodes))
        return all_nodes

    def get_start_date(self):
        start_date = self.Funds.start_date
        return start_date

    def get_end_date(self):
        end_date = self.Funds.end_date + pd.Timedelta(23,'h')
        return end_date

    #----------------
    #MAGIC TOOLBOX
    #----------------
    def get_date_range(self, freq='D', option=None):
        """ 
        Returns the date range we want to consider. More info at: 
            (1) <http://stackoverflow.com/questions/32168848/how-to-create-a-pandas-datetimeindex-with-year-as-frequency>
            (2) <http://stackoverflow.com/questions/22652774/error-in-appending-date-to-timeseries-data>
            (3) To use period range:<http://pandas.pydata.org/pandas-docs/stable/timeseries.html#periodindex-and-period-range> for more info
            (4) Convert period range and datetime index: <http://pandas.pydata.org/pandas-docs/stable/timeseries.html#period-dtypes>
            (5) Use the method to_period to convert to PeriodIndex

        
        """
        start = self.start_date
        end = self.end_date
        if freq == 'M':
            #Mensual indexing from the 1st of each month (otherwise it would be the last day of the month)
            date_range = pd.date_range(start=start, end=end, freq='MS') # 'MS' for 'month start'
            if option == 'include end_time':
                month_range = [date for date in date_range]
                end_time = pd.to_datetime(end)
                month_range.append(end_time)
                date_range = pd.DatetimeIndex(month_range)
                date_range = date_range.sort_values()
        elif freq == 'A':
            #Annual indexing including the first date
            january_range = pd.date_range(start=start, end=end, freq='AS-JAN')
            start_time = pd.to_datetime(start)
            annual_range = [date for date in january_range if not date == start_time]
            annual_range.append(start_time)
            date_range = pd.DatetimeIndex(annual_range)
            date_range = date_range.sort_values()
        elif freq == 'AS-APR':
            #Annual indexing from Arpil, 1st, including the first date
            april_range = pd.date_range(start=start, end=end, freq='AS-APR')
            start_time = pd.to_datetime(start)
            annual_range = [date for date in april_range if not date == start_time]
            annual_range.append(start_time)
            if option == 'include end_time':
                end_time = pd.to_datetime(end)
                annual_range.append(end_time)
            date_range = pd.DatetimeIndex(annual_range)
            date_range = date_range.sort_values()
        else:
            #Usually if freq == 'D' or freq == 'H'
            date_range = pd.date_range(start=start, end=end, freq=freq)
        return date_range

    def get_hydro_year_markers(self, freq='H'):
        ''' Output: (list) Groups the start date and end date for each hydrological year in the date range.
            >>> [(hydro_start_date, hydro_end_date) for each hydro year] '''
        date_range = self.get_date_range(freq='AS-APR', option='include end_time')
        #Initialization
        hydro_year_markers = list()
        #Iteration
        for i in range(len(date_range)-1):
            start = date_range[i]
            end = date_range[i+1]
            if not end == date_range.max():
                end = end - pd.Timedelta(1,'h')
            marker = (start, end)
            hydro_year_markers.append(marker)
        return hydro_year_markers

    def get_month_markers(self, freq='H'):
        ''' Output: (list) Groups the start date and end date for each month in the date range.
            >>> [(month_start, month_end) for month in date range]'''
        date_range = self.get_date_range(freq='M', option='include end_time')
        #Initialization
        month_markers = list()
        #Iteration
        for i in range(len(date_range)-1):
            start = date_range[i]
            end = date_range[i+1]
            if not end == date_range.max():
                end = end - pd.Timedelta(1,'h')
            marker = (start, end)
            month_markers.append(marker)
        return month_markers

    def get_month_range_series(self, freq='H'):
        ''' Output: (dict of list) Groups the hours in date range by month
            >>> {month: [hours in the month]} '''
        date_range = self.get_date_range(freq='M', option='include end_time')
        #Initialization
        month_range_series = dict()
        #Iteration
        for i in range(len(date_range)-1):
            start = date_range[i]
            end = date_range[i+1]
            if not end == date_range.max():
                end = end - pd.Timedelta(1,'h')
            month_range_series[start] = pd.date_range(start, end, freq=freq)
        return month_range_series
    
    def set_date_range_series(self, groupby='H/M'):
        ''' Output: (dict of lists) Contains all the dates of the hour range grouped by hour and by month
            >>> list_of_dates[index] where index=(hour, month) '''
        date_range = self.get_date_range(freq='H')
        date_range_series = defaultdict(list)
        if groupby == 'H/M':
            for date in date_range:
                index = (date.hour, date.month)
                date_range_series[index].append(date)
        if groupby == 'M/A':
            for date in date_range:
                index = (date.month, date.year)
                date_range_series[index].append(date)
        return dict(date_range_series)

    def set_date_range_dataframe(self, groupby='H/M'):
        ''' Output: (df) Contains the list of dates in hour range grouped by hour and by month.
        >>> list_of_dates(daily hours, annual months) '''
        date_range_series = self.set_date_range_series(groupby)
        if groupby == 'H/M':
            #We group by hour and month
            hours = range(24)
            months = range(1, 13)
            date_range_dataframe = pd.DataFrame(index=hours, columns=months)
        if groupby == 'M/A':
            #We group by month and year
            months = range(1, 13)
            year_range = [date.year for date in date_range]
            years = list(set(year_range))
            date_range_dataframe = pd.DataFrame(index=months, columns=years)
        for index in date_range_series:
            date_range_dataframe.at[index] = date_range_series[index]
        return date_range_dataframe

    def get_full_matrix(self, fill_value, date_range, columns):
        shape = (len(date_range), len(columns))
        data = np.full(shape=shape, fill_value=fill_value)
        df = pd.DataFrame(data, index=date_range, columns=columns)
        return df

    def get_full_series(self, fill_value, date_range):
        data = {date: fill_value for date in date_range}
        s = pd.Series(data)
        return s

    #---------------------------
    #LOAD SPOT PRICE MODELLING
    #---------------------------
    def check_nodes_price_projection(self):
        ''' Makes sure there exist a price projection at each node '''
        bool = False
        for node in self.all_nodes:
            if not node in self.Price.Dict_Nudos:
                bool = True
                print('Price projection for node {} has not been specified.'.format(node))
        if bool:
            self.__exit__()

    def set_SPOT_price_monthly_modelling(self):
        ''' Output: (dict of dataframes) Represents for each node, the monthly SPOT price for each month in the date range, for each hydro scenario.
            >>> {nodes: price(months, hydro scenarios) [USD/MWh]} '''
        for node in self.all_nodes:
            monthly_modelling = self.Price.get_monthly_price_modelling(node)
            self.SPOT_price_monthly_modelling[node] = monthly_modelling

    def set_SPOT_price_hourly_modelling(self, multiprocessing=False):
        ''' Output: (dict of series) For each node, the hourly modelling ratio for each hour of the date range 
            >>> {nodes: ratio(hours)}'''
        date_range_series = self.set_date_range_series(groupby='H/M')
        for node in self.all_nodes:
            hourly_modelling = self.Price.get_hourly_price_modelling(node)
            indices = [(hourly_modelling.at[index], date_range) for index, date_range in date_range_series.items()]
            if multiprocessing:
                n_cpu = mp.cpu_count()
                pool = mp.Pool(processes=n_cpu)
                frames = pool.map(set_SPOT_price_hourly_modelling_series_helper, indices)
                pool.close()
                pool.join()
            else:
                frames = [set_SPOT_price_hourly_modelling_series_helper(index) for index in indices]
            hourly_modelling_series = pd.concat(frames, axis=0)
            hourly_modelling_series = hourly_modelling_series.sort_index()
            self.SPOT_price_hourly_modelling[node] = hourly_modelling_series

    def set_SPOT_price_modelling(self):
        ''' Output: (dict of dataframes) SPOT price for each node, for each hydrological scenario, for each hour in the date range.
            >>> {node: price(hours, hydro scenarios)} '''
        date_range = self.get_date_range(freq='H')
        for node in self.all_nodes:
            #We load the modellings
            monthly_modelling = self.SPOT_price_monthly_modelling[node]
            hourly_modelling = self.SPOT_price_hourly_modelling[node]
            #We reindex monthly modelling to hourly frequency
            monthly_modelling = monthly_modelling.reindex(index=date_range, method='ffill')
            #We adjust the ñonthly modelling by applying an hourly ratio
            price_modelling = monthly_modelling.multiply(hourly_modelling, axis=0)
            self.SPOT_price_modelling[node] = price_modelling
            print(u'>>> ... Loading Price for {} ---> OK'.format(node))

    def check_SPOT_price_modelling_date_range(self):
        ''' Check for each node if the projection price fit well with the date range '''
        date_range = self.get_date_range(freq='M')
        start_funds = date_range.min()
        end_funds = date_range.max()
        exit = False
        ''' Funds' date range ends the last day of the month whereas the SPOT price are indexing on the first day of the month '''
        for node in self.all_nodes:
            SPOT_price_modelling = self.SPOT_price_modelling[node]
            date_range_SPOT = SPOT_price_modelling.index
            start_SPOT = date_range_SPOT.min()
            end_SPOT = date_range_SPOT.max()
            if start_SPOT > start_funds:
                print(u'SPOT price error: {} price projection starts after the funds starting date.'.format(node))
                print('start_SPOT: ({})'.format(start_SPOT))
                print('start_funds: ({})'.format(start_funds))
                exit = True
            if end_SPOT < end_funds:
                print(u'SPOT price error: {} price projection ends before the funds ending date.'.format(node))
                print('end_SPOT: ({})'.format(end_SPOT))
                print('end_funds: ({})'.format(end_funds))
                exit = True
        if exit:
            self.__exit__()

    #------------------------------
    #LOAD ENERGY DEMAND MODELLING
    #------------------------------
    def set_Energy_Demand_modelling(self):
        ''' 
        Creates a modelling series of the evolution of the daily demand (MWh),
        for each consumer, for each hour along the date range.

        Result:
        -------
        dict of series: {consumers: demand(hours) [MWh]}
        '''
        date_range = self.get_date_range(freq='H')
        for consumer in self.Consumers:
            energy_demand = consumer.get_energy_demand()
            #We normalize the dates to make sure they fit well
            energy_demand = energy_demand.reindex(index=date_range, fill_value=0)
            self.Energy_Demand_modelling[consumer.name] = energy_demand
            print(u'>>> ... Loading Demand for {} ---> OK'.format(consumer.name))

    #----------------------------------------
    #LOAD ENERGY GENERATION MODELLING
    #----------------------------------------
    def set_Energy_Generation_probabilities(self):
        ''' 
        Creates a generation dataframe for each probibility of energy generation (or hydrologic 
        scenario if the project is hydrologic), 
        for each generator, for each hour in the date range.
        
        Result: 
        -------
        dict of dataframes: {generators: generation(hours, probabilities)}
        '''
        def applyParallel(func, indices, multiprocessing=False):
            if multiprocessing:
                with mp.Pool(processes=mp.cpu_count()) as pool:
                    frames = pool.map(func, indices)
            else:
                frames = [func(index) for index in indices]
            return frames

        def set_energy_generation_probabilities_series(index):
            def set_series(date_range, probability, generation_probability):
                generation = pd.Series(index=date_range, name=probability)
                for date in date_range:
                    generation[date] = generation_probability.at[date.hour, date.month]
                return generation
            return set_series(*index)

        date_range = self.get_date_range(freq='H')
        for generator in self.Generators:
            generation_modelling = generator.get_energy_generation_modelling() 
            #Note: {probabilities_or_hydro_scenarios: {generation(hours, months)}}
            probabilities = generation_modelling.keys()
            indices = [(date_range, probability, generation_modelling[probability]) for probability in probabilities]
            frames = applyParallel(set_energy_generation_probabilities_series, indices)
            generation_probabilities = pd.concat(frames, axis=1)
            generation_probabilities = generation_probabilities.sort_index(axis=1)
            self.Energy_Generation_probabilities[generator.name] = generation_probabilities
            print(u'>>> ... Loading Generation for {} ---> OK'.format(generator.name))

    def set_Energy_Generation_probabilities_interpolation(self):
        ''' Output: (dict of dataframes) for each generator, a generation dataframe extrapolated linearly for each probability between 0 and 100, for each date in the date_range'''
        for generator in self.Generators:
            generation_probabilities = self.Energy_Generation_probabilities[generator.name]
            if generator.tecnologia == 'Hidro':
                #The interpolation is exactly the generation probabilities matrix because all the hydraulic scenarios are equiprobable
                interpolation = generation_probabilities
            else:
                #We get the columns and convert it from [P25,...,P95] to [25,...,95]
                probabilities = [int(probability.replace('P', '')) for probability in generation_probabilities.columns]
                interpolation = generation_probabilities
                step = 5
                #We interpolate linearly between the maximum and minimum probability index of generation probability
                columns = ['P{}'.format(x) for x in range(min(probabilities), max(probabilities)+1, step)]
                interpolation = interpolation.reindex(columns=columns).interpolate(axis=1)
                #We fill the first probability indexes using the backfill method: use next valid observation to fill gap
                columns = ['P{}'.format(x) for x in range(0, max(probabilities)+1, step)]
                interpolation = interpolation.reindex(columns=columns).fillna(method='bfill', axis=1)
                #We fill the last probability indexes using the ffill method: propagate last valid observation forward to next valid
                columns = ['P{}'.format(x) for x in range(0, 101, step)]
                interpolation = interpolation.reindex(columns=columns).fillna(method='ffill', axis=1)
            self.Energy_Generation_probabilities_interpolation[generator.name] = interpolation

    #-------------------------------
    #LOAD PURCHASE POWER AGREEMENTS
    #-------------------------------
    def set_PPA_Demand_series(self):
        ''' Output: (dict of series) Power Puchase Agreement for each consumer along the date range (usually, consumer PPA varies on annual frequency, but the results are presented in an hourly frequency)
            >>> {consumer: PPA(months)} '''
        date_range = self.get_date_range(freq='M')
        for consumer in self.Consumers:
            PPA_series = consumer.get_PPA_series()
            #We reindex the series to make sure it fits well with the date range
            PPA_series = PPA_series.reindex(index=date_range, fill_value=0) 
            self.PPA_Demand_series[consumer.name] = PPA_series

    #-------------------------------
    #GENERATE SPOT PRICE SCENARIOS
    #-------------------------------
    def set_Hydrological_scenarios(self):
        ''' Output: (dataframe) hydrological scenario for each hydro year from april (1st) to march (31st) for each simulation.
            >>> hydrological_scenario(hydro years, simulations) '''
        date_range = self.get_date_range(freq='AS-APR')
        columns = self.simulations
        n_simulation = len(columns)
        #n_simulation = self.parameters.n_simulation
        n_hydro_scenarios = int(self.Price.anoshidro)
        scenario_range = range(1, n_hydro_scenarios+1) #We want a list of the form [1,..,40]
        size = (len(date_range), n_simulation)
        data = np.random.choice(scenario_range, size=size)
        self.Hydrological_scenarios.value = pd.DataFrame(data, index=date_range, columns=columns)

    def set_SPOT_price3(self):
        ''' '''
        def applyParallel(func, indices, multiprocessing=False):
            if multiprocessing:
                with mp.Pool(mp.cpu_count()) as pool:
                    frames = pool.map(func, indices)
            else:
                frames = [func(index) for index in indices]
            return pd.concat(frames, axis=1).sort_index(axis=1)

        def set_spot_price_series_helper(index):
            def set_spot_price_series(simulation, hour_range, month_range_series, scenarios, spot_price_modelling):
                spot_price_series = pd.Series(name=simulation, index=hour_range, dtype=float)
                for start_month, date_range in month_range_series.items():
                    scenario = scenarios.at[start_month, simulation]
                    spot_price_series.loc[date_range] = spot_price_modelling.loc[date_range, scenario]
                return spot_price_series
            return set_spot_price_series(*index)

        hour_range = self.get_date_range('H')
        month_range = self.get_date_range('M')
        simulations = self.simulations
        scenarios = self.Hydrological_scenarios.value
        scenarios = scenarios.reindex(index=month_range, method='ffill')
        #simulation_series = self.parameters.simulation_series
        month_range_series = self.get_month_range_series()
        for node in self.all_nodes:
            spot_price_modelling = self.SPOT_price_modelling[node]
            #print(u'>>> ... Cycle {} out of {}: Node {}'.format(num_series, max_series, node))
            print(u'>>> ... SPOT price calculation for {node}'.format(node=node))
            indices = [(simulation, hour_range, month_range_series, 
                        scenarios, spot_price_modelling) for simulation in simulations]
            spot_price = applyParallel(set_spot_price_series_helper, indices)
            spot_price.index.name = node
            self.SPOT_price[node] = spot_price
            print(u'>>> ... SPOT price calculation for {node} ---> OK'.format(node=node))             

    #-----------------------------------
    #GENERATE ENERGY DEMAND SCENARIOS
    #-----------------------------------
    def check_trapezoidal_demand_distribution_parameters(self):
        ''' Check the availability of all the trapezoidal distribution parameters '''
        default_distribution = {'a': 0, 'b': 0, 'c': 0, 'd': 0}
        for consumer in self.Consumers:
            bool = False
            distribution = consumer.get_demand_distri_parameters()
            for str in list('abcd'):
                if not isinstance(distribution[str], float):
                    print(u'Missing distribution constant \'{}\' for consumer {}.'.format(str, consumer.name))
                    bool = True
            if bool:
                print(u'Distribution parameters set by default for consumer {}: {}'.format(consumer.name, default_distribution))
                distribution = default_distribution

    def set_trapezoidal_distribution(self):
        ''' Output: (dict) for each consumer, a trapezoidal density distribution that varies around 100% for each day for each simulation
            >>> {consumer: density_distribution(days, simulations)} '''
        simulations = self.simulations
        date_range = self.get_date_range(freq='M')
        for consumer in self.Consumers:
            distri_parameters = consumer.get_demand_distri_parameters()
            a, b, c, d = (distri_parameters[str] for str in list('abcd'))
            delta = trapezoidal_distribution(a, b, c, d, date_range, simulations)
            demand_distribution = 1 + delta
            self.demand_distribution[consumer.name] = demand_distribution

    def set_Energy_Demand(self, multiprocessing=False):
        ''' Output: (dict) for each consumer, a daily demand dataframe with a different daily distribution for each simulation
            >>> {consumers: demand(hours, simulations)} '''
        date_range = self.get_date_range(freq='H')
        count = 0
        for consumer in self.Consumers:
            #We load the inputs
            energy_demand_modelling = self.Energy_Demand_modelling[consumer.name] #type:series, freq='H'
            demand_distribution = self.demand_distribution[consumer.name] #type:dataframe, freq='D'
            #We reindex the demand distribution from a daily to an hourly frequency
            demand_distribution = demand_distribution.reindex(index=date_range, method='ffill')
            #We adjust the energy demand by applying a demand variation
            energy_demand = demand_distribution.multiply(energy_demand_modelling, axis=0)
            energy_demand.index.name = consumer.name
            self.Energy_Demand[consumer.name] = energy_demand
            print(u'>>> ... Generating Demand for {} ---> OK'.format(consumer.name))

    #---------------------------------------
    #GENERATE ENERGY GENERATION SCENARIOS
    #---------------------------------------
    def set_Generation_scenarios(self):
        ''' Output: (dict of dataframes) for each generator, a probability scenario dataframe for each month for each simulation
            >>> {generators: scenario(months, simulations)} '''
        simulations = self.simulations
        n_simulation = len(simulations)
        date_range = self.get_date_range(freq='M')
        for generator in self.Generators:
            if generator.tecnologia == 'Hidro':
                generation_scenarios = self.Hydrological_scenarios.value
                generation_scenarios = generation_scenarios.reindex(index=date_range, method='ffill') 
            else:
                generation_probabilities_interpolation = self.Energy_Generation_probabilities_interpolation[generator.name]
                probabilities = generation_probabilities_interpolation.columns
                size = (len(date_range), n_simulation)
                data = np.random.choice(probabilities, size=size)
                generation_scenarios = pd.DataFrame(data, index=date_range, columns=simulations)
            self.Generation_scenarios[generator.name] = generation_scenarios

    def set_Generation_depreciation(self):
        '''Output: (dict of series) for each generator, the depreciation rate calculated for each year in the date range.
            >>> {generator: depreciation_rate(years)} '''
        date_range = self.get_date_range('A')
        for generator in self.Generators:
            depreciation = float(generator.depreciacion) / 100
            enumerate_date_range = {i: date for i, date in enumerate(date_range)}
            dict_depreciation = {date: (1 - depreciation)**i for i, date in enumerate_date_range.items()}
            Generation_depreciation = pd.Series(dict_depreciation)
            self.Generation_depreciation[generator.name] = Generation_depreciation

    def set_Energy_Generation(self, multiprocessing=False):
        ''' Output: (dict) for each generator, a hourly generation dataframe for each simulation
            >>> {generators: generation(hours, simulations)} '''
        month_markers = self.get_month_markers(freq='H')
        for generator in self.Generators:
            generation_probabilities_interpolation = self.Energy_Generation_probabilities_interpolation[generator.name] #Dataframe, hourly freq
            scenarios = self.Generation_scenarios[generator.name] #Dataframe, monthly freq
            simulations = self.simulations
            indices = [(month_markers, scenarios[simulation], generation_probabilities_interpolation) for simulation in simulations]
            if multiprocessing:
                n_cpu = mp.cpu_count()
                pool = mp.Pool(processes=n_cpu)
                frames = pool.map(set_energy_generation_series_helper, indices)
                pool.close()
                pool.join()
            else:
                frames = [set_energy_generation_series_helper(index) for index in indices]
            energy_generation = pd.concat(frames, axis=1).sort_index(axis=1)
            # We apply the depreciation rate
            date_range = self.get_date_range('H')
            generation_depreciation = self.Generation_depreciation[generator.name]
            generation_depreciation = generation_depreciation.reindex(index=date_range, method='ffill')
            energy_generation = energy_generation.multiply(generation_depreciation, axis=0)
            # We rename the energy generation dataframe
            energy_generation.index.name = generator.name
            self.Energy_Generation[generator.name] = energy_generation
            print(u'>>> ... Generating Production for {} ---> OK'.format(generator.name))

    #---------------------------
    #SET RANDOM WALK MODELLING
    #---------------------------
    def check_availability_Random_Walk(self):
        volatility = self.Price.Volatilidad
        oil_price = self.Price.Precio_Petroleo
        default_volatility = 0
        default_oil_price = 0
        if not isinstance(volatility, float):
            print('Missing Volatility for oil price. Default value set: {}'.format(default_volatility))
            self.Price.Volatilidad = default_volatility
        if not isinstance(oil_price, float):
            print('Missing Oil Price for oil price. Default value set: {}'.format(default_oil_price))
            self.Price.Precio_Petroleo = default_oil_price

    def set_Random_Walk(self):
        date_range = self.get_date_range(freq='A')
        volatility = self.Price.Volatilidad
        OilAdjust = self.parameters.OilAdjust
        simulations = self.simulations
        #random_seed = self.parameters.random_seed
        self.Random_Walk.value = Random_Walk(volatility, OilAdjust, date_range, simulations)

    #-----------------------------------
    #INTRODUCE SPOT PRICE VARIABILITY
    #-----------------------------------
    def check_Random_Walk_SPOT_Price_parameters(self):
        ''' Check the availability of the price adjustement parameters for each node.'''
        alpha = {node: self.Price.Dict_Nudos[node].Pendiente for node in self.all_nodes}
        beta = {node: self.Price.Dict_Nudos[node].Constante for node in self.all_nodes}
        default_alpha = 0
        default_beta = 0
        for node in self.all_nodes:
            if not isinstance(alpha[node], float):
                print(u'Missing Slope for oil price in node {}. Default value set: {}'.format(node, default_alpha))
                self.Price.Dict_Nudos[node].Pendiente = default_alpha
            if not isinstance(beta[node], float):
                print(u'Missing Constant for oil price in node {}. Default value set: {}'.format(node, default_beta))
                self.Price.Dict_Nudos[node].Constante = default_beta

    def apply_SPOT_Price_Variability(self):
        ''' Output: (dict of dataframes) SPOT price reajusted with Brent Random Walk, for each node, for each hour in the date range, for each simulation.
            >>> {node: price(hours, simulations)} '''
        date_range = self.get_date_range(freq='H')
        delta_Random_Walk = self.Random_Walk.value - 1
        delta_Random_Walk = delta_Random_Walk.reindex(index=date_range, method='ffill')
        for node in self.all_nodes:
            spot_price = self.SPOT_price[node]
            name = spot_price.index.name
            alpha = self.Price.Dict_Nudos[node].Pendiente
            beta = self.Price.Dict_Nudos[node].Constante
            delta_spot_price = alpha*delta_Random_Walk + beta
            spot_price = spot_price.multiply(1 + delta_spot_price)
            spot_price.index.name = name
            self.SPOT_price[node] = spot_price

    #-----------------------------------
    #INTRODUCE PPA DEMAND VARIABILITY
    #-----------------------------------
    def set_PFI_Demand(self):
        ''' Output: (dict of df) PFI calculated fr each consumer, for each simulation, for each hour in the date range
            >>> {consumer: PFI(years, simulations)} '''
        date_range = self.get_date_range(freq='A')
        #alpha_base = self.Price.get_alpha_base_test()
        #beta_base = self.Price.get_beta_base_test()
        alpha_base = self.Price.get_alpha_base()
        beta_base = self.Price.get_beta_base()
        indices = self.Price.get_price_indices() # diesel, fuel, carbon, gnl, brent
        oil_price = self.Price.Precio_Petroleo
        Random_Walk = self.Random_Walk.value
        #Random_Walk = Random_Walk.reindex(index=date_range, method='ffill')
        PI = {index: alpha_base[index]*oil_price*Random_Walk + beta_base[index] for index in indices} #Projected price (dict of dataframes)
        for consumer in self.Consumers:
            #if True:
            if consumer.brent_indexation_Flag:
                #brent = 100
                brent = consumer.brent_0
                PI_base = {index: alpha_base[index]*brent + beta_base[index] for index in indices} #Projected price por each consumer (dict of floats)
                #a = consumer.get_weighted_elements_test()
                a = consumer.get_weighted_elements() #{a_fuel, a_brent, a_carbon, a_diesel, a_gnl}
                ''' 
                The sum of all the weighted elements must equal 1:
                    a_fuel + a_brent + a_carbon + a_diesel + a_gnl + a_6 = 1
                Thus, we deduce the value of the ultimate element (the 6th one): '''
                a_6 = 1 - a.sum()
                indexation = {index: a[index]*(PI[index] / PI_base[index]) for index in indices}
                PFI = sum(indexation.values()) + a_6
            else:
                simulations = self.simulations
                PFI = pd.DataFrame(1, index=date_range, columns=simulations) 
            self.PFI_Demand[consumer.name] = PFI

    def set_PPA_Demand(self):
        date_range = self.get_date_range(freq='M')
        for consumer in self.Consumers:
            PPA_series = self.PPA_Demand_series[consumer.name] #type:series, freq='M'
            PFI = self.PFI_Demand[consumer.name] #type:dataframe, freq='A'
            #We reindex the PFI from an annual to a mensual frequency
            PFI = PFI.reindex(index=date_range, method='ffill')
            PPA = PFI.multiply(PPA_series, axis=0)
            self.PPA_Demand[consumer.name] = PPA

    #---------------------------------------
    #INTRODUCE PPA GENERATION VARIABILITY
    #---------------------------------------
    def check_PPA_Generation_parameters(self):
        ''' Check the availability of the PPA ajustment parameters for each generator. '''
        default_PPAbase = 0
        default_PPAalt = 0
        default_alpha = 0
        default_beta = 0
        default_variable_cost_0 = 0
        alpha = {generator.name: generator.vc_pendiente for generator in self.Generators}
        beta = {generator.name: generator.vc_constante for generator in self.Generators}
        marginal_cost_0 = {generator.name: generator.variable_cost_0 for generator in self.Generators}
        for generator in self.Generators:
            if generator.brent_indexation_Flag:
                if not isinstance(alpha[generator.name], float):
                    print(u'Missing marginal cost coefficient alpha for generator {}. Default value set: {}'.format(generator.name, default_alpha))
                    generator.vc_pendiente = default_alpha
                if not isinstance(beta[generator.name], float):
                    print(u'Missing marginal cost constant beta for generator {}. Default value set: {}'.format(generator.name, default_beta))
                    generator.vc_constante = default_beta
            else:
                if generator.variable_cost_Flag:
                    if not isinstance(marginal_cost_0[generator.name], float):
                        print(u'Missing marginal cost constant by default for generator {}. Default value set: {}'.format(generator.name, default_variable_cost_0))
                        generator.variable_cost_0 = default_variable_cost_0
                else:
                    if not isinstance(generator.PPAbase, float):
                        print(u'Missing Base PPA for generator {}. Default value set: {}'.format(generator.name, default_PPAbase))
                        generator.PPAbase = default_PPAbase
                    if not isinstance(generator.PPAalt, float):
                        print(u'Missing Alternative PPA for generator {}. Default value set: {}'.format(generator.name, default_PPAalt))
                        generator.PPAalt = default_PPAalt

    def apply_PPA_Generation_Variability(self):
        ''' Output: (dict of dataframes) PPA (base PPA or alternative PPA) for each generator, for each year in the date_range, for each simulation.
            >>> {generator: PPA(years, simulations)} '''
        date_range = self.get_date_range(freq='A')
        for generator in self.Generators:
            if generator.brent_indexation_Flag:
                alpha = generator.vc_pendiente
                beta = generator.vc_constante
                Random_Walk = self.Random_Walk.value
                oil_price = self.Price.Precio_Petroleo
                oil_price = oil_price * Random_Walk
                PPAbase = alpha*oil_price + beta
                PPAalt = alpha*oil_price + beta
            else:
                simulations = self.simulations
                ones = pd.DataFrame(1, index=date_range, columns=simulations)
                if generator.variable_cost_Flag:
                    variable_cost_0 = generator.variable_cost_0
                    PPAbase = ones.multiply(variable_cost_0)
                    PPAalt = ones.multiply(variable_cost_0)
                else:
                    PPAbase = ones.multiply(generator.PPAbase)
                    PPAalt = ones.multiply(generator.PPAalt)
            self.PPAbase[generator.name] = PPAbase
            self.PPAalt[generator.name] = PPAalt

    #------------------------------------------
    #INTRODUCE ENERGY GENERATION VARIABILITY
    #------------------------------------------
    def adjust_Energy_Generation(self):
        ''' If variable cost option is activated, the generator will only produce energy when the SPOT price is greater than its own variable costs.
        Output: (dict of dataframes) Energy generation for each generator, for each hour in the date range, for each simulation.
            >>> {generator: generation(hours, simulations)} '''
        date_range = self.get_date_range(freq='H')
        for generator in self.Generators:
            if generator.variable_cost_Flag:
                #load matrices
                energy_generation = self.Energy_Generation[generator.name] #Hourly freq
                spot_price = self.SPOT_price[generator.subestacion] #Hourly freq
                variable_cost = self.PPAbase[generator.name] #Yearly freq
                #We reindex the variable costs matrix from yearly to hourly freq
                variable_cost = variable_cost.reindex(index=date_range, method='ffill')
                #Generation start only when spot prices are greater than marginal costs, otherwise there is no generation
                energy_generation = energy_generation.where(spot_price > variable_cost, 0)
                self.Energy_Generation[generator.name] = energy_generation

    #-------------------------
    #REINDEX DATA FREQUENCY
    #-------------------------
    def reindex_Energy_Demand_frequency(self, freq):
        date_range = self.get_date_range(freq=freq)
        for consumer in self.Consumers:
            energy_demand = self.Energy_Demand[consumer.name]
            if freq == 'H':
                energy_demand = energy_demand.reindex(index=date_range, method='ffill')
            elif freq == 'M':
                energy_demand = energy_demand.resample('MS').sum() # 'MS' for 'Month start'
            self.Energy_Demand[consumer.name] = energy_demand

    def reindex_Energy_Generation_frequency(self, freq):
        date_range = self.get_date_range(freq=freq)
        for generator in self.Generators:
            energy_generation = self.Energy_Generation[generator.name]
            if freq == 'H':
                energy_generation = energy_generation.reindex(index=date_range, method='ffill')
            elif freq == 'M':
                energy_generation = energy_generation.resample('MS').sum() # 'MS' for 'Month start'
            self.Energy_Generation[generator.name] = energy_generation

    #------------------------------
    #CREATE ENERGY DEMAND PROFIT
    #------------------------------
    def set_Purchased_Energy_Demand_Cost(self):
        ''' Output: (dict of dataframes) Cost induced by the purchase of energy to the SPOT market. The calculation is performed on an hourly basis, then, we resample the result in a montly frequency.
            >>> {consumer: cost(months, simulations)}'''
        date_range = self.get_date_range(freq='M') #To reindex the result in a monthly frequency
        for consumer in self.Consumers:
            energy_demand = self.Energy_Demand[consumer.name] #Dataframe, hourly freq
            spot_price = self.SPOT_price[consumer.subestacion] #Dataframe, hourly freq
            purchased_energy_cost = energy_demand.multiply(spot_price)
            purchased_energy_cost = purchased_energy_cost.resample('MS').sum()
            self.Purchased_Energy_Demand_Cost[consumer.name] = purchased_energy_cost

    def resample_Energy_Demand_to_monthly_frequency(self):
        ''' Output: (dict of dataframes) We resample the energy demand to an hourly frequency along the date range.
            >>> {consumer: energy_demand(months, simulations)} '''
        for consumer in self.Consumers:
            energy_demand = self.Energy_Demand[consumer.name] #Dataframe, hourly freq
            energy_demand = energy_demand.resample('MS').sum()
            self.Energy_Demand[consumer.name] = energy_demand

    def set_Sold_Energy_Demand_Incomes(self):
        ''' Output: (dict of dataframes) Incomes induced by the sales of energy to the final consumers.
            >>> {consumer: incomes(months, simulations)}'''
        for consumer in self.Consumers:
            energy_demand = self.Energy_Demand[consumer.name] #Dataframe, monthly freq
            PPA_Demand = self.PPA_Demand[consumer.name] #Dataframe, monthly freq
            sold_energy_incomes = energy_demand.multiply(PPA_Demand)
            self.Sold_Energy_Demand_Incomes[consumer.name] = sold_energy_incomes

    def set_Sold_Energy_Demand_Profit(self):
        ''' Output: (dict of dataframes) Profit induced by difference between sales incomes and purchasing costs.
            >>> {consumer: profit(months, simulations)}'''
        for consumer in self.Consumers:
            sold_energy_incomes = self.Sold_Energy_Demand_Incomes[consumer.name]
            purchased_energy_cost = self.Purchased_Energy_Demand_Cost[consumer.name]
            sold_energy_profit = sold_energy_incomes - purchased_energy_cost
            self.Sold_Energy_Demand_Profit[consumer.name] = sold_energy_profit

    def set_Purchased_Energy_Demand_Price(self):
        ''' Output: (dict of dataframes)
            >>> {consumer: Purchase_Price(months, simulations)}'''
        for consumer in self.Consumers:
            costs = self.Purchased_Energy_Demand_Cost[consumer.name]
            purchased_volume = self.Energy_Demand[consumer.name]
            purchase_price = costs / purchased_volume
            self.Purchased_Energy_Demand_Price[consumer.name] = purchase_price

    #-- Total energy
    def set_Total_Energy_Demand(self):
        ''' Output: (dataframe) Sum of all the consumers' energy demand for each simulation for each month in the date range.
            >>> demand(hours, simulation)'''
        total_energy_demand = sum(self.Energy_Demand[consumer.name] for consumer in self.Consumers)
        total_energy_demand = total_energy_demand.resample('MS').sum() # 'MS' for 'Month start'
        total_energy_demand.index.name = "Total Energy Demand"
        self.Total_Energy_Demand.value = total_energy_demand

    #----------------------------------
    #CREATE ENERGY GENERATION PROFIT
    #----------------------------------
    def set_Energy_Generation_Threshold(self):
        ''' Output: (dict of dataframes) Mensual energy generation volume thresholds [GWh] for each generators, for each month in the date range, for each threshold (Threshold 1, Threshold 2).
            >>> {generator: Maximum_Generation(months, ['Threshold 1', 'Threshold 2']) [GWh] '''
        date_range = self.get_date_range(freq='M')
        for generator in self.Generators:
            GenMax = generator.get_GenMax()
            thresholds = GenMax.columns
            energy_generation_threshold = pd.DataFrame(index=date_range, columns=thresholds)
            indices = [(date, threshold) for date in date_range for threshold in thresholds]
            for date, threshold in indices:
                energy_generation_threshold.at[date, threshold] = GenMax.at[date.month, threshold]
            self.Energy_Generation_Threshold[generator.name] = energy_generation_threshold

    def set_Monthly_Cumulative_Generation(self):
        ''' Output: (dict of dataframes) Mensual accumulated energy generation, for each generator, for each hour in the date range, for each simulation.
            >>> {generator: mensual_cumsum_generation(hours, simulation)'''
        month_range_series = self.get_month_range_series()
        for generator in self.Generators:
            energy_generation = self.Energy_Generation[generator.name]
            frames = []
            for date_range in month_range_series.values():
                monthly_generation = energy_generation.loc[date_range]
                cumulative_generation = monthly_generation.cumsum(axis=0)
                frames.append(cumulative_generation)
            monthly_cumsum_generation = pd.concat(frames, axis=0)
            monthly_cumsum_generation = monthly_cumsum_generation.sort_index()
            self.Monthly_Cumulative_Generation[generator.name] = monthly_cumsum_generation

    def set_Energy_Generation_P90(self):
        ''' Output: (dict of dataframes) P90 represents the monthly energy generated volume that is produced below the threshold 1, for each generator, for each month of the date range, for each simulation.
            >>> {generator: Energy_P90(months, simulations)} '''
        date_range = self.get_date_range(freq='H')
        for generator in self.Generators:
            energy_generation = self.Energy_Generation[generator.name]
            cumsum_generation = self.Monthly_Cumulative_Generation[generator.name]
            generation_threshold = self.Energy_Generation_Threshold[generator.name]
            threshold_1 = generation_threshold['Threshold 1'] # Series, monthly frequency
            threshold_1 = threshold_1.reindex(index=date_range, method='ffill')
            is_under_threshold_1 = cumsum_generation.le(threshold_1, axis=0) #Dataframe, True when cumsum_generation <= threshold, else False ('le' means lower or equal)
            energy_P90 = energy_generation.where(is_under_threshold_1, 0)
            self.Energy_Generation_P90[generator.name] = energy_P90

    def set_Energy_Generation_P50(self):
        ''' Output: (dict of dataframes) P50 represents the monthly energy generated volume that is produced between the threshold 1 and the threshold 2, for each generator, for each month of the date range, for each simulation.
            >>> {generator: Energy_P50(months, simulations)} '''
        date_range = self.get_date_range(freq='H')
        for generator in self.Generators:
            energy_generation = self.Energy_Generation[generator.name]
            cumsum_generation = self.Monthly_Cumulative_Generation[generator.name]
            generation_threshold = self.Energy_Generation_Threshold[generator.name]
            threshold_1 = generation_threshold['Threshold 1'] # Series, monthly frequency
            threshold_2 = generation_threshold['Threshold 2'] # Series, monthly frequency
            threshold_1 = threshold_1.reindex(index=date_range, method='ffill')
            threshold_2 = threshold_2.reindex(index=date_range, method='ffill')
            is_above_threshold_1 = cumsum_generation.gt(threshold_1, axis=0) #Dataframe, True when cumsum_generation > threshold_1, else False ('gt' means greater)
            is_under_threshold_2 = cumsum_generation.le(threshold_2, axis=0) #Dataframe, True when cumsum_generation <= threshold_2, else False ('le' means lower or equal)
            is_between_threshold_1_2 = is_above_threshold_1 & is_under_threshold_2 #Dataframe, True when threshold_1 < cumsum_generation <= threshold_2, else False
            energy_P50 = energy_generation.where(is_between_threshold_1_2, 0)
            self.Energy_Generation_P50[generator.name] = energy_P50

    #Useful to calcultate extra oprational margins
    def set_Sold_Energy_Generation_P90_Incomes(self):
        ''' Output: (dict of dataframes) Incomes induced by the sales of P90 energy, for each generator, for each month of the date range, for each simulation.
            >>> {generator: Incomes_P90(months, simulations)} '''
        for generator in self.Generators:
            energy_P90 = self.Energy_Generation_P90[generator.name] #Hourly freq
            spot_price = self.SPOT_price[generator.subestacion] #Hourly freq
            sold_energy_P90_incomes = energy_P90.multiply(spot_price)
            sold_energy_P90_incomes = sold_energy_P90_incomes.resample('MS').sum()
            self.Sold_Energy_Generation_P90_Incomes[generator.name] = sold_energy_P90_incomes

    #Useful to calcultate extra oprational margins
    def set_Sold_Energy_Generation_P50_Incomes(self):
        ''' Output: (dict of dataframes) Incomes induced by the sales of P50 energy, for each generator, for each month of the date range, for each simulation.
            >>> {generator: Incomes_P50(months, simulations)} '''
        for generator in self.Generators:
            energy_P50 = self.Energy_Generation_P50[generator.name] #Hourly freq
            spot_price = self.SPOT_price[generator.subestacion] #Hourly freq
            sold_energy_P50_incomes = energy_P50.multiply(spot_price)
            sold_energy_P50_incomes = sold_energy_P50_incomes.resample('MS').sum()
            self.Sold_Energy_Generation_P50_Incomes[generator.name] = sold_energy_P50_incomes

    def set_Sold_Energy_Generation_Incomes(self):
        ''' Output: (dict of dataframes) Incomes induced by the sales of usable energy (P50 + P90), for each generator, for each month of the date range, for each simulation.
            >>> {generator: Incomes(months, simulations)} '''
        for generator in self.Generators:
            sold_energy_P50_incomes = self.Sold_Energy_Generation_P50_Incomes[generator.name] #Monthly freq
            sold_energy_P90_incomes = self.Sold_Energy_Generation_P90_Incomes[generator.name] #Monthly freq
            sold_energy_incomes = sold_energy_P50_incomes + sold_energy_P90_incomes
            self.Sold_Energy_Generation_Incomes[generator.name] = sold_energy_incomes
    
    def resample_Energy_Generation_to_monthly_frequency(self):
        ''' Output: (dict of dataframes) We resample the energy generation to an hourly frequency along the date range.
            >>> {generator: energy_generation(months, simulations)}'''
        for generator in self.Generators:
            energy_generation = self.Energy_Generation[generator.name]
            energy_generation = energy_generation.resample('MS').sum()
            self.Energy_Generation[generator.name] = energy_generation

    def resample_Energy_Generation_P90_to_monthly_frequency(self):
        ''' Output: (dict of dataframes) We resample the energy generation to an hourly frequency along the date range.
            >>> {generator: energy_generation_P90(months, simulations)} '''
        for generator in self.Generators:
            energy_generation_P90 = self.Energy_Generation_P90[generator.name]
            energy_generation_P90 = energy_generation_P90.resample('MS').sum()
            self.Energy_Generation_P90[generator.name] = energy_generation_P90

    def resample_Energy_Generation_P50_to_monthly_frequency(self):
        ''' Output: (dict of dataframes) We resample the energy generation to an hourly frequency along the date range.
            >>> {generator: energy_generation_P50(months, simulations)}'''
        for generator in self.Generators:
            energy_generation_P50 = self.Energy_Generation_P50[generator.name]
            energy_generation_P50 = energy_generation_P50.resample('MS').sum()
            self.Energy_Generation_P50[generator.name] = energy_generation_P50

    def set_Energy_Generation_P90_P50(self):
        ''' Output: (dict of dataframes) We sum the P90 and P50 energy generation to get the usable energy volume for each generator, for each month in the date range, for each simulation.
            >>> {generator: energy_generation_P90_P50(months, simulations)} '''
        for generator in self.Generators:
            energy_generation_P90 = self.Energy_Generation_P90[generator.name] #Monthly freq
            energy_generation_P50 = self.Energy_Generation_P50[generator.name] #Monthly freq
            energy_generation_P90_P50 = energy_generation_P90 + energy_generation_P50
            self.Energy_Generation_P90_P50[generator.name] = energy_generation_P90_P50

    def set_Purchased_Energy_Generation_P90_Cost(self):
        ''' Output: (dict of dataframes) Cost of purchase of P90 energy, for each generator, for each month in the date range, for each simulation.
            >>> {generator: P90_purchase_cost(months, simulations)} '''
        date_range = self.get_date_range(freq='M')
        for generator in self.Generators:
            PPAbase = self.PPAbase[generator.name] #Dataframe, Yearly freq
            PPAbase = PPAbase.reindex(index=date_range, method='ffill')
            energy_generation_P90 = self.Energy_Generation_P90[generator.name]
            purchased_energy_P90_cost = energy_generation_P90.multiply(PPAbase)
            self.Purchased_Energy_Generation_P90_Cost[generator.name] = purchased_energy_P90_cost

    def set_Purchased_Energy_Generation_P50_Cost(self):
        ''' Output: (dict of dataframes) Cost of purchase of P50 energy, for each generator, for each month in the date range, for each simulation.
            >>> {generator: P50_purchase_cost(months, simulations)} '''
        date_range = self.get_date_range(freq='M')
        for generator in self.Generators:
            PPAalt = self.PPAalt[generator.name] #Dataframe, Yearly freq
            PPAalt = PPAalt.reindex(index=date_range, method='ffill')
            energy_generation_P50 = self.Energy_Generation_P50[generator.name]
            purchased_energy_P50_cost = energy_generation_P50.multiply(PPAalt)
            self.Purchased_Energy_Generation_P50_Cost[generator.name] = purchased_energy_P50_cost

    def set_Purchased_Energy_Generation_Cost(self):
        ''' Output: (dict of dataframes) Cost of purchase of usable energy (P50 + P90), for each generator, for each month in the date range, for each simulation.
            >>> {generator: cost_of_purchase(months, simulations)'''
        for generator in self.Generators:
            purchased_energy_P50_cost = self.Purchased_Energy_Generation_P50_Cost[generator.name]
            purchased_energy_P90_cost = self.Purchased_Energy_Generation_P90_Cost[generator.name]
            purchased_energy_cost = purchased_energy_P50_cost + purchased_energy_P90_cost
            self.Purchased_Energy_Generation_Cost[generator.name] = purchased_energy_cost

    def set_Sold_Energy_Generation_Profit(self):
        ''' Output: (dict of dataframes) Profit induced by difference between sales incomes and purchasing costs.
            >>> {generator: Profit(months, simulations)} '''
        for generator in self.Generators:
            sold_energy_incomes = self.Sold_Energy_Generation_Incomes[generator.name]
            purchased_energy_cost = self.Purchased_Energy_Generation_Cost[generator.name]
            sold_energy_profit = sold_energy_incomes - purchased_energy_cost
            self.Sold_Energy_Generation_Profit[generator.name] = sold_energy_profit

    def set_Sold_Energy_Generation_Price(self):
        ''' Output: (dict of dataframes)
            >>> {generator: Sale_Price(months, simulations)} '''
        for generator in self.Generators:
            incomes = self.Sold_Energy_Generation_Incomes[generator.name]
            sold_volume = self.Energy_Generation_P90_P50[generator.name]
            sale_price = incomes / sold_volume
            self.Sold_Energy_Generation_Price[generator.name] = sale_price

    # Total energy
    def set_Total_Energy_Generation_P90_P50(self):
        ''' Output: (dataframes) Sum of all the generation' energy generation (P90 + P50) for each simulation for each month in the date range.
            >>> generation_P90_P50(hours, simulation) '''
        total_energy_generation = sum(self.Energy_Generation[generator.name] for generator in self.Generators)
        total_energy_generation = total_energy_generation.resample('MS').sum() # 'MS' for 'Month start'
        total_energy_generation.index.name = 'Total Energy Generation P90 P50'
        self.Total_Energy_Generation_P90_P50.value = total_energy_generation

    #----------------------------------
    #RESAMPLE DATA BEFORE ASSESSMENT
    #----------------------------------
    def resample_SPOT_price_to_monthly_frequency(self):
        ''' Output: (dict of dataframes) Resample the SPOT price from a hourly to a monthly frequency. We obtain monthly price average for each month in date range, for each simulation.
        Note: We do this because otherwise, the excel assessment lasts too long
            >>> {node: SPOT_price(months, simulation)} '''
        for node in self.all_nodes:
            spot_price = self.SPOT_price[node]
            spot_price = spot_price.resample('MS').mean() # 'MS' for 'Month Start'
            self.SPOT_price[node] = spot_price

    def resample_SPOT_price_modelling_to_monthly_frequency(self):
        ''' Output: (dict of dataframes) Resample the SPOT price modelling from a hourly to a monthly frequency. We obtain monthly price average for each month in date range, for each hydrological scenario.
        Note: We do this because otherwise, the excel assessment lasts too long
            >>> {node: SPOT_price_modelling(months, hydro scenarios)} '''
        for node in self.all_nodes:
            spot_price_modelling = self.SPOT_price_modelling[node]
            spot_price_modelling = spot_price_modelling.resample('MS').mean() # 'MS' for 'Month Start'
            self.SPOT_price_modelling[node] = spot_price_modelling

    def resample_Energy_Generation_probabilities_to_monthly_frequency(self):
        ''' Output: (dict of dataframes) Resample the energy generation probabilities from a hourly to a monthly frequency. We obtain monthly cumulated generation for each month in date range, for each probability (P25, P50, ..., P90, P95).
        Note: We do this because otherwise, the excel assessment lasts too long
            >>> {node: generation_probabilities(months, simulation)} '''
        for generator in self.Generators:
            generation_probabilities = self.Energy_Generation_probabilities[generator.name]
            generation_probabilities = generation_probabilities.resample('MS').sum()
            self.Energy_Generation_probabilities[generator.name] = generation_probabilities

    def resample_Energy_Generation_probabilities_interpolation_to_monthly_frequency(self):
        ''' Output: (dict of dataframes) Resample the energy generation probabilities from a hourly to a monthly frequency. We obtain monthly cumulated generation for each month in date range, for each probability (P25, P50, ..., P90, P95).
        Note: We do this because otherwise, the excel assessment lasts too long
            >>> {node: generation_probabilities(months, simulation)}
        '''
        for generator in self.Generators:
            generation_interpolation = self.Energy_Generation_probabilities_interpolation[generator.name]
            generation_interpolation = generation_interpolation.resample('MS').sum()
            self.Energy_Generation_probabilities_interpolation[generator.name] = generation_interpolation


    #-----------------------------
    #CREATE FINANCIAL ASSESSMENT
    #-----------------------------
    def set_Gross_Profit(self):
        energy_generation_profit = sum(self.Sold_Energy_Generation_Profit[generator.name] for generator in self.Generators)
        energy_demand_profit = sum(self.Sold_Energy_Demand_Profit[consumer.name] for consumer in self.Consumers)
        gross_profit = energy_generation_profit + energy_demand_profit
        gross_profit.index.name = 'Gross Profit (USD)'
        self.Gross_Profit.value = gross_profit #Running total of profit

    def set_Operating_Margin(self):
        Operating_Cost = float(self.Funds.CostesOperativos) / 12
        Operating_Margin = self.Gross_Profit.value - Operating_Cost
        #We need to get a copy to give it an index name, otherwise --> troubles
        Operating_Margin = copy.deepcopy(Operating_Margin)
        Operating_Margin.index.name = 'Operating Margin (USD)'
        self.Operating_Margin.value = Operating_Margin

    def set_Extra_Operating_Margin(self):
        Operating_Cost = float(self.Funds.CostesOperativos) / 12
        Extra_Operating_Incomes = sum(self.Sold_Energy_Generation_P50_Incomes[generator.name] for generator in self.Generators)
        Extra_Operating_Cost = sum(self.Purchased_Energy_Generation_P50_Cost[generator.name] for generator in self.Generators)
        #Extra Operating Profit
        Extra_Operating_Profit = Extra_Operating_Incomes - Extra_Operating_Cost
        #Extra Operating Margin
        Extra_Operating_Margin = Extra_Operating_Profit - Operating_Cost
        Extra_Operating_Margin.index.name = 'Extra Operating Margin (USD)'
        self.Extra_Operating_Margin.value = Extra_Operating_Margin

    def set_Operating_Margin_Taxes(self):
        Operating_Margin = self.Operating_Margin.value
        Annual_Operating_Margin = Operating_Margin.resample('A').sum() #We get the annual Operating Margin on December, 31st
        tax_rate = float(self.Funds.TasaImpositiva) / 100
        #Taxes will be paid if the annual Operating Margin is positive
        Positive_Annual_Operating_Margin = Annual_Operating_Margin.clip(lower=0) #.where(Annual_Operating_Margin > 0, 0)
        Operating_Margin_Taxes = Positive_Annual_Operating_Margin.multiply(tax_rate)
        #We rename the taxes date (April, the 1st of year n+1 replaces December, 31st of year n)
        date_range = Operating_Margin_Taxes.index
        april_range = {date: date.replace(year=date.year+1, month=4, day=1) for date in date_range}
        Operating_Margin_Taxes = Operating_Margin_Taxes.rename(index=april_range)
        #We reindex the taxes
        date_range = self.get_date_range(freq='M')
        Operating_Margin_Taxes = Operating_Margin_Taxes.reindex(index=date_range, fill_value=0)
        #We need to get a copy to give it an index name, otherwise --> troubles
        Operating_Margin_Taxes = copy.deepcopy(Operating_Margin_Taxes)
        Operating_Margin_Taxes.index.name = 'Operating Margin Taxes (USD)'
        self.Operating_Margin_Taxes.value = Operating_Margin_Taxes

    def set_Net_Incomes(self):
        Net_Incomes = self.Operating_Margin.value - self.Operating_Margin_Taxes.value
        #We need to get a copy to give it an index name, otherwise --> troubles
        Net_Incomes = copy.deepcopy(Net_Incomes)
        Net_Incomes.index.name = 'Net Incomes (USD)'
        self.Net_Incomes.value = Net_Incomes

    def set_Net_Liquid_Assets_and_Dividend(self):
        Capital = float(self.Funds.Capital) * 1e6
        dividend_rate = float(self.Funds.UmbralReparticion) / 100
        threshold = Capital * dividend_rate
        #We get the monthly date range
        date_range = self.get_date_range(freq='M')
        start = date_range.min()
        end = date_range.max()
        #We get an annual range that starts from the first dividend distribution
        start_dividend = self.Funds.start_pago_date
        freq = __Annual_Start__[start_dividend.month]
        dividend_range = pd.date_range(start=start_dividend, end=end, freq=freq)
        #Initialization
        Net_Incomes = self.Net_Incomes.value
        columns = Net_Incomes.columns
        Net_Liquid_Assets = pd.DataFrame(0, index=date_range, columns=columns)
        Dividend = pd.DataFrame(0, index=date_range, columns=columns)
        #Let's go !
        Net_Liquid_Assets.loc[start] = Capital + Net_Incomes.loc[start]
        for date in date_range[1:]:
            Net_Liquid_Assets.loc[date] = Net_Liquid_Assets.loc[date-1] + Net_Incomes.loc[date]
            if date in dividend_range:
                Liquid_Assets = copy.deepcopy(Net_Liquid_Assets.loc[date])
                #Net Liquid Assets calculation
                Net_Liquid_Assets.loc[date] = Liquid_Assets.clip(upper=threshold)
                #Dividend Calculation
                delta = Liquid_Assets - threshold
                Dividend.loc[date] = delta.clip(lower=0)
        Dividend = copy.deepcopy(Dividend) #Otherwise, Dividend.index.name will be equal to Net_Liquid_Assets.index.name
        Dividend.index.name = 'Dividend (USD)'
        Net_Liquid_Assets = copy.deepcopy(Net_Liquid_Assets)
        Net_Liquid_Assets.index.name = 'Net Liquid Assets (USD)'
        self.Net_Liquid_Assets.value = Net_Liquid_Assets
        self.Dividend.value = Dividend

    def set_Benchmark(self):
        date_range = self.get_date_range(freq='M')
        Capital = float(self.Funds.Capital) * 1e6
        annual_rate = float(self.Funds.UmbralInicial) / 100
        mensual_rate = (1 + annual_rate)**(1/12.) - 1
        Dividend = self.Dividend.value
        columns = Dividend.columns
        Benchmark = pd.DataFrame(0, index=date_range, columns=columns)
        #Initialization
        start = date_range.min()
        Benchmark.loc[start] = -Capital + Dividend.loc[start]
        #Recurrence
        for date in date_range[1:]:
            Benchmark.loc[date] = Benchmark.loc[date-1]*(1 + mensual_rate) + Dividend.loc[date]
        Benchmark.index.name = 'Benchmark (USD)'
        self.Benchmark.value = Benchmark

    def set_Success_Fee(self):
        date_range = self.get_date_range('M')
        Dividend = self.Dividend.value
        Benchmark = self.Benchmark.value
        Success_Fee = Dividend.applymap(lambda x: self._calculation_Success_Fee(x))
        Success_Fee[Benchmark <= 0] = 0
        #We need to get a copy to give it an index name, otherwise --> troubles
        Success_Fee = copy.deepcopy(Success_Fee)
        Success_Fee.index.name = 'Success Fee (USD)'
        self.Success_Fee.value = Success_Fee

    def _calculation_Success_Fee(self, dividend):
        Capital = float(self.Funds.Capital) * 1e6
        VAT = float(self.Funds.iva) / 100
        #Performance thresholds
        threshold_1 = float(self.Funds.UmbralPerformance1) / 100
        threshold_2 = float(self.Funds.UmbralPerformance2) / 100
        #Commission Rates
        commission_rate_1 = float(self.Funds.ComisionTramo1) / 100
        commission_rate_2 = float(self.Funds.ComisionTramo2) / 100
        #Let's go !
        if dividend <= Capital * threshold_1:
            success_fee = 0
        elif dividend <= Capital * threshold_2:
            success_fee = (dividend - Capital*threshold_1)*commission_rate_1*(1 + VAT)
        else:
            success_fee_1 = (dividend - Capital*threshold_2)*commission_rate_2*(1 + VAT)
            success_fee_2 = Capital*threshold_1*commission_rate_1*(1 + VAT)
            success_fee = success_fee_1 + success_fee_2
        return success_fee

    def set_Shareholder_Cash_Flow(self):
        Shareholder_Cash_Flow = self.Dividend.value - self.Success_Fee.value
        Capital = float(self.Funds.Capital) * 1e6
        start = Shareholder_Cash_Flow.index.min()
        Shareholder_Cash_Flow.loc[start] -= Capital
        #We need to get a copy to give it an index name, otherwise --> troubles
        Shareholder_Cash_Flow = copy.deepcopy(Shareholder_Cash_Flow)
        Shareholder_Cash_Flow.index.name = 'Shareholder Cash Flow (USD)'
        self.Shareholder_Cash_Flow.value = Shareholder_Cash_Flow

    def set_IRR_Shareholder_Cash_Flow(self):
        ''' Output: (series) Internal Rate Return calculation on Shareholder Cash flow for each simulation.
        Note: annual_IRR = (1 + mensual_IRR)^(number of n periods in 1 t period) - 1
            >>> IRR(simulations) '''
        shareholder_cash_flow = self.Shareholder_Cash_Flow.value
        IRR = shareholder_cash_flow.apply(lambda x: (1 + np.irr(x.values))**12 - 1)
        IRR.name = 'IRR (%)'
        self.IRR_Shareholder_Cash_Flow.value = IRR

    def set_NPV_Shareholder_Cash_Flow(self):
        ''' Output: (series) Net Present Value calculation on Shareholder Cash flow for each simulation.
        Note: annual_NPV = NPV(mensual_rate, range of projected values)
        where: mensual_rate = (1 + annual_rate)^(1/12) - 1 '''
        discount_rate = self.parameters.discount_rate / 100
        mensual_rate = (1 + discount_rate)**(1./12) - 1
        shareholder_cash_flow = self.Shareholder_Cash_Flow.value
        NPV = shareholder_cash_flow.apply(lambda x: np.npv(mensual_rate, x.values))
        NPV.name = 'NPV (USD)'
        self.NPV_Shareholder_Cash_Flow.value = NPV

    #--------------------------------------
    # CREATE INSOLVENCY ANALYSIS
    #--------------------------------------
    def set_Insolvency(self):
        Net_Liquid_Assets = self.Net_Liquid_Assets.to_frame()
        Insolvency = Net_Liquid_Assets.where(Net_Liquid_Assets < 0, 0)
        #We need to get a copy to give it an index name, otherwise --> troubles
        Insolvency = copy.deepcopy(Insolvency)
        Insolvency.index.name = 'Insolvency (USD)'
        self.Insolvency.value = Insolvency

    def set_Insolvency_rate(self):
        ''' Output: (series) Evolution of the insolvency rate for each month in the date range.
            >>> Insolvency_rate(months) '''
        n_simulation = self.parameters.n_simulation
        insolvency = self.Insolvency.value
        count = insolvency.where(insolvency == 0, 1)
        cumsum_count = count.cumsum(axis=0)
        is_insolvency = cumsum_count.where(cumsum_count == 0, 1)
        count_insolvency = is_insolvency.sum(axis=1)
        insolvency_rate = count_insolvency.multiply(100./n_simulation)
        insolvency_rate.index.name = None #we delete the index name inherrited from Insolvency
        insolvency_rate.name = 'Insolvency rate (%)'
        self.Insolvency_rate.value = insolvency_rate

    def set_Insolvency_Coverage_Cost(self):
        ''' Output: (series) Evolution of the theoretical cost (USD) in function of the insolvency coverage (%). '''
        # Insolvemcy
        insolvency = self.Insolvency.value
        # Minimum of insolvency for each simulation
        min_insolvency = insolvency.min(axis=0) 
        # Insolvency Coverage Cost
        percentiles = [x/100. for x in range(0, 101, 5)]
        dropList = ['count', 'mean', 'std', 'min', 'max']
        insolvency_coverage_cost = get_summary_series(-min_insolvency, percentiles, dropList)
        insolvency_coverage_cost.name = 'Insolvency Coverage Cost (USD)'
        self.Insolvency_Coverage_Cost.value = insolvency_coverage_cost

    def set_Theoretical_Capital(self):
        ''' Output: (series) Evolution of the theoretical capital (USD) in function of the insolvency coverage (%).
            >>> Theoretical_Capital(percentiles) '''
        Capital = float(self.Funds.Capital) * 1e6
        insolvency_coverage_cost = self.Insolvency_Coverage_Cost.value
        # Theoretical Capital
        theoretical_capital = Capital + insolvency_coverage_cost
        theoretical_capital.name = '''Theoretical Capital (USD) for different 
                                    levels of insolvency coverage (%)'''
        self.Theoretical_Capital.value = theoretical_capital

    def set_Theoretical_Shareholder_Cash_Flow(self):
        ''' Output: (dict of dataframe) Theoretical Shareholder Cash Flow for each level of insolvency coverage by adding the cost of the insolvency coverage at the first date.
            >>> {coverage_levels: Shareholder_Cash_Flow(months, simulations)} '''
        insolvency_coverage_cost = self.Insolvency_Coverage_Cost.value
        coverage_levels = insolvency_coverage_cost.index
        for level in coverage_levels:
            theoretical_shareholder_cash_flow = copy.deepcopy(self.Shareholder_Cash_Flow.value)
            start = theoretical_shareholder_cash_flow.index.min()
            theoretical_shareholder_cash_flow.loc[start] -= insolvency_coverage_cost[level]
            # Name
            name = 'Theoretical Shareholder Cash Flow at {level} of insolvency coverage'.format(level=level)
            theoretical_shareholder_cash_flow.index.name = name
            # Return
            self.Theoretical_Shareholder_Cash_Flow[level] = theoretical_shareholder_cash_flow

    def set_Theoretical_IRR(self):
        ''' Output: (dataframe) Theoretical Internal Rate of Return for each level of insolvency coverage.
            >>> Theoretical_IRR(coverage_levels, simulations) '''
        theoretical_shareholder_cash_flow = self.Theoretical_Shareholder_Cash_Flow.to_dict()
        coverage_levels = theoretical_shareholder_cash_flow.keys()
        frames = []
        for level in coverage_levels:
            # Theoretical IRR Series
            theoretical_IRR_series = theoretical_shareholder_cash_flow[level].apply(lambda x: (1 + np.irr(x.values))**12 - 1)
            theoretical_IRR_series.name = level
            frames.append(theoretical_IRR_series)
        # Theoretical IRR
        theoretical_IRR = pd.concat(frames, axis=1).T #We transpose the matrix to coverage levels in indices and simulations in columns
        theoretical_IRR.index.name = 'Theoretical IRR (%) for different levels of insolvency coverage (%)'
        # Return
        self.Theoretical_IRR.value = theoretical_IRR

    def set_Theoretical_NPV(self):
        ''' Output: (dataframe) Theoretical Net Present Value for each level of insolvency coverage.
            >>> Theoretical_NPV(coverage_levels, simulations) '''
        discount_rate = self.parameters.discount_rate / 100.
        mensual_rate = (1 + discount_rate)**(1./12) - 1
        theoretical_shareholder_cash_flow = self.Theoretical_Shareholder_Cash_Flow.to_dict()
        coverage_levels = theoretical_shareholder_cash_flow.keys()
        frames = []
        for level in coverage_levels:
            # Theroretical NPV Series
            theoretical_NPV_series = theoretical_shareholder_cash_flow[level].apply(lambda x: np.npv(mensual_rate, x.values))
            theoretical_NPV_series.name = level
            frames.append(theoretical_NPV_series)
        # Theoretical NPV
        theoretical_NPV = pd.concat(frames, axis=1).T #We transpose the matrix to coverage levels in indices and simulations in columns
        theoretical_NPV.index.name = 'Theoretical NPV (USD) for different levels of insolvency coverage (%)'
        # Return
        self.Theoretical_NPV.value = theoretical_NPV

    #--------------------------------------
    #SET PARAMETERS TABLE
    #--------------------------------------
    def set_Computation_Parameters(self):
        ''' Output: (Series) Summary concerning the computation parameters.'''
        ordered_list = [
            ('Price Forecast', self.Price.name),
            ('Price Simulation Mode (Fast)', self.parameters.Price_Calculation_Fast_Mode),
            ('Number of Simulations', self.parameters.n_simulation),
            ('Discount Rate (%)', self.parameters.discount_rate),
            ('Valuation Date', self.parameters.Valuation_date.strftime('%Y-%m-%d')),
            ('Random Seed', self.parameters.random_seed),
            ('Oil Price Adjustment', self.parameters.OilAdjust),
            ('Override Uniform Generation', self.parameters.Override_Uniform_Generation)
        ]
        dict_summary = OrderedDict(ordered_list)
        computation_parameters = pd.Series(dict_summary, name='Computation Parameters')
        self.Computation_Parameters = computation_parameters

    def set_Portofolio_Features(self):
        ''' Output: (Series) Summary concerning the portofolio features.'''
        ordered_list = [
            ('Operational Costs (USD/year)', round(self.Funds.CostesOperativos, 1)),
            ('Capital (MUSD)', round(self.Funds.Capital, 1)),
            ('Treasury Rate (%)', self.Funds.Rendimiento),
            ('Profit Sharing Point (%)', self.Funds.UmbralReparticion),
            ('Tax Rate (%)', self.Funds.TasaImpositiva),
            ('Generators Profit Share Point (%)', self.Funds.BeneficioDistGeneradores),
            ('Commission Rate (%)', self.Funds.Comision),
            ('Profit Share Starting Point', self.Funds.start_pago_date.strftime('%b %Y'))
        ]
        dict_summary = OrderedDict(ordered_list)
        portofolio_features = pd.Series(dict_summary, name='Portofolio Features')
        self.Portofolio_Features = portofolio_features

    def set_Generation_Assets(self):
        '''Output: (dataframe) Features the overhall generation summary for each generator.
            >>> summary(index, [Name, Company, Contract, Substation, Life Span, Energy Contracted, PPA])'''
        frames = []
        for generator in self.Generators:
            # Life Span
            start = generator.start_date.strftime('%b %Y')
            end = generator.end_date.strftime('%b %Y')
            life_span = '{start} to {end}'.format(start=start, end=end)
            # Energy Contracted
            max_generation = generator.get_GenMax()
            scale = 1e-3 # We convert from MWh to GWh
            energy_contracted = max_generation['Threshold 2'].sum() * scale
            # PPA Ponderated
            PPAbase = generator.PPAbase
            PPAalt = generator.PPAalt
            Volume1 = max_generation['Threshold 1'].sum() # Volume P90
            Volume2 = max_generation['Threshold 2'].sum() - Volume1 # Volume P50
            PPA = (PPAbase*Volume1 + PPAalt*Volume2) / (Volume1 + Volume2)
            # Generation Assets Summary
            ordered_list = [
                ('Name', generator.name),
                ('Company', generator.empresa),
                #('Contract', generator.version),
                ('Substation', generator.subestacion),
                ('Life Span', life_span),
                ('Energy Contracted<br>(GWh/year)', round(energy_contracted,1)),
                ('PPA<br>(USD/MWh)', round(PPA,1))
            ]
            dict_summary = OrderedDict(ordered_list)
            # Generation Summary Series
            generation_summary = pd.Series(dict_summary)
            frames.append(generation_summary)
        # Generation Assets DataFrame
        generation_assets = pd.concat(frames, axis=1).T # We transpose the index and columns
        generation_assets.index.name = 'Generation Assets'
        # Note: the indices of the dataframe are 0,1,2...
        self.Generation_Assets = generation_assets

    def set_Withdrawal_PPAs(self):
        '''Output: (dataframe) Features the overhall demand summary for each consumer.
            >>> summary(index, [Name, Company, Contract, Substation, Life Span, Energy Contracted, PPA])'''
        frames = []
        for consumer in self.Consumers:
            # Life Span
            start = consumer.start_date.strftime('%b %Y')
            end = consumer.end_date.strftime('%b %Y')
            life_span = '{start} to {end}'.format(start=start, end=end)
            # Energy Contracted
            energy_demand = consumer.get_energy_demand()
            energy_demand = energy_demand.resample('MS').sum() # Monthly sum
            scale = 1e-3 # We convert from MWh to GWh
            energy_contracted = energy_demand.mean() * scale # Monthly mean
            # PPA Ponderated
            PPA_series = consumer.get_PPA_series()
            PPA = PPA_series.multiply(energy_demand).sum() / energy_demand.sum()
            # Withdrawal PPAs Summary
            ordered_list = [
                ('Name', consumer.name),
                ('Company', consumer.empresa),
                #('Contract', consumer.version),
                ('Substation', consumer.subestacion),
                ('Life Span', life_span),
                ('Energy Contracted<br>(GWh)', round(energy_contracted,1)),
                ('PPA<br>(USD/MWh)', round(PPA,1))
            ]
            dict_summary = OrderedDict(ordered_list)
            # Demand Summary Series
            demand_summary = pd.Series(dict_summary)
            frames.append(demand_summary)
        # Withdrawal PPAs Dataframe
        withdrawal_PPAs = pd.concat(frames, axis=1).T # We transpose the index and columns
        withdrawal_PPAs.index.name = 'Withdrawal PPAs'
        # Note: the indices of the dataframe are 0,1,2...
        self.Withdrawal_PPAs = withdrawal_PPAs

    def set_Portofolio_Summary(self):
        '''Output: (series) Summary concerning the Funds.'''
        # Portofolio life
        start = self.start_date.strftime('%B %Y')
        end = self.end_date.strftime('%B %Y')
        portofolio_life = 'From {start} to {end}.'.format(start=start, end=end)
        # Initial Capital
        initial_capital = 'MUSD {}'.format(self.Funds.Capital)
        # Total Energy Managed
        total_demand = self.Total_Energy_Demand.to_frame().sum(axis=0).mean()
        total_generation = self.Total_Energy_Generation_P90_P50.to_frame().sum(axis=0).mean()
        # Portofolio Summary
        ordered_list = [
            ('Funds Name', self.Funds.name),
            ('Portofolio Life', portofolio_life),
            #('Initial Capital', initial_capital),
            #('Total Energy Generated', total_generation),
            #('Total Energy Consumed', total_demand),
            #('Total Energy Managed', total_demand + total_generation)
        ]
        dict_summary = OrderedDict(ordered_list)
        portofolio_summary = pd.Series(dict_summary, name='Portofolio Summary')
        self.Portofolio_Summary = portofolio_summary

    def set_Success_Fee_Parameters(self):
        '''Output: (series) Summary concerning the success fee parameters.'''
        dict_summary = {
            'Initial Capital (MUSD)': float(self.Funds.Capital),
            'First Threshold (%)': float(self.Funds.UmbralPerformance1),
            'Second Threshold (%)': float(self.Funds.UmbralPerformance2),
            'Success Rate First Step (%)': float(self.Funds.ComisionTramo1),
            'Success Rate Second Step (%)': float(self.Funds.ComisionTramo2),
            'VAT (%)': float(self.Funds.iva),
            'Minimum Payment Threshold (%)': float(self.Funds.UmbralInicial)}
        success_fee_parameters = pd.Series(dict_summary, name='Success Fee Parameters')
        self.Success_Fee_Parameters = success_fee_parameters

    def set_Energy_Source(self):
        '''Output: (series)'''
        energy_list = defaultdict(list)
        for generator in self.Generators:
            energy = self.Energy_Generation_P90_P50.getAll(generator.name)
            total_energy = energy.sum(axis=0) #Total of energy for each simulation
            mean_energy = total_energy.mean() #Mean of total energy of each simulation
            source = generator.tecnologia
            energy_list[source].append(mean_energy)
        dict_energy_source = {source: sum(energy) for source, energy in energy_list.items()}
        energy_source = pd.Series(dict_energy_source, name='Total Energy Generation')
        self.Energy_Source = energy_source

    def set_Shareholder_Cash_Flow_Statistics(self):
        ''' Output: (dataframe) This dataframe will be used to construct the financial table in the report.'''
        # Shareholder Cash Flow Last Date Value Statistics
        date_range = self.get_date_range('M')
        end_date = date_range.max()
        cash_flow = self.Shareholder_Cash_Flow.to_frame().cumsum(axis=0) #Type: Dataframe
        final_cash_flow = cash_flow.loc[end_date, :] #Type: Series
        final_cash_flow_statistics = get_summary_series(final_cash_flow)
        final_cash_flow_statistics.name = cash_flow.index.name
        # NPV Statistics
        NPV = self.NPV_Shareholder_Cash_Flow.to_frame(axis=0) #Type:Series
        NPV_statistics = get_summary_series(NPV)
        # IRR Statistics
        IRR = self.IRR_Shareholder_Cash_Flow.to_frame(axis=0) #Type: Series
        IRR_statistics = get_summary_series(IRR)
        # Shareholder Cash Flow Statistics
        frames = [final_cash_flow_statistics, NPV_statistics, IRR_statistics]
        shareholder_cash_flow_statistics = pd.concat(frames, axis=1)
        # Name
        date = end_date.strftime('%B %Y')
        name = 'Shareholder Cash Flow statistic analysis on {date}'.format(date=date)
        shareholder_cash_flow_statistics.index.name = name
        # Return
        self.Shareholder_Cash_Flow_Statistics = shareholder_cash_flow_statistics
        print IRR, NPV

    def set_Generation_and_Demand(self):
        ''' Output: (dataframe) Represents the overhall average demand and generation for each month in the date range.
            >>> energy(months, [demand, generation]) '''
        #Total Energy Generation
        generation = self.Total_Energy_Generation_P90_P50.to_frame().mean(axis=1) #Type: Series
        generation.name = 'Generation'
        #Total Energy Demand
        demand = self.Total_Energy_Demand.to_frame().mean(axis=1) #Type: Series
        demand.name = 'Demand'
        # Demand and Generation
        generation_and_demand = pd.concat([generation, demand], axis=1)
        generation_and_demand.index.name = 'Generation and Demand (MWh)'
        self.Generation_and_Demand = generation_and_demand

    def set_Generation_series(self):
        ''' Output: (series) Represents the monthly average generation for each generator over all the simulations. This series will be used to construct the Donnut plot on the report.'''
        dict_series = {generator.name: self.Energy_Generation_P90_P50.getAll(generator.name).mean(axis=1).mean(axis=0) for generator in self.Generators}
        generation_series = pd.Series(dict_series, name='Generation')
        self.Generation_series = generation_series

    def set_Demand_series(self):
        ''' Output: (series) Represents the monthly average demand for each consumer over all the simulations. This series will be used to construct the Donnut plot on the report.'''
        dict_series = {consumer.name: self.Energy_Demand.getAll(consumer.name).mean(axis=1).mean(axis=0) for consumer in self.Consumers}
        demand_series = pd.Series(dict_series, name='Demand')
        self.Demand_series = demand_series

    #-----------------------
    #SET EXCEL ASSESSMENT
    #-----------------------
    def set_SPOT_price_Excel_Assessment(self):
        ''' Stores the SPOT price data in an excel file. '''
        #SPOT price
        dict_df = self.SPOT_price.to_dict()
        dict_dict_df = {'SPOT price': self.SPOT_price.to_dict()}
        self.excel.store_dict_dict_df(dict_dict_df, category='Price')
        #Random Walk
        dict_df = {'Random Walk': self.Random_Walk.to_frame()}
        self.excel.store_dict_df(dict_df, category='Price')
        #Price modelling
        dict_dict_df = {'Price modelling': self.SPOT_price_modelling.to_dict()}
        self.excel.store_dict_dict_df(dict_dict_df, category='Price')
        #Price modelling (hourly and monthly only)
        dict_df_hourly_modelling = {node: self.Price.get_hourly_price_modelling(node) 
                                    for node in self.all_nodes}
        dict_df_monthly_modelling = {node: self.Price.get_monthly_price_modelling(node)
                                     for node in self.all_nodes}
        dict_dict_df = {
            'SPOT price monthly modelling': dict_df_hourly_modelling,
            'SPOT price hourly modelling': dict_df_hourly_modelling,
        }
        self.excel.store_dict_dict_df(dict_dict_df, category='Price', virgin=True)

    def set_Hydrological_scenarios_Excel_Assessment(self):
        ''' Stores the hydrological scenarios in an excel file. '''
        dict_df = {'Hydrological scenarios': self.Hydrological_scenarios.to_frame()}
        self.excel.store_dict_df(dict_df, category='Price', summary_mode=['full'])

    def set_Energy_Demand_Excel_Assessment(self):
        ''' Stores the energy demand data in an excel file. '''
        dict_dict_df = {
            'Energy Demand': self.Energy_Demand.to_dict(),
            'Purchased Energy Demand Cost': self.Purchased_Energy_Demand_Cost.to_dict(),
            'Sold Energy Demand Incomes': self.Sold_Energy_Demand_Incomes.to_dict(),
            'Sold Energy Demand Profit': self.Sold_Energy_Demand_Profit.to_dict(),
            'Purchased Energy Demand Price': self.Purchased_Energy_Demand_Price.to_dict()
        }
        self.excel.store_dict_dict_df(dict_dict_df, category='Demand')
        #Total Energy Demand
        dict_df = {'Total Energy Demand': self.Total_Energy_Demand.to_frame()}
        self.excel.store_dict_df(dict_df, category='Demand')

    def set_Energy_Generation_Excel_Assessment(self):
        ''' Stores the energy generation data in an excel file. '''
        dict_dict_df = {
            'Energy Generation': self.Energy_Generation.to_dict(),
            'Energy Generation P90': self.Energy_Generation_P90.to_dict(),
            'Energy Generation P50': self.Energy_Generation_P50.to_dict(),
            'Energy Generation P90 P50': self.Energy_Generation_P90_P50.to_dict(),
            'Purchased Energy Generation P90 Cost': self.Purchased_Energy_Generation_P90_Cost.to_dict(),
            'Purchased Energy Generation P50 Cost': self.Purchased_Energy_Generation_P50_Cost.to_dict(),
            'Purchased Energy Generation Cost': self.Purchased_Energy_Generation_Cost.to_dict(),
            'Sold Energy Generation P90 Incomes' : self.Sold_Energy_Generation_P90_Incomes.to_dict(),
            'Sold Energy Generation P50 Incomes' : self.Sold_Energy_Generation_P50_Incomes.to_dict(),
            'Sold Energy Generation Incomes': self.Sold_Energy_Generation_Incomes.to_dict(),
            'Sold Energy Generation Profit': self.Sold_Energy_Generation_Profit.to_dict(),
            'Sold Energy Generation Price': self.Sold_Energy_Generation_Price.to_dict()
        }
        self.excel.store_dict_dict_df(dict_dict_df, category='Generation')
        #Total Energy Generation
        dict_df = {'Total Energy Generation P90 P50': self.Total_Energy_Generation_P90_P50.to_frame()}
        self.excel.store_dict_df(dict_df, category='Generation')

    def set_Energy_Generation_Probabilities_Excel_Assessment(self):
        ''' Stores the energy generation probabilities data in an excel file. '''
        # Resample the energy generation probabilities from a hourly to a monthly frequency
        generation_interpolation = self.Energy_Generation_probabilities_interpolation.to_dict()
        generation_interpolation = {key: dataframe.resample('MS').sum() for key, dataframe in generation_interpolation.items()}
        # DICT OF DICT OF DF
        dict_dict_df = {
            'Energy Generation scenarios': self.Generation_scenarios.to_dict(),
            'Energy Generation Probabilities': self.Energy_Generation_probabilities.to_dict(),
            'Energy Generation Probabilities Interpolation': generation_interpolation
        }
        self.excel.store_dict_dict_df(dict_dict_df, category='Generation', virgin=True)

    def set_Financial_Excel_Assessment(self):
        ''' Stores the financial data in an excel file. '''
        dict_df = {
            'Gross Profit': self.Gross_Profit.to_frame(),
            'Operating Margin': self.Operating_Margin.to_frame(),
            'Extra Operating Margin': self.Extra_Operating_Margin.to_frame(),
            'Operating Margin Taxes': self.Operating_Margin_Taxes.to_frame(),
            'Net Incomes': self.Net_Incomes.to_frame(),
            'Net Liquid Assets': self.Net_Liquid_Assets.to_frame(),
            'Insolvency': self.Insolvency.to_frame(),
            'Dividend': self.Dividend.to_frame(),
            'Benchmark': self.Benchmark.to_frame(),
            'Success Fee': self.Success_Fee.to_frame(),
            'Shareholder Cash Flow': self.Shareholder_Cash_Flow.to_frame().cumsum(axis=0)
        }
        self.excel.store_dict_df(dict_df, category='Finance')

    def set_IRR_NPV_Excel_Assessment(self):
        ''' Stores the IRR and NPV data in an excel file. '''
        dict_series = {
            'IRR': self.IRR_Shareholder_Cash_Flow.to_frame(axis=0),
            'NPV': self.NPV_Shareholder_Cash_Flow.to_frame(axis=0)}
        #dataframe = dict_series_to_frame(dict_series)
        dataframe = pd.concat(dict_series.values(), axis=1)
        dict_df = {'NPV IRR': dataframe}
        self.excel.store_dict_df(dict_df, category='Finance')

    def set_Parameter_Tables_Excel_Assessment(self):
        ''' Stores the tables of parameters in an excel file. '''
        dict_df = {
            'Computation Parameters': self.Computation_Parameters.to_frame(),
            'Success Fee Parameters': self.Success_Fee_Parameters.to_frame(),
            'Portofolio Features': self.Portofolio_Features.to_frame(),
            'Generation Assets': self.Generation_Assets,
            'Withdrawal PPAs': self.Withdrawal_PPAs,
        }
        self.excel.store_dict_df(dict_df, category='tables', summary_mode=[''])

    def set_Summary_Tables_Excel_Assessment(self):
        ''' Stores the summary tables in an excel file. '''
        dict_df = {
            'Portofolio Summary': self.Portofolio_Summary.to_frame(),
            'Energy Source': self.Energy_Source.to_frame(),
            'Shareholder Cash Flow Statistics': self.Shareholder_Cash_Flow_Statistics,
            'Generation and Demand': self.Generation_and_Demand,
            'Generation series': self.Generation_series.to_frame(),
            'Demand series': self.Demand_series.to_frame()
        }
        self.excel.store_dict_df(dict_df, category='tables', summary_mode=[''])

    #----------------------------------
    #CREATE HTML REPORT
    #----------------------------------
    def create_Summary_html_section(self):
        row = self.report.row
        plot = self.report.plot
        table = self.report.table
        color = self.report.color
        column = self.report.column
        toolbox = self.report.toolbox

        self.report.new_page([

            # Row 1
            row.Row([

                column.Column([
                    table.textTable(self.Portofolio_Summary),
                ], className='twelve columns', header='dark'),

            ]),

            # Row 2
            row.Row([

                column.Column([
                    table.sweetyTable(self.Portofolio_Features),
                ], className='one-half column', header='light'),

                column.Column([
                    table.sweetyTable(self.Computation_Parameters),
                ], className='one-half column', header='light'),

            ]),

            # Row 3
            row.Row([

                # XLarge column
                column.Column([
                    table.sweetyTable(self.Generation_Assets, indexed=False, auto_width=False),
                    table.sweetyTable(self.Withdrawal_PPAs, indexed=False, auto_width=False),
                ], className='twelve columns', header='light'), 

            ]),

        ])

    def create_Characteristics_and_Results_html_section(self):
        row = self.report.row
        plot = self.report.plot
        table = self.report.table
        color = self.report.color
        column = self.report.column
        toolbox = self.report.toolbox
        '''
        #Energy demand
        energy_demand_series = self.Total_Energy_Demand.value.mean(axis=1)
        energy_demand_series.name = 'Demand'
        #Energy Generation
        energy_generation_series = self.Total_Energy_Generation_P90_P50.mean(axis=1)
        energy_generation_series.name = 'Generation'
        #Total Energy
        Total_Energy = pd.concat([energy_generation_series, energy_demand_series], axis=1)
        Total_Energy.index.name = 'Generation and Demand (MWh)'
        '''
        #Energy demand
        #energy_demand = sum([self.Energy_Demand.getAll(consumer.name) for consumer in self.Consumers])
        energy_demand = sum(self.Energy_Demand.to_dict().values())
        avg_energy_demand = energy_demand.mean(axis=1) # Type: Series
        avg_energy_demand.name = 'Demand'
        #Energy generation P90 P50
        #energy_generation = sum([self.Energy_Generation_P90_P50[generator.name] for generator in self.Generators])
        energy_generation = sum(self.Energy_Generation_P90_P50.to_dict().values())
        avg_energy_generation = energy_generation.mean(axis=1)
        avg_energy_generation.name = 'Generation'
        #Generation and Demand
        frames = [avg_energy_generation, avg_energy_demand]
        generation_and_demand = pd.concat(frames, axis=1)
        generation_and_demand.index.name = 'Generation and Demand (MWh)'

        #ENERGY GENERATION by generator
        dict_series = {generator.name: self.Energy_Generation_P90_P50.getAll(generator.name).mean().mean() for generator in self.Generators}
        Generation_series = pd.Series(dict_series, name='Generation')

        #ENERGY DEMAND by consumer
        dict_series = {consumer.name: self.Energy_Demand.getAll(consumer.name).mean().mean() for consumer in self.Consumers}
        Demand_series = pd.Series(dict_series, name='Demand')


        self.report.new_page([

            # Row 1
            row.Row([

                # Column 1
                column.Column([
                    table.sweetyTable(self.Shareholder_Cash_Flow_Statistics)
                ], className='one-third column', header='dark'),

                # Column 2
                column.Column([
                    plot.simpleScatter(generation_and_demand, category=''),
                    plot.yummyDonut(Generation_series, category='Generation', title=None, color=color.antuko['blue1']),
                    plot.yummyDonut(Demand_series, category='Demand', title=None, color=color.antuko['blue2'])
                ], className='two-thirds column', header='light'),

            ])

        ])

    def create_Net_Liquid_Assets_and_Dividend_html_section(self):
        row = self.report.row
        plot = self.report.plot
        table = self.report.table
        color = self.report.color
        column = self.report.column
        toolbox = self.report.toolbox

        self.report.new_page([

            # Row 1
            row.Row([

                # Column 1

                column.Column([
                    plot.visualStatistics(self.Net_Liquid_Assets.to_frame(), category='Finance'),
                ], className='two-thirds column', header='dark'),

                # Column 2
                column.Column([
                    table.sweetyTable(
                        toolbox.last_date_summary_series(self.Net_Liquid_Assets.to_frame(), numFormat='currency')
                    )
                ], className='one-third column', header='light'),

            ]),

            # Row 2

            row.Row([

                # Column 1
                column.Column([
                    plot.visualStatistics(self.Dividend.to_frame().cumsum(axis=0), category='Finance'),
                ], className='two-thirds column', header='dark'),

                # Column 2

                column.Column([
                    table.sweetyTable(
                        toolbox.last_date_summary_series(self.Dividend.to_frame().cumsum(axis=0), numFormat='currency')
                    )
                ], className='one-third column', header='light'),

            ]),

        ])

    def create_Profit_Analysis_html_section(self):
        row = self.report.row
        plot = self.report.plot
        table = self.report.table
        color = self.report.color
        column = self.report.column
        toolbox = self.report.toolbox

        self.report.new_page([

            # Row 1
            row.Row([

                # Column 1
                column.Column([
                    table.sweetyTable(
                        toolbox.monthly_yearly_summary_df(self.Gross_Profit.to_frame(), method='sum', numFormat='currency')
                    )
                ], className='one-third column', header='dark'),

                # Column 2
                column.Column([
                    plot.visualStatistics(self.Gross_Profit.to_frame(), category='Finance'),
                ], className='two-thirds column', header='light'),

            ]),

            # Row 2
            row.Row([

                # Column 1
                column.Column([

                    table.sweetyTable(
                        toolbox.monthly_yearly_summary_df(self.Net_Incomes.to_frame(), method='sum', numFormat='currency')
                    )

                ], className='one-third column', header='dark'),

                # Column 2
                column.Column([
                    plot.visualStatistics(self.Net_Incomes.to_frame(), category='Finance'),
                ], className='two-thirds column', header='light'),

            ]),

        ])

    def create_Shareholder_Cashflow_html_section(self):
        row = self.report.row
        plot = self.report.plot
        table = self.report.table
        color = self.report.color
        column = self.report.column
        toolbox = self.report.toolbox

        self.report.new_page([

            # Row 1
            row.Row([

                # Column 1
                column.Column([
                    plot.visualStatistics(self.Shareholder_Cash_Flow.to_frame().cumsum(axis=0), category='Finance'),
                ], className='two-thirds column', header='light'),

                # Column 2
                column.Column([
                    table.sweetyTable(self.Shareholder_Cash_Flow_Statistics)
                ], className='one-third column', header='dark'),

            ]),

        ])

    def create_Energy_Generation_P90_P50_html_section(self):
        row = self.report.row
        plot = self.report.plot
        table = self.report.table
        color = self.report.color
        column = self.report.column
        toolbox = self.report.toolbox

        Energy_Generation_P90_P50_series = toolbox.parse_dict_to_series(self.Energy_Generation_P90_P50.to_dict())

        for list_Energy_Generation_P90_P50 in Energy_Generation_P90_P50_series:

            self.report.new_page([


                # Row 1
                row.Row([

                    # Column 1
                    column.Column([
                        table.sweetyTable(
                            toolbox.monthly_yearly_summary_df(energy_generation, method='sum', numFormat='energy')
                        )
                    ], className='one-third column', header='dark'),

                    # Column 2
                    column.Column([
                        plot.visualStatistics(energy_generation, category='Generation'),
                    ], className='two-thirds column', header='light'),

                ]) for energy_generation in list_Energy_Generation_P90_P50

            ])

    def create_Energy_Demand_html_section(self):
        row = self.report.row
        plot = self.report.plot
        table = self.report.table
        color = self.report.color
        column = self.report.column
        toolbox = self.report.toolbox

        Energy_Demand_series = toolbox.parse_dict_to_series(self.Energy_Demand.to_dict())
        for list_Energy_Demand in Energy_Demand_series:

            self.report.new_page([


                # Row 1

                row.Row([

                    # Column 1
                    column.Column([
                        table.sweetyTable(
                            toolbox.monthly_yearly_summary_df(energy_demand, method='sum', numFormat='energy')
                        )
                    ], className='one-third column', header='dark'),

                    # Column 2
                    column.Column([
                        plot.visualStatistics(energy_demand, category='Demand'),
                    ], className='two-thirds column', header='light'),

                ]) for energy_demand in list_Energy_Demand

            ])

    def create_SPOT_price_html_section(self):
        row = self.report.row
        plot = self.report.plot
        table = self.report.table
        color = self.report.color
        column = self.report.column
        toolbox = self.report.toolbox

        SPOT_price_series = toolbox.parse_dict_to_series(self.SPOT_price.to_dict())
        for list_SPOT_price in SPOT_price_series:

            self.report.new_page([


                # Row 1

                row.Row([

                    # Column 1
                    column.Column([
                        table.sweetyTable(
                            toolbox.monthly_yearly_summary_df(spot_price, method='mean', numFormat='currency')
                        )
                    ], className='one-third column', header='dark'),

                    # Column 2
                    column.Column([
                        plot.visualStatistics(spot_price, category='Price'),
                    ], className='two-thirds column', header='light'),

                ]) for spot_price in list_SPOT_price

            ])


    def compile_HTML_Report(self):
        self.report.create()

#--------------------------------------------------------------------#
# USEFUL FUNCTIONS FOR MULTIPROCESSING
#--------------------------------------------------------------------#

#SPOT PRICE
def set_SPOT_price_hourly_modelling_series_helper(index):
    return set_SPOT_price_hourly_modelling_series(*index)

def set_SPOT_price_hourly_modelling_series(hourly_ratio, date_range):
    hourly_modelling_series = pd.Series(hourly_ratio, index=date_range)
    return hourly_modelling_series

def set_SPOT_price_series_helper(index):
    return set_SPOT_price_series(*index)

def set_SPOT_price_series(hydro_year_markers, simulation, spot_price_modelling, scenarios):
    frames = list()
    for start, end in hydro_year_markers:
        scenario = scenarios[start]
        series = spot_price_modelling.loc[start:end, scenario]
        frames.append(series)
    price_series = pd.concat(frames, axis=0)
    price_series = price_series.sort_index()
    price_series.name = simulation
    return price_series

#TEST
def set_SPOT_price_series_helper_2(index):
    return set_SPOT_price_series_2(*index)

def set_SPOT_price_series_2(scenarios, hydro_year_markers, spot_price_modelling, multiprocessing=True):
    spot_price = pd.DataFrame(dtype=float)
    simulations = scenarios.columns
    indices = [(scenarios[simulation], hydro_year_markers, spot_price_modelling) for simulation in simulations]
    if multiprocessing:
        n_cpu = mp.cpu_count()
        pool = mp.Pool(processes=n_cpu)
        frames = pool.map(set_SPOT_price_series_mp_helper, indices)
        pool.close()
        pool.join()
    else:
        frames = [set_SPOT_price_series_mp_helper(index) for index in indices]
    spot_price = pd.concat(frames, axis=1)
    spot_price = spot_price.sort_index(axis=1)
    return spot_price

def set_SPOT_price_series_mp_helper(index):
    return set_SPOT_price_series_mp(*index)

def set_SPOT_price_series_mp(scenarios, hydro_year_markers, spot_price_modelling):
    print('>>> set_SPOT_price_series_mp <<<')
    simulation = scenarios.name
    price_series = pd.Series(name=simulation)
    for start, end in hydro_year_markers:
        scenario = scenarios.at[start]
        series = spot_price_modelling.loc[start:end, scenario]
        price_series = pd.concat([price_series, series], axis=0)
    price_series = price_series.sort_index()
    print('<<< set_SPOT_price_series_mp >>>')
    return price_series


#ENERGY GENERATION
def set_energy_generation_series_helper(index):
    return set_energy_generation_series(*index)

def set_energy_generation_series(month_markers, scenarios, generation_probabilities):
    ''' Input:
        - scenario: (series) of the form probabilities(hours), where probabilities = [P25,..,P99]
        - generation_probabilities: (df) of the dorm generation(hours, probailities)

        Output: (series) of the form generation(hours)
    '''
    simulation = scenarios.name #scenarios is a Series
    generation_series = pd.Series()
    for month_start, month_end in month_markers:
        scenario = scenarios[month_start]
        series = generation_probabilities.loc[month_start:month_end, scenario]
        generation_series = pd.concat([generation_series, series], axis=0)
    generation_series.name = simulation
    return generation_series.sort_index()

#--------------------------------------------------------------------#
# MAGIC TOOLBOX
#--------------------------------------------------------------------#

def set_simulation_series(n_simulation):
    ''' Output: (dict of lists) divides the total number of simulations in various series to better compute the multiprocessing '''
    n_cpu = mp.cpu_count()
    n_calcultation_per_cpu = 10
    n_simulation_per_series = n_cpu * n_calcultation_per_cpu
    simulation_series = defaultdict(list)
    count = 0
    n_series = 1
    for simulation in range(n_simulation):
        if not count < n_simulation_per_series:
            n_series += 1
            count = 0
        simulation_series[n_series].append(simulation)
        count += 1
    return dict(simulation_series)

def validstr(str):
    #Make sure all invalid arguments are excluded
    invalid_excel_args = ['\\', '/', '#', '*', '?', '!', '[', ']']
    validstr = ''.join(e for e in str if not e in invalid_excel_args)
    #Make sure string length stay smaller than the valid maximum length
    max_length = 31
    validstr = validstr[:max_length]
    return validstr

def describe_dataframe(df, mode, drop_list=['count']):
    if mode == 'Summary':
        date_range = df.index
        frames = list()
        for date in date_range:
            s = df.loc[date].describe()
            frames.append(s.to_frame().T)
        dataframe = pd.concat(frames, axis=0)
        dataframe = dataframe.drop(drop_list, axis=1)
    elif mode == 'Full':
        dataframe = df
    return dataframe

def describe_series(s, mode, drop_list=[]):
    if mode == 'Summary':
        series = s.describe()
        series = series.drop(drop_list)
    elif mode == 'Full':
        series = s
    return series


#------------------------------------
# TO IMPLEMENT IN A TOOLBOX CLASS
#------------------------------------
def get_summary_df(dataframe, percentiles=[.25,.5,.75], dropList=[]):
    '''Output: (dataframe) Returns the total count, the average, the standard deviation, the min, 25%, 50%, 75% and the max, for each date in the date range.
        >>> value(date, [count, mean, std, min, 25%, 50%, 75%, max]) 
    '''
    name = dataframe.index.name
    frames = list()
    indices = dataframe.index
    for index in indices:
        s = dataframe.loc[index].describe(percentiles)
        frames.append(s.to_frame().T)
    df = pd.concat(frames, axis=0)
    df = df.drop(dropList, axis=1)
    #df = dataframe.describe(percentiles).drop(dropList)
    df.index.name = name
    return df

def get_summary_series(series, percentiles=[.25,.5,.75], dropList=[]):
    '''Output: (series) Returns the total count, the average, the standard deviation, the min, 25%, 50%, 75% and the max.
        >>> value([count, mean, std, min, 25%, 50%, 75%, max])
    '''
    if isinstance(series, pd.DataFrame):
        df = series
        series = df[df.columns[0]]
        series.name = df.index.name
    s = series.describe(percentiles).drop(dropList)
    s.name = series.name
    return s

def dict_series_to_frame(dict_series, name=''):
    ''' Output: (dataframe) Concatenate series horizontally and return a dataframe.
    '''
    frames = dict_series.values()
    df = pd.concat(frames, axis=1)
    df.index.name = name
    return df


#--------------------------------------------------------------------#
# MODELLING FUNCTIONS
#--------------------------------------------------------------------#
def Random_Walk(volatility, OilAdjust, date_range, simulations):
    n_simulation = len(simulations)
    if OilAdjust:
        sigma = volatility/100.
        mu = -sigma**2/2.
        size = (len(date_range), n_simulation)
        normal_data = np.random.normal(mu, sigma, size=size)
        data = np.cumsum(normal_data, axis=0)
        Random_Walk_log = pd.DataFrame(data, index=date_range, columns=simulations)
        Random_Walk = np.exp(Random_Walk_log)
    else:
        shape = (len(date_range), n_simulation)
        data = np.ones(shape=shape)
        Random_Walk = pd.DataFrame(data, index=date_range, columns=simulations)
    return Random_Walk

def trapezoidal_distribution(a, b, c, d, date_range, simulations):
    n_simulation = len(simulations)
    triangle1 = (b - a)/2.
    triangle2 = (d - c)/2.
    square = c - b
    total_aera = triangle1 + square + triangle2
    if total_aera:
        len1 = int(n_simulation * triangle1/total_aera)
        len2 = int(n_simulation * triangle2/total_aera)
        len3 = n_simulation - len1 - len2
        #Important: We work on transposed matrix because otherwise, the distribution of the variation is not well distributed along the date range axis
        distri1 = np.random.triangular(left=a, mode=b, right=b, size=(len1, len(date_range)))
        distri2 = np.random.triangular(left=c, mode=c, right=d, size=(len2, len(date_range)))
        distri3 = np.random.uniform(low=b, high=c, size=(len3, len(date_range)))
        distri = np.concatenate((distri1, distri2, distri3))
        distri = np.random.permutation(distri)
        #Important: We transpose the distribution matrix
        distri = np.transpose(distri)
        df = pd.DataFrame(distri, index=date_range, columns=simulations)
    else:
        df = pd.DataFrame(0, index=date_range, columns=simulations)
    df = df / 100.
    return df

