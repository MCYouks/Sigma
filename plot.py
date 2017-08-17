# -*- coding: utf-8 -*-
'''

The objective is to build all the plotting functions to display graphs and figures.

@author: Andréas

@date: Monday, August the 14th
'''

import plotly
import plotly.plotly as py
import plotly.offline as pyol
import plotly.graph_objs as pygo
import plotly.figure_factory as pyff
pyol.init_notebook_mode()

import pandas as pd
import toolbox
import os

import html

from path import MyPath2

# HEX code of Antuko colors
antuko_colors = {
    'grey1': '#CCCCCC', #rgb(204, 204, 204)
    'grey2': '#A5A7AB', #rgb(165, 167, 171)
    'blue1': '#4CC7F1', #rgb(76, 199, 241)
    'blue2': '#1A75BB' #rgb(26, 117, 187) 
}

def Filename(path, name):
    ''' Returns the filename of the figure. '''
    return os.path.join(path, name+'.html')

def Url(fig, filename):
    ''' Returns the url of the plot. '''
    url = pyol.plot(fig, filename=filename, auto_open=False, show_link=False)
    return url

def Name(df_or_series):
    ''' Returns the name of the dataframe's index. '''
    def get_name(df_or_series):
        if isinstance(df_or_series, pd.DataFrame):
            return df_or_series.index.name
        elif isinstance(df_or_series, pd.Series):
            return df_or_series.name

    name = get_name(df_or_series)
    if name:
        return name
    else:
        raise AttributeError('Dataframe has no index name. Please give it a name before plotting the figure.')

#-------------------------------
# FIGURES
#-------------------------------
def visualStats(dataframe):
    '''
    Returns a figure that display the time evolution of the data percentiles.
    '''
    df = toolbox.describe_df(dataframe, dropList=['count', 'std'])

    color = {'min': antuko_colors['grey1'],
             '25%': antuko_colors['grey2'],
             '50%': antuko_colors['blue1'],
             'mean': antuko_colors['blue2'],
             '75%': antuko_colors['grey2'],
             'max': antuko_colors['grey1']}

    showlegend = {'min': False,
                  '25%': False,
                  '50%': True,
                  'mean': True,
                  '75%': False,
                  'max': False}

    name = {'min': 'Min',
            '25%': '25%',
            '50%': 'Median',
            'mean': 'Mean',
            '75%': '75%',
            'max': 'Max'}

    mode = {col: 'lines' for col in df.columns}

    width = {'min': 0,
            '25%': 0,
            '50%': 3.5,
            'mean': 3.5,
            '75%': 0,
            'max': 0}

    line = {col: pygo.Line(width=width[col], color=color[col]) for col in df.columns}

    fill = {'min': 'none',
            '25%': 'none',
            '50%': 'none',
            'mean': 'none',
            '75%': 'tonexty',
            'max': 'tonexty'}
    # DATA
    trace = {col: pygo.Scatter(x=df.index, y=df[col], showlegend=showlegend[col], name=name[col], mode=mode[col], line=line[col], fill=fill[col]) for col in df.columns}
    data = [trace['min'], trace['max'], trace['25%'], trace['75%'], trace['50%'], trace['mean']]
    
    # LAYOUT
    xaxis = pygo.XAxis(tickformat='%b %Y')
    yaxis = pygo.YAxis(ticksuffix='')
    margin = dict(l=34, r=0, t=3, b=30, pad=0)
    legend = dict(orientation='v', xanchor='right') #bgcolor='transparent' for transparent background
    layout = pygo.Layout(xaxis=xaxis, yaxis=yaxis, autosize=True, margin=margin, legend=legend, showlegend=False, hovermode='closest')
    # Figure
    fig = pygo.Figure(data=data, layout=layout)
    return fig


def simpleScatter(self, df, colors=[]):
    '''
    '''
    if not colors:
        start_color = antuko_colors['blue1']
        finish_color = antuko_colors['blue2']
        colors = toolbox.hex_linear_gradient(start_color, finish_color, len(df.columns))

    # DATA
    data = [pygo.Scatter(x=df.index, y=df[col], name=df[col].name, mode='lines', line=pygo.Line(width=2.5, color=colors[i])) for i, col in enumerate(df.columns)]
    
    # LAYOUT
    xaxis = pygo.XAxis()
    yaxis = pygo.YAxis(ticksuffix='')
    margin = dict(l=34, r=0, t=3, b=30)
    legend = dict(orientation='v', xanchor='right') #bgcolor='transparent' for transparent background
    layout = pygo.Layout(xaxis=xaxis, yaxis=yaxis, autosize=True, margin=margin, legend=legend, showlegend=False)
    
    # LAYOUT
    fig = pygo.Figure(data=data, layout=layout)
    return fig


def energyDonut(series, color='#CCCCCC'):
    '''
    Returns a donnut figure for energy values given in input.
    '''
    #If the series is null then we replace it by a tiny value, otherwise the donut will not display
    if sum(series.values) == 0:
        series += 1e-6
    series = series.sort_values(ascending=False)
    labels = series.index
    values = series.values
    total_energy = sum(values)
    #colors = [self.color.antuko['blue2'], self.color.antuko['blue1'], self.color.antuko['grey2'], self.color.antuko['grey1'],]
    colors = toolbox.hex_linear_gradient(color, '#FFFFFF', n=len(labels)+1)

    # DATA
    trace = pygo.Pie(labels=labels, values=values,
                   hoverinfo='percent+value+label', 
                   textinfo='percent', 
                   textfont=dict(size=10),
                   textposition='outside',
                   marker=dict(colors=colors),
                   hole=.85)
    data = [trace]
    
    # LAYOUT
    number, unit = toolbox.get_energyFormat(total_energy)
    text = '<b>{number}</b><br>{unit}/mth'.format(number=number, unit=unit)
    annotations = [dict(text=text, showarrow=False, font=dict(size=15, color=color))]
    margin = dict(l=0, r=0, t=25, b=0)
    legend = dict(orientation='h', bgcolor='transparent')
    titlefont = dict(color=color)
    layout = pygo.Layout(title=series.name, titlefont=titlefont, annotations=annotations, margin=margin, legend=legend, showlegend=False)
    
    # FIGURE
    fig = pygo.Figure(data=data, layout=layout)
    return fig


class plot(object):
    """
    Defines all the plotting we can do
    """
    def __init__(self, funds_name='', localtime=''):
        ''' The class uses the path class. '''
        self.path = MyPath2(funds_name, localtime)

    def iframe(self, url, **param):
        ''' Sets the iframe style to display a plot.
            Otherwise, it hardly works through css styling. '''
        param.update(dict(width='100%'))
        return html.Iframe(src=url, seamless='seamless', style=dict(border='0'), 
                        frameBoder='0', **param)

    def visualStats(self, dataframe, category='', **param):
        '''
        Returns the html iframe of the statistical figure plot.
        '''
        fig = visualStats(dataframe)
        name = Name(dataframe)
        category_path = self.path._set_figure_category_path(category)
        filename = Filename(category_path, name)
        url = Url(fig, filename)
        iframe = self.iframe(url, **param)
        return iframe

    def simpleScatter(self, dataframe, category='', colors=[], **param):
        '''
        Returns the html iframe of the simple scatter figure plot.
        '''
        fig = simpleScatter(dataframe, colors)
        name = Name(dataframe)
        category_path = self.path._set_figure_category_path(category)
        filename = Filename(category_path, name)
        url = Url(fig, filename)
        frame = self.iframe(url, **param)
        return frame

    def energyDonut(self, series, category='', color='#CCCCCC', **param):
        '''
         Returns the html iframe of the energy donnut figure plot..
        '''
        fig = energyDonut(series, color)
        name = Name(series)
        category_path = self.path._set_figure_category_path(category)
        filename = Filename(category_path, name)
        url = Url(fig, filename)
        iframe = self.iframe(url, **param)
        return iframe


'''
==========================================================
'''
if __name__ == '__main__':

    from report import report
    import numpy as np
    from html import *

    data = np.random.randn(20, 100)
    date_range = pd.date_range('2017-01', freq='MS', periods=20)
    dataframe = pd.DataFrame(data, index=date_range).cumsum(axis=0).apply(np.log)
    dataframe.index.name = 'my dataframe'

    Plot = plot()
    Report = report()

    content = Div([

        H6('Hola la compagnie créole!', className='gs-header gs-section-header padded'),

        Plot.visualStats(dataframe, height='250')

    ], className='row')

    Report.New_Page(content)

    Report.Build()
