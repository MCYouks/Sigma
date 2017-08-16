# -*- coding: utf-8 -*-
'''

The objective is to build all the functions to create html tables and charts.

@author: Andréas

@date: Monday, August the 14th
'''
from bs4 import BeautifulSoup

from report import report

import pandas as pd

import html

def dashTable(dataframe, indexed=True, auto_width=True, headed=True):
    '''
    Sets an html table from a dataframe.
    '''
    def set_indexed_df():
        if indexed:
            return dataframe
        else:
            return dataframe.set_index([dataframe.columns[0]])
    def set_width(df):
        if auto_width:
            return ''
        else:
            return '{}%'.format(100. / (len(df.columns) + 1))
    def set_name(df):
        if df.index.name:
            return df.index.name
        else:
            return ''
    def clean_content(table):
        content = html.set_content(table)
        content.replace('width=""', '')
        return content
    def set_table_body(df, width):
        table = ''
        for index, row in df.iterrows():
            cells = [html.Td(row[i], style={'text-align':'right'}, width=width) 
                        for i in range(len(row))]
            # Left index insertion
            cells.insert(0, html.Td(index, width=width))
            table += html.Tr(cells)
        return table
    def set_table_head(df, width):
        html_cells = [html.Td(col, style={'text-align':'right', 'valign':'top'}, width=width) 
                      for col in df.columns]
        # Top index column space insertion
        if indexed:
            html_cells.insert(0, html.Td('', width=width))
        else:
            name = set_name(df)
            html_cells.insert(0, html.Td(name, width=width))
        html_row = html.Tr(html_cells, style={'background':'white', 'font-weight':'bold'})
        return html_row
    def set_body():
        df = set_indexed_df()
        width = set_width(df)
        table = set_table_body(df, width)
        return table
    def set_head():
        if headed:
            df = set_indexed_df()
            width = set_width(df)
            head = set_table_head(df, width)
            return head
        else:
            return ''

    head = set_head()
    body = set_body()
    table = head + body
    content = clean_content(table)
    table = html.Table(content)
    return table

def seriesTable(series):
    '''
    Sets an html table from a series.
    '''
    dataframe = series.to_frame()
    table = dashTable(dataframe, headed=False)
    return table


if __name__ == '__main__':
    
    index = ['ourson', 'chaton', 'croco']
    data = {
        'mange' : ['poisson', 'pate', 'chevre'],
        'boit': ['eau', 'lait', 'eau'],
        'couleur': ['marron', 'blanc', 'vert']
    }
    df = pd.DataFrame(data, index=index)

    d = {key: ''.join(value) for key, value in data.items()}
    s = pd.Series(d)

    table = seriesTable(s)

    content = html.Div(table, className='row')

    Report = report()
    Report.New_Page(content)

    Report.Build()

