# -*- coding: utf-8 -*-
"""
Created on July 1st, 2017

Useful Classes for Arkenstone's Reporting

@author: Andréas
"""

import plotly
import plotly.plotly as py
import plotly.offline as pyol
import plotly.graph_objs as pygo
import plotly.figure_factory as pyff
pyol.init_notebook_mode()

from bs4 import BeautifulSoup

from collections import defaultdict, OrderedDict

import pandas as pd
import webbrowser
import os

import math

from path import MyPath2

from debug import debug, decorate_classmethods
debugOption = True

class MyHTML:
    def _set_html_style(self, style):
        return '; '.join([('{}:{}'.format(key, value)) for key, value in style.items()])

    def _set_html_content(self, content_or_list):
        if isinstance(content_or_list, str) or isinstance(content_or_list, unicode):
            content = self.get_validstr(content_or_list)
        elif isinstance(content_or_list, list):
            list_of_content = content_or_list
            #content = '\n'.join([str(element).encode('utf-8') for element in list_of_content])
            content = '\n'.join([self.get_validstr(element) for element in list_of_content])
        else:
            print 'Content does not fit with the format requiered !'
            print type(content_or_list), content_or_list
            content = '***Error content!***'
        return content

    def get_validstr(self, content):
        if isinstance(content, str):
            validstr = content
        elif isinstance(content, unicode):
            validstr = content.encode('utf-8')
        else:
            try:
                validstr = str(content)
            except:
                print('***Error: Fail to convert content into a valid string.')
                print type(content), content
                validstr = '***Error: invalid string!***'
        return validstr

    def _set_html_content_for_page_row_or_column(self, content_or_list):
        if isinstance(content_or_list, str):
            content = content_or_list
        elif isinstance(content_or_list, list):
            list_of_content = content_or_list
            content = []
            for element in list_of_content:
                if isinstance(element, list):
                    for subelement in element:
                        content.append(subelement)
                elif isinstance(element, str):    
                    content.append(element)
        return content

    def _clean_html_tag(self, html_str):
        to_remove = [' class=""', ' style=""', ' seamless=""',
                     ' width=""', ' height=""', ' colspan=""',
                     ' align=""', ' background=""']
        for str in to_remove:
            html_str = html_str.replace(str, '')
        return html_str

    def Div(self, content=[], className='', style={}, background=''):
        content = self._set_html_content(content)
        style = self._set_html_style(style)
        html_div = '''
        <div class="{className}" style="{style}" background="{background}">
            {content}
        </div>
        '''.format(content=content, className=className, style=style, background=background)
        return html_div

    def A(self, content=[], className='', href='', style={}):
        text = self._set_html_content(content)
        style = self._set_html_style(style)
        html_a = '''
        <a class="{className}" style="{style}" href="{href}">{text}</a>
        '''.format(text=text, className=className, style=style, href=href)
        return html_a

    def H1(self, content=[], className='', style={}):
        text = self._set_html_content(content)
        style = self._set_html_style(style)
        html_h1 = '''
        <h1 class="{className}" style="{style}">{text}</h1>
        '''.format(text=text, className=className, style=style)
        return html_h1

    def H5(self, content=[], className='', style={}):
        text = self._set_html_content(content)
        style = self._set_html_style(style)
        html_h5 = '''
        <h5 class="{className}" style="{style}">{text}</h5>
        '''.format(text=text, className=className, style=style)
        return html_h5

    def H6(self, content=[], className='', style={}):
        text = self._set_html_content(content)
        style = self._set_html_style(style)
        html_h6 = '''
        <h6 class="{className}" style="{style}">{text}</h6>
        '''.format(text=text, className=className, style=style)
        return html_h6

    def Br(self, content=[]):
        text = self._set_html_content(content)
        html_br = '''<br>{text}</br>'''.format(text=text)
        return html_br

    def Strong(self, text='', className='', style={}):
        style = self._set_html_style(style)
        html_strong = '''
        <strong class="{className}" style="{style}">
            {text}
        </strong>
        '''.format(text=text, className=className, style=style)
        return html_strong

    def P(self, text, className='', style={}):
        style = self._set_html_style(style)
        text = self.get_validstr(text)
        html_p = '''
        <p class="{className}" style="{style}">
            {text}
        </p>
        '''.format(text=text, className=className, style=style)
        return html_p

    def Span(self, text='', className='', style={}):
        style = self._set_html_style(style)
        html_span = '''
        <span class="{className}" style="{style}">{text}</span>
        '''.format(text=text, className=className, style=style)
        return html_span

    def Iframe(self, src, seamless='', style={}, width='', height=''):
        style = self._set_html_style(style)
        src = self.get_validstr(src)
        html_iframe = '''
        <iframe src="{src}" seamless="{seamless}" style="{style}" width="{width}" height="{height}"></iframe>
        '''.format(src=src, seamless=seamless, style=style, width=width, height=height)
        return html_iframe

    def Image(self, src="{src}", width='', height=''):
        src = self.get_validstr(src)
        html_image = '''
        <image src="{src}" width="{width}" height="{height}"></image>
        '''.format(src=src, width=width, height=height)
        return html_image

    def Td(self, content=[], className='', style={}, colspan='', width=''):
        cell = self._set_html_content(content)
        style = self._set_html_style(style)
        html_td = '''
        <td class="{className}" style="{style}" colspan="{colspan}" width='{width}'>{cell}</td>
        '''.format(cell=cell, className=className, style=style, colspan=colspan, width=width)
        return html_td

    def Tr(self, content=[], style={}):
        row = self._set_html_content(content)
        style = self._set_html_style(style)
        html_tr = '''
        <tr style="{style}">
            {row}
        </tr>
        '''.format(row=row, style=style)
        return  html_tr

    def Table(self, content=[], className='', style={}):
        rows = self._set_html_content(content)
        style = self._set_html_style(style)
        html_table = '''
        <table class="{className}" style="{style}">
            {rows}
        </table>
        '''.format(rows=rows, className=className, style=style)
        return html_table

    def Script(self, src):
        html_script = '''<script type="text/JavaScript" src="{src}"></script>'''.format(src=src)
        return html_script

    def Css(self, link):
        html_css = '''<link rel="stylesheet" href="{link}"></link>'''.format(link=link)
        return html_css

    def HTML(self, title='', layout='', css='', scripts=''):
        html_html = '''
        <html>
            <head>
                <meta charset="UTF-8"/>
                <title>{title}</title>
                {css}
            </head>
            <body>
                {layout}
            </body>
            <footer>
                {scripts}
            </footer>
        </html>
        '''.format(title=title, layout=layout, css=css, scripts=scripts)
        soup = BeautifulSoup(html_html, 'html.parser')
        html_html = soup.prettify()
        return self._clean_html_tag(html_html)


class MyToolBox:
    """Defines each manipulation we do with dataframes or series."""
    def __init__(self):
        pass

    def _descriptive_pandas(self, df_or_series, dropList=[], percentiles=[.25,.5,.75]):
        '''Output: (dataframe or series) Returns the total count, the average, the standard deviation, the min, 25%, 50%, 75% and the max.
            >>> if input == dataframe:
            value(date, [count, mean, std, min, 25%, 50%, 75%, max])
            >>> if input == series:
            value([count, mean, std, min, 25%, 50%, 75%, max])'''
        if isinstance(df_or_series, pd.DataFrame):
            df = df_or_series
            return self.summary_df(df, dropList, percentiles)
        elif isinstance(df_or_series, pd.Series):
            series = df_or_series
            return self.summary_series(series, dropList, percentiles)


    def summary_df(self, dataframe, dropList=[], percentiles=[.25,.5,.75]):
        '''Output: (dataframe) Returns the total count, the average, the standard deviation, the min, 25%, 50%, 75% and the max, for each date in the date range.
            >>> value(date, [count, mean, std, min, 25%, 50%, 75%, max])'''
        frames = list()
        name = dataframe.index.name
        indices = dataframe.index
        for index in indices:
            s = dataframe.loc[index].describe(percentiles)
            frames.append(s.to_frame().T)
        df = pd.concat(frames, axis=0)
        df = df.drop(dropList, axis=1)
        df.index.name = name
        return df

    def monthly_yearly_summary_df(self, dataframe, method='sum', numFormat=''):
        ''' Output: (dataframe) Return a montly and yearly descriptive table.
            >>> summary([monhly, yearly], [count, mean, std, min, 25%, 50%, 75%, max])'''
        #Series
        m_series = self.resample_pandas(dataframe, freq='MS', method=method) #Monthly average for each simulation
        y_series = self.resample_pandas(dataframe, freq='A', method=method) #Yearly average for each simulation
        #Monthly summary series
        m_summary_series = self.summary_series(m_series, dropList=['count'], percentiles=[.25,.5,.75])
        m_summary_series.name = 'Monthly'
        #Yearly summary series
        y_summary_series = self.summary_series(y_series, dropList=['count'], percentiles=[.25,.5,.75])
        y_summary_series.name = 'Yearly'
        #Format
        dict_format = {'energy': self.energyFormat, 'currency': self.currencyFormat, '%': self.percFormat}
        if numFormat:
            f = dict_format[numFormat]
            m_summary_series = m_summary_series.apply(lambda x: f(x))
            y_summary_series = y_summary_series.apply(lambda x: f(x))
        #Summary df
        summary_df = pd.concat([m_summary_series, y_summary_series], axis=1)
        name = 'Summary'
        summary_df.index.name = name
        #Return
        return summary_df

    def last_date_summary_series(self, dataframe, numFormat=''):
        ''' Output: (series) Returns a statistical summary on the last date of the date range.
            >>> value([mean, std, min, 25%, 50%, 75%, max]) '''
        #Summary series
        date_range = dataframe.index
        end_date = date_range.max()
        series = dataframe.loc[end_date, :]
        summary_series = self.summary_series(series, dropList=['count'], percentiles=[.25,.5,.75])
        #Format
        dict_format = {'energy': self.energyFormat, 'currency': self.currencyFormat, '%': self.percFormat}
        if numFormat:
            f = dict_format[numFormat]
            summary_series = summary_series.apply(lambda x: f(x))
        #Name
        date = end_date.strftime('%B %Y')
        name = 'Summary on {date}'.format(date=date)
        summary_series.name = name
        #Return
        return summary_series

    def summary_series(self, series, dropList=[], percentiles=[.25,.5,.75]):
        '''Output: (series) Returns the total count, the average, the standard deviation, the min, 25%, 50%, 75% and the max.
            >>> value([count, mean, std, min, 25%, 50%, 75%, max])'''
        return series.describe(percentiles).drop(dropList)

    def resample_pandas(self, df_or_series, freq='MS', method='sum'):
        ''' Output: (series) Returns a monthly and yearly sampled df or series.'''
        if method == 'sum':
            resampled_pandas = df_or_series.resample(freq).sum().mean(axis=0)
        elif method == 'mean':
            resampled_pandas = df_or_series.resample(freq).mean().mean(axis=0)
        return resampled_pandas

    #---------------------------------
    #FORMATING NUMBER AND CURRENCIES
    #---------------------------------
    def percFormat(self, number, decimal=0):
        if not math.isnan(number) or math.isinf(number):
            return number
        else:
            return '{0:,.{1}f}%'.format(number, decimal)

    def currencyFormat(self, number, currency='$', decimal=1):
        if math.isnan(number) or math.isinf(number):
            return number
        else:
            number, currency = self.get_currencyFormat(number, currency, decimal)
            currency = currency + ' ' # We insert a space between the currency and the number
            return self.numFormat(number, currency=currency, decimal=decimal)

    def get_currencyFormat(self, number, currency='$', decimal=1):
        number, power = self.millify(number)
        number = round(number, decimal)
        if currency == 'USD':
            currency = power + currency # We insert the power before the 'USD' and a space before the number
        elif currency == '$':
            currency = currency + power # We insert the power after the '$' sign
        return (number, currency)

    def energyFormat(self, number, decimal=1):
        if math.isnan(number) or math.isinf(number):
            return number
        else:
            number, unit = self.get_energyFormat(number, decimal)
            unit = ' ' + unit # We insert a space between the number and the units
            return self.numFormat(number, unit=unit, decimal=decimal)

    def get_energyFormat(self, number, decimal=1):
        ''' Output: (tuple) Returns the adjusted number and the power.
            >>> In : get_energyFormat(12345, 3)
            >>> Out: 12.3 GWh'''
        number, power = self.millify(number, 3)
        number = round(number, decimal)
        dict_unit = {'': 'MWh', 'k': 'GWh', 'M': 'TWh'}
        unit = dict_unit[power]
        return (number, unit)

    def numFormat(self, number, currency='', unit='', decimal=0):
        return '{0}{1:,.{2}f}{3}'.format(currency, number, decimal, unit)

    def millify(self, number, max_power=4):
        ''' Output: (tuple) Returns the number millified and the power associated.
            >>> In : millify(12345)
            >>> Out: (12.345, k)'''
        power = ['', 'k', 'M', 'B']
        millnames = power[:max_power] # We consider the max_power first powers
        n = float(number)
        floor = int(math.floor(0 if n == 0 else math.log10(abs(n))/3))
        millidx = max(0, min(len(millnames)-1, floor))
        number = n/10**(3 * millidx)
        power = millnames[millidx]
        return (number, power)

    #---------------------------
    # PARSE DICT TO SERIES
    #---------------------------
    def parse_dict_to_series(self, hdfstore, parse_size=3):
        ''' Output: (list of list) '''
        print('>>> parse_dict_to_series <<<')
        series = defaultdict(list)
        count = 0
        n_series = 1
        for key in hdfstore.keys():
            if count >= parse_size:
                n_series +=1
                count = 0
            series[n_series].append(hdfstore[key])
            count += 1
        print('<<< parse_dict_to_series >>>')
        return series.values()

    def parse_keys_to_series(self, dict, parse_size=3):
        ''' Output: (list of list) '''
        series = defaultdict(list)
        count = 0
        n_series = 1
        for key in dict.keys():
            if count >= parse_size:
                n_series +=1
                count = 0
            series[n_series].append(key)
            count += 1
        return series.values()

    def parse_list_to_series(self, _list, parse_size=3):
        ''' Output: (list of list) '''
        series = defaultdict(list)
        count = 0
        n_series = 1
        for element in _list:
            if count >= parse_size:
                n_series +=1
                count = 0
            series[n_series].append(element)
            count += 1
        return series.values()





class MyColor:
    """ 
    Defines all the color styling that is goign to be used in the plots.
    Source: https://bsou.io/posts/color-gradients-with-python
    """
    def __init__(self):
        self.antuko = {'grey1': '#CCCCCC', #rgb(204, 204, 204)
                       'grey2': '#A5A7AB', #rgb(165, 167, 171)
                       'blue1': '#4CC7F1', #rgb(76, 199, 241)
                       'blue2': '#1A75BB'} #rgb(26, 117, 187)

    def linear_gradient(self, start_hex, finish_hex="#FFFFFF", n=10):
        ''' Returns a list of hex gradient '''
        return self._linear_gradient(start_hex, finish_hex, n)['hex']

    #-----------------------------
    #COLORS AS POINT IN 3D SPACE
    #-----------------------------
    def hex_to_RGB(self, hex):
        ''' "#FFFFFF" -> [255,255,255] '''
        # Pass 16 to the integer function for change of base
        return [int(hex[i:i+2], 16) for i in range(1,6,2)]

    def RGB_to_hex(self, RGB):
        ''' [255,255,255] -> "#FFFFFF" '''
        # Components need to be integers for hex to make sense
        RGB = [int(x) for x in RGB]
        return "#"+"".join(["0{0:x}".format(v) if v < 16 else
                "{0:x}".format(v) for v in RGB])

    #--------------------------------------------
    #LINEAR GRADIENTS AND LINEAR INTERPOLATION
    #--------------------------------------------
    def color_dict(self, gradient):
        ''' Takes in a list of RGB sub-lists and returns dictionary of
        colors in RGB and hex form for use in a graphing function
        defined later on '''
        return {"hex":[self.RGB_to_hex(RGB) for RGB in gradient],
          "r":[RGB[0] for RGB in gradient],
          "g":[RGB[1] for RGB in gradient],
          "b":[RGB[2] for RGB in gradient]}

    def _linear_gradient(self, start_hex, finish_hex="#FFFFFF", n=10):
        ''' returns a gradient list of (n) colors between
        two hex colors. start_hex and finish_hex
        should be the full six-digit color string,
        inlcuding the number sign ("#FFFFFF") '''
        # Starting and ending colors in RGB form
        s = self.hex_to_RGB(start_hex)
        f = self.hex_to_RGB(finish_hex)
        # Initilize a list of the output colors with the starting color
        RGB_list = [s]
        # Calcuate a color at each evenly spaced value of t from 1 to n
        for t in range(1, n):
            # Interpolate RGB vector for color at the current value of t
            curr_vector = [int(s[j] + (float(t)/(n-1))*(f[j]-s[j]))
                           for j in range(3)]
            # Add it to our list of output colors
            RGB_list.append(curr_vector)
        # Return
        return self.color_dict(RGB_list)

    #--------------------------------------------------------
    #MULTIPLE LINEAR GRADIENTS ==> POLYLINEAR INTERPOLATION
    #--------------------------------------------------------
    def rand_hex_color(self, num=1):
        ''' Generate random hex colors, default is one,
        returning a string. If num is greater than
        1, an array of strings is returned. '''
        colors = [self.RGB_to_hex([x*255 for x in np.random.rand(3)])
                  for i in range(num)]
        if num == 1:
            return colors[0]
        else:
            return colors

    def polylinear_gradient(self, colors, n):
        ''' returns a list of colors forming linear gradients between
          all sequential pairs of colors. "n" specifies the total
          number of desired output colors '''
        # The number of colors per individual linear gradient
        n_out = int(float(n) / (len(colors) - 1))
        # returns dictionary defined by color_dict()
        gradient_dict = self._linear_gradient(colors[0], colors[1], n_out)
        if len(colors) > 1:
            for col in range(1, len(colors) - 1):
                next = self._linear_gradient(colors[col], colors[col+1], n_out)
                for k in ("hex", "r", "g", "b"):
                    # Exclude first point to avoid duplicates
                    gradient_dict[k] += next[k][1:]
        return gradient_dict




class MyFigure:
    """Defines each type of figures that are going to be used."""
    def __init__(self, funds_name='', localtime=''):
        self.toolbox = MyToolBox()
        self.path = MyPath2(funds_name, localtime)   
        self.color = MyColor()

    # MAIN PLOTS
    def _visualStatistics(self, df, category=''):
        fig = self._make_statistical_figure(df)
        filename = self._set_filename(category, df.index.name)
        url = self._set_url(fig, filename=filename)
        return url

    def _yummyDonut(self, series, category='', color='#CCCCCC'):
        fig = self._make_donut_figure(series, color)
        filename = self._set_filename(category, series.name)
        url = self._set_url(fig, filename=filename)
        return url

    def _superKernel(self, series, category='', invert_x_y=False):
        fig = self._make_kernel_figure(series, invert_x_y)
        filename = self._set_filename(category, series.name)
        url = self._set_url(fig, filename=filename)
        return url

    def _simpleScatter(self, df, category='', colors=[]):
        fig = self._make_simpleScatter(df, colors)
        filename = self._set_filename(category, df.index.name)
        url = self._set_url(fig, filename=filename, auto_open=False)
        return url

    # TOOLBOX
    def _set_filename(self, category, filename):
        #_category_path = self.path._set_category_path(category)
        _category_path = self.path._set_figure_category_path(category)
        return os.path.join(_category_path, filename+'.html')

    def _set_url(self, fig, filename, auto_open=False):
        return pyol.plot(fig, filename=filename, auto_open=auto_open, show_link=False)

    # KITCHEN
    def _make_statistical_figure(self, dataframe):
        df = self._make_statistical_df(dataframe)

        color = {'min': self.color.antuko['grey1'],
                 '25%': self.color.antuko['grey2'],
                 '50%': self.color.antuko['blue1'],
                 'mean': self.color.antuko['blue2'],
                 '75%': self.color.antuko['grey2'],
                 'max': self.color.antuko['grey1']}

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
        xaxis = pygo.XAxis()
        yaxis = pygo.YAxis(ticksuffix='')
        margin = dict(l=34, r=0, t=3, b=30)
        legend = dict(orientation='v', xanchor='right') #bgcolor='transparent' for transparent background
        layout = pygo.Layout(xaxis=xaxis, yaxis=yaxis, autosize=True, margin=margin, legend=legend, showlegend=False)
        # LAYOUT
        fig = pygo.Figure(data=data, layout=layout)
        return fig

    def _make_simpleScatter(self, df, colors=[]):
        if not colors:
            start_color = self.color.antuko['blue1']
            finish_color = self.color.antuko['blue2']
            colors = self.color.linear_gradient(start_color, finish_color, len(df.columns))

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


    def _make_statistical_df(self, dataframe):
        '''Output: (dataframe) For each date in the date range, the min, 25%, 50%, 75% and the max.
            >>> value(date, [min, 25%, 50%, 75%, max])'''
        return self.toolbox.summary_df(dataframe, dropList=['count', 'std'])

    def _make_donut_figure(self, series, color='#CCCCCC'):
        #If the series is null then we replace it by a tiny value, otherwise the donut will not display
        if sum(series.values) == 0:
            series += 1e-6
        series = series.sort_values(ascending=False)
        labels = series.index
        values = series.values
        total_energy = sum(values)
        #colors = [self.color.antuko['blue2'], self.color.antuko['blue1'], self.color.antuko['grey2'], self.color.antuko['grey1'],]
        colors = self.color.linear_gradient(color, '#FFFFFF', n=len(labels)+1)


        # DATA
        trace = pygo.Pie(labels=labels, values=values,
                       hoverinfo='percent+value+label', textinfo='percent', 
                       textfont=dict(size=10),
                       textposition='outside',
                       marker=dict(colors=colors),
                       hole=.85)
        data = [trace]
        # LAYOUT
        number, unit = self.toolbox.get_energyFormat(total_energy)
        text = '<b>{number}</b><br>{unit}/mth'.format(number=number, unit=unit)
        annotations = [dict(text=text, showarrow=False, font=dict(size=15, color=color))]
        margin = dict(l=0, r=0, t=25, b=0)
        legend = dict(orientation='h', bgcolor='transparent')
        titlefont = dict(color=color)
        layout = pygo.Layout(title=series.name, titlefont=titlefont, annotations=annotations, margin=margin, legend=legend, showlegend=False)
        # FIGURE
        fig = pygo.Figure(data=data, layout=layout)
        return fig

    def _make_kernel_figure(self, series, invert_x_y=False, xticksuffix='', yticksuffix=''):
        # DATA & GROUPLABEL
        data = [series.dropna()]
        grouplabel = [series.name]
        # LAYOUT
        xtitle = ''
        ytitle = ''
        xaxis = pygo.XAxis(title=xtitle, ticksuffix=xticksuffix)
        yaxis = pygo.YAxis(title=ytitle, ticksuffix=yticksuffix)
        margin = dict(l=34, r=0, t=0, b=30)
        legend = dict(orientation='v', xanchor='right') #bgcolor='transparent' for transparent background
        layout = pygo.Layout(xaxis=xaxis, yaxis=yaxis, margin=margin, legend=legend, showlegend=False)
        # FIGURE
        colors = [self.color.antuko['blue2'], self.color.antuko['blue1'],
                  self.color.antuko['grey2'], self.color.antuko['grey1']]
        histnorm = 'probability' #histnorm: 'probability' or 'probability density'
        _min = min(series.values)
        _max = max(series.values)
        bin_number = 15
        bin_size = (_max - _min)/bin_number
        fig = pyff.create_distplot(data, grouplabel, bin_size=bin_size, show_hist=False, show_rug=False, histnorm=histnorm, colors=colors)
        fig['layout'].update(layout)
        # INVERT X Y AXIS
        if invert_x_y:
            fig = self._invert_x_y(fig)
        return fig

    def _invert_x_y(self, fig):
        #Layout inversion
        xaxis, yaxis = fig['layout']['xaxis'], fig['layout']['yaxis']
        fig['layout']['xaxis'], fig['layout']['yaxis'] = yaxis, xaxis
        #Data inversion
        for trace in fig['data']:
            trace['x'], trace['y'] = trace['y'], trace['x']
        #Data suppression
        del fig['layout']['xaxis1'], fig['layout']['yaxis1']
        del fig['layout']['hovermode']
        del fig['layout']['legend']
        #Return
        return fig


class MyPlot:
    """Define each type of plot that are going to be used."""
    def __init__(self, funds_name='', localtime=''):
        self.html = MyHTML()
        self.figure = MyFigure(funds_name, localtime)

    # MAIN PLOTS
    def visualStatistics(self, df_or_list, category=''):
        if isinstance(df_or_list, pd.DataFrame):
            df = df_or_list
            content = self._make_visualStatistics(df, category)
        elif isinstance(df_or_list, list):
            list_of_df = df_or_list
            content = []
            for df in list_of_df:
                iframe = self._make_visualStatistics(df, category)
                content.append(iframe)
        return content

    def yummyDonut(self, series, category='', title='', color='#CCCCCC'):
        return self._make_yummyDonut(series, category, title, color)

    def superKernel(self, series, category='', invert_x_y=False):
        return self._make_superKernel(series, category, invert_x_y)

    def simpleScatter(self, df, category='', colors=[]):
        return self._make_simpleScatter(df, category, colors)

    # KITCHEN
    def _make_visualStatistics(self, df, category):
        title = df.index.name
        url = self.figure._visualStatistics(df, category)
        return self._make_iframe(title=title, src=url, height=260)

    def _make_yummyDonut(self, series, category, title, color):
        title = series.name if not title == None else title
        url = self.figure._yummyDonut(series, category, color)
        return self._make_iframe(title=title, src=url, height=170, width='49%')

    def _make_superKernel(self, series, category, invert_x_y=False):
        dict_title = {'IRR (%)': 'Internal Rate of Return (%) distribution'}
        title = dict_title[series.name] if series.name in dict_title else series.name
        url = self.figure._superKernel(series, category, invert_x_y)
        return self._make_iframe(title=title, src=url, height=180)

    def _make_simpleScatter(self, df, category, colors):
        title = df.index.name
        url = self.figure._simpleScatter(df, category, colors)
        return self._make_iframe(title=title, src=url)

    # TOOLBOX
    def _make_iframe(self, src='', title='', classNameTitle='{classNameTitle}', seamless="seamless", style=dict(border=0), width="100%", height="250"):
        html_iframe = self.html.Iframe(src=src, seamless=seamless, style=style, width=width, height=height)
        html_title = self._make_title(title=title, className=classNameTitle)
        iframe = [html_title, html_iframe] if not title == None else [html_iframe]
        return iframe


    def _make_title(self, title='', className=''):
        return self.html.H6(title, className=className)

class MyTable:
    """Define each type of table that are going to be used."""
    def __init__(self):
        self.html = MyHTML()

    # TEXT TABLE
    def textTable(self, series):
        '''Creates a table that highlight important information.
        Note: works with series only'''
        title = self._make_html_title(series)
        table = self._make_text_table(series)
        return title + table

    def _make_text_table(self, series):
        html_text_table = ''
        for index in series.index:
            subtitle = self.html.Strong(index)
            content = self.html.P(series[index], className="blue-text")
            html_text_table += subtitle + content
        #We had a little tiny space a the end of the table
        #html_text_table += '<br>'
        return html_text_table

    # SWEETY TABLE
    def sweetyTable(self, list_or_df_or_series, indexed=True, auto_width=True, style={}):
        '''Returns a beautiful titled table.'''
        if isinstance(list_or_df_or_series, list):
            list_of_df_or_series = list_or_df_or_series
            content = [self._make_sweetyTable(df_or_series, indexed, auto_width, style) for df_or_series in list_of_df_or_series]
        elif isinstance(list_or_df_or_series, pd.DataFrame) or isinstance(list_or_df_or_series, pd.Series):
            df_or_series = list_or_df_or_series
            content = self._make_sweetyTable(df_or_series, indexed, auto_width, style)
        return content

    def _make_sweetyTable(self, df_or_series, indexed=True, auto_width=True, style={}):
        title = self._make_html_title(df_or_series)
        table = self._make_html_table(df_or_series, indexed, auto_width, style)
        return title + table

    def _make_html_table(self, df_or_series, indexed=True, auto_width=True, style={}):
        table = self._make_table(df_or_series, indexed, auto_width)
        return self.html.Table(table, style=style)

    def _make_table(self, df_or_series, indexed=True, auto_width=True):
        if isinstance(df_or_series, pd.DataFrame):
            df = df_or_series
            table = self._make_table_df(df, indexed, auto_width)
        elif isinstance(df_or_series, pd.Series):
            series = df_or_series
            table = self._make_table_series(series)
        return table

    def _make_table_df(self, df, indexed=True, auto_width=True):
        '''Returns a table with a heading row corresponding to the dataframe's columns.'''
        head = self._make_heading_row(df, indexed, auto_width)
        table = self._make_dash_table(df, indexed, auto_width)
        table.insert(0, head)
        return table

    def _make_table_series(self, series):
        # The series must be converted into a frame, otherwise, the function _make_dash_table will not work
        df = series.to_frame()
        table = self._make_dash_table(df)
        return table

    def _make_heading_row(self, dataframe, indexed=True, auto_width=True):
        df = dataframe if indexed else dataframe.set_index([dataframe.columns[0]])
        width = '' if auto_width else '{}%'.format(100. / (len(df.columns) + 1))
        html_cells = [self.html.Td(col, style={'text-align':'right', 'valign':'top'}, width=width) for col in df.columns]
        # Top index column space insertion
        if indexed:
            html_cells.insert(0, self.html.Td('', width=width))
        else:
            html_cells.insert(0, self.html.Td(df.index.name, width=width))
        html_row = self.html.Tr(html_cells, style={'background':'white', 'font-weight':'bold'})
        return html_row

    def _make_dash_table(self, dataframe, indexed=True, auto_width=True):
        '''Returns a list of HTML rows from a Pandas dataframe.'''
        df = dataframe if indexed else dataframe.set_index([dataframe.columns[0]])
        width = '' if auto_width else '{}%'.format(100. / (len(df.columns) + 1))
        table = []
        for index, row in df.iterrows():
            html_row = [self.html.Td([row[i]], style={'text-align':'right'}, width=width) for i in range(len(row))]
            # Left index insertion
            html_row.insert(0, self.html.Td([index], width=width))
            table.append(self.html.Tr(html_row))
        return table

    def _make_html_title(self, df_or_series):
        if isinstance(df_or_series, pd.DataFrame):
            df = df_or_series
            title = df.index.name
        elif isinstance(df_or_series, pd.Series):
            series = df_or_series
            title = series.name
        html_title = self.html.H6(title, className='{classNameTitle}')
        return html_title

class MyColumn:
    '''Contains all the CSS style of the report.'''
    def __init__(self):
        self.html = MyHTML()

    # MAIN COLUMNS
    def tiny(self, content_or_list):
        pass

    def little(self, content_or_list):
        content = self._apply_column_className(content_or_list, className="gs-header gs-text-header padded")
        return self._make_column(content, className="four columns")

    def middle(self, content_or_list):
        content = self._apply_column_className(content_or_list, className="gs-header gs-table-header padded")
        return self._make_column(content, className="six columns")

    def medium(self, content_or_list):
        content = self._apply_column_className(content_or_list, className="gs-header gs-table-header padded")
        return self._make_column(content, className="eight columns")

    def large(self, content_or_list):
        pass

    def xlarge(self, content_or_list):
        content = self._apply_column_className(content_or_list, className="gs-header gs-table-header padded")
        return self._make_column(content, className="twelve columns")

    #GEOMETRIC COLUMN
    def Column(self, content_or_list, className='four columns', header='light'):
        dict_tone = {'light': 'subsection', 'dark': 'section'}
        tone = dict_tone[header]
        # Content
        classNameContent = "gs-header gs-{tone}-header padded".format(tone=tone)
        content = self._apply_column_className(content_or_list, className=classNameContent)
        # Column 
        return self._make_column(content, className=className)


    # KITCHEN
    def _make_column(self, content_or_list=[], className=''):
        content = self.html._set_html_content_for_page_row_or_column(content_or_list)
        return self.html.Div(content, className=className)

    def _apply_column_className(self, content_or_list, className):
        content = self.html._set_html_content_for_page_row_or_column(content_or_list)
        return [element.format(classNameTitle=className) for element in content]


class MyRow:
    '''Contains all the CSS style of the report.'''
    def __init__(self):
        self.html = MyHTML()

    def Row(self, content_or_list):
        return self._make_row(content_or_list, className="row")

    def _make_row(self, content_or_list=[], className="row"):
        content = self.html._set_html_content_for_page_row_or_column(content_or_list)
        return self.html.Div(content, className=className)


class MyTemplate:
    '''Contains all the CSS style of the report.'''
    def __init__(self):
        self.html = MyHTML()
        self.path = MyPath2()
        # CSS
        self.css = self._get_css()
        # JAVA SCRIPT
        self.scripts = self._get_scripts()
        # PRINT PDF
        self.print_button = self.html.A(['Print PDF'],
                                        className="button no-print",
                                        style=dict(position="absolute", top=-40, right=0))
        # PAGE HEADER
        '''
        self.page_header = self.html.Div([

                            self.html.Div([
                                self.html.H5('Antuko Valuation and Risk Assessment Report'),
                                self.html.H6('Summary', style=dict(color='#7F90AC')),
                            ], className = "nine columns padded"),

                            self.html.Div([
                                self.html.H1('<span style="opacity:0.5">03</span><span>17</span>'),
                                self.html.H6('Monthly Fund Update')
                            ], className = "three columns gs-header gs-accent-header padded", style=dict(float='right')),

                        ], className="row gs-header gs-text-header")
        '''
        self.page_header = self.html.Div([

                self.html.Div([

                    self.html.Image(src=os.path.join(self.path.images_path, 'image1.png'), height='50px')

                ], style=dict(position='relative', left='5px', top='5px')),

                self.html.Div([

                    self.html.Image(src=os.path.join(self.path.images_path, 'image2.png'), height='60px')

                ], style=dict(position='absolute', left='85.8%', top='0px')),

                self.html.Div([

                    self.html.H5('Valuation and Risk Assessment Report', style={'text-align':'right'})

                ], className='gs-header', style=dict(position='absolute', right='24px', top='13px')),

                self.html.Div(style={'background-color':'#4A75B3','position':'relative', 'height':'5px', 'top':'8px'})

            ], className='header')

        # PAGE FOOTER
        '''
        self.page_footer = self.html.Div([
                            'This footer will always be positioned at the bottom of the page, but',
                            self.html.Strong('not too strong'), '.'
                        ], className="footer")
        '''
        self.page_footer = self.html.Div([

                '@ 2017  |  Antuko Energy S.A.  |  ',

                self.html.A('antuko.com', href='http://en.antuko.com/', style=dict(color='white'))

            ], className='footer')

    def _get_css(self):
        _css_path = self.path._css_path
        links = [os.path.join(_css_path, 'css{}.css'.format(i)) for i in [1,2,3,4,5]]
        css = [self.html.Css(link=link) for link in links]
        return css

    def _get_scripts(self):
        _scripts_path = self.path._scripts_path
        sources = [os.path.join(_scripts_path, 'js{}.js'.format(i)) for i in [1,2,3]]
        scripts = [self.html.Script(src=src) for src in sources]
        return scripts

debugOption = True
@decorate_classmethods(debug(debugOption))
class MyReport:
    '''Contains all the CSS style of the report.'''
    def __init__(self, funds_name='', localtime=''):
        self.html = MyHTML()
        self.path = MyPath2(funds_name, localtime)
        self.template = MyTemplate()
        '''STYLE'''
        self.css = '\n'.join(self.template.css)
        self.scripts = '\n'.join(self.template.scripts)
        self.pages = []
        self.html_html = ''
        self.title = 'Report - {name}'.format(name=funds_name)
        self.html_title = os.path.join(self.path.localtime_dir_path, self.title+'.html')
        '''HTML CONSTRUCTORS'''
        self.plot = MyPlot(funds_name, localtime)
        self.table = MyTable()
        self.column = MyColumn()
        self.row = MyRow()
        self.color = MyColor()
        self.toolbox = MyToolBox()



    def new_page(self, content, title=''):
        self.pages.append(self._make_page(content, title))

    def _make_page(self, content_or_list, title):
        content = self.html._set_html_content(content_or_list)
        html_page = self.html.Div([
            # Print PDF button
            self.template.print_button,
            # Header
            self.template.page_header,
            # Subpage
            self.html.Div([
                # Page content
                content,
            ], className="subpage"),
            # Page footer
            self.template.page_footer
        ], className="page")
        return html_page

    def create(self):
        self._build_layout()
        self._build_html()
        self._write_html()
        self._open_html()

    def _build_layout(self):
        self.layout = self.html.Div('\n'.join(self.pages))

    def _build_html(self):
        self.html_html = self.html.HTML(title=self.title, layout=self.layout, css=self.css, scripts=self.scripts)

    def _write_html(self):
        f = open(self.html_title, 'w')
        f.write(self.html_html.encode('utf8')) # Solve the problem of spanish accents
        f.close()

    def _open_html(self, auto_open=True):
        if auto_open:
            new = 2 # open in a new tab, if possible
            webbrowser.open(self.html_title, new=new)

def read_excel_dict_df(io, name=''):
    xl = pd.ExcelFile(io)
    dict_df = {}
    for sheet_name in xl.sheet_names:
        df = xl.parse(sheet_name)
        df = df.set_index([df.columns[0]])
        dict_df[sheet_name] = df
    return dict_df

def read_excel_df(io, name=''):
    df = pd.read_excel(io)
    df = df.set_index([df.columns[0]])
    if name:
        df.index.name = name
    return df

def read_excel_series(io, name=''):
    df = read_excel_df(io, name)
    df = pd.read_excel(io)
    series = df[df.columns[0]]
    if name:
        series.name = name
    return series

def read_csv(io, name=''):
    df = pd.read_csv(io)
    df = df.set_index([df.columns[0]])
    if name:
        df.index.name = name
    return df

def get_excel_filename(path, name):
    return os.path.join(path, name)


if __name__ == '__main__':
    html = MyHTML()
    table = MyTable()
    column = MyColumn()
    figure = MyFigure()
    toolbox = MyToolBox()
    row= MyRow()
    plot = MyPlot()
    report = MyReport()
    color = MyColor()

    background = 'url(Antuko_Header_banner2.png)' 

    #TABLES
    tables_path = 'Data_Store\\Excel Assessment\\tables\\'

    io = tables_path + 'Portofolio Summary.xlsx'
    Portofolio_Summary = read_excel_series(io)

    io = tables_path + 'Portofolio Features.xlsx'
    Portofolio_Features = read_excel_series(io)

    io = tables_path + 'Computation Parameters.xlsx'
    Computation_Parameters = read_excel_series(io)

    io = tables_path + 'Generation Assets.xlsx'
    Generation_Assets = read_excel_df(io)

    io = tables_path + 'Withdrawal PPAs.xlsx'
    Withdrawal_PPAs = read_excel_df(io)
    
    #LOAD DATAFRAMES
    full_excel_path = 'Data_Store\\Excel Assessment\\full\\'

    io = full_excel_path + 'Generation\\Energy Generation P90 P50.xlsx'
    Energy_Generation_P90_P50 = read_excel_dict_df(io)

    io = full_excel_path + 'Demand\\Energy Demand.xlsx'
    Energy_Demand = read_excel_dict_df(io)

    #ENERGY GENERATION
    dict_series = {generator: int(Energy_Generation_P90_P50[generator].mean(axis=1).mean(axis=0)) for generator in Energy_Generation_P90_P50}
    generation_series = pd.Series(dict_series, name='Generation')

    #ENERGY DEMAND
    dict_series = {consumer: int(Energy_Demand[consumer].mean(axis=1).mean(axis=0)) for consumer in Energy_Demand}
    demand_series = pd.Series(dict_series, name='Demand')

    
    #----------------------------
    # PAGE1: SUMMARY
    #----------------------------

    report.new_page([

        # Row 1
        row.Row([

            column.Column([
                table.textTable(Portofolio_Summary),
            ], className='twelve columns', header='dark'),

        ]),

        # Row 2
        row.Row([

            column.Column([
                table.sweetyTable(Portofolio_Features),
            ], className='one-half column', header='light'),

            column.Column([
                table.sweetyTable(Computation_Parameters),
            ], className='one-half column', header='light'),

        ]),

        # Row 3
        row.Row([

            # XLarge column
            column.Column([
                table.sweetyTable(Generation_Assets, indexed=False, auto_width=False),
                table.sweetyTable(Withdrawal_PPAs, indexed=False, auto_width=False),
            ], className='twelve columns', header='light'), 

        ]),

    ])

    #---------------------------------------------
    # PAGE2: CHARACTERISTICS AND RESULTS SUMMARY
    #---------------------------------------------

    #SHAREHOLDER CASH FLOW STATISTICS
    io = tables_path + 'Shareholder Cash Flow Statistics.xlsx'
    Shareholder_Cash_Flow_Statistics = read_excel_df(io)

    #Energy demand
    energy_demand = sum(Energy_Demand.values())
    avg_energy_demand = energy_demand.mean(axis=1) # Type: Series
    avg_energy_demand.name = 'Demand'
    #Energy generation P90 P50
    energy_generation = sum(Energy_Generation_P90_P50.values())
    avg_energy_generation = energy_generation.mean(axis=1)
    avg_energy_generation.name = 'Generation'
    #Generation and Demand
    frames = [avg_energy_generation, avg_energy_demand]
    generation_and_demand = pd.concat(frames, axis=1)
    generation_and_demand.index.name = 'Generation and Demand (MWh)'

    report.new_page([

        # Row 1
        row.Row([

            # Column 1
            column.Column([
                table.sweetyTable(Shareholder_Cash_Flow_Statistics)
            ], className='one-third column', header='dark'),

            # Column 2
            column.Column([
                plot.simpleScatter(generation_and_demand, category='', colors=[color.antuko['blue1'], color.antuko['blue2']]),
                plot.yummyDonut(generation_series, category='Generation', title=None, color=color.antuko['blue1']),
                plot.yummyDonut(demand_series, category='Demand', title=None, color=color.antuko['blue2'])
            ], className='two-thirds column', header='light'),

        ])

    ])

    #---------------------------------------
    # PAGE 3: NET LIQUID ASSETS DIVIDEND 
    #---------------------------------------

    #NET LIQUID ASSETS
    io = full_excel_path + 'Finance\\Net Liquid Assets.xlsx'
    Net_Liquid_Assets = read_excel_df(io)

    #DIVIDEND
    io = full_excel_path + 'Finance\\Dividend.xlsx'
    Dividend = read_excel_df(io)
    Dividend = Dividend.cumsum(axis=0)

    report.new_page([

        # Row 1
        row.Row([

            # Column 1

            column.Column([
                plot.visualStatistics(Net_Liquid_Assets, category='Finance'),
            ], className='two-thirds column', header='dark'),

            # Column 2
            column.Column([
                table.sweetyTable(
                    toolbox.last_date_summary_series(Net_Liquid_Assets, numFormat='currency')
                )
            ], className='one-third column', header='light'),

        ]),

        # Row 2

        row.Row([

            # Column 1
            column.Column([
                plot.visualStatistics(Dividend, category='Finance'),
            ], className='two-thirds column', header='dark'),

            # Column 2

            column.Column([
                table.sweetyTable(
                    toolbox.last_date_summary_series(Dividend, numFormat='currency')
                )
            ], className='one-third column', header='light'),

        ]),

    ])

    #----------------------------
    # PAGE4: PROFITS ANALYSIS
    #----------------------------

    #GROSS PROFIT
    io = full_excel_path + 'Finance\\Gross Profit.xlsx'
    Gross_Profit = read_excel_df(io)

    #NET INCOMES
    io = full_excel_path + 'Finance\\Net Incomes.xlsx'
    Net_Incomes = read_excel_df(io)

    report.new_page([

        # Row 1

        row.Row([

            # Column 1
            column.Column([
                table.sweetyTable(
                    toolbox.monthly_yearly_summary_df(Gross_Profit, method='sum', numFormat='currency')
                )
            ], className='one-third column', header='dark'),

            # Column 2
            column.Column([
                plot.visualStatistics(Gross_Profit, category='Finance'),
            ], className='two-thirds column', header='light'),

        ]),

        # Row 2

        row.Row([

            # Column 1

            column.Column([

                table.sweetyTable(
                    toolbox.monthly_yearly_summary_df(Net_Incomes, method='sum', numFormat='currency')
                )

            ], className='one-third column', header='dark'),

            # Column 2
            column.Column([
                plot.visualStatistics(Net_Incomes, category='Finance'),
            ], className='two-thirds column', header='light'),

        ]),

    ])


    #---------------------------------------
    # PAGE 5: SHAREHOLDER CASHFLOW 
    #---------------------------------------

    #SHAREHOLDER CASH FLOW
    io = full_excel_path + 'Finance\\Shareholder Cash Flow.xlsx'
    Shareholder_Cash_Flow = read_excel_df(io)

    #SHAREHOLDER CASH FLOW STATISTICS
    io = tables_path + 'Shareholder Cash Flow Statistics.xlsx'
    Shareholder_Cash_Flow_Statistics = read_excel_df(io)

    report.new_page([

        # Row 1

        row.Row([

            # Column 1
            column.Column([
                plot.visualStatistics(Shareholder_Cash_Flow, category='Finance'),
            ], className='two-thirds column', header='light'),

            # Column 2
            column.Column([
                table.sweetyTable(Shareholder_Cash_Flow_Statistics)
            ], className='one-third column', header='dark'),

        ]),

    ])

    #---------------------------------------
    # PAGE 5: ENERGY GERENATION 
    #---------------------------------------

    # ENERGY GENERATION P90 P50
    io = full_excel_path + 'Generation\\Energy Generation P90 P50.xlsx'
    Energy_Generation_P90_P50 = read_excel_dict_df(io)

    generation_P90_P50_series = toolbox.parse_dict_to_series(Energy_Generation_P90_P50)
    for list_generation_P90_P50 in generation_P90_P50_series:

        report.new_page([


            # Row 1

            row.Row([

                # Column 1

                column.little([

                    table.sweetyTable(

                        toolbox.monthly_yearly_summary_df(generation_P90_P50, method='sum', numFormat='energy')

                    )

                ]),

                # Column 2

                column.medium([

                    plot.visualStatistics(generation_P90_P50, category='Generation'),

                ]),

            ]) for generation_P90_P50 in list_generation_P90_P50

        ])

    #---------------------------------------
    # PAGE 6: ENERGY DEMAND
    #---------------------------------------

    # ENERGY DEMAND
    io = full_excel_path + 'Demand\\Energy Demand.xlsx'
    Energy_Demand = read_excel_dict_df(io)

    demand_series = toolbox.parse_dict_to_series(Energy_Demand)
    for list_demand in demand_series:

        report.new_page([


            # Row 1

            row.Row([

                # Column 1

                column.little([

                    table.sweetyTable(

                        toolbox.monthly_yearly_summary_df(demand, method='sum', numFormat='energy')

                    )

                ]),

                # Column 2

                column.medium([

                    plot.visualStatistics(demand, category='Demand'),

                ]),

            ]) for demand in list_demand

        ])

    #---------------------------------------
    # PAGE 7: SPOT PRICE
    #---------------------------------------

    # ENERGY DEMAND
    io = full_excel_path + 'Price\\SPOT Price.xlsx'
    SPOT_Price = read_excel_dict_df(io)

    spot_price_series = toolbox.parse_dict_to_series(SPOT_Price)
    for list_spot_price in spot_price_series:

        report.new_page([


            # Row 1

            row.Row([

                # Column 1

                column.little([

                    table.sweetyTable(

                        toolbox.monthly_yearly_summary_df(price, method='mean', numFormat='currency')

                    )

                ]),

                # Column 2

                column.medium([

                    plot.visualStatistics(price, category='Price'),

                ]),

            ]) for price in list_spot_price

        ])

    #---------------------------------------
    # REPORT COMPILATION
    #---------------------------------------
    report.create()
    