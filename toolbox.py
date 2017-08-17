# -*- coding: utf-8 -*-
'''

The objective is set useful snippets of code that will be useful in several modules.

@author: Andréas

@date: Sunday, August the 12th
'''

import pandas as pd
import numpy as np
import math


#---------------------------------
# DESCRIBE FRAMES
#---------------------------------
def describe_df(dataframe, dropList=['count'], percentiles=[.25,.5,.75]):
    '''
    Returns the total count, the average, the standard deviation, 
    the min, 25%, 50%, 75% and the max, for each date in the date range.

    Parameters:
    -----------
    dropList: list of the columns we want to drop, default ['count']

    Output:
    --------
    dataframe:  value(date, [count, mean, std, min, 25%, 50%, 75%, max])
    '''
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

def describe_series(series, dropList=[], percentiles=[.25,.5,.75]):
    '''
    Returns the total count, the average, the standard deviation, 
    the min, 25%, 50%, 75% and the max, for each date in the date range.

    Parameters:
    -----------
    dropList: list of the columns we want to drop, default ['count']

    Output:
    --------
    series:  value([count, mean, std, min, 25%, 50%, 75%, max])
    '''
    return series.describe(percentiles).drop(dropList)

def describe(df_or_series, dropList=['count'], percentiles=[.25,.5,.75]):
    ''' Returns a summary of a dataframe or series. '''
    if isinstance(df_or_series, pd.Series):
        return describe_series(df_or_series, dropList, percentiles)
    elif isinstance(df_or_series, pd.DataFrame):
        return describe_df(df_or_series, dropList, percentiles)

#---------------------------------
# FORMATING NUMBER AND CURRENCIES
#---------------------------------
def numFormat(number, currency='', unit='', decimal=0):
    ''' Formats the number into q string. '''
    return '{0}{1:,.{2}f}{3}'.format(currency, number, decimal, unit)

def percentFormat(number, decimal=0):
    ''' 
    Returns a number in a percentage format. 

    Parameters:
    -----------
    number: scalar
    decimal: int, number of decimals after the comma.

    Output:
    ------
    string: number in a percentage format.
    '''
    if math.isnan(number) or math.isinf(number):
        return number
    else:
        return '{0:,.{1}f}%'.format(number, decimal)

def millify(number, max_power=4):
    ''' 
    Returns the number millified and the power associated.

    Ouputs:
    -------
    tuple: (number millified, power)

    Example:
    --------
        >>> In : millify(12345)
        >>> Out: (12.345, k)
    '''
    power = ['', 'k', 'M', 'B']
    millnames = power[:max_power] # We consider the max_power first powers
    n = float(number)
    floor = int(math.floor(0 if n == 0 else math.log10(abs(n))/3))
    millidx = max(0, min(len(millnames)-1, floor))
    number = n/10**(3 * millidx)
    power = millnames[millidx]
    return (number, power)

def currencyFormat(number, currency='$', decimal=1):
    '''
    Return a currency value with the right financial format.

    Output:
    -------
    string: number in a currency format.

    Example:
    --------
        >>> In : currencyFormat(12345)
        >>> Out: k$ 12.3
    '''
    def get_currencyFormat(number, currency='$', decimal=1):
        '''
        Returns the couple (value, currency). The currency is adjusted depending on the nature of the
        currency sign ('$' or 'USD').
        '''
        number, power = millify(number)
        number = round(number, decimal)
        if currency == 'USD':
            currency = power + currency # We insert the power before the 'USD' and a space before the number
        elif currency == '$':
            currency = currency + power # We insert the power after the '$' sign
        return (number, currency)

    if math.isnan(number) or math.isinf(number):
        return number
    else:
        number, currency = get_currencyFormat(number, currency, decimal)
        currency = currency + ' ' # We insert a space between the currency and the number
        number = numFormat(number, currency=currency, decimal=decimal)
        return number

def energyFormat(number, decimal=1):
    '''
    Return a energy value with the right scientific format.

    Parameters:
    -----------
    number: float, energy value in (MWh)

    Output:
    -------
    string: energy value in a scientific format.

    Example:
    --------
        >>> In : energyFormat(12345)
        >>> Out: 12.3 GWh
    '''
    def get_energyFormat(number, decimal=1):
        '''
        Returns the couple (adjusted value, power unit).
        '''
        number, power = millify(number, 3)
        number = round(number, decimal)
        dict_unit = {'': 'MWh', 'k': 'GWh', 'M': 'TWh'}
        unit = dict_unit[power]
        return (number, unit)

    if math.isnan(number) or math.isinf(number):
        return number
    else:
        number, unit = get_energyFormat(number, decimal)
        unit = ' ' + unit # We insert a space between the number and the units
        return numFormat(number, unit=unit, decimal=decimal)

def formatSeries(series, format, decimal=1):
    '''
    Formats a series of data.

    Parameters:
    -----------
    format: string, {percent, currency, energy}
    '''
    def formator(format, decimal):
        def wrapper(number):
            dict_f = {'percent': percentFormat, 'currency': currencyFormat,
                      'energy': energyFormat}
            f = dict_f[format]
            return f(number, decimal=decimal)
        return wrapper
    return series.apply(formator(format, decimal))

def formatDataFrame(dataframe, format, decimal=1, axis=1):
    '''
    Formats a dataframe.

    Parameters:
    -----------
    format: string, {percent, currency, energy}
    '''
    def format_all():
        def formator(format, decimal):
            def wrapper(series):
                return formatSeries(series, format, decimal)
            return wrapper
        return dataframe.apply(formator(format, decimal))
    def format_by_columns():
        frames = []
        for i, frmt in enumerate(format):
            series = dataframe[dataframe.columns[i]]
            frames.append(formatSeries(series, frmt, decimal))
        return pd.concat(frames, axis=1)
    def format_by_index():
        frames = []
        for i, frmt in enumerate(format):
            series = dataframe.loc[dataframe.index[i]]
            frames.append(formatSeries(series, frmt, decimal))
        return pd.concat(frames, axis=1).T

    if isinstance(format, str):
        return format_all()
    elif isinstance(format, list):
        if axis == 0:
            return format_by_index()
        elif axis == 1:
            return format_by_columns()

#---------------------------------
# COLOR GRADIENT GENERATOR
#---------------------------------
"""
Source: https://bsou.io/posts/color-gradients-with-python
"""

#COLORS AS POINT IN 3D SPACE

def hex_to_RGB(hex):
    ''' "#FFFFFF" -> [255,255,255] '''
    # Pass 16 to the integer function for change of base
    return [int(hex[i:i+2], 16) for i in range(1,6,2)]

def RGB_to_hex(RGB):
    ''' [255,255,255] -> "#FFFFFF" '''
    # Components need to be integers for hex to make sense
    RGB = [int(x) for x in RGB]
    return "#"+"".join(["0{0:x}".format(v) if v < 16 else
            "{0:x}".format(v) for v in RGB])


#LINEAR GRADIENTS AND LINEAR INTERPOLATION

def color_dict(gradient):
    ''' 
    Takes in a list of RGB sub-lists and returns dictionary of
    colors in RGB and hex form for use in a graphing function
    defined later on.
    '''
    return {"hex":[RGB_to_hex(RGB) for RGB in gradient],
      "r":[RGB[0] for RGB in gradient],
      "g":[RGB[1] for RGB in gradient],
      "b":[RGB[2] for RGB in gradient]}

def linear_gradient(start_hex, finish_hex="#FFFFFF", n=10):
    ''' 
    Returns a gradient list of (n) colors between
    two hex colors. start_hex and finish_hex
    should be the full six-digit color string,
    inlcuding the number sign ("#FFFFFF").
    '''
    # Starting and ending colors in RGB form
    s = hex_to_RGB(start_hex)
    f = hex_to_RGB(finish_hex)
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
    return color_dict(RGB_list)

def hex_linear_gradient(start_hex, finish_hex="#FFFFFF", n=10):
    ''' Returns a list of hex gradient '''
    return linear_gradient(start_hex, finish_hex, n)['hex']


#MULTIPLE LINEAR GRADIENTS ==> POLYLINEAR INTERPOLATION

def rand_hex_color(num=1):
    ''' 
    Generate random hex colors, default is one,
    returning a string. If num is greater than
    1, an array of strings is returned. 
    '''
    colors = [RGB_to_hex([x*255 for x in np.random.rand(3)])
              for i in range(num)]
    if num == 1:
        return colors[0]
    else:
        return colors

def polylinear_gradient(colors, n):
    ''' 
    Returns a list of colors forming linear gradients between
    all sequential pairs of colors. "n" specifies the total
    number of desired output colors 
    '''
    # The number of colors per individual linear gradient
    n_out = int(float(n) / (len(colors) - 1))
    # returns dictionary defined by color_dict()
    gradient_dict = linear_gradient(colors[0], colors[1], n_out)
    if len(colors) > 1:
        for col in range(1, len(colors) - 1):
            next = linear_gradient(colors[col], colors[col+1], n_out)
            for k in ("hex", "r", "g", "b"):
                # Exclude first point to avoid duplicates
                gradient_dict[k] += next[k][1:]
    return gradient_dict

if __name__ == '__main__':
    data = {
        'boit': [12345, 123456, 1234567],
        'mange' : [12, 123, 1234],
        'couleur': [12345678, 123456789, 1234567890]
    }
    dataframe = pd.DataFrame(data).T

    print formatDataFrame(dataframe, ['currency', 'percent', 'energy'], decimal=1, axis=0)


