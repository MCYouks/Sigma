# -*- coding: utf-8 -*-
'''

The objective is to build html code using python decorators.

@author: Andréas

@date: Sunday, August the 12th
'''
from bs4 import BeautifulSoup

def set_content(str_or_list):
    def set_text(str_or_list):
        def set_input(str_or_list):
            def unlist(aList):
                t = tuple()
                for element in aList:
                    if isinstance(element, list):
                        item = unlist(element)
                    else:
                        item = tuple([element])
                    t += item
                return t
            if isinstance(str_or_list, list):
                return unlist(str_or_list)
            else:
                return tuple([str_or_list])
        def set_unicode(text):
            if isinstance(text, str):
                return text.decode('utf-8')
            elif isinstance(text, unicode):
                return text
        return [set_unicode(text) for text in set_input(str_or_list)]
    return ''.join(set_text(str_or_list))

def set_css_param(param):
    def set_param(param):
        def set_key(key):
            if key == 'className':
                return 'class'
            else:
                return key
        def set_value(value):
            if isinstance(value, str):
                return value
            elif isinstance(value, dict):
                return '; '.join(['{k}:{v}'.format(k=k, v=v) for k, v in value.items()])
        return ' '.join(['{k}="{v}"'.format(k=set_key(key), v=set_value(value)) for key, value in param.items()])
    return set_param(param)

def tag(tag_name):
    def decorator(func):
        def wrapper(str_or_list='', **param):
            content = set_content(str_or_list)
            css_param = set_css_param(param)
            return u'<{tag} {param}>{content}</{tag}>'.format(tag=tag_name, content=content, param=css_param)
        return wrapper
    return decorator

@tag('h6')
def H6(*args, **param):
    pass

@tag('h5')
def H5(*args, **param):
    pass

@tag('strong')
def Strong(*args, **param):
    pass

@tag('iframe')
def Iframe(*args, **param):
    pass

@tag('image')
def Image(*args, **param):
    pass

@tag('link')
def Link(*args, **param):
    pass

@tag('script')
def Script(*args, **kwargs):
    pass

@tag('head')
def Head(*args, **kwargs):
    pass

@tag('meta')
def Meta(*args, **kwargs):
    pass

@tag('title')
def Title(*args, **kwargs):
    pass

@tag('footer')
def Footer(*args, **kwargs):
    pass

@tag('html')
def Html(*args, **kwargs):
    pass

@tag('div')
def Div(*args, **param):
    pass

@tag('br')
def Br(*args, **kwargs):
    pass

@tag('a')
def A(*args, **kwargs):
    pass

@tag('p')
def P(*args, **kwargs):
    pass

@tag('td')
def Td(*args, **kwargs):
    pass

@tag('tr')
def Tr(*args, **kwargs):
    pass

@tag('table')
def Table(*args, **kwargs):
    pass

@tag('body')
def Body(*args, **kwargs):
    pass

@tag('footer')
def Footer(*args, **kwargs):
    pass

def Page(*args, **param):
    param.update(dict(className='page'))
    return Div(*args, **param)

def Subpage(*args, **param):
    param.update(dict(className='subpage'))
    return Div(*args, **param)
    


if __name__ == '__main__':

    pass


    