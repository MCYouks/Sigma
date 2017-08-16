# -*- coding: utf-8 -*-
'''

The objective is to build a report with html code generated from python.

@author: Andréas

@date: Sunday, August the 12th
'''

from bs4 import BeautifulSoup
from path import MyPath2
from html import *

import webbrowser
import os

class template(object):
    """
    That class aims to build the template for the report.
    """
    def __init__(self, title='HTML Report by Andy'):
        self.title = title
        self.path = MyPath2()

    @property
    def Head(self):
        ''' Defines the HTML head. '''
        head = Head([

            Meta(charset='utf-8'),

            Title(self.title),

            self.Css,

        ])

        return head

    @property
    def Footer(self):
        ''' Defines the HTML footer. '''
        footer = Footer([

            self.Scripts

        ])

        return footer

    @property
    def Css(self):
        ''' Load all the css stylesheets. '''
        def get_css():
            css_path = self.path._css_path
            hrefs = [os.path.join(css_path, 'css{}.css'.format(i)) for i in [1,2,3,4,5]]
            css = [Link(href=href, rel='stylesheet') for href in hrefs]
            return css

        return get_css()

    @property
    def Scripts(self):
        ''' Load all the java scripts. '''
        def get_scripts():
            scripts_path = self.path._scripts_path
            srcs = [os.path.join(scripts_path, 'js{}.js'.format(i)) for i in [1,2,3]]
            scripts = [Script(src=src, type='text/JavaScript') for src in srcs]
            return scripts

        return get_scripts()

    @property
    def Print_Button(self):
        ''' Defines the 'print to PDF' button. '''
        print_button = A([

            'Print PDF'

        ], className='button no-print', style=dict(position='absolute', top='-40', right='0'))

        return print_button

    @property
    def Page_Header(self):
        ''' Defines the page header. '''
        page_header = Div([

            Div([

                Div([

                    Image(src=os.path.join(self.path.images_path, 'image1.png'), height='100%')

                ], className="header-image1"),

                Div([

                    Image(src=os.path.join(self.path.images_path, 'image2.png'), height='100%')

                ], className="header-image2"),

                Div([

                    H5('Valuation and Risk Assessment Report', style={'text-align':'right'})

                ], className='gs-header header-text'),

                Div(className='header-subline')

            ], className="header-background")

        ], className='header')

        return page_header

    @property
    def Page_Footer(self):
        ''' Defines the page footer. '''
        page_footer = Div([

            '@ 2017  |  Antuko Energy S.A.  |  ',

            A('antuko.com', href='http://en.antuko.com/', style=dict(color='white'))

        ], className='footer')

        return page_footer


    def Page(self, content=''):
        ''' Defines the HTML page format. '''
        page = Page([

            self.Print_Button,

            self.Page_Header,

            Subpage(content),

            self.Page_Footer,

        ])

        return page


class report(template):
    """
    That class aims to build the HTML report.
    """
    def __init__(self, *args, **kwargs):
        ''' The class inherits the methods from the class template. '''
        super(report, self).__init__(*args, **kwargs)
        self.filename = os.path.join(self.path.localtime_dir_path, self.title+'.html')
        self.Pages = []

    def New_Page(self, content=''):
        ''' Each new page is stored into a list. '''
        new_page = self.Page(content)
        self.Pages.append(new_page)

    @property
    def Body(self):
        ''' Creates the body structure of the HTML doc. ''' 
        return Body(self.Pages)

    @property
    def Html(self):
        def prettify(html_doc):
            soup = BeautifulSoup(html_doc, 'html.parser')
            return soup.prettify()

        html = Html([

            self.Head,

            self.Body,

            self.Footer

        ])

        return prettify(html)

    def Build(self):
        def check_if_path_exists():
            self.path._check_if_path_exists(self.path.localtime_dir_path)
        def write_html():
            with open(self.filename, 'w') as f:
                f.write(self.Html.encode('utf8')) # Solves the problem of spanish accents
        def open_html(auto_open=True):
            if auto_open:
                new = 2 # open in a new tab, if possible
                webbrowser.open(self.filename, new=new)


        check_if_path_exists()
        write_html()
        open_html()
        return self.Html


if __name__ == '__main__':

    Report = report()

    Report.New_Page()
    Report.New_Page()

    html_doc = Report.Build()

    print html_doc
    