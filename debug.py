# *- coding: utf-8 -*-
"""
Created on Tuesday July the 18th, 2017

Useful Debugging Classes for Arkenstone

@author: Andréas

Source: http://www.linuxtopia.org/online_books/programming_books/python_programming/python_ch26s03.html
"""

import traceback
import inspect
import sys

def PrintException(aFunc, exception):
    ''' Display the *original* exception '''
    #Error message
    print('''\n
    *****************************************
        Error Message: Exception raised
    *****************************************
    ''')
    #Print func name
    print('\nFunc:\n*************')
    print aFunc.__name__
    #Print func doc
    print('\nDoc:\n*************')
    print aFunc.__doc__
    #Exception
    print('\nException:\n*************')
    print exception
    #Traceback
    exc_info = sys.exc_info()
    print('\nTraceback:\n*************')
    traceback.print_exception(*exc_info)
    del exc_info

def debug(debugOption):
    """
    1. This decorator is created from the argument, theSetting.

    2.If theSetting is True, the concrete decorator will create the result function 
    named debugFunc, which prints a message and then uses the argument function.

    3.If theSetting is False, the concrete descriptor will simply return the argument 
    function without any overhead.
    """
    def trace(aFunc):
        """Trace entry, exit and exceptions."""
        if debugOption:
            def loggedFunc(*args, **kwargs):
                print '>>> {name} <<<'.format(name=aFunc.__name__)
                try:
                    result= aFunc(*args, **kwargs)
                except Exception, e:
                    PrintException(aFunc, e)
                    raise
                print '<<< {name} >>>'.format(name=aFunc.__name__)
                return result
            loggedFunc.__name__= aFunc.__name__
            loggedFunc.__doc__= aFunc.__doc__
            return loggedFunc
        else:
            return aFunc
    return trace

def decorate_classmethods(decorator, prefix='_'):
    """ To apply a decorator to all the method of a class. """
    def decorate(cls):
        for name, method in inspect.getmembers(cls, inspect.ismethod):
            if not name.startswith(prefix):
                setattr(cls, name, decorator(method))
        return cls
    return decorate


debugOption = True
@decorate_classmethods(debug(debugOption))
class MyClass(object):
    def __init__( self, someValue ):
        """Create a MyClass instance."""
        self.value= someValue

    def doSomething( self, anotherValue ):
        """Update a value."""
        self.value += anotherValue


if __name__ == '__main__':
    mc = MyClass(23)
    mc.doSomething([])
    print mc.value