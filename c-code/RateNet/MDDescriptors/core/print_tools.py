# -*- coding: utf-8 -*-
"""
print_tools.py
The purpose of this script is to store all printing functions.

AUTHOR(S):
    Alex K. Chew (alexkchew@gmail.com)
    
FUNCTIONS:
    store_figure: function to store figures
"""

### FUNCTION THAT DEALS WITH SAVING FIGURE
def store_figure(fig, path, fig_extension = 'png',
                 save_fig=False, 
                 bbox_inches = None,
                 dpi=1200):
    '''
    The purpose of this function is to store a figure.
    INPUTS:
        fig: [object]
            figure object
        path: [str]
            path to location you want to save the figure (without extension)
        fig_extension: [str]
            figure extension
        bbox_inches: [str, default None]
            'tight' if you want to save picture tightly
    OUTPUTS:
        void
    '''
    ## STORING FIGURE
    if save_fig is True:
        ## DEFINING FIGURE NAME
        fig_name =  path + '.' + fig_extension
        print("Printing figure: %s"%(fig_name) )
        fig.savefig( fig_name, 
                     format=fig_extension, 
                     dpi = dpi,    
                     bbox_inches = bbox_inches,
                     )
    return
