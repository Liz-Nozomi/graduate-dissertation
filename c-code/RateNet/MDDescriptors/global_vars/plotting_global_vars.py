# -*- coding: utf-8 -*-
"""
plotting_global_vars.py
The purpose of this script is to contain all global variables for plotting purposes. Things like font size, etc. should be placed here!

CREATED ON : 05/06/2018

AUTHOR(S):
    Alex K. Chew (alexkchew@gmail.com)
"""

### DEFINING GLOBAL PLOTTING VARIABLES
FONT_SIZE=16
FONT_NAME="Arial" 

LABELS = {
            'fontname': FONT_NAME,
            'fontsize': FONT_SIZE
            }

### DEFINING COLOR LIST
from matplotlib import colors
COLOR_LIST=['k','b','r','g','m','y','c'] + list(colors.CSS4_COLORS.keys()) ## LOTS OF COLORS

### DEFINING LINE STYLE
LINE_STYLE={
            "linewidth": 1.4, # width of lines
            }
### DEFINING SAVING STYLE
DPI_LEVEL=600
