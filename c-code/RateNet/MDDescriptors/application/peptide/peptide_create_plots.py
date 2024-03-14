# -*- coding: utf-8 -*-
"""
peptide_create_plots.py
this is the main script for surfactant analysis

CREATED ON: 04/10/2020

AUTHOR(S):
    Bradley C. Dallin (brad.dallin@gmail.com)
    
** UPDATES **

TODO:

"""
##############################################################################
## IMPORTING MODULES
##############################################################################
import os
import numpy as np
import matplotlib.pyplot as plt
## FUNCTION TO SAVE AND LOAD PICKLE FILES
from MDDescriptors.application.sams.pickle_functions import load_pkl

class set_plot_defaults:
    r'''
    class adusts matplotlib defaults to more high quality images
    '''
    def __init__(self, pub_type = "jacs" ):
        '''
        '''    
        ## SET FONT PARAMETERS
        plt.rcParams['font.sans-serif'] = 'Arial'
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.size'] = 8.0
        
        ## SET GRID PARAMETERS
        plt.rcParams['axes.grid'] = True
    #    plt.rcParams['axes.grid'] = False
        plt.rcParams['grid.linestyle'] = ":"
        plt.rcParams['grid.linewidth'] = "0.5"
        plt.rcParams['grid.color'] = "grey"
        
        ## SET LEGEND PARAMETERS
        plt.rcParams['legend.loc'] = "best"
        plt.rcParams['legend.fancybox'] = True
        plt.rcParams['legend.facecolor'] = "white"
        plt.rcParams['legend.edgecolor'] = "black"
        
        if pub_type == "jacs":
            self.jacs()
        
    def jacs(self):
        # JACS format: single column 3.33 in. wide
        #              double column 4.167- 7 in. wide
        #              all 9.167 in. height (including caption)  
        #              font >=4.5 pt
        #              linewidth >=0.5 pt
        #              font style Helvetica or Arial  
        ## SET FIGURE DIMENSIONS
        self.width = 3.33 # in
        self.height = 0.75 * self.width # 4:3 aspect ratio
 
        ## LIST OF LINESTYLES
        self.linestyles = [ "-.",
                            "--",
                            "-" ]

        ## LIST OF LINESTYLES
        self.markers = [ "o",
                         "s",
                         "^",
                         "v",
                         ">",
                         "<",]
        
        ## LIST OF COLORS
        self.colors = [ "dimgrey",
                        "tomato",
                        "slateblue",
                        "darkseagreen",
                        "mediumturquoise",
                        "teal",
                        "darkorange",
                        "deepskyblue",
                        "plum",
                        "crimson",
                        "lightsalmon",
                        "orangered", ]

def plot_line( path_fig, 
               data,
               savefig = False ):
    r"""
    """
    ## SET PLOT DEFAULT
    plot_details = set_plot_defaults("jacs")    
    ## LOOP THROUGH PLOTS
    for title, y_data in data.items():
        ## CREATING LINE PLOT
        fig, ax = plt.subplots()
        fig.subplots_adjust( left = 0.15, bottom = 0.16, right = 0.99, top = 0.97 )
        ## ADD TITLE TO PLOT
        plt.title( title, loc = 'center' )
        ## ASSUMES TIME (100K PS)
        x = np.arange( 0, 1e5, 10 )
#        x = np.arange( 0, len(y_data), 1 )
        y = y_data[:-1]
#        y = y_data
        plt.plot( x,
                  y,
                  linestyle = '-',
                  linewidth = 1.5,
                  color = "dimgray" )
#        # SET X AND Y AXES
        ax.set_xlabel( "Sim. Time (ps)" )
        ax.set_xlim( -1e4, 1.1e5 )
        ax.set_xticks( np.arange( 0, 1.1e5, 2e4 ), minor = False )    # sets major ticks
        ax.set_xticks( np.arange( -1e4, 1.2e5, 2e4 ), minor = True )  # sets minor ticks
#        ax.set_xlabel( "Atom number" )
#        ax.set_xlim( -5, 55 )
#        ax.set_xticks( np.arange( 0, 60, 10 ), minor = False )    # sets major ticks
#        ax.set_xticks( np.arange( -5, 65, 10 ), minor = True )  # sets minor ticks
        ax.set_ylabel( "RMSD (nm)" )
        ax.set_ylim( -0.1, 1.1 )
        ax.set_yticks( np.arange( 0, 1.2, 0.2 ), minor = False )    # sets major ticks
        ax.set_yticks( np.arange( -0.1, 1.3, 0.2 ), minor = True )  # sets minor ticks
#        ax.set_ylabel( "RMSF (nm)" )
#        ax.set_ylim( 2.1, 3.5 )
#        ax.set_yticks( np.arange( 2.2, 3.6, 0.2 ), minor = False )    # sets major ticks
#        ax.set_yticks( np.arange( 2.1, 3.7, 0.2 ), minor = True )  # sets minor ticks
        
        fig.set_size_inches( plot_details.width, plot_details.height )
        fig.tight_layout()
        if savefig is True:
            print( "FIGURE SAVED TO: %s" % path_fig.format( title.lower() ) )
            fig.savefig( path_fig.format( title.lower() ), dpi = 300, facecolor = 'w', edgecolor = 'w' )        
