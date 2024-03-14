"""
plotting.py
this is the main script sam plotting functions

CREATED ON: 02/28/2020

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
import mdtraj as md
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

##############################################################################
## CLASSES AND FUNCITONS
##############################################################################
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
                         "^" ]
        
        ## LIST OF COLORS
        self.colors = [ "dimgrey",
                        "tomato",
                        "slateblue" ]

#        # LIST OF COLORS
#        self.colors = [ "dimgrey",
#                        "slateblue",
#                        "darkseagreen",
#                        "tomato" ] 

        ## LIST OF COLORS
#        self.colors = [ "dimgrey",
#                        "deepskyblue",
#                        "slateblue",
#                        "darkseagreen",
#                        "tomato" ]

        ## LIST OF COLORS
#        self.colors = [ "dimgrey",
#                        "rosybrown",
#                        "deepskyblue",
#                        "slateblue",
#                        "navy",
#                        "darkseagreen",
#                        "forestgreen",
#                        "tomato",
#                        "red" ]
    

def plot_data( path_fig, 
               data_obj, 
               key,
               x_label = r"x label",
               x_ticks = [],
               y_label = r"y label",
               y_ticks = [],
               line_plot = True,
               add_line = False,
               savefig = False ):
    r"""
    """
    ## SET PLOT DEFAULT
    plot_details = set_plot_defaults("jacs")
    
    ## EXTRACT GROUP LISTS
    labels = list(data_obj.keys())
    group_labels = list(data_obj[labels[0]].keys())
    
    ## LOOP THROUGH LABELS
    for gl in group_labels:
        ## CREATE LINE PLOT
        fig, ax = plt.subplots()
        fig.subplots_adjust( left = 0.15, bottom = 0.16, right = 0.99, top = 0.97 )
        ii = 0
        for ll in labels:
            x = data_obj[ll][gl][key][:,0]
            y = data_obj[ll][gl][key][:,1]
        
            ## PLOT DATA                
            plt.plot( x, y,
                         linestyle = "-",
                         linewidth = 1.5,
                         color = plot_details.colors[ii],
                         label = ll )
            ii += 1
            
        if add_line is True:
            x_line = data_obj[ll][gl]['add_line'][0,:]
            y_line = data_obj[ll][gl]['add_line'][1,:]
            ## PLOT DATA                
            plt.plot( x_line, y_line,
                              linestyle = "-",
                              linewidth = 1.5,
                              color = "black",
                              label = None )
        ## SET LEGEND
        ax.legend()
        
        ## SET TITLE
        plt.title( gl, loc = "center" )
        
        # SET X AND Y AXES
        ax.set_xlabel( x_label )
        if len(x_ticks) > 0:
            x_diff = x_ticks[1] - x_ticks[0]
            x_min = x_ticks.min()
            x_max = x_ticks.max() + x_diff
            ax.set_xlim( x_min - 0.5*x_diff, x_max + 0.5*x_diff )
            ax.set_xticks( np.arange( x_min, x_max + x_diff, x_diff ), minor = False )                  # sets major ticks
            ax.set_xticks( np.arange( x_min - 0.5*x_diff, x_max + 1.5*x_diff, x_diff ), minor = True )  # sets minor ticks
        ax.set_ylabel( y_label )
        if len(y_ticks) > 0:
            y_diff = y_ticks[1] - y_ticks[0]
            y_min = y_ticks.min()
            y_max = y_ticks.max() + y_diff
            ax.set_ylim( y_min - 0.5*y_diff, y_max + 0.5*y_diff )
            ax.set_yticks( np.arange( y_min, y_max + y_diff, y_diff ), minor = False )                  # sets major ticks
            ax.set_yticks( np.arange( y_min - 0.5*y_diff, y_max + 1.5*y_diff, y_diff ), minor = True )  # sets minor ticks
    
        fig.set_size_inches( plot_details.width, plot_details.height )
        fig.tight_layout()
        if savefig is True:
            print( "FIGURE SAVED TO: %s" % path_fig )
            fig.savefig( path_fig, dpi = 300, facecolor = 'w', edgecolor = 'w' )

def plot_hbond_config( path_fig, 
                       data_obj, 
                       key,
                       y_label = r"y label",
                       y_ticks = [],
                       line_plot = True,
                       add_line = False,
                       savefig = False ):
    r"""
    """
    ## SET PLOT DEFAULT
    plot_details = set_plot_defaults("jacs")
    
    ## EXTRACT HBOND CONFIGURATION DATA
    data = data_obj[key]
    
    ## EXTRACT LINE LABELS
    line_labels = list(data.keys())
    
    ## EXTRACT X LABELS
    x_labels = list(data[line_labels[0]].keys())
    x = np.arange( 0, len(x_labels), 1 )
    
    ## EXTRACT PLOT LABELS
    plot_labels = list(data[line_labels[0]][x_labels[0]].keys())
    
    ## LOOP THROUGH PLOTS
    for pl in plot_labels:
        ## CREATE NEW LINE PLOT
        fig, ax = plt.subplots()
        fig.subplots_adjust( left = 0.15, bottom = 0.16, right = 0.99, top = 0.97 )   
        
        ## LOOP THROUGH LINES
        ii = 0
        for ll in line_labels:
            ## CREATE EMPTY Y LIST
            y = []
            ## LOOP THROUGH X LABELS
            for xl in x_labels:
                y.append( data[ll][xl][pl] )

            ## PLOT DATA          
            plt.plot( x, y,
                         linestyle = "-",
                         linewidth = 1.5,
                         color = plot_details.colors[ii],
                         label = ll )
            ii += 1
                
        ## SET LEGEND
        ax.legend()
        
        ## SET TITLE
        plt.title( pl, loc = "center" )
        
        # SET X AND Y AXES
        ax.set_xlabel( "Atom groups" )
#        ax.set_xlim( -0.5, float(len(x))+0.5 )
        ax.set_xticks( np.arange( 0, len(x), 1 ), minor = False )      # sets major ticks
        ax.set_xticks( np.arange( -0.5, len(x)+0.5, 1 ), minor = True )  # sets minor ticks
        ax.set_xticklabels( x_labels, rotation = 45 )
        ax.set_ylabel( y_label )
        if len(y_ticks) > 0:
            y_diff = y_ticks[1] - y_ticks[0]
            y_min = y_ticks.min()
            y_max = y_ticks.max() + y_diff
#            ax.set_ylim( y_min - 0.5*y_diff, y_max + 0.5*y_diff )
            ax.set_yticks( np.arange( y_min, y_max + y_diff, y_diff ), minor = False )                  # sets major ticks
            ax.set_yticks( np.arange( y_min - 0.5*y_diff, y_max + 1.5*y_diff, y_diff ), minor = True )  # sets minor ticks
    
#        fig.set_size_inches( plot_details.width, plot_details.height )
#        fig.tight_layout()
        if savefig is True:
            print( "FIGURE SAVED TO: %s" % path_fig )
            fig.savefig( path_fig, dpi = 300, facecolor = 'w', edgecolor = 'w' )

def plot_bar( path_fig, 
              data_obj,
              key,
              y_label = r"y label",
              y_ticks = [],
              savefig = False ):
    r"""
    """
    ## SET PLOT DEFAULT
    plot_details = set_plot_defaults("jacs")
    
    ## EXTRACT HBOND CONFIGURATION DATA
    data = data_obj[key]
    
    ## EXTRACT X LABELS
    x_labels = list(data.keys())
    x = np.arange( 0, len(x_labels), 1 )
    bar_width = 1 / float(len(x) + 1)
    shift = np.arange( 0, 1. - bar_width, bar_width ) - ( 0.5 - bar_width )
    
    ## EXTRACT BAR LABELS
    bar_labels = list(data[x_labels[0]].keys())
    
    ## CREATING BAR PLOT
    fig, ax = plt.subplots()
    fig.subplots_adjust( left = 0.15, bottom = 0.16, right = 0.99, top = 0.97 )
            
    ## LOOP THROUGH LINES
    ii = 0
    add_label = True
    for xl in x_labels:
        ## CREATE EMPTY Y LIST
        y = []
        ## LOOP THROUGH X LABELS
        jj = 0
        for bl in bar_labels:
            y = data[xl][bl]
            if add_label is True:
                label = bl
            else: 
                label = None
            ## PLOT DATA          
            plt.bar( x[ii] + shift[jj], 
                     y, 
                     linestyle = "None",
                     color = plot_details.colors[jj],
                     width = bar_width,
                     edgecolor = "black", 
                     linewidth = 0.5,
                     label = label )
            
            jj += 1
        ## TURN LABELS OFF AFTER FIRST TIME THROUGH
        add_label = False
        ii += 1    

    ## PLOT LINE AT Y=0
    plt.plot( [ x[0]-1, x[-1]+1 ], 
              [ 0, 0 ],
              linestyle = "-",
              linewidth = 0.5,
              color = "black",
              )

    ## SET LEGEND
    ax.legend( ncol = 3, loc = "upper center" )
                
    # SET X AND Y AXES
    ax.set_xlim( -0.5, len(x)-0.5 )
    ax.set_xticks( np.arange( 0, len(x), 1 ), minor = False )      # sets major ticks
    ax.set_xticks( np.arange( -0.5, len(x)+0.5, 1 ), minor = True )  # sets minor ticks
    ax.set_xticklabels( x_labels, rotation = 45 )
    ax.set_ylabel( y_label )
    if len(y_ticks) > 0:
        y_diff = y_ticks[1] - y_ticks[0]
        y_min = y_ticks.min()
        y_max = y_ticks.max() + y_diff
        ax.set_ylim( y_min - 0.5*y_diff, y_max + 0.5*y_diff )
        ax.set_yticks( np.arange( y_min, y_max + y_diff, y_diff ), minor = False )                  # sets major ticks
        ax.set_yticks( np.arange( y_min - 0.5*y_diff, y_max + 1.5*y_diff, y_diff ), minor = True )  # sets minor ticks
    
    fig.set_size_inches( plot_details.width, plot_details.height )
    fig.tight_layout()
    if savefig is True:
        print( "FIGURE SAVED TO: %s" % path_fig )
        fig.savefig( path_fig, dpi = 300, facecolor = 'w', edgecolor = 'w' )
