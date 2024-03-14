# -*- coding: utf-8 -*-
"""
surfactant_analysis.py
this is the main script for surfactant analysis

CREATED ON: 02/24/2020

AUTHOR(S):
    Bradley C. Dallin (brad.dallin@gmail.com)
    
** UPDATES **

TODO:

"""
##############################################################################
## IMPORTING MODULES
##############################################################################
import os, sys
import numpy as np
import matplotlib.pyplot as plt
## FUNCTION TO SAVE AND LOAD PICKLE FILES
from MDDescriptors.application.sams.pickle_functions import load_pkl
## PLOT DATA FUNCTION
from MDDescriptors.application.sams.sam_create_plots import set_plot_defaults
##############################################################################
## FUNCTIONS
##############################################################################
def plot_data( data,
               path_fig = None,
               **kwargs ):
    r"""
    """
    ## SET PLOT DEFAULT
    plot_details = set_plot_defaults("jacs")
        
    ## CREATING LINE PLOT
    fig, ax = plt.subplots()
    fig.subplots_adjust( left = 0.15, bottom = 0.16, right = 0.99, top = 0.97 )
    ## ADD TITLE
    if "plot_title" in data:
        if data["plot_title"] is not None:
            plt.title( data["plot_title"], loc = "center" )
    # SET X AND Y LABELS
    if "x_label" in data:
        ax.set_xlabel( data["x_label"] )
    else:
        ax.set_xlabel( "x label" )
    if "y_label" in data:
        ax.set_ylabel( data["y_label"] )
    else:
        ax.set_ylabel( "y label" )
    ## PLOT DATA
    ii = 0
    for line_label, line_data in data["data"].items():
        x = line_data["x"]
        y, y_err = [], []        
        for y_data in line_data["y"]:
            y.append( np.mean(y_data) )
            y_err.append( np.std(y_data) )
        ## CREATE LINE PLOT
        plt.plot( x, y,
                  marker = plot_details.markers[ii % plot_details.n_markers],
                  linestyle = "-",
                  linewidth = 1.5,
                  color = plot_details.colors[ii % plot_details.n_colors],
                  label = line_label
                  )
        ## CREATE ERROR BAR PLOT
        plt.errorbar( x, y,
                      marker = 'None',
                      linestyle = "None",
                      yerr = y_err,
                      ecolor = plot_details.colors[ii % plot_details.n_colors],
                      elinewidth = 0.5,
                      capsize = 1,
                      capthick = 0.5
                      )
#        if ii < 1:
#            xmin, xmax = np.min(x), np.max(x)
#            ymin, ymax = np.min(y), np.max(y)
#        else:
#            if xmin > np.min(x):
#                xmin = np.min(x)
#            if xmax < np.max(x):
#                xmax = np.max(x)
#            if ymin > np.min(y):
#                ymin = np.min(y)
#            if ymax < np.max(y):
#                ymax = np.max(y)            
        ## ADD ONE TO INCREMENT
        ii += 1
    ## SET LEGEND
    if "ncol_legend" in kwargs:
        ax.legend( loc = 'best', ncol = kwargs["ncol_legend"] )
    else:
        ax.legend( loc = 'best', ncol = 1 )
    ## SET X AND Y TICKS
    if "x_ticks" in data:
        decimal = 2
        x_diff = np.round( data["x_ticks"][1] - data["x_ticks"][0], decimals = decimal )
        x_min = np.round( data["x_ticks"].min(), decimals = decimal )
        x_max = np.round( data["x_ticks"].max() + x_diff, decimals = decimal )
        x_lower = np.round( x_min - 0.5*x_diff, decimals = decimal )
        x_upper = np.round( x_max + 0.5*x_diff, decimals = decimal )        
        ax.set_xlim( x_lower, x_upper )
        ax.set_xticks( np.arange( x_min, np.round( x_max + x_diff, decimals = decimal ), x_diff ), minor = False )       # sets major ticks
        ax.set_xticks( np.arange( x_lower, np.round( x_max + 1.5*x_diff, decimals = decimal ), x_diff ), minor = True )  # sets minor ticks
    if "y_ticks" in data:
        decimal = 4
        y_diff = np.round( data["y_ticks"][1] - data["y_ticks"][0], decimals = decimal )
        y_min = np.round( data["y_ticks"].min(), decimals = decimal )
        y_max = np.round( data["y_ticks"].max() + y_diff, decimals = decimal )
        y_lower = np.round( y_min - 0.5*y_diff, decimals = decimal )
        y_upper = np.round( y_max + 0.5*y_diff, decimals = decimal )        
        ax.set_ylim( y_lower, y_upper )
        ax.set_yticks( np.arange( y_min, np.round( y_max + y_diff, decimals = decimal ), y_diff ), minor = False )       # sets major ticks
        ax.set_yticks( np.arange( y_lower, np.round( y_max + 1.5*y_diff, decimals = decimal ), y_diff ), minor = True )  # sets minor ticks
    ## SET FIGURE SIZE AND LAYOUT
    fig.set_size_inches( plot_details.width, plot_details.height )
    fig.tight_layout()
    if path_fig is not None:
        print( "FIGURE SAVED TO: %s" % path_fig )
        fig.savefig( path_fig, dpi = 300, facecolor = 'w', edgecolor = 'w' )
    

#%%
##############################################################################
## MAIN SCRIPT
##############################################################################
if __name__ == "__main__":
    ## VARIABLES
    want_overwrite     = False
    subtract_reference = False
    reference_line     = [ "CH3", ] # "bulk"
    
    ## OUTPUT DIRECTORY
    filename    = r"hydration_residence_time_charge_scale"
    input_dir   = r"C:\Users\bdallin\Box Sync\univ_of_wisc\manuscripts\mixed_polar_sams\raw_data"
    input_file  = os.path.join( input_dir, filename + ".pkl" )
    output_dir  = r"C:\Users\bdallin\Box Sync\univ_of_wisc\manuscripts\mixed_polar_sams\raw_figures"
    output_file_png = os.path.join( output_dir, filename + ".png" )
    output_file_svg = os.path.join( output_dir, filename + ".svg" )
    ## CHECK FOR INPUT DIRECTORY 
    if os.path.exists(input_file) is False:
        sys.exit("Input file does not exist")
    ## CREATE OUTPUT DIRECTORY
    if os.path.exists(output_dir) is False:
        os.mkdir( output_dir )
    ## LOAD DATA FROM PKL
    data = load_pkl( input_file )
    ## PLOT DATA
    plot_data( data, path_fig = output_file_png, ncol_legend = 2 )
    plot_data( data, path_fig = output_file_svg, ncol_legend = 2 )
        