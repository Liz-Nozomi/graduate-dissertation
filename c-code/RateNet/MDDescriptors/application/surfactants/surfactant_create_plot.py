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
import pickle
import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

## MDBUILDERS FUNCTIONS
from MDBuilder.core.check_tools import check_server_path

## LOAD PICKLE FUNCTION
def load_pkl( path_pkl ):
    r'''
    Function to load data from pickle file
    '''
    print( "LOADING PICKLE FROM %s" % ( path_pkl ) )
    with open( path_pkl, 'rb' ) as input:
        data = pickle.load( input )
        
    return data

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
                        "crimson", ]

def plot_line( path_fig, 
               data,
               plot_title = None,
               x_labels = None,
               x_label = r"x label",
               x_ticks = [],
               y_label = r"y label",
               y_ticks = [],
               errorbars = False,
               shaded_error = False,
               savefig = False ):
    r"""
    """
    ## SET PLOT DEFAULT
    plot_details = set_plot_defaults("jacs")
    
    ## EXTRACT LINE LABELS
    line_labels = list(data.keys())
        
    ## CREATING LINE PLOT
    fig, ax = plt.subplots()
    fig.subplots_adjust( left = 0.15, bottom = 0.16, right = 0.99, top = 0.97 )
    if plot_title is not None:
        plt.title( plot_title, loc = "center" )
            
    ## LOOP THROUGH LINES
    for ii, ll in enumerate( line_labels ):
        ## GATHER X AND Y LISTS/ARRAYS
        x = data[ll][:,0]
        xerr = data[ll][:,1]
        y = data[ll][:,2]
        yerr = data[ll][:,3]
        
        ## PLOT LINES        
        if errorbars is True:
            plt.plot( x, y,
                         marker = plot_details.markers[ii],
                         linestyle = '-',
                         linewidth = 1.5,
                         color = plot_details.colors[ii],
                         label = ll )
            
            plt.errorbar( x, y,
                             marker = 'None',
                             linestyle = 'None',
                             xerr = xerr,
                             yerr = yerr,
                             ecolor = plot_details.colors[ii],
                             elinewidth = 0.5,
                             capsize = 1,
                             capthick = 0.5 )
        elif shaded_error is True: 
            plt.plot( x, y,
                         linestyle = '-',
                         linewidth = 1.5,
                         color = plot_details.colors[ii],
                         label = ll )
            
            ## PLOT ERROR
            plt.plot( x, y + yerr,
                         linestyle = ':',
                         linewidth = 1.0,
                         color = plot_details.colors[ii] )
            plt.plot( x, y - yerr,
                         linestyle = ':',
                         linewidth = 1.0,
                         color = plot_details.colors[ii] )
            plt.fill_between( x, y + yerr, y - yerr,
                                           color = plot_details.colors[ii],
                                           alpha = 0.5, )
        else:
            plt.plot( x, y,
                         linestyle = '-',
                         linewidth = 1.5,
                         color = plot_details.colors[ii],
                         label = ll )

    ## SET LEGEND
    ax.legend( loc = 'best' )
                
    # SET X AND Y AXES
    ax.set_xlabel( x_label )
    if len(x_ticks) > 0:
        x_diff = x_ticks[1] - x_ticks[0]
        x_min = np.round( x_ticks.min(), decimals = 2 ) # round to prevent precision error
        x_max = np.round( x_ticks.max(), decimals = 2 )
        ax.set_xlim( x_min - 0.5*x_diff, x_max + 0.5*x_diff )
        ax.set_xticks( np.arange( x_min, x_max + x_diff, x_diff ), minor = False )                  # sets major ticks
        ax.set_xticks( np.arange( x_min - 0.5*x_diff, x_max + 1.5*x_diff, x_diff ), minor = True )  # sets minor ticks
        if x_labels is not None:
            ax.set_xticklabels( x_labels )
    ax.set_ylabel( y_label )
    if len(y_ticks) > 0:
        y_diff = y_ticks[1] - y_ticks[0]
        y_min = np.round( y_ticks.min(), decimals = 3 )           # round to prevent precision error
        y_max = np.round( y_ticks.max() + y_diff, decimals = 3 )
        ax.set_ylim( y_min - 0.5*y_diff, y_max + 0.5*y_diff )
        ax.set_yticks( np.arange( y_min, y_max+y_diff, y_diff ), minor = False )                           # sets major ticks
        ax.set_yticks( np.arange( y_min - 0.5*y_diff, y_max + 1.5*y_diff, y_diff ), minor = True )  # sets minor ticks
    
    fig.set_size_inches( plot_details.width, plot_details.height )
    fig.tight_layout()
    if savefig is True:
        print( "FIGURE SAVED TO: %s" % path_fig )
        fig.savefig( path_fig, dpi = 300, facecolor = 'w', edgecolor = 'w' )

def plot_bar( path_fig, 
              data,
              x_labels,
              plot_title = None,
              y_label = r"y label",
              y_ticks = [],
              savefig = False ):
    r"""
    """
    ## SET PLOT DEFAULT
    plot_details = set_plot_defaults("jacs")
    
    ## EXTRACT BAR LABELS
    bar_labels = list(data.keys())
    bar_width = 1 / float(len(bar_labels)+1)
    shift = np.arange( 0, 1. - bar_width, bar_width ) - ( 0.5 - bar_width )
    ## CREATING BAR PLOT
    fig, ax = plt.subplots()
    fig.subplots_adjust( left = 0.15, bottom = 0.16, right = 0.99, top = 0.97 )
    if plot_title is not None:
        plt.title( plot_title, loc = "center" )
            
    ## LOOP THROUGH BARS
    for ii, bl in enumerate( bar_labels ):
        ## GATHER X AND Y LISTS/ARRAYS
        x = data[bl][:,0] + shift[ii]
        y = data[bl][:,2]
        yerr = data[bl][:,3]
        
        ## PLOT DATA
        plt.bar( x, y,
                    linestyle = "None",
                    color = plot_details.colors[ii],
                    width = bar_width,
                    edgecolor = "black", 
                    linewidth = 0.5,
                    yerr = yerr,
                    ecolor = "black",
                    capsize = 2.0,
                    label = bl ) 

    ## PLOT LINE AT Y=0
    plt.plot( [ -0.5, len(x_labels)-0.5 ], 
              [ 0, 0 ],
              linestyle = "-",
              linewidth = 0.5,
              color = "black",
              )

    ## SET LEGEND
    ax.legend( ncol = 3, loc = "upper center" )
    # SET X AND Y AXES
    ax.set_xlim( -0.5, len(x_labels)-0.5 )
    ax.set_xticks( np.arange( 0, len(x_labels), 1 ), minor = False )      # sets major ticks
    ax.set_xticks( np.arange( -0.5, len(x_labels)+0.5, 1 ), minor = True )  # sets minor ticks
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

#%%
##############################################################################
## MAIN SCRIPT
##############################################################################
if __name__ == "__main__":
    ## REWRITE
    save_fig = True
    extension = ".png"
    plot_title = None
    
    ## OUTPUT DIRECTORY
    output_dir = r"C:\Users\bdallin\Documents\Box Sync\univ_of_wisc\manuscripts\hexyl_surfactants\raw_figures"

    ## MAIN DIRECTORY
    main_dir = r"R:\surfactants\simulations\unbiased"
    
    ## FILES
    ## HBOND CONFIGURATIONS
    path_fig = os.path.join( output_dir, r"surfactant_water_hbond_configurations_water_models" + extension )
#    plot_title = r"$R=Am+$"
    paths_to_files = { 
                       r"$N<1$"             : { "SPCE-Am+"   : [ "surfactant_300K_hexylammonium_spce_CHARMM36/output_files/surfactant_prod_hbonds_configurations_Nlt1_c5.pkl"  ],
                                                "SPCE-Gdm+"  : [ "surfactant_300K_hexylguanidinium_spce_CHARMM36/output_files/surfactant_prod_hbonds_configurations_Nlt1_c5.pkl"  ], 
                                                "TIP3P-Am+"  : [ "surfactant_300K_hexylammonium_tip3p_CHARMM36/output_files/surfactant_prod_hbonds_configurations_Nlt1_c5.pkl" ], 
                                                "TIP3P-Gdm+" : [ "surfactant_300K_hexylguanidinium_tip3p_CHARMM36/output_files/surfactant_prod_hbonds_configurations_Nlt1_c5.pkl" ],
                                                "TIP4P-Am+"  : [ "surfactant_300K_hexylammonium_tip4p_CHARMM36/output_files/surfactant_prod_hbonds_configurations_Nlt1_c5.pkl" ],
                                                "TIP4P-Gdm+" : [ "surfactant_300K_hexylguanidinium_tip4p_CHARMM36/output_files/surfactant_prod_hbonds_configurations_Nlt1_c5.pkl" ],
                                                "RAMAN-Am+"  : [ "raman_300K/output_files/raman_am_plus_hbonds_configurations_Nlt1.pkl", ],
                                                "RAMAN-Gdm+" : [ "raman_300K/output_files/raman_gdm_plus_hbonds_configurations_Nlt1.pkl", ],
                                               },
                       r"$1 \leq N \leq 3$" : { "SPCE-Am+"   : [ "surfactant_300K_hexylammonium_spce_CHARMM36/output_files/surfactant_prod_hbonds_configurations_1geNle3_c5.pkl"  ],
                                                "SPCE-Gdm+"  : [ "surfactant_300K_hexylguanidinium_spce_CHARMM36/output_files/surfactant_prod_hbonds_configurations_1geNle3_c5.pkl"  ], 
                                                "TIP3P-Am+"  : [ "surfactant_300K_hexylammonium_tip3p_CHARMM36/output_files/surfactant_prod_hbonds_configurations_1geNle3_c5.pkl" ], 
                                                "TIP3P-Gdm+" : [ "surfactant_300K_hexylguanidinium_tip3p_CHARMM36/output_files/surfactant_prod_hbonds_configurations_1geNle3_c5.pkl" ],
                                                "TIP4P-Am+"  : [ "surfactant_300K_hexylammonium_tip4p_CHARMM36/output_files/surfactant_prod_hbonds_configurations_1geNle3_c5.pkl" ],
                                                "TIP4P-Gdm+" : [ "surfactant_300K_hexylguanidinium_tip4p_CHARMM36/output_files/surfactant_prod_hbonds_configurations_1geNle3_c5.pkl" ],
                                                "RAMAN-Am+"  : [ "raman_300K/output_files/raman_am_plus_hbonds_configurations_1geNle3.pkl", ],
                                                "RAMAN-Gdm+" : [ "raman_300K/output_files/raman_gdm_plus_hbonds_configurations_1geNle3.pkl", ],
                                               },
                       r"$N \geq 4$"        : { "SPCE-Am+"   : [ "surfactant_300K_hexylammonium_spce_CHARMM36/output_files/surfactant_prod_hbonds_configurations_Nge4_c5.pkl"  ],
                                                "SPCE-Gdm+"  : [ "surfactant_300K_hexylguanidinium_spce_CHARMM36/output_files/surfactant_prod_hbonds_configurations_Nge4_c5.pkl"  ], 
                                                "TIP3P-Am+"  : [ "surfactant_300K_hexylammonium_tip3p_CHARMM36/output_files/surfactant_prod_hbonds_configurations_Nge4_c5.pkl" ], 
                                                "TIP3P-Gdm+" : [ "surfactant_300K_hexylguanidinium_tip3p_CHARMM36/output_files/surfactant_prod_hbonds_configurations_Nge4_c5.pkl" ],
                                                "TIP4P-Am+"  : [ "surfactant_300K_hexylammonium_tip4p_CHARMM36/output_files/surfactant_prod_hbonds_configurations_Nge4_c5.pkl" ],
                                                "TIP4P-Gdm+" : [ "surfactant_300K_hexylguanidinium_tip4p_CHARMM36/output_files/surfactant_prod_hbonds_configurations_Nge4_c5.pkl" ],
                                                "RAMAN-Am+"  : [ "raman_300K/output_files/raman_am_plus_hbonds_configurations_Nge4.pkl", ],
                                                "RAMAN-Gdm+" : [ "raman_300K/output_files/raman_gdm_plus_hbonds_configurations_Nge4.pkl", ],
                                               },
                      }
                       
    ## SURFACTANT FILES
    ## HBOND CONFIGURATIONS (N>=4)
#    path_fig = os.path.join( output_dir, r"hbond_configurations_Nge4_spce" + extension )
#    plot_title = r"$N \geq 4$"
#    paths_to_files = { 
#                       "CH3"  : { "CH3" : [ "surfactant_300K_heptane_spce_CHARMM36/output_files/surfactant_prod_hbonds_configurations_Nge4_ch3.pkl", ],
#                                  "C1"  : [ "surfactant_300K_heptane_spce_CHARMM36/output_files/surfactant_prod_hbonds_configurations_Nge4_c1.pkl",  ],
#                                  "C2"  : [ "surfactant_300K_heptane_spce_CHARMM36/output_files/surfactant_prod_hbonds_configurations_Nge4_c2.pkl",  ],
#                                  "C3"  : [ "surfactant_300K_heptane_spce_CHARMM36/output_files/surfactant_prod_hbonds_configurations_Nge4_c3.pkl",  ],
#                                  "C4"  : [ "surfactant_300K_heptane_spce_CHARMM36/output_files/surfactant_prod_hbonds_configurations_Nge4_c4.pkl",  ],
#                                  "C5"  : [ "surfactant_300K_heptane_spce_CHARMM36/output_files/surfactant_prod_hbonds_configurations_Nge4_c5.pkl",  ],
#                                  "R"   : [ "surfactant_300K_heptane_spce_CHARMM36/output_files/surfactant_prod_hbonds_configurations_Nge4_r.pkl",   ],
#                                 },
#                       "Am"   : { "CH3" : [ "surfactant_300K_hexylamine_spce_CHARMM36/output_files/surfactant_prod_hbonds_configurations_Nge4_ch3.pkl", ],
#                                  "C1"  : [ "surfactant_300K_hexylamine_spce_CHARMM36/output_files/surfactant_prod_hbonds_configurations_Nge4_c1.pkl",  ],
#                                  "C2"  : [ "surfactant_300K_hexylamine_spce_CHARMM36/output_files/surfactant_prod_hbonds_configurations_Nge4_c2.pkl",  ],
#                                  "C3"  : [ "surfactant_300K_hexylamine_spce_CHARMM36/output_files/surfactant_prod_hbonds_configurations_Nge4_c3.pkl",  ],
#                                  "C4"  : [ "surfactant_300K_hexylamine_spce_CHARMM36/output_files/surfactant_prod_hbonds_configurations_Nge4_c4.pkl",  ],
#                                  "C5"  : [ "surfactant_300K_hexylamine_spce_CHARMM36/output_files/surfactant_prod_hbonds_configurations_Nge4_c5.pkl",  ],
#                                  "R"   : [ "surfactant_300K_hexylamine_spce_CHARMM36/output_files/surfactant_prod_hbonds_configurations_Nge4_r.pkl",   ],
#                                 },
#                       "Am+"  : { "CH3" : [ "surfactant_300K_hexylammonium_spce_CHARMM36/output_files/surfactant_prod_hbonds_configurations_Nge4_ch3.pkl", ],
#                                  "C1"  : [ "surfactant_300K_hexylammonium_spce_CHARMM36/output_files/surfactant_prod_hbonds_configurations_Nge4_c1.pkl",  ],
#                                  "C2"  : [ "surfactant_300K_hexylammonium_spce_CHARMM36/output_files/surfactant_prod_hbonds_configurations_Nge4_c2.pkl",  ],
#                                  "C3"  : [ "surfactant_300K_hexylammonium_spce_CHARMM36/output_files/surfactant_prod_hbonds_configurations_Nge4_c3.pkl",  ],
#                                  "C4"  : [ "surfactant_300K_hexylammonium_spce_CHARMM36/output_files/surfactant_prod_hbonds_configurations_Nge4_c4.pkl",  ],
#                                  "C5"  : [ "surfactant_300K_hexylammonium_spce_CHARMM36/output_files/surfactant_prod_hbonds_configurations_Nge4_c5.pkl",  ],
#                                  "R"   : [ "surfactant_300K_hexylammonium_spce_CHARMM36/output_files/surfactant_prod_hbonds_configurations_Nge4_r.pkl",   ],
#                                 },
#                       "Gdm+" : { "CH3" : [ "surfactant_300K_hexylguanidinium_spce_CHARMM36/output_files/surfactant_prod_hbonds_configurations_Nge4_ch3.pkl", ],
#                                  "C1"  : [ "surfactant_300K_hexylguanidinium_spce_CHARMM36/output_files/surfactant_prod_hbonds_configurations_Nge4_c1.pkl",  ],
#                                  "C2"  : [ "surfactant_300K_hexylguanidinium_spce_CHARMM36/output_files/surfactant_prod_hbonds_configurations_Nge4_c2.pkl",  ],
#                                  "C3"  : [ "surfactant_300K_hexylguanidinium_spce_CHARMM36/output_files/surfactant_prod_hbonds_configurations_Nge4_c3.pkl",  ],
#                                  "C4"  : [ "surfactant_300K_hexylguanidinium_spce_CHARMM36/output_files/surfactant_prod_hbonds_configurations_Nge4_c4.pkl",  ],
#                                  "C5"  : [ "surfactant_300K_hexylguanidinium_spce_CHARMM36/output_files/surfactant_prod_hbonds_configurations_Nge4_c5.pkl",  ],
#                                  "R"   : [ "surfactant_300K_hexylguanidinium_spce_CHARMM36/output_files/surfactant_prod_hbonds_configurations_Nge4_r.pkl",   ],
#                                 },
#                       }
    
    ## BULK FILES
#    path_fig = os.path.join( output_dir, r"bulk_water_hbond_configurations" + extension )
#    paths_to_files = { 
#                       r"$N<1$"        : { "SPCE"   : [ "bulk_300K_spce/output_files/bulk_prod_0_hbonds_configurations_Nlt1.pkl",
#                                                        "bulk_300K_spce/output_files/bulk_prod_1_hbonds_configurations_Nlt1.pkl",
#                                                        "bulk_300K_spce/output_files/bulk_prod_2_hbonds_configurations_Nlt1.pkl",
#                                                        "bulk_300K_spce/output_files/bulk_prod_3_hbonds_configurations_Nlt1.pkl",
#                                                        "bulk_300K_spce/output_files/bulk_prod_4_hbonds_configurations_Nlt1.pkl",   ],
#                                           "TIP3P"  : [ "bulk_300K_tip3p/output_files/bulk_prod_0_hbonds_configurations_Nlt1.pkl",
#                                                        "bulk_300K_tip3p/output_files/bulk_prod_1_hbonds_configurations_Nlt1.pkl",
#                                                        "bulk_300K_tip3p/output_files/bulk_prod_2_hbonds_configurations_Nlt1.pkl",
#                                                        "bulk_300K_tip3p/output_files/bulk_prod_3_hbonds_configurations_Nlt1.pkl",
#                                                        "bulk_300K_tip3p/output_files/bulk_prod_4_hbonds_configurations_Nlt1.pkl",  ],
#                                           "TIP4P"  : [ "bulk_300K_tip4p/output_files/bulk_prod_0_hbonds_configurations_Nlt1.pkl",
#                                                        "bulk_300K_tip4p/output_files/bulk_prod_1_hbonds_configurations_Nlt1.pkl",
#                                                        "bulk_300K_tip4p/output_files/bulk_prod_2_hbonds_configurations_Nlt1.pkl",
#                                                        "bulk_300K_tip4p/output_files/bulk_prod_3_hbonds_configurations_Nlt1.pkl",
#                                                        "bulk_300K_tip4p/output_files/bulk_prod_4_hbonds_configurations_Nlt1.pkl",  ],
#                                           "AMOEBA" : [ "bulk_300K_amoeba/output_files/bulk_prod_0_hbonds_configurations_Nlt1.pkl",
#                                                        "bulk_300K_amoeba/output_files/bulk_prod_1_hbonds_configurations_Nlt1.pkl",
#                                                        "bulk_300K_amoeba/output_files/bulk_prod_2_hbonds_configurations_Nlt1.pkl",
#                                                        "bulk_300K_amoeba/output_files/bulk_prod_3_hbonds_configurations_Nlt1.pkl",
#                                                        "bulk_300K_amoeba/output_files/bulk_prod_4_hbonds_configurations_Nlt1.pkl", ],
#                                           "RAMAN"  : [ "raman_300K/output_files/raman_bulk_hbonds_configurations_Nlt1.pkl", ],
#                                          },
#                       r"$1 \leq N<4$" : { "SPCE" :    [ "bulk_300K_spce/output_files/bulk_prod_0_hbonds_configurations_1geNle3.pkl",
#                                                         "bulk_300K_spce/output_files/bulk_prod_1_hbonds_configurations_1geNle3.pkl",
#                                                         "bulk_300K_spce/output_files/bulk_prod_2_hbonds_configurations_1geNle3.pkl",
#                                                         "bulk_300K_spce/output_files/bulk_prod_3_hbonds_configurations_1geNle3.pkl",
#                                                         "bulk_300K_spce/output_files/bulk_prod_4_hbonds_configurations_1geNle3.pkl",  ],
#                                           "TIP3P"  :  [ "bulk_300K_tip3p/output_files/bulk_prod_0_hbonds_configurations_1geNle3.pkl",
#                                                         "bulk_300K_tip3p/output_files/bulk_prod_1_hbonds_configurations_1geNle3.pkl",
#                                                         "bulk_300K_tip3p/output_files/bulk_prod_2_hbonds_configurations_1geNle3.pkl",
#                                                         "bulk_300K_tip3p/output_files/bulk_prod_3_hbonds_configurations_1geNle3.pkl",
#                                                         "bulk_300K_tip3p/output_files/bulk_prod_4_hbonds_configurations_1geNle3.pkl", ],
#                                           "TIP4P"  :  [ "bulk_300K_tip4p/output_files/bulk_prod_0_hbonds_configurations_1geNle3.pkl",
#                                                         "bulk_300K_tip4p/output_files/bulk_prod_1_hbonds_configurations_1geNle3.pkl",
#                                                         "bulk_300K_tip4p/output_files/bulk_prod_2_hbonds_configurations_1geNle3.pkl",
#                                                         "bulk_300K_tip4p/output_files/bulk_prod_3_hbonds_configurations_1geNle3.pkl",
#                                                         "bulk_300K_tip4p/output_files/bulk_prod_4_hbonds_configurations_1geNle3.pkl", ],
#                                           "AMOEBA" : [ "bulk_300K_amoeba/output_files/bulk_prod_0_hbonds_configurations_1geNle3.pkl",
#                                                        "bulk_300K_amoeba/output_files/bulk_prod_1_hbonds_configurations_1geNle3.pkl",
#                                                        "bulk_300K_amoeba/output_files/bulk_prod_2_hbonds_configurations_1geNle3.pkl",
#                                                        "bulk_300K_amoeba/output_files/bulk_prod_3_hbonds_configurations_1geNle3.pkl",
#                                                        "bulk_300K_amoeba/output_files/bulk_prod_4_hbonds_configurations_1geNle3.pkl", ],
#                                           "RAMAN"  : [ "raman_300K/output_files/raman_bulk_hbonds_configurations_1geNle3.pkl", ],
#                                          },
#                       r"$N \geq 4$"   : { "SPCE"   : [ "bulk_300K_spce/output_files/bulk_prod_0_hbonds_configurations_Nge4.pkl",
#                                                        "bulk_300K_spce/output_files/bulk_prod_1_hbonds_configurations_Nge4.pkl",
#                                                        "bulk_300K_spce/output_files/bulk_prod_2_hbonds_configurations_Nge4.pkl",
#                                                        "bulk_300K_spce/output_files/bulk_prod_3_hbonds_configurations_Nge4.pkl",
#                                                        "bulk_300K_spce/output_files/bulk_prod_4_hbonds_configurations_Nge4.pkl",   ],
#                                           "TIP3P"  : [ "bulk_300K_tip3p/output_files/bulk_prod_0_hbonds_configurations_Nge4.pkl",
#                                                        "bulk_300K_tip3p/output_files/bulk_prod_1_hbonds_configurations_Nge4.pkl",
#                                                        "bulk_300K_tip3p/output_files/bulk_prod_2_hbonds_configurations_Nge4.pkl",
#                                                        "bulk_300K_tip3p/output_files/bulk_prod_3_hbonds_configurations_Nge4.pkl",
#                                                        "bulk_300K_tip3p/output_files/bulk_prod_4_hbonds_configurations_Nge4.pkl",  ],
#                                           "TIP4P"  : [ "bulk_300K_tip4p/output_files/bulk_prod_0_hbonds_configurations_Nge4.pkl",
#                                                        "bulk_300K_tip4p/output_files/bulk_prod_1_hbonds_configurations_Nge4.pkl",
#                                                        "bulk_300K_tip4p/output_files/bulk_prod_2_hbonds_configurations_Nge4.pkl",
#                                                        "bulk_300K_tip4p/output_files/bulk_prod_3_hbonds_configurations_Nge4.pkl",
#                                                        "bulk_300K_tip4p/output_files/bulk_prod_4_hbonds_configurations_Nge4.pkl",  ],
#                                           "AMOEBA" : [ "bulk_300K_amoeba/output_files/bulk_prod_0_hbonds_configurations_Nge4.pkl",
#                                                        "bulk_300K_amoeba/output_files/bulk_prod_1_hbonds_configurations_Nge4.pkl",
#                                                        "bulk_300K_amoeba/output_files/bulk_prod_2_hbonds_configurations_Nge4.pkl",
#                                                        "bulk_300K_amoeba/output_files/bulk_prod_3_hbonds_configurations_Nge4.pkl",
#                                                        "bulk_300K_amoeba/output_files/bulk_prod_4_hbonds_configurations_Nge4.pkl", ],
#                                           "RAMAN"  : [ "raman_300K/output_files/raman_bulk_hbonds_configurations_Nge4.pkl", ],
#                                     },
#                      }

    ## INTERFACIAL RDF
#    path_fig = os.path.join( output_dir, r"interfacial_rdf_r" + extension )
#    plot_title = r"$R$"
#    paths_to_files = { 
#                       "CH3"  : { "None" : [ "surfactant_300K_heptane_spce_CHARMM36/output_files/surfactant_prod_interfacial_rdf_r.pkl", ],
#                                 },
#                       "Am"   : { "None" : [ "surfactant_300K_hexylamine_spce_CHARMM36/output_files/surfactant_prod_interfacial_rdf_r.pkl", ],
#                                 },
#                       "Am+"  : { "None" : [ "surfactant_300K_hexylammonium_spce_CHARMM36/output_files/surfactant_prod_interfacial_rdf_r.pkl", ],
#                                 },
#                       "Gdm+" : { "None" : [ "surfactant_300K_hexylguanidinium_spce_CHARMM36/output_files/surfactant_prod_interfacial_rdf_r.pkl", ],
#                                 },
#                       }

  
    data = {}
    ## LOOP THROUGH DATA TYPE
    for legend_label, sub_data in paths_to_files.items():
        x_labels = list(sub_data.keys())
        if 'None' not in x_labels:
            y = []
            yerr = []
            x = list( range( 0, len(x_labels) ) )
            xerr = [ 0 ] * len(x)
        
        ## LOOP THROUGH SUB DATA
        for axis_label, file_list in sub_data.items():
            ## LOOP THROUGH FILENAMES
            for ii, filename in enumerate(file_list):
                path_to_file = os.path.join( main_dir, filename )
                loaded_data = load_pkl( path_to_file )
                if ii < 1:
                    tmp_data = [loaded_data]
                else:
                    tmp_data.append( loaded_data )
            if 'None' not in x_labels:
                y.append( np.mean(tmp_data) )
                yerr.append( np.std(tmp_data) )
            else:
                xy_avg = np.mean( tmp_data, axis = 0 ).transpose()
                xy_std = np.std( tmp_data, axis = 0 ).transpose()
                x = xy_avg[:,0]
                xerr = xy_std[:,0]
                y = xy_avg[:,1]
                yerr = xy_std[:,1]
                
        data[legend_label] = np.array([ x, xerr, y, yerr ]).transpose()

    ## PLOT INTERFACIAL RDF
#    plot_line( path_fig = path_fig, 
#               data = data,
#               plot_title = plot_title,
#               x_label = r"r (nm)", 
#               x_ticks = np.arange( 0, 1.0, 0.2 ),
#               y_label = r"g(r)", 
#               y_ticks = np.arange( 0.0, 4.0, 0.5 ),
#               shaded_error = True,
#               savefig = save_fig )
    
#    # PLOT HBOND CONFIGURATIONS
#    plot_line( path_fig = path_fig, 
#               data = data,
#               plot_title = plot_title,
#               x_labels = x_labels,
#               x_label = r"Atom groups", 
#               x_ticks = np.array(x),
#               y_label = r"$\it{p}(\it{N})$", 
#               y_ticks = np.arange( 0.10, 0.35, 0.05 ),
#               errorbars = True,
#               savefig = save_fig )

    plot_bar( path_fig = path_fig, 
              data = data,
              plot_title = plot_title,
              x_labels = x_labels,
              y_label = r"$\it{p}(\it{N})$", 
              y_ticks = np.arange( 0.0, 1.0, 0.2 ),
              savefig = save_fig )
