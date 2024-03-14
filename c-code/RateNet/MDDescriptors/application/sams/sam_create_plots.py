# -*- coding: utf-8 -*-
"""
sam_create_plots.py
script to create plots the SAM simulations

CREATED ON: 02/24/2020

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

##############################################################################
## FUNCTIONS & CLASSES
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
        self.n_linestyles = len(self.linestyles)

        ## LIST OF LINESTYLES
        self.markers = [ "o",
                         "s",
                         "^",
                         "v",
                         ">",
                         "<",]
        self.n_markers = len(self.markers)
        
        ## LIST OF COLORS
        self.colors = [ "dimgrey",
                        "slateblue",
                        "darkseagreen",
                        "tomato",
                        "mediumturquoise",
                        "teal",
                        "darkorange",
                        "deepskyblue",
                        "plum",
                        "crimson",
                        "lightsalmon",
                        "orangered", ]
        self.n_colors = len(self.colors)

def plot_line( path_fig, 
               data,
               plot_title = None,
               x_labels = None,
               x_label = r"x label",
               x_ticks = [],
               y_label = r"y label",
               y_ticks = [],
               ncol_legend = 1,
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
    ax.legend( loc = 'best', ncol = ncol_legend )
                
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
        y_min = np.round( y_ticks.min(), decimals = 4 )           # round to prevent precision error
        y_max = np.round( y_ticks.max() + y_diff, decimals = 4 )
        print( y_min, y_max )
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
    save_fig = False
    extension = ".png"
    plot_title = None
    subtract_normal = False
    
    ## OUTPUT DIRECTORY
    output_dir = r"C:\Users\bdallin\Box Sync\univ_of_wisc\manuscripts\mixed_polar_sams\raw_figures"    
        
    ## MAIN DIRECTORY
    main_dir = r"R:\simulations\polar_sams\unbiased"
    
    ## FILES
    path_fig = os.path.join( output_dir, r"hbonds_all_c13oh_mixed_composition" + extension )
    normal_dist_file = "sam_single_12x12_300K_dodecanethiol_tip4p_nvt_CHARMM36/output_files/sam_prod_triplet_angle_distribution.pkl"
    paths_to_files = { 
                       "CH3"   : { "None" : [ "sample1/sam_single_12x12_300K_dodecanethiol_tip4p_nvt_CHARMM36/output_files/sam_prod_num_hbonds_all.pkl", ],
                                  },
#                       "k=0.0" : { "None" : [ "sam_single_12x12_300K_C13OH_k0.0_tip4p_nvt_CHARMM36/output_files/sam_prod_triplet_angle_distribution.pkl",   ],
#                                  },                                  
#                       "k=0.1" : { "None" : [ "sam_single_12x12_300K_C13OH_k0.1_tip4p_nvt_CHARMM36/output_files/sam_prod_triplet_angle_distribution.pkl",   ],
#                                  },
#                       "k=0.2" : { "None" : [ "sam_single_12x12_300K_C13OH_k0.2_tip4p_nvt_CHARMM36/output_files/sam_prod_triplet_angle_distribution.pkl",   ],
#                                  },
#                       "k=0.3" : { "None" : [ "sam_single_12x12_300K_C13OH_k0.3_tip4p_nvt_CHARMM36/output_files/sam_prod_triplet_angle_distribution.pkl",   ],
#                                  },
#                       "k=0.4" : { "None" : [ "sam_single_12x12_300K_C13OH_k0.4_tip4p_nvt_CHARMM36/output_files/sam_prod_triplet_angle_distribution.pkl",   ],
#                                  },
#                       "k=0.5" : { "None" : [ "sam_single_12x12_300K_C13OH_k0.5_tip4p_nvt_CHARMM36/output_files/sam_prod_triplet_angle_distribution.pkl",   ],
#                                  },
#                       "k=0.6" : { "None" : [ "sam_single_12x12_300K_C13OH_k0.6_tip4p_nvt_CHARMM36/output_files/sam_prod_triplet_angle_distribution.pkl",   ],
#                                  },
#                       "k=0.7" : { "None" : [ "sam_single_12x12_300K_C13OH_k0.7_tip4p_nvt_CHARMM36/output_files/sam_prod_triplet_angle_distribution.pkl",   ],
#                                  },
#                       "k=0.8" : { "None" : [ "sam_single_12x12_300K_C13OH_k0.8_tip4p_nvt_CHARMM36/output_files/sam_prod_triplet_angle_distribution.pkl",   ],
#                                  },
#                       "k=0.9" : { "None" : [ "sam_single_12x12_300K_C13OH_k0.9_tip4p_nvt_CHARMM36/output_files/sam_prod_triplet_angle_distribution.pkl",   ],
#                                  },
#                       "k=1.0" : { "None" : [ "sam_single_12x12_300K_C13OH_tip4p_nvt_CHARMM36/output_files/sam_prod_triplet_angle_distribution.pkl",        ],
#                                  },
#                       r"$\sigma_{R}$=0.00" : { "None" : [ "sam_single_12x12_300K_dodecanethiol_tip4p_nvt_CHARMM36/output_files/sam_prod_hbonds_average.pkl",                       ],
#                                  },
#                       r"$\sigma_{R}$=0.25" : { "None" : [ "sam_single_12x12_checker_300K_dodecanethiol0.75_C13OH0.25_tip4p_nvt_CHARMM36/output_files/sam_prod_hbonds_average.pkl", ],
#                                               },
#                       r"$\sigma_{R}$=0.40" : { "None" : [ "sam_single_12x12_checker_300K_dodecanethiol0.6_C13OH0.4_tip4p_nvt_CHARMM36/output_files/sam_prod_hbonds_average.pkl",   ],
#                                               },
#                       r"$\sigma_{R}$=0.50" : { "None" : [ "sam_single_12x12_checker_300K_dodecanethiol0.5_C13OH0.5_tip4p_nvt_CHARMM36/output_files/sam_prod_hbonds_average.pkl",   ],
#                                               },
#                       r"$\sigma_{R}$=0.75" : { "None" : [ "sam_single_12x12_checker_300K_dodecanethiol0.25_C13OH0.75_tip4p_nvt_CHARMM36/output_files/sam_prod_hbonds_average.pkl", ],
#                                               },
#                       r"$\sigma_{R}$=1.00" : { "None" : [ "sam_single_12x12_300K_C13OH_tip4p_nvt_CHARMM36/output_files/sam_prod_hbonds_average.pkl",                               ],
#                                  },
                      }
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
                print( loaded_data )
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
        if subtract_normal is True:
            path_to_file = os.path.join( main_dir, filename )
            path_to_normal = os.path.join( main_dir, normal_dist_file )
            loaded_data = load_pkl( path_to_normal )
            y_norm = loaded_data[1]
            y -= y_norm
            
        data[legend_label] = np.array([ x, xerr, y, yerr ]).transpose()

#    # PLOT DENSITY PROFILE
#    plot_line( path_fig = path_fig, 
#               data = data,
#               plot_title = plot_title,
#               x_label = r"$\theta$ (degrees)", 
#               x_ticks = np.arange( 40, 200, 20 ),
#               y_label = r"$\it{p}(\it{\theta})$", 
#               y_ticks = np.arange( -0.001, 0.001, 0.0005 ),
#               ncol_legend = 3,
#               shaded_error = True,
#               savefig = save_fig )
    
    ## PLOT HBOND CONFIGURATIONS
#    plot_line( path_fig = os.path.join( output_dir, r"hbond_configurations" + extension ), 
#               data = data,
#               plot_title = r"$N \geq 4$",
#               x_labels = x_labels,
#               x_label = r"Atom groups", 
#               x_ticks = np.array(x),
#               y_label = r"$\it{p}(\it{N})$", 
#               y_ticks = np.arange( 0.22, 0.32, 0.02 ),
#               errorbars = True,
#               savefig = save_fig )

#    plot_bar( path_fig = os.path.join( output_dir, r"bulk_water_hbond_configurations" + extension ), 
#              data = data,
#              x_labels = x_labels,
#              y_label = r"$\it{p}(\it{N})$", 
#              y_ticks = np.arange( 0.0, 0.7, 0.1 ),
#              savefig = save_fig )
