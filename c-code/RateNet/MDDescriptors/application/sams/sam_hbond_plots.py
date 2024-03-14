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
        self.markers = [ "s",
                         "s",
                         "s",
                         "o",
                         "o",
                         "o",]
        
        ## LIST OF COLORS
        self.colors = [ "slateblue",
                        "darkseagreen",
                        "tomato",
                        "slateblue",
                        "darkseagreen",
                        "tomato", ]

def plot_line( path_fig, 
               data,
               plot_title = None,
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
        y = data[ll][:,1]
        yerr = data[ll][:,2]
        
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
        x_min = np.round( x_ticks.min(), decimals = 0 ) # round to prevent precision error
        x_max = np.round( x_ticks.max(), decimals = 0 )
        ax.set_xlim( x_min - 0.5*x_diff, x_max + 0.5*x_diff )
        ax.set_xticks( np.arange( x_min, x_max + x_diff, x_diff ), minor = False )                  # sets major ticks
        ax.set_xticks( np.arange( x_min - 0.5*x_diff, x_max + 1.5*x_diff, x_diff ), minor = True )  # sets minor ticks
    ax.set_ylabel( y_label )
    if len(y_ticks) > 0:
        y_diff = y_ticks[1] - y_ticks[0]
        y_min = np.round( y_ticks.min(), decimals = 4 )           # round to prevent precision error
        y_max = np.round( y_ticks.max() + y_diff, decimals = 4 )
        ax.set_ylim( y_min - 0.5*y_diff, y_max + 0.5*y_diff )
        ax.set_yticks( np.arange( y_min, y_max+y_diff, y_diff ), minor = False )                           # sets major ticks
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
    extension = ".svg"
    plot_title = None
    subtract_normal = False
    
    ## OUTPUT DIRECTORY
    output_dir = r"C:\Users\bdallin\Box Sync\univ_of_wisc\manuscripts\mixed_polar_sams\raw_figures"    
        
    ## MAIN DIRECTORY
    main_dir = r"R:\simulations\polar_sams\unbiased"
    
    ## FILES
    filename = "output_files/sam_prod_height_difference.pkl"
    plot_title = None #r"SAM hydrogen bonds"
    x_label = r"Charge scaled (k)" # r"Composition ($\sigma_{\it{R}}$)"
    x_ticks = np.arange( 0.0, 1.0, 0.2 )
    y_label = r"$max(h_{wc})-min(h_{wc})$ (nm)" # r"$var(h_{wc})$ $(nm^{2})$"
    y_ticks = np.arange( 0, 0.15, 0.05 ) # np.arange( 0, 0.002, 0.0005 )
    path_fig = os.path.join( output_dir, r"willard_chandler_height_difference_charge_scaled" + extension )
    reference_label = "CH3"
    reference_files = [ "sample1/sam_single_12x12_300K_dodecanethiol_tip4p_nvt_CHARMM36/output_files/sam_prod_num_hbonds_all.pkl", ]
    paths_to_files = { 
                       "NH2"   : { "0.0" : [ "sample1/sam_single_12x12_300K_C13NH2_k0.0_tip4p_nvt_CHARMM36",   ],
                                   "0.1" : [ "sample1/sam_single_12x12_300K_C13NH2_k0.1_tip4p_nvt_CHARMM36",   ],
                                   "0.2" : [ "sample1/sam_single_12x12_300K_C13NH2_k0.2_tip4p_nvt_CHARMM36",   ],
                                   "0.3" : [ "sample1/sam_single_12x12_300K_C13NH2_k0.3_tip4p_nvt_CHARMM36",   ],
                                   "0.4" : [ "sample1/sam_single_12x12_300K_C13NH2_k0.4_tip4p_nvt_CHARMM36",   ],
                                   "0.5" : [ "sample1/sam_single_12x12_300K_C13NH2_k0.5_tip4p_nvt_CHARMM36",   ],
                                   "0.6" : [ "sample1/sam_single_12x12_300K_C13NH2_k0.6_tip4p_nvt_CHARMM36",   ],
                                   "0.7" : [ "sample1/sam_single_12x12_300K_C13NH2_k0.7_tip4p_nvt_CHARMM36",   ],
                                   "0.8" : [ "sample1/sam_single_12x12_300K_C13NH2_k0.8_tip4p_nvt_CHARMM36",   ],
                                   "0.9" : [ "sample1/sam_single_12x12_300K_C13NH2_k0.9_tip4p_nvt_CHARMM36",   ],
                                   "1.0" : [ "sample1/sam_single_12x12_300K_C13NH2_tip4p_nvt_CHARMM36",        ],
                                  },
                       "CONH2" : { "0.0" : [ "sample1/sam_single_12x12_300K_C12CONH2_k0.0_tip4p_nvt_CHARMM36", ],
                                   "0.1" : [ "sample1/sam_single_12x12_300K_C12CONH2_k0.1_tip4p_nvt_CHARMM36", ],
                                   "0.2" : [ "sample1/sam_single_12x12_300K_C12CONH2_k0.2_tip4p_nvt_CHARMM36", ],
                                   "0.3" : [ "sample1/sam_single_12x12_300K_C12CONH2_k0.3_tip4p_nvt_CHARMM36", ],
                                   "0.4" : [ "sample1/sam_single_12x12_300K_C12CONH2_k0.4_tip4p_nvt_CHARMM36", ],
                                   "0.5" : [ "sample1/sam_single_12x12_300K_C12CONH2_k0.5_tip4p_nvt_CHARMM36", ],
                                   "0.6" : [ "sample1/sam_single_12x12_300K_C12CONH2_k0.6_tip4p_nvt_CHARMM36", ],
                                   "0.7" : [ "sample1/sam_single_12x12_300K_C12CONH2_k0.7_tip4p_nvt_CHARMM36", ],
                                   "0.8" : [ "sample1/sam_single_12x12_300K_C12CONH2_k0.8_tip4p_nvt_CHARMM36", ],
                                   "0.9" : [ "sample1/sam_single_12x12_300K_C12CONH2_k0.9_tip4p_nvt_CHARMM36", ],
                                   "1.0" : [ "sample1/sam_single_12x12_300K_C12CONH2_tip4p_nvt_CHARMM36",      ],
                                  },
                       "OH"    : { "0.0" : [ "sample1/sam_single_12x12_300K_C13OH_k0.0_tip4p_nvt_CHARMM36",    ],
                                   "0.1" : [ "sample1/sam_single_12x12_300K_C13OH_k0.1_tip4p_nvt_CHARMM36",    ],
                                   "0.2" : [ "sample1/sam_single_12x12_300K_C13OH_k0.2_tip4p_nvt_CHARMM36",    ],
                                   "0.3" : [ "sample1/sam_single_12x12_300K_C13OH_k0.3_tip4p_nvt_CHARMM36",    ],
                                   "0.4" : [ "sample1/sam_single_12x12_300K_C13OH_k0.4_tip4p_nvt_CHARMM36",    ],
                                   "0.5" : [ "sample1/sam_single_12x12_300K_C13OH_k0.5_tip4p_nvt_CHARMM36",    ],
                                   "0.6" : [ "sample1/sam_single_12x12_300K_C13OH_k0.6_tip4p_nvt_CHARMM36",    ],
                                   "0.7" : [ "sample1/sam_single_12x12_300K_C13OH_k0.7_tip4p_nvt_CHARMM36",    ],
                                   "0.8" : [ "sample1/sam_single_12x12_300K_C13OH_k0.8_tip4p_nvt_CHARMM36",    ],
                                   "0.9" : [ "sample1/sam_single_12x12_300K_C13OH_k0.9_tip4p_nvt_CHARMM36",    ],
                                   "1.0" : [ "sample1/sam_single_12x12_300K_C13OH_tip4p_nvt_CHARMM36",         ],
                                  },
#                       "NH2-mix"   : { "0.0"  : [ "sample1/sam_single_12x12_300K_dodecanethiol_tip4p_nvt_CHARMM36",                          ],
#                                       "0.25" : [ "sample1/sam_single_12x12_checker_300K_dodecanethiol0.75_C13NH20.25_tip4p_nvt_CHARMM36",   ],
#                                       "0.4"  : [ "sample1/sam_single_12x12_checker_300K_dodecanethiol0.6_C13NH20.4_tip4p_nvt_CHARMM36",     ],
#                                       "0.5"  : [ "sample1/sam_single_12x12_checker_300K_dodecanethiol0.5_C13NH20.5_tip4p_nvt_CHARMM36",     ],
#                                       "0.75" : [ "sample1/sam_single_12x12_checker_300K_dodecanethiol0.25_C13NH20.75_tip4p_nvt_CHARMM36",   ],
#                                       "1.0"  : [ "sample1/sam_single_12x12_300K_C13NH2_tip4p_nvt_CHARMM36",                                 ],
#                                      },
#                       "CONH2-mix" : { "0.0"  : [ "sample1/sam_single_12x12_300K_dodecanethiol_tip4p_nvt_CHARMM36",                          ],
#                                       "0.25" : [ "sample1/sam_single_12x12_checker_300K_dodecanethiol0.75_C12CONH20.25_tip4p_nvt_CHARMM36", ],
#                                       "0.4"  : [ "sample1/sam_single_12x12_checker_300K_dodecanethiol0.6_C12CONH20.4_tip4p_nvt_CHARMM36",   ],
#                                       "0.5"  : [ "sample1/sam_single_12x12_checker_300K_dodecanethiol0.5_C12CONH20.5_tip4p_nvt_CHARMM36",   ],
#                                       "0.75" : [ "sample1/sam_single_12x12_checker_300K_dodecanethiol0.25_C12CONH20.75_tip4p_nvt_CHARMM36", ],
#                                       "1.0"  : [ "sample1/sam_single_12x12_300K_C12CONH2_tip4p_nvt_CHARMM36",                               ],
#                                      },
#                       "OH-mix"    : { "0.0"  : [ "sample1/sam_single_12x12_300K_dodecanethiol_tip4p_nvt_CHARMM36",                          ],
#                                       "0.25" : [ "sample1/sam_single_12x12_checker_300K_dodecanethiol0.75_C13OH0.25_tip4p_nvt_CHARMM36",    ],
#                                       "0.4"  : [ "sample1/sam_single_12x12_checker_300K_dodecanethiol0.6_C13OH0.4_tip4p_nvt_CHARMM36",      ],
#                                       "0.5"  : [ "sample1/sam_single_12x12_checker_300K_dodecanethiol0.5_C13OH0.5_tip4p_nvt_CHARMM36",      ],
#                                       "0.75" : [ "sample1/sam_single_12x12_checker_300K_dodecanethiol0.25_C13OH0.75_tip4p_nvt_CHARMM36",    ],
#                                       "1.0"  : [ "sample1/sam_single_12x12_300K_C13OH_tip4p_nvt_CHARMM36",                                  ],
#                                      },
#                       "NH2-sep"   : { "0.0"  : [ "sample1/sam_single_12x12_300K_dodecanethiol_tip4p_nvt_CHARMM36",                        ],
#                                       "0.25" : [ "sample1/sam_single_12x12_janus_300K_dodecanethiol0.75_C13NH20.25_tip4p_nvt_CHARMM36",   ],
#                                       "0.4"  : [ "sample1/sam_single_12x12_janus_300K_dodecanethiol0.58_C13NH20.42_tip4p_nvt_CHARMM36",   ],
#                                       "0.5"  : [ "sample1/sam_single_12x12_janus_300K_dodecanethiol0.5_C13NH20.5_tip4p_nvt_CHARMM36",     ],
#                                       "0.75" : [ "sample1/sam_single_12x12_janus_300K_dodecanethiol0.25_C13NH20.75_tip4p_nvt_CHARMM36",   ],
#                                       "1.0"  : [ "sample1/sam_single_12x12_300K_C13NH2_tip4p_nvt_CHARMM36",                               ],
#                                      },
#                       "CONH2-sep" : { "0.0"  : [ "sample1/sam_single_12x12_300K_dodecanethiol_tip4p_nvt_CHARMM36",                        ],
#                                       "0.25" : [ "sample1/sam_single_12x12_janus_300K_dodecanethiol0.75_C12CONH20.25_tip4p_nvt_CHARMM36", ],
#                                       "0.4"  : [ "sample1/sam_single_12x12_janus_300K_dodecanethiol0.58_C12CONH20.42_tip4p_nvt_CHARMM36"  ],
#                                       "0.5"  : [ "sample1/sam_single_12x12_janus_300K_dodecanethiol0.5_C12CONH20.5_tip4p_nvt_CHARMM36",   ],
#                                       "0.75" : [ "sample1/sam_single_12x12_janus_300K_dodecanethiol0.25_C12CONH20.75_tip4p_nvt_CHARMM36", ],
#                                       "1.0"  : [ "sample1/sam_single_12x12_300K_C12CONH2_tip4p_nvt_CHARMM36",                             ],
#                                      },
#                       "OH-sep"    : { "0.0"  : [ "sample1/sam_single_12x12_300K_dodecanethiol_tip4p_nvt_CHARMM36",                        ],
#                                       "0.25" : [ "sample1/sam_single_12x12_janus_300K_dodecanethiol0.75_C13OH0.25_tip4p_nvt_CHARMM36",    ],
#                                       "0.4"  : [ "sample1/sam_single_12x12_janus_300K_dodecanethiol0.58_C13OH0.42_tip4p_nvt_CHARMM36",    ],
#                                       "0.5"  : [ "sample1/sam_single_12x12_janus_300K_dodecanethiol0.5_C13OH0.5_tip4p_nvt_CHARMM36",      ],
#                                       "0.75" : [ "sample1/sam_single_12x12_janus_300K_dodecanethiol0.25_C13OH0.75_tip4p_nvt_CHARMM36",    ],
#                                       "1.0"  : [ "sample1/sam_single_12x12_300K_C13OH_tip4p_nvt_CHARMM36",                                ],
#                                      },
                      }

    ## LOAD REFERENCE FILES
    ref_data = []
    for ref_file in reference_files:
        path_to_file = os.path.join( main_dir, ref_file )
        ref_data.append( load_pkl( path_to_file ) )
    if len(ref_data) > 0:
        avg_ref = np.mean( ref_data )
        avg_std = np.std( ref_data )
         
    ## LOAD DATA
    data = {}
    ## LOOP THROUGH SAM TYPE
    for legend_label, sub_data in paths_to_files.items():
        ## GATHER X VALUES AND CREATE Y PLACEHOLDER
        x = np.array([ float(ii) for ii in sub_data.keys() ])
        y_avg = []
        y_std = []
        ## LOOP THROUGH SUBDATA
        for file_list in sub_data.values():
            for ii, path_analysis in enumerate( file_list ):
                path_to_file = os.path.join( main_dir, path_analysis, filename )
                loaded_data = load_pkl( path_to_file )
                if ii < 1:
                    tmp_data = [loaded_data]
                else:
                    tmp_data.append( loaded_data )
            ## APPEND Y VALUES
            y_avg.append( np.mean( tmp_data ) )
            y_std.append( np.std( tmp_data ) )
        ## PLACE X AND Y VALUES IN TO DATA STRUCTURE
        data[legend_label] = np.array([ x, np.array(y_avg), np.array(y_std) ]).transpose()
    
    ## PLOT HBOND CONFIGURATIONS
    plot_line( path_fig = path_fig, 
               data = data,
               plot_title = plot_title,
               x_label = x_label,
               x_ticks = x_ticks,
               y_label = y_label,
               y_ticks = y_ticks,
               ncol_legend = 1,
               errorbars = True,
               savefig = save_fig )

