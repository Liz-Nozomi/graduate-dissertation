"""
plot_indus_data.py

Author: Brad Dallin 
"""
# %% Import relevent libraries
import sys, os
import shutil
import numpy as np
#import matplotlib as mpl
#mpl.use('pdf')    
import matplotlib.pyplot as plt

# %% functions
def read_hbond_csv( filename ):
    '''
    read_xvg reads a xvg file and extracts the numeric values as floats
    INPUT: filename = name of xvg file to be read
    OUTPUT: numeric data values
    '''
    with open( filename, 'r' ) as outputfile: # Opens gro file and make it writable
        fileData = outputfile.readlines()
    
    # split file into lines
    lines = [ n.strip( '\n' ) for n in fileData ]
    lines.pop(0) # delete first line
    # extract keys from file
    keys = [ key for key in lines[0].split(',') ][1:] # ignore first entry (blank)
    for ii in range(len(keys)):
        k = list(keys[ii])
        for jj in range(len(k)):
            if k[jj] in [ " ", "-" ]:
                k[jj] = "_"
        keys[ii] = "".join(k)
    lines.pop(0) # delete second line
    
    # extract total
    totals = [ float(n) for n in lines[0].split(',')[1:] ] # ignore first entry (title)
    # assign totals to dictionary
    hbond_totals = {}
    for key, total in zip( keys, totals ):
        hbond_totals[key] = total
    lines.pop(0) # delete third line
    
    # extract averages
    averages = [ float(n) for n in lines[0].split(',')[1:] ] # ignore first entry (title)
    # assign totals to dictionary
    hbond_averages = {}
    for key, average in zip( keys, averages ):
        hbond_averages[key] = average
    lines.pop(0) # delete fourth line
    lines.pop(0) # delete fifth line
    lines.pop(0) # delete sixth line
    lines.pop(0) # delete seventh line
    
    # extract distributions
    hbond_dist = {}
    dist_data = [ line.split(',') for line in lines if line != '' ]
    dist_data = np.array([ [ float(n) for n in line if n != '' ] for line in dist_data ])
    hbond_dist["num_water"] = dist_data[:,0]
    for ii, key in enumerate(keys):
        hbond_dist[key] = dist_data[:,ii+1]
    
    return hbond_totals, hbond_averages, hbond_dist
        
def plot_hbonds( data_list, filename, labels, plot_total = False, plot_average = False, plot_distributions = False ):
    r"""
    """
    # JACS format: single column 3.33 in. wide
    #              double column 4.167- 7 in. wide
    #              all 9.167 in. height (including caption)  
    #              font >=4.5 pt
    #              linewidth >=0.5 pt
    #              font style Helvetica or Arial
    ## SET FIGURE DIMENSIONS
    width = 3.33 # in
    height = 0.75 * width # 4:3 aspect ratio
    
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
    
    ## LIST OF COLORS
    colors = [ "dimgrey",
               "slateblue",
               "darkseagreen",
               "tomato" ]
    
    ## CREATE PLOT
    if plot_total is True:
        fig, ax = plt.subplots()
        fig.subplots_adjust( left = 0.15, bottom = 0.16, right = 0.99, top = 0.97 )
        bar_width = 1 / float(len(labels) + 1)
        x = np.arange(len(data_list[0].items()))
        shift = np.arange( 0, 1. - bar_width, bar_width ) - ( 0.5 - bar_width )
        
        add_label = True
        keys = []
        jj = 0
        for key in data_list[0].keys():
            if "all_" in key:
                keys.append( key.split('_')[1] )
            elif "_" in key:
                keys.append( "-".join( [ k for k in key.split('_') if k != '_' ] ) )
            else:
                keys.append( key )
            
            for ii, label in enumerate( labels ):
                if add_label is True:
                    plt.bar( x[jj] + shift[ii], 1e-3 * data_list[ii][key], 
                                                linestyle = "None", 
                                                color = colors[ii],
                                                width = bar_width,
                                                edgecolor = "black", 
                                                linewidth = 0.5,
                                                label = label, )
                else:
                    plt.bar( x[jj] + shift[ii], 1e-3 * data_list[ii][key], 
                                                linestyle = "None", 
                                                color = colors[ii],
                                                width = bar_width,
                                                edgecolor = "black", 
                                                linewidth = 0.5, )
                    
            # icrement to next key    
            jj += 1
            
            # turn labels off
            add_label = False
                   
        plt.plot( [ -1, len(labels) ], [ 0, 0 ], linewidth = 0.5, linestyle = '-', color = 'black' )

        ## SET LEGEND
        ax.legend( ncol = 2 )
                    
        # SET X AND Y AXES
        ax.set_xlim( -0.5, len(keys)-0.5 )
        ax.set_xticks( np.arange( 0, len(keys), 1 ), minor = False )      # sets major ticks
        ax.set_xticks( np.arange( -0.5, len(keys)+0.5, 1 ), minor = True )  # sets minor ticks
        ax.set_xticklabels( keys, rotation = 45 )
        ax.set_ylabel( r"Total hbonds x $10^{3}$" )
        ax.set_ylim( -0.2, 2.6 )
        ax.set_yticks( np.arange( 0, 2.8, 0.4 ), minor = False )   # sets major ticks
        ax.set_yticks( np.arange( -0.2, 3.0, 0.4 ), minor = True )  # sets minor ticks

        fig.set_size_inches( width, height )
        fig.tight_layout()
        fig.savefig( filename, dpi = 300, facecolor = 'w', edgecolor = 'w' )
        
    if plot_average is True:
        fig, ax = plt.subplots()
        fig.subplots_adjust( left = 0.15, bottom = 0.16, right = 0.99, top = 0.97 )
        bar_width = 1 / float(len(labels) + 1)
        x = np.arange(len(data_list[0].items()))
        shift = np.arange( 0, 1. - bar_width, bar_width ) - ( 0.5 - bar_width )
        
        add_label = True
        keys = []
        jj = 0
        for key in data_list[0].keys():
            if "all_" in key:
                keys.append( key.split('_')[1] )
            elif "_" in key:
                keys.append( "-".join( [ k for k in key.split('_') if k != '_' ] ) )
            else:
                keys.append( key )
            
            for ii, label in enumerate( labels ):
                if add_label is True:
                    plt.bar( x[jj] + shift[ii], data_list[ii][key], 
                                                linestyle = "None", 
                                                color = colors[ii],
                                                width = bar_width,
                                                edgecolor = "black", 
                                                linewidth = 0.5,
                                                label = label, )
                else:
                    plt.bar( x[jj] + shift[ii], data_list[ii][key], 
                                                linestyle = "None", 
                                                color = colors[ii],
                                                width = bar_width,
                                                edgecolor = "black", 
                                                linewidth = 0.5, )
                    
            # icrement to next key    
            jj += 1
            
            # turn labels off
            add_label = False
                   
        plt.plot( [ -1, len(keys) ], [ 0, 0 ], linewidth = 0.5, linestyle = '-', color = 'black' )

        ## SET LEGEND
        ax.legend( ncol = 2 )
                    
        # SET X AND Y AXES
        ax.set_xlim( -0.5, len(keys)-0.5 )
        ax.set_xticks( np.arange( 0, len(keys), 1 ), minor = False )      # sets major ticks
        ax.set_xticks( np.arange( -0.5, len(keys)+0.5, 1 ), minor = True )  # sets minor ticks
        ax.set_xticklabels( keys, rotation = 45 )      
        ax.set_ylabel( r"Number of hydrogen bonds" )
        ax.set_ylim( -0.5, 5.5 )
        ax.set_yticks( np.arange( 0, 6.0, 1.0 ), minor = False )   # sets major ticks
        ax.set_yticks( np.arange( -0.5, 6.5, 1.0 ), minor = True )  # sets minor ticks
            
        fig.set_size_inches( width, height )
        fig.tight_layout()
        fig.savefig( filename, dpi = 300, facecolor = 'w', edgecolor = 'w' )

#%% Execute functions
if __name__ == "__main__":    
    
# --- USER SPECIFIED VALUES ---
    print( '\n\n--- plot_hbond_data.py ---' )
    print( '--------------------------\n' )

    plot_total            = True
    plot_average          = True
    plot_dist_all         = False
    plot_dist_water       = False
    plot_dist_sam         = False
    plot_dist_sam_sam     = False
    plot_dist_sam_water   = False
    plot_dist_water_water = False
    path_output = r"C:\Users\bdallin\Documents\Box Sync\univ_of_wisc\manuscripts\mixed_polar_sams\figure_hbonds"
    path_directories = r"R:\simulations\polar_sams\unbiased\tip4p"
    directories = { 
                    "C11CH3"           : "sam_single_12x12_300K_dodecanethiol_tip4p_nvt_CHARMM36",
#                    "C13NH2_25_mixed"  : "sam_single_12x12_checker_300K_dodecanethiol0.75_C13NH20.25_tip4p_nvt_CHARMM36",
#                    "C13NH2_40_mixed"  : "sam_single_12x12_checker_300K_dodecanethiol0.6_C13NH20.4_tip4p_nvt_CHARMM36",
#                    "C13NH2_50_mixed"  : "sam_single_12x12_checker_300K_dodecanethiol0.5_C13NH20.5_tip4p_nvt_CHARMM36",
#                    "C13NH2_75_mixed"  : "sam_single_12x12_checker_300K_dodecanethiol0.25_C13NH20.75_tip4p_nvt_CHARMM36",
                    "C13NH2"           : "sam_single_12x12_300K_C13NH2_tip4p_nvt_CHARMM36",
#                    "C12CONH2_25_mixed": "sam_single_12x12_checker_300K_dodecanethiol0.75_C12CONH20.25_tip4p_nvt_CHARMM36",
#                    "C12CONH2_40_mixed": "sam_single_12x12_checker_300K_dodecanethiol0.6_C12CONH20.4_tip4p_nvt_CHARMM36",
#                    "C12CONH2_50_mixed": "sam_single_12x12_checker_300K_dodecanethiol0.5_C12CONH20.5_tip4p_nvt_CHARMM36",
#                    "C12CONH2_75_mixed": "sam_single_12x12_checker_300K_dodecanethiol0.25_C12CONH20.75_tip4p_nvt_CHARMM36",
                    "C12CONH2"         : "sam_single_12x12_300K_C12CONH2_tip4p_nvt_CHARMM36",
#                    "C13OH_25_mixed"   : "sam_single_12x12_checker_300K_dodecanethiol0.75_C13OH0.25_tip4p_nvt_CHARMM36",
#                    "C13OH_40_mixed"   : "sam_single_12x12_checker_300K_dodecanethiol0.6_C13OH0.4_tip4p_nvt_CHARMM36",
#                    "C13OH_50_mixed"   : "sam_single_12x12_checker_300K_dodecanethiol0.5_C13OH0.5_tip4p_nvt_CHARMM36",
#                    "C13OH_75_mixed"   : "sam_single_12x12_checker_300K_dodecanethiol0.25_C13OH0.75_tip4p_nvt_CHARMM36",
                    "C13OH"            : "sam_single_12x12_300K_C13OH_tip4p_nvt_CHARMM36",
                  }

    total_data = []
    average_data = []
    distribution_data =[]
    for key, wd in directories.items():
        path_input = os.path.join( path_directories, wd, r"output_files" )
        hbond_file = os.path.join( path_input, r"hbond_properties.csv" )
        total_png = os.path.join( path_output, r"hbond_total_test.png" )
        average_png = os.path.join( path_output, r"hbond_average_test.png" )
        # read hbond data file
        hbond_total, hbond_average, hbond_distributions = read_hbond_csv( hbond_file )
        total_data.append( hbond_total )
        average_data.append( hbond_average )
        distribution_data.append( hbond_distributions )
    
    if plot_total is True:
        plot_hbonds( total_data, total_png, directories.keys(), plot_total = True )
    
    if plot_average is True:
        plot_hbonds( average_data, average_png, directories.keys(), plot_average = True )
