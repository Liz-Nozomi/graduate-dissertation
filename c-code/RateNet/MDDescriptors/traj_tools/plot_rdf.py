#/usr/bin/env python
"""
plot_rdf.py
The purpose of this script is to read files from gmx rdf and make a plot out of it.
This assumes you have a water/cosolvent system. Although, it does not have to. 

Author: Alex Chew (09/26/2019)
"""

# Importing necessary modules
import os
import sys # Used to import arguments
import numpy as np
import matplotlib

## IMPORT MATPLOTLIB
if sys.prefix == '/usr':
    matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
    import matplotlib.pyplot as plt
    plt.rcParams['agg.path.chunksize'] = 10000 # Had to apply this to work on the server
else:
    import matplotlib.pyplot as plt

## CHECKING TOOLS
from MDBuilder.core.check_tools import check_testing ## CHECKING PATH FOR TESTING

## IMPORTING READ XVG
from MDDescriptors.core.read_write_tools import read_xvg

## LIGAND BUILDER
from MDBuilder.builder.ligand_builder import get_ligand_args

## DEFINING MPLT DEFAULTS
from MDDescriptors.core.plot_tools import set_mpl_defaults

## SETTING DEFAULTS)
set_mpl_defaults()

## FORMATTING 
from matplotlib.ticker import ScalarFormatter

## FORMAT
class ScalarFormatterForceFormat(ScalarFormatter):
    def _set_format(self):  # Override function that finds format to use.
        self.format = "%1.1f"  # Give format here

## DEFINING GLOBAL VARIABLES
## COLOR FOR RDFS
RDF_COLORS=[
        'k',
        'b',
        'r',
        'g',
        'c',
        'm',
        ]

## FUNCTION TO EXTRACT THINGS WITHIN QUOTES
def extract_quoted_info_from_str( str_line ):
    '''
    The purpose of this function is to extract quoted values from a string.
    INPUTS:
        str_line: [str]
            string that you want to extract quotes from
    OUTPUTS:
        within_quotes: [str]
            string that has been extracted
    '''
    quoteIndices = [n for (n, e) in enumerate(str_line) if e == '\"']
    within_quotes = str_line[ quoteIndices[0] + 1 : quoteIndices[1] ] # e.g.  'reference TOL'
    return within_quotes


### FUNCTION TO GET X AND Y LABELS
def get_xy_labels(data):
    '''
    The purpose of this function is to get x and y labels.
    INPUTS:
        data: [list]
            list of the data file
    OUTPUTS:
        x_label: [str]
            label for x values
        y_label: [str]
            label for y values            
    '''
    x_idx = [ idx for idx, row in enumerate(data) if row.startswith('@') and 'xaxis' in row][0]
    y_idx = [ idx for idx, row in enumerate(data) if row.startswith('@') and 'yaxis' in row][0]
    ## EXTRACTING ANME FROM QUOTES
    x_label = extract_quoted_info_from_str(data[x_idx])
    y_label = extract_quoted_info_from_str(data[y_idx])
    
    return x_label, y_label

### FUNCTION TO GET THE LEGEND NAMES
def get_legend_names( data ):
    '''
    The purpose of this function is to get the legend names.
    INPUTS:
        data: [list]
            list of the data file
    OUTPUTS:
        legend_names_list: [list]
            list of legend names
        
    '''
    ## GETTING REFERENCE MOLECULE NAME
    reference_idx = [ idx for idx, row in enumerate(data) if row.startswith('@ s') and 'legend' in row]
    ## STORING LIST OF LEGEND NAMES
    legend_names_list = []
    ## LOOPING THROUGH EACH INDEX
    for each_index in reference_idx: 
        ## DEFINING CURRENT LINE
        line = data[each_index]
        ## FINDING INDICES WITHIN QUOTES
        within_quotes = extract_quoted_info_from_str(line)
        
        ## STORING
        legend_names_list.append(within_quotes)
    return legend_names_list
#########################################
### FUNCTION TO EXTRACT DATA FROM RDF ###
#########################################
class extract_gmx_rdf:
    '''
    The purpose of this function is to extract the GMX RDF details.
    INPUTS:
        data: [list]
    OUTPUTS:
        self.x_label: [str]
            label for x axis
        self.y_label: [str]
            label for y axis
        self.ref_molecule_name: [str]
            ref molecule name
        self.legend_names_list: [list]
            list of legend names
        self.combined_name_list: [list]
            list of combined name with ref
        self.Extract_data: [np.array]
            extracted rdf array
    '''
    def __init__(self, data):

        ## GETTING X AND Y LABELS
        self.x_label, self.y_label = get_xy_labels(data = data)
        
        ## GETTING REFERENCE MOLECULE NAME
        self.ref_molecule_name = self.get_ref_name(data = data)
        
        ## GETTING LEGEND LIST NAME
        self.legend_names_list = self.get_legend_names(data = data)
        
        ## UPDATING LEGEND NAME LIST
        self.combined_name_list = [ self.ref_molecule_name + "-" + each_name for each_name in self.legend_names_list ]
        
        ## EXTRACTING DATA
        self.Extract_data = self.get_rdf_array(data = data)
    
        return
        
    ### FUNCTION TO GET REFERENCE MOLECULE NAME
    @staticmethod
    def get_ref_name( data ):
        '''
        The purpose of this function is to get the reference name. We assume that 
        reference may look like: 
             '@ subtitle "reference TOL"\n'
        INPUTS:
            data: [list]
                list of the data file
        OUTPUTS:
            ref_molecule_name: [str]
                reference molecule name, e.g. TOL
        '''
        ## GETTING REFERENCE MOLECULE NAME
        reference_molecule_idx = [ idx for idx, row in enumerate(data) if row.startswith('@ subtitle') and 'reference' in row][0]
        
        ## EXTRACTING ANME FROM QUOTES
        within_quotes = extract_quoted_info_from_str(data[reference_molecule_idx])
        
        ## DEFINING REFERENCE MOLECULE NAME
        ref_molecule_name = within_quotes.split(' ')[1]
        return ref_molecule_name
    
    ### FINDING RDF VALUES
    @staticmethod
    def get_rdf_array( data ):
        '''
        The purpose of this function is to extract the RDF values from gmx 
        rdf function. 
        INPUTS:
            data: [list]
                list of the data file
        OUTPUTS
            Extract_data: [np.array, float]
                extracted data as float
        '''
        ## LOCATING FINAL INDICES
        Final_indices= [i for i, j in enumerate(data) if j.startswith('@') or j.startswith('#')][-1]
        
        ## EXTRACTING THE RDF DATA
        Extract_data = np.array([ x.split() for x in data[Final_indices+1:] ]).astype('float')
        
        return Extract_data
    
### FUNCTION TO CREATE DICTIONARY FOR RDF
def generate_rdf_dict( combined_name_list, 
                       Extract_data,
                       x_label="r",
                       y_label="g_r"):
    '''
    The purpose of this function is to generate RDF dictionary given a list 
    and an array of values. The list should be as follows:
        r-values: values of the radius -- assumed to be 0 index
        g(r) 1 values: values for RDF
        g(r) 2 values: values for RDF
        ... and so on
    INPUTS:
        legend_names_list: [list]
            list of legend names
        Extract_data: [np.array]
            numpy array of extracted values
    OUTPUTS:
        
    '''
    ## CREATING EMPTY DICTIONARY
    rdf_dict = {}
    ## LOOPING THROUGH LEGEND NAMES
    for leg_idx, each_legend in enumerate(combined_name_list):
        ## DEFINING INDEX
        index = leg_idx + 1 # Start count at 1
        ## STORING EACH LEGEND
        rdf_dict[each_legend] = {
                x_label: Extract_data[:, 0],
                y_label: Extract_data[:, index],
                }
    return rdf_dict

## COMBINING EACH NAME AND GENERATING RDF DICT
def combine_rdf_dict( storage_rdf ):
    '''
    The purpose of this function is to generate RDF based on some list of dicts
    INPUTS:
        storage_rdf: [list]
            list of rdf classes
    OUTPUTS:
        combined_name_list: [list]
            combined name list using all rdf functions
        Extract_data: [np.array]
            numpy array of extracted data
    '''        
    ## FOR EACH RDF
    for idx, each_rdf_class in enumerate(storage_rdf):
        ## SEEING IF IDX IS ZERO
        if idx == 0:
            ## STORING VARIABLE
            combined_name_list = each_rdf_class.combined_name_list
            Extract_data = each_rdf_class.Extract_data
        else:
            ## ADDING TO THE LIST
            combined_name_list.extend(each_rdf_class.combined_name_list)
            Extract_data = np.append(Extract_data, each_rdf_class.Extract_data[:,1:], axis = 1)
    return combined_name_list, Extract_data

### FUNCTION TO PLOT RDF
def plot_gmx_rdf( rdf_dict, 
                  x_label = 'r (nm)',
                  y_label = 'g(r)',
                  x_label_key = 'r',
                  y_label_key = "g_r",
                  fontsize = 12,
                  linewidth = 1.2,
                  separated=False,
                  want_line_at_y_1 = True,
                  ):
    '''
    The purpose of this function is to plot the rdf found from gmx rdf
    INPUTS:
        rdf_dict: [dict]
            dictionary of RDFs that you want to plot
        x_label: [str]
            label for x axis
        y_label: [str]
            label for y axis
        x_label_key: [str]
            key for x label within dictionary
        y_label_key: [str]
            key for y label within dictionary
        fontsize: [int]
            font size of axis
        linewidth: [float]
            line width
        separated: [logical]
            True if you want the plots separate in sub plots
        want_line_at_y_1: [logical]
            True if you want a line at y = 1
    OUTPUTS:
        fig, ax: figure and axis for the plot
    '''
    ## CREATING PLOT
    if separated is False:
        fig = plt.figure() 
        ax = fig.add_subplot(111)
    
        ## SETTING X AND Y LEGENDS
        ax.set_xlabel(x_label, fontsize = fontsize)
        ax.set_ylabel(y_label, fontsize = fontsize)
        
        ## LOOPING THROUGH RDF FUNCTION
        for idx, each_key in enumerate(rdf_dict):
            ## DEFINING R AND GR
            r = rdf_dict[each_key][x_label_key]
            g_r = rdf_dict[each_key][y_label_key]
            ax.plot(r, g_r, '-', linewidth=linewidth, color = RDF_COLORS[idx], label = each_key)
            
        ## ADDING HORIZONTAL LINE
        if want_line_at_y_1 is True:
            ax.axhline(y=1, linestyle='--', color='gray', linewidth=2)
    
        ## ADDING LEGEND
        ax.legend()
    
    else:
        ## SEPARATED SUBPLOTS
        total_plots = len(rdf_dict)
        
        ## GENERATING SUBPLOTS
        fig, ax = plt.subplots(total_plots, 1, sharex=True)
        
        ## SETTING AXIS LABELS
        # ax[0].set_xlabel(x_label, fontsize = fontsize)
        
        ## LOOPING THROUGH EACH RDF
        for idx, each_key in enumerate(rdf_dict):
            ## DEFINING R AND GR
            r = rdf_dict[each_key][x_label_key]
            g_r = rdf_dict[each_key][y_label_key]
            ax[idx].plot(r, g_r, '-', linewidth=linewidth, color = RDF_COLORS[idx], label = each_key)
            ax[idx].legend()
            ## ADDING HORIZONTAL LINE
            if want_line_at_y_1 is True:
                ax[idx].axhline(y=1, linestyle='--', color='gray', linewidth=2)
            ## SETTING Y AXIS LABEL
            ax[idx].set_xlabel(x_label, fontsize = fontsize)
            ax[idx].set_ylabel(y_label, fontsize = fontsize)
            
            ## FORMAT Y AXIS LABELS
            yfmt = ScalarFormatterForceFormat()
            yfmt.set_powerlimits((0,0))
            ax[idx].yaxis.set_major_formatter(yfmt)
            
            ## SHOWING GRID
            ax[idx].grid()
        ## TIGHT LAYOUT
        fig.tight_layout()
        ## REMOVAL OF HORIZONTAL SPACE BETWEEN AXISES
        fig.subplots_adjust(hspace=0)
    
    return fig,ax 

### FUNCTION TO EXTRACT DATA
def extract_xvg_data(path_to_directory,
                     xvg_list,
                     x_label = "r",
                     y_label = "g_r",
                     extraction_function = None):
    '''
    The purpose of this function is to extract xvg data. 
    INPUTS:
        path_to_directory: [str]
            path where all the xvgs are stored
        xvg_list: [list]
            list of xvg files
        x_label: [str]
            x label used for the dictionary
        y_label: [str]
            y label used for dictionary
        extraction_function: [func]
            function to extract the xvg data. If None, then the data is just 
            stored.
    OUTPUTS:
        data_dict: [dict]
            dictionary for the data
        plot_x_label: [str]
            plot x label
        plot_y_label: [str]
            plot y label
    '''
    
    ## DEFINING STORAGE
    storage_rdf = []
    
    ## LOOPING THROUGH EACH XVG
    for rdf_xvg in xvg_list:
        ## DEFINING PATH TO XVG
        path_xvg = os.path.join(path_to_directory,rdf_xvg)
        
        ## READING XVG FILE
        xvg_file = read_xvg(path_xvg)
        
        ## DEFINING DATA
        data = xvg_file.xvg_lines
        
        ## GETTING INFORMATION FROM RESIDUE-RESIDUE
        if extraction_function is not None:
            rdf = extraction_function(data = data)
        else:
            rdf = data
        ## STORING RDF INFO
        storage_rdf.append(rdf)

    ## DEFINING X AND Y LABELS
    if extraction_function is not None:
        ## COMBINING RDFS
        combined_name_list, Extract_data = combine_rdf_dict( storage_rdf )
        
        plot_x_label = storage_rdf[0].x_label
        plot_y_label = storage_rdf[0].y_label
    
        ## GENETING DICTIONARY FOR RDF
        data_dict = generate_rdf_dict(combined_name_list = combined_name_list,
                                      Extract_data = Extract_data,
                                      x_label = x_label,
                                      y_label = y_label)
    else:
        data_dict = storage_rdf
        plot_x_label = None
        plot_y_label = None
    return data_dict, plot_x_label, plot_y_label


## DEFINING MAIN
def main(path_to_directory, 
         rdf_xvg_list, 
         output_image_name = None,
         separated=False,
         extraction_function = extract_gmx_rdf,
         want_line_at_y_1 = True,
         store_image = True,):
    '''
    Main function goes through the following algorithm:
        - Loop through rdf xvg file list
        - Combine the data into a dictionary
        - Feed dictionary into a plotting RDF function
        - Print out RDF plot as a png
    INPUTS:
        path_to_directory: [str]
            path to the directory of interest
        rdf_xvg_list: [list]
            list of xvg files
        output_image_name: [str]
            string of the output name
        separated: [logical]
            True if you want the plots separate in sub plots
        want_line_at_y_1: [logical]
            True if you want a line at y = 1
        store_image: [logical]
            True if you want to store image
    OUTPUTS:
        fig, ax: [obj]
            figure objects
        rdf_dict: [dict]
            dictionary for data
    '''
    ## DEFAULT LABEL KEYS
    x_label_key="x"
    y_label_key="y"
    
    ## EXTRACTING THE DATA
    rdf_dict, x_label, y_label = extract_xvg_data(path_to_directory = path_to_directory,
                                                   xvg_list = rdf_xvg_list,
                                                   x_label = x_label_key,
                                                   y_label = y_label_key,
                                                   extraction_function = extraction_function)
    
    ## DEFINING FIGURE PATH
    fig_path = os.path.join(path_to_directory, output_image_name)
    fig_name, fig_extension = os.path.splitext(output_image_name)
    
    ## PLOTTING
    fig, ax = plot_gmx_rdf( rdf_dict, 
                            x_label = x_label,
                            y_label = y_label,
                            x_label_key = x_label_key,
                            y_label_key = y_label_key,
                            fontsize = 12,
                            linewidth = 2,
                            separated = separated,
                            )
    
    ## STORING FIGURE
    if store_image is True:
        fig.savefig(fig_path, format=fig_extension[1:], dpi=600)    
        ## PRINTING
        print("Storing image in %s"%(fig_path))
    return fig, ax, rdf_dict

#%%
## MAIN SCRIPT
if __name__=="__main__":
    
    # Testing parameters
    Testing = check_testing() # False if you are running this program on command line
    
    ## RUNNING TESTING
    if Testing == True:
    
        ## DEFINING FULL PATH
        path_to_directory = r"R:\scratch\nanoparticle_project\simulations\191221-Rerun_all_EAM_models_1_cutoff\EAM_300.00_K_2_nmDIAM_ROT012_CHARMM36jul2017_Trial_1"
        # r"/mnt/r/scratch/nanoparticle_project/prep_system/prep_mixed_solvents/7_toluene_10_massfrac_300"
        
        ## DEFINING RDF XVG
        rdf_xvg_list = ["np_rdf-AUNP_counterions.xvg",
                        "np_rdf-AUNP_lig.xvg",
                        "np_rdf-AUNP_SOL.xvg",
                        "np_rdf-AUNP_lig_nitrogens.xvg"]
        # [ "RDF_Res1_Res2.xvg", "RDF_Res2_Res2.xvg" ] 
        
        ## DEFINING NAME OF IMAGE
        output_image_name="np_rdf.png"
        
        ## DEFINING SEPARATION DESIRED
        separated = True
    else:
        # Importing parser to allow inputs
        from optparse import OptionParser # Used to allow commands within command line
        use = "Usage: %prog [options]"
        parser = OptionParser(usage = use)
        
        ## PATH INFORMATION
        parser.add_option('--path', dest = 'path_to_directory', help = 'Full input path to files', default = '.')
        
        ## RDF XVG
        parser.add_option("--rdf_xvg", dest="rdf_xvg_list", action="callback", type="string", callback=get_ligand_args,
                  help="RDF xvg file, separate each by comma (no whitespace)", default = None)

        ## PATH INFORMATION
        parser.add_option('--image', dest = 'output_image_name', help = 'Output image that is stored into path', default = '.')        
        
        ## DESIRED SEPARATE SUBPLOTS
        parser.add_option('--separated', dest = 'separated', action="store_true",
                          help = 'Set toggle if you want RDFs in separate subplots', default = False)        

        
        ### GETTING ARGUMENTS
        (options, args) = parser.parse_args() # Takes arguments from parser and passes them into "options" and "argument"
        
        ## DEFINING VARIABLES
        path_to_directory = options.path_to_directory
        rdf_xvg_list = options.rdf_xvg_list
        output_image_name = options.output_image_name
        separated = options.separated
        
    ## RUNNING MAIN CODE
    main(path_to_directory = path_to_directory, 
         rdf_xvg_list = rdf_xvg_list, 
         output_image_name = output_image_name,
         separated = separated)

        
        