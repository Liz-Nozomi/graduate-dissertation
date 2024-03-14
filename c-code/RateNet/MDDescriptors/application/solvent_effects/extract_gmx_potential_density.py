# -*- coding: utf-8 -*-
"""
extract_gmx_potential_density.py
The purpose of this script is to extract gromacs results for electrostatic 
potential and density

IMPORTANT NOTES
---------------
- This script assumes you already ran gmx potential and density.
- We will plot both potential and density

CREATED ON: 07/06/2018

AUTHORS:
    Alex K. Chew (alexkchew@gmail.com)
"""

### IMPORTING MODULES
import numpy as np

### CUSTOM MODULES
import MDDescriptors.core.import_tools as import_tools # Loading trajectory details
from MDDescriptors.core.check_tools import check_file_exist # checks if file exists

### DEFINING DICTIONARIES FOR XVG FILE
XVG_FILE_CLASSIFIER={
        'density.xvg':
            ['distance', 'density'], # nm, kg/m^3
        'potential.xvg':
            ['distance', 'potential'], # nm, Volts        
        }

### CLASS TO ANALYZE POTENTIAL AND DENSITY TOGETHER
class analyze_gmx_potential_density:
    '''
    The purpose of this function is to analyze gmx potential and density togther
    INPUTS:
        traj_data: [object]
            trajectory data indicating location of the files
        xvg_file: [str]
            name of the xvg file
        variable_definition_label: [str]
            label with the definitions of each of the columns. This is referring to the "GMX_XVG_VARIABLE_DEFINITION" global variable.
        
    OUTPUTS:
        ## VARIABLE INFORMATION
            self.variable_definition: [list]
                Here, you will define the variables such that you define the column, name, and type of variable.
                Note: the name of the variable will be used as a dictionary.
        
    '''
    ## INITIALIZING
    def __init__(self, traj_data, xvg_files, xvg_definitions=["density.xvg", "potential.xvg"] ):
        ## DEFINING INPUTS
        self.xvg_files = xvg_files
        self.xvg_definitions = xvg_definitions
        ## DEFINING VARIABLE
        self.output_xvg = []
        ## LOOPING THROUGH
        for idx, each_xvg_file in enumerate(xvg_files):
            self.output_xvg.append( import_tools.read_gromacs_xvg(      traj_data = traj_data,
                                                                        xvg_file = each_xvg_file,
                                                                        variable_definition = import_tools.GMX_XVG_VARIABLE_DEFINITION[xvg_definitions[idx]]
                                                                        ))
        # OUTPUT: results.output_xvg[0].output
        
        return
        


#%% MAIN SCRIPT
if __name__ == "__main__":
    
    ### DIRECTORY TO WORK ON    
    analysis_dir=r"190114-Expanded_NoSolute_Pure_Sims" # Analysis directory
    category_dir="NoSolute" # category directory
    specific_dir="Expand_8nm_300.00_6_nm_NoSolute_100_WtPercWater_spce_Pure" 
    
    ### DEFINING PATH TO ANALYSIS DIRECTORY
    path2AnalysisDir=r"R:\scratch\SideProjectHuber\analysis\\" + analysis_dir + '\\' + category_dir + '\\' + specific_dir + '\\' # PC Side
    
    ### DEFINING FILE NAMES
    gro_file=r"mixed_solv_prod.gro" # Structural file
    xtc_file=r"mixed_solv_prod_10_ns_whole.xtc" # r"sam_prod_10_ns_whole.xtc" # Trajectory file
    
    ### LOADING TRAJECTORY
    traj_data = import_tools.import_traj( directory = path2AnalysisDir, # Directory to analysis
                 structure_file = gro_file, # structure file
                  xtc_file = xtc_file, # trajectories
                  want_only_directories = True, # want only directories
                  )
    
    ### DEFINING INPUT VARIABLES
    input_details = {   'traj_data'         :           traj_data,                      # Trajectory information
                         'xvg_files'        : ['density.xvg', 'potential.xvg'],          #
                         'xvg_definitions'  : ["density.xvg", "potential.xvg"]       # Summary file
                         }
    
    ## ANALYZING
    results = analyze_gmx_potential_density( **input_details )
    
    
    #%%
    
    ### IMPORTING FUNCTION TO GET TRAJ
    from MDDescriptors.traj_tools.multi_traj import load_multi_traj_pickle, find_multi_traj_pickle, load_multi_traj_pickles
    from MDDescriptors.core.csv_tools import csv_info_add, csv_info_new, multi_csv_export, csv_info_export, csv_info_decoder_add, csv_dict
    
    ### IMPORTING PLOT
    from MDDescriptors.core.plot_tools import create_plot, save_fig_png
    from MDDescriptors.global_vars.plotting_global_vars import LINE_STYLE, LABELS
    
    #########################################
    ### CLASS FUNCTION TO EXTRACT GMX MSD ###
    #########################################
    class extract_gmx_potential_density:
        '''
        The purpose of this script is to extract the potential and density. 
        We can start by making plots of potential and density. Then, we can 
        find the maxima of the density to compute the bulk density. Afterwards, 
        we can compute the bulk electrostatic potential
        INPUTS:
            
        OUTPUTS:
            
        '''
        ### INITIALIZATION
        def __init__(self, class_object, pickle_name, decoder_type = 'solvent_effects'):
            ## STORING INPUTS
            self.pickle_name = pickle_name
            ## DEFINING PERCENT CUTOFF
            self.percent_cutoff=0.95
            ## STORING STRUCTURE
            self.class_object = class_object
            ## STORING INFORMATION FOR CSV
            self.csv_dict = csv_dict(file_name = pickle_name, decoder_type = decoder_type )
            ## FINDING VARIABLES
            self.find_variables()
            ## FINDING RELATIVE EXTREMA
            self.find_density_maxima()
            ## DEFINING CSV INFO
            self.csv_info = self.csv_dict.csv_info
        
        ### FUNCTION TO FIND VARIABLES OF INTEREST
        def find_variables(self):
            ''' This function simply defines density, potential and distance'''
            ## LOCATING DENSITY INDEX
            density_index = self.class_object.xvg_definitions.index("density.xvg")
            potential_index = self.class_object.xvg_definitions.index("potential.xvg")
            ## DEFINING DISTANCES
            self.density_distance =  self.class_object.output_xvg[density_index].output['distance']
            self.potential_distance =  self.class_object.output_xvg[potential_index].output['distance']
            ## DEFINING VALUES
            self.density_values = self.class_object.output_xvg[density_index].output['density']
            self.potential_values = self.class_object.output_xvg[potential_index].output['potential']

            return
        
        ### FUNCTION TO FIND MAXIMA
        def find_density_maxima(self):
            ''' Function to compute the maxima for the density profile '''
            ## FINDING LARGEST VALUES BASED ON MAXIMA
            self.density_largest_indices = np.where( self.density_values >= np.max(self.density_values) * self.percent_cutoff )[0]
            ## NORMALIZING DENSITY
            self.density_values_normalized = self.density_values / np.mean(self.density_values[self.density_largest_indices])
            ## COMPUTING AVERAGE DENSITY
            self.avg_density = np.mean(self.density_values_normalized[self.density_largest_indices])
            self.avg_potential = np.mean(self.potential_values[self.density_largest_indices])
            ## ADDING POTENTIAL
            self.csv_dict.add( data_title = 'Electrostatic potential(V)', data = [self.avg_potential] )
            return
            
        ### PLOTTING
        def plot_potential_density(self):
            ''' This function plots the potential and density in a single plot '''
            import sys
            if sys.prefix != '/usr':
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D # For 3D axes
                from MDDescriptors.core.plot_tools import color_y_axis, LABELS, LINE_STYLE
            ## CREATING PLOT
            fig, ax = plt.subplots()
            ## DEFINING X AND Y AXIS
            ax.set_xlabel('Distance (nm)', **LABELS)
            ax.set_ylabel('Normalized Density', **LABELS)
            ## PLOTTING
            ax.plot( self.density_distance, self.density_values_normalized, color='k', **LINE_STYLE )
            ## PLOTTING MAXIMA
            ax.plot( self.density_distance[self.density_largest_indices], 
                     self.density_values_normalized[self.density_largest_indices], 
                                            '.', color='r', **LINE_STYLE )
            ## PLOTTING STRAIGHT LINE
            ax.axhline( self.avg_density, linestyle = '--', color='r' )
            ## ADDING POENTIAL
            ax2 = ax.twinx()
            ax2.set_ylabel('Electrostatic potential (V)',color='b', **LABELS)
            ax2.plot( self.potential_distance, self.potential_values, color='b', **LINE_STYLE )
            ## PLOTTING MAXIMA
            ax2.plot( self.potential_distance[self.density_largest_indices], 
                     self.potential_values[self.density_largest_indices], 
                                            '.', color='aqua', **LINE_STYLE )
            ## PLOTTING STRAIGHT LINE
            ax2.axhline( self.avg_potential, linestyle = '--', color='aqua' )
            color_y_axis(ax2, 'b')
            
        
        
        
    #%%
    ## DEFINING CLASS
    Descriptor_class = analyze_gmx_potential_density
    ## DEFINING DATE
    Date='190116'
    ## DEFINING DESIRED DIRECTORY
    Pickle_loading_file=r"Expand_8nm_300.00_6_nm_NoSolute_10_WtPercWater_spce_dmso"
    Pickle_loading_file=r"Expand_8nm_300.00_6_nm_NoSolute_100_WtPercWater_spce_Pure"
    
    #%%
    ## SINGLE TRAJ ANALYSIS
    results = load_multi_traj_pickle(Date, Descriptor_class, Pickle_loading_file )
    # Pickle_loading_file = specific_dir
    ## EXTRACTION
    potential_density = extract_gmx_potential_density( class_object =  results,pickle_name = Pickle_loading_file )
    potential_density.plot_potential_density()
    
    ######
    # FOR ESI, EXTRACTION PROTOCOL
    ## DENSITY
    extract_density_distance = potential_density.density_distance
    extract_density_values_norm = potential_density.density_values_normalized
    extract_density_avg = potential_density.avg_density
    extract_density_distance_values_upper_95 = potential_density.density_distance[potential_density.density_largest_indices]
    extract_density_distance_values_norm_upper_95 = potential_density.density_values_normalized[potential_density.density_largest_indices]
    ## POTENTIAL
    extract_potential_distance = potential_density.potential_distance
    extract_potential_distance_upper_95 = potential_density.potential_distance[potential_density.density_largest_indices]
    extract_potential_values = potential_density.potential_values
    extract_potential_values_upper_95 = potential_density.potential_values[potential_density.density_largest_indices]
    extract_potential_avg = potential_density.avg_potential
    
    ######
    
    #%%
    
    #### RUNNING MULTIPLE CSV EXTRACTION
    multi_csv_export(Date = Date,
                    Descriptor_class = Descriptor_class,
                    desired_titles = None, 
                    export_class = extract_gmx_potential_density,
                    export_text_file_name = 'extract_gmx_potential_density',
                    )    

    
    