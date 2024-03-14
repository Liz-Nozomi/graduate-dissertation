# -*- coding: utf-8 -*-
"""
extract_gmx_gyrate.py
The purpose of this function is to extract gmx gyrate and run calcs on it

CREATED ON: 12/18/2018

AUTHOR(S):
    Alex K. Chew (alexkchew@gmail.com)
"""

### IMPORTING MODULES
import numpy as np
import MDDescriptors.core.import_tools as import_tools # Loading trajectory details
## IMPORTING GLOBAL VARIABLES
from MDDescriptors.application.nanoparticle.global_vars import GMX_XVG_VARIABLE_DEFINITION

### CLASS TO ANALYZE GYRATE DETAILS
class analyze_gmx_gyrate:
    '''
    The purpose of this function is to analyze gmx gyrate
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
    def __init__(self, traj_data, xvg_file, variable_definition_label="GYRATE" ):
        ## DEFINING VARIABLES
        self.variable_definition = GMX_XVG_VARIABLE_DEFINITION[variable_definition_label]
        
        ## READING FILES
        self.output_xvg = import_tools.read_gromacs_xvg(    traj_data = traj_data,
                                                            xvg_file = xvg_file,
                                                            variable_definition = self.variable_definition
                                                        )
        ## DEFINING AVG RMSF
        self.avg_radius_gyrate = np.mean( self.output_xvg.output['Rg'] )
        
        return
        
        


#%% MAIN SCRIPT
if __name__ == "__main__":
    
    ### DIRECTORY TO WORK ON    
    analysis_dir=r"181218-2nm_rot_ligands" # Analysis directory
    category_dir="EAM" # category directory
    specific_dir="EAM_310.15_K_2_nmDIAM_ROT005_CHARMM36_Trial_1"
    
    ### DEFINING PATH TO ANALYSIS DIRECTORY
    path2AnalysisDir=r"R:\scratch\nanoparticle_project\analysis\\" + analysis_dir + '\\' + category_dir + '\\' + specific_dir + '\\' # PC Side
    
    ### DEFINING FILE NAMES
    gro_file=r"sam_prod.gro" # Structural file
    xtc_file=r"sam_prod_10_ns_whole.xtc" # r"sam_prod_10_ns_whole.xtc" # Trajectory file
    
    ### LOADING TRAJECTORY
    traj_data = import_tools.import_traj( directory = path2AnalysisDir, # Directory to analysis
                 structure_file = gro_file, # structure file
                  xtc_file = xtc_file, # trajectories
                  want_only_directories = True, # want only directories
                  )
    
    ### INPUTS
    output_xvg="gyrate.xvg"
    
    

    
    ### DEFINING INPUT VARIABLES
    input_details = {   'traj_data'         :           traj_data,                      # Trajectory information
                         'xvg_file'         :           output_xvg,                      # File with moments of inertia
                         'variable_definition_label': "GYRATE",                # Variable definitions
                        }

    ## RUNNING ANALYSIS FOR GMX RMSF
    results = analyze_gmx_gyrate( **input_details ) 

    #%%
    ##################### EXTRACTION PROCESS #####################
    
    ### IMPORTING FUNCTION TO GET TRAJ
    from MDDescriptors.traj_tools.multi_traj import load_multi_traj_pickle, find_multi_traj_pickle, load_multi_traj_pickles
    from MDDescriptors.core.csv_tools import csv_info_add, csv_dict, csv_info_new, multi_csv_export, csv_info_export, csv_info_decoder_add

    ###########################################
    ### CLASS FUNCTION TO EXTRACT GMX HBOND ###
    ###########################################
    class extract_gmx_gyrate:
        '''
        The purpose of this function is to extract the gmx output function
        INPUTS:
            class_object: extraction class
            pickle_name: name of the directory of the pickle
        OUTPUTS:
            csv_info: updated csv info 
        '''
        ### INITIALIZATION
        def __init__(self, class_object, pickle_name, decoder_type = 'nanoparticle'):
            ## STORING STRUCTURE
            self.class_object = class_object
            ## STORING INFORMATION FOR CSV
            self.csv_dict = csv_dict(pickle_name, decoder_type)
            ## ADDING CSV DECODER INFORMATION
            self.csv_dict.add( data_title = 'avg_radius_gyrate', data =  [ np.mean( self.class_object.output_xvg.output['Rg'] ) ] )
            ## REDEFINING CSV INFO            
            self.csv_info = self.csv_dict.csv_info
            return

    ## DEFINING CLASS
    Descriptor_class = analyze_gmx_gyrate
    ## DEFINING DATE
    Date='181218'
    #%%
    ## DEFINING DESIRED DIRECTORY
    Pickle_loading_file=r"EAM_310.15_K_2_nmDIAM_ROT001_CHARMM36_Trial_1"
    ## SINGLE TRAJ ANALYSIS
    multi_traj_results = load_multi_traj_pickle(Date, Descriptor_class, Pickle_loading_file )
    ## EXTRACTING
    extracted_results = extract_gmx_gyrate( multi_traj_results,  Pickle_loading_file)
    
    
    #%%
    #### RUNNING MULTIPLE CSV EXTRACTION
    multi_csv_export(Date = Date,
                    Descriptor_class = Descriptor_class,
                    desired_titles = None, 
                    export_class = extract_gmx_gyrate,
                    export_text_file_name = 'extract_gmx_gyrate',
                    )    



