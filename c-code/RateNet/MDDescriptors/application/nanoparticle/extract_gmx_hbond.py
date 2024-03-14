# -*- coding: utf-8 -*-
"""
extract_gmx_hbond.py
The purpose of this function is to extract gmx_hbond code and run subsequent calculations on it

IMPORTANT NOTES
---------------
- This script assumes you already ran gmx hbond. In particular, we are interested in the number of hydrogen bonds
- Currently, this only takes in hbnum.xvg (could expand on this later)

CREATED ON: 10/04/2018

AUTHOR(S):
    Alex K. Chew (alexkchew@gmail.com)

"""
### IMPORTING MODULES
import numpy as np
import MDDescriptors.core.import_tools as import_tools # Loading trajectory details


#########################################
### CLASS FUNCTION TO READ HBNUM FILE ###
#########################################
class read_hbnum_xvg:
    '''
    The purpose of this function is to read the hbnum.xvg file from GROMACS gmx hbond
    INPUTS:
        file_path: [str] full path to the hbond file
    OUTPUTS:
        ## INITIAL VARIABLES
            self.file_path: [str]
                file path name
        ## FILE OUTPUTS
            self.data_full: [list] 
                full list of the original data
            self.data_extract: [list] 
                extracted data in a form of a list (i.e. no comments)
        ## VARIABLE OUTPUTS
            self.output_timestep: [np.array, shape=(timestep, 1)] 
                list of time steps as floating point numbers (picoseconds)
            self.output_axis: [np.array, shape=(timestep, 1)] 
                array with the number of hydrogen bonds
    FUNCTIONS:
        define_variables: function to define variables based on the outputs of "read" function
    '''
    ## INITIALIZING
    def __init__(self, file_path,  ):
        ## STORING INITIAL VARIABLES
        self.file_path = file_path
        ## READING THE FILE
        self.data_full, self.data_extract = import_tools.read_xvg(self.file_path)
        ## EXTRACTION OF VARIABLES
        self.define_variables()
        return

    ### FUNCTION TO DEFINE VARIABLES BASED ON EXTRACTION
    def define_variables(self, ):
        '''
        The purpose of this function is to define variables based on extracted results
        INPUTS:
            self.data_extract: [list] 
                extracted data in a form of a list (i.e. no comments)
        OUTPUTS:
            self.output_timestep: [np.array, shape=(timestep, 1)] 
                list of time steps as floating point numbers (picoseconds)
            self.output_axis: [np.array, shape=(timestep, 1)] 
                array with the number of hydrogen bonds
        '''
        self.output_timestep = np.array([ x[0] for x in self.data_extract]).astype("float") # output time step in picoseconds
        self.output_num_hbond = np.array([ x[1] for x in self.data_extract]).astype("float") # major, middle, and minor axis
        
########################################
### CLASS TO FULLY ANALYZE GMX HBOND ###
########################################        
class analyze_gmx_hbond:
    '''
    The purpose of this function is to analyze the outputs of gmx hbond
    INPUTS:
        traj_data: [null] 
            traj_data is only here to enable MDDescriptors and inclusion of file directory space
        hbnum_file: [str] 
            string name of the output hbnum file from gmx principal
    OUTPUTS:
        self.moi_extract: [class] extraction information from extract_moi_file class
    '''
    ## INITIALIZATION
    def __init__(self, traj_data, hbnum_file,  ):
    
        ### DEFINING FULL PATH
        full_path = traj_data.directory + '/' + hbnum_file
        
        ### READING MOI FILE
        file_data = read_hbnum_xvg(full_path)
        
        ### CALCULATING AVERAGE HYDROGEN BOND
        self.avg_h_bond =  np.mean(file_data.output_num_hbond)
    
        return
    



#%% MAIN SCRIPT
if __name__ == "__main__":
    
    ### DIRECTORY TO WORK ON    
    analysis_dir=r"180928-2nm_RVL_proposal_runs_fixed" # Analysis directory
    category_dir="EAM" # category directory
    specific_dir="EAM_310.15_K_2_nmDIAM_ROT001_CHARMM36_Trial_1"
    
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
    
    ### DEFINING INPUT VARIABLES
    input_details = {   'traj_data'         :           traj_data,                      # Trajectory information
                         'hbnum_file'         :           "hbnum.xvg",                      # File with moments of inertia
                        }

    ### RUNNING ANALYSIS FOR GMX PRINCIPAL
    hbond_results = analyze_gmx_hbond( **input_details ) 
    
    #%%
    ##################### EXTRACTION PROCESS #####################
    
    ### IMPORTING FUNCTION TO GET TRAJ
    from MDDescriptors.traj_tools.multi_traj import load_multi_traj_pickle, find_multi_traj_pickle, load_multi_traj_pickles
    from MDDescriptors.core.csv_tools import csv_info_add, csv_dict, csv_info_new, multi_csv_export, csv_info_export, csv_info_decoder_add

    ###########################################
    ### CLASS FUNCTION TO EXTRACT GMX HBOND ###
    ###########################################
    class extract_gmx_hbond:
        '''
        The purpose of this function is to extract the analyze_gmx_principal function
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
            self.csv_dict = csv_dict(pickle_name, decoder_type  )
            ## ADDING CSV DECODER INFORMATION
            self.csv_dict.add( data_title = 'avg_h_bond', data =  [ class_object.avg_h_bond ] )
            ## REDEFINING CSV INFO            
            self.csv_info = self.csv_dict.csv_info
            return

    ## DEFINING CLASS
    Descriptor_class = analyze_gmx_hbond
    ## DEFINING DATE
    Date='181004'
    #%%
    ## DEFINING DESIRED DIRECTORY
    Pickle_loading_file=r"EAM_310.15_K_2_nmDIAM_ROT001_CHARMM36_Trial_1"
    ## SINGLE TRAJ ANALYSIS
    multi_traj_results = load_multi_traj_pickle(Date, Descriptor_class, Pickle_loading_file )
    ## EXTRACTING
    extracted_results = extract_gmx_hbond( multi_traj_results,  Pickle_loading_file)
    
    
    #%%
    #### RUNNING MULTIPLE CSV EXTRACTION
    multi_csv_export(Date = Date,
                    Descriptor_class = Descriptor_class,
                    desired_titles = None, 
                    export_class = extract_gmx_hbond,
                    export_text_file_name = 'extract_gmx_hbond',
                    )    
    