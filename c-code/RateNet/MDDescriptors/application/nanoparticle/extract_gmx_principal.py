# -*- coding: utf-8 -*-
"""
extract_gmx_principal.py
The purpose of this function is to extract gmx_principal code and run subsequent calculations on it

IMPORTANT NOTES
---------------
- This script assumes you already ran gmx principal. In particular, we are interested in the moments of inertia.
- Currently, this only takes in moi.xvg

CREATED ON: 06/18/2018

AUTHOR(S):
    Alex K. Chew (alexkchew@gmail.com)

"""

### IMPORTING MODULES
import numpy as np
import MDDescriptors.core.import_tools as import_tools # Loading trajectory details

#######################################
### CLASS FUNCTION TO LOAD MOI FILE ###
#######################################
class read_moi_file:
    '''
    The purpose of this function is to read the moi.xvg file from GROMACS gmx principal
    INPUTS:
        file_path: [str] full path to the moi file
    OUTPUTS:
        ## INITIAL VARIABLES
            self.file_path: file path name
        ## FILE OUTPUTS
            self.data_full: [list] full list of the original data
            self.data_extract: [list] extracted data in a form of a list (i.e. no comments)
        ## VARIABLE OUTPUTS
            self.output_timestep: [np.array, shape=(timestep, 1)] list of time steps as floating point numbers (picoseconds)
            self.output_axis: [np.array, shape=(timestep, 3)] array all the major, middle, and minor axis values, e.g. [[major, middle, minor], ...]
    FUNCTIONS:
        define_variables: function to define variables based on the outputs of "read" function
    ALGORITHM:
        - load moi file
        - exclude all comments
        - delineate the values into variables
    '''
    ## INITIALIZING
    def __init__(self, file_path,  ):
        ## STORING INITIAL VARIABLES
        self.file_path = file_path
        ## READING THE FILE
        self.data_full, self.data_extract = import_tools.read_xvg(self.file_path)
        ## EXTRACTION OF VARIABLES
        self.define_variables()

    ### FUNCTION TO DEFINE VARIABLES BASED ON EXTRACTION
    def define_variables(self, ):
        '''
        The purpose of this function is to define variables based on extracted results
        INPUTS:
            self.data_extract: [list] extracted data in a form of a list (i.e. no comments)
        OUTPUTS:
            self.output_timestep: [np.array, shape=(timestep, 1)] list of time steps as floating point numbers (picoseconds)
            self.output_axis: [np.array, shape=(timestep, 3)] array all the major, middle, and minor axis values, e.g. [[major, middle, minor], ...]
        '''
        ## DEFINING TIME STAMPS
        self.output_timestep = np.array([ x[0] for x in self.data_extract]).astype("float") # output time step in picoseconds
        self.output_axis = np.array([ x[1:] for x in self.data_extract]).astype("float") # major, middle, and minor axis
    
#########################################################
### FUNCTION TO EXTRACT INFORMATION FROM THE MOI FILE ###
#########################################################
class extract_moi_file:
    '''
    The purpose of this function is to extract the moi.xvg file
    INPUTS:
        moi_file: [class] object from read_moi_file class
    OUTPUTS:
        
    '''
    ## INITIALIZING
    def __init__(self, moi_file,  ):
        ## STORING MOI FILE
        self.moi_file=moi_file
        
        ## CALCULATING ECCENTRICITY
        self.calc_eccentricity()
        
    ### FUNCTION TO CALCULATE ECCENTRICITY
    def calc_eccentricity(self):
        '''
        The purpose of this function is to calculate eccentricity
        e = 1 - I_min / I_avg
        Reference: Lebecque, S., Crowet, J. M., Nasir, M. N., Deleu, M. & Lins, L. Molecular dynamics study of micelles properties according to their size. J. Mol. Graph. Model. 72, 6–15 (2017).
        INPUTS:
            void
        OUTPUTS:
            self.eccentricity: [float] eccentricity value as determined by the equation
        '''
        ## FINDING AVERAGES
        averages = np.mean(self.moi_file.output_axis, axis = 1 )
        ## FINDING MINIMAS
        minimas =  np.min(self.moi_file.output_axis, axis = 1 )
        ## ECCENTRICITY OVER TIME
        eccentricity = 1 - minimas/averages
        ## DIVIDING AND AVERAGING
        self.eccentricity_avg = np.mean(eccentricity)
        self.eccentricity_std = np.std(eccentricity)
        return
    
    
############################################
### CLASS TO FULLY ANALYZE GMX PRINCIPAL ###
############################################        
class analyze_gmx_principal:
    '''
    The purpose of this function is to analyze the outputs of gmx principal
    INPUTS:
        traj_data: [null] traj_data is only here to enable MDDescriptors and inclusion of file directory space
        moi_file: [str] string name of the output moi file from gmx principal
    OUTPUTS:
        self.moi_extract: [class] extraction information from extract_moi_file class
    '''
    ## INITIALIZATION
    def __init__(self, traj_data, moi_file,  ):
    
        ### DEFINING FULL PATH
        full_moi_path = traj_data.directory + '/' + moi_file
        
        ### READING MOI FILE
        moi_file = read_moi_file(full_moi_path)
           
        ### EXTRACTING MOI FILE
        self.moi_extract = extract_moi_file(moi_file)
    
        return




#%% MAIN SCRIPT
if __name__ == "__main__":
    
    ### DIRECTORY TO WORK ON    
    analysis_dir=r"180607-Alkanethiol_Extended_sims" # Analysis directory
    category_dir="EAM" # category directory
    specific_dir="EAM_310.15_K_2_nmDIAM_hexadecanethiol_CHARMM36_Trial_1" # spherical_310.15_K_2_nmDIAM_octanethiol_CHARMM36" # Directory within analysis_dir r"mdRun_363.15_6_nm_tBuOH_100_WtPercWater_spce_Pure"    
    
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
                         'moi_file'         :           "moi.xvg",                      # File with moments of inertia
                        }
    
    ### RUNNING ANALYSIS FOR GMX PRINCIPAL
    moi_results = analyze_gmx_principal( **input_details ) 
    
    
    
    #%%
    
    ### IMPORTING FUNCTION TO GET TRAJ
    from MDDescriptors.traj_tools.multi_traj import load_multi_traj_pickle, find_multi_traj_pickle, load_multi_traj_pickles
    from MDDescriptors.core.csv_tools import csv_info_add, csv_info_new, multi_csv_export, csv_info_export, csv_info_decoder_add
       
    ###############################################
    ### CLASS FUNCTION TO EXTRACT GMX PRINCIPAL ###
    ###############################################
    class extract_gmx_principal:
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
            self.csv_info = csv_info_new(pickle_name)
            ## ADDING CSV DECODER INFORMATION
            self.csv_info = csv_info_decoder_add(self.csv_info, pickle_name, decoder_type)
            ## CALCULATING ECCENTRICITY
            self.find_eccentricity()
            return
        ### FUNCTION TO FIND ECCENTRICITY
        def find_eccentricity(self ):
            '''
            The purpose of this function is to find the eccentricity
            '''
            eccentricity_avg = self.class_object.moi_extract.eccentricity_avg
            eccentricity_std = self.class_object.moi_extract.eccentricity_std
            ## STORING INPUTS
            self.csv_info = csv_info_add(self.csv_info, data_title = 'eccentricity_avg', data = [eccentricity_avg] )
            self.csv_info = csv_info_add(self.csv_info, data_title = 'eccentricity_std', data = [eccentricity_std] )
        
    ## DEFINING CLASS
    Descriptor_class = analyze_gmx_principal
    ## DEFINING DATE
    Date='181206'
    
    #%%
    ## DEFINING DESIRED DIRECTORY
    Pickle_loading_file=r"EAM_310.15_K_2_nmDIAM_butanethiol_CHARMM36_Trial_1"
    ## SINGLE TRAJ ANALYSIS
    multi_traj_results = load_multi_traj_pickle(Date, Descriptor_class, Pickle_loading_file )
    
    #%%
    #### RUNNING MULTIPLE CSV EXTRACTION
    multi_csv_export(Date = Date,
                    Descriptor_class = Descriptor_class,
                    desired_titles = None, 
                    export_class = extract_gmx_principal,
                    export_text_file_name = 'extract_gmx_principal',
                    )    
    
    
    
    
    