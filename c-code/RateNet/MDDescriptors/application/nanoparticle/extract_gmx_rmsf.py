# -*- coding: utf-8 -*-
"""
extract_gmx_rmsf.py
The purpose of this function is to extract gmx_rmsf code and run subsequent calculations on it

IMPORTANT NOTES
---------------
- This script assumes you already ran gmx rmsf. In particular, we are interested in the average rmsf value
- This script only takes into account rmsf.xvg

CREATED ON: 10/04/2018

AUTHOR(S):
    Alex K. Chew (alexkchew@gmail.com)

"""
### IMPORTING MODULES
import numpy as np
import MDDescriptors.core.import_tools as import_tools # Loading trajectory details

#######################################
### CLASS FUNCTION TO READ XVG FILE ###
#######################################
class analyze_gmx_rmsf:
    '''
    The purpose of this function is to analyze the rmsf function using GROMACS
    

    
    '''
    def __init__(self, traj_data, xvg_file ):
        
        ## DEFINING FULL PATH
        self.file_path = traj_data.directory + '/' + xvg_file
        
        ## READING THE FILE
        self.data_full, self.data_extract = import_tools.read_xvg(self.file_path)
    
        ## VARIABLE EXTRACTION
        self.define_variables()
        
        ## ANALYSIS
        self.avg_rmsf = np.mean(self.output_rmsf_per_atom)
    
    ## EXTRACTION OF VARIABLES
    def define_variables(self, ):
        '''
        The purpose of this function is to define variables based on extracted results
        INPUTS:
            self.data_extract: [list] 
                extracted data in a form of a list (i.e. no comments)
        OUTPUTS:
            self.output_rmsf_per_atom: [np.array, shape=(num_atoms, 1)] 
                rmsf per atom average across time frame
        '''
        self.output_rmsf_per_atom = np.array([ x[1] for x in self.data_extract]).astype("float") # major, middle, and minor axis




#%% MAIN SCRIPT
if __name__ == "__main__":
    
    ### DIRECTORY TO WORK ON    
    analysis_dir=r"180928-2nm_RVL_proposal_runs_fixed" # Analysis directory
    category_dir="EAM" # category directory
    specific_dir="EAM_310.15_K_2_nmDIAM_ROT001_CHARMM36_Trial_1"
    
    ### DEFINING PATH TO ANALYSIS DIRECTORY
    path2AnalysisDir=r"R:\scratch\nanoparticle_project\simulations\190520-2nm_ROT_Sims_updated_forcefield_new_ligands\EAM_300.00_K_2_nmDIAM_ROT012_CHARMM36jul2017_Trial_1"
    
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
    output_xvg="rmsf.xvg"
    
    ### DEFINING INPUT VARIABLES
    input_details = {   'traj_data'         :           traj_data,                      # Trajectory information
                         'xvg_file'         :           output_xvg,                      # File with moments of inertia
                        }
    
    ## RUNNING ANALYSIS FOR GMX RMSF
    results = analyze_gmx_rmsf( **input_details ) 

    #%%
    ##################### EXTRACTION PROCESS #####################
    
    ### IMPORTING FUNCTION TO GET TRAJ
    from MDDescriptors.traj_tools.multi_traj import load_multi_traj_pickle, find_multi_traj_pickle, load_multi_traj_pickles
    from MDDescriptors.core.csv_tools import csv_info_add, csv_dict, csv_info_new, multi_csv_export, csv_info_export, csv_info_decoder_add

    ###########################################
    ### CLASS FUNCTION TO EXTRACT GMX HBOND ###
    ###########################################
    class extract_gmx_rmsf:
        '''
        The purpose of this function is to extract the analyze_gmx_rmsf function
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
            self.csv_dict.add( data_title = 'avg_rmsf', data =  [ class_object.avg_rmsf ] )
            ## REDEFINING CSV INFO            
            self.csv_info = self.csv_dict.csv_info
            return

    ## DEFINING CLASS
    Descriptor_class = analyze_gmx_rmsf
    ## DEFINING DATE
    Date='181206'
    #%%
    ## DEFINING DESIRED DIRECTORY
    Pickle_loading_file=r"EAM_310.15_K_2_nmDIAM_ROT001_CHARMM36_Trial_1"
    ## SINGLE TRAJ ANALYSIS
    multi_traj_results = load_multi_traj_pickle(Date, Descriptor_class, Pickle_loading_file )
    ## EXTRACTING
    extracted_results = extract_gmx_rmsf( multi_traj_results,  Pickle_loading_file)
    
    
    #%%
    #### RUNNING MULTIPLE CSV EXTRACTION
    multi_csv_export(Date = Date,
                    Descriptor_class = Descriptor_class,
                    desired_titles = None, 
                    export_class = extract_gmx_rmsf,
                    export_text_file_name = 'extract_gmx_rmsf',
                    )    
    

    
    
    