# -*- coding: utf-8 -*-
"""
gmx_hbond.py

IMPORTANT NOTES
---------------
- This script assumes you already ran gmx hbond. In particular, we are interested in the number of hydrogen bonds
- Currently, this only takes in hbnum.xvg (could expand on this later)

CREATED ON: 07/26/2019

AUTHOR(S):
    Alex K. Chew (alexkchew@gmail.com)
"""
## IMPORTING MODULES
import os

## IMPORTING CUSTOM MODULES
import MDDescriptors.core.pickle_tools as pickle_tools

## IMPORTING FUNCTION THAT WAS USED
from MDDescriptors.application.nanoparticle.extract_gmx_hbond import analyze_gmx_hbond
from MDDescriptors.application.nanoparticle.extract_gmx_principal import analyze_gmx_principal

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




#%% MAIN SCRIPT
if __name__ == "__main__":
    
    ## DEFINING PICKLE DIRECTORY TO LOOK INTO
    pickle_dir = r"R:\scratch\nanoparticle_project\scripts\analysis_scripts\PICKLE_ROT"
    ## DEFINING DESCRIPTOR CLASS
    descriptor_class = analyze_gmx_principal
    ## DEFINING PICKLE NAME
    pickle_name = r"EAM_300.00_K_2_nmDIAM_ROT001_CHARMM36jul2017_Trial_1"
    
    ## LOADING THE PICKLE
    results = pickle_tools.load_pickle_from_descriptor(pickle_dir = pickle_dir,
                                                       descriptor_class = descriptor_class,
                                                       pickle_name = pickle_name,
                                                       )
    
    #%%
    
    current_value = get_class_attribute(results, ['moi_extract', 'eccentricity_avg'])
    
    
    #%%
    ## ATTRIBUTES FOR RESULTS
    # results.__getattribute__('avg_h_bond')
    
    ## EXTRACTING GMX HBOND
    extracted_results = extract_gmx_hbond(
                                            class_object = results,
                                            pickle_name = pickle_name,
                                            decoder_type = 'nanoparticle',
                                            )
    
    
    
    