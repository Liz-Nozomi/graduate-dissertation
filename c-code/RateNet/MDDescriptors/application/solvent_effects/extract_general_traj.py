# -*- coding: utf-8 -*-
"""
extract_general_traj.py
This script extracts the results from general_traj_info.py

CLASSES:
    extract_nanoparticle_structure: extracts nanoparticle structure information

CREATED ON: 04/20/2018

AUTHOR(S):
    Alex K. Chew (alexkchew@gmail.com)

"""

### IMPORTING FUNCTION TO GET TRAJ
from MDDescriptors.traj_tools.multi_traj import load_multi_traj_pickle, find_multi_traj_pickle, load_multi_traj_pickles
from MDDescriptors.core.csv_tools import csv_info_add, csv_info_new, multi_csv_export, csv_info_export, csv_info_decoder_add
## IMPORTING FUNCTION THAT WAS USED
from MDDescriptors.core.general_traj_info import general_traj_info

#######################################################
### CLASS FUNCTION TO EXTRACT STRUCTURAL PROPERTIES ###
#######################################################
class extract_general_traj_info:
    '''
    The purpose of this class is to extract the nanoparticle structure information
    INPUTS:
        results: class from general traj information
        pickle_name: name of the directory of the pickle
    OUTPUTS:
        csv_info: updated csv info
    '''
    ### INITIALIZATION
    def __init__(self, results, pickle_name, decoder_type = 'solvent_effects'):
        ## STORING STRUCTURE
        self.results = results
        ## STORING INFORMATION FOR CSV
        self.csv_info = csv_info_new(pickle_name)
        ## ADDING CSV DECODER INFORMATION
        self.csv_info = csv_info_decoder_add(self.csv_info, pickle_name, decoder_type)
        ## ADDING INFORMATION
        self.csv_info = csv_info_add(self.csv_info, data_title = 'ens_vol(nm3)', data = [self.results.ens_volume] )
        self.csv_info = csv_info_add(self.csv_info, data_title = 'ens_length(nm)', data = [self.results.ens_length] )
        self.csv_info = csv_info_add(self.csv_info, data_title = 'total_atoms', data = [self.results.total_atoms] )
        self.csv_info = csv_info_add(self.csv_info, data_title = 'total_frames', data = [self.results.total_frames] )
        
        ## ADDING RESIDUE INFORMATION
        if decoder_type == "solvent_effects":
            for each_residue in results.residues.keys():
                ## SEEING IF WATER
                if each_residue == 'HOH':
                    self.csv_info = csv_info_add(self.csv_info, data_title = 'N_H2O', data = [self.results.residues[each_residue]] )
                else:
                    self.csv_info = csv_info_add(self.csv_info, data_title = 'N_org', data = [self.results.residues[each_residue]] )
        return


#%% MAIN SCRIPT
if __name__ == "__main__":
    ## DEFINING CLASS
    Descriptor_class = general_traj_info
    ## DEFINING DATE
    Date='190719' # '180719-FINAL'
    ## DEFINING DESIRED DIRECTORY
    Pickle_loading_file=r"mdRun_300.00_6_nm_NoSolute_10_WtPercWater_spce_dmso"
    #%%
    ## SINGLE TRAJ ANALYSIS
    multi_traj_results = load_multi_traj_pickle(Date, Descriptor_class, Pickle_loading_file )
    ## EXTRACTION
    traj_results = multi_traj_results
    ## CSV EXTRACTION
    extracted_results = extract_general_traj_info(traj_results, Pickle_loading_file)
    
    #%%
    ##### MULTI TRAJ ANALYSIS
    traj_results, list_of_pickles = load_multi_traj_pickles( Date, Descriptor_class)
    #%%
    #### RUNNING MULTIPLE CSV EXTRACTION
    multi_csv_export(Date = Date, 
                    Descriptor_class = Descriptor_class,
                    desired_titles = None, 
                    export_class = extract_general_traj_info,
                    export_text_file_name = 'extract_general_traj_info',
                    )    
  
    
    