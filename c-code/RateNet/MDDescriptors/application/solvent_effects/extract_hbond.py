# -*- coding: utf-8 -*-
"""
extract_hbond.py
The purpose of this script is to extract hydrogen bonding information

CREATED ON: 04/13/2018

Author(s):
    Alex K. Chew (alexkchew@gmail.com)
"""

## MATH FUNCTIONS
import numpy as np

### IMPORTING FUNCTION TO GET TRAJ
from MDDescriptors.traj_tools.multi_traj import load_multi_traj_pickle, find_multi_traj_pickle, load_multi_traj_pickles
from MDDescriptors.application.solvent_effects.h_bond import calc_hbond
from MDDescriptors.core.csv_tools import csv_info_add, csv_info_new, multi_csv_export, csv_info_export, csv_info_decoder_add

### FUNCTION TO FIND AVERAGE FOR EACH DICTIONARY KEY
def find_avg_dict_key( my_dict ):
    ''' This finds average for each key in the dictionary '''
    avg_dict = { each_key: np.mean(my_dict[each_key]) for each_key in my_dict.keys() }
    return avg_dict

### FUNCTION TO FIND AVG FOR DONOR
def find_avg_acceptor_donor(acceptor_donor_dict, donor_acceptor_key=['num_donor','num_acceptor']):
    ''' This function finds the average for acceptor/donor array '''
    avg_dict = { each_key:{each_donor_acceptor_pair: np.mean(acceptor_donor_dict[each_key][each_donor_acceptor_pair]) 
                            for each_donor_acceptor_pair in donor_acceptor_key} for each_key in acceptor_donor_dict.keys() }
    return avg_dict

#################################################################################
### CLASS FUNCTION TO EXTRACT CALCULATE PREFERENTIAL INTERACTION COEFFICIENTS ###
#################################################################################
class extract_hbond:
    '''
    The purpose of this function is to extract preferential interaction coefficient calculations
    INPUTS:
        pref_int: class from calc_pref_interaction_coefficients
        pickle_name: name of the directory of the pickle
    OUTPUTS:
        csv_info: updated csv info
    '''
    ### INITIALIZATION
    def __init__(self, results, pickle_name, decoder_type = 'solvent_effects'):
        ## STORING RESULTS
        self.results = results
        
        ## STORING INFORMATION FOR CSV
        self.csv_info = csv_info_new(pickle_name)
        
        ## ADDING CSV DECODER INFORMATION
        self.csv_info = csv_info_decoder_add(self.csv_info, pickle_name, decoder_type)
        
        ## FINDING AVG HYDROGEN BONDS
        self.hbond_btn_residues_avg = find_avg_dict_key( self.results.hbond_btn_residues )
        
        ## FINDING AVG NUMBER OF DONOTS
        self.acceptor_donor_hbond_avg = find_avg_acceptor_donor( self.results.solute_acceptor_donor_hbond )
        
        ## DEFINING KEYS OF INTEREST
        if len(self.results.solvent_name)>1:
            self.cosolvent_name =[ each_solvent for each_solvent in self.results.solvent_name if each_solvent != 'HOH'][0]
        
        ## DEFINING SOLUTE NAMES
        self.acceptor_solute_names = self.results.acceptors_solute_names
        
        ## LOOPING THROUGH RESIDUES AND INPUTTING
        for each_key in self.acceptor_donor_hbond_avg:
            current_name = each_key
            if len(self.results.solvent_name)>1:
                if self.cosolvent_name in current_name:
                    current_name = current_name.replace(self.cosolvent_name, 'cosolvent')
            ## LOOPING THROUGH DONOR AND ACCEPTOR
            for donor_and_acceptor in ['num_acceptor', 'num_donor']:
                ## DEFINING NAME OF DATA
                data_name = donor_and_acceptor + '_' + current_name
                self.csv_info = csv_info_add(self.csv_info, data_title = data_name, data = [self.acceptor_donor_hbond_avg[each_key][donor_and_acceptor]] )
        
                            

#%% MAIN SCRIPT
if __name__ == "__main__":
    ## DEFINING CLASS
    Descriptor_class = calc_hbond
    ## DEFINING DATE
    # Date='191102_PDO_gauche'
    Date='191114_THD_CHD'
    # '191102_PDO_trans'
    # '190317'
#    #%%
#    ## DEFINING DESIRED DIRECTORY
#    Pickle_loading_file=r"Mostlikely_433.15_6_nm_PDO_100_WtPercWater_spce_Pure"
#    # Pickle_loading_file=r"Mostlikely_433.15_6_nm_PDO_10_WtPercWater_spce_dioxane"
##    Pickle_loading_file=r"mdRun_433.15_6_nm_PDO_50_WtPercWater_spce_dmso"
#    ## SINGLE TRAJ ANALYSIS
#    multi_traj_results = load_multi_traj_pickle(Date, Descriptor_class, Pickle_loading_file )
#    ## EXTRACTION
#    traj_result = multi_traj_results
    
    
    #%%
    
    #%%
#    ## CSV EXTRACTION
#    extracted_results = extract_hbond(traj_result, Pickle_loading_file)

    #%%
    
    
#    exported_csv = csv_info_export(extracted_results.csv_info)

    #%%
    ##### MULTI TRAJ ANALYSIS
    # traj_results, list_of_pickles = load_multi_traj_pickles( Date, Descriptor_class)
    
    #### RUNNING MULTIPLE CSV EXTRACTION
    multi_csv_export(Descriptor_class = Descriptor_class,
                                        Date = Date, 
                                        desired_titles = None, # ['ligand_density_area_angs_per_ligand', 'final_number_adsorbed_ligands', 'num_ligands_per_frame'],
                                        export_class = extract_hbond,
                                        export_text_file_name = 'extract_hbond_THD_CHD',
                                        )    


