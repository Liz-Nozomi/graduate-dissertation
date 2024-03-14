# -*- coding: utf-8 -*-
"""
extract_pref_int.py
The purpose of this script is to extract preferential interaction coefficients

CREATED ON: 04/13/2018

Author(s):
    Alex K. Chew (alexkchew@gmail.com)
"""

## MATH FUNCTIONS
import numpy as np

### IMPORTING FUNCTION TO GET TRAJ
from MDDescriptors.traj_tools.multi_traj import load_multi_traj_pickle, find_multi_traj_pickle, load_multi_traj_pickles
from MDDescriptors.application.solvent_effects.preferential_interaction_coefficients import calc_pref_interaction_coefficients
from MDDescriptors.core.csv_tools import csv_info_add, csv_info_new, multi_csv_export, csv_info_export, csv_info_decoder_add

#################################################################################
### CLASS FUNCTION TO EXTRACT CALCULATE PREFERENTIAL INTERACTION COEFFICIENTS ###
#################################################################################
class extract_pref_interaction_cofficients:
    '''
    The purpose of this function is to extract preferential interaction coefficient calculations
    INPUTS:
        pref_int: class from calc_pref_interaction_coefficients
        pickle_name: name of the directory of the pickle
    OUTPUTS:
        csv_info: updated csv info
    '''
    ### INITIALIZATION
    def __init__(self, pref_int, pickle_name, decoder_type = 'solvent_effects'):
        ## STORING PREF INT
        self.pref_int = pref_int
        
        ## STORING INFORMATION FOR CSV
        self.csv_info = csv_info_new(pickle_name)
        
        ## ADDING CSV DECODER INFORMATION
        self.csv_info = csv_info_decoder_add(self.csv_info, pickle_name, decoder_type)
        
        ## FINDING PREFERENTIAL INTERACTION COEFFICIENT
        self.find_pref_int_coeff()
        
        ## EXTRATION OF RDF
        self.find_rdf_for_pref_int()
        
        ## FINDING OXYGEN PREFERENTIAL INTERACTION COEFFICIENT
        # self.find_pref_int_coeff_oxy()
        
        ## FINDING EXCESS COORDINATION NUMBER
        # self.find_excess_coord_num()
        
    ### FUNCTION TO EXTRACT PREFERENTIAL INTERACTION COEFFICIENTS
    def find_pref_int_coeff(self):
        '''
        The purpose of this function is to find the preferential interaction coefficient (e.g. nGamma, etc)
        INPUTS:
            self: class object
        OUTPUTS:
            Updated self.csv_info
        '''
        ## STORING INPUTS
        ## LOOPING THROUGH EACH SOLUTE
        for solute_index, solute_name in enumerate(self.pref_int.solute_name):
            try:
                self.csv_info = csv_info_add(self.csv_info, data_title = solute_name + '_ngamma_23', data = [self.pref_int.pref_int_coeff[solute_index]['ngamma_23']] )
            except Exception:
                self.csv_info = csv_info_add(self.csv_info, data_title = solute_name + '_ngamma_23', data = [ np.nan ] )
            self.csv_info = csv_info_add(self.csv_info, data_title = solute_name + '_cutoff_radius(nm)', data = [self.pref_int.equil_radius[solute_index]] )
        ''' Uncomment if you want the other parameters
        self.csv_info = csv_info_add(self.csv_info, data_title = 'n_1_local_avg', data = [self.pref_int.pref_int_coeff['n_1_local_avg']] )
        self.csv_info = csv_info_add(self.csv_info, data_title = 'n_1_bulk_avg', data = [self.pref_int.pref_int_coeff['n_1_bulk_avg']] )
        self.csv_info = csv_info_add(self.csv_info, data_title = 'n_3_local_avg', data = [self.pref_int.pref_int_coeff['n_3_local_avg']] )
        self.csv_info = csv_info_add(self.csv_info, data_title = 'n_3_bulk_avg', data = [self.pref_int.pref_int_coeff['n_3_bulk_avg']] )
        '''
        
        
    ### FUNCTION TO EXTRACT RDF
    def find_rdf_for_pref_int(self):
        ''' Function to extract RDF used for preferential interaction coefficient cutoff    '''
        ## LOOPING THROUGH EACH SOLUTE
        for solute_index, solute_name in enumerate(self.pref_int.solute_name):
            ## DEFINING RDF TITLES
            title_names = [ "water_rdf", "cosolvent_rdf" ]
            if len(self.pref_int.solvent_name) == 2:
                solvent_indices = [ self.pref_int.water_index, self.pref_int.cosolvent_index   ]
            else:
                solvent_indices = [ self.pref_int.water_index ]
            ## LOOPING THROUGH EACH INDEX
            for idx, solvent_index in enumerate(  solvent_indices  ):
                ## ATTEMPTING
                try:
                    ## IDENTIFYING WATER INDEX
                    self.csv_info = csv_info_add(self.csv_info, data_title = solute_name + title_names[idx], 
                                                 data = [ 
                                                         # self.pref_int.rdf.rdf_r[solute_index][solvent_index],
                                                         self.pref_int.rdf.rdf_g_r[solute_index][solvent_index],
                                                         ],
                                                 labels=['g_r']) # 'r(nm)',
                except Exception:
                    pass
            
#            
#            ## IDENTIFYING WATER INDEX
#            self.csv_info = csv_info_add(self.csv_info, data_title = solute_name + 'water_rdf', 
#                                         data = [ 
#                                                 self.pref_int.rdf.rdf_r[solute_index][self.pref_int.water_index],
#                                                 self.pref_int.rdf.rdf_g_r[solute_index][self.pref_int.water_index],
#                                                 ],
#                                         labels=['r(nm)', 'g_r'])
#            ## IDENTIFYING COSOLVENT INDEX
#            ## IDENTIFYING WATER INDEX
#            try:
#                self.csv_info = csv_info_add(self.csv_info, data_title = solute_name + 'cosolvent_rdf', 
#                                             data = [ 
#                                                     self.pref_int.rdf.rdf_r[solute_index][self.pref_int.cosolvent_index],
#                                                     self.pref_int.rdf.rdf_g_r[solute_index][self.pref_int.cosolvent_index],
#                                                     ],
#                                             labels=['r(nm)', 'g_r'])
#            except Exception:
#                pass
             
        
    
    
    ### FUNCTION TO EXTRACT EACH OXYGEN PREFERENTIAL INTERACTION COEFFICIENTS
    def find_pref_int_coeff_oxy(self):
        '''
        The purpose of this function is to find the preferential interaction coefficients of each oxygen group
        INPUTS:
            self: class object
        OUTPUTS:
            Updated self.csv_info
        '''
        ## LOOPING FOR EACH OXYGEN
        for index, each_oxygen in enumerate(self.pref_int.rdf.rdf_oxy_names):
            self.csv_info = csv_info_add(self.csv_info, data_title = self.pref_int.solute_name + '_' + each_oxygen + '_ngamma_23', data = [self.pref_int.pref_int_coeff_solute_atom[each_oxygen]['ngamma_23']] )
            self.csv_info = csv_info_add(self.csv_info, data_title = self.pref_int.solute_name + '_' + each_oxygen + '_cutoff_radius(nm)', data = [self.pref_int.solute_atom_cutoff_radius[index]] )
            
    ### FUNCTION TO EXTRACT EXCESSC OORDINATION NUMBER
    def find_excess_coord_num(self):
        '''
        The purpose of this function is toe find the excess corodination number
        INPUTS:
            self: class object
        OUTPUTS:
            Updated self.csv_info
        '''
        ## STORING INPUTS
        ## LOOPING THROUGH EACH SOLVENT
        for solvent_index, each_solvent in enumerate(self.pref_int.solvent_name):
            if each_solvent == 'HOH':
                self.csv_info = csv_info_add(self.csv_info, data_title = 'excess_coord_num_HOH', data = [self.pref_int.excess_coord_num[solvent_index] ] )
            else:
                self.csv_info = csv_info_add(self.csv_info, data_title = 'excess_coord_num_COS', data = [self.pref_int.excess_coord_num[solvent_index] ] )
            ## LOOPING THROUGH EACH OXYGEN
            if self.pref_int.want_oxy_pref_int_coeff is True:
                for oxygen_index, each_oxygen in enumerate(self.pref_int.rdf.rdf_oxy_names):
                    if each_solvent == 'HOH':
                        self.csv_info = csv_info_add(self.csv_info, data_title = 'excess_coord_num_oxy_' + each_oxygen +'_HOH', data = [self.pref_int.excess_coord_num_oxy[solvent_index][oxygen_index] ] )
                    else:
                        self.csv_info = csv_info_add(self.csv_info, data_title = 'excess_coord_num_oxy_' + each_oxygen +'_COS', data = [self.pref_int.excess_coord_num_oxy[solvent_index][oxygen_index] ] )
                    
                    

#%% MAIN SCRIPT
if __name__ == "__main__":
    ## DEFINING CLASS
    Descriptor_class = calc_pref_interaction_coefficients
    ## DEFINING DATE
    Date='190312'
    #%%
    ## DEFINING DESIRED DIRECTORY
    Pickle_loading_file=r"mdRun_433.15_6_nm_PDO_10_WtPercWater_spce_dioxane"
#    Pickle_loading_file=r"mdRun_433.15_6_nm_PDO_50_WtPercWater_spce_dmso"
    ## SINGLE TRAJ ANALYSIS
    multi_traj_results = load_multi_traj_pickle(Date, Descriptor_class, Pickle_loading_file )
    ## EXTRACTION
    traj_result = multi_traj_results
    #%%
    ## CSV EXTRACTION
    extracted_results = extract_pref_interaction_cofficients(traj_result, Pickle_loading_file)

    #%%
    
    
    exported_csv = csv_info_export(extracted_results.csv_info)

    #%%
    ##### MULTI TRAJ ANALYSIS
    # traj_results, list_of_pickles = load_multi_traj_pickles( Date, Descriptor_class)
    
    #### RUNNING MULTIPLE CSV EXTRACTION
    multi_csv_export(Descriptor_class = Descriptor_class,
                                        Date = Date, 
                                        desired_titles = None, # ['ligand_density_area_angs_per_ligand', 'final_number_adsorbed_ligands', 'num_ligands_per_frame'],
                                        export_class = extract_pref_interaction_cofficients,
                                        export_text_file_name = 'extract_pref_int',
                                        )    


