# -*- coding: utf-8 -*-
"""
extract_nanoparticle_bundling.py
This script extracts nanoparticle bundling groups

CREATED ON: 05/28/2018

AUTHOR(S):
    Alex K. Chew (alexkchew@gmail.com)
    
# CLASSES
    - extract_nanoparticle_bundling: extracts nanoparticle bundling groups
    
# FUNCTIONS
    - 
    
    
"""

import sys  
import numpy as np
# reload(sys)  
# sys.setdefaultencoding('utf8')

### IMPORTING FUNCTION TO GET TRAJ
from MDDescriptors.traj_tools.multi_traj import load_multi_traj_pickle, find_multi_traj_pickle, load_multi_traj_pickles
from MDDescriptors.core.csv_tools import csv_info_add, csv_info_new, multi_csv_export, csv_info_export, csv_info_decoder_add
## IMPORTING FUNCTION THAT WAS USED
from MDDescriptors.application.nanoparticle.nanoparticle_find_bundled_groups import calc_nanoparticle_bundling_groups, calc_nanoparticle_bundling_trans_ratio, find_all_assigned_vs_unassigned_ligands
import MDDescriptors.core.calc_tools as calc_tools
### IMPORTING GLOBAL VARIABLES
from MDDescriptors.global_vars.plotting_global_vars import COLOR_LIST, LABELS, LINE_STYLE
from MDDescriptors.core.plot_tools import create_plot, save_fig_png, create_3d_axis_plot

#######################################################
### CLASS FUNCTION TO EXTRACT STRUCTURAL PROPERTIES ###
#######################################################
class extract_nanoparticle_bundling:
    '''
    The purpose of this function is to extract nanoparticle bundling information
    INPUTS:
        bundling_class: class from nanoparticle_structure
        pickle_name: name of the directory of the pickle
    OUTPUTS:
        csv_info: updated csv info 
    '''
    ### INITIALIZATION
    def __init__(self, bundling_class, pickle_name, decoder_type = 'nanoparticle'):
        ## STORING PICKLE NAME
        self.pickle_name = pickle_name
        ## STORING STRUCTURE
        self.bundling = bundling_class
        ## STORING INFORMATION FOR CSV
        self.csv_info = csv_info_new(pickle_name)
        ## ADDING CSV DECODER INFORMATION
        self.csv_info = csv_info_decoder_add(self.csv_info, pickle_name, decoder_type)
        
        ## AVERAGE BUNDLING
        self.find_avg_bundling()
        
        ## NONASSIGNMENTS
        self.find_avg_nonassignments()
        
        ## GROUP SIZE
        self.find_avg_group_size()
        
        ## FINDING TOTAL LIGANDS BUNDLED
        self.find_total_ligs_assigned()
        
        ## TRANS RATIO
        # self.find_trans_ratio()
        
        ## PLOTTING TRANS RATIO AS A HISTOGRAM (Comment out if you do not want the plot!)
        # self.plot_bundling_groups_trans_distribution()
        
        return
        
    ### FUNCTION TO FIND AVERAGE AND STD BUNDLING GROUPS
    def find_avg_bundling(self, ):
        '''
        The purpose of this function is to extract average bundling
        INPUTS:
            self: class object
        OUTPUTS:
            Updated self.csv_info
        '''
        ## STORING INPUTS
        self.csv_info = csv_info_add(self.csv_info, data_title = 'bundling', data = [self.bundling.results_avg_std_bundling_grps] )
        return
    
    ### FUNCTION TO CALCULATE THE AVERAGE NUMBER OF UNASSIGNED GROUP FOR THE SIMULATION
    def find_avg_nonassignments(self, ):
        '''
        The purpose of this function is to calculate the average number of non-assignments
        INPUTS:
            self: class object
        OUTPUTS:
            avg_std_assignments: [dict] dictionary with the average and std of nonassignments
            Updated self.csv_info
        '''
        ## STORING INPUTS
        self.csv_info = csv_info_add(self.csv_info, data_title = 'non_assignments', data = [self.bundling.results_avg_std_nonassignments] )
        return

    ### FUNCTION TO FIND THE AVERAGE GROUP SIZE
    def find_avg_group_size(self):
        '''
        The purpose of this function is to find the average group size of ligands
        INPUTS:
            self: class object
        OUTPUTS:
            Updated self.csv_info
        '''        
        ## STORING INPUTS
        self.csv_info = csv_info_add(self.csv_info, data_title = 'group_size', data = [self.bundling.results_avg_std_group_size] )
        return
    
    ### FUNCTION TO FIND TOTAL ASSIGNMENTS
    def find_total_ligs_assigned(self,):
        '''
        The purpose of the function is to find the number of ligands bundled divided by the total ligands
        INPUTS:
            self: class object
        OUTPUTS:
            
        '''
        ## FINDING ASSIGNED AND UNASSIGNED LIGANDS
        self.assigned_ligands, self.unassigned_ligands = find_all_assigned_vs_unassigned_ligands( ligand_grp_list = self.bundling.lig_grp_list )  
        
        ## GETTING TOTAL ASSIGNED LIGANDS DIVIDED BY TOTAL LIGANDS
        assigned_ligand_totals = [ x.size for x in self.assigned_ligands]
        
        ## GETTING RATIO
        ratio_assigned_to_total = np.array(assigned_ligand_totals) / self.bundling.structure_np.total_ligands
        
        ## FINDING AVERAGE TOTAL LIGANDS AND ERROR
        results_avg_std_ratio_assigned_to_total = calc_tools.calc_avg_std_of_list( ratio_assigned_to_total )
        
        ## STORING INPUTS
        self.csv_info = csv_info_add(self.csv_info, data_title = 'ratio_assigned', data = [results_avg_std_ratio_assigned_to_total] )
        
        return
        
    ### FUNCTION TO INCLUDE TRANS RATIO
    def find_trans_ratio(self):
        '''
        The purpose of this function is to find the trans ratio of assigned & unassigned ligands
        INPUTS:
            self: class object
        OUTPUTS:
        '''
        ## STORING INPUTS
        self.csv_info = csv_info_add(self.csv_info, data_title = 'trans_ratio_unassigned', data = [self.bundling.trans_ratio.results_avg_std_trans_ratio_unassigned_ligands] )
        self.csv_info = csv_info_add(self.csv_info, data_title = 'trans_ratio_assigned', data = [self.bundling.trans_ratio.results_avg_std_trans_ratio_assigned_ligands] )
        return
    
    ### FUNCTION TO PLOT HISTOGRAM OF TRANS RATIO
    def plot_bundling_groups_trans_distribution(self, bin_width = 0.005, save_fig = True):
        '''
        The purpose of this script is to plot the distance distribution of gold based distance matrix
        INPUTS:
            bundling_groups: [class] bundling groups from "calc_nanoparticle_bundling_groups" class
            bin_width: [float] bin width for the histogram data
            save_fig: [logical] True if you want to save the plot
        OUTPUTS:
            plot of trans distribution for bundling groups
        FUNCTIONS:
            get_clean_data: cleans dihedral data in terms of nan's and outputs the correct number of dihedrals considered for trans ratio calculations
        '''
        ### FUNCTION TO GET THE CLEAN DATA
        def get_clean_data(trans_dihedrals, total_trans_dihedrals):
            '''
            The purpose of this script is to simply remove all nans, and correctly generate total number of trans
            INPUTS:
                trans_dihedrals: [list] trans ratio as a function of frames
                total_trans_dihedrals: [list] list of total number of dihedrals considered per trans ratio
            OUTPUTS:
                clean_trans_dihedrals:[numpy array] trans_dihedral without np.nans
                num_trans_dihedrals: [numpy array] total number of trans dihedrals found per frame basis
            '''
            ## CONVERTING TO NUMPY
            trans_dihedrals = np.array(trans_dihedrals)
            ## FINDING TOTAL NUMBER OF TRANS FOUND
            num_trans_dihedrals =trans_dihedrals* np.array(total_trans_dihedrals)
            ## REMOVING NAN FROM DATA
            clean_trans_dihedrals = trans_dihedrals[~np.isnan(trans_dihedrals)]
            ## SETTING TOTAL NUMBER OF NANS TO ZEROS
            num_trans_dihedrals[np.isnan(num_trans_dihedrals)] = 0
            
            return clean_trans_dihedrals, num_trans_dihedrals
        
        ## DEFINING BUNDLING GROUPS
        bundling_groups = self.bundling
        
        ## CREATING FIGURE
        fig, ax = create_plot()
        ## DEFINING TITLE
        ax.set_title('Trans ratio distribution between assigned and unassigned ligands')
        ## DEFINING X AND Y AXIS
        ax.set_xlabel('Trans ratio', **LABELS)
        ax.set_ylabel('Normalized number of occurances', **LABELS)  
        ## DEFINING LIMITS
        ax.set_ylim([0, 100])
        ## CREATING BINS
        bins = np.arange(0, 1, bin_width)
        
        #################################
        ### PLOTTING ASSIGNED LIGANDS ###
        #################################
        ## FINDING DATA ASSOCIATED WITH ASSIGNED LIGANDS
        assigned_lig_trans_dihedrals, assigned_lig_total_trans = get_clean_data( trans_dihedrals = bundling_groups.trans_ratio.trans_ratio_assigned_ligands_list,
                                                                                 total_trans_dihedrals = bundling_groups.trans_ratio.trans_ratio_assigned_total_dihedrals,
                                                                                 )
        
        
        ## PLOTTING HISTOGRAM
        n, bins, patches = ax.hist(assigned_lig_trans_dihedrals, bins = bins, color  = 'k' , density=True, label="Assigned ligands" )
        ## DRAWING AVERAGE
        ax.axvline( x = bundling_groups.trans_ratio.results_avg_std_trans_ratio_assigned_ligands['avg'], linestyle='--', color = 'k', )
        
        ###################################
        ### PLOTTING UNASSIGNED LIGANDS ###
        ###################################      
        ## FINDING DATA ASSOCIATED WITH UNASSIGNED LIGANDS
        unassigned_lig_trans_dihedrals, unassigned_lig_total_trans = get_clean_data( trans_dihedrals = bundling_groups.trans_ratio.trans_ratio_unassigned_ligands_list,
                                                                                 total_trans_dihedrals = bundling_groups.trans_ratio.trans_ratio_unassigned_total_dihedrals,
                                                                                 )

        ## PLOTTING HISTOGRAM
        ax.hist(unassigned_lig_trans_dihedrals, bins = bins, color  = 'r' , density=True, label="Unassigned ligands" )
        ## DRAWING AVERAGE
        ax.axvline( x = bundling_groups.trans_ratio.results_avg_std_trans_ratio_unassigned_ligands['avg'], linestyle='--', color = 'r', )
    
        ### DRAWING FOR ACCOUNTING ALL LIGANDS
        ## DEFINING DATA
        ## CONVERTING NAN'S TO ZEROS
        
        ############################
        ### PLOTTING ALL LIGANDS ###
        ############################
        ### DEFINING THE DATA AS THE AVERAGE OF THE ASSIGNED AND UNASSIGNED LIGANDS
        data = (unassigned_lig_total_trans + assigned_lig_total_trans) / (np.array(bundling_groups.trans_ratio.trans_ratio_assigned_total_dihedrals) + np.array(bundling_groups.trans_ratio.trans_ratio_unassigned_total_dihedrals) )
        ## PLOTTING HISTOGRAM
        ax.hist(data, bins = bins, color  = 'g' , density=True, label="All ligands" , alpha = 0.50)
        ## DRAWING AVERAGE
        ax.axvline( x = np.mean(data), linestyle='--', color = 'g', )
    
        ## DRAWING LEGEND
        ax.legend()
        
        ## STORING IMAGE
        save_fig_png(fig = fig, label = 'trans_dist_' + self.pickle_name, save_fig = save_fig)
        
        return
    
    


#%% MAIN SCRIPT
if __name__ == "__main__":
    ## DEFINING CLASS
    Descriptor_class = calc_nanoparticle_bundling_groups
    ## DEFINING DATE
    Date='180824'
    #%%
    ## DEFINING DESIRED DIRECTORY
    Pickle_loading_file=r"EAM_310.15_K_2_nmDIAM_hexadecanethiol_CHARMM36_Trial_3"
    ## SINGLE TRAJ ANALYSIS
    multi_traj_results = load_multi_traj_pickle(Date, Descriptor_class, Pickle_loading_file )
    print(multi_traj_results.structure_np.total_ligands)
    
    #%%
    ## EXTRACTION
    traj_results = multi_traj_results
    ## CSV EXTRACTION
    extracted_results = extract_nanoparticle_bundling(traj_results, Pickle_loading_file)
    
    #%%
    ##### MULTI TRAJ ANALYSIS
    traj_results, list_of_pickles = load_multi_traj_pickles( Date, Descriptor_class)
    #%%

    #%%
    #### RUNNING MULTIPLE CSV EXTRACTION
    multi_csv_export(Date = Date,
                    Descriptor_class = Descriptor_class,
                    desired_titles = None, 
                    export_class = extract_nanoparticle_bundling,
                    export_text_file_name = 'extract_nanoparticle_bundling',
                    )    
    # traj_results = traj_results, 
    # list_of_pickles = list_of_pickles,