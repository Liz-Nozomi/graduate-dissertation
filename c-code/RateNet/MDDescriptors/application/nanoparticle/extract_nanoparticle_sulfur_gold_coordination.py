# -*- coding: utf-8 -*-
"""
extract_nanoparticle_sulfur_gold_coordination.py
This script extracts the results from nanoparticle_sulfur_gold_coordination.py
Created on Mon Jul  2 15:33:42 2018

CREATED ON: 06/25/2018

AUTHOR(S):
    Alex K. Chew (alexkchew@gmail.com)
"""

### IMPORTING FUNCTION TO GET TRAJ
from MDDescriptors.traj_tools.multi_traj import load_multi_traj_pickle, find_multi_traj_pickle, load_multi_traj_pickles, load_multi_traj_multi_analysis_pickle, find_class_from_list
from MDDescriptors.core.csv_tools import csv_info_add, csv_info_new, multi_csv_export, csv_info_export, csv_info_decoder_add,csv_dict

### IMPORTING GLOBAL VARIABLES
from MDDescriptors.global_vars.plotting_global_vars import COLOR_LIST, LABELS, LINE_STYLE
from MDDescriptors.core.plot_tools import create_plot, save_fig_png, create_3d_axis_plot
import MDDescriptors.core.calc_tools as calc_tools
### MATH MODULES
import numpy as np


######################################################################################
### CLASS FUNCTION TO CORRELATE BETWEEN BUNDLING GROUPS AND NEARBY WATER STRUCTURE ###
######################################################################################
class extract_nanoparticle_sulfur_gold_coordination_bundling_groups:
    '''
    The purpose of this function is to extract sulfur gold coordination in conjunction with bundling groups
    INPUTS:
        analysis_classes: [list] list of multiple classes:
            calc_nanoparticle_bundling_groups: class containing bundling group information
            nanoparticle_sulfur_gold_coordination: class containing gold-sulfur coordination
    OUTPUTS:
        ## INPUT CLASSES
            self.sulfur_gold_coord_class: [class] sulfur-gold coordination class
            self.bundling_class: [class] nanoparticle bundling group class
    
    '''
    ### INITIALIZATION
    def __init__(self, analysis_classes, pickle_name, decoder_type = 'nanoparticle'):
        ## STORING INITIAL VARIABLES
        self.pickle_name = pickle_name
        
        ## DEFINING CLASSES
        self.bundling_class = find_class_from_list( analysis_classes, class_name = 'calc_nanoparticle_bundling_groups' )
        self.sulfur_gold_coord_class = find_class_from_list( analysis_classes, class_name = 'nanoparticle_sulfur_gold_coordination' )
        
        ## STORING INFORMATION FOR CSV
        self.csv_dict = csv_dict(file_name = pickle_name, decoder_type = decoder_type )
        
        ## FINDING ASSIGNED AND UNASSIGNED LIGANDS
        self.assigned_ligands, self.unassigned_ligands = find_all_assigned_vs_unassigned_ligands( ligand_grp_list = self.bundling_class.lig_grp_list )            
        
        ## FINDING FRACTIONS OF FACETS
        self.ratio_of_facets = self.sulfur_gold_coord_class.ratio_of_facet_surface_atoms
        
        ## FINDING INFORMATION ABOUT THE GOLD
        self.find_ratio_gold_facets()
        
        ## FINDING AVERAGE FACETS
        self.avg_facets = self.find_average_facets_assigned_vs_unassigned_ligands()
        
        ## FINDING ENSEMBLE AVERAGE QUANTITIES
        self.calc_ensem_facets()
        
        ## PLOTTING HISTOGRAM
        # self.plot_bundling_groups_facet_distribution()
        
        ## STORING INTO CSV INFO
        self.csv_info = self.csv_dict.csv_info
        
        return
    
    ### FUNCTION TO STORE INFORMATION ABOUT THE GOLD
    def find_ratio_gold_facets(self):
        '''
        The purpose of this script is to find the ratio of gold facet atoms to total number of atoms
        INPUTS:
            self.sulfur_gold_coord_class: class containing sulfur-gold coordination
        OUTPUTS:
            update to csv info: ratio_facets_to_total_surface_atoms
        '''
        num_facets = len(self.sulfur_gold_coord_class.gold_surface_facet_index)
        ratio_facets_to_total_surface_atoms = num_facets / float(self.sulfur_gold_coord_class.gold_surface_num_atom)
        ## STORING TO CSV FILE
        self.csv_dict.add( data_title = 'ratio_facets_to_total_surface_atoms',  data = [ ratio_facets_to_total_surface_atoms ]  )
        return
    
    ### FUNCTION TO FIND RELATIONSHIP BETWEEN SULFUR-GOLD TO BUNDLING ASSIGNED VS. UNASSIGNED LIGANDS
    def find_average_facets_assigned_vs_unassigned_ligands(self):
        '''
        The purpose of this function is to find the preference of faceted groups of assigned and unassigned ligands
        INPUTS:
            self.assigned_ligands: [list] ligand index of all assigned ligands
            self.unassigned_ligands: [list] ligand index of all unassigned ligands
            self.ratio_of_facets: [np.array, shape=(num_frames, num_ligands)] ratio of facets for each ligand
            self.bundling_class: [class] nanoparticle bundling group class
        OUTPUTS:
            avg_facets: [list] list of assigned and unassigned faceting group as a per frame basis
                avg_facets[0] <-- for assigned ligands
                avg_facets[1] <-- for unassigned ligands
        '''
        ## DEFINING EMPTY ARRAYS FOR UNASSIGNED AND ASSIGNED LIGANDS
        avg_facets = [[],[]]
        
        ## LOOPING THROUGH EACH TRAJECTORY
        for each_frame in range(self.bundling_class.total_frames):
            ## FINDING THE ASSIGNMENTS
            current_assigned_ligands, current_unassigned_ligands = self.assigned_ligands[each_frame], self.unassigned_ligands[each_frame]
            ## GETTING WATER CONTACT FOR EACH ASSIGNMENTS
            for assign_index, each_assignments in enumerate([  current_assigned_ligands, current_unassigned_ligands ]):
                ## FINDING FACETTING GROUPS
                current_facets = self.ratio_of_facets[each_frame, each_assignments]
                ## FINDING AVERAGE
#                if len(current_facets) != 0:
#                    current_avg_facets = np.nanmean(current_facets)
#                else:
#                    current_avg_facets = np.nan
                ## STORING
                avg_facets[assign_index].append(current_facets)
        return avg_facets
            
    ### FUNCTION TO FIND ENSEMBLE AVERAGE QUANTITIES
    def calc_ensem_facets(self):
        '''
        The purpose of this function is to find ensemble values
        INPUTS:
            self: class object
        OUTPUTS:
            self.results_assigned_ligands_average_water_contact: [dict] dictionary of average and std of water contact to assigned ligands
            self.results_unassigned_ligands_average_water_contact: [dict] dictionary of average and std of water contact to unassigned ligands
        '''
        ## FINDING AVERAGE WATER-LIGAND CONTACT FOR UNASSIGNED AND ASSIGNED GROUPS
        self.results_assigned_ligands_average_facets = calc_tools.calc_avg_std_of_list( np.concatenate(self.avg_facets[0]) )
        self.results_unassigned_ligands_average_facets = calc_tools.calc_avg_std_of_list( np.concatenate(self.avg_facets[1]) )
        ## STORING INTO CSV FILE
        self.csv_dict.add( data_title = 'assigned_lig_facets',  data = [ self.results_assigned_ligands_average_facets ]  )
        self.csv_dict.add( data_title = 'unassigned_lig_facets',  data = [ self.results_unassigned_ligands_average_facets ]  )
        return

    ### FUNCTION TO PLOT HISTOGRAM OF WATER CONTACT BETWEEN ASSIGNED AND UNASSIGNED
    def plot_bundling_groups_facet_distribution(self, bin_width = 0.02, save_fig = True):
        '''
        The purpose of this script is to plot the number of water contacts for assigned vs. unassigned ligands
        INPUTS:
            bin_width: [float, default=0.02] bin width of the histogram
            save_fig: [logical, default=True] Ture if you want to save the figure
        OUTPUTS:
            histogram plot of ligand-water contact distribution
        '''
        ## CREATING FIGURE
        fig, ax = create_plot()
        ## DEFINING TITLE
        ax.set_title('Facet distribution between assigned and unassigned ligands')
        ## DEFINING X AND Y AXIS
        ax.set_xlabel('Average ratio of facet to total gold atoms', **LABELS)
        ax.set_ylabel('Normalized number of occurances', **LABELS)  
        
        ## SETTING X AND Y LIMS
        ax.set_xlim([0, 1.1])
        # ax.set_ylim([0, 8])
        
        ## REMOVING ALL DATA POINTS WITH NAN
        # avg_facets = np.array(self.avg_facets)
        # avg_facets = avg_facets[~np.isnan(avg_facets)]
        
        ## FINDING MAXIMUM DATA
        # max_data_point = np.max(avg_facets)
        # min_data_point = np.min(avg_facets)
        ## CREATING BINS
        bins = np.arange(0, 1.1, bin_width)
        
        ############################
        ### PLOTTING ALL LIGANDS ###
        ############################
        ## DEFINING DATA
#        assigned_lig_values = np.array(self.avg_facets[0])
#        ## REMOVING ALL NANs
#        assigned_lig_values = assigned_lig_values[~np.isnan(assigned_lig_values)]
#        
#        ## PLOTTING HISTOGRAM
#        n, bins, patches = ax.hist(assigned_lig_values, bins = bins, color  = 'k' , density=True, label="Assigned ligands" )
        
        #################################
        ### PLOTTING ASSIGNED LIGANDS ###
        #################################
        ## DEFINING DATA
        assigned_lig_values = np.concatenate(np.array(self.avg_facets[0]))
        ## REMOVING ALL NANs
        # assigned_lig_values = assigned_lig_values[~np.isnan(assigned_lig_values)]
        
        ## PLOTTING HISTOGRAM
        n, bins, patches = ax.hist(assigned_lig_values, bins = bins, color  = 'k' , density=True, label="Assigned ligands" )

        ###################################
        ### PLOTTING UNASSIGNED LIGANDS ###
        ###################################
        ## DEFINING DATA
        unassigned_lig_values = np.concatenate(np.array(self.avg_facets[1]))
        ## REMOVING ALL NANs
        # unassigned_lig_values = unassigned_lig_values[~np.isnan(unassigned_lig_values)]
        
        ## PLOTTING HISTOGRAM
        n, bins, patches = ax.hist(unassigned_lig_values, bins = bins, color  = 'r' , density=True, label="Unassigned ligands", alpha = 0.50 )

        ## DRAWING LEGEND
        ax.legend()
        
        ## STORING IMAGE
        save_fig_png(fig = fig, label = 'facet_dist_' + self.pickle_name, save_fig = save_fig)
        
        return


    
    
#%% MAIN SCRIPT
if __name__ == "__main__":
    ## DEFINING CLASS
    from MDDescriptors.application.nanoparticle.nanoparticle_sulfur_gold_coordination import nanoparticle_sulfur_gold_coordination
    Descriptor_class = nanoparticle_sulfur_gold_coordination
    ## DEFINING DATE
    Date='180718' # '180622'
    ## DEFINING DESIRED DIRECTORY
    Pickle_loading_file=r"EAM_310.15_K_2_nmDIAM_butanethiol_CHARMM36_Trial_1"
    # Pickle_loading_file=r"EAM_310.15_K_2_nmDIAM_butanethiol_CHARMM36_Trial_1"
    #%%
    ## SINGLE TRAJ ANALYSIS
    multi_traj_results = load_multi_traj_pickle(Date, Descriptor_class, Pickle_loading_file )
    
    #%%
    #### MULTIPLE ANALYSIS TOOLS
    ## IMPORTING FUNCTION THAT WAS USED
    from MDDescriptors.application.nanoparticle.nanoparticle_sulfur_gold_coordination import nanoparticle_sulfur_gold_coordination
    from MDDescriptors.application.nanoparticle.nanoparticle_find_bundled_groups import calc_nanoparticle_bundling_groups, find_all_assigned_vs_unassigned_ligands
    ## DEFINING DATE AS A LIST
    # Dates = [ '180719-FINAL', '180719-FINAL' ]
    Dates = [ '180824', '180719-FINAL' ]
    Descriptor_classes = [ nanoparticle_sulfur_gold_coordination, calc_nanoparticle_bundling_groups ]
    
    ## LOADING MULTIPLE TRAJECTORY ANALYSIS
    multi_traj_results_list = load_multi_traj_multi_analysis_pickle(Dates = Dates, 
                                                                    Descriptor_classes = Descriptor_classes, 
                                                                    Pickle_loading_file_names = Pickle_loading_file,
                                                                    current_work_dir = None  )
    
    #%%
    
    ### RUNNING EXTRACTION TOOL
    multi_extract = extract_nanoparticle_sulfur_gold_coordination_bundling_groups( analysis_classes = multi_traj_results_list, pickle_name = Pickle_loading_file  )
    
    #%%
    #### RUNNING MULTIPLE CSV EXTRACTION
    multi_csv_export(Date = Dates,
                    Descriptor_class = Descriptor_classes,
                    desired_titles = None, 
                    export_class = extract_nanoparticle_sulfur_gold_coordination_bundling_groups,
                    export_text_file_name = 'extract_nanoparticle_sulfur_gold_coordination_bundling_groups',
                    )    

