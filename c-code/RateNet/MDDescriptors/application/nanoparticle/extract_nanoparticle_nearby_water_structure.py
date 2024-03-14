# -*- coding: utf-8 -*-
"""
extract_nearby_water_structure.py
This script extracts the results from nanoparticle_nearby_water_structure.py

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


#################################################
### CLASS FUNCTION TO EXTRACT WATER STRUCTURE ###
#################################################
class extract_nanoparticle_nearby_water_structure:
    '''
    The purpose of this function is to extract nanoparticle nearby water structure function
    INPUTS:
        analysis_class: class object
        pickle_name: name of the directory of the pickle
    OUTPUTS:
        csv_info: updated csv info 
    '''
    ### INITIALIZATION
    def __init__(self, analysis_class, pickle_name, decoder_type = 'nanoparticle'):
        ## STORING CLASS
        self.analysis_class = analysis_class
        ## STORING INFORMATION FOR CSV
        self.csv_dict = csv_dict(file_name = pickle_name, decoder_type = decoder_type )
        ## ADDING DETAILS
        self.find_ligand_water_contacts()        
        ## STORING INTO CSV INFO
        self.csv_info = self.csv_dict.csv_info
        
        return
        
    ### FUNCTION TO FIND TRANS INFORMATION
    def find_ligand_water_contacts(self, ):
        '''
        The purpose of this function is to extract the trans ratio from the nanoparticle structure
        INPUTS:
            self: class object
        OUTPUTS:
            Updated self.csv_info
        '''
        ## STORING INPUTS
        self.csv_dict.add( data_title = 'ligand_water_contacts',  data = [ self.analysis_class.results_num_water_ligand_contacts ]  )
        # self.csv_info = csv_info_add(self.csv_info, data_title = 'trans_ratio', data = [self.structure.dihedral_trans] )
        
        return


######################################################################################
### CLASS FUNCTION TO CORRELATE BETWEEN BUNDLING GROUPS AND NEARBY WATER STRUCTURE ###
######################################################################################
class extract_nanoparticle_nearby_water_structure_bundling_groups:
    '''
    The purpose of this function is to extract nearby water structures and correlate them to bundling groups
    INPUTS:
        analysis_classes: [list] list of multiple classes:
            calc_nanoparticle_bundling_groups: class containing bundling group information
            nanoparticle_nearby_water_structure: class containing nearby water structure information
    OUTPUTS:
        ## INPUT CLASSES
            self.nearby_water_structure_class: [class] nearby water structure class
            self.bundling_class: [class] nanoparticle bundling group class
        ## LIGAND ASSIGNMENTS
            self.assigned_ligands: [list] ligand index of all assigned ligands
            self.unassigned_ligands: [list] ligand index of all unassigned ligands
        ## RESULTS
            self.results_assigned_ligands_average_water_contact: [dict] dictionary of average and std of water contact to assigned ligands
            self.results_unassigned_ligands_average_water_contact: [dict] dictionary of average and std of water contact to unassigned ligands
    '''
    ### INITIALIZATION
    def __init__(self, analysis_classes, pickle_name, decoder_type = 'nanoparticle'):
        ## STORING PICKLE NAME
        self.pickle_name = pickle_name
        
        ## DEFINING CLASSES
        self.bundling_class = find_class_from_list( analysis_classes, class_name = 'calc_nanoparticle_bundling_groups' )
        self.nearby_water_structure_class = find_class_from_list( analysis_classes, class_name = 'nanoparticle_nearby_water_structure' )
        
        ## STORING INFORMATION FOR CSV
        self.csv_dict = csv_dict(file_name = pickle_name, decoder_type = decoder_type )
        
        ## FINDING ASSIGNED AND UNASSIGNED LIGANDS
        self.assigned_ligands, self.unassigned_ligands = find_all_assigned_vs_unassigned_ligands( ligand_grp_list = self.bundling_class.lig_grp_list )            
        
        ## FINDING AVERAGE WATER CONTACTS
        self.avg_water_contact = self.find_average_water_contact_assigned_vs_unassigned_ligands()
        
        ## FINDING ENSEMBLE AVERAGE QUANTITIES
        self.calc_ensem_water_contact()
        
        ## PLOTTING HISTOGRAM
        self.plot_bundling_groups_water_contact_distribution()
        
        ## STORING INTO CSV INFO
        self.csv_info = self.csv_dict.csv_info
        
        return
    
    ### FUNCTION TO FIND AVERAGE WATER CONTACT FOR ASSIGNED VS. UNASSIGNED LIGANDS
    def find_average_water_contact_assigned_vs_unassigned_ligands(self):
        '''
        The purpose of this function is to find the average water contact of assigned versus unassigned ligands
        INPUTS:
            self.assigned_ligands: [list] ligand index of all assigned ligands
            self.unassigned_ligands: [list] ligand index of all unassigned ligands
            self.nearby_water_structure_class: [class] nearby water structure class
            self.bundling_class: [class] nanoparticle bundling group class
        OUTPUTS:
            avg_water_contact: [list] list of assigned and unassigned water contacts as a per frame basis
                avg_water_contact[0] <-- water contacts for assigned ligands
                avg_water_contact[1] <-- water contacts for unassigned ligands
        '''
        ## DEFINING AVERAGE WATER CONTACT EMPTY ARRAY
        avg_water_contact = [[],[]]
        
        ## LOOPING THROUGH EACH TRAJECTORY
        for each_frame in range(self.bundling_class.total_frames):
            ## FINDING THE ASSIGNMENTS
            current_assigned_ligands, current_unassigned_ligands = self.assigned_ligands[each_frame], self.unassigned_ligands[each_frame]
            ## GETTING WATER CONTACT FOR EACH ASSIGNMENTS
            for assign_index, each_assignments in enumerate([  current_assigned_ligands, current_unassigned_ligands ]):
                ## FINDING WATER CONTACT ASSIGNMENTS
                current_water_contacts = self.nearby_water_structure_class.num_water_ligand_contacts[each_frame, each_assignments]
                ## FINDING AVERAGE WATER CONTACTS
                if len(current_water_contacts) != 0:
                    current_avg_water_contacts = np.mean(current_water_contacts)
                else:
                    current_avg_water_contacts = np.nan
                ## STORING
                avg_water_contact[assign_index].append(current_avg_water_contacts)
            
        return avg_water_contact
    
    ### FUNCTION TO FIND ENSEMBLE AVERAGE QUANTITIES
    def calc_ensem_water_contact(self):
        '''
        The purpose of this function is to find ensemble water-ligand contact
        INPUTS:
            self: class object
        OUTPUTS:
            self.results_assigned_ligands_average_water_contact: [dict] dictionary of average and std of water contact to assigned ligands
            self.results_unassigned_ligands_average_water_contact: [dict] dictionary of average and std of water contact to unassigned ligands
        '''
        ## FINDING AVERAGE WATER-LIGAND CONTACT FOR UNASSIGNED AND ASSIGNED GROUPS
        self.results_assigned_ligands_average_water_contact = calc_tools.calc_avg_std_of_list( self.avg_water_contact[0] )
        self.results_unassigned_ligands_average_water_contact = calc_tools.calc_avg_std_of_list( self.avg_water_contact[1] )
        ## STORING INTO CSV FILE
        self.csv_dict.add( data_title = 'assigned_lig_water_contact',  data = [ self.results_assigned_ligands_average_water_contact ]  )
        self.csv_dict.add( data_title = 'unassigned_lig_water_contact',  data = [ self.results_unassigned_ligands_average_water_contact ]  )
        return
        
    ### FUNCTION TO PLOT HISTOGRAM OF WATER CONTACT BETWEEN ASSIGNED AND UNASSIGNED
    def plot_bundling_groups_water_contact_distribution(self, bin_width = 0.02, save_fig = True):
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
        ax.set_title('Water contact distribution between assigned and unassigned ligands')
        ## DEFINING X AND Y AXIS
        ax.set_xlabel('Average number of water contacts per ligand', **LABELS)
        ax.set_ylabel('Normalized number of occurances', **LABELS)  
        
        ## SETTING X AND Y LIMS
        ax.set_xlim([0, 14])
        ax.set_ylim([0, 8])
        
        ## REMOVING ALL DATA POINTS WITH NAN
        avg_water_contact = np.array(self.avg_water_contact)
        avg_water_contact = avg_water_contact[~np.isnan(avg_water_contact)]
        
        ## FINDING MAXIMUM DATA
        max_data_point = np.max(avg_water_contact)
        min_data_point = np.min(avg_water_contact)
        ## CREATING BINS
        bins = np.arange(min_data_point, max_data_point, bin_width)
        
        #################################
        ### PLOTTING ASSIGNED LIGANDS ###
        #################################
        ## DEFINING DATA
        assigned_lig_water_contacts = np.array(self.avg_water_contact[0])
        ## REMOVING ALL NANs
        assigned_lig_water_contacts = assigned_lig_water_contacts[~np.isnan(assigned_lig_water_contacts)]
        
        ## PLOTTING HISTOGRAM
        n, bins, patches = ax.hist(assigned_lig_water_contacts, bins = bins, color  = 'k' , density=True, label="Assigned ligands" )

        ###################################
        ### PLOTTING UNASSIGNED LIGANDS ###
        ###################################
        ## DEFINING DATA
        unassigned_lig_water_contacts = np.array(self.avg_water_contact[1])
        ## REMOVING ALL NANs
        unassigned_lig_water_contacts = unassigned_lig_water_contacts[~np.isnan(unassigned_lig_water_contacts)]
        
        ## PLOTTING HISTOGRAM
        n, bins, patches = ax.hist(unassigned_lig_water_contacts, bins = bins, color  = 'r' , density=True, label="Unassigned ligands", alpha = 0.50 )

        ## DRAWING LEGEND
        ax.legend()
        
        ## STORING IMAGE
        save_fig_png(fig = fig, label = 'water_contact_dist_' + self.pickle_name, save_fig = save_fig)
        
        return
    
        
#%% MAIN SCRIPT
if __name__ == "__main__":
    from MDDescriptors.application.nanoparticle.nanoparticle_nearby_water_structure import nanoparticle_nearby_water_structure
    ## DEFINING CLASS
    Descriptor_class = nanoparticle_nearby_water_structure
    ## DEFINING DATE
    Date='180702' # '180622'
    ## DEFINING DESIRED DIRECTORY
    Pickle_loading_file=r"EAM_310.15_K_4_nmDIAM_butanethiol_CHARMM36_Trial_1"
    #%%
    ## SINGLE TRAJ ANALYSIS
    multi_traj_results = load_multi_traj_pickle(Date, Descriptor_class, Pickle_loading_file )
    
    #### RUNNING MULTIPLE CSV EXTRACTION
    df_1, dfs_2, dfs_2_names = multi_csv_export(Date = Date,
                                                Descriptor_class = Descriptor_class,
                                                desired_titles = None, 
                                                export_class = extract_nanoparticle_nearby_water_structure,
                                                export_text_file_name = 'extract_nearby_water_structure',
                                                )    
    
    #%%
    #### MULTIPLE ANALYSIS TOOLS
    ## IMPORTING FUNCTION THAT WAS USED
    from MDDescriptors.application.nanoparticle.nanoparticle_nearby_water_structure import nanoparticle_nearby_water_structure
    from MDDescriptors.application.nanoparticle.nanoparticle_find_bundled_groups import calc_nanoparticle_bundling_groups, find_all_assigned_vs_unassigned_ligands
    ## DEFINING DATE AS A LIST
    Dates = [ '180702', '180702' ]
    Descriptor_classes = [ nanoparticle_nearby_water_structure, calc_nanoparticle_bundling_groups ]
    
    ## LOADING MULTIPLE TRAJECTORY ANALYSIS
    multi_traj_results_list = load_multi_traj_multi_analysis_pickle(Dates = Dates, 
                                                                    Descriptor_classes = Descriptor_classes, 
                                                                    Pickle_loading_file_names = Pickle_loading_file,
                                                                    current_work_dir = None  )
    
    #%%
    
    ### RUNNING EXTRACTION TOOL
    multi_extract_nearby_water_bundling_groups = extract_nanoparticle_nearby_water_structure_bundling_groups( analysis_classes = multi_traj_results_list, pickle_name = Pickle_loading_file  )
        
    #%%
    #### RUNNING MULTIPLE CSV EXTRACTION
    df_1, dfs_2, dfs_2_names = multi_csv_export(Date = Dates,
                                                Descriptor_class = Descriptor_classes,
                                                desired_titles = None, 
                                                export_class = extract_nanoparticle_nearby_water_structure_bundling_groups,
                                                export_text_file_name = 'extract_nanoparticle_nearby_water_structure_bundling_groups',
                                                )    
    
    
    #%%
