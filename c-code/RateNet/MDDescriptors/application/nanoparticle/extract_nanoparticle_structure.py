# -*- coding: utf-8 -*-
"""
extract_nanoparticle_structure.py
This script extracts the results from nanoparticle_structure.py

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
from MDDescriptors.application.nanoparticle.nanoparticle_structure import nanoparticle_structure

#######################################################
### CLASS FUNCTION TO EXTRACT STRUCTURAL PROPERTIES ###
#######################################################
class extract_nanoparticle_structure:
    '''
    The purpose of this class is to extract the nanoparticle structure information
    INPUTS:
        structure: class from nanoparticle_structure
        pickle_name: name of the directory of the pickle
    OUTPUTS:
        csv_info: updated csv info
    '''
    ### INITIALIZATION
    def __init__(self, structure, pickle_name, decoder_type = 'nanoparticle'):
        ## STORING STRUCTURE
        self.structure = structure
        ## STORING INFORMATION FOR CSV
        self.csv_info = csv_info_new(pickle_name)
        ## ADDING CSV DECODER INFORMATION
        self.csv_info = csv_info_decoder_add(self.csv_info, pickle_name, decoder_type)
        ## ADDING DETAILS
        self.find_trans_ratio()
        self.find_min_distance()
        
    ### FUNCTION TO FIND TRANS INFORMATION
    def find_trans_ratio(self, ):
        '''
        The purpose of this function is to extract the trans ratio from the nanoparticle structure
        INPUTS:
            self: class object
        OUTPUTS:
            Updated self.csv_info
        '''
        ## STORING INPUTS
        self.csv_info = csv_info_add(self.csv_info, data_title = 'trans_ratio_avg', data = [self.structure.dihedral_trans_avg] )
        self.csv_info = csv_info_add(self.csv_info, data_title = 'trans_ratio_std', data = [self.structure.dihedral_trans_std] )
        
        return
    ### FUNCTION TO FIND MINIMUM DISTANCES
    def find_min_distance(self, ):
        '''
        The purpose of this function is to extract the minimum distance between end groups
        INPUTS:
            self: class object
        OUTPUTS:
            Updated self.csv_info
        '''
        ## STORING INPUTS
        self.csv_info = csv_info_add(self.csv_info, data_title = 'min_dist_end_grps_avg', data = [self.structure.terminal_group_min_dist_avg] )
        self.csv_info = csv_info_add(self.csv_info, data_title = 'min_dist_end_grps_std', data = [self.structure.terminal_group_min_dist_std] )
        return


#%% MAIN SCRIPT
if __name__ == "__main__":
    ## DEFINING CLASS
    Descriptor_class = nanoparticle_structure
    ## DEFINING DATE
<<<<<<< HEAD
    Date='180828' # '180719-FINAL'
=======
    Date='180921' # '180719-FINAL'
>>>>>>> 27973297dc387d7242200ce3da057bff973363c5
    ## DEFINING DESIRED DIRECTORY
    Pickle_loading_file=r"sam_8x8_300K_butanethiol_water_nvt_OPLS"
    #%%
    ## SINGLE TRAJ ANALYSIS
    multi_traj_results = load_multi_traj_pickle(Date, Descriptor_class, Pickle_loading_file )
    ## EXTRACTION
    traj_results = multi_traj_results
    #%%
    ## CSV EXTRACTION
    extracted_results = extract_nanoparticle_structure(traj_results, Pickle_loading_file)
    
    #%%
    ##### MULTI TRAJ ANALYSIS
    traj_results, list_of_pickles = load_multi_traj_pickles( Date, Descriptor_class, turn_off_multiple = True)
    #%%
    #### RUNNING MULTIPLE CSV EXTRACTION
    multi_csv_export(Date = Date, 
                    Descriptor_class = Descriptor_class,
                    desired_titles = None, 
                    export_class = extract_nanoparticle_structure,
                    export_text_file_name = 'extract_nanoparticle_structure',
                    )    
    
    #%%
    
    ########################### MULTIPLE ANALYIS TOOLS ###########################
    ## IMPORTING TRAJ FUNCTIONS
    from MDDescriptors.traj_tools.multi_traj import load_multi_traj_pickle, find_multi_traj_pickle, load_multi_traj_pickles, load_multi_traj_multi_analysis_pickle, find_class_from_list
    from MDDescriptors.core.csv_tools import csv_info_add, csv_info_new, multi_csv_export, csv_info_export, csv_info_decoder_add,csv_dict

    ## MATH FUNCTIONS
    import numpy as np
    
    ### IMPORTING GLOBAL VARIABLES
    from MDDescriptors.global_vars.plotting_global_vars import COLOR_LIST, LABELS, LINE_STYLE
    from MDDescriptors.core.plot_tools import create_plot, save_fig_png, create_3d_axis_plot
    
    ## IMPORTING FUNCTION THAT WAS USED
    from MDDescriptors.application.nanoparticle.nanoparticle_structure import nanoparticle_structure
    from MDDescriptors.application.nanoparticle.nanoparticle_find_bundled_groups import calc_nanoparticle_bundling_groups, find_all_assigned_vs_unassigned_ligands, find_all_assigned_vs_unassigned_ligands

    ## DEFINING DATE AS A LIST
    # Dates = [ '180719-FINAL', '180719-FINAL' ]
    Dates = [ '180824', '180824' ]
    
    ## DEFINING DESCRIPTOR CLASS
    Descriptor_classes = [ nanoparticle_structure, calc_nanoparticle_bundling_groups ]
    
    #%%
    ## DEFINING DESIRED DIRECTORY
    Pickle_loading_file=r"EAM_310.15_K_4_nmDIAM_hexadecanethiol_CHARMM36_Trial_3"
    

    ## LOADING MULTIPLE TRAJECTORY ANALYSIS
    multi_traj_results_list = load_multi_traj_multi_analysis_pickle(Dates = Dates, 
                                                                    Descriptor_classes = Descriptor_classes, 
                                                                    Pickle_loading_file_names = Pickle_loading_file,
                                                                    current_work_dir = None  )
    
    #%%
    
    import MDDescriptors.core.calc_tools as calc_tools
    
    ###########################################################################
    ### CLASS FUNCTION TO CORRELATE BETWEEN BUNDLING GROUPS AND TRANS RATIO ###
    ###########################################################################
    class extract_nanoparticle_structure_to_bundling_groups:
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
            self.structure_class = find_class_from_list( analysis_classes, class_name = 'nanoparticle_structure' )
                
            ## STORING INFORMATION FOR CSV
            self.csv_dict = csv_dict(file_name = pickle_name, decoder_type = decoder_type )
    
            ## FINDING ASSIGNED AND UNASSIGNED LIGANDS
            self.assigned_ligands, self.unassigned_ligands = find_all_assigned_vs_unassigned_ligands( ligand_grp_list = self.bundling_class.lig_grp_list )            
        
            ## FINDING TRANS RATIO OF ASSIGNED AND UNASSIGNED
            self.trans_ratio_assign_vs_unassign_list = self.find_trans_ratio_assigned_vs_unassigned_ligands()
            
            ## FINDING ENSEMBLE AVERAGE QUANTITIES
            self.calc_ensem_trans_ratio()
            
            ## PLOTTING HISTOGRAM
            # self.plot_bundling_groups_trans_ratio_per_ligand_distribution()
            
            ## STORING INTO CSV INFO
            self.csv_info = self.csv_dict.csv_info
            
            return
        
        ### FUNCTION TO FIND AVERAGE WATER CONTACT FOR ASSIGNED VS. UNASSIGNED LIGANDS
        def find_trans_ratio_assigned_vs_unassigned_ligands(self):
            '''
            The purpose of this function is to find the trans-ratio of assigned versus unassigned ligands
            INPUTS:
                self.assigned_ligands: [list] ligand index of all assigned ligands
                self.unassigned_ligands: [list] ligand index of all unassigned ligands
                self.structure_class: [class] structure class
                self.bundling_class: [class] nanoparticle bundling group class
            OUTPUTS:
                avg_water_contact: [list] list of assigned and unassigned water contacts as a per frame basis
                    avg_water_contact[0] <-- water contacts for assigned ligands
                    avg_water_contact[1] <-- water contacts for unassigned ligands
            '''
            ## DEFINING AVERAGE WATER CONTACT EMPTY ARRAY
            avg_assign_vs_unassign_list = [[],[]]
            
            ## LOOPING THROUGH EACH TRAJECTORY
            for each_frame in range(self.bundling_class.total_frames):
                ## FINDING THE ASSIGNMENTS
                current_assigned_ligands, current_unassigned_ligands = self.assigned_ligands[each_frame], self.unassigned_ligands[each_frame]
                ## GETTING WATER CONTACT FOR EACH ASSIGNMENTS
                for assign_index, each_assignments in enumerate([  current_assigned_ligands, current_unassigned_ligands ]):
                    ## FINDING VALUES FOR EACH ASSIGNMENT
                    current_values = self.structure_class.dihedral_ligand_trans_avg_per_frame[each_frame, each_assignments]
                    ## STORING
                    avg_assign_vs_unassign_list[assign_index].append(current_values)
                
            return avg_assign_vs_unassign_list
    
        ### FUNCTION TO FIND ENSEMBLE AVERAGE QUANTITIES
        def calc_ensem_trans_ratio(self):
            '''
            The purpose of this function is to find ensemble trans ratio
                self: class object
            OUTPUTS:
                self.results_assigned_ligands_average_water_contact: [dict] dictionary of average and std of water contact to assigned ligands
                self.results_unassigned_ligands_average_water_contact: [dict] dictionary of average and std of water contact to unassigned ligands
            '''
            ## FINDING AVERAGE WATER-LIGAND CONTACT FOR UNASSIGNED AND ASSIGNED GROUPS
            self.results_assigned_ligands = calc_tools.calc_avg_std_of_list( np.concatenate(self.trans_ratio_assign_vs_unassign_list[0]) )
            self.results_unassigned_ligands = calc_tools.calc_avg_std_of_list( np.concatenate(self.trans_ratio_assign_vs_unassign_list[1]) )
            ## STORING INTO CSV FILE
            self.csv_dict.add( data_title = 'trans_ratio_assigned',  data = [ self.results_assigned_ligands ]  )
            self.csv_dict.add( data_title = 'trans_ratio_unassigned',  data = [ self.results_unassigned_ligands ]  )
            return
    
        ### FUNCTION TO PLOT HISTOGRAM OF WATER CONTACT BETWEEN ASSIGNED AND UNASSIGNED
        def plot_bundling_groups_trans_ratio_per_ligand_distribution(self, bin_width = 0.02, save_fig = True):
            '''
            The purpose of this script is to plot the trans ratio for assigned vs. unassigned ligands
            INPUTS:
                bin_width: [float, default=0.02] bin width of the histogram
                save_fig: [logical, default=True] Ture if you want to save the figure
            OUTPUTS:
                histogram plot of ligand-water contact distribution
            '''
            ## CREATING FIGURE
            fig, ax = create_plot()
            ## DEFINING TITLE
            ax.set_title('Trans ratio distribution between assigned and unassigned ligands')
            ## DEFINING X AND Y AXIS
            ax.set_xlabel('Trans ratio for each ligand', **LABELS)
            ax.set_ylabel('Normalized number of occurances', **LABELS)  
            
            ## SETTING X AND Y LIMS
            ax.set_xlim([0, 1])
            # ax.set_ylim([0, 8])
            
#                ## REMOVING ALL DATA POINTS WITH NAN
#                avg_water_contact = np.array(self.avg_water_contact)
#                avg_water_contact = avg_water_contact[~np.isnan(avg_water_contact)]
#                
#                ## FINDING MAXIMUM DATA
#                max_data_point = np.max(avg_water_contact)
#                min_data_point = np.min(avg_water_contact)
            ## CREATING BINS
            bins = np.arange(0 , 1, bin_width)
            
            #################################
            ### PLOTTING ASSIGNED LIGANDS ###
            #################################
            ## DEFINING DATA
            assigned_lig_values = np.array(self.trans_ratio_assign_vs_unassign_list[0])
            ## CONCATENATING
            assigned_lig_values = np.concatenate( assigned_lig_values )
            
            ## PLOTTING HISTOGRAM
            n, bins, patches = ax.hist(assigned_lig_values, bins = bins, color  = 'b' , density=True, label="Assigned ligands", alpha = 0.75 )
    
            ###################################
            ### PLOTTING UNASSIGNED LIGANDS ###
            ###################################
            ## DEFINING DATA
            unassigned_lig_values = np.array(self.trans_ratio_assign_vs_unassign_list[1])
            
            ## CONCATENATING
            unassigned_lig_values = np.concatenate( unassigned_lig_values )
            
            ## PLOTTING HISTOGRAM
            n, bins, patches = ax.hist(unassigned_lig_values, bins = bins, color  = 'r' , density=True, label="Unassigned ligands", alpha = 0.75 )
    
            ## DRAWING LEGEND
            ax.legend()
            
            ## STORING IMAGE
            save_fig_png(fig = fig, label = 'trans_ratio_per_lig_dist_' + self.pickle_name, save_fig = save_fig)
            
            return
        
    #%%
        
    ### RUNNING EXTRACTION TOOL
    multi_extract = extract_nanoparticle_structure_to_bundling_groups( analysis_classes = multi_traj_results_list, pickle_name = Pickle_loading_file  )
    
    #%%
    
    #### RUNNING MULTIPLE CSV EXTRACTION
    multi_csv_export(Date = Dates,
                    Descriptor_class = Descriptor_classes,
                    desired_titles = None, 
                    export_class = extract_nanoparticle_structure_to_bundling_groups,
                    export_text_file_name = 'extract_nanoparticle_structure_to_bundling_groups',
                    )    
    
    
    
    
    
    
    
    