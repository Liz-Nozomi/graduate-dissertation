# -*- coding: utf-8 -*-
"""
extract_nanoparticle_sasa.py
This script extracts the results from nanoparticle_sasa.py

CLASSES:
    extract_nanoparticle_sasa: extracts nanoparticle sasa information

CREATED ON: 05/14/2018

AUTHOR(S):
    Alex K. Chew (alexkchew@gmail.com)

"""

### IMPORTING FUNCTION TO GET TRAJ
from MDDescriptors.traj_tools.multi_traj import load_multi_traj_pickle, find_multi_traj_pickle, load_multi_traj_pickles
from MDDescriptors.core.csv_tools import csv_info_add, csv_info_new, multi_csv_export, csv_info_export, csv_info_decoder_add
## IMPORTING FUNCTION THAT WAS USED
from MDDescriptors.application.nanoparticle.nanoparticle_sasa import nanoparticle_sasa

#######################################################
### CLASS FUNCTION TO EXTRACT STRUCTURAL PROPERTIES ###
#######################################################
class extract_nanoparticle_sasa:
    '''
    The purpose of this class is to extract the nanoparticle structure information
    INPUTS:
        sasa: class from nanoparticle_structure
        pickle_name: name of the directory of the pickle
    OUTPUTS:
        csv_info: updated csv info
    '''
    ### INITIALIZATION
    def __init__(self, sasa, pickle_name, decoder_type = 'nanoparticle'):
        ## STORING STRUCTURE
        self.sasa = sasa
        ## STORING INFORMATION FOR CSV
        self.csv_info = csv_info_new(pickle_name)
        ## ADDING CSV DECODER INFORMATION
        self.csv_info = csv_info_decoder_add(self.csv_info, pickle_name, decoder_type)
        ## ADDING DETAILS
        self.find_sasa()
        ## DEFINING TOTAL ATOM PER LIGAND
        self.csv_info = csv_info_add(self.csv_info, data_title = 'num_atom_per_lig', data = [ len(self.sasa.structure.ligand_atom_index_list[0]) ] )
        
    ### FUNCTION TO FIND SASA
    def find_sasa(self, ):
        '''
        The purpose of this function is to extract the trans ratio from the nanoparticle structure
        INPUTS:
            self: class object
        OUTPUTS:
            Updated self.csv_info
        '''
        ## LOOPING THROUGH EACH RESULT
        for each_group in self.sasa.sasa_avg_std:
            self.csv_info = csv_info_add(self.csv_info, data_title = 'sasa_'+each_group, data = [self.sasa.sasa_avg_std[each_group]] )
            self.csv_info = csv_info_add(self.csv_info, data_title = 'num_atoms_'+each_group, data = [len(self.sasa.atom_index_each_group_type[each_group])] )
        ## STORING INPUTS
        # self.csv_info = csv_info_add(self.csv_info, data_title = 'sasa_all_ligands', data = [self.sasa.sasa_avg_std['all_atom_ligands']] )
        # self.csv_info = csv_info_add(self.csv_info, data_title = 'sasa_end_grps', data = [self.sasa.sasa_avg_std['end_groups']] )
        # self.csv_info = csv_info_add(self.csv_info, data_title = 'optimal_cutoff', data = [self.sasa.optimal_cutoff_dict['end_groups'][0]] )
        
        return


#%% MAIN SCRIPT
if __name__ == "__main__":
    ## DEFINING CLASS
    Descriptor_class = nanoparticle_sasa
    ## DEFINING DATE
    Date='181206'
    ## DEFINING DESIRED DIRECTORY
    Pickle_loading_file=r"EAM_310.15_K_2_nmDIAM_ROT001_CHARMM36_Trial_1"
    #%%
    ## SINGLE TRAJ ANALYSIS
    multi_traj_results = load_multi_traj_pickle(Date, Descriptor_class, Pickle_loading_file )
    ## EXTRACTION
    traj_results = multi_traj_results
    print(traj_results.sasa_avg_std)
    ## CSV EXTRACTION
    extracted_results = extract_nanoparticle_sasa(traj_results, Pickle_loading_file)
    
    #%%
    ##### MULTI TRAJ ANALYSIS
    traj_results, list_of_pickles = load_multi_traj_pickles( Date, Descriptor_class)
    #%%
    #### RUNNING MULTIPLE CSV EXTRACTION
    multi_csv_export(Date = Date, 
                    Descriptor_class = Descriptor_class,
                    desired_titles = None, 
                    export_class = extract_nanoparticle_sasa,
                    export_text_file_name = 'extract_nanoparticle_sasa_rotello',
                    )    

    
    #%%
    ########################### MULTIPLE ANALYIS TOOLS ###########################
    ## IMPORTING TRAJ FUNCTIONS
    from MDDescriptors.traj_tools.multi_traj import load_multi_traj_pickle, find_multi_traj_pickle, load_multi_traj_pickles, load_multi_traj_multi_analysis_pickle, find_class_from_list
    from MDDescriptors.core.csv_tools import csv_info_add, csv_info_new, multi_csv_export, csv_info_export, csv_info_decoder_add,csv_dict

    ## MATH FUNCTIONS
    import numpy as np
    import MDDescriptors.core.calc_tools as calc_tools
    
    ### IMPORTING GLOBAL VARIABLES
    from MDDescriptors.global_vars.plotting_global_vars import COLOR_LIST, LABELS, LINE_STYLE
    from MDDescriptors.core.plot_tools import create_plot, save_fig_png, create_3d_axis_plot
    
    ## IMPORTING FUNCTION THAT WAS USED
    from MDDescriptors.application.nanoparticle.nanoparticle_sasa import nanoparticle_sasa
    from MDDescriptors.application.nanoparticle.nanoparticle_find_bundled_groups import calc_nanoparticle_bundling_groups, find_all_assigned_vs_unassigned_ligands, find_all_assigned_vs_unassigned_ligands

    ## DEFINING DATE AS A LIST
    # Dates = [ '180719-FINAL', '180719-FINAL' ]
    Dates = [ '180824', '180824' ]
    
    ## DEFINING DESIRED DIRECTORY
    Pickle_loading_file=r"EAM_310.15_K_4_nmDIAM_hexadecanethiol_CHARMM36_Trial_2"
    
    ## DEFINING DESCRIPTOR CLASS
    Descriptor_classes = [ nanoparticle_sasa, calc_nanoparticle_bundling_groups ]
    #%%
    ## LOADING MULTIPLE TRAJECTORY ANALYSIS
    multi_traj_results_list = load_multi_traj_multi_analysis_pickle(Dates = Dates, 
                                                                    Descriptor_classes = Descriptor_classes, 
                                                                    Pickle_loading_file_names = Pickle_loading_file,
                                                                    current_work_dir = None  )
    # sasa_all_atom_ligands_each_ligand
    #%%
    
    #################################################################
    ### CLASS FUNCTION TO CORRELATE SASA RATIO TO BUNDLING GROUPS ###
    #################################################################
    class extract_nanoparticle_sasa_to_bundling_groups:
        '''
        The purpose of this function is to extract nanoparticle sasa and correlate them to bundling groups
        INPUTS:
            analysis_classes: [list] list of multiple classes:
                calc_nanoparticle_bundling_groups: class containing bundling group information
                nanoparticle_nearby_water_structure: class containing nearby water structure information
            pickle_name: [str] name of the pickle file
            save_fig: [logical, default = False] True if you want to save the figure
            want_lig_dist_data: True if you want the ligand distribution data fo bundling groups sasa per ligand distribution
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
        def __init__(self, analysis_classes, pickle_name, save_fig = False, want_lig_dist_data = False, decoder_type = 'nanoparticle'):
            ## STORING PICKLE NAME
            self.pickle_name = pickle_name
            
            ## DEFINING CLASSES
            self.bundling_class = find_class_from_list( analysis_classes, class_name = 'calc_nanoparticle_bundling_groups' )
            self.sasa_class = find_class_from_list( analysis_classes, class_name = 'nanoparticle_sasa' )
                
            ## STORING INFORMATION FOR CSV
            self.csv_dict = csv_dict(file_name = pickle_name, decoder_type = decoder_type )
            
            ## FINDING ASSIGNED AND UNASSIGNED LIGANDS
            self.assigned_ligands, self.unassigned_ligands = find_all_assigned_vs_unassigned_ligands( ligand_grp_list = self.bundling_class.lig_grp_list )            
            
            ## FINDING SASA OF ASSIGNED AND UNASSIGNED
            self.sasa_assign_vs_unassign_list = self.find_sasa_assigned_vs_unassigned_ligands()
            
            ## FINDING ENSEMBLE SASA
            self.calc_ensem_sasa()
            
            ## PLOTTING HISTOGRAM
            self.plot_bundling_groups_sasa_per_ligand_distribution( save_fig = save_fig, want_lig_dist_data = want_lig_dist_data)
            
            ## STORING INTO CSV INFO
            self.csv_info = self.csv_dict.csv_info
            return
        
        ### FUNCTION TO FIND AVERAGE WATER CONTACT FOR ASSIGNED VS. UNASSIGNED LIGANDS
        def find_sasa_assigned_vs_unassigned_ligands(self):
            '''
            The purpose of this function is to find the sasa of assigned versus unassigned ligands
            INPUTS:
                self.assigned_ligands: [list] ligand index of all assigned ligands
                self.unassigned_ligands: [list] ligand index of all unassigned ligands
                self.sasa_class: [class] sasa class
                self.bundling_class: [class] nanoparticle bundling group class
            OUTPUTS:
                avg_assign_vs_unassign_list: [list] list of assigned and unassigned sasa as a per frame basis
                    avg_assign_vs_unassign_list[0] <-- variable output for assigned ligands
                    avg_assign_vs_unassign_list[1] <-- variable output for unassigned ligands
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
                    current_values = self.sasa_class.sasa_all_atom_ligands_each_ligand[each_frame, each_assignments]
                    ## STORING
                    avg_assign_vs_unassign_list[assign_index].append(current_values)
                
            return avg_assign_vs_unassign_list
        
        ### FUNCTION TO FIND ENSEMBLE AVERAGE QUANTITIES
        def calc_ensem_sasa(self):
            '''
            The purpose of this function is to find ensemble sasa
                self: class object
            OUTPUTS:
                self.results_assigned_ligands_average_water_contact: [dict] dictionary of average and std of water contact to assigned ligands
                self.results_unassigned_ligands_average_water_contact: [dict] dictionary of average and std of water contact to unassigned ligands
            '''
            ## FINDING AVERAGE WATER-LIGAND CONTACT FOR UNASSIGNED AND ASSIGNED GROUPS
            self.results_assigned_ligands = calc_tools.calc_avg_std_of_list( np.concatenate(self.sasa_assign_vs_unassign_list[0]) )
            self.results_unassigned_ligands = calc_tools.calc_avg_std_of_list( np.concatenate(self.sasa_assign_vs_unassign_list[1]) )
            ## STORING INTO CSV FILE
            self.csv_dict.add( data_title = 'assigned_sasa (nm2)',  data = [ self.results_assigned_ligands ]  )
            self.csv_dict.add( data_title = 'unassigned_sasa (nm2)',  data = [ self.results_unassigned_ligands ]  )
            return
        
        ### FUNCTION TO PLOT HISTOGRAM OF SASA BETWEEN ASSIGNED AND UNASSIGNED
        def plot_bundling_groups_sasa_per_ligand_distribution(self, bin_width = 0.02, save_fig = True, want_lig_dist_data = False):
            '''
            The purpose of this script is to plot the sasa for assigned vs. unassigned ligands
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
            ax.set_xlabel('SASA (nm^2) for each ligand', **LABELS)
            ax.set_ylabel('Normalized number of occurances', **LABELS)  
            
            ## SETTING X AND Y LIMS
            # ax.set_xlim([0, 1])
            # ax.set_ylim([0, 8])
            
            ## FINDING MAXIMUM DATA
            # max_data_point = np.max(self.sasa_class.sasa_all_atom_ligands_each_ligand)
            max_data_point = 3.5
            # min_data_point = np.min(sasa_assign_vs_unassign_list)
            ## CREATING BINS
            bins = np.arange(0 , max_data_point, bin_width)
            
#            #####################
#            ### PLOTTING FULL ### 
#            #####################
#            ## DEFINING DATA
#            all_lig_values = np.concatenate(self.sasa_class.sasa_all_atom_ligands_each_ligand)
#            
#            ## PLOTTING HISTOGRAM
#            n, bins, patches = ax.hist(all_lig_values, bins = bins, color  = 'k' , density=True, label="All_ligands", alpha = 0.75 )
            
            #################################
            ### PLOTTING ASSIGNED LIGANDS ###
            #################################
            ## DEFINING DATA
            assigned_lig_values = np.array(self.sasa_assign_vs_unassign_list[0])
            ## CONCATENATING
            assigned_lig_values = np.concatenate( assigned_lig_values )
            
            ## PLOTTING HISTOGRAM
            n, bins, patches = ax.hist(assigned_lig_values, bins = bins, color  = 'b' , density=True, label="Assigned ligands", alpha = 0.75 )
            if want_lig_dist_data is True:
                self.csv_dict.add( data_title = 'assigned_prob_dist_func',  data = [ bins, n ], labels=["bins", "PDF"]  )
            ###################################
            ### PLOTTING UNASSIGNED LIGANDS ###
            ###################################
            ## DEFINING DATA
            unassigned_lig_values = np.array(self.sasa_assign_vs_unassign_list[1])
            
            ## CONCATENATING
            unassigned_lig_values = np.concatenate( unassigned_lig_values )
            
            ## PLOTTING HISTOGRAM
            n, bins, patches = ax.hist(unassigned_lig_values, bins = bins, color  = 'r' , density=True, label="Unassigned ligands", alpha = 0.75 )
            if want_lig_dist_data is True:
                self.csv_dict.add( data_title = 'unassigned_prob_dist_func',  data = [ bins, n ], labels=["bins", "PDF"]  )
            ## DRAWING LEGEND
            ax.legend()
            
            ## STORING IMAGE
            save_fig_png(fig = fig, label = 'sasa_per_lig_dist_' + self.pickle_name, save_fig = save_fig)
            
            return
        
    #%%
    ### RUNNING EXTRACTION TOOL
    multi_extract = extract_nanoparticle_sasa_to_bundling_groups( analysis_classes = multi_traj_results_list, pickle_name = Pickle_loading_file, want_lig_dist_data = True  )
    
    #%%
    export_class_args = {
                'save_fig': False,
                'want_lig_dist_data': False,
                }
    
    #### RUNNING MULTIPLE CSV EXTRACTION
    multi_csv_export(Date = Dates,
                    Descriptor_class = Descriptor_classes,
                    desired_titles = None, 
                    export_class = extract_nanoparticle_sasa_to_bundling_groups,
                    export_text_file_name = 'extract_nanoparticle_sasa_to_bundling_groups',
                    **export_class_args
                    )    

    
    
    
    
    
    
    
    
    
    