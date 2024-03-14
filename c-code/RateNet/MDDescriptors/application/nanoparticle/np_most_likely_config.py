# -*- coding: utf-8 -*-
"""
np_most_likely_config.py
The purpose of this script is to find the most likely configuration of 
the nanoparticle. The idea is that we have bundling groups

Written by: Alex K. Chew (alexkchew@gmail.com, 03/27/2019)
"""

### IMPORTING MODULES
import numpy as np
import pandas as pd
import os
import mdtraj as md

## MDDESCRIPTOR MODULES
import MDDescriptors.core.import_tools as import_tools # Loading trajectory details
### IMPORTING NANOPARTICLE STRUCTURE CLASS
from MDDescriptors.application.nanoparticle.nanoparticle_structure import nanoparticle_structure
### IMPORTING GLOBAL VARIABLES
from MDDescriptors.application.nanoparticle.global_vars import GOLD_SULFUR_CUTOFF, GOLD_GOLD_CUTOFF_BASED_ON_SHAPE, NP_WORKING_DIR
## IMPORTING MULTI TRAJ FUNCTIONS
from MDDescriptors.traj_tools.multi_traj import load_multi_traj_pickle, find_multi_traj_pickle, load_multi_traj_pickles, load_multi_traj_multi_analysis_pickle, load_pickle_for_analysis
## IMPROTING BUNDLING GROUPS
from MDDescriptors.application.nanoparticle.nanoparticle_find_bundled_groups import calc_nanoparticle_bundling_groups
## IMPORTING PICKLE FUNCTIONS
from MDBuilder.core.pickle_funcs import store_class_pickle, load_class_pickle

## MDBUILDERS FUNCTIONS
from MDBuilder.core.check_tools import check_testing

## IMPORTING MD DESCRIPTOR CALC TOOLS FUNCTION
import MDDescriptors.core.calc_tools as calc_tools

## GLOBAL VARIABLES
from MDDescriptors.application.nanoparticle.global_vars import LIGAND_RESIDUE_NAMES, MDDESCRIPTORS_DEFAULT_VARIABLES

## IMPORTING PLOTTING TOOLS
import MDDescriptors.core.plot_tools as plot_tools

### FUNCTION TO STANDARDIZE ARRAY
def standardize( array ):
    ''' This function standardizes numerical features 
    INPUTS:
        array: [np.array]
            array you want to standardize
    OUTPUTS:
        standardized_array: [np.array]
            standardized features by subtracting the mean and dividing by the std
        mean: [float]
            mean of the features with the same number of columns as the features list
        std: [float]
            std of the features with the same number of columns as the features list
    '''
    ## FINDING MEAN AND STD
    mean = np.mean(array)
    std = np.std(array)
    ## CHECKING IF STD == 0
    if std == 0:
        ## MAKING SURE THAT NO STD IS EQUAL TO 0
        std = 1.0
    ## FEATURE STANDARDIZED
    standardized_array = (array - mean)/std
    return standardized_array, mean, std

### FUNCTION TO FIND THE SORTED INDEX
def find_sorted_array_based_on_desired_values( array, desired_array, order = None ):
    '''
    The purpose of this function is to find the sorted array based on desired values. 
    This array can take any shape, so long as the desired values have the same number of 
    columns. For instance, you can have two arrays of interest with shape=(num_frames, 1). 
    You can then column stack them, such as:
        array = np.column_stack( ( avg_bundling_standardized, cross_entropy_standardized ) )
        desired_values = np.column_stack( ( avg_bundling_value_standardized, cross_entropy_lowest_standardized ) )
    Then, use this function to sort the array. This function will first take the differences, and take the norm. 
    INPUTS:
        array: [np.array, shape=(num_frames, num_features)]
            array with number of frames and number of features
        desired_values: [np.array, shape=(1, num_features)]
            desired value for each feature
        ord: [type, default = None]
            order or the norm, default is 2-norm
    OUTPUT:
        sorted_index: [np.array, shape=(num_frames)]
            sorted index based on the trajectory index
        sorted_norms: [np.array, shape=(num_frames)]
            output norms
    '''
    ## FINDING DIFFERENCES
    differences = array - desired_array
    ## FINDING NORM
    norms = np.linalg.norm( differences, axis = 1, ord = order )
    ## SORTING SMALLEST TO LARGEST
    sorted_index = np.argsort( norms )
    ## SORTED NORMS
    sorted_norms = norms[sorted_index]
    
    return sorted_index, sorted_norms

## FUNCTION TO PLOT NUMBER OF BUNDLES
def plot_num_bundles_over_frames( time_list, total_bundling_groups, avg_bundling = None, round_bundling = None ):
    '''
    The purpose of this function is to plot number of bundles over frames. 
    INPUTS:
        total_bundling_groups: [np.array]
            number of bundling groups per frame
        time_list: [np.array]
            time list
    OUTPUTS:
        figure of number of bundles over frames
    '''

    ## FINDING AVERAGE
    if avg_bundling is None:
        avg_bundling = np.mean(total_bundling_groups)
    if round_bundling is None:
        round_bundling = np.round(avg_bundling)
    
    ## CREAITING AXIS PLOT
    fig, ax = plot_tools.create_plot()
    
    ## UPDATING AXIS
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel("Number of bundling groups")
    
    ## PLOTTING LINE
    ax.plot( time_list, total_bundling_groups, linestyle = '-', color ='k', linewidth = 1.6   )
    
    ## PLOTTING HORIZONTAL LINE
    ax.axhline( y = avg_bundling, linestyle='--', color = 'r', linewidth = 1.6, label = 'Average bundling')
    
    ## PLOTTING HORIZONTAL LINE
    ax.axhline( y = round_bundling, linestyle='--', color = 'b', linewidth = 1.6, label = 'Rounded bundling')
    
    ## ADDING LEGEND
    ax.legend()
    
    return fig, ax

### FUNCTION TO PRINT OUT PANDAS AND PICKLE
def print_panda_summary( pandas_obj, output_folder_path, output_file_name  ):
    '''
    The purpose of this function is to print out pandas summary
    INPUTS:
        pandas_obj: [obj]
            pandas object
        output_folder_path: [str]
            output folder path
        output_file_name: [str]
            output file name
    OUTPUT: null
    '''
    ## COMBINING PATHS
    output_full_path=os.path.join(output_folder_path, output_file_name)
    ## PRINTING
    print("Creating summary file in: %s"%(output_full_path) )
    ## PRINTING CLOSEST STRUCTURE
    pandas_obj.to_csv(output_full_path, sep='\t', index=None, mode='w')
    return


###############################################################
### CLASS FUNCTION TO FIND THE MOST PROBABLE BUNDLING GROUP ###
###############################################################
class find_most_probable_np:
    '''
    The purpose of this function is to find the most probable nanoparticle. 
    The constraints are as follows:
        - Has to be close to the number of clusters available
        - Uses the probability distribution of ligands within versus without a cluster.
    INPUTS:
        traj_data: [class]
            Data taken from import_traj class
        pickle_loading_file: [str]
            pickle loading file information
        ligand_names: [list]
            list of ligand names you are interested in
        analysis_classes: [list] ** DEPRECIATED
            list of list regarding the analysis class
        output_folder_path: [str, default = None]
            path to output folder. If None, no pickle / output file will be generated
        output_file_name: [str, default="most_likely_config"]
            name of output file
        want_pickle: [logical, default=True]
            True if you want to save a pickle
    OUTPUTS:
        self.bundling_groups: [obj]
            bundling group class
    FUNCTIONS
        plot_bundling_vs_time: [active] plots bundling versus time
        extract_bundling_info: extracts bundling information
    '''
    ### INITIALIZING
    def __init__(self, 
                 traj_data,
                 pickle_loading_file, 
                 ligand_names,
                 output_folder_path = None,
                 output_file_name = 'np_most_likely',
                 debug = False,
                 want_pickle = False,
                 bundling_groups_vars = {}
                 ):
        ### STORING INITIAL VARIABLES
        self.output_folder_path = output_folder_path
        self.output_file_name = output_file_name
        
#        ### GETTING ONLY ANALYSIS CLASSES
#        try:
#            self.bundling_groups = load_pickle_for_analysis(
#                                                            analysis_classes = analysis_classes, 
#                                                            function_name = 'calc_nanoparticle_bundling_groups', 
#                                                            pickle_file_name = pickle_loading_file,
#                                                            conversion_type = None,
#                                                            current_work_dir = NP_WORKING_DIR,
#                                                            )
#            ## WHEN FILE IS NOT FOUND, RUN BUNDLING GROUPS
#        except FileNotFoundError:
        ## COMPUTING BUNDLING GROUPS
        ## LOADING DEFAULT VARIABLES
        default_vars = MDDESCRIPTORS_DEFAULT_VARIABLES[calc_nanoparticle_bundling_groups.__name__]        
        ## CHANGING VARS
        for each_key in bundling_groups_vars:
            default_vars[each_key] = bundling_groups_vars[each_key]
        self.bundling_groups = calc_nanoparticle_bundling_groups(traj_data = traj_data,
                                                                 **default_vars )
            
        ## EXTRACTING BUNDLING GROUPS
        self.extract_bundling_info()
        
        ## FUNCTION TO COMPUTE CROSS ENTROPY OF LIGANDS
        self.compute_ligand_cross_entropy()
        
        ## STANDARDIZING THE FEATURES
        self.standardize_features()
        
        ## FINDING CLOSEST STRUCTURE
        self.find_closest_structure()
        
        ## CREATING DATABASE
        self.create_database_closest_structures()
        
        ## WRITING OUTPUT FILE IF PATH EXISTS
        if output_folder_path is not None:
            ## PRINTING PANDAS
            print_panda_summary( pandas_obj = self.closest_structure_database, 
                                 output_folder_path = self.output_folder_path,
                                 output_file_name = self.output_file_name,
                                )
            ## OUTPUTING TO PICKLE
            if want_pickle is True:
                ## EXTRACTION FILE NAME WITHOUT SUMMARY
                file_name_without_summary = os.path.splitext(self.output_file_name)[0]
                store_class_pickle(self, os.path.join(self.output_folder_path, file_name_without_summary + ".pickle"))
        
    ### FUNCTION TO FIND CLOSEST STRUCTURE
    def find_closest_structure(self):
        '''
        The purpose of this function is to find the closest structures subject to two constraints:
            1 -- the number of bundles is close to the average number of bundles
            2 -- the cross entropy is minimized
        Here, we will use a simple 1-norm to find the closest structure. We could have 
        used alternative norms, such as the Manhattan distance, etc. 
        INPUTS:
            self: [obj]
                class object
        '''
        ## DEFINING ARRAY OF INTEREST
        array_of_features = np.column_stack( ( self.total_bundling_groups_standardized, self.cross_entropy_summed_standardized ) )
        desired_values =  np.column_stack( ( self.avg_bundling_standardized, self.cross_entropy_lowest_standardized ) )
        
        ## FINDING SORTED INDEX
        self.closest_structure_time_index, self.closest_structure_norms = \
            find_sorted_array_based_on_desired_values( array = array_of_features, 
                                                       desired_array = desired_values,
                                                       order = 1) # order of 1 -norm
        return
        
        
    ### FUNCTION TO STANDARDIZE CROSS ENTROPY AND AVERAGE BUNDLING
    def standardize_features(self):
        '''
        The purpose of this function is to standardize the bundling and cross entropy 
        so they are comparable. Otherwise, the cross entropy dominates the feature space. 
        The standardization is as follows:
            standard_feature = ( feature - mean ) / std
        INPUTS:
            self: [obj]
                class object
        OUTPUTS:
            self.total_bundling_groups_standardized: [np.array]
                total bundling groups that are standardized
            self.avg_bundling_standardized: [float]
                average bundling groups that are standardized
            self.cross_entropy_summed_standardized: [np.array]
                cross entropy summed that is standardized
            self.cross_entropy_lowest_standardized: [float]
                lowest cross entropy that is standardized
        '''
        ## STANDARDIZING BUNDLING GROUPS
        self.total_bundling_groups_standardized, total_bundling_groups_stand_mean, total_bundling_groups_stand_std = \
                standardize(self.total_bundling_groups)
        self.avg_bundling_standardized = (self.avg_bundling - total_bundling_groups_stand_mean) / total_bundling_groups_stand_std
        
        ## STANDARDIZING CROSS ENTROPY
        self.cross_entropy_summed_standardized, cross_entropy_stand_mean, cross_entropy_stand_mean_std = \
            standardize(self.cross_entropy_summed)
        self.cross_entropy_lowest_standardized = (self.cross_entropy_lowest - cross_entropy_stand_mean) / cross_entropy_stand_mean_std
        
        return
        
    
    ### FUNCTION TO COMPUTE CROSS ENTROPY OF THE LIGANDS
    def compute_ligand_cross_entropy(self,):
        '''
        The purpose of this function is to compute cross entropy of the ligands. The algorithm 
        is follows:
            - We will need to compute the probability that a ligand is in or not in a bundle. 
            - We need to assign probabilities -- we do this with Laplacian estimate
            - With the probabilities, we need to compute the cross entropy
            - We can also sort the cross entropy -- though it is not that important
        INPUTS:
            self: [obj]
                class object
        OUTPUTS:
            self.ligand_assignments: [np.array]
                ligand assignments as a function of time
            self.prob_ligand_in_bundle: [np.array]
                probability of ligand in a bundle
            self.ligand_assignments_converted: [np.array]
                ligand assignments with 1 or 0 based on whether the ligand is in a bundle (1) or not (0)
            self.cross_entropy_summed: [np.array, shape=(num_frames)]
                cross entropy of all the ligands
            self.cross_entropy_sorted_index: [np.array, shape=(num_frames)]
                cross entropy sorted index from 0 to number of rames
            self.cross_entropy_summed_sorted: [np.array, shape=(num_frames)]
                cross entropy summed and sorted
            self.cross_entropy_lowest: [float]
                lowest cross entropy possible
        '''
        ## FINDING ASSIGNMENTS
        self.ligand_assignments = np.array(self.bundling_groups.lig_grp_assignment)
        
        ## COUNTING NUMBER OF NON  ASSIGNMENTS
        ligand_nonassigment_count = np.sum(self.ligand_assignments==-1,axis=0)
        
        ## COMPUTING PROBABILITY OF NON-ASSIGNMENTS
        prob_non_assignments = (ligand_nonassigment_count+1) / (self.total_frames+2) # Using laplace estimates
        ## COMPUTING PROBABILITY OF ASSIGNMENTS
        self.prob_ligand_in_bundle = 1-prob_non_assignments
        
        ## CONVERTING ASSIGNMENTS TO 0 OR 1
        self.ligand_assignments_converted = np.copy(self.ligand_assignments)
        self.ligand_assignments_converted[self.ligand_assignments>-1] = 1
        self.ligand_assignments_converted[self.ligand_assignments==-1] = 0
        
        ## COMPUTING CROSS ENTROPY
        cross_entropy = - (self.ligand_assignments_converted * np.log(self.prob_ligand_in_bundle) + \
                           (1 - self.ligand_assignments_converted) * np.log( 1 - self.prob_ligand_in_bundle) )
        self.cross_entropy_summed = np.sum(cross_entropy, axis = 1)
        
        ## SORING
        self.cross_entropy_sorted_index = np.argsort( self.cross_entropy_summed )
        ## SORTED CROSS ENTROPY
        self.cross_entropy_summed_sorted =  self.cross_entropy_summed[self.cross_entropy_sorted_index]
        ## DEFINING LOWEST CROSS ENTROPY 
        self.cross_entropy_lowest = self.cross_entropy_summed_sorted[0]
        
        return
        
    ### FUNCTION TO PLOT PROBABILITY OF LIGANDS IN BUNDLE VS. INDEX
    def plot_prob_ligand_vs_index(self, fig_label = 'np_most_likely_prob_ligand_vs_index', save_fig=False):
        ''' This function plots probability of ligand in bundle versus index '''
        ## DEFINING LIGAND INDEX
        ligand_index = np.arange(len(self.prob_ligand_in_bundle))
        
        ## CREATING AXIS PLOT
        fig, ax = plot_tools.create_plot()
        
        ## UPDATING AXIS
        ax.set_xlabel('Ligand index')
        ax.set_ylabel("Probability of bundle")
        
        ## PLOTTING LINE
        ax.scatter( ligand_index, self.prob_ligand_in_bundle, color ='k', linewidth = 1.6   )
        # ax.bar( ligand_index, self.prob_ligand_in_bundle, color ='k', align='center'   )
        ## SAVING FIGURE        
        if save_fig is True:
            plot_tools.save_fig_png( fig = fig, 
                                     label = fig_label )
        return fig, ax
        
    ### FUNCTION TO PLOT THE CROSS ENTROPY AS A FUNCTION OF TIME
    def plot_cross_entropy_vs_time(self, fig_label = 'np_most_likely_cross_entropy_vs_time', save_fig=False):
        ''' This function plots cross entropy as a function of time '''
        ## CREATING AXIS PLOT
        fig, ax = plot_tools.create_plot()
        
        ## UPDATING AXIS
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel("Cross entropy of ligands")
        
        ## PLOTTING LINE
        ax.plot(  self.traj_time, self.cross_entropy_summed, linestyle = '-', color ='k', linewidth = 1.6   )
        
        ## PLOTTING MINIMA
        ax.axvline( x = self.traj_time[self.cross_entropy_sorted_index[0]], linestyle='--', color = 'r', linewidth = 1.6, label = 'Minimum cross entropy time')
        ax.axhline( y = self.cross_entropy_summed[self.cross_entropy_sorted_index[0]], linestyle='--', color = 'b', linewidth = 1.6, label = 'Lowest cross entropy')
        
        ## ADDING LEGEND
        ax.legend()
        
        ## SAVING FIGURE        
        if save_fig is True:
            plot_tools.save_fig_png( fig = fig, 
                                     label = fig_label )
        return fig, ax
    
    ### FUNCTION TO PLOT NUMBER OF BUNDLES VERSUS CROSS ENTROPY
    def plot_cross_entropy_vs_num_bundles(self, fig_label = 'np_most_likely_cross_entropy_vs_num_bundles', save_fig=False):
        ''' This plots cross entropy versus number of bundles '''
        ## CREATING AXIS PLOT
        fig, ax = plot_tools.create_plot()
        
        ## UPDATING AXIS
        ax.set_xlabel('Number of bundles')
        ax.set_ylabel("Cross entropy of ligands")
        
        ## PLOTTING LINE
        ax.scatter( self.total_bundling_groups, self.cross_entropy_summed, color ='k', linewidth = 1.6   )
        
        ## PLOTTING MINIMA
        ax.axhline( y = self.cross_entropy_lowest, linestyle='--', color = 'r', linewidth = 1.6, label = 'Lowest cross entropy')
        ax.axvline( x = self.avg_bundling, linestyle='--', color = 'b', linewidth = 1.6, label = 'Average bundling group')
        
        ## ADDING LEGEND
        ax.legend(loc='upper left')
        ## SAVING FIGURE        
        if save_fig is True:
            plot_tools.save_fig_png( fig = fig, 
                                     label = fig_label )
        return fig, ax
        
    ### FUNCTION TO PLOT NUMBER OF BUNDLES VERSUS CROSS ENTROPY
    def plot_cross_entropy_vs_num_bundles_standardized(self, fig_label = 'np_most_likely_cross_entropy_vs_num_bundles_standardized', save_fig=False):
        ''' This plots cross entropy versus number of bundles standardized '''
        ## CREATING AXIS PLOT
        fig, ax = plot_tools.create_plot()
        
        ## UPDATING AXIS
        ax.set_xlabel('Standardized number of bundles')
        ax.set_ylabel("Standardized cross entropy of ligands")
        
        ## PLOTTING LINE
        ax.scatter( self.total_bundling_groups_standardized, self.cross_entropy_summed_standardized, color ='k', linewidth = 1.6   )
        
        ## PLOTTING MINIMA
        ax.axhline( y = self.cross_entropy_lowest_standardized, linestyle='--', color = 'r', linewidth = 1.6, label = 'Lowest cross entropy')
        ax.axvline( x = self.avg_bundling_standardized, linestyle='--', color = 'b', linewidth = 1.6, label = 'Average bundling group')
        
        ## ADDING LEGEND
        ax.legend(loc='upper left')
        ## SAVING FIGURE        
        if save_fig is True:
            plot_tools.save_fig_png( fig = fig, 
                                     label = fig_label )
        return fig, ax
    
    ### FUNCTION TO PLOT BUNDLING VERSUS TIME
    def plot_bundling_vs_time(self, fig_label = 'np_most_likely_bundling_vs_time', save_fig=False):
        ''' This plots bundling vs. time '''
        
        ## PLOTTING NUMBER OF BUNDLES
        fig, ax  = plot_num_bundles_over_frames(    time_list = self.traj_time, 
                                                    total_bundling_groups = self.total_bundling_groups,
                                                    avg_bundling = self.avg_bundling,
                                                    round_bundling = self.avg_bundling_rounded)
        ## SAVING FIGURE        
        if save_fig is True:
            plot_tools.save_fig_png( fig = fig, 
                                     label = fig_label )
        
        return fig, ax
    
    ### FUNCTION TO EXTRACT BUNDLING GROUPS
    def extract_bundling_info(self):
        '''
        The purpose of this function is to extract bundling information
        INPUTS:
            self: [obj]
                class object
        OUTPUTS:
            self.total_bundling_groups: [np.array]
                total bundling groups as a function of time
            self.time_list: [np.array]
                time list array for the simulation in ps
            self.total_frames: [int]
                total number of frames
            self.avg_bundling: [float]
                average bundling
            self.avg_bundling_rounded: [float]
                average bundling that is rounded to an integer value
            self.index_bundling_groups_in_avg: [tuple]
                indices of bundling groups
            self.num_bundling_within_avg: [int]
                number of bundling group within rounded average        
        '''
        ## FINDING TOTAL NUMBER OF BUNDLING GROUPS
        self.total_bundling_groups = self.bundling_groups.lig_total_bundling_groups
        
        ## DEFINING TIME
        self.traj_time = self.bundling_groups.structure_np.traj_time_list
        ## FINDING TOTAL FRAMES
        self.total_frames = self.bundling_groups.total_frames
        
        ## FINDING AVERAGE BUNDLING
        self.avg_bundling = self.bundling_groups.results_avg_std_bundling_grps['avg']
        self.avg_bundling_rounded = np.round( self.bundling_groups.results_avg_std_bundling_grps['avg'] )
        
        ## FINDING POINTS WHERE WE FIND THIS NUMBER OF BUNDLING GROUPS
        self.index_bundling_groups_in_avg = np.where( self.total_bundling_groups == self.avg_bundling_rounded )
        
        ## FINDING NUMBER OF BUNDLES
        self.num_bundling_within_avg = len(self.index_bundling_groups_in_avg[0])
        return
        
    ### FUNCTION TO CREATE DATABASE OF CLOSEST STRUCTURES
    def create_database_closest_structures(self):
        '''
        The purpose of this function is to create the closest structures in a form of a pandas table. Then, 
        we can output the table as a form of a text file. The idea here is that we would like to use the database 
        in the bash shell by exporting a specific configuration.
        INPUTS:
            self: [object]
                class object
        OUTPUTS:
             self.closest_structure_database: [pandas.DataFrame]
                 database containing frames and norm from maxima
                      Frame_index  Norm_from_maxima
                0             474          0.220377
                1             250          0.389766
        '''
        ## DEFINING STORAGE VECTOR
        self.closest_structure_database = None
        self.closest_structure_database = pd.DataFrame( columns = [ 'Index',
                                                                    'Python_index',
                                                                    'Frame_index',
                                                                    'Frame_time_ps',
                                                                    'Norm',
                                                                    ]
                                                        )
        ## LOOPING THROUGH EACH TIME FRAME TO GET THE CONFIGURATION
        for idx, each_frame in enumerate(self.closest_structure_time_index):
            ## ADDING TO THE PANDAS
            self.closest_structure_database = self.closest_structure_database.append(
                                                {
                                                 'Index': idx,   
                                                 'Python_index': each_frame,
                                                 'Frame_index': each_frame + 1,  
                                                 'Frame_time_ps': self.traj_time[each_frame],
                                                 'Norm' : self.closest_structure_norms[idx],
                                                 } , ignore_index = True)
        ## CONVERTING DATA TYPE
        self.closest_structure_database['Index'] = self.closest_structure_database.Index.astype(int)
        self.closest_structure_database['Python_index'] = self.closest_structure_database.Python_index.astype(int)
        self.closest_structure_database['Frame_index'] = self.closest_structure_database.Frame_index.astype(int)
        self.closest_structure_database['Frame_time_ps'] = self.closest_structure_database['Frame_time_ps'].map(lambda x: '%.3f' % x) # 3 decimal places
        self.closest_structure_database['Norm'] = self.closest_structure_database['Norm'].map(lambda x: '%.3f' % x) # 3 decimal places
        return
        
    ### STORING SELF INFORMATION
    



#%% MAIN SCRIPT
if __name__ == "__main__":
    ## TESTING
    testing = check_testing() #  check_testing()
    
    ## TESTING VARIABLES
    if testing is True:
    
        ## DEFINIGN MAIN DIRECTORY
        main_dir = r"R:\scratch\nanoparticle_project\simulations"
        
        ### DIRECTORY TO WORK ON    
        simulation_dir=r"190509-2nm_C11_Sims_updated_forcefield"
        # r"190510-2nm_C11_Sims_extraction" # Analysis directory
        
        ## DEFINING SPECIFIC DIRECTORY FOR ANALYSIS
        specific_dir="EAM_300.00_K_2_nmDIAM_dodecanethiol_CHARMM36jul2017_Trial_1" # spherical_310.15_K_2_nmDIAM_octanethiol_CHARMM36" # Directory within analysis_dir r"mdRun_363.15_6_nm_tBuOH_100_WtPercWater_spce_Pure"    
#        specific_dir="EAM_300.00_K_2_nmDIAM_C11OH_CHARMM36jul2017_Trial_1" # spherical_310.15_K_2_nmDIAM_octanethiol_CHARMM36" # Directory within analysis_dir r"mdRun_363.15_6_nm_tBuOH_100_WtPercWater_spce_Pure"    
#        specific_dir="EAM_300.00_K_2_nmDIAM_C11OH_CHARMM36jul2017_Trial_1" # spherical_310.15_K_2_nmDIAM_octanethiol_CHARMM36" # Directory within analysis_dir r"mdRun_363.15_6_nm_tBuOH_100_WtPercWater_spce_Pure"    
#        specific_dir="EAM_300.00_K_2_nmDIAM_C11NH2_CHARMM36jul2017_Trial_1" # spherical_310.15_K_2_nmDIAM_octanethiol_CHARMM36" # Directory within analysis_dir r"mdRun_363.15_6_nm_tBuOH_100_WtPercWater_spce_Pure"    
        
        ### DEFINING FULL PATH TO WORKING DIRECTORY
        # input_path=r"R:\scratch\nanoparticle_project\analysis\\" + analysis_dir + '\\' + category_dir + '\\' + specific_dir # PC Side
        input_path=os.path.join( main_dir, simulation_dir, specific_dir )
        ## DEFINING GRO AND XTC FILE
        gro_file=r"sam_prod.gro" # Structural file
        xtc_file=r"sam_prod_10_ns_whole_rot_trans.xtc" # r"sam_prod_10_ns_whole.xtc" # Trajectory file
        ## DEFINING VARIABLES
        want_pickle = True
        
        ## DEFINING OUTPUT PATH
        output_folder_path = input_path # r"R:\scratch\SideProjectHuber\Scripts\AnalysisScripts"
        output_file_name = r"np_most_likely"
        
        ## SAVING FIGURE ?
        save_fig = False
        fig_extension = "png"
        
        ## DEFINING DISPLACEMENT VECTOR TYPE
        displacement_vector_type = 'avg_heavy_atom'
        
    else:
        from optparse import OptionParser # for parsing command-line options
        ## RUNNING COMMAND LINE PROMPTS
        parser = OptionParser()
        
        ## PATH INFORMATION
        parser.add_option('-i', '--inputpath', dest = 'input_path', help = 'Full input path to files', default = '.')
        
        ## OUTPUT FILES
        parser.add_option('-o', '--outputpath', dest = 'output_folder_path', help = 'Full output path to file', default = '.')
        parser.add_option('-s', '--summary', dest = 'output_file_name', help = 'Summary file name', default = '.')
        
        ## INPUT FILE INFORMATION
        parser.add_option('-g', '--gro', dest = 'gro_file', help = 'input gro file', default = '.')
        parser.add_option('-x', '--xtc', dest = 'xtc_file', help = 'input xtc file', default = '.')
        parser.add_option('--itp', dest = 'itp_file', help = 'input itp file', default = 'sam.itp')
        
        ## LOGICALS
        parser.add_option('-p', '--pickle', dest = 'want_pickle', help = 'Stores pickle if flag is turned on. Default: False.',
                          default = False, action = 'store_true')
        
        ## BUNDLING TYPE
        parser.add_option('--bundling_type', dest = 'displacement_vector_type', help = 'Bundling type, either terminal_heavy_atom or avg_heavy_atom',
                          default = 'avg_heavy_atom', )
        
        ### GETTING ARGUMENTS
        (options, args) = parser.parse_args() # Takes arguments from parser and passes them into "options" and "argument"
        
        ### DEFINING VARIABLES
        input_path = options.input_path
        output_folder_path = options.output_folder_path
        output_file_name = options.output_file_name
        gro_file = options.gro_file
        xtc_file = options.xtc_file
        want_pickle = options.want_pickle
        displacement_vector_type = options.displacement_vector_type
        itp_file = options.itp_file

    ####################################################################
    ### MAIN SCRIPTS 
    ####################################################################
    ### DEFINING SPECIFIC DIRECTORY BASED ON NAME
    specific_dir = os.path.basename( input_path )
    ### LOADING TRAJECTORY
    traj_data = import_tools.import_traj( directory = input_path, # Directory to analysis
                                          structure_file = gro_file, # structure file
                                          xtc_file = xtc_file, # trajectories
                                          )
        
        
    #%%
    
    
    ## DEFINING BUNDLING GROUP VARS
    bundling_groups_vars = {'displacement_vector_type': displacement_vector_type,
                            'itp_file' : itp_file, }
    
    ### DEFINING INPUT DATA
    input_details = {   'traj_data'             :           traj_data,                      # Trajectory information
                         'ligand_names'         :           LIGAND_RESIDUE_NAMES,   # Name of the ligands of interest
#                         'analysis_classes'     : [[ calc_nanoparticle_bundling_groups, '180719-FINAL' ], ## REUSING SELF ASSEMBLY STRUCTURE CLASS
#                                                   ],
                         'pickle_loading_file'  : specific_dir,       ## SPECIFIC DIRECTORY FOR LOADING
                         'debug'                : False, # Debugging
                         'output_folder_path'   : output_folder_path,
                         'output_file_name'     : output_file_name,
                         'want_pickle'          : want_pickle,
                         'bundling_groups_vars' : bundling_groups_vars,
                        }

    ## FINDING MOST PROBABLE NP
    most_probable_np = find_most_probable_np( **input_details )
    # len(most_probable_np.bundling_groups.lig_grp_assignment)
    # Similarity metrics -- could be useful : http://dataaspirant.com/2015/04/11/five-most-popular-similarity-measures-implementation-in-python/
    # most_probable_np.plot_bundling_vs_time()
    # most_probable_np.plot_cross_entropy_vs_time()
    # most_probable_np.plot_prob_ligand_vs_index()
    # most_probable_np.plot_cross_entropy_vs_num_bundles()
    # most_probable_np.plot_cross_entropy_vs_num_bundles_standardized()
    
    #%%
#    save_fig = True
#    most_probable_np.plot_bundling_vs_time( save_fig = save_fig )
#    most_probable_np.plot_cross_entropy_vs_time( save_fig = save_fig )
#    most_probable_np.plot_prob_ligand_vs_index( save_fig = save_fig )
#    most_probable_np.plot_cross_entropy_vs_num_bundles( save_fig = save_fig )
#    most_probable_np.plot_cross_entropy_vs_num_bundles_standardized( save_fig = save_fig )
    #%%
    # Clustering with RMSD: http://mdtraj.org/1.9.3/examples/clustering.html
    
    
#    ### FUNCTION TO PLOT RMSD VS. TIME
#    def plot_rmsd_vs_time(time, 
#                          rmsd,
#                          avg_rmsd = None,
#                          ):
#        ''' 
#        This function plots RMSD versus time
#        INPUTS:
#            time: [np.array]
#                time array
#            rmsd: [np.array]
#                rmsd with the same length as the time
#        '''
#        ## CREATING AXIS PLOT
#        fig, ax = plot_tools.create_plot()
#        
#        ## UPDATING AXIS
#        ax.set_xlabel('Time (ps)')
#        ax.set_ylabel("RMSD")
#        
#        ## PLOTTING LINE
#        ax.plot(  time, rmsd, linestyle = '-', color ='k', linewidth = 1.6   )
#        
#        ## PLOTTING AVERAGE LINE
#        if avg_rmsd is not None:
#            ## HORIZONTAL LINE
#            ax.axhline( y = avg_rmsd, linestyle='--', color = 'r', linewidth = 1.6, label = 'Avg RMSD')
#            ## ADDING LEGEND
#            ax.legend()
#        return fig, ax
#    
#    
#    ### FUNCTION TO COMPUTE THE AVERAGE RMSE
#    def compute_avg_rmse_specific_index( rmsd, 
#                                         array_closest):
#        '''
#        The purpose of this function is to compute the average RMSD without the 
#        indices in question. 
#        INPUTS:
#            rmse: [np.array]
#                rmse with respect to each time frame
#            array_closest: [np.array]
#                array index that you do not want RMSE to be computed for
#        OUTPUTS:
#            avg_rmsd: [float]
#                average RMSD without using the array indices in 'array_closest'
#        '''
#        avg_rmsd = np.mean(np.delete(rmsd, array_closest))
#        return avg_rmsd
#        
#    
#    ## DEFINING DESIRED MOST LIKELY CONFIGURATION
#    num_desired_most_likely_config = 2
#    
#    ## FINDING ALL HEAVY ATOM INDEX (FLATTENED)
#    ligand_heavy_atom_index = np.array(
#            calc_tools.flatten_list_of_list(most_probable_np.bundling_groups.structure_np.ligand_heavy_atom_index) )
#    
#    ## DEFINING LIST
#    closest_structure_time_index_list = most_probable_np.closest_structure_time_index
#    
#    ## DEFINING MOST CLOSEST STRUCTURE TIME INDEX
#    closest_structure_time_index = closest_structure_time_index_list[0]
#    
#    ## DEFINING ARRAY CONTAINING
#    array_closest = np.array([closest_structure_time_index])
#
#    ## COMPUTING RMSF
#    rmsd = md.rmsd( target = traj_data.traj, 
#                    reference = traj_data.traj,
#                    frame = closest_structure_time_index, # frame we want to test for
#                    atom_indices = ligand_heavy_atom_index
#                    )
#    
#    ## FINDING AVERAGE OF AN RMSD ARRAY WITHOUT THESE INDICES
#    avg_rmsd = compute_avg_rmse_specific_index( rmsd = rmsd, 
#                                                array_closest = array_closest)
#    
#    ## PLOTTING RMSD
#    fig, ax = plot_rmsd_vs_time( time = most_probable_np.traj_time,
#                                 rmsd = rmsd,
#                                 avg_rmsd = avg_rmsd,
#                                 )
#    
#    #%% 
#    
#    ## SEEING NEXT INDEX
#    next_time_indices = np.delete(most_probable_np.closest_structure_time_index, closest_structure_time_index)[0]
#    
#    
#    ## COMPUTING RMSF
#    rmsd_list = [ md.rmsd( target = traj_data.traj, 
#                    reference = traj_data.traj,
#                    frame = array_closest, # frame we want to test for
#                    atom_indices = ligand_heavy_atom_index
#                    ) for each_] 
#    ## COMPUTING AVERAGE RMSE FOR EACH ARRAY CLOSEST
#    # avg_rmsd_list = 
#    
#    
#    
#    
#        
#    ## SAVING FIGURE
#    fig_name = specific_dir + '_rmsd_vs_time'
#
#    ## STORING FIGURE
#    if save_fig == True:
#        ## SAVING FIGURE
#        plot_tools.save_fig_png( fig = fig, 
#                                 label = fig_name )
#    
#    
#    
#    ## DEFINING 
#    
#    
#    
#    
#    
#    
#    #%%
#    
#    ## RMSD STANDARDIZED
#    rmsd_standardized = standardize(rmsd)[0]
#    
#    ## FINDING RELATIVE RMSD
#    rmsd_standardized_relative = rmsd_standardized - rmsd_standardized[closest_structure_time_index]
#    
#    ## FINDING DIFFERENCE OF CROSS ENTROPIES
#    bundling_groups_relative = most_probable_np.total_bundling_groups_standardized - \
#                                most_probable_np.total_bundling_groups_standardized[closest_structure_time_index]
#    cross_entropy_relative = most_probable_np.cross_entropy_summed_standardized - \
#                                most_probable_np.cross_entropy_summed_standardized[closest_structure_time_index]
#                                
#                                
#    ## COMBINING ARRAY
#    coordinate_array = np.column_stack( (bundling_groups_relative, cross_entropy_relative, rmsd_standardized_relative ))
#    ## RETURNS: num_frames, 3
#    
#    ## FINDING NORM
#    norms = np.linalg.norm( coordinate_array, axis = 1, ord = 2 )
#    
#    ## SORTING LARGEST TO SMALLEST
#    sorted_index = np.argsort( norms )[::-1]
#    
#    ## SORTED NORMS
#    sorted_norms = norms[sorted_index]
#    
#    #%%
#    
#                                
#    ## CREATING 3D
#    fig , ax = plot_tools.create_3d_axis_plot() 
#    
#    ## UPDATING AXIS
#    ax.set_xlabel('Number of bundling groups')
#    ax.set_ylabel("Cross entropy")
#    ax.set_zlabel("RMSD")
#    
#    ## PLOT SCATTER
#    ax.scatter(bundling_groups_relative, 
#               cross_entropy_relative, 
#               rmsd_standardized_relative,
#               linestyle = '-', 
#               color ='k', 
#               linewidth = 1.6   )
#    
#    ## PLOT SCATTER
#    ax.scatter(bundling_groups_relative[sorted_index[0]], 
#               cross_entropy_relative[sorted_index[0]], 
#               rmsd_standardized_relative[sorted_index[0]],
#               linestyle = '-', 
#               color ='red', 
#               linewidth = 2   )
#    
#    ## MAKING AXIS EQUAL
##    ax.axis('equal')
#    
#    ## TIGHT LAYOUT    
#    fig.tight_layout()
#    
#    ## SAVING FIGURE
#    fig_name = specific_dir + '_bundling_cross_rmsd'
#
#    ## SAVING FIGURE
#    plot_tools.save_fig_png( fig = fig, 
#                             label = fig_name )
#
#    
#    ## PLOTTING OTHERS
#    most_probable_np.plot_prob_ligand_vs_index( fig_label = specific_dir + "_prob_lig_vs_index",
#                                                save_fig =save_fig)
#    most_probable_np.plot_cross_entropy_vs_num_bundles_standardized( 
#                                                                    fig_label = specific_dir + "_cross_entropy_vs_num",
#                                                                    save_fig = save_fig )
#    
#    
#    
#    #%%
#    
#    ## IMPORTING MODULES
#    import matplotlib.pyplot as plt
#    from mpl_toolkits.mplot3d import Axes3D # For 3D axes
#        
#    ## DEFINING GLOBAL PLOTTING VARIABLES
#    FONT_SIZE=12
#    FONT_NAME="Arial" 
#    
#    LABELS = {
#                'fontname': FONT_NAME,
#                'fontsize': FONT_SIZE
#                }
#    
#    ## DEFAULT FIGURE SIZE
#    plt.rcParams['figure.figsize'] = [10, 10] # inches
#    plt.rcParams.update()
#    ### FUNCTION TO PLOT SCATTER
#    def plot_scatter( positions, frame, atom_index, fig = None, ax = None, **plot_dicts  ):
#        '''
#        The purpose of this function is to plot the system.
#        INPUTS:
#            positions: [np.array, shape=(n_frames, n_atoms, 3)]
#                atomic positions of your system
#            frame: [int]
#                frame that you want to plot
#            atom_index: [np.array, shape=(N_atoms, 1)]
#                atom index for the positions
#            fig: [obj, default=None]
#                figure object
#            ax: [obj, default=None]
#                axis object
#            plot_dicts: [dict]
#                plotting dictionary
#        OUTPUTS:
#            fig: [obj, default=None]
#                figure object
#            ax: [obj, default=None]
#                axis object
#        '''
#        ## IMPORTING TOOLS
#        # import matplotlib.pyplot as plt
#        # from mpl_toolkits.mplot3d import Axes3D # For 3D axes
#        
#        ## CREATING FIGURE
#        if fig == None or ax == None:
#            fig = plt.figure(); ax = fig.add_subplot(111, projection='3d', aspect='equal')
#        
#            ## ADDING X, Y, Z LABELS
#            ax.set_xlabel('x (nm)', **LABELS)
#            ax.set_ylabel('y (nm)', **LABELS)
#            ax.set_zlabel('z (nm)', **LABELS)
#    
#        ## PLOTTING POINTS
#        ax.scatter(positions[frame][atom_index, 0], positions[frame][atom_index, 1], positions[frame][atom_index, 2], 
#                       **plot_dicts )
#        
#        return fig, ax
#    
#    
#    ### FUNCTION TO PLOT EACH ATOM INDEX
#    def plot_each_atom_index( positions, frame, plot_atom_index_dict, plot_order ):
#        '''
#        This function plots each atom index consecutively
#        INPUTS:
#            positions: [np.array, shape=(n_frames, n_atoms, 3)]
#                atomic positions of your system
#            frame: [int]
#                frame that you want to plot
#            plot_atom_index_dict: [dict]
#                dictionary containing all atom indices
#            plot_order: [list]
#                order of how you want to plot
#        OUTPUTS:
#            fig: [obj, default=None]
#                figure object
#            ax: [obj, default=None]
#                axis object
#        '''
#        ## DEFING FIGURE AND AXIS
#        fig, ax = None, None
#        ## LOOPING THROUGH EACH ATOM INDEX
#        for each_atom_type in plot_order:
#            ## DEFINING ATOM INDEX
#            atom_index = plot_atom_index_dict[each_atom_type]['atom_index']
#            ## DEFINING PLOTTING INFORMATION
#            plot_dicts = plot_atom_index_dict[each_atom_type]['plotting_dict']
#            ## PLOTTING SCATTER FUNCTION
#            fig, ax = plot_scatter( positions, frame, atom_index = atom_index, fig = fig, ax = ax, **plot_dicts  )
#                   
#        return fig, ax
#    
#    
#     ### FUNCTION TO GET CMAP
#    def get_cmap(n, name='hsv'):
#        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
#        RGB color; the keyword argument name must be a standard mpl colormap name.
#        This function is useful to generate colors between red and purple without having to specify the specific colors
#        USAGE:
#            ## GENERATE CMAP
#            cmap = get_cmap(  len(self_assembly_coord.gold_facet_groups) )
#            ## SPECIFYING THE COLOR WITHIN A FOR LOOP
#            for ...
#                current_group_color = cmap(idx) # colors[idx]
#                run plotting functions
#        '''
#        ## IMPORTING FUNCTIONS
#        import matplotlib.pyplot as plt
#        return plt.cm.get_cmap(name, n + 1)
#        
#    ### FUNCTION TO FIND GROUP ASSIGNMENTS
#    def find_group_assignments( assignments ):
#        '''
#        The purpose of this function is to find the group assignments with a given array of assignments.
#        For example, suppose you are given a list of numbers [ 0 , 1 ,2, 0 ], and you want to create a new list:
#            0: [0, 3]
#            1: [1]
#            2: [2]
#        In this case, we are generating list of lists to get all the indexes that are matching
#        INPUTS:
#            assignments: [np.array, shape=(N,1)] assignments for each index
#        OUTPUTS:
#            group_list: [dict] dictionary of the group list
#        '''
#        ## FINDING UNIQUE ASSIGNMENTS
#        unique_assignments = np.unique( assignments )
#        ## CREATING EMPTY DICTIONARY LIST
#        group_list = {}    
#        ## LOOPING THROUGH EACH ASSIGNMENT AND FINDING ALL INDICES WHERE TRUE
#        for each_assignment in unique_assignments:
#            ## FINDING ALL INDEXES
#            indices_with_assignments = np.where(assignments == each_assignment)[0] # ESCAPING TUPLE
#            ## CREATING A DICTIONARY AND STORING THE VALUES
#            group_list[str(each_assignment)] = indices_with_assignments
#        return group_list
#        
#    ## QUICK FUNCTION TO REORDER LIST OF INT
#    def sort_list_of_int(listofstrings):
#        '''This function simply sorts a list of strings'''
#        convert_to_int = [ int(each_str) for each_str in listofstrings ]
#        sorted_convert_to_int = sorted(convert_to_int)
#        convert_to_str = [ str(each_int) for each_int in sorted_convert_to_int ]
#        return convert_to_str
#        
#    ### FUNCTION TO PLOT ASSIGNMENTS
#    def plot_assignments( positions, atom_index, frame, group_list, fig = None, ax = None, **plot_dict):
#        '''
#        The purpose of this function is to plot the assignments for a given frame
#        INPUTS:
#            positions: [np.array, shape=(n_frames, n_atoms, 3)]
#                atomic positions of your system
#            frame: [int]
#                frame that you want to plot
#            atom_index: [np.array, shape=(N_atoms, 1)]
#                atom index for the positions
#            group_list: [dict]
#                group list for the assignments of atom_index
#            fig: [obj, default=None]
#                figure object
#            ax: [obj, default=None]
#                axis object
#        OUTPUTS:
#            fig: [obj, default=None]
#                figure object
#            ax: [obj, default=None]
#                axis object
#        '''
#        ## GENERATING COLOR LIST
#        cmap_colors = get_cmap(n=len(group_list[frame]))
#        
#        ## CONVERTING GROUP LIST TO ORDER CORRECTLY
#        group_list_keys_sorted = sort_list_of_int(group_list[frame].keys())
#        
#        ## PLOTTING ASSIGNMENTS
#        for idx, each_group in enumerate(group_list_keys_sorted):
#    
#            ## DEFINING  COLOR
#            if each_group == '-1':
#                current_color = "gray"
#            else:
#                current_color = cmap_colors(idx)
#    
#            ## ADDING TO PLOT
#            ax.scatter( positions[frame][end_group_index[group_list[frame][each_group]], 0],
#                        positions[frame][end_group_index[group_list[frame][each_group]], 1],
#                        positions[frame][end_group_index[group_list[frame][each_group]], 2],
#                       color=current_color,
#                       label = each_group,
#                       **plot_dict
#                      )
#        ## ADDING LEGEND
#        ax.legend()
#            
#        return fig, ax
#    
#    #---------------
#    
#    for frame_type in ["most_likely", "least_likely"]:
#        # frame_type = "most_likely"
#    #    frame_type = "least_likely"
#        
#        ## DEFINING FIG NAME
#        fig_name = specific_dir
#        
#    
#        
#        if frame_type == "most_likely":
#            frame = most_probable_np.closest_structure_time_index[0]
#            fig_name = fig_name + "_most_likely_frame_%d"%(frame)
#        elif frame_type =="least_likely":
#            frame = sorted_index[0]
#            fig_name = fig_name + "_least_likely_frame_%d"%(frame)
#        
#        ## DEFINING ATOM INDEX
#        ligand_heavy_atom_index_list = np.array(most_probable_np.bundling_groups.structure_np.ligand_heavy_atom_index[:])
#        ## DEFINING GOLD ATOM INDEX
#        gold_atom_index_list = most_probable_np.bundling_groups.structure_np.gold_atom_index[:]
#        ## DEFINING SULFUR HEAD GROUP LIST
#        sulfur_head_group_index_list = np.array(most_probable_np.bundling_groups.structure_np.head_group_atom_index)
#        ## DEFINING END GROUP INDEX
#        end_group_index = ligand_heavy_atom_index_list[:,-1]
#        
#        ## DEFINING GROUP LIST
#        group_list = most_probable_np.bundling_groups.lig_grp_list
#        
#        ## DEFINING VARIABLES FOR ATOM INDEX
#        plot_atom_index_dict={
#            'LIGAND': { 'atom_index': ligand_heavy_atom_index_list[:,1:].flatten(),
#                        'plotting_dict': { 'marker': 'o',
#                                           'edgecolors': 'black',
#                                           'color' : 'black',
#                                            'linewidth': 2 ,
#                                           'alpha':0.1,
#                                          }},
#            'AUNP': { 'atom_index': gold_atom_index_list,
#                        'plotting_dict': { 'marker': 'o',
#                                           'edgecolors': 'orange',
#                                           'color' : 'orange',
#                                           'linewidth': 1 ,
#                                          }},
#            'SULFURS': { 'atom_index': sulfur_head_group_index_list,
#                        'plotting_dict': { 'marker': 'o' ,
#                                           'edgecolors': 'yellow',
#                                           'color' : 'yellow',
#                                            'linewidth': 2,
#                                          }},
#            }
#                 
#    
#        ## ADDING END GROUPS TO PLOTTING DICTIONARY
#        plot_atom_index_dict['END_GRPS']={ 'atom_index': end_group_index,
#                                            'plotting_dict': { 'marker': 'o',
#                                                               'edgecolors': 'red',
#                                                               'color' : 'red',
#                                                                'linewidth': 3 ,
#                                                               'alpha':1,
#                                                              }}
#        
#        ## DEFINING PLOTTING ORDER
#        plot_order = ['AUNP', 'SULFURS','LIGAND',] # ,'LIGAND', 'END_GRPS'
#        ## PLOTITNG EACH               
#        fig, ax = plot_each_atom_index(positions = traj_data.traj.xyz, 
#                                        frame = frame,
#                                        plot_atom_index_dict = plot_atom_index_dict,
#                                        plot_order = plot_order)
#        
#        ## DEFINING PLOT DICTIONARY FOR ASSIGNMENTS
#        assignment_plot_dict = {
#            'linewidth': 5
#        }
#             
#        
#        ## PLOTTING ASSIGNMENTS
#        fig, ax = plot_assignments( positions = traj_data.traj.xyz,
#                                   atom_index = end_group_index,
#                                   group_list = group_list,
#                                   frame = frame,
#                                   fig = fig,
#                                   ax = ax,
#                                   **assignment_plot_dict
#                                  )
#        
#        ## SAVING FIGURE
#        ## STORING FIGURE
#        if save_fig == True:
#            fig_name = fig_name + '.' + fig_extension
#            print("Printing figure: %s"%(fig_name) )
#            fig.savefig( fig_name, 
#                         format=fig_extension, 
#                         dpi = 600,    )
        

