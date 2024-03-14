# -*- coding: utf-8 -*-
"""
extract_gmx_dipoles.py
The purpose of this script is to extract gmx_dipoles

IMPORTANT NOTES
---------------
- This script assumes you already ran gmx dipoles. In particular, we are interested in dielectric constants

CREATED ON: 07/06/2018

AUTHOR(S):
    Alex K. Chew (alexkchew@gmail.com)
"""
### IMPORTING MODULES
import numpy as np
import MDDescriptors.core.import_tools as import_tools # Loading trajectory details
from MDDescriptors.core.check_tools import check_file_exist # checks if file exists

###################################################
### CLASS FUNCTION TO ANALYZE FILES FOR DIPOLES ###
###################################################
class analyze_gmx_dipoles:
    '''
    The purpose of this function is to analyze gmx dipole outputs
    INPUTS:
        traj_data: [null] traj_data is only here to enable MDDescriptors and inclusion of file directory space
        summary_file: [str] string name of the output from gmx_dipoles
        sampling_directory:[str] name of the sampling directory that contains information about sampling time
    OUTPUTS:
        self.dielectric_constant: [float] dielectric constant
    '''
    ## INITIALIZING
    def __init__(self, traj_data, summary_file, sampling_directory="gmx_dipoles_sampling_time", sampling_time_prefix="gmx_dipoles_sampling"  ):
        ## DEFINING FULL PATH
        full_path = traj_data.directory + '/' + summary_file
        ## FINDING DIELECTRIC CONSTANT
        self.dielectric_constant = self.find_dielectric_constant(full_path = full_path)
        ## DEFINING FULL PATH TO SAMPLING DIRECTORY
        full_path_sampling_dir= traj_data.directory + '/' + sampling_directory
        print(full_path_sampling_dir)
        ## SEEING IF SAMPLING DIRECTORY EXISTS
        if check_file_exist(full_path_sampling_dir) == True:
            ## PRINTING
            print("Found sampling time directory! Running analysis on sampling time...")
            ## STORING INPUTS
            self.full_path_sampling_dir = full_path_sampling_dir
            self.sampling_time_prefix = sampling_time_prefix
            ## RUNNIGN EXTRACTION PROTOCOL FOR SAMPLING TIME
            self.find_dielectric_constant_sampling_time()
    
    ### FUNCTION TO LOCATE DIELECTRIC CONSTANT
    def find_dielectric_constant(self, full_path):
        '''
        The purpose of this function is to find the dieletric constant value
        INPUTS:
            self: class object
            full_path: [str] full path to where your GROMACS output for gmx dipole is
        OUTPUTS:
            dielectric_constant: [float] dielectric constant
        '''
        ## READING FILE AS LINES
        summary_file_data = import_tools.read_file_as_line(full_path)
        
        ## FINDING LINE INDEX WITH EPSILON
        line_index_epsilon = [ idx for idx, each_line in enumerate(summary_file_data) if 'Epsilon' in each_line][-1]
        
        ## DEFINING EPSILON DATA
        epsilon_line = summary_file_data[line_index_epsilon]
        
        ## FINDING THE '=' IN EPSILON LINE
        equals_index = epsilon_line.index('=')
        
        ## FINDING EPSILON DATA
        dielectric_constant = float(epsilon_line[equals_index + 1:])
        return dielectric_constant
    
    ### FUNCTION TO LOCATE SAMPLING TIME
    def find_dielectric_constant_sampling_time(self, ):
        '''
        The purpose of this function is to go through each of the file of the sampling time and extract dielectric constant. 
        INPUTS:
            self: class object
        OUTPUTS:
            self.dielectric_constant_sampling_values: [list] dielectric constant for various sampling times
            self.dielectric_constant_sampling_time_interval: [list] list of list containing the time interval that the dielectric constant was considered
        '''
        ## IMPORTING MODULES
        import glob
        import os
        ## LOCATING ALL SAMPLING TIME DIRECTORIES AND SORTING
        list_of_sampling_time_files = sorted(glob.glob( self.full_path_sampling_dir + '/' + self.sampling_time_prefix + '*' ))
        ## LOOPING THROUGH EACH SAMPLING AND GETTING DIELECTRIC CONSTANT
        self.dielectric_constant_sampling_values = [ self.find_dielectric_constant(full_path=each_file) for each_file in list_of_sampling_time_files ]
        ## FINDING BASENAME
        current_sampling_files_basename = [ os.path.basename(filename) for filename in list_of_sampling_time_files ]
        ## USING THE BASENAME TO FIND THE SAMPLING TIME INTERVAL
        self.dielectric_constant_sampling_time_interval = [ self.sampling_time_conversion_of_file_name(each_basename) for each_basename in current_sampling_files_basename]
        
        return
        
    ### FUNCTION TO GET THE SAMPLING TIME BEGINNING AND END FOR SAMPLNG FILE BASENAME
    @staticmethod
    def sampling_time_conversion_of_file_name(file_name):
        '''
        The purpose of this function is to convert filenames for sampling time file information
        INPUTS:
            file_name: [str] name of the file, e.g. 'gmx_dipoles_sampling_7_15000_50000.info'
        OUTPUTS:
            sampling_time_interval: [list] begin and end sampling time:
                begin_sampling_time: [float] beginning sampling time
                end_sampling_time: [float] ending sampling time
        '''
        ## REMOVING THE END
        clean_file_name = file_name.replace('.info','')
        ## SPLITTING
        split_file_name = clean_file_name.split('_')
        ## RETURNS: ['gmx', 'dipoles', 'sampling', '1', '45000', '50000' ]
        begin_sampling_time = float(split_file_name[4])
        end_sampling_time = float(split_file_name[5])
        ## DEFINING SAMPLING TIME INTERVAL
        sampling_time_interval = [begin_sampling_time, end_sampling_time]
        return sampling_time_interval
        

#%% MAIN SCRIPT
if __name__ == "__main__":
    
    ### DIRECTORY TO WORK ON    
    analysis_dir=r"180713-NoSolute_Sims_Other_Solvents" # Analysis directory
    category_dir="NoSolute" # category directory
    specific_dir="mdRun_300.00_6_nm_NoSolute_10_WtPercWater_spce_dmso" 
    
    ### DEFINING PATH TO ANALYSIS DIRECTORY
    path2AnalysisDir=r"R:\scratch\SideProjectHuber\analysis\\" + analysis_dir + '\\' + category_dir + '\\' + specific_dir + '\\' # PC Side
    
    ### DEFINING FILE NAMES
    gro_file=r"mixed_solv_prod.gro" # Structural file
    xtc_file=r"mixed_solv_prod_10_ns_whole.xtc" # r"sam_prod_10_ns_whole.xtc" # Trajectory file
    
    ### LOADING TRAJECTORY
    traj_data = import_tools.import_traj( directory = path2AnalysisDir, # Directory to analysis
                 structure_file = gro_file, # structure file
                  xtc_file = xtc_file, # trajectories
                  want_only_directories = True, # want only directories
                  )
    
    ### DEFINING INPUT VARIABLES
    input_details = {   'traj_data'         :           traj_data,                      # Trajectory information
                         'summary_file'     :           "gmx_dipole_summary.txt",       # Summary file
                         }
    
    ## ANALYZING
    dipoles = analyze_gmx_dipoles( **input_details )
    
    #%%
    
    
    ### IMPORTING FUNCTION TO GET TRAJ
    from MDDescriptors.traj_tools.multi_traj import load_multi_traj_pickle, find_multi_traj_pickle, load_multi_traj_pickles
    from MDDescriptors.core.csv_tools import csv_info_add, csv_info_new, multi_csv_export, csv_info_export, csv_info_decoder_add, csv_dict
    
    ### IMPORTING PLOT
    from MDDescriptors.core.plot_tools import create_plot, save_fig_png
    from MDDescriptors.global_vars.plotting_global_vars import LINE_STYLE, LABELS
    
    #########################################
    ### CLASS FUNCTION TO EXTRACT GMX MSD ###
    #########################################
    class extract_gmx_dipoles:
        '''
        The purpose of this function is to extract the analyze_gmx_msd function
        INPUTS:
            class_object: extraction class
            pickle_name: name of the directory of the pickle
        OUTPUTS:
            csv_info: updated csv info 
        '''
        ### INITIALIZATION
        def __init__(self, class_object, pickle_name, decoder_type = 'solvent_effects'):
            ## STORING INPUTS
            self.pickle_name = pickle_name
            ## STORING STRUCTURE
            self.class_object = class_object
            ## STORING INFORMATION FOR CSV
            self.csv_dict = csv_dict(file_name = pickle_name, decoder_type = decoder_type )
            ## CALCULATING DIFFUSION COEFFICIENT
            self.find_dielectric_constant()        
            ## PLOTTING SAMPLING TIME
            # self.plot_sampling_time()
            ## DEFINING CSV INFO
            self.csv_info = self.csv_dict.csv_info
            return
        ### FUNCTION TO FIND ECCENTRICITY
        def find_dielectric_constant(self):
            '''
            The purpose of this function is to find the dielectric constant
            '''
            dielectric_constant= self.class_object.dielectric_constant
            ## STORING INPUTS
            self.csv_dict.add( data_title = 'Dielectric_constant', data = [dielectric_constant] )
            return
        
        ### FUNCTION TO PLOT THE SAMPLING TIME
        def plot_sampling_time(self, save_fig = True):
            '''
            The purpose of this function is to find the sampling time of dielectric constant
            INPUTS:
                self: class object
            OUTPUTS:
                
            '''
            ## DEFINING TIME INTERVAL
            time_intervals = self.class_object.dielectric_constant_sampling_time_interval
            ## DEFINING DIELECTRIC SAMPLING TIME
            dielectric_sampling_time = self.class_object.dielectric_constant_sampling_values
            ## FINDING SAMPLING TIME INTERVAL VALUES
            sampling_time_interval_values = [ each_interval[1]-each_interval[0] for each_interval in time_intervals]
            ## GENERATING PLOTS
            fig, ax = create_plot()
            ## SETTING X AND Y LAVELS
            ax.set_xlabel('Sampling time (ps)', **LABELS)
            ax.set_ylabel('Dielectric constant', **LABELS)
            ## PLOTTING
            ax.plot(sampling_time_interval_values,dielectric_sampling_time,'.-', color='k', **LINE_STYLE  )
            
            ## STORING IMAGE
            save_fig_png(fig = fig, label = 'dielectric_sampling_time_' + self.pickle_name, save_fig = save_fig)
            
            return
            
        
    
    
    ## DEFINING CLASS
    Descriptor_class = analyze_gmx_dipoles
    ## DEFINING DATE
    Date='190118'
    #%%
    ## DEFINING DESIRED DIRECTORY
    Pickle_loading_file=r"mdRun_300.00_6_nm_NoSolute_0_WtPercWater_spce_dmso"
    
    ## SINGLE TRAJ ANALYSIS
    multi_traj_results = load_multi_traj_pickle(Date, Descriptor_class, Pickle_loading_file )
    
    ## EXTRACTION
    dielectric_extract = extract_gmx_dipoles( class_object =  multi_traj_results,pickle_name = Pickle_loading_file )
    
    
    #%%
    
    #### RUNNING MULTIPLE CSV EXTRACTION
    multi_csv_export(Date = Date,
                    Descriptor_class = Descriptor_class,
                    desired_titles = None, 
                    export_class = extract_gmx_dipoles,
                    export_text_file_name = 'extract_gmx_dipoles',
                    )    

    