# -*- coding: utf-8 -*-
"""
extract_gmx_msd.py
The purpose of this function is to extract gmx msd code and run subsequent calculations on it

IMPORTANT NOTES
---------------
- This script assumes you already ran gmx msd. In particular, we are interested in the diffusion coefficients

CREATED ON: 07/06/2018

AUTHOR(S):
    Alex K. Chew (alexkchew@gmail.com)
"""

### IMPORTING MODULES
import numpy as np
import MDDescriptors.core.import_tools as import_tools # Loading trajectory details

### MODULE TO IMPORT FILE
from MDDescriptors.application.nanoparticle.extract_gmx_principal import read_xvg

### FUNCTION TO ANALYZE THE XVG FILE
class analyze_gmx_msd:
    '''
    The purpose of this function is to analyze the outputs of gmx msd
    INPUTS:
        traj_data: [null] traj_data is only here to enable MDDescriptors and inclusion of file directory space
        xvg_file: [str] string name of the output file from gmx msd
    OUTPUTS:
        
    '''
    ## INITIALIZATION
    def __init__(self, traj_data, xvg_file,  ):
        
        ## DEFINING FULL PATH
        full_xvg_path = traj_data.directory + '/' + xvg_file
        
        ## READING FILE
        self.data_full, self.data_extract =  read_xvg( file_path = full_xvg_path)
    
        ## FINDING DIFFUSION COEFFICIENTS
        self.find_diff_coeff()
        
        return
    
    ## FUNCTION TO FIND DIFFUSION COEFFICIENT
    def find_diff_coeff(self):
        '''
        The purpose of this funcction is to find the diffusion coefficient value
        INPUTS:
            self: class object
        OUTPUTS:
            self.diff_coeff_value: [float] diffusion coefficient value
            self.diff_coeff_err: [float] diffusion coefficient error
            self.diff_coeff_units: [str] units of the diffusion coefficient
            self.msd_data: [np.array] msd data as a function os picoseconds
        '''
        ## FINDING LINE THAT HAS THE LAST '#'
        self.line_index_last_pound = [ index for index, each_line in enumerate(self.data_extract) if each_line[0]=='#' ][-1]
        ## DEFINING LINE EXTRACT
        line_extract = self.data_extract[self.line_index_last_pound]
        ## DEFINING DIFFFUSION COEFFICIENT BASED ON NUMBERS
        # e.g. ['#', 'D[', 'AUNP]', '=', '0.1238', '(+/-', '0.0425)', '(1e-5', 'cm^2/s)']
        self.diff_coeff_value = float(line_extract[4])
        self.diff_coeff_err   = float(line_extract[6].replace(')','')) # Removes all percent signs from the error
        self.diff_coeff_units =  line_extract[7] + line_extract[8]

        ## DEFINING DATA PAST THE POUND SIGNS
        self.msd_data = np.array(self.data_extract[self.line_index_last_pound+1:]).astype('float64')

        return        
    
        

#%% MAIN SCRIPT
if __name__ == "__main__":
    
    ### DIRECTORY TO WORK ON    
    analysis_dir=r"180702-Trial_1_spherical_EAM_correction" # Analysis directory
    category_dir="EAM" # category directory
    specific_dir="EAM_310.15_K_2_nmDIAM_hexadecanethiol_CHARMM36_Trial_1" # spherical_310.15_K_2_nmDIAM_octanethiol_CHARMM36" # Directory within analysis_dir r"mdRun_363.15_6_nm_tBuOH_100_WtPercWater_spce_Pure"    
    
    ### DEFINING PATH TO ANALYSIS DIRECTORY
    path2AnalysisDir=r"R:\scratch\nanoparticle_project\analysis\\" + analysis_dir + '\\' + category_dir + '\\' + specific_dir + '\\' # PC Side
    
    ### DEFINING FILE NAMES
    gro_file=r"sam_prod.gro" # Structural file
    xtc_file=r"sam_prod_10_ns_whole.xtc" # r"sam_prod_10_ns_whole.xtc" # Trajectory file
    
    ### LOADING TRAJECTORY
    traj_data = import_tools.import_traj( directory = path2AnalysisDir, # Directory to analysis
                 structure_file = gro_file, # structure file
                  xtc_file = xtc_file, # trajectories
                  want_only_directories = True, # want only directories
                  )
    
    ### DEFINING INPUT VARIABLES
    input_details = {   'traj_data'         :           traj_data,                      # Trajectory information
                         'xvg_file'         :           "msd.xvg",                      # File with moments of inertia
                         }
    
    ### RUNNING ANALYSIS TOOL
    msd = analyze_gmx_msd( **input_details )
    # data_full, data_extract = read_xvg( file_path = traj_data.directory + "msd.xvg" )
    
    
    #%%
    #%%
    
    ### IMPORTING FUNCTION TO GET TRAJ
    from MDDescriptors.traj_tools.multi_traj import load_multi_traj_pickle, find_multi_traj_pickle, load_multi_traj_pickles
    from MDDescriptors.core.csv_tools import csv_info_add, csv_info_new, multi_csv_export, csv_info_export, csv_info_decoder_add, csv_dict
    
    #########################################
    ### CLASS FUNCTION TO EXTRACT GMX MSD ###
    #########################################
    class extract_gmx_msd:
        '''
        The purpose of this function is to extract the analyze_gmx_msd function
        INPUTS:
            class_object: extraction class
            pickle_name: name of the directory of the pickle
        OUTPUTS:
            csv_info: updated csv info 
        '''
        ### INITIALIZATION
        def __init__(self, class_object, pickle_name, decoder_type = 'nanoparticle'):
            ## STORING INPUTS
            self.pickle_name = pickle_name
            ## STORING STRUCTURE
            self.class_object = class_object
            ## STORING INFORMATION FOR CSV
            self.csv_dict = csv_dict(file_name = pickle_name, decoder_type = decoder_type )
            ## CALCULATING DIFFUSION COEFFICIENT
            self.find_diffusion_coefficient()        
            ## PLOTTING COEFFICIENTS
            self.plot_msd_vs_time()
            ## DEFINING CSV INFO
            self.csv_info = self.csv_dict.csv_info
            return
        ### FUNCTION TO FIND ECCENTRICITY
        def find_diffusion_coefficient(self):
            '''
            The purpose of this function is to find the diffusion coefficient
            '''
            diff_coeff_avg = self.class_object.diff_coeff_value
            diff_coeff_std = self.class_object.diff_coeff_err
            diff_coeff_units = self.class_object.diff_coeff_units
            ## STORING INPUTS
            self.csv_dict.add( data_title = 'Diffusion_coeff_values', data = [diff_coeff_avg] )
            self.csv_dict.add( data_title = 'Diffusion_coeff_std', data = [diff_coeff_std] )
            self.csv_dict.add( data_title = 'Diffusion_coeff_units', data = [diff_coeff_units] )
    
        ### FUNCTION TO PLOT DIFFUSION AS A FUNCTION OF TIME
        def plot_msd_vs_time(self, save_fig = True):
            '''
            The purpose of this function is to plot the msd as a function of time
            INPUTS:
                self: class object
            OUTPUTS:
                
            '''
            ## IMPORTING
            from MDDescriptors.global_vars.plotting_global_vars import COLOR_LIST, LABELS, LINE_STYLE
            from MDDescriptors.core.plot_tools import create_plot, save_fig_png, create_3d_axis_plot
            
            ## CREATING PLOT
            fig, ax = create_plot()
            
            ## DEFINING TITLE
            ax.set_title('Mean-squared displacement (MSD) as a function of time ')
            ## DEFINING X AND Y AXIS
            ax.set_xlabel('Time (ps)', **LABELS)
            ax.set_ylabel('MSD (1E-5 cm^2/s)', **LABELS)  
            ## PLOTTING
            ax.plot( self.class_object.msd_data[:, 0], self.class_object.msd_data[:, 1], '.-', color='k', **LINE_STYLE)
            ## SAVING FIGURE
            save_fig_png(fig = fig, label = 'msd_vs_time_' + self.pickle_name, save_fig = save_fig)
            return
        
        
    ## DEFINING CLASS
    Descriptor_class = analyze_gmx_msd
    ## DEFINING DATE
    Date='180806'
    ## DEFINING DESIRED DIRECTORY
    
    #%%
    Pickle_loading_file=r"EAM_310.15_K_2_nmDIAM_butanethiol_CHARMM36_Trial_1"
    ## SINGLE TRAJ ANALYSIS
    multi_traj_results = load_multi_traj_pickle(Date, Descriptor_class, Pickle_loading_file )
    
    ## EXTRACTION
    msd_extract = extract_gmx_msd( class_object =  multi_traj_results,pickle_name = Pickle_loading_file )
    
    
    #%%
    
    #### RUNNING MULTIPLE CSV EXTRACTION
    multi_csv_export(Date = Date,
                    Descriptor_class = Descriptor_class,
                    desired_titles = None, 
                    export_class = extract_gmx_msd,
                    export_text_file_name = 'extract_gmx_msd',
                    )    
    
    
    
    
    
    
    
    