# -*- coding: utf-8 -*-
"""
extract_nanoparticle_gold_to_ligand_structure.py
The purpose of this script is to extract the gold to ligand structure

CREATED ON: 06/29/2018

AUTHOR(S):
    Alex K. Chew (alexkchew@gmail.com)

"""

### IMPORTING FUNCTION TO GET TRAJ
from MDDescriptors.traj_tools.multi_traj import load_multi_traj_pickle, find_multi_traj_pickle, load_multi_traj_pickles, load_multi_traj_multi_analysis_pickle
from MDDescriptors.core.csv_tools import csv_info_add, csv_info_new, multi_csv_export, csv_info_export, csv_info_decoder_add,csv_dict

### IMPORTING GLOBAL VARIABLES
from MDDescriptors.global_vars.plotting_global_vars import COLOR_LIST, LABELS, LINE_STYLE
from MDDescriptors.core.plot_tools import create_plot, save_fig_png, create_3d_axis_plot
import MDDescriptors.core.calc_tools as calc_tools
### MATH MODULES
import numpy as np

## INCLUDING SCRIPTs

## IMPORTING FUNCTION THAT WAS USED
from MDDescriptors.application.nanoparticle.nanoparticle_rdf import plot_rdf

## IMPORTING TOOLS
from MDDescriptors.geometry.rdf_extract import find_first_solvation_shell_rdf


#####################################
### CLASS FUNCTION TO EXTRACT RDF ###
#####################################
class extract_nanoparticle_gold_to_ligand_structure:
    '''
    The purpose of this function is to extract the nanoparticle structural function
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
        ## STORING PICKLE NAME
        self.pickle_name = pickle_name
        ## STORING INFORMATION FOR CSV
        self.csv_dict = csv_dict(file_name = pickle_name, decoder_type = decoder_type )
        
        ## PLOTTING RDFs
        self.plot_gold_gold_rdf()
        self.plot_gold_surface_sulfur_rdf()
        
        ## FINDING FIRST SOLVATION SHELL
        self.find_first_solv_shell()
        
        ## STORING INTO CSV INFO
        self.csv_info = self.csv_dict.csv_info

        return
    
    ### FUNCTION TO FIND THE FIRST SOLVATION SHELL
    def find_first_solv_shell(self):
        '''
        The purpose of this function is to find the first solvation shell information
        INPUTS:
            self: class object
            
        '''
        ## FINDING FIRST SOLVATION SHELL
        gold_gold_first_solv_shell = find_first_solvation_shell_rdf(g_r = self.analysis_class.gold_rdf_g_r,
                                                          r = self.analysis_class.gold_rdf_r)
        ## FINDING FIRST SOLVATION SHELL
        gold_surface_sulfur_first_solv_shell = find_first_solvation_shell_rdf(g_r = self.analysis_class.gold_surface_sulfur_rdf_g_r,
                                                          r = self.analysis_class.gold_surface_sulfur_rdf_r)
        
        ## STORING INPUTS
        self.csv_dict.add( data_title = 'gold_gold_first_solv_min_r_nm',  data = [ gold_gold_first_solv_shell['min']['r'] ]  )
        self.csv_dict.add( data_title = 'gold_surface_sulfur_first_solv_min_r_nm',  data = [ gold_surface_sulfur_first_solv_shell['min']['r'] ]  )
        
        return
    
    ### FUNCTION TO PLOT THE RDF
    def plot_gold_gold_rdf(self,save_fig=True):
        '''
        The purpose of this function is to plot the rdf
        INPUTS:
            self: class object
            save_fig: [logical, default=True] True if you want to save the figure
        OUTPUTS:
            void
        '''
        ## PLOTTING THE RDF
        fig, ax = plot_rdf(r = self.analysis_class.gold_rdf_r,
                           g_r = self.analysis_class.gold_rdf_g_r)
        ## STORING IMAGE
        save_fig_png(fig = fig, label = 'nanoparticle_rdf_gold_gold_' + self.pickle_name, save_fig = save_fig)
        return
    
    ### FUNCTION TO PLOT THE RDF
    def plot_gold_surface_sulfur_rdf(self,save_fig=True):
        '''
        The purpose of this function is to plot the rdf between surface gold atoms and sulfur atoms
        INPUTS:
            self: class object
            save_fig: [logical, default=True] True if you want to save the figure
        OUTPUTS:
            void
        '''
        ## PLOTTING THE RDF
        fig, ax = plot_rdf(r = self.analysis_class.gold_surface_sulfur_rdf_r,
                           g_r = self.analysis_class.gold_surface_sulfur_rdf_g_r)
        ## STORING IMAGE
        save_fig_png(fig = fig, label = 'nanoparticle_rdf_gold_surface_sulfur_' + self.pickle_name, save_fig = save_fig)
        return
    
    
    




#%% MAIN SCRIPT
if __name__ == "__main__":
    from MDDescriptors.application.nanoparticle.nanoparticle_gold_to_bundling_ligand_structure import nanoparticle_gold_to_ligand_structure
    
    ## DEFINING CLASS
    Descriptor_class = nanoparticle_gold_to_ligand_structure
    ## DEFINING DATE
    Date='180806' # '180622'
    ## DEFINING DESIRED DIRECTORY
    Pickle_loading_file=r"EAM_310.15_K_4_nmDIAM_hexadecanethiol_CHARMM36_Trial_1"
    ## SINGLE TRAJ ANALYSIS
    multi_traj_results = load_multi_traj_pickle(Date, Descriptor_class, Pickle_loading_file )
    
    #%%
    
    ## RUNNING EXTRACTION PROCEDURE
    extract_results = extract_nanoparticle_gold_to_ligand_structure(analysis_class = multi_traj_results, 
                                                                    pickle_name = Pickle_loading_file)
    
    #%%
    
    #### RUNNING MULTIPLE CSV EXTRACTION
    df_1, dfs_2, dfs_2_names = multi_csv_export(Date = Date,
                                                Descriptor_class = Descriptor_class,
                                                desired_titles = None, 
                                                export_class = extract_nanoparticle_gold_to_ligand_structure,
                                                export_text_file_name = 'extract_nanoparticle_gold_to_ligand_structure',
                                                )    