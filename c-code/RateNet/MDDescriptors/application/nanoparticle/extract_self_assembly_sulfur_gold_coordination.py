# -*- coding: utf-8 -*-
"""
extract_self_assembly_sulfur_gold_coordination.py
This script extracts the results from self_assembly_sulfur_gold_coordination.py

CREATED ON: 09/07/2018

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

### FUNCTION TO CONVERT SURFACE DENSITY TO GRAFTING DENSITY
def convert_surface_density_to_grafting_density( surface_density ):
    '''
    The purpose of this function is to convert surface densities in Ang^2 / lig to grafting densities in lig / nm^2
    INPUTS:
        surface_density: [float]
            surface density in Angstroms^2 / lig
    OUTPUTS:
        grafting_density: [float]
            ligand grafting density in lig / nm^2
    '''
    grafting_density = 1 / surface_density * 100
    return grafting_density

#######################################################
### CLASS FUNCTION TO EXTRACT STRUCTURAL PROPERTIES ###
#######################################################
class extract_self_assembly_sulfur_gold_coordination:
    '''
    The purpose of this class is to extract self_assembly_sulfur_gold_coordination
    INPUTS:
        structure: class from self_assembly_structure
        pickle_name: name of the directory of the pickle
    OUTPUTS:
        csv_info:
            num_ligands_per_frame: number of ligands (y) per frame (x)
            ligand_density_area_angs_per_ligand: ligand density in Angstroms^2/ligand
            final_number_adsorbed_ligands: final number of adsorbed ligands
    '''
    ### INITIALIZATION
    def __init__(self, self_assembly_gold_coordination, pickle_name, decoder_type='self_assembly_np'):
        ## STORING 
        self.self_assembly_gold_coordination = self_assembly_gold_coordination
        ## STORING INFORMATION FOR CSV
        self.csv_dict = csv_dict(file_name = pickle_name, decoder_type = decoder_type)
        
        ## ADDING SURFACE AREAS
        self.csv_dict.add( data_title = 'gold_facet_convex_hull_surface_area (nm2)', data = [self.self_assembly_gold_coordination.gold_facet_convex_hull_surface_area] ) 
        self.csv_dict.add( data_title = 'gold_facet_convex_hull_edge_surface_area (nm2)', data = [self.self_assembly_gold_coordination.gold_facet_convex_hull_edge_surface_area] ) 
        
        ## ADDING GRAFTING DENSITIES
        self.csv_dict.add( data_title = 'facet_grafting_density (lig/nm2)', data = [ convert_surface_density_to_grafting_density(self.self_assembly_gold_coordination.planar_density_angs_per_lig) ] ) 
        self.csv_dict.add( data_title = 'edge_grafting_density (lig/nm2)', data = [ convert_surface_density_to_grafting_density(self.self_assembly_gold_coordination.planar_density_angs_per_lig_edge) ] ) 
        
        ## ADDING FACET TO SURFACE ATOM RATIO
        
        ## EXTRACTING CSV INFO
        self.csv_info = self.csv_dict.csv_info        
        
        return
    


    
        
#%% MAIN SCRIPT
if __name__ == "__main__":
    
    from MDDescriptors.application.nanoparticle.self_assembly_sulfur_gold_coordination import self_assembly_sulfur_gold_coordination
    ## DEFINING CLASS
    Descriptor_class = self_assembly_sulfur_gold_coordination
    
    ## DEFINING DATE
    Date='180913'
    
    ## DEFINING DESIRED DIRECTORY
    # Pickle_loading_file=r"mdRun_433.15_6_nm_ACE_50_WtPercWater_spce_dmso"
    Pickle_loading_file=r"EAM_2_nmDIAM_300_K_2_nmEDGE_5_AREA-PER-LIG_4_nm_300_K_butanethiol_Trial_1"
    
    ## IMPORTING PLOTTING TOOLS
    # from MDDescriptors.application.nanoparticle.plot_self_assembly_structure import plot_self_assembly_structure, extract_self_assembly_structure
    #%%
    #### SINGLE TRAJ ANALYSIS
    multi_traj_results = load_multi_traj_pickle(Date, Descriptor_class, Pickle_loading_file )   
    
    
    #%%
    

    
    
    #%%
    
    #### RUNNING MULTIPLE CSV EXTRACTION
    from MDDescriptors.core.csv_tools import multi_csv_export
    multi_csv_export(Date = Date,
                    Descriptor_class = Descriptor_class,
                    desired_titles = None, # ['ligand_density_area_angs_per_ligand', 'final_number_adsorbed_ligands', 'num_ligands_per_frame'],
                    export_class = extract_self_assembly_sulfur_gold_coordination,
                    export_text_file_name = 'extract_self_assembly_sulfur_gold_coordination',
                    )
    
    