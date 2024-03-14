# -*- coding: utf-8 -*-
"""
accessible_hydroxyl_frac_extract.py
The purpose of this script is to extract the accessible hydroxyl fraction

Author(s):
    Alex K. Chew (alexkchew@gmail.com)
"""
### IMPORTING FUNCTION TO GET TRAJ
from MDDescriptors.traj_tools.multi_traj import load_multi_traj_pickle, find_multi_traj_pickle
from MDDescriptors.geometry.accessible_hydroxyl_fraction import calc_accessible_hydroxyl_fraction

#%% MAIN SCRIPT
if __name__ == "__main__":
    ## DEFINING CLASS
    Descriptor_class = calc_accessible_hydroxyl_fraction
    ## DEFINING DATE
    Date='180521'
    ## DEFINING DESIRED DIRECTORY
    Pickle_loading_file=r"mdRun_373.15_6_nm_FRI_100_WtPercWater_spce_Pure"
#    Pickle_loading_file=r"mdRun_433.15_6_nm_PDO_50_WtPercWater_spce_dmso"
    ## SINGLE TRAJ ANALYSIS
    multi_traj_results = load_multi_traj_pickle(Date, Descriptor_class, Pickle_loading_file )
    ## EXTRACTION
    hydroxyl_frac = multi_traj_results
    print(hydroxyl_frac.accessible_hydroxyl_frac)
    
    

