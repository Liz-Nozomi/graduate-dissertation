# -*- coding: utf-8 -*-
"""
extract_traj_info_general.py
The purpose of this script is to run a full general extraction protocol. 
The idea is that we by-pass the multidescriptor approach in generating pickles 
and go straight to extraction of number of atoms, ensemble volume, and so on. 

Written by: Alex K. Chew (alexkchew@gmail.com, 11/25/2019)


"""

## IMPORTING IMPORTANT MODULES
import os
import glob
import pandas as pd

### IMPORTING FUNCTION TO GET TRAJ
from MDDescriptors.traj_tools.multi_traj import load_multi_traj_pickle, find_multi_traj_pickle, load_multi_traj_pickles
from MDDescriptors.core.csv_tools import csv_info_add, csv_info_new, multi_csv_export, csv_info_export, csv_info_decoder_add
## IMPORTING FUNCTION THAT WAS USED
from MDDescriptors.core.general_traj_info import general_traj_info

## IMPORTING TOOLS FOR GENERAL TRAJ INFO
from MDDescriptors.application.solvent_effects.extract_general_traj import extract_general_traj_info

## LOADING TOOLS
import MDDescriptors.core.import_tools as import_tools # Loading trajectory details

## DECODER
from MDDescriptors.core.decoder import decode_name


#%% MAIN SCRIPT
if __name__ == "__main__":
    
    
    ## DEFINING DIRECTORY
    main_dir=r"R:/scratch/SideProjectHuber/Simulations/TOPICS_IN_CATALY_FINAL_FRU_LEV_SIGMA"
    # r"R:/scratch/SideProjectHuber/Simulations/TOPICS_IN_CATALY_FINAL_CYO_CYE_SIGMA"
    
    ## DEFINING GRO AND XTC FILE
    gro_file="mixed_solv_prod.gro"
    xtc_file="mixed_solv_prod.xtc"
    
    ## DEFINING OUTPUT SPREADSHEET NAME
    output_csv="extract_traj_info_general.csv"
    
    ## DEFINING DECODER TYPE
    decode_type = "solvent_effects"
    
    ## DEFINING PATH
    path_csv = os.path.join(main_dir, output_csv)
    
    ## GLOBBING EACH DIRECTORY
    dir_list=[ each_dir for each_dir in glob.glob(main_dir + os.path.sep + '*' + os.path.sep)
                                     if os.path.isdir(each_dir) ]
    
    ## LIMITING DIR NAME FOR NOW
    # dir_list = dir_list[0:2]
    
    ### DEFINING INPUT DATA
    input_details={ 'verbose': True
                    }
    
    ## STORING LIST
    traj_output_list = []
    
    ## LOOPING THROUGH EACH DIRECTORY
    for each_dir in dir_list:
        ## GETTING BASENAME
        current_dir_basename = os.path.basename(os.path.dirname(each_dir))
        
        ## DECODING THE NAME
        decoded_name_dict = decode_name(name = current_dir_basename, 
                                        decode_type = decode_type)
        
        ### LOADING TRAJECTORY
        traj_data = import_tools.import_traj( directory = each_dir, # Directory to analysis
                                              structure_file = gro_file, # structure file
                                              xtc_file = xtc_file, # trajectories
                                              )
        
        ## RUNNING GENERAL INFO
        traj_info  = general_traj_info(traj_data, **input_details)
        
        ## CREATING DICTIONARY
        traj_dict = {
                "Name" : current_dir_basename,
                'ens_vol(nm3)': traj_info.ens_volume,
                'ens_length(nm)': traj_info.ens_length,
                'total_atoms': traj_info.total_atoms,
                'total_frames': traj_info.total_frames,
                }
        
        ## ADDING DECODED NAME ETO DICTIONARY
        traj_dict.update(decoded_name_dict)
        
        ## ADDING RESIDUE INFORMATION
        if decode_type == "solvent_effects":
            for each_residue in traj_info.residues.keys():
                ## SEEING NUMBER 
                num_residues = traj_info.residues[each_residue]
                ## SEEING IF WATER
                if each_residue == 'HOH':
                    traj_dict['N_H2O'] = num_residues
                else:
                    ## MAKING SURE > 1 RESIDUES
                    if num_residues > 1:
                        traj_dict['N_org'] = num_residues
        ## APPENDING
        traj_output_list.append(traj_dict)

    
    #%%
    
    ## CONVERTING TO PANDAS
    df = pd.DataFrame(traj_output_list)
    
    ## PRINTING
    print("Storing CSV in: %s"%( path_csv ) )
    
    ## EXPORTING PANDAS
    df.to_csv(path_csv)
    
    