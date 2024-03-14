# -*- coding: utf-8 -*-
"""
decoder.py
This script contains all functions that decodes names. For example, decoding a directory name to be comprehensible for python

Created on: 03/25/2018

FUNCTIONS:
    decode_name: decoding name function that can decode a directory name
    convert_pickle_file_names: converts pickle file names based on a conversion type -- great if you have two simulations with different nomenclatures

Author(s):
    Alex K. Chew (alexkchew@gmail.com)
"""

### IMPORTING MODUELS
import sys

### FUNCTION TO DECODE THE DIRECTORY NAMES
def decode_name(name, decode_type=None):
    '''
    The purpose of this function is to decode a directory name. 
    INPUTS:
        name: string of name
        decode_type: type of decoding
            'solvent_effects': decoding for solvent effects
                e.g. mdRun_433.15_6_nm_PRO_10_WtPercWater_spce_dmso
            'self_assembly_np': decoding for nanoparticle
                e.g. hollow_2_nmDIAM_300_K_2_nmEDGE_5_AREA-PER-LIG_4_nm_300_K_butanethiol
            'nanoparticle': decoding for nanoparticle
                e.g. hollow_310.15_K_2_nmDIAM_dodecanethiol_CHARMM36
    OUTPUTS:
        dictionary with the corresponding nomenclature
    NOTE: You can add more to the decoding by including "if decode_type...."
    '''
    ## SPLITTING NAME OF THE DIRECTORY
    split_name = name.split('_')
    
    #######################
    ### SOLVENT EFFECTS ###
    #######################
    ### EXAMPLE: mdRun_433.15_6_nm_PRO_10_WtPercWater_spce_dmso
    ### RESULTS: ['mdRun', '433.15', '6', 'nm', 'PRO', '10', 'WtPercWater', 'spce', 'dmso']
    if decode_type == "solvent_effects":
        ## SEEING IF THE FIRST ONE IS A SPECIFIC NAME
        if split_name[0] ==  'HYDTEST2':
            ## DEFINING DICTIONARY BASED ON SPLIT_NAME RESULTS, e.g. HYDTEST2_1.00nm_300.00_6_nm_xylitol_10_WtPercWater_spce_dmso
            dir_dict= {
                       'distance':              split_name[1],  # Temperature 
                       'temp':                  float(split_name[2]),  # Temperature 
                       'box_size':              int(split_name[3]),  # Initial box length
                       'solute_residue_name':   split_name[5],  # Residue name of the solute
                       'mass_frac_water':       int(split_name[6]),  # Mass fraction of water
                       'cosolvent_name':        split_name[-1], # Name of the cosolvent
                       }
        elif split_name[0] ==  'Expand':
            ## DEFINING DICTIONARY BASED ON EXPANDED NAME Expand_8nm_300.00_6_nm_NoSolute_100_WtPercWater_spce_Pure
            dir_dict= {
                       'temp':                  float(split_name[2]),  # Temperature 
                       'box_size':              int(split_name[3]),  # Initial box length
                       'solute_residue_name':   split_name[5],  # Residue name of the solute
                       'mass_frac_water':       int(split_name[6]),  # Mass fraction of water
                       'cosolvent_name':        split_name[-1], # Name of the cosolvent
                       }
        else:
            ## DEFINING DICTIONARY BASED ON SPLIT_NAME RESULTS
            dir_dict= {
                       'temp':                  float(split_name[1]),  # Temperature 
                       'box_size':              int(split_name[2]),  # Initial box length
                       'solute_residue_name':   split_name[4],  # Residue name of the solute
                       'mass_frac_water':       int(split_name[5]),  # Mass fraction of water
                       'cosolvent_name':        split_name[-1], # Name of the cosolvent
                       }
        ## DEFINING EXCEPTIONS
        try:
            if dir_dict['cosolvent_name'] == 'L' and split_name[-2] == 'GVL':
                dir_dict['cosolvent_name'] = 'GVL'
        except Exception:
            pass
        
        
    ########################
    ### SELF ASSEMBLY NP ###
    ########################
    ### EXAMPLE: hollow_2_nmDIAM_300_K_2_nmEDGE_5_AREA-PER-LIG_4_nm_300_K_butanethiol
    #### RESULTS: ['hollow','2','nmDIAM','300', 'K', '2', 'nmEDGE', '5', 'AREA-PER-LIG', '4', 'nm', '300', 'K', 'butanethiol']
    elif decode_type == "self_assembly_np":
        ## DEFINING DICTIONARY BASED ON SPLIT_NAME RESULTS
        if 'Planar' in name:
            dir_dict= {
                       'shape':                 split_name[0],          # Shape of the gold core
                       'num_lig':              split_name[1],   # Diameter in nm
                       'temperature':           float(split_name[2]),   # Temperature in K
                       'trial'          :       int(split_name[-1]),    # Trial number
                       }
        else:
            dir_dict= {
                       'shape':                 split_name[0],          # Shape of the gold core
                       'diameter':              float(split_name[1]),   # Diameter in nm
                       'temperature':           float(split_name[3]),   # Temperature in K
                       'trial'          :       int(split_name[-1]),    # Trial number
                       }
    ####################
    ### NANOPARTICLE ###
    ####################        
    ### EXAMPLE: hollow_310.15_K_2_nmDIAM_dodecanethiol_CHARMM36
    elif decode_type == "nanoparticle":
        ## DEFINING DICTIONARY BASED ON SPLIT_NAME RESULTS
        if 'Planar' not in name:  # split_name[0] != 'Planar':
            ## SWITCH SOLVENT SIMS
            if split_name[0] == 'switch':
                dir_dict= {
                           'shape'          :       split_name[1].split('-')[-1],          # Shape of the gold core
                           'diameter'       :       float(split_name[4]),   # Diameter in nm
                           'temperature'    :       float(split_name[2]),   # Temperature in K
                           'ligand'         :       split_name[6],          # ligand name
                           'trial'          :       int(split_name[-1]),    # Trial number
                           'cosolvent'      :       split_name[1].split('-')[-2],
                           }
            elif split_name[0] == "MostlikelynpNVTspr":
                
                ## DEFINING MODEL
                np_model_name = ''.join(name.split('-')[1:]).split("_")
                
                dir_dict= {
                           'spring_constant':       split_name[1].split('-')[0], 
                           'shape'          :       np_model_name[0],          # Shape of the gold core
                           'diameter'       :       float(np_model_name[3]),   # Diameter in nm
                           'temperature'    :       float(np_model_name[1]),   # Temperature in K
                           'ligand'         :       np_model_name[5],          # ligand name
                           'trial'          :       int(np_model_name[-3]),    # Trial number
                           }
                
                
            else:
                ## NORMAL SIMS
                dir_dict= {
                           'shape'          :       split_name[0],          # Shape of the gold core
                           'diameter'       :       float(split_name[3]),   # Diameter in nm
                           'temperature'    :       float(split_name[1]),   # Temperature in K
                           'ligand'         :       split_name[5],          # ligand name
                           'trial'          :       int(split_name[-1]),    # Trial number
                           'cosolvent'      : 'None',               # No cosolvent
                           }
            
        else:
            ### FOR PLANAR CASES
            ### EXAMPLE: Planar_310.15_K_dodecanethiol_10x10_CHARMM36_intffGold
            if split_name[0] == 'NVTspr':
                ## DEFINING DICTIONARY BASED ON SPLIT_NAME RESULTS
                dir_dict= {
                           'spring_constant': split_name[1],
                           'shape'          :       split_name[2],          # Shape of the gold core
                           'diameter'       :       None,   # Diameter in nm
                           'temperature'    :       float(split_name[3]),   # Temperature in K
                           'ligand'         :       split_name[5],          # ligand name
                           'trial'          :       int(split_name[-2].split('-')[0]),    # Trial number
                           }
            else:
                ## DEFINING DICTIONARY BASED ON SPLIT_NAME RESULTS
                dir_dict= {
                           'shape'          :       split_name[0],          # Shape of the gold core
                           'diameter'       :       None,   # Diameter in nm
                           'temperature'    :       float(split_name[1]),   # Temperature in K
                           'ligand'         :       split_name[3],          # ligand name
                           'trial'          :       int(split_name[-1]),    # Trial number
                           }
    
    else:
        print("ERROR! Check decoding type (currently: %s)"%(decode_type))
        sys.exit()
    return dir_dict

### FUNCTION TO CONVERT PICKLE FILE NAMES BASED ON YOUR DESIRED SETTINGS
def convert_pickle_file_names(pickle_file_name, conversion_type = None):
    '''
    The purpose of this function is to convert the pickle file name so you can correctly load the pickle. This is useful if you have different nomenclature for different simulation setups.
    INPUTS:
        pickle_file_name: [str]
            name of the pickle file you are running
        conversion_type: [str, default=None]
            conversion type that you want. The list of conversions are below:
                'spherical_np_to_self_assembled_structure': converts nanoparticle systems to self assembled structure
    OUTPUTS:
        updated_pickle_file_name: [str]
            name of the updated pickle file
    '''
    from MDDescriptors.core.decoder import decode_name
    if conversion_type == None:
        updated_pickle_file_name = pickle_file_name
    ## THIS CONVERSION TYPE CONVERTS SPHERICAL NP RUNS TO SELF ASSEMBLED STRUCTURE NOMENCLATURE
    # Ex: EAM_310.15_K_2_nmDIAM_dodecanethiol_CHARMM36_Trial_1 -> EAM_2_nmDIAM_300_K_2_nmEDGE_5_AREA-PER-LIG_4_nm_300_K_butanethiol_Trial_1
    elif conversion_type == "spherical_np_to_self_assembled_structure":
        ## USING DECODING FUNCTIONS
        decoded_name = decode_name(pickle_file_name, decode_type="nanoparticle")
        ## UPDATING PICKLE FILE NAME, e.g. EAM_2_nmDIAM_300_K_2_nmEDGE_5_AREA-PER-LIG_4_nm_300_K_butanethiol_Trial_2
        updated_pickle_file_name = "%s_%d_nmDIAM_300_K_2_nmEDGE_5_AREA-PER-LIG_4_nm_300_K_butanethiol_Trial_%d"%(decoded_name['shape'], decoded_name['diameter'], decoded_name['trial'] ) 
        
    return updated_pickle_file_name

#%% MAIN SCRIPT
if __name__ == "__main__":
    
    ## TESTING NAME
    decoded_name = decode_name( name =  "EAM_300.00_K_2_nmDIAM_ROT014_CHARMM36jul2017_Trial_1",
                                decode_type = 'nanoparticle')

