#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
global_vars.py
This code contains all global vars for the NP hydrophobicity project

Written by: Alex K. Chew (03/16/2020)

USAGE EXAMPLE:
    from MDDescriptors.application.np_hydrophobicity.global_vars import PARENT_SIM_PATH

"""
## IMPORTING CHECK TOOLS
import MDDescriptors.core.check_tools as check_tools

## DEFINING IMAGE LOCATION
IMAGE_LOCATION = check_tools.check_path(r"S:\np_hydrophobicity_project\output_images")
# r"C:\Users\akchew\Box Sync\VanLehnGroup\2.Research Documents\Alex_RVL_Meetings\20200323\images\np_hydrophobicity"
#  r"/Users/alex/Box/VanLehnGroup/2.Research Documents/Alex_RVL_Meetings/20200323/images"

## DEFINING PARENT SIM PATH
PARENT_SIM_PATH = check_tools.check_path(r"S:\np_hydrophobicity_project\simulations")
NP_SIM_PATH = check_tools.check_path(r"R:\scratch\nanoparticle_project\simulations")        
        
## DEFINING NEIGHBORS LOCATION
COMBINE_NEIGHBORS_DIR="compute_neighbors"
COMBINE_NEIGHBORS_LOG="neighbors.log"

## DEFINING GRID LOCATION
GRID_LOC="grid-45000_50000"
GRID_OUTFILE="out_willard_chandler.dat"

## DEFINING MU OUTOUT
OUT_HYDRATION_PDB = "out_hydration.pdb"
MU_PICKLE="mu.pickle"

## DEFINING PROD GRO FILE
PROD_GRO="sam_prod.gro"

## DEFINING PATH SICT
PATH_DICT={
        'Planar_SAMs': '20200403-Planar_SAMs-5nmbox_vac_with_npt_equil',
        'Planar_SAMs_unrestrained': r"20200413-Planar_SAMs-5nmbox_vac_with_npt_equil_NORESTRAINT",
        'GNP': '20200618-Most_likely_GNP_water_sims_FINAL',
            # '20200401-Renewed_GNP_sims_with_equil',
        'GNP_PEG': '20200401-Renewed_GNP_sims_with_equil_other_molecules_PEG',
        'GNP_MIXED': '20200411-mixed_sam_frozen',
        'GNP_6nm': r"20200414-6nm_particles_frozen" ,
        'GNP_unsaturated': '20200618-Most_likely_GNP_water_sims_FINAL',
            # r"20200419-unsaturated_frozen",
        'GNP_branched': '20200618-Most_likely_GNP_water_sims_FINAL',
            # r"20200421-branch_frozen",
        'GNP_least_likely': r"20200421-2nm_Least_likely_config",
        'Planar_C3': r"20200515-planar_short_lig_frozen",
        }

## DEFINING PURE WATER SIM DICT
PURE_WATER_SIM_DICT={
        'main_dir': 'PURE_WATER_SIMS',
        'parent_dir': 'wateronly-6_nm-tip3p-300.00_K',
        'wc_folder': '0.24-0.33-2000-50000',
        'mu_pickle': MU_PICKLE,
        }

## DEFINING RELABEL DICT
RELABEL_DICT = {
        'dodecanethiol': "CH3",
        "C11OH": "OH",
        "C11COOH": "COOH",
        "C11NH2": "NH2",
        "C11CONH2": "CONH2",
        "C11CF3": "CF3",
    
        ## UNSAT        
        "C11double67OH": "OH_unsat",
        'dodecen-1-thiol':  "CH3_unsat",
        "C11double67NH2": "NH2_unsat",
        "C11double67COOH": "COOH_unsat",
        "C11double67CONH2": "CONH2_unsat",
        "C11double67CF3": "CF3_unsat",
        
        ## BRANCHED
        'C11branch6CF3': 'CF3_br',
        'C11branch6CH3': 'CH3_br',
        'C11branch6CONH2': 'CONH2_br',
        'C11branch6COOH': 'COOH_br',
        'C11branch6NH2': 'NH2_br',
        "C11branch6OH": "OH_br",
        
        ## C4
        'butanethiol': 'C3_CH3',
        'C3OH': 'C3_OH',
        }

# COLORS: https://matplotlib.org/3.1.0/gallery/color/named_colors.html
LIGAND_COLOR_DICT = {
        'dodecanethiol': 'black',
        'C11OH': 'red',
        'C11COOH': 'brown',
            # 'magenta',
        'C11NH2': 'blue',
        'C11CONH2': 'orange',
        'C11CF3': 'gray',
            # 'cyan',
        'C11COO': 'orange',
        'C11NH3': 'green',
        'ROT001': 'purple',
        'ROT002': 'olive',
        'ROT003': 'lightblue',
        'ROT004': 'goldenrod',
        'ROT005': 'magenta',
        'ROT006': 'maroon',
        'ROT007': 'slategrey',
        'ROT008': 'indigo',
        'ROT009': 'chocolate',
        'ROTPE1': 'greenyellow',
        'ROTPE2': 'powderblue',
        'ROTPE3': 'darksalmon',
        'ROTPE4': 'deeppink',



        ## MIXED SAM COLORS
        "C11COO,C11COOH": 'purple',
        "C11NH3,C11NH2": 'cyan',
        
        ## UNSATURATED GROUPS
        'dodecen-1-thiol':  'black',
        "C11double67NH2": 'blue',
        "C11double67COOH": 'magenta',
        "C11double67CONH2": 'orange',
        "C11double67CF3": 'cyan',
        'C11double67OH': 'red',
        
        ## BRANCHED GROUPS
        'C11branch6CF3': 'cyan',
        'C11branch6CH3': 'black',
        'C11branch6CONH2': 'orange',
        'C11branch6COOH': 'magenta',
        'C11branch6NH2': 'blue',
        "C11branch6OH": "red",
         
        ## SHORTER LIGANDS
        'butanethiol': 'black',
        'C3OH': 'red',
        
        }

## DEFINING DEFAULT WC ANALYSIS
DEFAULT_WC_ANALYSIS="26-0.24-0.1,0.1,0.1-0.33-all_heavy-2000-50000-wc_45000_50000"

## DEFINING PREFIX FOR GNP AND PLANAR
PREFIX_SUFFIX_DICT={
        'GNP': {
                'prefix': 'MostlikelynpNVTspr_50-EAM_300.00_K_2_nmDIAM_',
                'suffix': '_CHARMM36jul2017_Trial_1_likelyindex_1',
                },
        'GNP_6nm': {
                'prefix': 'MostlikelynpNVTspr_50-EAM_300.00_K_6_nmDIAM_',
                'suffix': '_CHARMM36jul2017_Trial_1_likelyindex_1',
                },
        'Planar': {
                'prefix': 'NVTspr_50_Planar_300.00_K_',
                'suffix': '_10x10_CHARMM36jul2017_intffGold_Trial_1-5000_ps',
                },
        'Planar_no_restraint': {
                'prefix': 'NVTspr_0_Planar_300.00_K_',
                'suffix': '_10x10_CHARMM36jul2017_intffGold_Trial_1-5000_ps',
                },
        'GNP_PEG': {
                'prefix': 'MostlikelynpNVTspr_50-EAM_300.00_K_2_nmDIAM_',
                'suffix': '_CHARMM36jul2017_Trial_1_likelyindex_1',
                },
        'GNP_MIXED':{
                'prefix': 'MostlikelynpNVTspr_50-EAM_300.00_K_2_nmDIAM_',
                'suffix': '_0.53,0.47_CHARMM36jul2017_Trial_1_likelyindex_1',
                },
        'GNP_least_likely': {
                'prefix': 'MostlikelynpNVTspr_50-EAM_300.00_K_2_nmDIAM_',
                'suffix': '_CHARMM36jul2017_Trial_1_likelyindex_4001',
                },
                
        }