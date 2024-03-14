# -*- coding: utf-8 -*-
"""
global_vars.py
The purpose of this script is to contain all global variables for plotting purposes. Things like font size, etc. should be placed here!
This script is purposely designed for the nanoparticle project

## VARIABLES:
- GOLD_ATOM_NAME: Gold atom names
- LIGAND_HEAD_GROUP_ATOM_NAME: head group for the ligands
- GOLD_SULFUR_CUTOFF: gold-sulfur cutoff to denote a bond
- GOLD_GOLD_CUTOFF_BASED_ON_SHAPE: gold-gold cutoff distance based on the gold shape

CREATED ON : 05/10/2018

AUTHOR(S):
    Alex K. Chew (alexkchew@gmail.com)
"""

from MDDescriptors.core.initialize import checkPath2Server
## ADDING PATH FUNCTIONS 
import MDDescriptors.core.check_tools as check_tools

## TRY TO IMPORT
try:
    
    ### IMPORTING LIGAND REISDUE NAMES
    from MDDescriptors.application.nanoparticle.core.find_ligand_residue_names import load_all_ligand_residue_names
    ## LOADING ALL RESIDUE NAMES
    LIGAND_RESIDUE_NAMES = load_all_ligand_residue_names()
except Exception:
    pass

# ligand_residue_list as LIGAND_RESIDUE_NAMES
### DEFINING GLOBAL VARIABLES
GOLD_ATOM_NAME="Au" # Atom name for gold: used to find the center of the gold lattice
GOLD_RESIDUE_NAME="AUNP" # Residue name for gold, which we will use to find self-assembly proccesses
LIGAND_HEAD_GROUP_ATOM_NAME="S1" # Atom name for the head group of ligands. This is as a reference

## DEFINING CUTOFF FOR GOLD-SULFUR --- based on:
#   Djebaili, T., Richardi, J., Abel, S. & Marchi, M. Atomistic simulations of the surface coverage of large gold nanocrystals. J. Phys. Chem. C 117, 17791–17800 (2013).
GOLD_SULFUR_CUTOFF = 0.327 # nms (3.27 angstroms)
## TAKEN FROM MD BUILDERS
# from MDBuilders.application.nanoparticle.global_vars import GOLD_SULFUR_CUTOFF

## DEFINING GOLD_GOLD_CUTOFF: FOUND FROM GOLD-GOLD RDF FIRST SOLVATION PEAK MINIMUM
GOLD_GOLD_CUTOFF_BASED_ON_SHAPE={ 'EAM' : 0.34, # nm
                                  'spherical': 0.37, # nm
                                  'hollow': 0.41, # nm
                                  'Planar': 0.29, # nm
                                  }

## DEFINING DEFAULT IMAGE
DEFAULT_IMAGE_OUTPUT = check_tools.check_path(r"/home/akchew/scratch/nanoparticle_project/output_images")
# checkPath2Server() 
#  
# 


## DEFINING LIGANDS NAMES
#LIGAND_RESIDUE_NAMES = ['BUT', 'OCT', 'HEX', 'DOD', 'DEC', 'EIC', 'HEP', 'PEN', 'PED', 'COH', 'COO', 'NP5', 'RNS', 'HED', 'NH3', 'COH', 'COO',
#                        'RO1', 'RO2', 'RO3', 'RO4', 'RO5', 'RO6', 'RO7', 'RO8', 'RO9', 'R10', 'R11', 'R12',
#                        'NH2', 'CON', 'UF3']

## DEFINING WORKING DIRECTORY
NP_WORKING_DIR=checkPath2Server(r"R:\scratch\nanoparticle_project\scripts\analysis_scripts")


## DEFINING DEFAULT VARIABLES
MDDESCRIPTORS_DEFAULT_VARIABLES = {
        'calc_nanoparticle_bundling_groups':
            {'ligand_names': LIGAND_RESIDUE_NAMES, # name of all possible ligands
             'min_cluster_size':  3, # minimum constraint for a cluster
             'weights': [0.5, 0.5], # weights between angle and distance between end groups
             'save_disk_space': True,
             'separated_ligands': False, # They are often not bound
             'itp_file': 'sam.itp' # itp file
             }
        
    
        }

## DEFINING VARIABLES
GMX_XVG_VARIABLE_DEFINITION = {'GYRATE': 
                                    [ 
                                    [0, 'frame',    int],
                                    [1, 'Rg',       float ],
                                    [2, 'Rg_X',     float],
                                    [3, 'Rg_Y',     float],
                                    [4, 'Rg_Z',     float],
                                ]
                               }



## GLOBAL VARIABLE FOR DIFFERENT R GROUPS
'''
To add R groups, simply open up VMD, then load the entire ligand, e.g. R:\scratch\nanoparticle_project\prep_system\prep_ligand\final_ligands\ROT010
Type in command: show_atom_labels_serial
'''
R_GROUPS_PER_LIGAND={
        'r_group':{
            'RO1': [ 'N63', 
                     'C68', 'H69', 'H70', 'H71',
                     'C72', 'H73', 'H74', 'H75',
                     'C64', 'H65', 'H66', 'H67'
                    ],
            'RO2': [ 'N63',
                     'C79', 'H80', 'H81', 'H82',
                     'C75', 'H76', 'H77', 'H78',
                     'C64', 'H65', 'H66',
                     'C67', 'H68', 'H69',
                     'C70', 'H71', 'H72',
                     'O73', 'H74',
                    ],
            'RO3': [ 'N63',
                     'C76', 'H77', 'H78', 'H79',
                     'C80', 'H81', 'H82', 'H83',
                     'C64', 'H65', 'H66',
                     'C67', 'H68', 'H69',
                     'C70', 'H71', 'H72',
                     'N73', 'H74', 'H75',
                    ],
            'RO4': [ 'N63',
                     'C75', 'H76', 'H77', 'H78',
                     'C79', 'H80', 'H81', 'H82',
                     'C64', 'C66', 'H70', 'C69', 'H73', 'C65', 'H74', 'C68', 'H72', 'C67', 'H71'                        
                    ],
            'RO5': [ 'N63',
                     'C83', 'H84', 'H85', 'H86',
                     'C87', 'H88', 'H89', 'H90',
                     'C64', 'H65', 'H66',
                     'C67', 'H68', 'H69',
                     'C70', 'H71', 'H72',
                     'C73', 'H74', 'H75',
                     'C76', 'H77', 'H78',
                     'C79', 'H80', 'H81', 'H82'   
                    ],
            'RO6': [ 'N63',
                     'C81', 'H82', 'H83', 'H84',
                     'C85', 'H86', 'H87', 'H88',
                     'C64', 'H75',
                     'C68', 'H73', 'H79',
                     'C66', 'H71', 'H77',
                     'C67', 'H72', 'H78',
                     'C65', 'H70', 'H76',
                     'C69', 'H74', 'H80'
                    
                    ],
            'RO7': [
                    'N63', 
                    'C91', 'H92', 'H93', 'H94',
                    'C95', 'H96', 'H97', 'H98',
                    'C64', 'H75',
                    'C68', 'H73', 'H78',
                    'C66', 'H77', 'H71',
                    'C67', 'H72', 
                    'C65', 'H76', 'H70',
                    'C69', 'H74', 'H79',
                    'C80', 
                    'C83', 'H87',
                    'C84', 'H88',
                    'C81', 'H90',
                    'C85', 'H89',
                    'C82', 'H86',
                    ],
            'RO8': [
                    'N63',
                    'C93', 'H94', 'H95', 'H96',
                    'C97', 'H98', 'H99', 'H100',
                    'C64', 'H75', 
                    'C68', 'H73', 'H78', 
                    'C66', 'H71', 'H77',
                    'C67', 'H72',
                    'C65', 'H76', 'H70',
                    'C69', 'H74', 'H79',
                    'C80',
                    'C89', 'H90', 'H91', 'H92',
                    'C81', 'H82', 'H83', 'H84',
                    'C85', 'H86', 'H87', 'H88',                            
                    ],
            'RO9': [
                    'O63', 'H64'
                    ],
            'R10': [
                    'N63', 
                    'C64', 'H65', 'H66', 'H67',
                    'C68', 'H69' , 'H70', 'H71',
                    'C72', 'H73', 'H74',
                    'C75', 'H76', 'H77', 'H78',                    
                    ],
            'R11': [
                    'N63', 
                    'C64', 'H65', 'H66', 'H67',
                    'C68', 'H69', 'H70', 'H71', 
                    'C72', 'H73', 'H74',
                    'C75', 'H76', 'H77', 
                    'C78', 'H79', 'H80',
                    'C81', 'H82', 'H83', 'H84',
                    ],
            'R12': [
                    'N63', 
                    'C68', 'H69', 'H70', 'H71', 
                    'C64', 'H65', 'H66', 'H67',
                    'C72', 'H73', 'H74',
                    'C75', 'H76', 'H77',
                    'C78', 'H79', 'H80',
                    'C81', 'H82', 'H83',
                    'C84', 'H85', 'H86',
                    'C87', 'H88', 'H89',
                    'C90', 'H91', 'H92',
                    'C93', 'H94', 'H95',
                    'C96', 'H97', 'H98',
                    'C99', 'H100', 'H101', 'H102',
                    ],
                    },
        'r_group_c_only':{
            'RO1': [ 
                     'C68',
                     'C72',
                     'C64',
                    ],
            'RO2': [ 
                     'C79',
                     'C75',
                     'C64',
                     'C67',
                     'C70',
                    ],
            'RO3': [ 
                     'C76', 
                     'C80', 
                     'C64', 
                     'C67', 
                     'C70', 
                    ],
            'RO4': [ 
                     'C75', 
                     'C79', 
                     'C64', 'C66',  'C69', 'C65', 'C68', 'C67'               
                    ],
            'RO5': [ 
                     'C83', 
                     'C87', 
                     'C64', 
                     'C67', 
                     'C70', 
                     'C73', 
                     'C76', 
                     'C79',   
                    ],
            'RO6': [ 
                     'C81',
                     'C85', 
                     'C64', 
                     'C68', 
                     'C66', 
                     'C67', 
                     'C65', 
                     'C69',
                    
                    ],
            'RO7': [
                    
                    'C91', 
                    'C95', 
                    'C64', 
                    'C68', 
                    'C66', 
                    'C67',  
                    'C65', 
                    'C69', 
                    'C80', 
                    'C83', 
                    'C84', 
                    'C81', 
                    'C85', 
                    'C82', 
                    ],
            'RO8': [
                    
                    'C93', 
                    'C97', 
                    'C64',  
                    'C68', 
                    'C66', 
                    'C67', 
                    'C65', 
                    'C69', 
                    'C80',
                    'C89', 
                    'C81', 
                    'C85',                             
                    ],
            'RO9': [
                    
                    ],
            'R10': [
                    'C64', 
                    'C68', 
                    'C72', 
                    'C75', 
                    ],
            'R11': [
                    'C64', 
                    'C68',  
                    'C72', 
                    'C75', 
                    'C78', 
                    'C81', 
                    ],
            'R12': [ 
                    'C68', 
                    'C64', 
                    'C72',
                    'C75', 
                    'C78', 
                    'C81', 
                    'C84', 
                    'C87', 
                    'C90', 
                    'C93', 
                    'C96', 
                    'C99', 
                    ],
                    
                    },
            }