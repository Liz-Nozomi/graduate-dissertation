# -*- coding: utf-8 -*-
"""
global_vars.py
This code contains all global variables.

Created on: 03/15/2019

Author(s):
    - Alex K. Chew (alexkchew@gmail.com)


"""
## IMPORTING FUNCTIONS
import os

## INCLUDING CHECK PATH
from core.check_tools import check_path_to_server

## DEFINING PATH TO MAIN DATA
try:
    PATH_MAIN_PROJECT = check_path_to_server(r"/Volumes/akchew/scratch/storage/2020_SolventNet_Chem_Sci/")
except Exception:
    PATH_MAIN_PROJECT = ""
    pass

# Variable containing all solutes and their reaction temperatures in K
SOLUTE_TO_TEMP_DICT={
        'ETBE': '343.15',
        'tBuOH': '363.15',
        'LGA':  '403.15',
        'PDO': '433.15',
        'FRU': '373.15',
        'CEL': '403.15',
        'XYL': '403.15',
        }

## DEFINING LIST OF POSSIBLE SOLVENTS
POSSIBLE_SOLVENTS=['HOH','DIO','THF','GVLL', 'dmso', 'ACE', 'ACN']

## DEFINING PREFERRED ORDERED OF REACTANTS
ORDER_OF_REACTANTS = ['ETBE', 
                      'tBuOH', 'TBA', 
                      'LGA',
                      'PDO',
                      'FRU',
                      'CEL',
                      'XYL',
                      'GLU',
                      'DIO',
                      'GVL',
                      'THF',
                      ]

## DEFINING K_H2O CONSTANTS FOR REACTION, units of L mol^-1 s^-1
# All constants taken from ESI of EES 2018 paper:
#   https://pubs.rsc.org/en/content/articlelanding/2018/ee/c7ee03432f
K_H2O_CONSTANTS = { 
        'XYL': 1.04e-4,
        'FRU': 1.95e-4,
        'CEL': 1.44e-2,
        'PDO': 1.26e-4,
        'LGA': 2.27e-2,
        'TBA': 1.39e-3,
        'tBuOH': 1.39e-3, # same as TBA
        'ETBE': 1.04e-4
        }

### CREATING DICTIONARY FOR COLOR
COSOLVENT_COLORS={
            'DIO': 'red',
            'GVL': 'blue',
            'THF': 'green',
            'dmso': 'purple',
            }

## DICTIONARY FOR DEFAULT RUNS FOR CNN NETWORKS
CNN_DICT = {
        'validation_split': 0.2,
        'batch_size': 18, # higher batches results in faster training, lower batches can converge faster
        'metrics': ['mean_squared_error'],
        'shuffle': True, # True if you want to shuffle the training data
        }

## DEFINING SAMPLING DICT        
SAMPLING_DICT = {
        'name': 'stratified_learning',
        'split_percentage': 0.6, # 3,
        }

## DEFINING PATH TO CNN PROJECT
PATH_CNN_PROJECT=r"R:\scratch\3d_cnn_project"

## DEFINING PATH DICTIONARY
DEFAULT_PATH_DICT = {
        'database_path': os.path.join(PATH_CNN_PROJECT, "database"),
        'class_file_path': os.path.join(PATH_CNN_PROJECT, "database", "Experimental_Data", "solvent_effects_regression_data.csv"),
        'combined_database_path': os.path.join(PATH_CNN_PROJECT, "combined_data_set"),
        'path_image_dir': r"C:\Users\akchew\Box Sync\VanLehnGroup\0.Manuscripts\Solvent_effects_3D_CNNs\Images",
        'sim_path': os.path.join(PATH_CNN_PROJECT, "simulations"),
        'path_pickle': os.path.join(PATH_CNN_PROJECT, "storage"),
        }


## DEFINING INPUTS FOR DESCRIPTOR FUNCTION
INPUTS_FOR_DESCRIPTOR_FXN ={
        'path_csv': DEFAULT_PATH_DICT['class_file_path'],
        'col_names': [ 'delta' , 'gamma', 'tau'], # , 
        'col_matching_list': [['solute','cosolvent','mass_frac'],
                                  ['solute','cosolvent','mass_frac_water']
                                  ],
        }