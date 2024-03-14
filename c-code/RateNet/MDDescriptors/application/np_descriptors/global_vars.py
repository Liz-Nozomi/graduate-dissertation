# -*- coding: utf-8 -*-
"""
global_vars.py
This contains global variables for the np descriptor project

Variables:
    NPLM_SIM_DICT:
        simulation dictionary for nplm sims
    IMAGE_LOC:
        image location to store all images

Written by: Alex K. Chew (02/28/2020)
"""
## IMPORTING CHECK TOOLS
import os
import MDDescriptors.core.check_tools as check_tools
import glob

## DEFINING SIM DICT
SIM_DICT={          
        'ROT_WATER_SIMS' : 'ROT_WATER_SIMS',
        'ROT_DMSO_SIMS' : 'ROT_DMSO_SIMS',
        }
            
## DEFINING SAVE IMAGE LOCATION
IMAGE_LOC =check_tools.check_path(r"R:\scratch\nanoparticle_project\nanoparticle_project\output_images")
#  r"/Users/alex/Box/VanLehnGroup/2.Research Documents/Alex_RVL_Meetings/20200323/images/nplm_project"
# r"C:\Users\akchew\Box Sync\VanLehnGroup\2.Research Documents\Alex_RVL_Meetings\20200316\images\nplm_project"
# r"C:\Users\akchew\Box Sync\VanLehnGroup\2.Research Documents\Alex_RVL_Meetings\20200224\images\nplm_project"

## DEFINING PARENT SIM PATH
PARENT_DIR = check_tools.check_path(r"R:\scratch\nanoparticle_project")
PARENT_SIM_PATH = os.path.join(PARENT_DIR,
                               "simulations")

## DEFINING GOLD NAME
GOLD_RESNAME="AUNP"

### DEFINING ANALYSIS FOLDER
ANALYSIS_FOLDER = "analysis"

## DEFINING PICKLE NAME
PICKLE_NAME="results.pickle"

## DEFINING PATH TO DATABASE
PATH_DATABASE = os.path.join(PARENT_DIR,
                             "database")