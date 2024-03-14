# -*- coding: utf-8 -*-
"""
willard_chandler_global_vars.py
This script contains all global variables used. 

VARIABLES:
    WC_DEFAULTS:
        Defaults used for the Willard-Chandler surface. 
    TECHNICAL_DEFAULTS:
        Technical defaults used for running calculations.
"""

## DEFINING WILLARD CHANDLER VARIABLES
WC_DEFAULTS={
        'alpha' : 0.24, # Describes the width. This is the same value used in the WC paper.
        'sigma' : 0.24, # Describes the width. This is the same value used in the WC paper.
        'contour': 25.6, # 80% of the bulk water density. 
                         #    contour = 16. # 0.5 bulk <-- used in the WC paper
                         #    contour = 20.8 # 0.65 bulk
                         #    contour = 25.6 # 0.8 bulk 
        'mesh': [0.1, 0.1, 0.1] , # Grid mesh 0.1
        }

## DEFINING TECHNICAL DEFAULTS
TECHNICAL_DEFAULTS={
        'n_procs': 20, # Number of processors
        'n_frames': 50000, # 100000 - Number of rames
        'n_chunks': 5, # Number of chunks you want to create
        }

## DEFINING MAX NUMBER OF MOLECULES
MAX_N = 14 # Overestimating the maximum to recompute mu values
R_CUTOFF = 0.33 # Radius of cavity

## DEFINING MU MIN AND MAX
MU_MIN=8
MU_MAX=12
