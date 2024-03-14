# -*- coding: utf-8 -*-
"""
sam_compile_data_to_plot.py
this script compiles data to plot

CREATED ON: 04/22/2020

AUTHOR(S):
    Bradley C. Dallin (brad.dallin@gmail.com)
    
** UPDATES **

TODO:

"""
##############################################################################
## IMPORTING MODULES
##############################################################################
import os, sys
import numpy as np
## ADD MDDescriptors TO PATH
if r"R:\bin\python_modules" not in sys.path:
    sys.path.append( r"R:\bin\python_modules" )
## FUNCTION TO SAVE AND LOAD PICKLE FILES
from MDDescriptors.application.sams.pickle_functions import load_pkl, save_pkl

##############################################################################
## FUNCTIONS
##############################################################################
## DATA COMPILING FUNCTION
def compile_data_to_plot( output_file,
                          main_dir,
                          filename,
                          paths,
                          data_type   = "y", ):
    r'''
    Function that loads data from paths and computes statistics
    '''
    ## LOOP THROUGH PLOT/LINE/BAR LABEL
    data = {}
    for plot_key, x_points in paths.items():
        x, y = [], []
        for x_point, path_samples in x_points.items():
            x.append( float(x_point) )
            y_data = []
            for y_path in path_samples:
                file_to_load = os.path.join( main_dir, y_path, filename )
                y_data.append( load_pkl( file_to_load ) )
            y.append( y_data )
        data[plot_key] = { "x": x,
                           "y": y }
    return data

#%%
##############################################################################
## MAIN SCRIPT
##############################################################################
if __name__ == "__main__":
    ## VARIABLES
    subtract_reference = False
    # input_files = [ r"output_files\sam_prod_num_hbonds_all.pkl",
    #                 r"output_files\sam_prod_num_hbonds_sam.pkl",
    #                 r"output_files\sam_prod_num_hbonds_water.pkl",
    #                 r"output_files\sam_prod_num_hbonds_sam-sam.pkl",
    #                 r"output_files\sam_prod_num_hbonds_sam-water.pkl",
    #                 r"output_files\sam_prod_num_hbonds_water-water.pkl" ]
    # output_files = [ r"num_hbonds_all_composition.pkl",
    #                  r"num_hbonds_sam_composition.pkl",
    #                  r"num_hbonds_water_composition.pkl",
    #                  r"num_hbonds_sam-sam_composition.pkl",
    #                  r"num_hbonds_sam-water_composition.pkl",
    #                  r"num_hbonds_water-water_composition.pkl" ]
    # titles = [ "All hydrogen bonds",
    #            "SAM hydrogen bonds",
    #            "Water hydrogen bonds",
    #            "SAM-SAM hydrogen bonds",
    #            "SAM-water hydrogen bonds",
    #            "Water-water hydrogen bonds" ]
    input_files = [ r"output_files\sam_prod_hydration_residence_time.pkl" ]
    output_files = [ r"hydration_residence_time_charge_scale.pkl" ]
    titles = [ None ]
    for file_to_analyze, outfile, title in zip( input_files, output_files, titles ):
        ## OUTPUT DIRECTORY
        output_dir  = r"C:\Users\bdallin\Box Sync\univ_of_wisc\manuscripts\mixed_polar_sams\raw_data"
        output_file = os.path.join( output_dir, outfile )
        ## CREATE OUTPUT DIRECTORY
        if os.path.exists(output_dir) is False:
            os.mkdir( output_dir )        
        ## MAIN DIRECTORY
        main_dir = r"R:\simulations\polar_sams\unbiased"    
        ## COMPILE DATA TO PLOT
        data_to_plot = {}
        data_to_plot["plot_type"]  = "line"
        data_to_plot["plot_title"] = title
        data_to_plot["x_label"]    = r"Scaled partial charge ($\it{k}$)"
        data_to_plot["x_ticks"]    = np.arange( 0.0, 1.0, 0.2 )
        data_to_plot["y_label"]    = r"Hydration residence time (ps)"
        data_to_plot["y_ticks"]    = np.arange( 50.0, 80.0, 10.0 )
        ## DICTIONARY CONTAINING DATA TO COMPILE
        ## STRUCTURE { PLOT/LINE/BAR LABEL: { POINT/LINE LABEL: FILEPATH } }
        data_type = "y" # xy or y
        paths_to_files = { 
                            "CH3"   : { "0.0" : [ "sample1/sam_single_12x12_300K_dodecanethiol_tip4p_nvt_CHARMM36",
                                                  "sample2/sam_single_12x12_300K_dodecanethiol_tip4p_nvt_CHARMM36",
                                                  "sample3/sam_single_12x12_300K_dodecanethiol_tip4p_nvt_CHARMM36" ],
                                        "1.0" : [ "sample1/sam_single_12x12_300K_dodecanethiol_tip4p_nvt_CHARMM36",
                                                  "sample2/sam_single_12x12_300K_dodecanethiol_tip4p_nvt_CHARMM36",
                                                  "sample3/sam_single_12x12_300K_dodecanethiol_tip4p_nvt_CHARMM36" ],
                                        },
                            "NH2"   : { "0.0" : [ "sample1/sam_single_12x12_300K_C13NH2_k0.0_tip4p_nvt_CHARMM36",
                                                  "sample2/sam_single_12x12_300K_C13NH2_k0.0_tip4p_nvt_CHARMM36",
                                                  "sample3/sam_single_12x12_300K_C13NH2_k0.0_tip4p_nvt_CHARMM36"   ],
                                        "0.1" : [ "sample1/sam_single_12x12_300K_C13NH2_k0.1_tip4p_nvt_CHARMM36",
                                                  "sample2/sam_single_12x12_300K_C13NH2_k0.1_tip4p_nvt_CHARMM36",
                                                  "sample3/sam_single_12x12_300K_C13NH2_k0.1_tip4p_nvt_CHARMM36"   ],
                                        "0.2" : [ "sample1/sam_single_12x12_300K_C13NH2_k0.2_tip4p_nvt_CHARMM36",
                                                  "sample2/sam_single_12x12_300K_C13NH2_k0.2_tip4p_nvt_CHARMM36",
                                                  "sample3/sam_single_12x12_300K_C13NH2_k0.2_tip4p_nvt_CHARMM36"   ],
                                        "0.3" : [ "sample1/sam_single_12x12_300K_C13NH2_k0.3_tip4p_nvt_CHARMM36",
                                                  "sample2/sam_single_12x12_300K_C13NH2_k0.3_tip4p_nvt_CHARMM36",
                                                  "sample3/sam_single_12x12_300K_C13NH2_k0.3_tip4p_nvt_CHARMM36"   ],
                                        "0.4" : [ "sample1/sam_single_12x12_300K_C13NH2_k0.4_tip4p_nvt_CHARMM36",
                                                  "sample2/sam_single_12x12_300K_C13NH2_k0.4_tip4p_nvt_CHARMM36",
                                                  "sample3/sam_single_12x12_300K_C13NH2_k0.4_tip4p_nvt_CHARMM36"   ],
                                        "0.5" : [ "sample1/sam_single_12x12_300K_C13NH2_k0.5_tip4p_nvt_CHARMM36",
                                                  "sample2/sam_single_12x12_300K_C13NH2_k0.5_tip4p_nvt_CHARMM36",
                                                  "sample3/sam_single_12x12_300K_C13NH2_k0.5_tip4p_nvt_CHARMM36"   ],
                                        "0.6" : [ "sample1/sam_single_12x12_300K_C13NH2_k0.6_tip4p_nvt_CHARMM36",
                                                  "sample2/sam_single_12x12_300K_C13NH2_k0.6_tip4p_nvt_CHARMM36",
                                                  "sample3/sam_single_12x12_300K_C13NH2_k0.6_tip4p_nvt_CHARMM36"   ],
                                        "0.7" : [ "sample1/sam_single_12x12_300K_C13NH2_k0.7_tip4p_nvt_CHARMM36",
                                                  "sample2/sam_single_12x12_300K_C13NH2_k0.7_tip4p_nvt_CHARMM36",
                                                  "sample3/sam_single_12x12_300K_C13NH2_k0.7_tip4p_nvt_CHARMM36"   ],
                                        "0.8" : [ "sample1/sam_single_12x12_300K_C13NH2_k0.8_tip4p_nvt_CHARMM36",
                                                  "sample2/sam_single_12x12_300K_C13NH2_k0.8_tip4p_nvt_CHARMM36",
                                                  "sample3/sam_single_12x12_300K_C13NH2_k0.8_tip4p_nvt_CHARMM36"   ],
                                        "0.9" : [ "sample1/sam_single_12x12_300K_C13NH2_k0.9_tip4p_nvt_CHARMM36",
                                                  "sample2/sam_single_12x12_300K_C13NH2_k0.9_tip4p_nvt_CHARMM36",
                                                  "sample3/sam_single_12x12_300K_C13NH2_k0.9_tip4p_nvt_CHARMM36"   ],
                                        "1.0" : [ "sample1/sam_single_12x12_300K_C13NH2_tip4p_nvt_CHARMM36",
                                                  "sample2/sam_single_12x12_300K_C13NH2_tip4p_nvt_CHARMM36",
                                                  "sample3/sam_single_12x12_300K_C13NH2_tip4p_nvt_CHARMM36"        ],
                                        },
                            "CONH2" : { "0.0" : [ "sample1/sam_single_12x12_300K_C12CONH2_k0.0_tip4p_nvt_CHARMM36",
                                                  "sample2/sam_single_12x12_300K_C12CONH2_k0.0_tip4p_nvt_CHARMM36",
                                                  "sample3/sam_single_12x12_300K_C12CONH2_k0.0_tip4p_nvt_CHARMM36" ],
                                        "0.1" : [ "sample1/sam_single_12x12_300K_C12CONH2_k0.1_tip4p_nvt_CHARMM36",
                                                  "sample2/sam_single_12x12_300K_C12CONH2_k0.1_tip4p_nvt_CHARMM36",
                                                  "sample3/sam_single_12x12_300K_C12CONH2_k0.1_tip4p_nvt_CHARMM36" ],
                                        "0.2" : [ "sample1/sam_single_12x12_300K_C12CONH2_k0.2_tip4p_nvt_CHARMM36",
                                                  "sample2/sam_single_12x12_300K_C12CONH2_k0.2_tip4p_nvt_CHARMM36",
                                                  "sample3/sam_single_12x12_300K_C12CONH2_k0.2_tip4p_nvt_CHARMM36" ],
                                        "0.3" : [ "sample1/sam_single_12x12_300K_C12CONH2_k0.3_tip4p_nvt_CHARMM36",
                                                  "sample2/sam_single_12x12_300K_C12CONH2_k0.3_tip4p_nvt_CHARMM36",
                                                  "sample3/sam_single_12x12_300K_C12CONH2_k0.3_tip4p_nvt_CHARMM36" ],
                                        "0.4" : [ "sample1/sam_single_12x12_300K_C12CONH2_k0.4_tip4p_nvt_CHARMM36",
                                                  "sample2/sam_single_12x12_300K_C12CONH2_k0.4_tip4p_nvt_CHARMM36",
                                                  "sample3/sam_single_12x12_300K_C12CONH2_k0.4_tip4p_nvt_CHARMM36" ],
                                        "0.5" : [ "sample1/sam_single_12x12_300K_C12CONH2_k0.5_tip4p_nvt_CHARMM36",
                                                  "sample2/sam_single_12x12_300K_C12CONH2_k0.5_tip4p_nvt_CHARMM36",
                                                  "sample3/sam_single_12x12_300K_C12CONH2_k0.5_tip4p_nvt_CHARMM36" ],
                                        "0.6" : [ "sample1/sam_single_12x12_300K_C12CONH2_k0.6_tip4p_nvt_CHARMM36",
                                                  "sample2/sam_single_12x12_300K_C12CONH2_k0.6_tip4p_nvt_CHARMM36",
                                                  "sample3/sam_single_12x12_300K_C12CONH2_k0.6_tip4p_nvt_CHARMM36" ],
                                        "0.7" : [ "sample1/sam_single_12x12_300K_C12CONH2_k0.7_tip4p_nvt_CHARMM36",
                                                  "sample2/sam_single_12x12_300K_C12CONH2_k0.7_tip4p_nvt_CHARMM36",
                                                  "sample3/sam_single_12x12_300K_C12CONH2_k0.7_tip4p_nvt_CHARMM36" ],
                                        "0.8" : [ "sample1/sam_single_12x12_300K_C12CONH2_k0.8_tip4p_nvt_CHARMM36",
                                                  "sample2/sam_single_12x12_300K_C12CONH2_k0.8_tip4p_nvt_CHARMM36",
                                                  "sample3/sam_single_12x12_300K_C12CONH2_k0.8_tip4p_nvt_CHARMM36" ],
                                        "0.9" : [ "sample1/sam_single_12x12_300K_C12CONH2_k0.9_tip4p_nvt_CHARMM36",
                                                  "sample2/sam_single_12x12_300K_C12CONH2_k0.9_tip4p_nvt_CHARMM36",
                                                  "sample3/sam_single_12x12_300K_C12CONH2_k0.9_tip4p_nvt_CHARMM36" ],
                                        "1.0" : [ "sample1/sam_single_12x12_300K_C12CONH2_tip4p_nvt_CHARMM36",
                                                  "sample2/sam_single_12x12_300K_C12CONH2_tip4p_nvt_CHARMM36",
                                                  "sample3/sam_single_12x12_300K_C12CONH2_tip4p_nvt_CHARMM36"      ],
                                        },
                            "OH"    : { "0.0" : [ "sample1/sam_single_12x12_300K_C13OH_k0.0_tip4p_nvt_CHARMM36",
                                                  "sample2/sam_single_12x12_300K_C13OH_k0.0_tip4p_nvt_CHARMM36",
                                                  "sample3/sam_single_12x12_300K_C13OH_k0.0_tip4p_nvt_CHARMM36"    ],
                                        "0.1" : [ "sample1/sam_single_12x12_300K_C13OH_k0.1_tip4p_nvt_CHARMM36",
                                                  "sample2/sam_single_12x12_300K_C13OH_k0.1_tip4p_nvt_CHARMM36",
                                                  "sample3/sam_single_12x12_300K_C13OH_k0.1_tip4p_nvt_CHARMM36"    ],
                                        "0.2" : [ "sample1/sam_single_12x12_300K_C13OH_k0.2_tip4p_nvt_CHARMM36",
                                                  "sample2/sam_single_12x12_300K_C13OH_k0.2_tip4p_nvt_CHARMM36",
                                                  "sample3/sam_single_12x12_300K_C13OH_k0.2_tip4p_nvt_CHARMM36"    ],
                                        "0.3" : [ "sample1/sam_single_12x12_300K_C13OH_k0.3_tip4p_nvt_CHARMM36",
                                                  "sample2/sam_single_12x12_300K_C13OH_k0.3_tip4p_nvt_CHARMM36",
                                                  "sample3/sam_single_12x12_300K_C13OH_k0.3_tip4p_nvt_CHARMM36"    ],
                                        "0.4" : [ "sample1/sam_single_12x12_300K_C13OH_k0.4_tip4p_nvt_CHARMM36",
                                                  "sample2/sam_single_12x12_300K_C13OH_k0.4_tip4p_nvt_CHARMM36",
                                                  "sample3/sam_single_12x12_300K_C13OH_k0.4_tip4p_nvt_CHARMM36"    ],
                                        "0.5" : [ "sample1/sam_single_12x12_300K_C13OH_k0.5_tip4p_nvt_CHARMM36",
                                                  "sample2/sam_single_12x12_300K_C13OH_k0.5_tip4p_nvt_CHARMM36",
                                                  "sample3/sam_single_12x12_300K_C13OH_k0.5_tip4p_nvt_CHARMM36"    ],
                                        "0.6" : [ "sample1/sam_single_12x12_300K_C13OH_k0.6_tip4p_nvt_CHARMM36",
                                                  "sample2/sam_single_12x12_300K_C13OH_k0.6_tip4p_nvt_CHARMM36",
                                                  "sample3/sam_single_12x12_300K_C13OH_k0.6_tip4p_nvt_CHARMM36"    ],
                                        "0.7" : [ "sample1/sam_single_12x12_300K_C13OH_k0.7_tip4p_nvt_CHARMM36",
                                                  "sample2/sam_single_12x12_300K_C13OH_k0.7_tip4p_nvt_CHARMM36",
                                                  "sample3/sam_single_12x12_300K_C13OH_k0.7_tip4p_nvt_CHARMM36"    ],
                                        "0.8" : [ "sample1/sam_single_12x12_300K_C13OH_k0.8_tip4p_nvt_CHARMM36",
                                                  "sample2/sam_single_12x12_300K_C13OH_k0.8_tip4p_nvt_CHARMM36",
                                                  "sample3/sam_single_12x12_300K_C13OH_k0.8_tip4p_nvt_CHARMM36"    ],
                                        "0.9" : [ "sample1/sam_single_12x12_300K_C13OH_k0.9_tip4p_nvt_CHARMM36",
                                                  "sample2/sam_single_12x12_300K_C13OH_k0.9_tip4p_nvt_CHARMM36",
                                                  "sample3/sam_single_12x12_300K_C13OH_k0.9_tip4p_nvt_CHARMM36"    ],
                                        "1.0" : [ "sample1/sam_single_12x12_300K_C13OH_tip4p_nvt_CHARMM36",
                                                  "sample2/sam_single_12x12_300K_C13OH_tip4p_nvt_CHARMM36",
                                                  "sample3/sam_single_12x12_300K_C13OH_tip4p_nvt_CHARMM36"         ],
                                        },
                            # "NH2-mix"   : { "0.0"  : [ "sample1/sam_single_12x12_300K_dodecanethiol_tip4p_nvt_CHARMM36",
                            #                             "sample2/sam_single_12x12_300K_dodecanethiol_tip4p_nvt_CHARMM36",
                            #                             "sample3/sam_single_12x12_300K_dodecanethiol_tip4p_nvt_CHARMM36",                          ],
                            #                 "0.25" : [ "sample1/sam_single_12x12_checker_300K_dodecanethiol0.75_C13NH20.25_tip4p_nvt_CHARMM36",
                            #                             "sample2/sam_single_12x12_checker_300K_dodecanethiol0.75_C13NH20.25_tip4p_nvt_CHARMM36",
                            #                             "sample3/sam_single_12x12_checker_300K_dodecanethiol0.75_C13NH20.25_tip4p_nvt_CHARMM36",   ],
                            #                 "0.4"  : [ "sample1/sam_single_12x12_checker_300K_dodecanethiol0.6_C13NH20.4_tip4p_nvt_CHARMM36",
                            #                             "sample2/sam_single_12x12_checker_300K_dodecanethiol0.6_C13NH20.4_tip4p_nvt_CHARMM36",
                            #                             "sample3/sam_single_12x12_checker_300K_dodecanethiol0.6_C13NH20.4_tip4p_nvt_CHARMM36",     ],
                            #                 "0.5"  : [ "sample1/sam_single_12x12_checker_300K_dodecanethiol0.5_C13NH20.5_tip4p_nvt_CHARMM36",
                            #                             "sample2/sam_single_12x12_checker_300K_dodecanethiol0.5_C13NH20.5_tip4p_nvt_CHARMM36",
                            #                             "sample3/sam_single_12x12_checker_300K_dodecanethiol0.5_C13NH20.5_tip4p_nvt_CHARMM36",     ],
                            #                 "0.75" : [ "sample1/sam_single_12x12_checker_300K_dodecanethiol0.25_C13NH20.75_tip4p_nvt_CHARMM36",   
                            #                             "sample2/sam_single_12x12_checker_300K_dodecanethiol0.25_C13NH20.75_tip4p_nvt_CHARMM36",   
                            #                             "sample3/sam_single_12x12_checker_300K_dodecanethiol0.25_C13NH20.75_tip4p_nvt_CHARMM36",   ],
                            #                 "1.0"  : [ "sample1/sam_single_12x12_300K_C13NH2_tip4p_nvt_CHARMM36",                                 
                            #                             "sample2/sam_single_12x12_300K_C13NH2_tip4p_nvt_CHARMM36",                                 
                            #                             "sample3/sam_single_12x12_300K_C13NH2_tip4p_nvt_CHARMM36",                                 ],
                            #                 },
                            # "CONH2-mix" : { "0.0"  : [ "sample1/sam_single_12x12_300K_dodecanethiol_tip4p_nvt_CHARMM36",
                            #                             "sample2/sam_single_12x12_300K_dodecanethiol_tip4p_nvt_CHARMM36",
                            #                             "sample3/sam_single_12x12_300K_dodecanethiol_tip4p_nvt_CHARMM36",                          ],
                            #                 "0.25" : [ "sample1/sam_single_12x12_checker_300K_dodecanethiol0.75_C12CONH20.25_tip4p_nvt_CHARMM36",
                            #                             "sample2/sam_single_12x12_checker_300K_dodecanethiol0.75_C12CONH20.25_tip4p_nvt_CHARMM36",
                            #                             "sample3/sam_single_12x12_checker_300K_dodecanethiol0.75_C12CONH20.25_tip4p_nvt_CHARMM36", ],
                            #                 "0.4"  : [ "sample1/sam_single_12x12_checker_300K_dodecanethiol0.6_C12CONH20.4_tip4p_nvt_CHARMM36",
                            #                             "sample2/sam_single_12x12_checker_300K_dodecanethiol0.6_C12CONH20.4_tip4p_nvt_CHARMM36",
                            #                             "sample3/sam_single_12x12_checker_300K_dodecanethiol0.6_C12CONH20.4_tip4p_nvt_CHARMM36",   ],
                            #                 "0.5"  : [ "sample1/sam_single_12x12_checker_300K_dodecanethiol0.5_C12CONH20.5_tip4p_nvt_CHARMM36",
                            #                             "sample2/sam_single_12x12_checker_300K_dodecanethiol0.5_C12CONH20.5_tip4p_nvt_CHARMM36",
                            #                             "sample3/sam_single_12x12_checker_300K_dodecanethiol0.5_C12CONH20.5_tip4p_nvt_CHARMM36",   ],
                            #                 "0.75" : [ "sample1/sam_single_12x12_checker_300K_dodecanethiol0.25_C12CONH20.75_tip4p_nvt_CHARMM36",   
                            #                             "sample2/sam_single_12x12_checker_300K_dodecanethiol0.25_C12CONH20.75_tip4p_nvt_CHARMM36",   
                            #                             "sample3/sam_single_12x12_checker_300K_dodecanethiol0.25_C12CONH20.75_tip4p_nvt_CHARMM36", ],
                            #                 "1.0"  : [ "sample1/sam_single_12x12_300K_C12CONH2_tip4p_nvt_CHARMM36",                                 
                            #                             "sample2/sam_single_12x12_300K_C12CONH2_tip4p_nvt_CHARMM36",                                 
                            #                             "sample3/sam_single_12x12_300K_C12CONH2_tip4p_nvt_CHARMM36",                               ],
                            #                 },
                            # "OH-mix"    : { "0.0"  : [ "sample1/sam_single_12x12_300K_dodecanethiol_tip4p_nvt_CHARMM36",
                            #                             "sample2/sam_single_12x12_300K_dodecanethiol_tip4p_nvt_CHARMM36",
                            #                             "sample3/sam_single_12x12_300K_dodecanethiol_tip4p_nvt_CHARMM36",                          ],
                            #                 "0.25" : [ "sample1/sam_single_12x12_checker_300K_dodecanethiol0.75_C13OH0.25_tip4p_nvt_CHARMM36",
                            #                             "sample2/sam_single_12x12_checker_300K_dodecanethiol0.75_C13OH0.25_tip4p_nvt_CHARMM36",
                            #                             "sample3/sam_single_12x12_checker_300K_dodecanethiol0.75_C13OH0.25_tip4p_nvt_CHARMM36",    ],
                            #                 "0.4"  : [ "sample1/sam_single_12x12_checker_300K_dodecanethiol0.6_C13OH0.4_tip4p_nvt_CHARMM36",
                            #                             "sample2/sam_single_12x12_checker_300K_dodecanethiol0.6_C13OH0.4_tip4p_nvt_CHARMM36",
                            #                             "sample3/sam_single_12x12_checker_300K_dodecanethiol0.6_C13OH0.4_tip4p_nvt_CHARMM36",      ],
                            #                 "0.5"  : [ "sample1/sam_single_12x12_checker_300K_dodecanethiol0.5_C13OH0.5_tip4p_nvt_CHARMM36",
                            #                             "sample2/sam_single_12x12_checker_300K_dodecanethiol0.5_C13OH0.5_tip4p_nvt_CHARMM36",
                            #                             "sample3/sam_single_12x12_checker_300K_dodecanethiol0.5_C13OH0.5_tip4p_nvt_CHARMM36",      ],
                            #                 "0.75" : [ "sample1/sam_single_12x12_checker_300K_dodecanethiol0.25_C13OH0.75_tip4p_nvt_CHARMM36",    
                            #                             "sample2/sam_single_12x12_checker_300K_dodecanethiol0.25_C13OH0.75_tip4p_nvt_CHARMM36",   
                            #                             "sample3/sam_single_12x12_checker_300K_dodecanethiol0.25_C13OH0.75_tip4p_nvt_CHARMM36",    ],
                            #                 "1.0"  : [ "sample1/sam_single_12x12_300K_C13OH_tip4p_nvt_CHARMM36",                                 
                            #                             "sample2/sam_single_12x12_300K_C13OH_tip4p_nvt_CHARMM36",                                 
                            #                             "sample3/sam_single_12x12_300K_C13OH_tip4p_nvt_CHARMM36",                                  ],
                            #                 },
                            # "NH2-sep"   : { "0.0"  : [ "sample1/sam_single_12x12_300K_dodecanethiol_tip4p_nvt_CHARMM36",
                            #                             "sample2/sam_single_12x12_300K_dodecanethiol_tip4p_nvt_CHARMM36",
                            #                             "sample3/sam_single_12x12_300K_dodecanethiol_tip4p_nvt_CHARMM36",                          ],
                            #                 "0.25" : [ "sample1/sam_single_12x12_janus_300K_dodecanethiol0.75_C13NH20.25_tip4p_nvt_CHARMM36",
                            #                             "sample2/sam_single_12x12_janus_300K_dodecanethiol0.75_C13NH20.25_tip4p_nvt_CHARMM36",
                            #                             "sample3/sam_single_12x12_janus_300K_dodecanethiol0.75_C13NH20.25_tip4p_nvt_CHARMM36",     ],
                            #                 "0.4"  : [ "sample1/sam_single_12x12_janus_300K_dodecanethiol0.58_C13NH20.42_tip4p_nvt_CHARMM36",
                            #                             "sample2/sam_single_12x12_janus_300K_dodecanethiol0.58_C13NH20.42_tip4p_nvt_CHARMM36",
                            #                             "sample3/sam_single_12x12_janus_300K_dodecanethiol0.58_C13NH20.42_tip4p_nvt_CHARMM36",     ],
                            #                 "0.5"  : [ "sample1/sam_single_12x12_janus_300K_dodecanethiol0.5_C13NH20.5_tip4p_nvt_CHARMM36",
                            #                             "sample2/sam_single_12x12_janus_300K_dodecanethiol0.5_C13NH20.5_tip4p_nvt_CHARMM36",
                            #                             "sample3/sam_single_12x12_janus_300K_dodecanethiol0.5_C13NH20.5_tip4p_nvt_CHARMM36",       ],
                            #                 "0.75" : [ "sample1/sam_single_12x12_janus_300K_dodecanethiol0.25_C13NH20.75_tip4p_nvt_CHARMM36",   
                            #                             "sample2/sam_single_12x12_janus_300K_dodecanethiol0.25_C13NH20.75_tip4p_nvt_CHARMM36",    
                            #                             "sample3/sam_single_12x12_janus_300K_dodecanethiol0.25_C13NH20.75_tip4p_nvt_CHARMM36",     ],
                            #                 "1.0"  : [ "sample1/sam_single_12x12_300K_C13NH2_tip4p_nvt_CHARMM36",                                  
                            #                             "sample2/sam_single_12x12_300K_C13NH2_tip4p_nvt_CHARMM36",                                 
                            #                             "sample3/sam_single_12x12_300K_C13NH2_tip4p_nvt_CHARMM36",                                 ],
                            #                 },
                            # "CONH2-sep" : { "0.0"  : [ "sample1/sam_single_12x12_300K_dodecanethiol_tip4p_nvt_CHARMM36",
                            #                             "sample2/sam_single_12x12_300K_dodecanethiol_tip4p_nvt_CHARMM36",  
                            #                             "sample3/sam_single_12x12_300K_dodecanethiol_tip4p_nvt_CHARMM36",                          ],
                            #                 "0.25" : [ "sample1/sam_single_12x12_janus_300K_dodecanethiol0.75_C12CONH20.25_tip4p_nvt_CHARMM36",
                            #                             "sample2/sam_single_12x12_janus_300K_dodecanethiol0.75_C12CONH20.25_tip4p_nvt_CHARMM36",
                            #                             "sample3/sam_single_12x12_janus_300K_dodecanethiol0.75_C12CONH20.25_tip4p_nvt_CHARMM36",   ],
                            #                 "0.4"  : [ "sample1/sam_single_12x12_janus_300K_dodecanethiol0.58_C12CONH20.42_tip4p_nvt_CHARMM36",
                            #                             "sample2/sam_single_12x12_janus_300K_dodecanethiol0.58_C12CONH20.42_tip4p_nvt_CHARMM36",
                            #                             "sample3/sam_single_12x12_janus_300K_dodecanethiol0.58_C12CONH20.42_tip4p_nvt_CHARMM36",   ],
                            #                 "0.5"  : [ "sample1/sam_single_12x12_janus_300K_dodecanethiol0.5_C12CONH20.5_tip4p_nvt_CHARMM36",
                            #                             "sample2/sam_single_12x12_janus_300K_dodecanethiol0.5_C12CONH20.5_tip4p_nvt_CHARMM36",  
                            #                             "sample3/sam_single_12x12_janus_300K_dodecanethiol0.5_C12CONH20.5_tip4p_nvt_CHARMM36",     ],
                            #                 "0.75" : [ "sample1/sam_single_12x12_janus_300K_dodecanethiol0.25_C12CONH20.75_tip4p_nvt_CHARMM36",   
                            #                             "sample2/sam_single_12x12_janus_300K_dodecanethiol0.25_C12CONH20.75_tip4p_nvt_CHARMM36",   
                            #                             "sample3/sam_single_12x12_janus_300K_dodecanethiol0.25_C12CONH20.75_tip4p_nvt_CHARMM36",   ],
                            #                 "1.0"  : [ "sample1/sam_single_12x12_300K_C12CONH2_tip4p_nvt_CHARMM36",                                 
                            #                             "sample2/sam_single_12x12_300K_C12CONH2_tip4p_nvt_CHARMM36",                                 
                            #                             "sample3/sam_single_12x12_300K_C12CONH2_tip4p_nvt_CHARMM36",                               ],
                            #                 },
                            # "OH-sep"    : { "0.0"  : [ "sample1/sam_single_12x12_300K_dodecanethiol_tip4p_nvt_CHARMM36",
                            #                             "sample2/sam_single_12x12_300K_dodecanethiol_tip4p_nvt_CHARMM36",
                            #                             "sample3/sam_single_12x12_300K_dodecanethiol_tip4p_nvt_CHARMM36",                          ],
                            #                 "0.25" : [ "sample1/sam_single_12x12_janus_300K_dodecanethiol0.75_C13OH0.25_tip4p_nvt_CHARMM36",
                            #                             "sample2/sam_single_12x12_janus_300K_dodecanethiol0.75_C13OH0.25_tip4p_nvt_CHARMM36",
                            #                             "sample3/sam_single_12x12_janus_300K_dodecanethiol0.75_C13OH0.25_tip4p_nvt_CHARMM36",      ],
                            #                 "0.4"  : [ "sample1/sam_single_12x12_janus_300K_dodecanethiol0.58_C13OH0.42_tip4p_nvt_CHARMM36",
                            #                             "sample2/sam_single_12x12_janus_300K_dodecanethiol0.58_C13OH0.42_tip4p_nvt_CHARMM36",
                            #                             "sample3/sam_single_12x12_janus_300K_dodecanethiol0.58_C13OH0.42_tip4p_nvt_CHARMM36",      ],
                            #                 "0.5"  : [ "sample1/sam_single_12x12_janus_300K_dodecanethiol0.5_C13OH0.5_tip4p_nvt_CHARMM36",
                            #                             "sample2/sam_single_12x12_janus_300K_dodecanethiol0.5_C13OH0.5_tip4p_nvt_CHARMM36",
                            #                             "sample3/sam_single_12x12_janus_300K_dodecanethiol0.5_C13OH0.5_tip4p_nvt_CHARMM36",        ],
                            #                 "0.75" : [ "sample1/sam_single_12x12_janus_300K_dodecanethiol0.25_C13OH0.75_tip4p_nvt_CHARMM36",   
                            #                             "sample2/sam_single_12x12_janus_300K_dodecanethiol0.25_C13OH0.75_tip4p_nvt_CHARMM36",   
                            #                             "sample3/sam_single_12x12_janus_300K_dodecanethiol0.25_C13OH0.75_tip4p_nvt_CHARMM36",      ],
                            #                 "1.0"  : [ "sample1/sam_single_12x12_300K_C13OH_tip4p_nvt_CHARMM36",                                 
                            #                             "sample2/sam_single_12x12_300K_C13OH_tip4p_nvt_CHARMM36",                                 
                            #                             "sample3/sam_single_12x12_300K_C13OH_tip4p_nvt_CHARMM36",                                  ],
                            #                 },
                          }
        ## COMBINE DATA FROM PATHS TO SINGLE FILE
        data = compile_data_to_plot( output_file = output_file,
                                     main_dir    = main_dir,
                                     filename    = file_to_analyze,
                                     paths       = paths_to_files,
                                     data_type   = data_type,                                 
                                    )
        ## ADD DATA TO DICT
        data_to_plot["data"] = data
        ## SAVE DATA TO PKL FILE
        save_pkl( data_to_plot, output_file )
