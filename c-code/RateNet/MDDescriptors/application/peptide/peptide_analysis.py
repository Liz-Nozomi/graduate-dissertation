# -*- coding: utf-8 -*-
"""
surfactant_analysis.py
this is the main script for surfactant analysis

CREATED ON: 02/24/2020

AUTHOR(S):
    Bradley C. Dallin (brad.dallin@gmail.com)
    
** UPDATES **

TODO:

"""
##############################################################################
## IMPORTING MODULES
##############################################################################
## IMPORT OS
import os
## MDBUILDERS FUNCTIONS
from MDBuilder.core.check_tools import check_server_path
## IMPORTING ANALYSIS FUNCTION
from MDDescriptors.application.peptide.root_mean_sq import compute_rmsd, compute_rmsf
from MDDescriptors.application.peptide.peptide_create_plots import plot_line

##############################################################################
## FUNCTIONS & CLASSES
##############################################################################
### FUNCTION TO CHECK IF YOU ARE RUNNING ON SPYDER
def check_spyder():
    ''' This function checks if you are running on spyder '''
    if any('SPYDER' in name for name in os.environ):
        return True
    else:
        return False
#%%
##############################################################################
## MAIN SCRIPT
##############################################################################
if __name__ == "__main__":
    ## REWRITE
    want_rewrite = False
    save_fig = True
    ## OUTPUT DIRECTORY
    output_dir = r"C:\Users\bdallin\Box Sync\univ_of_wisc\manuscripts\beta_peptides\figure_stability"    
    ## GROMACS OUTPUT
    input_prefix = "peptide_prod"    
    ## MAIN DIRECTORY
    main_dir = r"R:\peptides\simulations\unbiased\restrained"    
    ## SUB DIRECTORIES
    sub_directories = {
                        "GA-B3K+"  : "peptide_300K_ga_b3k_3plus_tip4p_nvt_CHARMM36",
                        "GA-B3K"   : "peptide_300K_ga_b3k_tip4p_nvt_CHARMM36",
                        "GA-KKQ+"  : "peptide_300K_ga_kkq_2plus_tip4p_nvt_CHARMM36",
                        "GA-KKQ"   : "peptide_300K_ga_kkq_tip4p_nvt_CHARMM36",
                        "GA-KQK+"  : "peptide_300K_ga_kqk_2plus_tip4p_nvt_CHARMM36",
                        "GA-KQK"   : "peptide_300K_ga_kqk_tip4p_nvt_CHARMM36",
                        "GA-KQQ+"  : "peptide_300K_ga_kqq_1plus_tip4p_nvt_CHARMM36",
                        "GA-KQQ"   : "peptide_300K_ga_kqq_tip4p_nvt_CHARMM36",
                        "GA-QKK+"  : "peptide_300K_ga_qkk_2plus_tip4p_nvt_CHARMM36",
                        "GA-QKK"   : "peptide_300K_ga_qkk_tip4p_nvt_CHARMM36",
                        "GA-QKQ+"  : "peptide_300K_ga_qkq_1plus_tip4p_nvt_CHARMM36",
                        "GA-QKQ"   : "peptide_300K_ga_qkq_tip4p_nvt_CHARMM36",
                        "GA-QQK+"  : "peptide_300K_ga_qqk_1plus_tip4p_nvt_CHARMM36",
                        "GA-QQK"   : "peptide_300K_ga_qqk_tip4p_nvt_CHARMM36",
                        "GA-B3Q"   : "peptide_300K_ga_b3q_tip4p_nvt_CHARMM36",
                        "iso-B3K+" : "peptide_300K_non_ga_b3k_3plus_tip4p_nvt_CHARMM36",
                        "iso-B3K"  : "peptide_300K_non_ga_b3k_tip4p_nvt_CHARMM36",
                        "iso-KKQ+" : "peptide_300K_non_ga_kkq_2plus_tip4p_nvt_CHARMM36",
                        "iso-KKQ"  : "peptide_300K_non_ga_kkq_tip4p_nvt_CHARMM36",
                        "iso-KQK+" : "peptide_300K_non_ga_kqk_2plus_tip4p_nvt_CHARMM36",
                        "iso-KQK"  : "peptide_300K_non_ga_kqk_tip4p_nvt_CHARMM36",
                        "iso-KQQ+" : "peptide_300K_non_ga_kqq_1plus_tip4p_nvt_CHARMM36",
                        "iso-KQQ"  : "peptide_300K_non_ga_kqq_tip4p_nvt_CHARMM36",
                        "iso-QKK+" : "peptide_300K_non_ga_qkk_2plus_tip4p_nvt_CHARMM36",
                        "iso-QKK"  : "peptide_300K_non_ga_qkk_tip4p_nvt_CHARMM36",
                        "iso-QKQ+" : "peptide_300K_non_ga_qkq_1plus_tip4p_nvt_CHARMM36",
                        "iso-QKQ"  : "peptide_300K_non_ga_qkq_tip4p_nvt_CHARMM36",
                        "iso-QQK+" : "peptide_300K_non_ga_qqk_1plus_tip4p_nvt_CHARMM36",
                        "iso-QQK"  : "peptide_300K_non_ga_qqk_tip4p_nvt_CHARMM36",
                        "iso-B3Q"  : "peptide_300K_non_ga_b3q_tip4p_nvt_CHARMM36",
                       }    
    ## LIST OF ANALYSIS FUNCTIONS
    analysis_list = { 
                      "rmsd" : compute_rmsd,
#                      "rmsf" : compute_rmsf,
                      }
    ## LOOP THROUGH DIRECTORIES
    data = {}
    for key, sd in sub_directories.items():
        ## PREPARE DIRECTORY FOR ANALYSIS
        path_to_sim = check_server_path( os.path.join( main_dir, sd ) )
        ## COMPUTE ANALYSIS IN ANALYSIS_LIST
        for analysis_key, function_obj in analysis_list.items():
            print( "\n--- executing %s function ---" % analysis_key )            
            ## NOTE: FUNCTION INPUT CAN PROBABLY BE IMPROVED SOMEHOW
            analysis_kwargs = { 'sim_working_dir'    : path_to_sim,
                                'input_prefix'       : input_prefix,
                                'add_bonds_from_itp' : False,
                                'rewrite'            : want_rewrite,
                               }
            ## EXECUTE FUNCTIONS
            data[key] = function_obj(**analysis_kwargs)

    ## PLOT RMSF FIGURES
#    path_fig = os.path.join( output_dir, r"rmsf_{:s}.png" )        
#    plot_line( path_fig, 
#               data,
#               savefig = save_fig )

    ## PLOT RMSD FIGURES
    path_fig = os.path.join( output_dir, r"rmsd_{:s}_S_restrained.png" )        
    plot_line( path_fig, 
               data,
               savefig = save_fig )
        