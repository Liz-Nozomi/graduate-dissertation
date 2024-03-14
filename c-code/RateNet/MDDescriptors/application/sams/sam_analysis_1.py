"""
sam_analysis.py
this is the main script for SAM analysis

CREATED ON: 02/18/2020

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
#from MDDescriptors.application.sams.rdf import compute_interfacial_rdf
from MDDescriptors.application.sams.density import compute_density_profile
from MDDescriptors.application.sams.triplet_angle import compute_triplet_angle_distribution
from MDDescriptors.application.sams.h_bond_properties import compute_num_hbonds
from MDDescriptors.application.sams.willard_chandler import compute_wc_statistic
from MDDescriptors.application.sams.triplet_entropy import compute_triplet_entropy
from MDDescriptors.application.sams.hydration_dynamics import compute_hydration_residence_time
##############################################################################
## FUNCTIONS
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
    ## USER SPECIFIED INPUTS
    z_cutoff     = 0.3       # nm, z distance cutoff, only includes water below this value
    r_cutoff     = 0.33      # nm, radius cutoff for small cavities
    use_com      = True      # use com of water, false oxygen atom used as ref
    n_procs      = 1        # number of processors to use in parallel
    periodic     = True      # use pbc in calculations
    residue_list = [ "SOL", "HOH" ] # list of residues to include in calculations
    verbose      = True      # print out to cmd line
    print_freq   = 100       # print after N steps
    want_rewrite = False     # rewrite files
    ## GROMACS OUTPUT
    input_prefix = "sam_prod" # sys.argv[1]
    ## MAIN DIRECTORY
    main_dir = r"R:\simulations\polar_sams\unbiased\sample1"    
    ## SUB DIRECTORIES
    sub_directories = [ 
                        "sam_single_12x12_300K_dodecanethiol_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_300K_C13NH2_k0.0_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_300K_C13NH2_k0.1_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_300K_C13NH2_k0.2_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_300K_C13NH2_k0.3_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_300K_C13NH2_k0.4_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_300K_C13NH2_k0.5_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_300K_C13NH2_k0.6_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_300K_C13NH2_k0.7_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_300K_C13NH2_k0.8_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_300K_C13NH2_k0.9_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_300K_C13NH2_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_300K_C12CONH2_k0.0_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_300K_C12CONH2_k0.1_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_300K_C12CONH2_k0.2_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_300K_C12CONH2_k0.3_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_300K_C12CONH2_k0.4_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_300K_C12CONH2_k0.5_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_300K_C12CONH2_k0.6_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_300K_C12CONH2_k0.7_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_300K_C12CONH2_k0.8_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_300K_C12CONH2_k0.9_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_300K_C12CONH2_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_300K_C13OH_k0.0_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_300K_C13OH_k0.1_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_300K_C13OH_k0.2_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_300K_C13OH_k0.3_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_300K_C13OH_k0.4_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_300K_C13OH_k0.5_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_300K_C13OH_k0.6_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_300K_C13OH_k0.7_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_300K_C13OH_k0.8_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_300K_C13OH_k0.9_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_300K_C13OH_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_checker_300K_dodecanethiol0.75_C12CONH20.25_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_checker_300K_dodecanethiol0.6_C12CONH20.4_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_checker_300K_dodecanethiol0.5_C12CONH20.5_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_checker_300K_dodecanethiol0.25_C12CONH20.75_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_janus_300K_dodecanethiol0.75_C12CONH20.25_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_janus_300K_dodecanethiol0.58_C12CONH20.42_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_janus_300K_dodecanethiol0.5_C12CONH20.5_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_janus_300K_dodecanethiol0.25_C12CONH20.75_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_checker_300K_dodecanethiol0.75_C13NH20.25_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_checker_300K_dodecanethiol0.6_C13NH20.4_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_checker_300K_dodecanethiol0.5_C13NH20.5_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_checker_300K_dodecanethiol0.25_C13NH20.75_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_janus_300K_dodecanethiol0.75_C13NH20.25_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_janus_300K_dodecanethiol0.58_C13NH20.42_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_janus_300K_dodecanethiol0.5_C13NH20.5_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_janus_300K_dodecanethiol0.25_C13NH20.75_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_checker_300K_dodecanethiol0.75_C13OH0.25_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_checker_300K_dodecanethiol0.6_C13OH0.4_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_checker_300K_dodecanethiol0.5_C13OH0.5_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_checker_300K_dodecanethiol0.25_C13OH0.75_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_janus_300K_dodecanethiol0.75_C13OH0.25_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_janus_300K_dodecanethiol0.58_C13OH0.42_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_janus_300K_dodecanethiol0.5_C13OH0.5_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_janus_300K_dodecanethiol0.25_C13OH0.75_tip4p_nvt_CHARMM36",                        
                        "sam_single_12x12_checker_300K_dodecanethiol0.6_C13NH30.4_tip4p_nvt_CHARMM36",
                        "sam_single_12x12_janus_300K_dodecanethiol0.58_C13NH30.42_tip4p_nvt_CHARMM36",
                       ]    
    ## LIST OF ANALYSIS FUNCTIONS
    analysis_list = { 
#                      "interfacial_rdf"            : compute_interfacial_rdf,
#                      "density_profile"            : compute_density_profile,
#                      "triplet_angle_distribution" : compute_triplet_angle_distribution,
#                      "triplet_entropy"            : compute_triplet_entropy,
#                      "num_hbonds"                 : compute_num_hbonds,
#                      "wc_height"                  : compute_wc_statistic,
                      "hydration_residence_time"   : compute_hydration_residence_time,
                      }
    ## LOOP THROUGH DIRECTORIES
    for sd in sub_directories:
        ## PREPARE DIRECTORY FOR ANALYSIS
        path_to_sim = check_server_path( os.path.join( main_dir, sd ) )
        ## COMPUTE ANALYSIS IN ANALYSIS_LIST
        for analysis_key, function_obj in analysis_list.items():
            print( "\n--- executing %s function ---" % analysis_key )            
            ## NOTE: FUNCTION INPUT CAN PROBABLY BE IMPROVED SOMEHOW
            analysis_kwargs = { 'sim_working_dir' : path_to_sim,
                                'input_prefix'    : input_prefix,
                                'rewrite'         : want_rewrite,
                                'z_cutoff'        : z_cutoff,
                                'r_cutoff'        : r_cutoff,
                                'use_com'         : use_com,
                                'n_procs'         : n_procs,
                                'periodic'        : periodic,
                                'residue_list'    : residue_list,
                                'verbose'         : verbose,
                                'print_freq'      : print_freq }
            ## EXECUTE FUNCTIONS
            function_obj(**analysis_kwargs)
            