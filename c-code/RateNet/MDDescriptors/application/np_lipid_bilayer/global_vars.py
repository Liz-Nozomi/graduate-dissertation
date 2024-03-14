# -*- coding: utf-8 -*-
"""
global_vars.py
This contains global variables for the nplm project

Variables:
    NPLM_SIM_DICT:
        simulation dictionary for nplm sims
    IMAGE_LOC:
        image location to store all images
FUNCTIONS:
    find_us_sampling_key:
        function to find umbrella sampling kets

Written by: Alex K. Chew (02/28/2020)
"""
## IMPORTING CHECK TOOLS
import os
import MDDescriptors.core.check_tools as check_tools
import glob

## DEFINING COLORS
GROUP_COLOR_DICT={
    'GOLD':  'gold',
    'RGRP': 'black',
    'ALK': 'gray', # 'tab:pink',
    'PEG': 'green',
    'NGRP': 'purple',
    'NGRP_RGRP': 'red',
    'C1': 'red',
    'C10': 'black',
    }

## DEFINING SIM DICT
NPLM_SIM_DICT={
        ## UMBRELLA SAMPLING FOR PULLING FROM BULK TO LM
        'us_forward_R12':
            {'main_sim_dir': r"US_Z_PULLING_SIMS",
             'specific_sim': r"US-1.3_5_0.2-NPLM-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1"},
        'us_forward_R01':
            {# 'main_sim_dir': r"20200120-US-sims_NPLM_rerun_stampede",
             'main_sim_dir': r"US_Z_PULLING_SIMS",
             'specific_sim': r"US-1.3_5_0.2-NPLM-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT001_1"},
        
        ## FOR BENZENE
        'us_forward_R17':
            {
             'main_sim_dir': r"20200613-US_otherligs_z-com",
             'specific_sim': r"US-NPLM-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT017_1"},
        
        ## UMBRELLA SAMPLING FOR REVERSE (PULLING FROM LM)
        'us_reverse_R12':
            {'main_sim_dir': r"US_Z_PUSHING_SIMS",
             'specific_sim': r"USrev-1.5_5_0.2-pullnplm-1.300_5.000-0.0005_2000-DOPC_196-EAM_2_ROT012_1"},
        'us_reverse_R01':
            {'main_sim_dir': r"US_Z_PUSHING_SIMS",
                # r"20200207-US-sims_NPLM_reverse_ROT001_aci",
             'specific_sim': r"USrev-1.5_5_0.2-pullnplm-1.300_5.000-0.0005_2000-DOPC_196-EAM_2_ROT001_1"},
        
        ## PUSHING NP INTO LM
        'pushing_to_lm_R12':
            {'main_sim_dir': r"20200123-Pulling_sims_from_US",
             'specific_sim': r"pullnplm-1.300_5.000-0.0005_2000-DOPC_196-EAM_2_ROT012_1"},
        'pushing_to_lm_R01':
            {'main_sim_dir': r"20200123-Pulling_sims_from_US",
             'specific_sim': r"pullnplm-1.300_5.000-0.0005_2000-DOPC_196-EAM_2_ROT001_1"},
        
        ## PULLING THE NP FROM LIPID BILAYER
        'pulling_from_lm_R12':
            {'main_sim_dir': r"20200113-NPLM_PULLING",
             'specific_sim': r"NPLM-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1"},
        'pulling_from_lm_R01':
            {'main_sim_dir': r"20200113-NPLM_PULLING",
             'specific_sim': r"NPLM-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT001_1"},
             
        ## UNBIASED SIMULATIONS FOR R12
        'unbiased_ROT012_1.300':
            {'main_sim_dir': r"20200128-unbiased_ROT012",
             'specific_sim': r"NPLM_unb-1.300_2000-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1"},
        'unbiased_ROT012_1.900':
            {'main_sim_dir': r"20200128-unbiased_ROT012",
             'specific_sim': r"NPLM_unb-1.900_2000-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1"},
        'unbiased_ROT012_2.100':
            {'main_sim_dir': r"20200128-unbiased_ROT012",
             'specific_sim': r"NPLM_unb-2.100_2000-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1"},

        'unbiased_ROT012_3.500':
            {'main_sim_dir': r"20200414-Unbiased_at_3.500nm_C1_C10",
             'specific_sim': r"NPLM_unb-3.500_2000-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1"},

        ## UNBIASED FOR ROT001
        'unbiased_ROT001_1.700':
            {'main_sim_dir': r"20200205-unbiased_ROT001",
             'specific_sim': r"NPLM_unb-1.700_2000-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT001_1"},
        'unbiased_ROT001_1.900':
            {'main_sim_dir': r"20200205-unbiased_ROT001",
             'specific_sim': r"NPLM_unb-1.900_2000-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT001_1"},
        'unbiased_ROT001_2.100':
            {'main_sim_dir': r"20200205-unbiased_ROT001",
             'specific_sim': r"NPLM_unb-2.100_2000-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT001_1"},
        'unbiased_ROT001_2.300':
            {'main_sim_dir': r"20200205-unbiased_ROT001",
             'specific_sim': r"NPLM_unb-2.300_2000-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT001_1"},
        
        'unbiased_ROT001_3.500':
            {'main_sim_dir': r"20200414-Unbiased_at_3.500nm_C1_C10",
             'specific_sim': r"NPLM_unb-3.500_2000-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT001_1"},
        
        
        ## REVERSE SIMS FOR ROT001
        'unbiased_ROT001_4.700_rev':
            {'main_sim_dir': r"20200407-unbiased_ROT001_rev",
             'specific_sim': r"NPLM_rev_unb-4.900_2000-1.5_5_0.2-pullnplm-1.300_5.000-0.0005_2000-DOPC_196-EAM_2_ROT001_1"},
        
        
        ## UNBIASED SIMULATIONS AT EQUILIBRIUM
        'unbiased_ROT012_5.300':
            {'main_sim_dir': r"20200330-unbiased_C12_at_minimum",
             'specific_sim': r"NPLM_unb-5.300_2000-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1"},
        'unbiased_ROT012_5.300_rev':
            {'main_sim_dir': r"20200330-unbiased_C12_at_minimum",
             'specific_sim': r"NPLM_rev_unb-5.300_2000-1.5_5_0.2-pullnplm-1.300_5.000-0.0005_2000-DOPC_196-EAM_2_ROT012_1"}, 
             
        ## PULLING, THEN UNBIASED
        'pullthenunbias_ROT001':
            {'main_sim_dir': r"20200423-pulling_unbiased_full",
             'specific_sim': r"NPLMpulling2unb-4.900_2000-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT001_1"},
        'pullthenunbias_ROT012':
            {'main_sim_dir': r"20200423-pulling_unbiased_full",
             'specific_sim': r'NPLMpulling2unb-5.100_2000-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1_extend_equil'},
             
        'pullthenunbias_ROT004':
            {'main_sim_dir': r"20200423-pulling_unbiased_full",
             'specific_sim': r'NPLMpulling2unb-5.100_2000-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT004_1'},
             # 'specific_sim': r"NPLMpulling2unb-5.100_2000-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1"},             
             
        ## PLUMED PULLING SIMS
        'plumed_pulling_ROT012':
            {'main_sim_dir': r"20200517-plumed_test_pulling_new_params_debugging_stride1000_6ns",
             'specific_sim': r"NPLMplumedcontactspulling-5.100_2_50_1000_0.35-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1",
             'prefix': 'nplm_pulling'},
        'plumed_pulling_ROT001':
            {'main_sim_dir': r"20200517-plumed_test_pulling_new_params_debugging_stride1000_6ns",
             'specific_sim': r"NPLMplumedcontactspulling-5.100_2_50_1000_0.35-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT001_1",
             'prefix': 'nplm_pulling'},
        
        
        ## UMBRELLA SAMPLING FOR PLUMED
        'us-PLUMED_ROT012':
            {
            'main_sim_dir': r"20200522-plumed_US_initial",
            'specific_sim': r"US_1-NPLMplumedcontactspulling-5.100_2_50_1000_0.35-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1",
            'reference_for_index': {'sim_key': 'plumed_pulling_ROT012', # Store reference information for indexing information
                                    'prefix': 'nplm'},
                    },
        'us-PLUMED_ROT001':
            {
            'main_sim_dir': r"20200522-plumed_US_initial",
            'specific_sim': r"US_1-NPLMplumedcontactspulling-5.100_2_50_1000_0.35-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT001_1",
            'reference_for_index': {'sim_key': 'plumed_pulling_ROT001', # Store reference information for indexing information
                                    'prefix': 'nplm'},
                    },
                                    
        ## PLUMED PULLING WITH HYDROPHOBIC CONTACTS
        'plumed_hydrophobic_pulling_ROT012':
            {'main_sim_dir': r"20200608-Pulling_with_hydrophobic_contacts",
             'specific_sim': r"NPLMplumedhydrophobiccontactspulling-5.100_2_50_1000_0.35-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1",
             'prefix': 'nplm_pulling',
                    },
        ## FOR C1
        'plumed_hydrophobic_pulling_ROT001':
            {'main_sim_dir': r"20200608-Pulling_with_hydrophobic_contacts",
             'specific_sim': r"NPLMplumedhydrophobiccontactspulling-5.100_2_50_1000_0.35-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT001_1",
             'prefix': 'nplm_pulling',
                    },
             
        ## UNBIASED AFTER PLUMED
        'plumed_unbiased_after_US_ROT012':
            {'main_sim_dir': r"20200619-Unbiased_after_plumed_US",
             'specific_sim': r"NPLM_unb-130-UShydrophobic_10-NPLMplumedhydrophobiccontactspulling-5.100_2_50_1000_0.35-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1",
             'prefix': 'nplm_prod',
                    },
             
        'plumed_unbiased_after_US_ROT001':
            {'main_sim_dir': r"20200619-Unbiased_after_plumed_US",
             'specific_sim': r"NPLM_unb-130-UShydrophobic_10-NPLMplumedhydrophobiccontactspulling-5.100_2_50_1000_0.35-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT001_1",
             'prefix': 'nplm_prod',
                    },
             
        ## UNBIASED AFTER PLUMED US SIMS, 20 NS EXTRACTIONS
        'plumed_unbiased_20ns_US_ROT012_40':
            {'main_sim_dir': r"20200713-Unbiased_after_plumed_US_20ns",
             'specific_sim': r"NPLM_unb-40.0-20000.000_ps-UShydrophobic_10-NPLMplumedhydrophobiccontactspulling-5.100_2_50_1000_0.35-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1",
             'prefix': 'nplm_prod',
                    },  
             
        'plumed_unbiased_20ns_US_ROT017_120':
            {'main_sim_dir': r"20200713-Unbiased_after_plumed_US_20ns",
             'specific_sim': r"NPLM_unb-120.0-20000.000_ps-UShydrophobic_10-NPLMplumedhydrophobiccontactspulling-5.100_2_50_300_0.35-50000_ns-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT017_1",
             'prefix': 'nplm_prod',
                    },
             
        'plumed_unbiased_20ns_US_ROT017_40':
            {'main_sim_dir': r"20200713-Unbiased_after_plumed_US_20ns",
             'specific_sim': r"NPLM_unb-40.0-20000.000_ps-UShydrophobic_10-NPLMplumedhydrophobiccontactspulling-5.100_2_50_300_0.35-50000_ns-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT017_1",
             'prefix': 'nplm_prod',
                    },
             
             
        ## TRIALS FOR ROT012
        'plumed_unbiased_15ns_US_ROT012_40':
            {'main_sim_dir': r"20200713-Unbiased_after_plumed_US_20ns",
             'specific_sim': r"NPLM_unb-40.0-15000.000_ps-UShydrophobic_10-NPLMplumedhydrophobiccontactspulling-5.100_2_50_1000_0.35-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1",
             'prefix': 'nplm_prod',
                    },  
             
        'plumed_unbiased_10ns_US_ROT012_40':
            {'main_sim_dir': r"20200713-Unbiased_after_plumed_US_20ns",
             'specific_sim': r"NPLM_unb-40.0-10000.000_ps-UShydrophobic_10-NPLMplumedhydrophobiccontactspulling-5.100_2_50_1000_0.35-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1",
             'prefix': 'nplm_prod',
                    },  
             
        ## ADDITIONAL PLUMED UNBIASED
        'plumed_unbiased_20ns_US_ROT001_40':
            {'main_sim_dir': r"20200713-Unbiased_after_plumed_US_20ns",
             'specific_sim': r"NPLM_unb-40.0-20000.000_ps-UShydrophobic_10-NPLMplumedhydrophobiccontactspulling-5.100_2_50_1000_0.35-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT001_1",
             'prefix': 'nplm_prod',
                    },  
        'plumed_unbiased_15ns_US_ROT001_40':
            {'main_sim_dir': r"20200713-Unbiased_after_plumed_US_20ns",
             'specific_sim': r"NPLM_unb-40.0-15000.000_ps-UShydrophobic_10-NPLMplumedhydrophobiccontactspulling-5.100_2_50_1000_0.35-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT001_1",
             'prefix': 'nplm_prod',
                    },  
        'plumed_unbiased_10ns_US_ROT001_40':
            {'main_sim_dir': r"20200713-Unbiased_after_plumed_US_20ns",
             'specific_sim': r"NPLM_unb-40.0-10000.000_ps-UShydrophobic_10-NPLMplumedhydrophobiccontactspulling-5.100_2_50_1000_0.35-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT001_1",
             'prefix': 'nplm_prod',
                    },  
             

        ### UNBIASED AFTER US
        'unbiased_after_us_ROT017_1.900nm':
            {'main_sim_dir': r"20200720-unbiased_after_US_1.9_Benzene",
             'specific_sim': r"NPLM_unb-1.900_2000-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT017_1",
             'prefix': 'nplm_prod',
             }, 
             
        ## DEFINING US SIM
        'modified_FF_us_forward_R17':
            {'main_sim_dir': r"20200818-Bn_US_modifiedFF",
             'specific_sim': r"US-NPLM-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT017_1"},
             
             
        ### UNBIASED WITH MODIFIED FF
        'modified_FF_unbiased_after_us_ROT017_1.900nm':
            {'main_sim_dir': r"20200818-Bn_US_modifiedFF_UNBIASED_AFTER_US",
             'specific_sim': r"NPLM_unb-1.900_2000-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT017_1",
             'prefix': 'nplm_prod',
             }, 
             
        
        ### UNBIASED AFTER PMF - HYDROPHOBIC CONTACTS FOR Bn
        'modified_FF_hydrophobic_pmf_unbiased_ROT017_150_10000':
            {'main_sim_dir': r"20200822-Hydrophobic_contacts_PMF_Bn_new_FF_UNBIASED_SIMS",
             'specific_sim': r"NPLM_unb-150.0-10000.000_ps-UShydrophobic_10-NPLMplumedhydrophobiccontactspulling-5.100_2_50_300_0.35-50000_ns-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT017_1",
             'prefix': 'nplm_prod',
                    },
             
        'modified_FF_hydrophobic_pmf_unbiased_ROT017_150_15000':
            {'main_sim_dir': r"20200822-Hydrophobic_contacts_PMF_Bn_new_FF_UNBIASED_SIMS",
             'specific_sim': r"NPLM_unb-150.0-15000.000_ps-UShydrophobic_10-NPLMplumedhydrophobiccontactspulling-5.100_2_50_300_0.35-50000_ns-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT017_1",
             'prefix': 'nplm_prod',
                    },
             
        'modified_FF_hydrophobic_pmf_unbiased_ROT017_150_20000':
            {'main_sim_dir': r"20200822-Hydrophobic_contacts_PMF_Bn_new_FF_UNBIASED_SIMS",
             'specific_sim': r"NPLM_unb-150.0-20000.000_ps-UShydrophobic_10-NPLMplumedhydrophobiccontactspulling-5.100_2_50_300_0.35-50000_ns-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT017_1",
             'prefix': 'nplm_prod',
                    },
             
        ## USING 40 CONTACTS
        'modified_FF_hydrophobic_pmf_unbiased_ROT017_40_10000':
            {'main_sim_dir': r"20200822-Hydrophobic_contacts_PMF_Bn_new_FF_UNBIASED_SIMS",
             'specific_sim': r"NPLM_unb-40.0-10000.000_ps-UShydrophobic_10-NPLMplumedhydrophobiccontactspulling-5.100_2_50_300_0.35-50000_ns-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT017_1",
             'prefix': 'nplm_prod',
                    },
             
        'modified_FF_hydrophobic_pmf_unbiased_ROT017_40_15000':
            {'main_sim_dir': r"20200822-Hydrophobic_contacts_PMF_Bn_new_FF_UNBIASED_SIMS",
             'specific_sim': r"NPLM_unb-40.0-15000.000_ps-UShydrophobic_10-NPLMplumedhydrophobiccontactspulling-5.100_2_50_300_0.35-50000_ns-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT017_1",
             'prefix': 'nplm_prod',
                    },
             
        'modified_FF_hydrophobic_pmf_unbiased_ROT017_40_20000':
            {'main_sim_dir': r"20200822-Hydrophobic_contacts_PMF_Bn_new_FF_UNBIASED_SIMS",
             'specific_sim': r"NPLM_unb-40.0-20000.000_ps-UShydrophobic_10-NPLMplumedhydrophobiccontactspulling-5.100_2_50_300_0.35-50000_ns-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT017_1",
             'prefix': 'nplm_prod',
                    },

        ### UNBIASED AFTER PMF USING 50 NS
        'plumed_unbiased_50ns_US_ROT001_40':
            {'main_sim_dir': r"20200925-Hydrophobic_contacts_Unbiased_sims_50ns",
             'specific_sim': r"NPLM_unb-40.0-50000.000_ps-UShydrophobic_10-NPLMplumedhydrophobiccontactspulling-5.100_2_50_1000_0.35-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT001_1",
             'prefix': 'nplm_prod',
                    },  
        'plumed_unbiased_50ns_US_ROT012_40':
            {'main_sim_dir': r"20200925-Hydrophobic_contacts_Unbiased_sims_50ns",
             'specific_sim': r"NPLM_unb-40.0-50000.000_ps-UShydrophobic_10-NPLMplumedhydrophobiccontactspulling-5.100_2_50_1000_0.35-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1",
             'prefix': 'nplm_prod',
                    },  
        'plumed_unbiased_50ns_US_ROT017_40':
            {'main_sim_dir': r"20200925-Hydrophobic_contacts_Unbiased_sims_50ns",
             'specific_sim': r"NPLM_unb-40.0-50000.000_ps-UShydrophobic_10-NPLMplumedhydrophobiccontactspulling-5.100_2_50_300_0.35-50000_ns-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT017_1",
             'prefix': 'nplm_prod',
                    },  
        
        
        }
            
## DEFINING SAVE IMAGE LOCATION
IMAGE_LOC=check_tools.check_path(r"R:\scratch\nanoparticle_project\nplm_sims\output_images")
#  r"/Users/alex/Box/VanLehnGroup/2.Research Documents/Alex_RVL_Meetings/20200323/images/nplm_project"
# r"C:\Users\akchew\Box Sync\VanLehnGroup\2.Research Documents\Alex_RVL_Meetings\20200316\images\nplm_project"
# r"C:\Users\akchew\Box Sync\VanLehnGroup\2.Research Documents\Alex_RVL_Meetings\20200224\images\nplm_project"

## DEFINING PARENT SIM PATH
PARENT_SIM_PATH = check_tools.check_path(r"R:\scratch\nanoparticle_project\nplm_sims")
NP_PARENT_SIM_PATH = check_tools.check_path(r"R:\scratch\nanoparticle_project\simulations")
            
##################################################
### CLASS FUNCTION TO ANALYZE THE JOB FOR NPLM ###
##################################################
class nplm_job_types:
    '''
    This code stores all information for the nplm job
    INPUTS:
        parent_sim_path: [str]
            path to sim directory
        main_sim_dir: [str]
            main sim directory
        specific_sim: [str]
            specific simulation name
    OUTPUTS:
        self.job_type: [str]
            job type name
        self.distance_xvg: [str]
            name of the  xvg file that contains the distances
        self.path_simulations: [str]
            full path to simulations
        self.frame_rate: [int]
            frame rate
        self.config_library: [list]
            list of configurations that your simulation contains (multiple for 
            umbrella sampling, none for normal unbiased)
    '''
    def __init__(self, 
                 parent_sim_path,
                 main_sim_dir,
                 specific_sim,
                 np_itp_file = "sam.itp"):
        ## STORING INPUTS
        self.parent_sim_path = parent_sim_path
        self.main_sim_dir = main_sim_dir
        self.specific_sim = specific_sim
        self.np_itp_file = np_itp_file
        
        ## GETTING JOB TYPE
        self.job_type, self.distance_xvg = self.find_job_type(specific_sim = self.specific_sim)
        
        ## FINDING PATH TO SIM
        self.find_path_to_sim()
        
        ## FINDING CONFIGURATION LIBRARIES
        self.config_library, self.path_simulation_list = self.find_config_libs()
        
        ## DEFINING GRO AND XTC FILES
        self.gro_file, self.xtc_file = self.find_gro_and_xtc_files()
        
        return
        
    ### FUNCTION TO DEFINE JOB TYPE
    @staticmethod
    def find_job_type(specific_sim):
        '''
        The purpose of this function is to find the job type and output the xvg
        for that job type.
        INPUTS:
            specific_sim: [str]
                specific simulation name
        OUTPUTS:
            job_type: [str]
                job type
            distance_xvg: [str]
                xvg file name
        '''
        ## DEFINING DEFAULT XVG FOR DISTANCE
        distance_xvg="pull_summary.xvg"
        
        ## DEFINING JOB TYPE
        if specific_sim.startswith("US"):
            job_type="US"
        elif specific_sim.startswith("NPLM_unb"):
            job_type="UNBIASED"
            distance_xvg="push_summary.xvg"
        elif specific_sim.startswith("NPLM_rev_unb"):
            job_type="UNBIASED"
            distance_xvg="push_summary.xvg"
        elif specific_sim.startswith("NPLM-"):
            job_type="PULL"
            distance_xvg="pull_summary.xvg"
        elif specific_sim.startswith("pullnplm"):
            job_type="PUSH"
            distance_xvg="push_summary.xvg"
        elif specific_sim.startswith("NPLMpulling2unb"):
            job_type="UNBIASED"
            distance_xvg="push_summary.xvg"
        else:
            job_type="UNDECIDED"
            distance_xvg="None.xvg"
            # print("Error, no specific sim starting with: %s"%(specific_sim) )
        return job_type, distance_xvg
    
    ### FUNCTION TO GET PATH TO SIM
    def find_path_to_sim(self):
        '''
        The purpose of this function is to get the path to simulations
        INPUTS:
            self: [obj]
                class object 
        OUTPUTS:
            self.path_simulations: [str]
                full path to simulations
            self.frame_rate: [int]
                frame rate
        '''
        ## DEFINING CONFIGURATION FILE
        if self.job_type=="US":
            sim_dir = r"4_simulations"
            ## DEFINING FRAME RATE
            self.frame_rate = 100 # ps/frame
        elif self.job_type=="PULL" or self.job_type=="PUSH" or self.job_type=="UNBIASED":
            sim_dir = r""
            ## DEFINING FRAME RATE
            self.frame_rate = 10 # ps/frame
        else:
            sim_dir = r""
            self.frame_rate = 10
        
        ## PATH TO SIMULATION
        self.path_simulations = os.path.join(self.parent_sim_path,
                                             self.main_sim_dir,
                                             self.specific_sim,
                                             sim_dir)
        
        return
    
    ### FUNCTION TO FIND ALL CONFIGURATION LIBRARIES
    def find_config_libs(self):
        '''
        The purpose of this function is to find all configuration libraries
        INPUTS:
            self: [obj]
                class object
        OUTPUTS:
            config_library: [list]
                list of configuration libraries
            path_simulation_list: [list]
                list of simulations
        '''
        
        ## GETTING ALL CONFIGURATION LIBRARIES
        if self.job_type=="US":
            config_library = [ os.path.basename(each_dir) for each_dir in glob.glob(self.path_simulations + "/*") ]
            ## FINDING ITP FILE
            self.np_itp_file =  os.path.join(os.path.dirname(self.path_simulations),
                                             "1_input_files",
                                             self.np_itp_file)
            
                                             
                                             
        else:
            config_library = [""]
        
        ## GETTING PATH SIM LIST
        path_simulation_list = [ os.path.join(self.path_simulations, specific_config) for idx, specific_config in enumerate(config_library) ]
        
        
        return config_library, path_simulation_list
    
    ### FUNCTION TO FIND GRO AND XTC FILES
    def find_gro_and_xtc_files(self):
        '''
        This function finds the gro and xtc file based on your input job type
        INPUTS:
            self: [obj]
                class object
        OUTPUTS:
           gro_file: [str]
               gro file string
           xtc_file: [str]
               xtc file string
        '''
        ## DEFINING GRO AND XTC FILE
        if self.job_type != "US":
            gro_file = "nplm_prod_skip_1_non-Water_0_-1.gro"
            xtc_file = "nplm_prod_skip_1_non-Water_0_-1.xtc"
        else:
            gro_file = "nplm_prod_skip_10_non-Water_0_50000.gro"
            xtc_file = "nplm_prod_skip_10_non-Water_0_50000.xtc"
        
        return gro_file, xtc_file

### FUNCTION TO GET UMBRELLA SAMPLING KEY
def find_us_sampling_key(sim_type, 
                         want_config_key = False,
                         want_shorten_lig_name = False):
    '''
    The purpose of this function is to find the umbrella sampling key from 
    unbiased simulation key
    INPUTS:
        sim_type: [str]
            key for unbiased simulations
        want_config_key; [logical]
            True if you want the config key
        want_shorten_lig_name: [logical]
            True if you want shortened lig name
    OUTPUTS:
        us_dir_key: [str]
            corresponding key for umbrella sampling simulations
        config_key: [str] (optional)
            config key telling you the current distance. It is useful for debugging.
        want_shorten_lig_name: [logical]
            True if you want shorten lig name
    '''
    ### GETTING DETAILS FROM UMBRELLA SAMPLING SIMULATIONS
    if "unbiased" in sim_type:
        ## FINDING UMBRELLA SAMPLING THAT MATCHES IT
        if "rev" in sim_type:
            if "ROT001" in sim_type:
                us_dir_key = "us_reverse_R01"
                lig_name="C1"
            elif "ROT012" in sim_type:
                us_dir_key = "us_reverse_R12"
                lig_name="C10"
            else:
                print("Error! No reverse sim of this type is found: %s"%(sim_type) )
        else:
            if "ROT001" in sim_type:
                us_dir_key = "us_forward_R01"
                lig_name="C1"
            elif "ROT012" in sim_type:
                us_dir_key = "us_forward_R12"
                lig_name="C10"
            elif "ROT017" in sim_type:
                us_dir_key = "modified_FF_us_forward_R17"
                lig_name="Bn"
            else:
                print("Error! No forward sim of this type is found: %s"%(sim_type) )
    ## FINDING CONFIG
    if want_config_key is True:
        
        ## GETTING DICTIONARY
        dict_key = NPLM_SIM_DICT[sim_type]['specific_sim']
        
        config_key = dict_key.split('-')[1]
        
        ## CHECKING CONFIG KEY
        if '_' in config_key:
            ## SPLIT AGAIN
            config_key = config_key.split('_')[0]
        
        
        if want_shorten_lig_name is True:
            return us_dir_key, config_key, lig_name
        else:
            return us_dir_key, config_key
    else:
        return us_dir_key
    
    
