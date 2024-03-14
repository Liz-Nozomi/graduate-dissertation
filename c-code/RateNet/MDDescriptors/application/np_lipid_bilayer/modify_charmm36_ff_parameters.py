#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
modify_charmm36_ff_parameters.py

The purpose of this script is to modify charmm36 parameters according to the 
article below:
    Khan, H. M. et al. Improving the Force Field Description of Tyrosine-Choline Cation-π Interactions: 
    QM Investigation of Phenol-N(Me)4+ Interactions. J. Chem. Theory Comput. 12, 5585–5595 (2016).

This article goes through aromatic-cation interactions and suggests that 
inclusion of specific pairs could improve the accuracy of pi-cation interactions. 


Written by: Alex K. Chew (08/12/2020)

PHENOL CARBON: (sigma, epsilon)
CG2R61     6    12.011000    0.000  A  0.355005321205  0.29288

TETRAMETHYLAMMONIUM NITROGEN AND CARBON
NG3P0     7    14.007000    0.000  A  0.329632525712  0.83680
CG334     6    12.011000    0.000  A  0.394668132136  0.32217

Between phenol carbon and N, default epsilon: sqrt(0.29288 * 0.83680) / 4.184 = 0.1183 kcal/mol (matching the paper)
Between phenol carbon and C_TMA: default epsilon: sqrt(0.29288 * 0.32217) / 4.184 = 0.0734 (matching the paper)

To get sigma_ij: sigma_ij = (sigma_i + sigma_j) / 2
To get Rmin: Take derivative of LJ potential and set to zero, resulting in:
    Rmin = (1/2)^(-1/6) * sigma_ij
    
    Therefore:
        for phenol carbon and C
            (0.355005321205 + 0.394668132136) / 2 * (1/2)^(-1/6) = 0.42074 nm
        for phenol carbon and N: (matching the paper)
            (0.355005321205 + 0.329632525712) / 2 * (1/2)^(-1/6) = 0.38424 nm

        
Therefore, sigma values and combination rules of sigma remain unchanged

We need to add a [ nonbond_parms ] section immediately after the [ atomtypes ] section


Examples of ParmEd from Jonathan (Not used thoguh!):
    
    from parmed import gromacs
    TiO2_top=gromacs.GromacsTopologyFile(out_TiO2_itpfile_name)
    gmx_top=gromacs.GromacsTopologyFile(working_path+'.top',xyz=working_path+'.gro')
    
    working_path='R:/AMBER//CBCr_top/CBCr_GMX_new'
    lig_top=gromacs.GromacsTopologyFile(working_path+'.top')
    lig_top.atoms[1].charge+=charge_per_lig/2
    lig_top.atoms[0].charge+=charge_per_lig/2
    lig_top.write(working_path+'_edit.top')


Example of non_bond parms (Buckingham pi). NOTE. For Charmm36, we use LJ potential, so the "func" is going to be 2
[ nonbond_params ]
  ; i    j func       V(c6)        W(c12)
    O    O    1 0.22617E-02   0.74158E-06
    O   OA    1 0.22617E-02   0.13807E-05
    .....

"""

## LOADING MODULES
import os

## COPY/PASTE TOOLS
import shutil  
from optparse import OptionParser # Used to allow commands within command line

## LOADING CHECK TESTING
from MDBuilder.core.check_tools import check_testing ## CHECKING PATH FOR TESTING

## LOADING SPECIFIC MODULES
import MDDescriptors.core.read_write_tools as read_write_tools

## NONBONDED FUNCTION
NBFUNC=1
# 2 for LJ parameters, consistent with the defaults of charmm force field

## DEFINING DEFAULTS FROM PAPER
EPSILON_KCALPERMOL = {
        'C_phenol-C_TMA': 0.2081, # Negative sign is omitted
        'C_phenol-N_TMA': 0.2400,        
        }

## CONVERTING TO KJ/MOL
EPSILON_KJPERMOL = { each_key: EPSILON_KCALPERMOL[each_key] * 4.184  for each_key in EPSILON_KCALPERMOL}

## GETTING SIGMA VALUES
SIGMA_NM = {
        'C_phenol-C_TMA': 0.3748367266705, # (0.355005321205 + 0.394668132136) / 2
        'C_phenol-N_TMA': 0.3423189234585, # (0.355005321205 + 0.329632525712) / 2
        }

## DEFINING ATOMTYPES FOR PHENOL CARBON
PHENOL_CARBON_ATOMTYPES = [
        'CG2R61',
        ]

## DEFINING TMA ATOM TYPES
TMA_CARBON_ATOMTYPES = [
        'CG334',
        ## ADDING FOR DOPC
        "CTL5",
        ]
#        "CTL2",

## DEFINING TMA ATOM TYPES
TMA_NITROGEN_ATOMTYPES = [
        'NG3P0',
        ## ADDING FOR DOPC
        "NTL",
        ]

### FUNCTION TO GENERATE ATOM TYPES
def combine_phenol_tma_nb_pairs(PHENOL_CARBON_ATOMTYPES = PHENOL_CARBON_ATOMTYPES,
                                TMA_CARBON_ATOMTYPES = TMA_CARBON_ATOMTYPES,
                                TMA_NITROGEN_ATOMTYPES = TMA_NITROGEN_ATOMTYPES,
                                SIGMA_NM = SIGMA_NM,
                                EPSILON_KJPERMOL = EPSILON_KJPERMOL,
                                NBFUNC = NBFUNC,
                                ):
    '''
    This function combines the nonbonded pair information for phenol-TMA.
    INPUTS:
        PHENOL_CARBON_ATOMTYPES: [list]
            list of phenol carbon types
        TMA_CARBON_ATOMTYPES: [list]
            list of TMA carbon types
        TMA_NITROGEN_ATOMTYPES: [list]
            list of TMA nitrogen types
        SIGMA_NM: [dict]
            dictionary of sigma in nanometers
        EPSILON_KJPERMOL: [dict]
            dictionary of epsilon values in kj/mol
        NBFUNC: [int]
            nonbonded function type, 2 is LJ parameters
    OUTPUTS:
        MODIFIED_FF_NBTYPES: [list]
            list of forcefield nonbonded types
    '''
    ## CREATING EMPTY LIST
    MODIFIED_FF_NBTYPES = []

    ## LOOPING THROUGH PHENOL AND CARBON
    for ph_atom_types in PHENOL_CARBON_ATOMTYPES:
        ## LOOPING THROUGH TMA CARBON
        for tma_c_atomtypes in TMA_CARBON_ATOMTYPES:
            ## CREATING LIST
            new_entry = [ph_atom_types,
                         tma_c_atomtypes,
                         NBFUNC,
                         SIGMA_NM['C_phenol-C_TMA'],
                         EPSILON_KJPERMOL['C_phenol-C_TMA'],
                         ]
            ## APPENDING
            MODIFIED_FF_NBTYPES.append(new_entry)
        
        ## LOOPING THROUGH NITROGEN ATOMS
        for tma_n_atomtypes in TMA_NITROGEN_ATOMTYPES:
            ## CREATING LIST
            new_entry = [ph_atom_types,
                         tma_n_atomtypes,
                         NBFUNC,
                         SIGMA_NM['C_phenol-N_TMA'],
                         EPSILON_KJPERMOL['C_phenol-N_TMA'],
                         ]
            ## APPENDING
            MODIFIED_FF_NBTYPES.append(new_entry)
        
    return MODIFIED_FF_NBTYPES

## GETTING MODIFIED FF TYPES
MODIFIED_FF_NBTYPES = combine_phenol_tma_nb_pairs(PHENOL_CARBON_ATOMTYPES = PHENOL_CARBON_ATOMTYPES,
                                                  TMA_CARBON_ATOMTYPES = TMA_CARBON_ATOMTYPES,
                                                  TMA_NITROGEN_ATOMTYPES = TMA_NITROGEN_ATOMTYPES,
                                                  SIGMA_NM = SIGMA_NM,
                                                  EPSILON_KJPERMOL = EPSILON_KJPERMOL,
                                                  NBFUNC = NBFUNC,
                                                  )

### FUNCTION TO CREATE NEW FOLDER
def create_new_ff_folder(path_to_ff,
                         path_to_new_ff_folder):
    '''
    This function creates a new force field folder based on path names
    INPUTS:
        path_to_ff: [str]
            input folder path
        path_to_new_ff_folder: [str]
            output folder path
    OUTPUTS:
        destination: [str]
            destination of the new folder
    '''
    ## CHECKING IF FF FILE PATH IS TRUE
    if os.path.isdir(path_to_new_ff_folder) is True:
        print("--------------------------------------")
        print("Removing force field folder duplicate!")
        print("Path: %s"%(path_to_new_ff_folder))
        print("--------------------------------------")
        shutil.rmtree(path_to_new_ff_folder)

    
    ## COPYING FF FOLDER
    print("*** Copying forcefield files! ***")
    print("--> Orig path: %s"%(path_to_ff))
    print("--> New path: %s"%(path_to_new_ff_folder))
    destination = shutil.copytree(path_to_ff, path_to_new_ff_folder) 
    
    return destination

### FUNCTION TO WRITE NEW NBIFX
def update_nbfix_file(path_nbfix,
                      path_output_nbfix,
                      MODIFIED_FF_NBTYPES = MODIFIED_FF_NBTYPES):
    '''
    This function updates the nbfix file to an output file.
    INPUTS:
        path_nbfix: [str]
            path to input nbfix file
        path_output_nbfix: [str]
            path to output nbfix file
        MODIFIED_FF_NBTYPES: [dict]
            dictionary contianin all nb types
    OUTPUTS:
        nbfix: [list]
            list of nbfix information
    '''
    ### FUNCTION TO LOAD NBFIX
    nbfix = read_write_tools.read_file_as_line(path_nbfix)
    
    ## ADDING TO LIST
    added_list = []
    for each_new_type in MODIFIED_FF_NBTYPES:
        new_type_string = " %s  %s  %d  %.12f  %.12f"%(tuple(each_new_type))
        added_list.append(new_type_string)
        
    ## APPENDING
    nbfix.extend(added_list)
    
    ## OUTPUT
    print("Writing new NBFIX file")
    print("Path: %s"%(path_output_nbfix))
    with open(path_output_nbfix, 'w') as f:
        for each_item in nbfix:
            f.write("%s\n"%(each_item))
            
    return nbfix

## DEFINING MAIN FUNCTION
def main_modify_charmm36_parameters(main_path,
                                    ff_folder,
                                    new_ff_folder,
                                    ff_nbfix,
                                    ):
    '''
    Main function that modifies the charmm36 force field parameters.
    INPUTS:
        main_path: [str]
            main path to folder where force field is located
        ff_folder: [str]
            force field folder name
        new_ff_folder: [str]
            new force field folder name
        ff_nbfix: [str]
            file within nbfix.itp
    OUTPUTS:
        destination: [str]
            destination of output folder
        nbfix: [list]
            list of nbfix outputs
    '''
    ##### DEFINING PATHS ####
    ## DEFINING PATH TO FF
    path_to_ff = os.path.join(main_path,
                              ff_folder
                              )

    ## DEFINING PATH TO NEW FOLDER
    path_to_new_ff_folder = os.path.join(main_path, new_ff_folder)
    
    ## DEFINING PATH TO NBFIX
    path_nbfix = os.path.join(path_to_ff,
                              ff_nbfix)
    
    ## DEFINING PATH OUTPUT
    path_output_nbfix = os.path.join(path_to_new_ff_folder,
                                     ff_nbfix
                                     )
    
    

    
    ## CREATING NEW FOLDER
    destination = create_new_ff_folder(path_to_ff = path_to_ff,
                                       path_to_new_ff_folder = path_to_new_ff_folder)
    
    ## UPDATING NBFIX
    nbfix = update_nbfix_file(path_nbfix,
                          path_output_nbfix)
    
    return destination, nbfix

#%%
###############################################################################
### MAIN SCRIPT
###############################################################################
if __name__ == "__main__":
    
    ### TURNING TEST ON / OFF
    testing = check_testing()  # False if you're running this script on command prompt!!!
    
    ## TESTING IS ON
    if testing is True:
    
        ## DEFINING PATH TO LOCATION
        main_path = r"/Volumes/akchew/scratch/nanoparticle_project/nplm_sims/20200713-US_PLUMED_iter2/UShydrophobic_10-NPLMplumedhydrophobiccontactspulling-5.100_2_50_300_0.35-25000_ns-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT017_1/1_input_files"
        
        ## FORCE FIELD NAME
        ff_folder = r"charmm36-jul2017.ff"
        
        ## NEW FF FOLDER NAME
        new_ff_folder =  r"charmm36-jul2017-mod.ff"
        
        ## DEFINING NONBONDING FILE
        ff_nbfix="nbfix.itp"
        
    else:
        ### DEFINING PARSER OPTIONS
        # Adding options for command line input (e.g. --ligx, etc.)
        use = "Usage: %prog [options]"
        parser = OptionParser(usage = use)
        
        ## PREPARATION SIMULATION GOLD FOLDER
        parser.add_option("--main_path", dest="main_path", action="store", type="string", help="Main path to ff folder", default=".")
        parser.add_option("--input_ff_folder", dest="ff_folder", action="store", type="string", help="Input force field folder", default=r"charmm36-jul2017.ff")
        parser.add_option("--output_ff_folder", dest="new_ff_folder", action="store", type="string", help="Input force field folder", default=r"charmm36-jul2017-mod.ff")
        parser.add_option("--ff_nbfix", dest="ff_nbfix", action="store", type="string", help="NBFix file within ff folder", default=r"nbfix.itp")
        
        ## PARSING ARGUMENTS
        (options, args) = parser.parse_args() # Takes arguments from parser and passes them into "options" and "argument"
        
        ## RESTORING ARGUMENTS
        main_path = options.main_path
        ff_folder = options.ff_folder
        new_ff_folder = options.new_ff_folder
        ff_nbfix = options.ff_nbfix
        
    ## DEFINING INPUTS
    inputs_ff = {
            'main_path': main_path,
            'ff_folder': ff_folder,
            'new_ff_folder': new_ff_folder,
            'ff_nbfix': ff_nbfix,
            }
    
    ## CREATING NB FIX FOLDER
    destination, nbfix = main_modify_charmm36_parameters(**inputs_ff)
    
    