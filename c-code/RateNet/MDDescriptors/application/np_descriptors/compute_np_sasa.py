#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_np_sasa.py
The purpose of this script is to compute the SASA for a gold nanoparticle. 

Written by: Alex K. Chew (03/31/2020)

Algorithm:
    - Center gold core with only the nanoparticle
    - Use GMX SASA to compute the SASA across the group
    
    
Example GROMACS code:
    
    
1. Create index file
gmx make_ndx -f sam_prod.tpr -o sam_prod_center_GNP_only.ndx
keep 0
r AUNP | r R12
name 1 GNP
q

2. TRJCONV to get the trajectory

# XTC file
gmx trjconv -f sam_prod.xtc -s sam_prod.tpr -o sam_prod_center_GNP_only.xtc -n sam_prod_center_GNP_only.ndx -center -pbc mol -dt 100
GNP
GNP

# GRO FILE
gmx trjconv -f sam_prod.xtc -s sam_prod.tpr -o sam_prod_center_GNP_only.gro -n sam_prod_center_GNP_only.ndx -center -pbc mol -dump 0
GNP
GNP

# TPR FILE
gmx convert-tpr -s sam_prod.tpr -n sam_prod_center_GNP_only.ndx -o sam_prod_center_GNP_only.tpr
GNP

3. GMX SASA
gmx sasa -f sam_prod_center_GNP_only.xtc -s sam_prod_center_GNP_only.tpr

Note that our gold atoms are not well-defined! It will need an output 
file. 

Debug on SWARM:
python3.6 /home/akchew/bin/pythonfiles/modules/MDDescriptors/application/np_descriptors/compute_np_sasa.py


FUNCTIONS:
    print_vdw_radii:
        prints the vdw radius dat file
    generate_gnp_only:
        uses gmx trjconv to extract gold nanoparticles laone
    compute_gmx_sasa:
        function that runs gmx sasa on the system
    main_compute_np_sasa:
        main function that computes nanoparticle sasa

"""


## IMPORTING FUNCTIONS
import os
import mdtraj as md

## LIGAND RESIDUE NAMES
from MDDescriptors.application.nanoparticle.core.find_ligand_residue_names import get_ligand_names_within_traj

## CHECK TOOLS
import MDDescriptors.core.check_tools as check_tools

## TRJCONV COMMANDS
from MDDescriptors.traj_tools.trjconv_commands import generate_gromacs_command, run_bash_command, GMX_COMMAND_PREFIX

## IMPORTING TOOLS
import MDDescriptors.core.import_tools as import_tools # Loading trajectory details

## GLOBAL VARS
from MDDescriptors.application.np_descriptors.global_vars import GOLD_RESNAME

### FUNCTION TO PRINT VDW RADII SCRIPT
def print_vdw_radii(output_file="vdwradii.dat"):
    '''
    The purpoes of this function is to print the vdw radii data file used for 
    GMX SASA. The VDW radii should not chnage from Bondi's 1964 version.
    INPUTS:
        output_file: [str]
            string to the output file
    OUTPUTS:
        output_data: [str]
            output data for the vdw radii. In addition, the file will otuput hte radius 
            in terms of a *.dat file.
    '''
    
    output_data=\
    '''\
; Very approximate VanderWaals radii
; only used for drawing atoms as balls or for calculating atomic overlap.
; longest matches are used
; '???' or '*' matches any residue name
; 'AAA' matches any protein residue name
; Source: http://en.wikipedia.org/wiki/Van_der_Waals_radius
; These come from A. Bondi, "van der Waals Volumes and Radii",
; J. Phys. Chem. 68 (1964) 441-451
???  H     0.12
???  C     0.17
???  N     0.155
???  O     0.152
???  F     0.147
???  P     0.18
???  S     0.18
???  Cl    0.175
???  Au    0.213
???  VS    0.213
; Water charge sites
SOL  MW    0
SOL  LP    0
; Masses for vsite construction
GLY  MN1   0
GLY  MN2   0
ALA  MCB1  0
ALA  MCB2  0
VAL  MCG1  0
VAL  MCG2  0
ILE  MCG1  0
ILE  MCG2  0
ILE  MCD1  0
ILE  MCD2  0
LEU  MCD1  0
LEU  MCD2  0
MET  MCE1  0
MET  MCE2  0
TRP  MTRP1 0
TRP  MTRP2 0
THR  MCG1  0
THR  MCG2  0
LYSH MNZ1  0
LYSH MNZ2  0

'''
    
    ## OPENING OUTPUT FILE
    with open(output_file, 'w') as f:
        f.write(output_data)
    
    return output_data

### FUNCTION TO GENERATE GRO XTC FOR SASA
def generate_gnp_only(wd,
                      input_prefix,
                      output_prefix,
                      dt = 100,
                      rewrite=False):
    '''
    This script generates gnp only systems by indexing and finding only 
    GNP atoms.
    INPUTS:
        wd: [str]
            working directory location
        input_prefix: [str]
            input prefix for .gro, .xtc, .tpr
        output_prefix: [str] 
            output prefix for .gro, .xtc, .tpr
        df: [int]
            dt time for trajectory
        rewrite: [str]
            True if you want to rewrite
    OUTPUTS:
        gro, ndx, tpr, xtc file names
    '''
    
    ## DEFINING OUTPUTS
    output_gro = output_prefix + '.gro'
    output_tpr = output_prefix + '.tpr'
    output_xtc = output_prefix + '.xtc'
    output_ndx = output_prefix + '.ndx'
    
    ## IMPORTING TRAJECTORY FOR SINGLE FRAME
    path_to_gro = os.path.join(wd,
                                input_prefix + ".gro")
    
    ## CHECKING PATH
    traj = md.load( path_to_gro ) 
    
    ## GETTING RESIDUE NAMES
    lig_res_name = get_ligand_names_within_traj(traj = traj)
    
    ## GETTING STRING FOR GNP
    residue_name_string = ' | '.join(["r %s"%(GOLD_RESNAME)] + ["r %s"%(each_lig) for each_lig in lig_res_name])
    
    ## DEFINING OUTPUT NAME
    output_name="GNP"
    
    ##################
    ### INDEX FILE ###
    ##################
    
    ## INPUTTING
    ndx_inputs=\
        """
        keep 0
        %s
        name 1 %s
        q
        """%(residue_name_string,
            output_name)
    
    ### GENERATING GROMACS COMMAND FOR INDEX
    ndx_command = generate_gromacs_command(GMX_COMMAND_PREFIX,
                                           arguments=['make_ndx'],
                                           input_files={'-f': input_prefix + '.tpr'},
                                           output_files={'-o': output_ndx,
                                                          }
                                           )

    ## RUNNING COMMAND
    run_bash_command(command = ndx_command,
                     wd = wd,
                     string_input = ndx_inputs,
                     path_check = os.path.join(wd, output_ndx),
                     rewrite = rewrite,
                     )
        
    ################
    ### XTC FILE ###
    ################

    ## XTC FILE
    xtc_args = ['trjconv', '-pbc', 'mol', '-center', '-dt', "%d"%(dt) ]
    xtc_command = generate_gromacs_command(GMX_COMMAND_PREFIX,
                                           arguments = xtc_args,
                                           input_files={'-s': input_prefix + '.tpr',
                                                        '-f': input_prefix + '.xtc',
                                                        '-n': output_ndx,},
                                           output_files={'-o': output_xtc,
                                                          }
                                           )
    ## INPUTS FOR XTC
    output_selection = \
    """
    %s
    %s
    """%(output_name,
        output_name)

    ## RUNNING COMMAND
    run_bash_command(command = xtc_command,
                     wd = wd,
                     string_input = output_selection,
                     path_check = os.path.join(wd, output_xtc),
                     rewrite = rewrite,
                     )
    
    ################
    ### GRO FILE ###
    ################
    gro_args = xtc_args + [ '-dump', '0' ]
    ### GENERATING GROMACS COMMAND FOR INDEX
    gro_command = generate_gromacs_command(GMX_COMMAND_PREFIX,
                                           arguments = gro_args,
                                           input_files={'-s': input_prefix + '.tpr',
                                                        '-f': input_prefix + '.xtc',
                                                        '-n': output_ndx,},
                                           output_files={'-o': output_gro,
                                                          }
                                        )

    ## RUNNING COMMAND
    run_bash_command(command = gro_command,
                     wd = wd,
                     string_input = output_selection,
                     path_check = os.path.join(wd, output_gro),
                     rewrite= rewrite,
                     )
    
    ################
    ### TPR FILE ###
    ################

    ### GENERATING GROMACS COMMAND FOR INDEX
    # gmx convert-tpr -s sam_prod.tpr -o sam_prod-center_pbc_mol.tpr
    tpr_command = generate_gromacs_command(GMX_COMMAND_PREFIX,
                                           arguments=['convert-tpr'],
                                           input_files={'-s': input_prefix + '.tpr',
                                                        '-n': output_ndx},
                                           output_files={'-o': output_tpr,
                                                          }
                                        )
    
    ## DEFINING OUTPUTS
    ## INPUTS FOR XTC
    tpr_inputs = \
    """%s
    """%(output_name,
         )

    ## RUNNING COMMAND
    run_bash_command(command = tpr_command,
                     wd = wd,
                     string_input = tpr_inputs,
                     path_check = os.path.join(wd, output_tpr),
                     rewrite= rewrite,
                     )
    
    return output_gro, output_tpr, output_xtc, output_ndx


## RUNNING GMX SASA
def compute_gmx_sasa(wd,
                     tpr_file,
                     xtc_file,
                     output_sasa_file = 'sasa.xvg',
                     rewrite = False):
    '''
    The purpose of this function is to compute gmx sasa. Ideally, you should have prepared 
    a script to extract the trajectory (i.e. remove solvents). Otherwise, 
    these solvents are included in the calculation, which may construe 
    your final results.
    INPUTS:
        wd: [str]
            working directory
        tpr_file: [str]
            tpr file to compute the sasa
        xtc_file: [str]
            xtc file to compute the sasa
        output_sasa_file: [str]
            output sasa_file name
        rewrite: [logical]
            True if you want to rewrite the file
    OUTPUTS:
        output_sasa_file: [str]
            output sasa file name
    '''
    ## PRINTING VDW RADII FILE 
    print_vdw_radii(output_file=os.path.join(wd,"vdwradii.dat") )
    
    ## DEFINING SASA INPUTS
    sasa_inputs="System"
    
    ### GENERATING GROMACS COMMAND
    sasa_command = generate_gromacs_command(GMX_COMMAND_PREFIX,
                                           arguments=['sasa'],
                                           input_files={'-s': tpr_file,
                                                        '-f': xtc_file},
                                           output_files={'-o': output_sasa_file,
                                                          }
                                           )

    ## RUNNING COMMAND
    run_bash_command(command = sasa_command,
                     wd = wd,
                     string_input = sasa_inputs,
                     path_check = os.path.join(wd, output_sasa_file),
                     rewrite = rewrite,
                     )
        
    return output_sasa_file

### FUNCTION TO COMPUTE GMX SASA
def main_compute_np_sasa(path_to_sim,
                         input_prefix,
                         dt =  100,
                         rewrite = False,
                         ):
    '''
    Main function to compute SASA of gold nanoparticles. The script 
    first generates a gold nanoparticle alone and centered. Then, it 
    uses GMX SASA to compute SASA values. Note that this script generates 
    its own vdwradii in compliance with Bondi et al 1964. 
    
    INPUTS:
        path_to_sim: [str]
            path to the simulation
        input_prefix: [str]
            prefix for your inputs
        dt: [int]
            time step for the analysis
        rewrite: [logical]
            True if you want to rewrite the data
    OUTPUTS:
        sasa_full_data: [str]
            output full data for gmx sasa
        sasa_extract_data: [str]
            output extracted data, e.g.:
                [['0.000', '251.150'],
                 ['100.000', '244.643'],
                 ['200.000', '253.470'],
                 ['300.000', '250.106'],
                 ['400.000', '246.267'],
                 ['500.000', '243.262'],
                 ['600.000', '235.550'],
                 ['700.000', '230.803'],
                 ['800.000', '230.829'],
                 ['900.000', '228.287']]
            Here, the SASA of each time step is evaluated in nm^2
    '''
    ## DEFINING INPUTS
    gnp_only_inputs={
            'wd': path_to_sim,
            'input_prefix': input_prefix,
            'output_prefix': input_prefix + "_GNP_only",
            'dt': dt,
            'rewrite': rewrite,
            }
    
    ## GENERATING GRO, TPR, AND XTC
    output_gro, output_tpr, output_xtc, output_ndx = generate_gnp_only(**gnp_only_inputs)        
    
    ## DEFINING SASA FILE NAME
    sasa_file_name = os.path.basename(os.path.splitext(output_xtc)[0]) + "_sasa.xvg"
    
    ## DEFINING INPUTS
    sasa_inputs={
            'wd': path_to_sim,
            'xtc_file': output_xtc,
            'tpr_file': output_tpr,
            'output_sasa_file': sasa_file_name,
            'rewrite': rewrite,
            }
    
    ## RUNNING GMX SASA
    output_sasa_file = compute_gmx_sasa(**sasa_inputs)

    ## READING SASA FILE
    sasa_full_data, sasa_extract_data = import_tools.read_xvg(os.path.join(path_to_sim,
                                                                           output_sasa_file))
    
    return sasa_full_data, sasa_extract_data

#%% MAIN SCRIPT
if __name__ == "__main__":
    
    ## DEFINING PATH
    path_to_sim = r"/Volumes/akchew/scratch/nanoparticle_project/simulations/ROT_WATER_SIMS/EAM_300.00_K_2_nmDIAM_ROT012_CHARMM36jul2017_Trial_1"
    # r"/home/akchew/scratch/nanoparticle_project/simulations/ROT_WATER_SIMS/EAM_300.00_K_2_nmDIAM_ROT012_CHARMM36jul2017_Trial_1"
    
    ## DEFINING INPUT PREFIX
    input_prefix="sam_prod"
    
    ## DEFINING OUTPUT PREFIX
    output_prefix=input_prefix + "_GNP_only"
    
    ## CHECKING PATH
    path_to_sim = check_tools.check_path(path_to_sim)
    
    ## DEFINING INPUT PREFIX
    input_prefix = 'sam_prod'
 
    ## DEFINING INPUTS FOR MAIN FUNCTION
    inputs_for_main = {
            'path_to_sim': path_to_sim,
            'input_prefix': input_prefix,
            'rewrite': False,
            }
    
    ## RUNNING MAIN FUNCTION
    sasa_full_data, sasa_extract_data = main_compute_np_sasa(**inputs_for_main)
        
    

    

