# -*- coding: utf-8 -*-
"""
compute_rmsf_of_ligands.py
The purpose of this script is to compute the root-mean-squared fluctuations of 
the ligands on the nanoparticle. 

This code will use the GROMACS commands to run the RMSF. Then, we will 
take the outputs and extract all the information. 

Written by: Alex K. Chew (alexkchew@gmail.com, 02/06/2020)


GROMACS COMMANDS FOR RMSD


REMOVING ROTATION AND TRANSLATION

function rot_trans_ {
    ## DEFINING  VARIABLES
    input_tpr_file="$1"
    input_xtc_file="$2"
    res_name="$3"
    system_name="$4"
    output_xtc_file="$5"
    ## DEFINING INDEX FILE
    index_file="rot_trans.ndx"

### CREATING INDEX FILE
gmx make_ndx -f "${input_tpr_file}" -o "${index_file}" << INPUT
keep 0
r ${res_name}
q
INPUT

### CHECKING IF THE RESIDUE IS WITHIN (Checks if string is empty)
if [ -z "$(grep ${res_name} ${index_file})" ]; then
gmx make_ndx -f "${input_tpr_file}" -o ${index_file} << INPUT
r ${res_name}
q
INPUT
fi

### EXTRACTION AND PLACES CENTER OF MASSES WITHIN BOX
gmx trjconv -s "${input_tpr_file}" -f "${input_xtc_file}" -o "${input_xtc_file%.xtc}_center" -pbc mol -center -n ${index_file} << INPUT
${res_name}
${system_name}
INPUT
# CENTERING ON RESIDUE, OUTPUTTING SYSTEM

### RESTRAINING ROTATION AND TRANSLATIONAL DEGREES OF FREEDOM
gmx trjconv -s "${input_tpr_file}" -f "${input_xtc_file%.xtc}_center" -o "${output_xtc_file}" -fit rot+trans -center -n ${index_file} << INPUT
${res_name}
${res_name}
${system_name}
INPUT
# CENTERING AND ROT+TRANS ON RESIDUE, OUTPUTTING SYSTEM
}


############################
### FINDING RESIDUE NAME ###
############################

## FINDING NAME OF DIRECTORY
dir_name=$(basename "${path_to_analysis_dir}")

## FINDING LIGAND NAME
ligand_name=$(extract_lig_name_from_dir "${dir_name}")

## FINDING RESIDUE NAME
residue_name_intersection="$(find_residue_name_from_ligand_txt ${ligand_name})"

###########################################
### CREATING INDEX FILE FOR LIGAND ONLY ###
###########################################

## CREATING INDEX FILE (LIGAND ONLY)
gmx make_ndx -f "${input_tpr_file_}" -o "${output_ligand_index}" << INPUTS
keep 1
keep 0
r ${residue_name_intersection}
q
INPUTS

############################
### RUNNING GMX COMMANDS ###
############################

## USING RMSD TO GET PDB FILE
gmx rmsf -s "${input_tpr_file_}" \
         -f "${input_xtc_file_}" \
         -ox "${rmsf_pdb_file_}" \
         -n "${output_ligand_index}" \
         -o "${output_file}" << INPUTS
${residue_name_intersection}
INPUTS

## USING RMS COMMAND TO GET THE ROOT MEAN SQUARED ERROR OVER TIME
gmx rms -s "${rmsf_pdb_file_}" \
        -f "${input_xtc_file_}" \
        -o "${output_rms_over_time}" << INPUTS
${residue_name_intersection}
${residue_name_intersection}
INPUTS

## CLEANING UP ANY EXTRANEOUS DATA
rm \#*


"""
### IMPORTING MODULES
import numpy as np
import pandas as pd
import os
import mdtraj as md

## MDBUILDERS FUNCTIONS
from MDBuilder.core.check_tools import check_testing, check_server_path

## IMPORTING COMMANDS 
from MDDescriptors.traj_tools.trjconv_commands import convert_with_trjconv, generate_gro_xtc_with_center_AUNP

## IMPORTING TOOLS
import MDDescriptors.core.import_tools as import_tools # Loading trajectory details

## IMPORTING FINDING LIG RESIDUE NAMES
from MDDescriptors.application.nanoparticle.core.find_ligand_residue_names import get_ligand_names_within_traj

#######################################
### CLASS FUNCTION TO READ XVG FILE ###
#######################################
class compute_gmx_rmsf:
    '''
    This function analyzes RMSF output
    INPUTS:
        path_xvg: [str]
            path to xvg file
        path_gro_file: [str, default = None]
            path to gro file. This is useful to load all the atoms thare are 
            associated with the RMSF function.
    OUTPUTS:
        self.avg_rmsf: [float]
            average RMSF
        self.gro_file: [obj]
            md trajectory gro file object
        self.output_rmsf_per_atom: [np.array]
            RMSF per atom basis
    '''
    ## INITIALIZING
    def __init__(self,
                 path_xvg,
                 path_gro_file = None):
        
        ## STORING
        self.path_xvg = path_xvg
        
        ## READING FILE 
        self.data_full, self.data_extract = import_tools.read_xvg(self.path_xvg)
        
        ## VARIABLE EXTRACTION
        self.define_variables()
        
        ## ANALYSIS
        self.avg_rmsf = np.mean(self.output_rmsf_per_atom)
        
        return
    
    ## EXTRACTION OF VARIABLES
    def define_variables(self, ):
        '''
        The purpose of this function is to define variables based on extracted results
        INPUTS:
            self.data_extract: [list] 
                extracted data in a form of a list (i.e. no comments)
        OUTPUTS:
            self.output_rmsf_per_atom: [np.array, shape=(num_atoms, 1)] 
                rmsf per atom average across time frame
        '''
        self.output_rmsf_per_atom = np.array([ x[1] for x in self.data_extract]).astype("float") # major, middle, and minor axis
        

### FUNCTION TO COMPUTE RMSF
def main_compute_gmx_rmsf(path_to_sim,
                          input_prefix,
                          rewrite = False,
                          output_suffix = None,
                          center_residue_name = 'AUNP',
                          rmsf_inputs={},
                          **args):
    '''
    Main function to compute ligand RMSF information. This function will 
    utilize GMX RMSF functionality to quickly compute RMSF.
    INPUTS:
        path_to_sim: [str]
            path to the simulations
        input_prefix: [str]
            input prefix
        output_suffix: [str]
            output prefix
        func_inputs: [dict]
            dictionary to compute_gmx_rmsf functionality
        **args:
            arguments for centering gold nanoparticles, etc.
    OUTPUTS:
        rmsf: [obj]
            object containing RMSF information
        func_inputs: [dict]
            dictionary of the function inputs. This may be useful in recovering 
            the residue name. (though probably not necessary)
    '''
    
    ## GETTING TRAJECTORY
    output_gro, output_xtc = generate_gro_xtc_with_center_AUNP(path_to_sim = path_to_sim,
                                                               input_prefix = input_prefix,
                                                               output_suffix = output_suffix,
                                                               rewrite = rewrite,
                                                               center_residue_name = center_residue_name,
                                                               **args
                                                               )
    ## LOADING GRO FILE
    path_gro_file = os.path.join(path_to_sim, output_gro)
    
    ## CHECKING PATH TO GRO FILE
    if path_gro_file is not None:
        traj = md.load(path_gro_file)
    else:
        traj = None
    
    ## GETTING LIGAND NAMES
    ligand_names = get_ligand_names_within_traj(traj)

    ## DEFINING OUTPUT PREFIX
    output_prefix = os.path.splitext(output_gro)[0]
    
    ## DEFINING INPUTS
    func_inputs = {
            'input_gro_file': input_prefix + ".gro",
            'input_tpr_file': output_prefix + ".tpr",
            'input_xtc_file': output_prefix + ".xtc",
            'index_file':     output_prefix + ".ndx",
            'input_top_file': "sam.top",
            'top_comment_out': [ "lig_posre.itp", 
                                 "sulfur_posre.itp",
                                 "gold_posre.itp"],
            'input_mdp_file': "nvt_double_prod_gmx5_charmm36_frozen_gold.mdp",
            'rewrite': rewrite,
            }

    ## STORING LIGAND NAMES
    func_inputs['rmsf_residue_name'] = ligand_names
    
    ## UPDATING INPUTS
    for each_key in rmsf_inputs:
        func_inputs[each_key]  = rmsf_inputs[each_key]
    
    ## GENERATING TRJCONV
    trjconv_output = convert_with_trjconv(wd = path_to_sim)
    output_pdb_file, output_rmsf_file, output_rmsf_vs_time_file = trjconv_output.compute_gmx_rmsf(**func_inputs)
    
    ## EXTRACTION OF RMSF FILE
    rmsf = compute_gmx_rmsf(path_xvg= os.path.join(path_to_sim, output_rmsf_file),
                            )
    
    return rmsf, func_inputs

#%% MAIN SCRIPT
if __name__ == "__main__":
    
    ## DEFINING MAIN DIRECTORY
    main_dir = check_server_path(r"R:\scratch\nanoparticle_project\simulations")
    
    ### DIRECTORY TO WORK ON    
    simulation_dir=r"20200212-Debugging_GNP_spring_constants_heavy_atoms"
    
    ## DEFINING SPECIFIC DIR
    specific_dir = r"MostlikelynpNVTspr_1000-EAM_300.00_K_2_nmDIAM_C11OH_CHARMM36jul2017_Trial_1_likelyindex_1"
    
    ## DEFINING PATH
    path_to_sim = os.path.join(main_dir,
                               simulation_dir,
                               specific_dir)
    
    ## DEFINING INPUT GRO AND XTC
    input_prefix = "sam_prod"
    
    ## DEFINING OUTPUT PREFIX
    output_prefix="sam_prod-center_pbc_mol"
    
    ## COMPUTING RMSF
    rmsf, func_inputs= main_compute_gmx_rmsf(path_to_sim = path_to_sim,
                                             input_prefix = input_prefix)
    

    
    
    