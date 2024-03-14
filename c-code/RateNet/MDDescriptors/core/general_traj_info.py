# -*- coding: utf-8 -*-
"""
general_traj_info.py
The purpose of this script is to get general trajectory information (e.g. box length, etc.)
This is important for publishable data that requires a supplementary information about the residues, atoms, etc.

Written by: Alex K. Chew (alexkchew@gmail.com, 08/08/2018)

FUNCTIONS:
    
CLASSES:
    general_traj_info: class object that can obtain general trajectory information
    
UPDATES:
    - 20180808 - AKC - Generated draft of script

"""
### IMPORTING MODULES
import MDDescriptors.core.import_tools as import_tools # Loading trajectory details
import mdtraj as md
import MDDescriptors.core.calc_tools as calc_tools
import numpy as np

##################################################################
### CLASS FUNCTION TO CALCULATE GENERAL TRAJECTORY INFORMATION ###
##################################################################
class general_traj_info:
    '''
    The purpose of this class is to take the any trajectory and calculate vital, general information about it
    INPUTS:
        traj_data: 
            Data taken from import_traj class
        verbose: [logical, default=True] 
            True if you want print statements
    OUTPUTS:
        ## BOX INFORMATION
        self.ens_volume: [float] ensemble volume
        self.ens_length: [float] ensemble box length
        
        ## RESIDUES
        self.residues: [dict] dictionary listing total number of residues
        
        ## ATOMS
        self.total_atoms_per_residue: [dict] total atoms for each residue
        self.total_atoms: [int] total number of atoms in the system
        
        ## FRAMES
        self.total_frames: [int] total frames in your system
    '''
    ## INITIALIZING
    def __init__(self, traj_data, verbose = True):
        ### PRINTING
        if verbose == True:
            print("**** CLASS: %s ****"%(self.__class__.__name__))
        
        ## DEFINING TRAJECTORY
        traj = traj_data.traj
        
        ## GETTING THE VOLUME
        self.ens_volume = calc_tools.calc_ensemble_vol(traj = traj)
        
        ## GETTING ENSEMBLE VOLUME (ASSUMING SQUARE)
        self.ens_length = np.mean(traj.unitcell_lengths)
        
        ## STORING RESIDUES
        self.residues = traj_data.residues
        ## FINDING TOTAL ATOM PER RESIDUE
        self.find_total_atom_for_each_residue(traj = traj)
        
        ## FINDING TOTAL NUMBER OF ATOMS
        self.total_atoms = len([ atom.index for atom in traj.topology.atoms])
        
        ## TOTAL FRAMES
        self.total_frames = len(traj)        
        
        ## PRINTING SUMMARY
        if verbose == True:
            self.print_summary()
        
    ## PRINTING FUNCTION
    def print_summary(self):
        ''' This function prints a summary '''
        print("TOTAL FRAMES: %d" %(self.total_frames) )
        print("ENSEMBLE VOLUME (nm^3): %.1f"%(self.ens_volume) )
        print("ENSEMBLE LENGTH (nm): %.1f"%(self.ens_length) )
        print("TOTAL ATOMS: %d"%(self.total_atoms) )
        print("RESIDUES: %s"%(self.residues) )
        print("TOTAL ATOMS PER RESIDUE: %s"%(self.total_atoms_per_residue) )
        
    ## FUNCTION TO FIND THE TOTAL ATOM FOR EACH RESIDUE
    def find_total_atom_for_each_residue(self, traj):
        '''
        The purpose of this function is to find the total number of atom per residue
        INPUTS:
            self: class object
            traj: trajectory from md.traj
        OUTPUTS:
            self.total_atoms_per_residue: [dict] total atoms for each residue
        '''
        ## CREATING BLANK DICTIONARY
        self.total_atoms_per_residue = {}
        ## LOOPING THROUGH EACH RESIDUE NAME, FINDING TOTAL NUMBER OF ATOMS
        for each_residue_name in self.residues.keys():
            self.total_atoms_per_residue[each_residue_name] = calc_tools.find_total_atoms(traj = traj, resname = each_residue_name)[0]
        return
        

#%% MAIN SCRIPT
if __name__ == "__main__":
    
    ### DIRECTORY TO WORK ON
    # analysis_dir=r"180328-ROT_SPHERICAL_TRANSFER_LIGANDS_6nm" # Analysis directory
    analysis_dir=r"SELF_ASSEMBLY_FULL" # Analysis directory
    # category_dir="spherical" # category directory
    category_dir="EAM" # category directory spherical
    # specific_dir="spherical_6_nmDIAM_300_K_2_nmEDGE_5_AREA-PER-LIG_4_nm_300_K_butanethiol" # Directory within analysis_dir r"mdRun_363.15_6_nm_tBuOH_100_WtPercWater_spce_Pure"
    diameter="4"
    specific_dir= category_dir + "_"+ diameter +"_nmDIAM_300_K_2_nmEDGE_5_AREA-PER-LIG_4_nm_300_K_butanethiol_Trial_1" # Directory within analysis_dir r"mdRun_363.15_6_nm_tBuOH_100_WtPercWater_spce_Pure"
    # 180328-ROT_SPHERICAL_TRANSFER_LIGANDS_6nm/spherical
    ### DEFINING PATH
    path2AnalysisDir=r"R:\scratch\nanoparticle_project\analysis\\" + analysis_dir + '\\' + category_dir + '\\' + specific_dir # PC Side
    
    ### DEFINING FILE NAMES
    gro_file=r"gold_ligand_equil.gro" # Structural file
    xtc_file=r"gold_ligand_equil_whole.xtc" # Trajectory file
    
    ### LOADING TRAJECTORY
    traj_data = import_tools.import_traj( directory = path2AnalysisDir, # Directory to analysis
                 structure_file = gro_file, # structure file
                  xtc_file = xtc_file, # trajectories
                  )
    #%%
    ### DEFINING INPUT DATA
    input_details={ 'verbose': True
                    }
    
    ### FINDING SELF ASSEMBLY STRUCTURE
    traj_info = general_traj_info(traj_data, **input_details)