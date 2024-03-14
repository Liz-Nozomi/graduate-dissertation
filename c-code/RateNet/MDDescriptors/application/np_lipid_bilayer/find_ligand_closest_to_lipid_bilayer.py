#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
find_ligand_closest_to_lipid_bilayer.py
The purpose of this script is to find the ligand that is nearest to the lipid bilayer. 
This script will then output the deatils of the ligand that is the closest. For 
simplicity, it will also compute the z-distance between the ligand and the 
center of mass of the lipid bilayer. 

Written by: Alex K. Chew (04/22/2020)

FUNCTIONS:
    plot_mayavi_np_with_last_heavy_atoms:
        plots mayavi nanoparticle with the last heavy atoms.

"""

import os
import numpy as np

## IMPORTING TOOLS
import MDDescriptors.core.import_tools as import_tools # Loading trajectory details

## CALC TOOLS
import MDDescriptors.core.calc_tools as calc_tools # Loading trajectory details

## PICKLE FUNCTIONS
import MDDescriptors.core.pickle_tools as pickle_funcs

## IMPORTING STRUCTURE
from MDDescriptors.application.nanoparticle.nanoparticle_structure import nanoparticle_structure
from MDDescriptors.application.nanoparticle.core.find_ligand_residue_names import get_ligand_names_within_traj

## PLOTTING TOOLS
import MDDescriptors.core.plot_tools as plot_funcs

## CHECK TESTING TOOLS
from MDBuilder.core.check_tools import check_testing ## CHECKING PATH FOR TESTING

### FUNCTION TO PLOT NP WITH POINTS
def plot_mayavi_np_with_last_heavy_atoms(traj_data,
                                         last_lig_heavy_atom_index = None,
                                         frame = 0):
    '''
    This function plots the nanoparticle with the last heavy atoms as green 
    dots. 
    INPUTS:
        traj_data: [obj]
            trajectory data object
        last_lig_heavy_atom_index: [np.array]
            last lig heavy atom index. If None, we will only plot the nanoparticle.
        frame: [int]
            frame to plot
    OUTPUTS:
        fig: [object]
            figure object
    '''

    ## PLOTTING MAYAVI NANOPARTICLE
    fig = plot_funcs.plot_mayavi_nanoparticle(traj_data = traj_data,
                                              frame = frame )
    
    ## PLOTTING LAST HEAVY ATOMS
    if last_lig_heavy_atom_index is not None:
        atom_index = last_lig_heavy_atom_index
        traj = traj_data.traj
        
        ## DEFINING POSITIONS
        x_pos = traj.xyz[frame, atom_index, 0]
        y_pos = traj.xyz[frame, atom_index, 1]
        z_pos = traj.xyz[frame, atom_index, 2]
        
        shape_atom_size = 3 * np.ones(len(atom_index))
        color = tuple(plot_funcs.COLOR_CODE_DICT['green'])
        
        from mayavi import mlab
        
        ## PLOTTING POINTS
        mlab.points3d(
                    x_pos, 
                    y_pos, 
                    z_pos,
                    shape_atom_size,
                    figure = fig,
                    scale_factor=.25, 
                    opacity=1.0,
                    mode = "sphere", # "sphere",
                    color = color
                     )

    return fig


################################################
### CLASS FUNCTION TO COMPUTE CLOSEST LIGAND ###
################################################
class find_lig_closest_to_lipid_membrane:
    '''
    The purpose of this object is to find the ligand closest to the lipid membrane. 
    INPUTS:
        traj_data: [obj]
            trajectory object
        np_structure: [obj]
            nanoparticle structure object
        frame: [int]
            frame index to compute closest lig for
    OUTPUTS:
        self.last_lig_heavy_atom_index: [np.array]
            numpy array of ligand heavy atom index
        self.last_lig_heavy_atom_name: [np.array]
            numpy array of unique atom names
            
        self.lm_com: [np.array] 
            lipid membrnae center of mass when taking account all atoms
        self.lm_com_z: [float]
            lipid membrane z dimension center of mass
        
        self.lig_closest_idx: [int]
            ligand closest index (out of all ligands)
        self.lig_closest_atom_index: [int]
            atom index for the last heavy atom that is the closest
        self.lig_distance_to_lm_com_z: [float]
            ligand last heavy atom distance to the center of mass of lipid membrane
        
    FUNCTIONS:
        print_summary: 
            prints summary of the results
        get_np_last_atom_index:
            gets nanoparticle last heavy atom index
        compute_lm_com_z:
            computes lipid membrane center of mass in the z-axis
        compute_closest_lig_index:
            computes ligand that is cloesst to the lipid membrane
        plot_mayavi_lig_heavy_atoms:
            plots nanoparticle last heavy atom index for debugging purposes
    PLOTTING USAGE:
        Function that plots ligand heavy atoms:
            np_closest.plot_mayavi_lig_heavy_atoms(traj_data = traj_data)
            
    '''
    def __init__(self,
                 traj_data,
                 np_structure,
                 lm_resname = 'DOPC',
                 frame = 0):
        ## STORING STRUCTURE
        self.np_structure = np_structure
        self.frame = frame
        self.lm_resname = lm_resname
        
        ## FINIDNG NP LAST LIGAND HEAVY ATOM INDEX
        self.get_np_last_atom_index(traj_data = traj_data)
        
        ## GETTING LIPID MEMBRANE CENTER OF MASS
        self.compute_lm_com_z(traj_data = traj_data)
        
        ## GETTING CLOSEST LIGAND DISTNACE
        self.compute_closest_lig_index(traj_data = traj_data)
        
        ## PRINTING SUMMARY
        self.print_summary()
        
        return
    
    ### FUNCTION TO PRINT SUMMARY
    def print_summary(self):
        ''' This function prints the summary '''
        print("--- SUMMARY ---")
        print("Unique last heavy atom names: %s"%( np.array2string(self.last_lig_heavy_atom_name, separator = ',') ) )
        print("Total number of ligands: %d"%( len( self.np_structure.ligand_heavy_atom_index ) ) )
        print("-------")
        print("Lipid membrane COM: %s"%(self.lm_com) )
        print("Ligand closest index: %d"%( self.lig_closest_idx ) )
        print("Ligand atom index: %d"%(self.lig_closest_atom_index) )
        print("Ligand z distance to lipid membrane (nm): %.5f"%( self.lig_distance_to_lm_com_z ))
        
        return
    
    ### FUNCTION TO GET HEAVY ATOM INDEX
    def get_np_last_atom_index(self,
                               traj_data):
        '''
        This function gets the nanoparticle last ligand heavy index.
        INPUTS:
            traj_data: [obj]
                trajectory object
        OUTPUTS:
            self.last_lig_heavy_atom_index: [np.array]
                numpy array of ligand heavy atom index
            self.last_lig_heavy_atom_name: [np.array]
                numpy array of unique atom names
        '''
        
        ## GETTING LAST LIGAND HEAVY ATOM INDEX FOR EACH LIGAND
        self.last_lig_heavy_atom_index = np.array([ each_lig_atom_index[-1] 
                                               for each_lig_atom_index in self.np_structure.ligand_heavy_atom_index])
        
        self.last_lig_heavy_atom_name_list = [traj_data.topology.atom(each_index).name for each_index in self.last_lig_heavy_atom_index]
        ## GETTING LAST LIGAND HEAVY ATOM NAME
        self.last_lig_heavy_atom_name = np.unique(self.last_lig_heavy_atom_name_list)
        
        return
    
    ### FUNCTION TO GET THE LM COM
    def compute_lm_com_z(self, traj_data):
        '''
        This function computes the lipid membrane center of mass. 
        INPUTS:
            traj_data: [obj]
                trajectory object
        OUTPUTS:
            self.lm_com: [np.array] 
                lipid membrnae center of mass when taking account all atoms
            self.lm_com_z: [float]
                lipid membrane z dimension center of mass
        '''
        ## GETTING LM COM
        self.lm_com = calc_tools.find_center_of_mass(traj = traj_data.traj,
                                                residue_name = self.lm_resname,
                                                combine_all_res = True)
        ## GETTING Z DIMENSION
        self.lm_com_z = self.lm_com[self.frame][0][-1]
        return
        
    ### FUNCTION TO COMPUTE CLOSEST LIGAND DISTANCE
    def compute_closest_lig_index(self,
                                  traj_data):
        '''
        The purpose of this function is to compute the closest ligand 
        to the lipid membrane using the center of mass z-distance and the z-distance 
        of the last heavy atom of the ligands. 
        INPUTS:
            traj_data: [obj]
                trajectory object
        OUTPUTS:
            self.lig_closest_idx: [int]
                ligand closest index (out of all ligands)
            self.lig_closest_atom_index: [int]
                atom index for the last heavy atom that is the closest
            self.lig_distance_to_lm_com_z: [float]
                ligand last heavy atom distance to the center of mass of lipid membrane
        '''
        
        ## GETTING ALL Z DISTANCES FOR NP
        np_lig_z_distances = traj_data.traj.xyz[self.frame,self.last_lig_heavy_atom_index,-1]
        
        ## GETTING DIFFERENCE
        diff = np_lig_z_distances - self.lm_com_z
        
        ## GETTING MIN
        self.lig_closest_idx = np.argmin(diff)
        
        ## FINDING LIGAND HEAVY ATOM INDEX
        self.lig_closest_atom_index = self.last_lig_heavy_atom_index[self.lig_closest_idx]
        self.lig_distance_to_lm_com_z = diff[self.lig_closest_idx]
        self.lig_atom_name = self.last_lig_heavy_atom_name_list[self.lig_closest_idx]
        
        return
    
    ### FUNCTION TO PLOT MAYAVI
    def plot_mayavi_lig_heavy_atoms(self,
                                    traj_data):
        ''' Function that plots nanoparticle with last heavy atoms '''
        fig = plot_mayavi_np_with_last_heavy_atoms(traj_data = traj_data,
                                                   last_lig_heavy_atom_index = self.last_lig_heavy_atom_index,
                                                   frame = self.frame)
        return fig
    
### PRINT SUMMARY FILE
def print_summary_file(path_summary, 
                       np_closest):
    '''
    This function prints a summary file that we could use in the terminal.
    INPUTS:
        path_summary: [str]
            path to summary file
        np_closest: [obj]
            nanoparticle closest object
    '''
    print("Printing summary file: %s"%(path_summary))
    with open(path_summary, 'w') as f:
        f.write("CLOSEST_LIG_INDEX: %d\n"%( np_closest.lig_closest_idx ) )
        f.write("CLOSEST_ATOM_INDEX: %d\n"%( np_closest.lig_closest_atom_index  ))
        f.write("ATOM_DISTANCE_TO_LM: %.5f\n"%( np_closest.lig_distance_to_lm_com_z ) )        
        f.write("ATOM_NAME: %s\n"%( np_closest.lig_atom_name ) )
    return

### MAIN FUNCTION TO COMPUTE LIPIDS CLOSEST TO LIPID MEMBRANE
def main_compute_lig_closest_to_lm(path_to_sim,
                                   gro_file,
                                   itp_file,
                                   frame = 0,
                                   lm_resname = "DOPC",
                                   output_summary="closest_lig.summary",
                                   output_pickle ="closest_lig.pickle",
                                   ):
    '''
    Main function that computes ligand closest to lipid membrane.
    INPUTS:
        path_to_sim: [str]
            path to the simulation
        gro_file: [str]
            gro file that would be loaded
        frame: [int, default = 0]
            frame that is loaded
        itp_file: [str]
            itp file
        lm_resname: [str]
            lipid membrane residue name
        output_summary: [str]
            summary file name
        output_pickle: [str]
            pickle file name
    OUTPUTS:
        np_closest: [obj]
            object for nanoparticle closest
    '''
    
    ## LOADING THE GRO FILE
    traj_data = import_tools.import_traj(directory = path_to_sim, # Directory to analysis
                                         structure_file = gro_file, # structure file
                                         xtc_file = gro_file, # trajectories
                                         )
    
    ## FINDING ALL LIGAND NAMES
    ligand_names = get_ligand_names_within_traj(traj_data.traj)
    
    ## DEFINING INPUTS
    inputs_structure={
            'traj_data': traj_data,
            'ligand_names': ligand_names,
            'itp_file': itp_file,
            'separated_ligands': False,
            'structure_types': None,
            }
    
    ## GETTING NANOPARTICLE STRUCTURE
    np_structure = nanoparticle_structure(**inputs_structure)
    
    ## DEFININNG INPUTS
    inputs_to_np_closest = {
            'traj_data': traj_data,
            'np_structure': np_structure,
            'frame': frame,
            'lm_resname' : lm_resname,
            }

    ## ANALYZING
    np_closest = find_lig_closest_to_lipid_membrane(**inputs_to_np_closest)
    
    ## PLOTTING
    '''
    np_closest.plot_mayavi_lig_heavy_atoms(traj_data = traj_data)
    '''
    
    ## WRITING SUMMARY
    path_summary = os.path.join(path_to_sim,
                                output_summary)
    path_pickle = os.path.join(path_to_sim,
                                output_pickle)
    ## WRITING
    print_summary_file(path_summary = path_summary,
                       np_closest = np_closest)
    
    ## STORING PICKLE
    pickle_funcs.pickle_results(results = np_closest,
                                pickle_path = path_pickle,
                                verbose = True)
    
    return np_closest
#%% MAIN SCRIPT
if __name__ == "__main__":
    
    ## CHECKING TESTING
    testing = check_testing()
    
    if testing is True:
        ## DEFINING PATH TO GRO
        path_to_sim=r"/Volumes/akchew/scratch/nanoparticle_project/nplm_sims/20200422-pulling_unbiased/NPLMpulling_unb-5.100_2000-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1"
        
        ## DEFINING GRO
        gro_file="nplm.gro"
        
        ## DEFINING ITP FILE
        itp_file="sam.itp"
        
        ## DEFINING OUTPUT SUMMARY
        output_summary="closest_lig.summary"
        output_pickle ="closest_lig.pickle"
        
        ## DEFINING LIPID MEMBRANE RESNAME
        lm_resname = 'DOPC'
    else:
        from optparse import OptionParser # Used to allow commands within command line
        # Adding options for command line input (e.g. --ligx, etc.)
        use = "Usage: %prog [options]"
        parser = OptionParser(usage = use)
        ## DEFINING INPUT FOLDER
        parser.add_option("--path_sim", dest="path_to_sim", action="store", type="string", help="Path to simulation", default=".")
        parser.add_option("--gro_file", dest="gro_file", action="store", type="string", help="Gro file", default=".")
        parser.add_option("--itp_file", dest="itp_file", action="store", type="string", help="Nanoparticle itp file", default=".")
        parser.add_option("--summary", dest="output_summary", action="store", type="string", help="Output summary file", default=".")
        parser.add_option("--lm_resname", dest="lm_resname", action="store", type="string", help="Lipid membrane residue name", default=".")
        parser.add_option("--pickle", dest="output_pickle", action="store", type="string", help="Output pickle file", default=".")
        ## COLLECTING ARGUMENTS
        (options, args) = parser.parse_args() # Takes arguments from parser and passes them into "options" and "argument"
        
        path_to_sim = options.path_to_sim
        gro_file = options.gro_file
        itp_file = options.itp_file
        output_summary = options.output_summary
        output_pickle = options.output_pickle
        lm_resname = options.lm_resname

    ## DEFINING FUNCTION INPUTS
    func_inputs ={
            'path_to_sim': path_to_sim,
            'gro_file': gro_file,
            'lm_resname': lm_resname,
            'itp_file': itp_file,
            'output_pickle': output_pickle,
            'output_summary': output_summary,
            }
    
    ## COMPUTING NANOPARTICLE CLOSESTS
    np_closest = main_compute_lig_closest_to_lm(**func_inputs)
    
        
    