#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
divide_lipid_membrane_leaflets.py

The purpose of this script is to load a gro file of a system containing a 
lipid membrane, then find all atom indices on the top and bottom leaflet. 

We assume that your DOPC lipid membrane is in-tact and not divided between the 
periodic boundary condition. 

INPUTS:
    - gro file
    - residue name of the lipid membrane
    
OUTPUTS:
    - index information for top and bottom leaflet
    
ALGORITHM:
    - Load the gro file with md.traj
    - Find all DOPC lipid membranes
    - Locate only the heavy atoms of DOPC
    - Find center of mass of DOPC
    - Identify top and bottom leaflet residues
    - Check to see if top and bottom are the same number -- good way to see if 
    we have correctly divided the groups
    - Plot top and bottom leaflet to visualize the different groups

Written by: Alex K. Chew (05/15/2020)
"""
import os
import mdtraj as md
import numpy as np
import time
from optparse import OptionParser # Used to allow commands within command line

## READING FILE AS LINES
from MDDescriptors.core.read_write_tools import import_index_file

## CALC TOOLS
import MDDescriptors.core.calc_tools as calc_tools

from MDDescriptors.application.np_lipid_bilayer.compute_NPLM_contacts import get_nplm_heavy_atom_details
from MDDescriptors.application.np_lipid_bilayer.compute_NPLM_contacts_extract import generate_rotello_np_groups, generate_lm_groups

## CHECK TESTING FUNCTION
from MDBuilder.core.check_tools import check_testing ## CHECKING PATH FOR TESTING

import MDDescriptors.core.plot_tools as plot_funcs
## SETTING DEFAULTS
plot_funcs.set_mpl_defaults()

### CLASS FUNCTION TO DIVID LIPID BILAYERS
class divide_lipid_bilayers:
    '''
    The purpoes of this code is to divide the lipid bilayers in to upper and 
    lower leaflets. The idea is then to compute COM of the lipid bilayer and 
    use that as a way of distinguishing top and bottom. The COM of each 
    lipid residue will be computed as well. This code assumes that the lipid 
    membrane is not broken -- so it may be useful to define the trajectory 
    such that the lipids are not across a periodic boundary condition. 
    
    INPUTS:
        traj: [obj]
            trajectory object
        lm_residue_name: [str]
            lipid membrane residue name
        
    OUTPUTS:
        
        ATOM INDICES OF TOP AND BOTTOM LEAFLET        
        self.top_atom_indexes: [list]
            list of atom indices on the top (no hydrogens)
        self.bot_atom_indexes: [list]
            list of atom indices of the bottom (no hydrogens)
    
    NOTES:
        - You could plot the top and bottom leaflet by:
            ## PLOTTING TOP AND BOTTOM LEAFLET
            divided_lm.plot_mayavi_top_and_bottom(traj = traj)
    '''
    def __init__(self,
                 traj,
                 lm_residue_name = "DOPC"
                 ):
        ## STORING INPUTS
        self.lm_residue_name = lm_residue_name
        
        ## COMPUTING LM CENTER OF MASS
        self.com_each_residue, self.com_all_residues = self.compute_lm_com(traj = traj)
        
        ## COMPUTING INDICES THAT ARE ABOVE AND BELOW
        self.indices_above, self.indices_below = self.compute_indices_above_and_below(com_each_residue = self.com_each_residue,
                                                                                      com_all_residues = self.com_all_residues)
        
        ## GETTING RESIDUE INDEX
        self.get_residue_indices(traj = traj)
        
        ## GETTING TOTAL TOP AND BOTOTM
        self.n_top = len(self.top_res_index)
        self.n_bot = len(self.bot_res_index)
        
        ## PRINTING SUMMARY
        self.print_summary()
        
        return
    
    ## PRINTING SUMMARY
    def print_summary(self):
        ''' This function prints the summary '''
        print("----- SUMMARY FOR divide_lipid_bilayers  -----")
        print("Lipid membrane residue name: %s"%(self.lm_residue_name) )
        print("Total residues: %d"%(len(self.residue_index)) )
        print("Total residues on top leaflet: %d"%(self.n_top) )
        print("Total residues on bottom leaflet: %d"%(self.n_bot) )
        
        ## THROW ERROR IF TOP AND BOTTOM ARE NOT THE SAME
        if self.n_top != self.n_bot:
            print("Warning! Top and bottom leaflets do not have the same residue count")
            print("This may be normal if you have an asymmetric bilayer")
            print("If not, then please check the divide_lipid_bilayers code!")
            print("Pausing 5 seconds so you can see this message")
            time.sleep(5)
        
        return
    
    ## COMPUTE CENTER OF MASS
    def compute_lm_com(self, traj):
        '''
        This function computes the center of mass of the lipid membranes
        INPUTS:
            traj: [obj]
                trajectory object
        OUTPUTS:
            com_each_residue: [np.array. shape= (N,3)]
                center  of mass of each residue
            com_all_residues: [np.array, shape = (1,3)]
                center of mass of all residues combined
        '''
        
        ## COMPUTING CENTER OF MASS OF EACH RESIDUE
        com_each_residue = calc_tools.find_center_of_mass(traj = traj, 
                                                        residue_name = self.lm_residue_name,
                                                        combine_all_res = False)[0]
    
        
        ## COMPUTING CENTER OF MASS TOGETHER
        com_all_residues = calc_tools.find_center_of_mass(traj = traj, 
                                                        residue_name = self.lm_residue_name,
                                                        combine_all_res = True)[0][0]
        
        return com_each_residue, com_all_residues
    
    ## STATIC FUNCTION TO COMPUTE INDICES
    @staticmethod
    def compute_indices_above_and_below(com_each_residue,
                                        com_all_residues):
        '''
        This function computes the indices above and below the center of mass. 
        INPUTS:
            com_each_residue: [np.array. shape= (N,3)]
                center  of mass of each residue
            com_all_residues: [np.array, shape = (1,3)]
                center of mass of all residues combined
        OUTPUTS:
            indices_above: [np.array]
                indices that are above the center of mass
            indices_below: [np.array]
                indices that are below the center of mass
        '''
        
        ## FINDING ALL LIPIDS THAT ARE ABOVE
        indices_above = np.where(com_each_residue[:,-1] > com_all_residues[-1])[0]
        indices_below = np.where(com_each_residue[:,-1] < com_all_residues[-1])[0]
        
        return indices_above, indices_below
        
    
    ### FUNCTION TO GET THE TOP AND BOTTOM INDICES
    def get_residue_indices(self, traj):
        '''
        This function gets the indices using the indices above and below.
        INPUTS:
            traj: [obj]
                trajectory object
        OUTPUTS:
            self.top_atom_indexes: [list]
                list of atom indices on the top (no hydrogens)
            self.bot_atom_indexes: [list]
                list of atom indices of the bottom (no hydrogens)
        '''
        ## FINDING ATOM AND RESIDUE INDICES
        self.residue_index, self.atom_index = calc_tools.find_residue_atom_index(traj = traj, 
                                                                       residue_name = self.lm_residue_name, 
                                                                           atom_names = None)
        
        ## CONVERTING RES INDEX TO ARRAY
        self.residue_index = np.array(self.residue_index)
        
        ## LOOPING TO GET RESIDUE INDEX OF TOP AND BOTTOM
        self.top_res_index = self.residue_index[self.indices_above]
        self.bot_res_index = self.residue_index[self.indices_below]
        
        ## ATOM INDEXES OF THE TOP AND BOTTOM
        self.top_atom_indexes = np.array([ atom.index for each_res_index in self.top_res_index for atom in traj.topology.residue(each_res_index).atoms if atom.element.symbol != "H" ])
        self.bot_atom_indexes = np.array([ atom.index for each_res_index in self.bot_res_index for atom in traj.topology.residue(each_res_index).atoms if atom.element.symbol != "H" ])
        
    ### FUNCTION TO PLOT MAYAVI ATOM
    def plot_mayavi_top_and_bottom(self, traj):
        '''
        This function plots the top and bottom lipid membranes as red and blue, respectively. 
        The idea is to make sure that the selection of lipid membranes are correct 
        between top and bottom leaflet.
        INPUTS:
            traj: [obj]
                trajectory object
        OUTPUTS:
            fig: [obj]
                mayavi figure
        '''
        ## GENERATING PLOT FOR TOP
        fig = plot_funcs.plot_mayavi_atoms(traj = traj,
                                           atom_index = self.top_atom_indexes,
                                           desired_atom_colors = 'red')
        
        
        ## GENERATING PLOT FOR BOTTOM
        fig = plot_funcs.plot_mayavi_atoms(traj = traj,
                                           atom_index = self.bot_atom_indexes,
                                           desired_atom_colors = 'blue',
                                           figure = fig)
        
        return fig
    
    
### FUNCTION TO GET THE TOP AND BOTTOM INDICES
def compute_indices_divided_lm_using_groups(traj,
                                            divided_lm,
                                            lm_groups,):
    '''
    The purpose of this function is to divide the top and bottom indices
    by head groups, tail groups, and so forth.
    INPUTS:
        traj: [obj]
            trajectory object
        divided_lm: [obj]
            divided lipid membrane object
        lm_groups: [dict]
            dictionary of lipid membrane groups
            
    OUTPUTS:
        top_bot_dict: [dict]
            top and bottom dictionary containing each group atom index, e.g.
                    
                {'TOP': {'HEADGRPS': array([ 8692.,  8693.,  8696., ..., 35622., 35623., 35624.]),
                  'TAILGRPS': array([ 8715.,  8716.,  8719., ..., 35730., 35733., 35736.])},
                 'BOT': {'HEADGRPS': array([35740., 35741., 35744., ..., 62670., 62671., 62672.]),
                  'TAILGRPS': array([35763., 35764., 35767., ..., 62778., 62781., 62784.])}}
        
    '''

    ## GENERATING LIST OF INDICES
    input_indices_dict = {
            'TOP': divided_lm.top_atom_indexes,
            'BOT': divided_lm.bot_atom_indexes,
            }
    
    ## CREATING A DICT
    top_bot_dict = {}
    
    ## LOOPING
    for divided_key in input_indices_dict.keys():
    
        ## GENERATING A LIST FOR EACH INDEX
        atom_dict_storage = { each_key: np.array([]).astype(int) for each_key in lm_groups.keys()}
        
        ## DEFINING ATOM INDICES
        atom_indices = input_indices_dict[divided_key]
        
        ## LOOPING FOR THE TOP
        for atom_idx in atom_indices:
            ## SEEING IF EITHER HEAD OR TAIL GROUP
            atom_name = traj.topology.atom(atom_idx).name
            ## LOOPING
            for each_key in lm_groups:
                atom_list = lm_groups[each_key]
                if atom_name in atom_list:
                    atom_dict_storage[each_key] = np.append( atom_dict_storage[each_key], atom_idx )
                    
        ## STORING
        top_bot_dict[divided_key] = atom_dict_storage.copy()
    return top_bot_dict

### MAIN FUNCTION TO OUTPUT INDEX FILE
def main_get_lm_split_ndx(path_to_sim,
                          gro_file,
                          index_file,
                          lm_residue_name = "DOPC"):
    '''
    Main function to get the index groups for lipid membrane top and bottom 
    of the bilayer. It simply uses center of mass z- to compute the top and 
    bottom portions. Then, we generate a new index file for it.
    INPUTS:
        path_to_sim: [str]
            path to simulation
        gro_file: [str]
            gro file to load
        index_file: [str]
            indexing file that you want to modify
        lm_residue_name: [str]
            lipid membrane residue name
    OUTPUTS:
        top_bot_dict: [dict]
            top and bottom indices for head/tail groups
        ndx_file: [str]
            index file
    '''
       
    ## LOADING FILE
    path_to_gro = os.path.join(path_to_sim,
                               gro_file
                               )
    ## LOADING TRAJECTORY
    traj = md.load(path_to_gro)
    
    ## COMPUTING DIVIDED GROUPS
    divided_lm = divide_lipid_bilayers(traj = traj,
                                       lm_residue_name = "DOPC")
    
    ''' Good way to visualize top and bottom head groups.
    ## GENERATING PLOT FOR TOP
    fig = plot_funcs.plot_mayavi_atoms(traj = traj,
                                       atom_index = list(top_bot_dict['TOP']['HEADGRPS']),
                                       desired_atom_colors = 'red')
    
    
    ## GENERATING PLOT FOR BOTTOM
    fig = plot_funcs.plot_mayavi_atoms(traj = traj,
                                       atom_index = list(top_bot_dict['BOT']['HEADGRPS']),
                                       desired_atom_colors = 'blue',
                                       figure = fig)
    '''
    
    ## GETTING LIPID MEMBRANE HEAVY ATOMS AND ATOM INDEX
    lm_heavy_atom_index, lm_heavy_atom_names = get_nplm_heavy_atom_details(traj = traj,
                                                                           lm_res_name = divided_lm.lm_residue_name,
                                                                           atom_detail_type="lm")
    
    ## FINDING ATOM INDEX
    lm_groups = generate_lm_groups(traj = traj,
                                   atom_names = lm_heavy_atom_names,
                                   lm_heavy_atom_index = lm_heavy_atom_index,
                                   verbose = False,
                                   )
    
    ## GETTING TOP AND BOTTOM INDICES
    top_bot_dict = compute_indices_divided_lm_using_groups(traj = traj,
                                                           divided_lm = divided_lm,
                                                           lm_groups =lm_groups,)
    
    ## DEFINING PATH TOI NDEX
    path_to_index = os.path.join(path_to_sim,
                                 index_file)
    
    ## GETTING INDEX FILE
    ndx_file = import_index_file(path_to_index)
    
    ## LOOPING THROUGH THE DICT
    for top_n_bot_key in top_bot_dict:
        ## DEFINING DICT
        current_dict = top_bot_dict[top_n_bot_key]
        ## LOOPING THROUGH EACH GROUP
        for each_groups_key in current_dict:
            ## GETTING KEY NAME
            group_key_name = '_'.join([top_n_bot_key, each_groups_key])
            ## GETTING INDEX
            current_indices = current_dict[each_groups_key]
            ## STORING IT INTO INDEX
            ndx_file.write(index_key = group_key_name,
                           index_list = current_indices)
    
    return top_bot_dict, ndx_file


#%% RUN FOR TESTING PURPOSES 
if __name__ == "__main__":
    
    ### TURNING TEST ON / OFF
    testing = check_testing() # False if you're running this script on command prompt!!!`
    
    ## TESTING
    if testing is True:
        ## DEFINING PATH TO SIM
        path_to_sim = "/Volumes/akchew/scratch/nanoparticle_project/nplm_sims/20200120-US-sims_NPLM_rerun_stampede/US-1.3_5_0.2-NPLM-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1/4_simulations/5.100"
        # "/Volumes/akchew/scratch/nanoparticle_project/nplm_sims/20200120-US-sims_NPLM_rerun_stampede/US-1.3_5_0.2-NPLM-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1/4_simulations/4.700"
        
        ## DEFINING GRO FILE
        gro_file="nplm_prod-non-Water.gro"
        
        ## DEFINING INDEX FILE
        index_file="nplm_prod-DOPC-AUNP.ndx"
        
        ## DEFINING DOPC LIPID BILAYERS
        lm_residue_name = "DOPC"
        
    ### TESTING IS OFF
    else:
        ### DEFINING PARSER OPTIONSn
        # Adding options for command line input (e.g. --ligx, etc.)
        use = "Usage: %prog [options]"
        parser = OptionParser(usage = use)
        
        ## INPUT FOLDER
        parser.add_option("--path_to_sim", 
                          dest="path_to_sim", 
                          action="store", 
                          type="string", 
                          help="Path to simulation", default=".")
        
        parser.add_option("--gro_file", 
                          dest="gro_file", 
                          action="store", 
                          type="string", 
                          help="Gro file", default=".")
        
        parser.add_option("--index_file", 
                          dest="index_file", 
                          action="store", 
                          type="string", 
                          help="Index file", default=".")
        
        parser.add_option("--lm_residue_name", 
                          dest="lm_residue_name", 
                          action="store", 
                          type="string", 
                          help="Lipid membrane residue name", default="DOPC")
        
        ### PARSING ARGUMENTS
        (options, args) = parser.parse_args() # Takes arguments from parser and passes them into "options" and "argument"
        
        ## INPUT FILES
        path_to_sim = options.path_to_sim
        gro_file = options.gro_file
        index_file = options.index_file
        lm_residue_name = options.lm_residue_name

    ## MAIN FUNCTION THAT SPLITS THE LIPID MEMBRANE
    top_bot_dict, ndx_file = main_get_lm_split_ndx(path_to_sim,
                                                   gro_file,
                                                   index_file,
                                                   lm_residue_name =lm_residue_name)
    

    
