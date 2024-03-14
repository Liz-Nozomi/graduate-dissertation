# -*- coding: utf-8 -*-
"""
find_water_near_reactant.py
The purpose of this function is to find the frames where the water is nearest 
to the reactant oxygens.

Created on : 10/23/2019

## CREATING XTC FILE
create_xtc_test_cutoff mixed_solv_prod.tpr mixed_solv_prod.xtc 190000

Author(s):
    Alex K. Chew (alexkchew@gmail.com)

"""

## SYSTEM TOOLS
import numpy as np
import mdtraj as md # Running calculations
import sys
import os
import pandas as pd

### IMPORTING MODULES
import MDDescriptors.core.import_tools as import_tools # Loading trajectory details
import MDDescriptors.core.calc_tools as calc_tools # Loading calculation tools

## CHECKING PATH TOOLS
from MDDescriptors.core.initialize import checkPath2Server

###############################
### DEFINING CLASS FUNCTION ###
###############################
class find_nearby_solvents:
    '''
    The purpose of this function is to find nearest solvents around a reactant 
    molecule.
    INPUTS:
        traj: [obj]
            trajectory
    OUTPUTS:
        
    '''
    def __init__(self,
                 traj_data,
                 solute_name = 'PDO',
                 solvent_name = 'HOH',
                 cutoff_radius = 0.35, # nanometers
                 path_csv = 'output.csv'
                 ):
        
        ## STORING INPUTS
        self.solute_name = solute_name
        self.solvent_name = solvent_name
        self.cutoff_radius = cutoff_radius
        
        ## DEFINING TRAJECTORY
        traj = traj_data.traj
        
        ## GETTING ALL HEAVY ATOM INDEX
        self.solute_heavy_atom_index  = calc_tools.find_residue_heavy_atoms( traj = traj,
                                                                             residue_name = self.solute_name )
        
        ## GETTING ALL OXYGEN ATOMS
        self.solute_oxy_atom_index = [ each_index for each_index in self.solute_heavy_atom_index 
                                              if traj.topology.atom(each_index).element.symbol == 'O']
        
        ## GETTING OXYGAN NAMES
        self.solute_oxy_atom_names = [ traj.topology.atom(each_index).name for each_index in self.solute_oxy_atom_index]
        
        ## GETTING ALL HEAVY ATOMS OF WATER
        self.solvent_heavy_atom_index  = calc_tools.find_residue_heavy_atoms( traj = traj,
                                                                             residue_name = self.solvent_name )
    
        ## COMPUTING DISTANCES
        self.distances = calc_tools.calc_pair_distances_between_two_atom_index_list(traj = traj,
                                                                                    atom_1_index = self.solute_oxy_atom_index,
                                                                                    atom_2_index = self.solvent_heavy_atom_index )
        ## RETURNS 1001 FRAMES, 2 OXYGEN REACTANTS, 402 SOLVENT ATOMS
        
        ## GETTING PAIRS OF DISTANCES
        self.distances_within_cutoff = np.sum(self.distances < self.cutoff_radius ,axis=2)
        # RTETURNS 1001 FRAME, 2 OXYGEN REACTANTS, 1, 2, 0, ETC.
        '''e.g.
            array([[[2.259848  , 1.2647136 , 1.6502651 , ..., 1.2507606 ,
                     1.247052  , 3.1531172 ],
                    [2.3939092 , 1.5153599 , 1.946167  , ..., 1.4255291 ,
                     1.3864852 , 2.8906608 ]],
        '''
        
        ## FINDING UNIQUE COMBINATIONS
        self.unique_combinations = np.unique(self.distances_within_cutoff, axis = 0)
        # REUTURNS N COMBINATIONS, 2 OXYGEN REACTANTS,
        ''' e.g.
            array([[0, 0],
                   [0, 1],
                   ...
        '''
        ## DEFINING EMPTY DICTIONARY TO STORE COMBINATIONS
        self.combo_frames_dict = {}
        
        ## LOOPING THROUGH EACH COMBINATION
        for each_combo in self.unique_combinations:
            ## GETTING NAME
            dict_name = str(each_combo)
            ## FINDING ALL EQUIVALENCE
            indices = np.where(np.all(self.distances_within_cutoff == each_combo, axis=1))[0]
            ## STORING
            self.combo_frames_dict[dict_name] = indices[:]
        
        ## GETTING DATAFRAME
        self.df = self.create_dataframe_for_combo( combo_frames_dict = self.combo_frames_dict)
        
        ## DEFINING NAMES
        output_csv_name = 'water_near_reactant-' + self.solute_name + '-' + '_'.join(self.solute_oxy_atom_names) + '.csv'
        
        ## DEFINING PATH
        full_path_csv = os.path.join(path_csv, output_csv_name)
        
        ## PRINTING TO CSV
        self.df.to_csv(full_path_csv)
        print("Writing CSV file to: %s"%(full_path_csv) )
        
        return

    ### FUNCTION TO CREATE DATAFRAME
    @staticmethod
    def create_dataframe_for_combo(combo_frames_dict):
        '''
        The purpose of this function is to create a dataframe, such as below:
                 [0 0]  [0 1]  [0 2]  [1 0]  [1 1]  ...  [2 0]  [2 1]  [2 2]  [3 0]  [3 1]
            0        0      5     10      3      1  ...     58     95     54    224    133
            1        6     16     96      4      2  ...     73    225    760    480    228
            
        This contains all the frames that has these combinations
        INPUTS:
            combo_frames_dict: [dict]
                dictionary containing frames
        OUTPUTS:
            df: [dataframe]
                dataframe with the number of occurances. NOTE, if the frame is 
                -1, that means that from that point on, these values do not exist
        '''
        ## RENAMING
        d = combo_frames_dict
        ## CREATING DATAFRAME WITH DIFFERENT LENGTHS
        df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in d.items() ]))
        df = df.fillna(-1) # Not present
        ## CONVERT EACH COLUMN INTO INT
        for each_column in df.columns:
            df[each_column] = df[each_column].astype(int)

        return df
        
        


#%% MAIN SCRIPT
if __name__ == "__main__":
    ## DEFINING PATH TO GRO AND XTC
    path_sim=checkPath2Server(r"R:\scratch\SideProjectHuber\Simulations\190922-PDO_most_likely_gauche")
    
    ## DEFINING SIM FILE
    sim_file="Mostlikely_433.15_6_nm_PDO_10_WtPercWater_spce_dmso"
    sim_file="Mostlikely_433.15_6_nm_PDO_100_WtPercWater_spce_Pure"
    
    ## EFINING PATH
    full_path_sim = os.path.join(path_sim, sim_file)
    
    ## DEFINING GRO AND XTC FILE
    gro_file="mixed_solv_prod.gro"
    xtc_file="mixed_solv_prod.xtc"
    # "mixed_solv_prod_190000.xtc"
    
    ### LOADING TRAJECTORY
    traj_data = import_tools.import_traj( directory = full_path_sim, # Directory to analysis
                 structure_file = gro_file, # structure file
                  xtc_file = xtc_file, # trajectories
                  )
    
    #%%
    
    ### DEFINING INPUT DATA
    input_details={
                    'traj_data'                      : traj_data,
                    'solute_name'               : 'PDO',                    # Solute of interest
                    'solvent_name'              : 'HOH',   # Solvents you want radial distribution functions for
                    'cutoff_radius'             : 0.35,                     # Radius of cutoff for the RDF (OPTIONAL)
                    'path_csv'                  : full_path_sim,
                    }

    
    ## RUNNING CODE
    nearby_solvents = find_nearby_solvents( **input_details )
    
