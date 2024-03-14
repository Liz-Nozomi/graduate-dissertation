#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_np_intercalation.py

This function computes the nanoparticle intercalation into the lipid membrane.

Written by: Alex K. Chew (04/03/2020)


"""

## IMPORTING TOOLS
import os
import numpy as np
import MDDescriptors.core.import_tools as import_tools
import MDDescriptors.core.plot_tools as plot_funcs

## IMPORTING GLOBAL VARS
from MDDescriptors.application.np_lipid_bilayer.global_vars import \
    NPLM_SIM_DICT, IMAGE_LOC, PARENT_SIM_PATH, nplm_job_types

## SETTING PLOT DEFAULTS
plot_funcs.set_mpl_defaults()

FIGURE_SIZE=plot_funcs.FIGURE_SIZES_DICT_CM['1_col_landscape']

## CALC TOOLS
import MDDescriptors.core.calc_tools as calc_tools

## FUNCTIONS
from MDDescriptors.application.np_lipid_bilayer.compute_NPLM_contacts_extract import generate_rotello_np_groups, generate_lm_groups
from MDDescriptors.application.np_lipid_bilayer.compute_NPLM_contacts import get_nplm_heavy_atom_details

## IMPORTING FUNCTION FOR LIGAND NAMES
from MDDescriptors.application.nanoparticle.core.find_ligand_residue_names import get_ligand_names_within_traj

### IMPORTING NANOPARTICLE STRUCTURE CLASS
from MDDescriptors.application.nanoparticle.nanoparticle_structure import nanoparticle_structure

## IMPORTING COMMANDS 
from MDDescriptors.traj_tools.trjconv_commands import convert_with_trjconv

## IMPORTING LAST FRAME TOOL
from MDDescriptors.core.traj_itertools import get_traj_from_last_frame


## PLOTTING RESIDUE WITHIN VS. TIME
def plot_lig_intercolated_vs_time(time_array,
                                  num_residue_within,
                                  fig_size_cm=FIGURE_SIZE,
                                  line_dict={'color':'k'},
                                  fig = None,
                                  ax = None):
    '''
    This function plots the number of ligands intercolated over time.
    INPUTS:
        time_array: [np.array]
            time array
        num_residue_within: [np.array]
            number of unique ligands within the lipid bilayer
    OUTPUTS:
        fig, ax:
            figure and axis for plot
            
    '''
    ## CREATING FIGURE
    if fig is None or ax is None:
        fig, ax = plot_funcs.create_fig_based_on_cm(fig_size_cm=fig_size_cm)
        
        ## SETTING AXIS
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel("Number of ligands intercalated")
    
    ## PLOTTING
    ax.plot(time_array, num_residue_within, **line_dict)
    
    fig.tight_layout()
    
    return fig,ax

### FUNCTION TO FIND ALL ATOM INDEXES FOR EACH GROUP
def find_atom_index_for_each_group(traj,
                                   resname,
                                   group_dict):
    '''
    This function finds the atom indexes for each group and stores them 
    in a new dictionary.
    INPUTS:
        traj: [obj]
            trajectory object
        resname: [str]
            residue name
        group_dict: [dict]
            dictionary with the atom names
    OUTPUTS:
        group_dict_index: [key]
            group dictionary index
    '''
    ## CREATING NEW DICT
    group_dict_index = {}
    
    ## LOOPING THROUGH AND GETTING ATOM INDEX
    for each_key in group_dict:
        ## GETTING ATOM INDEX OF ONLY HEAD GROUPS
        atom_index = calc_tools.find_residue_atom_index(traj = traj,
                                                       residue_name = resname,
                                                       atom_names = group_dict[each_key])[1]
        
        ## FLATTENING
        atom_index = np.array(atom_index).flatten()
        
        ## STORING
        group_dict_index[each_key] = atom_index[:]
    
    return group_dict_index

###########################################
### CLASS FUNCTION TO ANALYZE LM GROUPS ###
###########################################
class analyze_lm_groups:
    '''
    This function analyzes lipid membrane details.
    INPUTS:
        traj: [obj]
            trajectory object
        lm_res_name: [str]
            residue name of the lipid membrane
    OUTPUTS:
        
    FUNCTIONS:
        find_com:
            finds com of head groups
        find_top_and_bottom_leaflet_positions:
            find top and bottom leaflet positions
    '''
    ## INITIALIZING
    def __init__(self,
                 traj,
                 lm_res_name):
        ## STORING LM RESNAME
        self.lm_res_name = lm_res_name
        
        ## GETTING INFORMATION FOR LIGAND DETAILS
        self.lm_heavy_atom_index, self.lm_heavy_atom_names = \
                get_nplm_heavy_atom_details(traj = traj,
                                            lm_res_name = lm_res_name,
                                            atom_detail_type = 'lm')
        
        ## GETTING LM GROUP
        self.lm_groups = generate_lm_groups(
                       lm_heavy_atom_index = self.lm_heavy_atom_index,
                       atom_names = self.lm_heavy_atom_names,
                       )
        
        ## GETTING LM ATOM INDEX
        self.lm_groups_atom_index = find_atom_index_for_each_group(traj = traj,
                                           resname = lm_res_name,
                                           group_dict = self.lm_groups)
        
        ## GETTING ATOM INDEX FOR ALL TAIL GROUPS
        self.lm_headgrps_atom_names=self.lm_groups['HEADGRPS']
        ## SHAPE: (392, 10) for 392 lipids and 10 atoms each
        return
    
    ## FUNCTION TO FIND COM
    def find_com(self,
                 traj):
        '''
        This function finds the lm center of mass
        '''
        ## COM CALCULATION
        center_of_mass = calc_tools.find_center_of_mass( traj = traj, 
                                                         residue_name = self.lm_res_name, 
                                                         atom_names = self.lm_headgrps_atom_names )
        
        ## SHAPE: (501, 392, 3) -- num frames, 392 lipids, 3 (xyz)
        return center_of_mass
    
    ## FUNCTION TO FIND THE TOP AND BOTTOM MEAN POSITIONS
    def find_top_and_bottom_leaflet_positions(self,
                                              traj):
        '''
        This function finds the top and bottom leaflet positions.
        INPUTS:
            traj: [obj]
                trajectory object
        OUTPUTS:
            mean_z_top: [float]
                mean value of the top leaflet z position
            mean_z_bot: [float]
                mean value of bottom leaflet z position
        '''
        ## GETTING COM
        center_of_mass = self.find_com(traj = traj)
        
        ## GETTING Z VALUES ONLY
        com_z_values  = center_of_mass[:,:,-1]
        
        ## COMPUTING Z AVG
        z_avg_com = np.mean(com_z_values, axis = 1) # RETURNS NUM_FRAMES, 1
        
        ## GETTING ALL GROUPS GREATER / BELOW THAN Z AVG (num_frames, num_residues)
        values_greater_than_z_avg = com_z_values > z_avg_com[:,np.newaxis]
        values_less_than_z_avg = com_z_values <= z_avg_com[:,np.newaxis]
        
        ## GETTING ALL RESIDUE INDEXES 
        mean_z_top = np.mean(com_z_values[np.where(values_greater_than_z_avg)])
        mean_z_bot = np.mean(com_z_values[np.where(values_less_than_z_avg)])
        
        return mean_z_top, mean_z_bot, center_of_mass

### FUNCTION TO GET THE NP WITHIN LM
def find_all_np_positions_within_lm(traj,
                                    mean_z_top,
                                    mean_z_bot):
    '''
    This function finds all nanoparticle positions within the lipid 
    membrane. It does so by finding all np positions, then looks for in-between 
    z-dimensions.
    INPUTS:
        traj: [obj]
            trajectory object
        mean_z_top: [float]
            upper threshold for the z-dimension
        mean_z_bot: [float]
            lower threshold for the z-dimension
    OUTPUTS:
        atom_index_within: [list]
            list of atom indices of the nanoparticle that is in between
        ligand_names: [list]
            list of ligand names
        np_heavy_atom_index: [list]
            heavy atom index
        np_heavy_atom_names: [list]
            np heavy atom index
    '''
    ### GETTING NP WITHIN
    ligand_names, np_heavy_atom_index, np_heavy_atom_names = get_nplm_heavy_atom_details(traj = traj,
                                                                                         atom_detail_type='np')
    ##  GETTING ALL Z POSITIONS FOR NP
    np_z_positions = traj.xyz[:,np_heavy_atom_index,-1]    ## SHAPE: (NUM FRAMES, Z POSITION)
    
    ## FINDING ALL Z POSITIONS WITHIN THE TOP AND BOTTOM
    in_btwn = np.logical_and( np_z_positions < mean_z_top, np_z_positions > mean_z_bot)
    
    ## GETTING ATOM INDEX
    atom_index_within = [ np_heavy_atom_index[each_frame_array] for each_frame_array in in_btwn]
    
    return atom_index_within, ligand_names, np_heavy_atom_index, np_heavy_atom_names

### FUNCTION TO PLOT MAYAVI IMAGE
def plot_nplm_intercalation(traj_data,
                            lm_details,
                            np_intercalation,
                            frame = 0 ,
                            ):
    '''
    This function plots the nanoparticle intercalation into the lipid membrane
    INPUTS:
        traj_data: [obj]
            trajectory data
        lm_details: [obj]
            lipid membrane information
        np_intercalation: [obj]
            nanoparticle intercalation details
        frame: [int]
            frame of the intercalation
    OUTPUTS:
        fig: 
            figure of the intercalation
    '''
    try:
        import mayavi.mlab as mlab
        mlab.close()
    except AttributeError:
        pass
    
    ## PLOTTING LM
    fig = plot_funcs.plot_mayavi_atoms(traj = traj_data.traj,
                              atom_index = lm_details.lm_groups_atom_index['HEADGRPS'],
                              frame = frame,
                              figure = None,
                              dict_atoms = plot_funcs.ATOM_DICT,
                              dict_colors = plot_funcs.COLOR_CODE_DICT)
    
    ## GETTING ALL ATOM INDICES FOR LIGANDS THAT ARE OVERLAPPING
    res_index_within = np_intercalation.resindex_within[frame]
    
    ## GETTING ATOM INDEX
    np_heavy_atom_index_within = np_intercalation.ligand_heavy_atom_index[res_index_within].flatten()
    
    fig = plot_funcs.plot_mayavi_atoms(traj = traj_data.traj,
                              atom_index = np_heavy_atom_index_within,
                              # np_intercalation.ligand_heavy_atom_index[0],
                              frame = frame,
                              figure = fig,
                              dict_atoms = plot_funcs.ATOM_DICT,
                              dict_colors = plot_funcs.COLOR_CODE_DICT)
    return fig

##############################################
### CLASS FUNCTION TO FIND NP INTERCALATED ###
##############################################
class compute_np_intercalation:
    '''
    The purpose of this function is to compute the number of residues 
    that are intercalated into the lipid membrane. This function should 
    also be able to plot the ligands that are in fact intercalated. We use 
    the lipid membrane information computed in a previous function 
    to get the extent of intercalation. 
    
    INPUTS:
        traj_data: [obj]
            trajectory object
        z_top: [float]
            z top threshold
        z_bot: [float]
            z bottom threshold
        itp_file: [str]
            sam itp file name
    OUTPUTS:
       self.num_residue_within: [np.array]
           number of residues within lm over time
    '''
    ## INITIALIZING
    def __init__(self,
                 traj_data,
                 z_top,
                 z_bot,
                 itp_file = "sam.itp"
                 ):
    
        ## STORING VARIABLES
        self.z_top = z_top
        self.z_bot = z_bot
        self.itp_file = itp_file
        
        ## DEFINING TRAJ
        traj = traj_data.traj
        
        ## STORING TIME
        self.time_array = traj.time
        
        ## STORING UNIT CELL LENGTHS
        self.traj_box_sizes = traj.unitcell_lengths
        
        ## FINDING ALL NP POSITIONS WITH LIPID MEMBRANE
        self.atom_index_within, self.ligand_names, self.np_heavy_atom_index, np_heavy_atom_names \
                        = find_all_np_positions_within_lm(traj = traj,
                                                          mean_z_top = self.z_top,
                                                          mean_z_bot = self.z_bot)
                        
        ## GETTING NP STRUCTURE
        self.structure_np = nanoparticle_structure(traj_data           = traj_data,                # trajectory data
                                                ligand_names        = self.ligand_names,        # ligand names
                                                itp_file            = self.itp_file,                 # defines the itp file
                                                structure_types      = None,                     # checks structural types
                                                separated_ligands    = False    # True if you want separated ligands 
                                                )                
        
        ## GETTING LIGAND HEAVY ATOMS
        self.ligand_heavy_atom_index = np.array(self.structure_np.ligand_heavy_atom_index)
        # RETURNS NUM LIG X NUM ATOMS
        
        ## FIND ALL UNIQUE RESIDUE INDEXES WITHIN LAYER
        self.resindex_within = self.compute_resindex_within()
        
        ## COUNT TOTAL EACH RES INDEX
        self.num_residue_within = np.array([ len(each_list) for each_list in self.resindex_within])
        
        return
    
    
    ## FUNCTION TO GET RESINDEX WITHIN
    def compute_resindex_within(self):
        '''
        The purpose of this function is to get the residue index 
        of the given atom index
        INPUTS:
            np_intercalation: [obj]
                np_intercalation object
        OUTPUTS:
            resindex_within: [np.array]
                residue indexes that are within
        '''
        ## GETTING RESINDEX WITHIN
        resindex_within = []
        
        for each_index_list in self.atom_index_within:
            ## DEFINING CURRENT RES LIS
            current_res_list = []
            for each_atom in each_index_list:
                indexes = np.where(self.ligand_heavy_atom_index == each_atom)
                ## CHECKING IF THERE IS A LIG, SOMETIMES IT COULD BE A GOLD CORE ATOM
                if len(indexes[0]) > 0:
                    res_index = np.unique(indexes[0][0])
                    current_res_list.append(res_index)
            ## APPENDING
            resindex_within.append(np.unique(current_res_list))
        ## CONVERTING TO NUMPY
        resindex_within = np.array(resindex_within)
        return resindex_within
        
### FUNCTION TO PLOT THE SURFACES
def plot_z_surfaces(z_value,
                    box_size,
                    surface_dict={'color': 'black',
                                  },
                    fig = None,
                    ax = None):
    '''
    This function plots the z surfaces to visualize the locations.
    Reference: https://stackoverflow.com/questions/3461869/plot-a-plane-based-on-a-normal-vector-and-a-point-in-matlab-or-matplotlib
    INPUTS:
        z_value: [float]
            z value to plot
        box_size: [np.array, shape = 3]
            box size of your trajectory
        fig, ax:
            figure and axis
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    ## DEFINING THE POINT
    point  = np.array([box_size[0], box_size[1], z_value])
    # np.array(box_size)/2
    # np.array([0, 0, 0])
    normal = np.array([0, 0, 1])
    
    # a plane is a*x+b*y+c*z+d=0
    # [a,b,c] is the normal. Thus, we have to calculate
    # d and we're set
    d = -point.dot(normal)
    
    x_array = np.linspace(0, box_size[0], 100)
    y_array = np.linspace(0, box_size[1], 100)
    
    # create x,y
    xx, yy = np.meshgrid(x_array, y_array)
    
    # calculate corresponding z
    z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]
    
    # plot the surface
    if fig is None or ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        
    ## PLOTTING THE SURFACE
    ax.plot_surface(xx, yy, z, **surface_dict) # , 

    return fig, ax

### MAIN FUNCTION TO COMPUTE INTERCALATION
def main_compute_np_intercalation(path_to_sim,
                                  input_prefix,
                                  last_time_ps = 50000,
                                  selection = 'non-Water',
                                  lm_res_name = 'DOPC',
                                  itp_file = 'sam.itp',
                                  rewrite=False
                                  ):
    '''
    This function computes the nanoparticle intercalation into the lipid bilayer.
    INPUTS:
        path_to_sim: [str]
            path to the simulation
        selection: [str]
            selection to run the contacts with
        lm_res_name: [str]
            lipid membrane residue name
        itp_file: [str]
            itp file for nanoparticle
    OUTPUTS:
        np_intercalation: [obj]
            nanoparticle intercalation object
        lm_details: [obj]
            details of the lipid membrane
        
    '''
    ## CONVERTING TRAJECTORY
    trjconv_func = convert_with_trjconv(wd = path_to_sim)
    ## GETTING ONLY SPECIFIC SELECTION
    gro_file, xtc_file, ndx_file = trjconv_func.generate_gro_xtc_specific_selection(input_prefix = input_prefix,
                                                                                    selection = selection,
                                                                                    rewrite = rewrite)
    ## LOADING FILES
    traj_data = import_tools.import_traj(directory = path_to_sim,
                                         structure_file = gro_file,
                                         xtc_file = xtc_file,
                                         )
    ## UPDATING TRAJECTORY BASED ON TIME
    traj_data.traj = get_traj_from_last_frame(traj = traj_data.traj,
                                              last_time_ps = last_time_ps)
    
    ## GETTING LM GROUPS
    lm_details = analyze_lm_groups(traj = traj_data.traj,
                                   lm_res_name = lm_res_name)

    ## FINDING COM
    mean_z_top, mean_z_bot, center_of_mass = lm_details.find_top_and_bottom_leaflet_positions(traj = traj_data.traj)

    ## COMPUTING INTERCaLATION
    np_intercalation = compute_np_intercalation(traj_data = traj_data,
                                                z_top = mean_z_top,
                                                z_bot = mean_z_bot,
                                                itp_file = itp_file)
    
    return np_intercalation, lm_details, center_of_mass



#%% RUN FOR TESTING PURPOSES 
if __name__ == "__main__":
    
    sim_type_list = [
            'unbiased_ROT012_1.300'
            ]
    
    ## GRO AND FILE
    input_prefix="nplm_prod-center_pbc_mol"
    gro_file = input_prefix + ".gro"
    xtc_file = input_prefix + ".xtc"
    
    ## DEFINING SIMULATION TYPE
    for sim_type in sim_type_list:
        ## DEFINING MAIN SIMULATION DIRECTORY
        main_sim_dir= NPLM_SIM_DICT[sim_type]['main_sim_dir']
        specific_sim= NPLM_SIM_DICT[sim_type]['specific_sim']
        
        ## PATH TO SIMULATION
        path_to_sim=os.path.join(PARENT_SIM_PATH,
                              main_sim_dir,
                              specific_sim)
        
        ## DEFINING INPUTS
        inputs_np_inter={
                'path_to_sim': path_to_sim,
                'input_prefix': 'nplm_prod',
                'last_time_ps': 50000,
                'selection': 'non-Water',
                'lm_res_name': 'DOPC',
                'itp_file': 'sam.itp',
                'rewrite': False,
                }
                
        ## GETTING NANOPARTICLE INTERCALATION
        np_intercalation, lm_details, center_of_mass = main_compute_np_intercalation(**inputs_np_inter)

        
        #%%
        

        
        
        #%% PLOTTING EXAMPLES
        
        ## PLOTTING
        fig, ax = plot_lig_intercolated_vs_time(time_array = np_intercalation.time_array,
                                                num_residue_within = np_intercalation.num_residue_within)
        
        
        #%%

        ## PLOTTING Z SURFACES
        fig, ax = plot_z_surfaces(z_value = mean_z_top,
                                  box_size = np_intercalation.traj_box_sizes[0],
                                  surface_dict={'color': 'black'},
                                  fig = None,
                                  ax = None)
        
        
        ## PLOTTING Z SURFACES
        fig, ax = plot_z_surfaces(z_value = mean_z_bot,
                                  box_size = np_intercalation.traj_box_sizes[0],
                                  surface_dict={'color': 'blue'},
                                  fig = fig,
                                  ax = ax)
        
        #%%
        
        ## PLOTTING INTERCALATION
        fig = plot_nplm_intercalation(traj_data = traj_data,
                                      lm_details = lm_details,
                                      np_intercalation = np_intercalation,
                                      frame = -1 ,
                                     )
    
        
    