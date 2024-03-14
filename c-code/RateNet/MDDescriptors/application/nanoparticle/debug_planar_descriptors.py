# -*- coding: utf-8 -*-
"""
debug_planar_descriptors.py
The purpose of this script is to check the planar SAMs by measuring the tilt angle, 
monolayer height, and so on to see if the top and bottom SAM are similar

CREATED ON: 12/08/2019

AUTHOR(S):
    Alex K. Chew (alexkchew@gmail.com)
"""

### IMPORTING MODULES
import MDDescriptors.core.import_tools as import_tools # Loading trajectory details
import numpy as np
import MDDescriptors.core.calc_tools as calc_tools # calc tools
import MDDescriptors.core.read_write_tools as read_write_tools # Reading itp file tool
import mdtraj as md
import sys
import glob
import time
import os

### DEFINING GLOBAL VARIABLES
from MDDescriptors.application.nanoparticle.global_vars import GOLD_ATOM_NAME, LIGAND_HEAD_GROUP_ATOM_NAME

### IMPORTING STRUCTURE
from MDDescriptors.application.nanoparticle.nanoparticle_structure import nanoparticle_structure

## PLOTTING DISTRIBUTION OF TILT ANGLES
import MDDescriptors.core.plot_tools as plot_tools    

## DEFINING COLOR DICT
PLANAR_COLOR_DICT={
        'top': 'black',
        'bot': 'red'
        }

### FUNCTION TO PLOT DISTRIBUTION
def plot_dist_between_top_and_bottom_monolayer( dist_dict,
                                                step_size = 1,
                                                bin_range = [30, 40],
                                                xlabel="Value"):
    '''
    The purpose of this function is to plot the distribution between top and
    bottom monolayers. The idea would be to compare the distribution and see 
    how they differ.
    INPUTS:
        dist_dict: [dict]
            dictionary containing 'top' and 'bot' labels with the corresponding tilt angle distribution
        step_size: [float]
            step size within the histogram
        bin_range: [list, size=2]
            minimum and maximum bin ranges
        
    OUTPUTS:
        fig, ax: figure and axis
    '''

    ## COMPUTING BINS
    bins = np.arange(bin_range[0], bin_range[-1], step_size) + 0.5 * step_size
    
    ## CREATING FIGURE
    fig, ax = plot_tools.create_plot()
    
    ## ADDING AXIS
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of occurrences ")
    
    ## DEFINING LINE STYLE
    LINESTYLE = {
            'linewidth' : 2,
            }
    
    ## ADDING HISTOGRAM
    for each_key in dist_dict:
        ## GETTING DISTRIBUTION
        dist = dist_dict[each_key]
        ## USING NUMPY HISTOGRAM
        ps = np.histogram( dist, bins = bins.size, range = ( bins[0], bins[-1]) )[0]
        ## GETTING COLOR
        try:
            color = PLANAR_COLOR_DICT[each_key]
        except Exception:
            color = None
        
        ## PLOTTING
        ax.plot(bins, ps, color = color, label = each_key, **LINESTYLE)
    
    
    ## ADDING LEGEND
    ax.legend()
    return fig, ax

### FUNCTION TO PLOT A BAR GRAPH BETWEEN MONOLAYERS
def plot_bar_btn_top_and_bottom(dist_dict,
                                ylabel="Distance to grid"):
    '''
    The purpose of this function is to plot the bar graph between top and 
    bottom monolayer.
    INPUTS:
        dist_dict: [dict]
            dictionary between top and bottom layers
        
    '''
    import matplotlib.pyplot as plt
    ## CREATING FIGURE
    fig, ax = plot_tools.create_plot()
    ## SETTING LABELS
    ax.set_ylabel(ylabel)
    
    ## GETTING POS
    pos  = np.arange(len(dist_dict))
    y_values = [dist_dict[each_key] for each_key in dist_dict]
    
    ## PLOTTING BAR PLOT
    ax.bar(pos, y_values, align='center', color='k')
    ## CHANGING XTICKS
    plt.xticks( pos, dist_dict.keys()  )

    return fig, ax


### FUNCTION TO GET ANGLE RELATIVE TO A REFERENCE
def compute_angle_btn_displacement_to_ref_vec(displacements,
                                              ref_vector):
    '''
    The purpose of this function is to compute the angles in degrees between 
    a set of vectors relative to a reference vector. This function uses 
    the dot product rule:
        cos (theta) = ( a \dot b ) / ( |a| * |b| )
    Therefore, we first normalize the reference, compute the dot product, 
    then we divide by the norms of the vectors. Afterwards, we compute a 
    arccosine function, which outputs everything in terms of radians. We 
    lastly convert everything into degrees. 
    INPUTS:
        displacements: [np.array, shape=(frame, num_vec, 3)]
            displacement vectors
        ref_vector: [np.array, shape=(3)]
            reference vector in x, y z
    OUTPUTS:
        angles: [np.array, shape=(frame, num_vec)]
            angle in degrees between each vector and the corresponding 
            reference line.
    '''
    ## GETTING NORMALIZED REFERENCE
    norm_ref_vector = calc_tools.unit_vector(vector = ref_vector)
    
    ## GETTING DOT PRODUCT
    dot_product = np.dot(displacements, norm_ref_vector)
    
    ## FINDING NORMS OF DISPLACEMENT
    displacement_norms = np.linalg.norm(displacements, axis = 2)    
    
    ## DIVIDING DOT PRODUCT BY NORM
    dot_product_normalized = dot_product / displacement_norms
    
    ## GETTING ANGLE
    angles = np.degrees(np.arccos(np.clip(dot_product_normalized, -1.0, 1.0)))
    return angles

### FUNCTION TO GET DISPLACEMENTS BETWEEN HEAD AND GROUP
def compute_head_to_end_group_displacements(traj, structure):
    '''
    The purpose of this function is to compute the displacement between 
    the sulfur head group to the end group. 
    INPUTS:
        traj: [traj]
            trajectory object
        structure: [obj]
            structure from nanoparticle_structure class
    OUTPUTS:
        displacements: [np.array, shape=(num_frames, pairs, 3)]
            displacements between head and end groups
    '''

    ## COMPUTING INDEXES OF ALL TERMINAL GROUPS
    structure.find_all_terminal_groups()
    # OUTPUTS: structure.terminal_group_index
    
    ## GETTING ALL ATOM PAIRS
    pairs = np.concatenate( ( np.array(structure.head_group_atom_index)[:,np.newaxis], 
                             np.array(structure.terminal_group_index)[:,np.newaxis] ), axis = 1 )
    
    ## GETTING THE VECTOR BETWEEN HEAD AND TAIL GROUPS
    displacements = md.compute_displacements(traj = traj, 
                                             atom_pairs = pairs,
                                             periodic = True)
    return displacements

### FUNCTION TO GET THE TOP AND BOTTOM INDEX FOR PLANAR SAM
def find_top_and_bottom_index_planar_sam(sulfur_z_positions):
    '''
    The purpose of this function is to get the top and bottom indices 
    given the sulfur z positions
    INPUTS:
        sulfur_z_positions: [np.array, (num_frames, num_sulfurs) ]
            sulfur z positions
    OUTPUTS:
        top_index: [np.array]
            top sulfur indices
        bot_index: [np.array]
            bottom sulfur indices
    '''    
    ## GETTING ALL AVERAGE POSITIONS
    avg_z_position = np.mean(sulfur_z_positions)
    
    ## GETTING ALL INDEXES LESS THAN AND GREATER
    top_index = np.unique(np.where(sulfur_z_positions > avg_z_position)[-1]) # [1]
    bot_index = np.unique(np.where(sulfur_z_positions <= avg_z_position)[-1]) # [1]
    return top_index, bot_index

### FUNCTION TO GET TILT ANGLE OF TOP AND BOTOTM
def compute_tilt_angle_planar_monolayer(displacements, top_index, bot_index):
    '''
    The purpose of this function is to compute tilt angle for a planar monolayer 
    INPUTS:
        displacements: [np.array, shape=(num_frames, pairs, 3)]
            displacements between head and end groups
        top_index: [np.array]
            top sulfur indices
        bot_index: [np.array]
            bottom sulfur indices
    OUTPUTS:
        tilt_angles_top, tilt_angles_bot: [np.array, shape=(num_frames, num_pairs)]
            tilt angle of the top and bottom monolayer
    '''

    ## GETTING DISPLACEMENTS OF TOP AND BOTTOM
    displacements_top, displacements_bot = displacements[:,top_index], displacements[:,bot_index]

    ## GETTING ANGLES
    tilt_angles_top = compute_angle_btn_displacement_to_ref_vec(displacements = displacements_top,
                                                                ref_vector = np.array([0, 0, 1]))
    
    ## GETTING ANGLES
    tilt_angles_bot = compute_angle_btn_displacement_to_ref_vec(displacements = displacements_bot,
                                                                ref_vector = np.array([0, 0, -1]))
    return tilt_angles_top, tilt_angles_bot
    
### FUNCTION TO COMPUTE MONOLAYER HEIGHT
def compute_height_planar_monolayer(displacements,  top_index, bot_index ):
    '''
    The purpose of this function is tocompute the monolayer height given the 
    displacements between head group and terminal group.
    INPUTS:
        displacements: [np.array, shape=(num_frames, pairs, 3)]
            displacements between head and end groups
        top_index: [np.array]
            top sulfur indices
        bot_index: [np.array]
            bottom sulfur indices
    OUTPUTS:
        monolayer_height_top, monolayer_height_bot: [np.array, shape=(num_frames, num_pairs)]
            monolayer height of the top and bottom monolayer
    '''
    ## DEFINING MONOLAYER HEIGHT
    monolayer_height_distances = displacements[:,:,-1]
    monolayer_height_top, monolayer_height_bot = np.abs(monolayer_height_distances[:, top_index]), np.abs(monolayer_height_distances[:, bot_index])
    return monolayer_height_top, monolayer_height_bot

### FUNCTION TO LOAD WC DATAFILE
def load_datafile(path_to_file):
    '''
    The purpose of this function is to load the WC data file
    INPUTS:
        path_to_file: [str]
            path to the data file
    OUTPUTS:
        data: [np.array, shape = (n_points, (x,y,z,value))]
            data from the wc interface
    '''
    ## OPENING DATA FILE
    with open( path_to_file ) as raw_data:
        data = raw_data.readlines()
    
    ## TRANSFORMING DATA TO NUMPY ARRAY
    data = np.array( [ [ float(el) for el in line.split(',') ] for line in data[2:] ] )
    
    return data

### FUNCTION TO DEBUG PLANAR MONOLAYER
class compute_monolayer_properties:
    '''
    The purpose of this function is to compute monolayer properties
    INPUTS:
        traj_data: [traj]
            traj data
        structure: [object]
            structural object from nanoparticle_structure

    OUTPUTS:
        
    '''
    ## INITIALIZING
    def __init__(self,
                 traj_data,
                 structure):
    
        ## DEFINING TRAJECTORY
        traj = traj_data.traj
        
        ## COMPUTING TILT ANGLE AND MONOLAYER HEIGHT
        self.compute_tilt_angle_and_monolayer_height(traj = traj, structure = structure)
        
        ## GETTING END GROUP Z POSITIONS
        self.end_group_z_positions = traj.xyz[:, structure.terminal_group_index, -1]
        
        return
    
    ### FUNCTION TO GET MONOLAYER HEIGHT AND SO ON
    def compute_tilt_angle_and_monolayer_height(self, traj, structure):
        '''
        The purpose of this function is to compute tilt angle and monolayer height 
        for a given trajectory
        INPUTS:
            traj: [md.traj]
                trajectory object
            structure: [obj]
                structure from nanoparticle_structure
        OUTPUTS:
            self.displacements: [np.array, shape=(num_frames, num_pairs, 3)]
                displacement vectors
            self.top_index: [np.array]
                top index of ligands
            self.bot_index: [np.array]
                bottom index of ligands
            self.monolayer_height_dist_dict: [dict]
                dictionary of top and bottom height distribution
            self.tilt_angle_dist_dict: [dict]
                dictionary of top and bottom tilt angle distribution
        '''
        
        ## COMPUTING HEAD TO END GROUP DISPLACEMENTS
        self.displacements = compute_head_to_end_group_displacements(traj = traj, 
                                                                     structure = structure)
        
        ## GETTING ALL INDICES BETWEEN TOP AND BOTTOM MONOLAYERS
        sulfur_z_positions = traj.xyz[:, structure.head_group_atom_index, -1]
        
        ## GETTING TOP AND BOTTOM INDEX
        self.top_index, self.bot_index = find_top_and_bottom_index_planar_sam(sulfur_z_positions = sulfur_z_positions)
        
        ## GETTING TILT ANGLE
        tilt_angles_top, tilt_angles_bot = compute_tilt_angle_planar_monolayer(displacements = self.displacements, 
                                                                               top_index = self.top_index, 
                                                                               bot_index = self.bot_index)
        
        ## GETTING MONOLAYER HEIGHT
        monolayer_height_top, monolayer_height_bot = compute_height_planar_monolayer(displacements = self.displacements, 
                                                                                     top_index = self.top_index, 
                                                                                     bot_index = self.bot_index)

        ## DEFINING INPUTS
        self.monolayer_height_dist_dict = {'top': monolayer_height_top,
                                      'bot': monolayer_height_bot}
        
        ## DEFINING INPUTS
        self.tilt_angle_dist_dict = {'top': tilt_angles_top,
                                'bot': tilt_angles_bot}
        
        return
    
    ### FUNCTION TO PLOT FIGURE AND AXIS FOR TILT ANGLE
    def plot_tilt_angle_dist(self):
        ''' Function that plots tilt angle distribution'''
        ## PLOTTING DISTRIBUTION FOR TILT ANGLE
        fig, ax = plot_dist_between_top_and_bottom_monolayer( dist_dict = self.tilt_angle_dist_dict,
                                                              step_size = 1,
                                                              bin_range = [25, 45],
                                                              xlabel = "Tilt angle")
        return fig, ax
    
    ## # FUNCTION TO PLOT MONOLAYER HEIGHT
    def plot_monolayer_height(self):
        ''' Function that plots monolayer height '''
        ## PLOTTING DISTRIBUTION MONOLAYER HEIGHT
        fig, ax = plot_dist_between_top_and_bottom_monolayer( dist_dict = self.monolayer_height_dist_dict,
                                                              step_size = 0.1,
                                                              bin_range = [1, 2],
                                                              xlabel = "Monolayer height (nm)")
        return fig, ax
    
    
    ### FUNCTION TO GET DIFFERENCE
    def compute_grid_difference(self, path_grid):
        '''
        The purpose of this function is to get the difference between z-dimensions 
        of the grid points.
        INPUTS:
            path_grid: [str]
                path to grid file
        OUTPUTS:
            self.top_bot_dict: [dict]
                top and bottom value between grid points and end group atoms in terms of 
                z-positions
            self.bot_grid_z: [np.array]
                bottom grid z
            self.top_grid_z: [np.array]
                top grid z
        '''        
        ## GETTING DATA
        grid = load_datafile(path_to_file = path_grid)
        
        ## GETTING TOP AND BOTTOM GRID
        top_grid, bot_grid = find_top_and_bottom_index_planar_sam(sulfur_z_positions = grid[:,-1])
        
        ## GETTING ALL Z POSITIONS FOR TOP AND BOTTOM GRID
        self.top_grid_z, self.bot_grid_z = grid[top_grid,-1], grid[bot_grid,-1]
        
        ## GETTING AVERAGE Z DISTANCE FOR SULFURS
        end_z_top, end_z_bot = self.end_group_z_positions[:, self.top_index], self.end_group_z_positions[:, self.bot_index]
        
        ## GETTING DIFFERENCE
        top_diff = np.abs(np.mean(self.top_grid_z) - np.mean(end_z_top))
        bot_diff = np.abs(np.mean(self.bot_grid_z) - np.mean(end_z_bot))
        
        ## DEFINING DICT
        self.top_bot_dict = {
                'top': top_diff,
                'bot': bot_diff}
        return
    
    ### FUNCTION TO PLOT GRID DIFFERENCE
    def plot_bar_grid_position_difference(self):
        ''' This plots a bar plot between top and bottom monolayer '''
        ## PLOTTING BAR PLOT
        fig, ax = plot_bar_btn_top_and_bottom(dist_dict = self.top_bot_dict,
                                              ylabel="End group distance to grid (nm)")
        return fig, ax



#%% MAIN SCRIPT
if __name__ == "__main__":
    
    ## DEFINING SIM DICT
    sim_dict={
            'Normal': r"191205-Planar_SAMs_frozen_trial_2_NH2",
            'Translated': r"191206-Planar_SAMs_frozen_trial_3_translated",
            'Unfrozen': r"191208-Planar_SAMs_trans_unfr",
            'Anneal': "191210-annealing_try4",
                # r"191209-anneal",
            'NPT_Anneal': r"191208-Planar_SAM_temp_annealing_try3",
            }
    
    ## DEFINING PATH TO SAVE FIG
    path_save_fig = r"C:\Users\akchew\Box Sync\VanLehnGroup\2.Research Documents\Alex_RVL_Meetings\20191216\images\debug_hydrophobicity"
    # r"C:\Users\akchew\Box Sync\VanLehnGroup\2.Research Documents\Alex_RVL_Meetings\20191202\Images\pymol_debug_planar_sams\monolayer_properties"
    save_fig = True
    
    ## DEFINING JOB LIST
    job_type_list = ['Anneal'] # 'Normal', 'Translated', 'Unfrozen' 'Anneal'
    
    ## DEFINING SIMULATION PATH
    sim_path = r"S:\np_hydrophobicity_project\simulations"
    
    ## DEFINING JOB TYPE
    for job_type in job_type_list:
        
        if job_type == "NPT_Anneal":
            sim_path = r"R:/scratch/nanoparticle_project/simulations/"
        ## DEFINING SIM 
        if job_type != "Unfrozen" and job_type != "Anneal":
            sim_name = r"FrozenPlanar_300.00_K_dodecanethiol_10x10_CHARMM36jul2017_intffGold_Trial_1-50000_ps"
            if job_type == "NPT_Anneal":
                sim_name = r"Planar_300.00_K_dodecanethiol_10x10_CHARMM36jul2017_intffGold_Trial_1"
        else:
            sim_name = r"FrozenGoldPlanar_300.00_K_dodecanethiol_10x10_CHARMM36jul2017_intffGold_Trial_1-50000_ps"

        
        ## DEFINING FULL PATH
        path_to_analysis=os.path.join(sim_path,
                                      sim_dict[job_type],
                                      sim_name
                                      )
        
        ## DEFINING GRO AND XTC
        gro_file=r"sam_prod.gro"
        xtc_file=r"sam_prod_last_10ns.xtc"
        # r"sam_prod.xtc"
        
        ## DEFINING STRIDE
        stride=None
        
        ### LOADING TRAJECTORY
        traj_data = import_tools.import_traj( directory = path_to_analysis, # Directory to analysis
                     structure_file = gro_file, # structure file
                      xtc_file = xtc_file, # trajectories
                      stride = stride,
                      )
        
        #%%
        ### DEFINING INPUT DATA
        input_details = {
                            'ligand_names': ['DOD'],      # Name of the ligands of interest
                            'itp_file': 'charmm36-jul2017.ff/dodecanethiol.itp', # 'match', # ,                      # ITP FILE
                            'structure_types': ['trans_ratio'],         # Types of structurables that you want , 'distance_end_groups' # 'trans_ratio'
                            'save_disk_space': False,                    # Saving space
                            'separated_ligands':True,
                            }
        
        ## RUNNING CLASS
        structure = nanoparticle_structure(traj_data, **input_details )
        
        #%%
    
        ## GENERATING MONOLAYER PROPERTIES
        monolayer_properties = compute_monolayer_properties( traj_data = traj_data,
                                                             structure = structure)
            
        
        #%%
        ## GETTING GRID POINT DISTANCE
        # grid_folder = r"hyd_analysis_translated\grid-0_1000-0.1\out_willard_chandler.dat"
        grid_folder = r"hyd_analysis_translated\grid-0_1000\out_willard_chandler.dat"
        path_grid = os.path.join(path_to_analysis, grid_folder)
        
        ## GETTING GRID DIFFERENCE
        monolayer_properties.compute_grid_difference(path_grid)
        
        ## DEFINING PREFIX
        fig_prefix = os.path.join(path_save_fig,job_type)
        
        ## TILT ANGLE PLOT
        fig, ax = monolayer_properties.plot_tilt_angle_dist()
        plot_tools.save_fig_png(fig = fig, 
                                label = fig_prefix + "-tilt_angle", 
                                save_fig = save_fig)
        
        ## MONOLAYER HEIGHT
        fig, ax = monolayer_properties.plot_monolayer_height()
        plot_tools.save_fig_png(fig = fig, label = 
                                fig_prefix + "-height", 
                                save_fig = save_fig)
        
        ## GRID DIFERENCE
        fig, ax = monolayer_properties.plot_bar_grid_position_difference()
        plot_tools.save_fig_png(fig = fig, label = 
                                fig_prefix + "-grid_diff", 
                                save_fig = save_fig)
    
    

    #%%
    
    ################################################
    ### CLASS FUNCTION TO COMPUTE PLANAR DENSITY ###
    ################################################
    class compute_monolayer_density:
        '''
        The purpose of this function is to compute the density across the monolayer. 
        The idea is to get all the z-positions, bin the z, then plot a density. 
        INPUTS:
            
        OUTPUTS:
            
        '''
        ## INITIALIZING
        def __init__(self, traj_data, 
                     structure, 
                     monolayer_properties,
                     bin_width = 0.01):
            ## STORING
            self.bin_width = bin_width
            
            ## DEFINING TRAJECTORY
            traj = traj_data.traj
            
            ## STORING STRUCTURE AND MONOLAYER PROPERTIES
            self.get_system_details( traj = traj,
                                     structure = structure,
                                     monolayer_properties = monolayer_properties)
            
            
            ## CREATING STORAGE FOR DENSITY
            self.density_storage = {}
            

            
            return
        
        ## GETTING THE SYSTEM DETAILS
        def get_system_details(self, traj, structure, monolayer_properties):
            '''
            The purpose of this function is to get the system information, such 
            as z-dimensions and so on. 
            INPUTS:
                traj: [obj]
                    trajectory object
                structure: [obj]
                    structure object from nanoparticle_structure
                monolayer_properties: [obj]
                    monolayer structural properties
            OUTPUTS:
                self.box_size: [np.array]
                    box sizes
                        x, y, z are in self.box_size_x, self.box_size_y, self.box_size_z
                self.z_position_top_ref: [np.array]
                    position of the top sulfur
            '''
            ###########################
            ### DEFINING THE SYSTEM ###
            ###########################
            
            ## SULFUR INDEX
            top_sulfur_index = np.array(structure.head_group_atom_index)[monolayer_properties.top_index]
            bot_sulfur_index = np.array(structure.head_group_atom_index)[monolayer_properties.bot_index]
            
            ## FINDNIG Z POSITIONS OF SULFURS FOR EACH FRAME
            # self.z_position_top_ref = np.mean(traj.xyz[:,top_sulfur_index,2],axis=1) # NUM_FRAMES X 1
            self.z_position_top_ref = np.mean( (np.mean(traj.xyz[:,top_sulfur_index,2], axis = 1), 
                                                np.mean(traj.xyz[:,bot_sulfur_index,2], axis = 1)), axis = 0)
            self.z_position_bot_sulfur = np.mean(traj.xyz[:,bot_sulfur_index,2],axis=1) # NUM_FRAMES X 1
            
            ## FINDING BOX SIZE DETAILS
            self.box_size = traj.unitcell_lengths
            self.box_size_x = self.box_size[:,0] # NUM_FRAMES X 1 -- x dimensions 
            self.box_size_y = self.box_size[:,1] # NUM_FRAMES X 1 -- y dimensions 
            self.box_size_z = self.box_size[:,2] # NUM_FRAMES X 1 -- z dimensions 
            
            ## FINDING DISTANCE VECTOR BETWEEN TOP AND BOTTOM MONOLAYER
            self.dist_vector = self.box_size_z
            # self.z_position_bot_sulfur + self.box_size_z - self.z_position_top_ref
            self.max_dist = np.max(self.dist_vector)
            self.max_bin = int(np.floor(self.max_dist / self.bin_width))
            
            ## BIN VOLUME
            self.bin_volume = np.mean(self.box_size_x * self.box_size_y * self.bin_width) # Average bin volume
            
            ## NUMBER OF FRAMES
            self.num_frames = traj.time.size
            
            return
        
        ### FUNCTION TO COMPUTE ATOMIC DISPLACEMENTS (NO PBC)
        def compute_displacements_for_atoms(self, 
                                            traj,
                                            residue_name, 
                                            atom_names = None,
                                            verbose = True):
            '''
            The purpose of this function is to compute displacements for all 
            atoms in the z-direction
            INPUTS:
                traj: [obj]
                    trajectory object
                residue_name: [str]
                    residue name of interest
                atom_names: [str]
                    atom names of interest
                verbose: [logical]
                    True if you want to print details
            OUTPUTS:
                disp_storage: [dict]
                    storage of the:
                        residue atom name
                        atom_per_residue (used for normalization)
                        z_dis -- z-displacements with no periodic boundaries enforced
            '''
            ########################################
            ### GETTING EACH RESIDUE AND BINNING ###
            ########################################
            ## FINDING RESIDUE AND ATOM INDICES
            residue_index, atom_index = calc_tools.find_residue_atom_index(traj = traj,
                                                                           residue_name = residue_name, 
                                                                           atom_names = atom_names)
            
            ## GETTING ATOM NAMES
            if atom_names is None:
                atom_names = np.unique(calc_tools.find_atom_names(traj = traj, 
                                                        residue_name = residue_name))
            ## GETTING ATOM NAME AND RESIDUE NAME
            residue_atom_name = residue_name + '-' + '_'.join(atom_names)
            
            # CREATING COPY OF TRAJ
            copied_traj = traj[:]
            
            ## GETTING ALL ATOM INDEX
            all_atom_index = np.array([ atom.index for atom in traj.topology.atoms ])
            
            ## GETTING CONCATENATED ATOM INDEX
            atom_index_flatten = np.concatenate(atom_index)
            
            ## PRINTING
            if verbose is True:
                print("--------------------------------------------------------")
                print("Computing displacements between top sulfur and %s"%(residue_atom_name) )
                print("Total number of atoms: %d\n"%(len(atom_index_flatten)) )
            
            # FINDING INDEX OF THE FIRST ATOM IN TOP MONOLAYER (OR "BOTTOM" WITH PBC)
            index_to_change=np.setdiff1d( all_atom_index, atom_index_flatten )[0] # First index
            
            # CHANGING POSITIONS OF THE SULFUR ATOM (Z-POSITION)
            copied_traj.xyz[:, index_to_change, 2] = self.z_position_top_ref
            
            ## GENERATING ATOM PAIRS
            atom_pairs = np.concatenate( (np.array([index_to_change] * len(atom_index_flatten) )[:,np.newaxis],
                                          atom_index_flatten[:, np.newaxis]), axis = 1 )
            
            ## GETTING ATOM PER RESIDUE
            atom_per_residue = float(len(atom_pairs) / len(residue_index))
            
            ## USING MDTRAJ TO CALCULATE DISPLACEMENT WITH NO PERIODIC BOUNDARY CONDITIONS
            displacements = md.compute_displacements( traj=copied_traj, 
                                                      atom_pairs = atom_pairs, 
                                                      periodic = False)
            ## RETURNS DISPLACEMENTS IN NUM_FRAMES, NUM_ATOMS, 3
            
            ## GETTING Z DISPLACEMENTS AND SUBTRACTING TO BRING DOWN THE Z-DISTANCE W.R.T SULFUR MONOLAYER
            z_dis = displacements[:,:,2]
            
            ## CREATING DICTIONARY
            disp_storage= {
                    'residue_atom_name' : residue_atom_name,
                    'atom_per_residue'  : atom_per_residue,
                    'z_dis'             : z_dis,
                    }
            
            return disp_storage
    
        ### FUNCTION TO COMPUTE DENSITY DISTRIBUTION
        def compute_density_dist(self,disp_storage):
            '''
            The purpose of this function is to compute the density distribution 
            based on the density storage output from the displacement function.
            INPUTS:
                disp_storage: [dict]
                    dictionary containing residue atom name, z_dis, etc.
                
            OUTPUTS:
                
            '''
            
            ## FINDING Z DISTRIBUTION
            z_dis = disp_storage['z_dis'] ## NUM_FRAMES X NUM_ATOMS
            ## GETTING ATOM PER RESIDUE
            atom_per_residue = disp_storage['atom_per_residue']
            
            ## CORRECTING Z DISPLACEMENTS FOR PBC
            for time_index in range(len(z_dis)):
                # Finding all z-displacements below the top layer
                isLessThan = np.where(z_dis[time_index] < 0 )
                # For all those displacements, simply add the box length in the z-dimensions
                z_dis[time_index, isLessThan] += self.box_size_z[time_index] # Adding L
            
            ## GETTING R ARRAY
            z_array = np.arange( 0, self.max_dist, self.bin_width ) + 0.5 * self.bin_width
            ## USING NUMPY HISTOGRAM
            hist = np.histogram(z_dis, bins = z_array.size, range=(0, self.max_dist))[0]
            
            ## GETTING HISTOGRAM NORMALIZED
            hist_norm = hist / self.bin_volume / self.num_frames / atom_per_residue 
            
            return z_array, hist_norm
  
        ### FUNCTION TO PLOT DENSITY
        @staticmethod
        def plot_planar_density(z_array, 
                                hist_norm,
                                fig = None,
                                ax = None,
                                **plot_config):
            '''
            The purpose of this function is to plot the planar density. 
            INPUTS:
                z_array: [np.array]
                    z-array in nm
                hist_norm: [np.array]
                    normalized histogram
                fig, ax: [figure obj]
                    input figure and axis
            OUTPUTS:
                fig, ax: figure and axis
            '''
            if fig is None or ax is None:
                ## CREATING FIGURE
                fig, ax = plot_tools.create_plot()
                
                ## ADDING AXIS
                ax.set_xlabel("z (nm)")
                ax.set_ylabel("Density (num/nm^3)")
            
            ## PLOTTING
            ax.plot(z_array, hist_norm, **plot_config)
            
            return fig, ax
    
    
    ## DEFINING DEFAULTS
    bin_width=0.01
    
    
    ### DICTIONARY OF THE DIFFERENT TYPES
    library_residues = {
            'WATER':
                {'residue_name': 'HOH',
                 'atom_names'  : None, # ['O']
                 'color'       : 'blue', 
                 },
            'GOLD':
                {'residue_name': 'AUI',
                 'atom_names'  : None,
                 'color'       : 'gold',
                 },
            'LIGAND':
                {'residue_name': 'DOD',
                 'atom_names'  : None,
                 'color'       : 'red',
                 },
#            'SULFUR':
#                {'residue_name': 'DOD',
#                 'atom_names'  : 'S1',
#                 'color'       : 'yellow',
#                 },
            }
    
    ## COMPUTING MONOLAYER DENSITY
    density = compute_monolayer_density( traj_data = traj_data, 
                                         structure = structure,
                                         monolayer_properties = monolayer_properties,
                                         bin_width = bin_width)
    ## DEFINING TRAJECTORY    
    traj = traj_data.traj
    
    ## GENERATE STORAGE DICT
    storage_dict = {}
    
    ## LOOPING THROUGH LIBRARY
    for each_combination in library_residues:
        ## COMPUTING DISPLACEMENTS
        disp_storage = density.compute_displacements_for_atoms(traj = traj,
                                                               residue_name = library_residues[each_combination]['residue_name'],
                                                               atom_names = library_residues[each_combination]['atom_names'],
                                                               verbose = True,)
        ## COMPUTING DENSITY DISTRIBUTION
        z_array, hist_norm = density.compute_density_dist(disp_storage = disp_storage)
        ## STORING
        storage_dict[each_combination] = {
                'z_array': z_array,
                'hist_norm': hist_norm,
                }


    #%%

    ## DEFINING PLOT FEATURES
    plot_features = {
            'linewidth': 2,
            }
    
    ## DEFINING NO FIGURE, AXIS
    fig, ax = None, None
    
    ## LOOPING THROUGH LIBRARY
    for each_combination in library_residues:
        ## GETTING COLOR
        color = library_residues[each_combination]['color']
        ## STORING COLOR
        plot_features['color'] = color
        plot_features['label'] = each_combination
        ## PLOTTING
        fig, ax = density.plot_planar_density(fig = fig, ax = ax, **storage_dict[each_combination], **plot_features)
        
    ### FUNCTION TO GET RELATIVE POSITION
    def get_relative_position(value, z_ref, box_length):
        '''
        This function gets the relative position of z by subtracting the ref, then adding 
        box length L if the value turns out to be negative. 
        INPUTS:
            value: [float]
                value you want relative positions for
            z_ref: [float]
                relative z value
            box_length: [float]
                box length that will fix the value based on PBC
        OUTPUTS:
            relative_value: [float]
                value relative to reference and box dimension
        '''
        relative_value = value - z_ref
        if relative_value < 0:
            relative_value += box_length
        return relative_value
    
    ## DEFINING TOP Z REFERENCE
    top_z_ref = np.mean(density.z_position_top_ref)
    
    ## ADDING GRIDDING 
    grid_z_bot = np.mean(monolayer_properties.bot_grid_z)
    grid_z_top = np.mean(monolayer_properties.top_grid_z)
    
    ## GETTING RELATIVE VALUES
    grid_z_bot_rel = get_relative_position(value = grid_z_bot,
                                           z_ref = top_z_ref,
                                           box_length = np.mean(density.box_size_z))
    grid_z_top_rel = get_relative_position(value = grid_z_top,
                                           z_ref = top_z_ref,
                                           box_length = np.mean(density.box_size_z))
    
    ## ADDING TO PLOT
    ax.axvline(x = grid_z_top_rel, color = 'g', linestyle = '--', label = 'top')
    ax.axvline(x = grid_z_bot_rel, color = 'k', linestyle = '--', label = 'bot')
    
    ## ADDING LEGEND
    ax.legend()
    
    #%%
    
    ## DEFINING WATER DICTIONARY
    water_dict = storage_dict['WATER']
    
    ## GETTING ALL POSITIONS
    water_hist = water_dict['hist_norm']
    
    
    
    ax.axhline(y = 33, color = 'k', linestyle = '--', label = 'bot')
    
    

    
    
    


    
    
     