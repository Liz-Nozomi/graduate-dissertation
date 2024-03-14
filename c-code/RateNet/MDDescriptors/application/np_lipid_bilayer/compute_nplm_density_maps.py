# -*- coding: utf-8 -*-
"""
compute_nplm_density_maps.py
The purpose of this function is to compute a density profile for a nanoparticle 
lipid membrane simulation. The idea is that we want to visualize a large number 
of frames quickly. To do this, we will use ideas from spatial distribution maps
and previous Reid Van Lehn's papers to generate cylindrical maps / normal 
density maps. At the end, we would like to visualize the lipid membrane and see 
what is going on in these systems. 

Author: Alex K. Chew (02/14/2020)

EDITING TRAJECTORY SO WE HAVE CONSISTENT COMPARISONS:
    ## EDITING TRAJECTORY
    # XTC FILE
    gmx trjconv -f nplm_prod.xtc -s nplm_prod.tpr -o nplm_prod_last_5_ns_centered_np.xtc -center -pbc mol -b 95000 -skip 10
    # GRO FILE
    gmx trjconv -f nplm_prod.xtc -s nplm_prod.tpr -o nplm_prod_last_5_ns_centered_np.gro -center -pbc mol -dump 95000
    Selection:
        AUNP
        non-Water
    # TPR FILE
    gmx convert-tpr -s nplm_prod.tpr -o nplm_prod_last_5_ns_centered_np.tpr
    Selection:
        non-Water

    # --- don't need below, centered on gold nanoparticle is sufficent
    # XTC FILE
    gmx trjconv -f nplm_prod_last_5_ns_centered_np.xtc -s nplm_prod_last_5_ns_centered_np.tpr -o nplm_prod_last_5_ns_centered_lm.xtc -center -pbc mol
    gmx trjconv -f nplm_prod_last_5_ns_centered_np.gro -s nplm_prod_last_5_ns_centered_np.tpr -o nplm_prod_last_5_ns_centered_lm.gro -center -pbc mol
    
    Selection:
    DOPC
    System

INPUTS:
    - gro and xtc file
    - grid spacing
OUTPUTS:
    - pickle file with the stored densities -- will use the loop traj function? 


"""
## IMPORTING TOOLS
import os
import MDDescriptors.core.import_tools as import_tools
import MDDescriptors.core.calc_tools as calc_tools
import numpy as np
import mdtraj as md
## IMPORTING PATHS
from MDDescriptors.core.check_tools import check_path

## IMPORTING FUNCTION FOR LIGAND NAMES
from MDDescriptors.application.nanoparticle.core.find_ligand_residue_names import get_ligand_names_within_traj
from MDDescriptors.application.np_lipid_bilayer.compute_NPLM_contacts_extract import generate_rotello_np_groups, generate_lm_groups

## IMPORTING COMMANDS 
from MDDescriptors.traj_tools.trjconv_commands import convert_with_trjconv, generate_gro_xtc_with_center_AUNP

## IMPORTING LAST FRAME TOOL
from MDDescriptors.core.traj_itertools import get_traj_from_last_frame


### FUNCTION TO COMPUTE DISPLACEMENTS
def compute_displacements(traj, 
                          reference_centers, 
                          atom_index,
                          periodic = True):
    '''
    The purpose of this function is to compute displacements given a trajectory 
    and reference centers. The code will first find an atom index that 
    is not in use to replace as reference center. Then, it will use traj to 
    compute displacments.
    INPUTS:
        traj: [obj]
            trajectory object
        reference_centers: [np.array, shape = (num_frames, 3)]
            reference center for your displacement
        atom_index: [np.array]
            list of the atom indices that you care about
        periodic: [logical]
            True if you want PBC to be accounted for
    OUTPUTS:
        displacements: [np.array, shape = (num_frames, num_atom_index, 3)]
            displacements from the center
    '''
    ## COPYING TARAJECTORY
    copied_traj = traj[:]
    
    ## GETTING ATOM INDICES THAT ARE NOT IN THE ATOM INDEX
    atom_not_in_index = [ x.index for x in traj.topology.atoms if x.index not in atom_index]
    if len(atom_not_in_index) == 0:
        print("Error! All atoms are used in the atom indices")
        print("This can cause errors since we have no more atoms to move and compute displacements")
    else:
        ## ARBITRARILY SELECTING A POINT TO MOVE
        index_to_change = atom_not_in_index[0]
    
    ## CHANGING POSITION OF THE SOLUTE
    copied_traj.xyz[:, index_to_change, :] = reference_centers[:]
    
    ## GENERATING ATOM PAIRS
    atom_pairs = [ [index_to_change, x] for x in atom_index]
    
    ## COMPUTING DISPLACEMENTS
    displacements = md.compute_displacements( traj=copied_traj, 
                                              atom_pairs = atom_pairs, 
                                              periodic = periodic)

    return displacements

### FUNCTION TO CONVERT CARTESIAN TO POLAR
def cart2pol(x, y):
    '''
    The purpose of this function is to convert cartesian to polar coordinates. 
    INPUTS:
        x: [np.array]
            x positions
        y: [np.array]
            y positions
    OUTPUTS:
        
    '''
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

### FUNCTION TO CONVERT DISPLACEMENT ARRAY INTO A SEPARATE POLAR COORD
def convert_xyz_array_cart2polar( array ):
    '''
    The purpose of this function is to convert an x, y, z array into 
    a polar coordinate array. 
    
    INPUTS:
        array: [np.array, shape = (num_frames, num_displacements, 3 )]
            x,y,z array
    OUTPUTS:
        polar_array: [np.array, same shape as array]
            r, theta, z array. Note that theta is in radians
    '''
    ## COPYING 
    polar_array = array.copy()
    
    ## COMPUTING RADIUS AND THETA
    rho, phi = cart2pol(x = array[:,:,0],
                        y = array[:,:,1])
    
    ## STORING
    polar_array[:,:,0] = rho[:]
    polar_array[:,:,1] = phi[:]
    
    return polar_array

###########################################
### CLASS FUNCTION TO COMPUTE DENSITIES ###
###########################################
class compute_nplm_density:
    '''
    The purpose of this code is to compute the densities across nanoparticle-
    lipid membrane simulations. 
    INPUTS:
        traj_data: [obj]
            trajectory object
        r_range: [tuple]
            range of the radius to search
        z_range: [tuple]
            z-range to search for
        bin_width: [int]
            width of the bin
        lm_res_name: [str]
            lipid membrane residue name
    OUTPUTS:
        
        
    ''' 
    ## INITATION
    def __init__(self,
                 traj_data,
                 lm_res_name = "DOPC",
                 r_range = (0, 5),
                 z_range =(-7, 7),
                 bin_width = 0.2,
                 ):
        
        ## STORING INPUTS
        self.lm_res_name = lm_res_name
        self.r_range = r_range
        self.z_range = z_range
        self.bin_width = bin_width
        
        ## GETTING HISTOGRAM INFORMATION
        self.arange, self.bins = self.compute_histogram_bins()
        
        ## DEFINING TRAJ
        traj = traj_data.traj
        
        ## STORE TIME ARRAY
        self.time_array = traj.time[:]
        self.time_length = len(self.time_array)
        
        ## STORING VOLUME AND LENGTHS
        self.unitcell_volumes = traj.unitcell_volumes[:]
        self.unitcell_lengths = traj.unitcell_lengths[:]
        
        ## FINDING LIGAND NAMES
        self.ligand_names = get_ligand_names_within_traj(traj=traj)
        ## GETTING NANOPARTICLE HEAVY ATOMS
        self.np_heavy_atom_index = np.array([ calc_tools.find_residue_heavy_atoms(traj = traj,
                                          residue_name = each_ligand) for each_ligand in self.ligand_names ]).flatten()
        
        ## GETTING HEAVY ATOM INDEX OF MEMBRANE
        self.lm_heavy_atom_index = calc_tools.find_residue_heavy_atoms(traj = traj,
                                                                       residue_name = lm_res_name)
        
        ## LOCATING GROUPS
        self.np_groups = generate_rotello_np_groups(traj = traj, 
                                                    np_heavy_atom_index = self.np_heavy_atom_index, )
        self.lm_groups = generate_lm_groups(traj = traj, 
                                            lm_heavy_atom_index = self.lm_heavy_atom_index)
        
        ## GETTING CENTER OF MASS OF GOLD
        self.gold_com = np.squeeze(calc_tools.find_center_of_mass(traj, 
                                                                  residue_name = self.ligand_names[0], 
                                                                  atom_names = self.np_groups['GOLD'] ))
        # RETURNS: num frames, 3(xyz)
        ## GENERATING ATOM INDEXES
        self.generate_atom_indices(traj = traj)
        
        return
    
    ### FUNCTION TO GENERATE DICTIONARIES
    def generate_atom_indices(self, traj):
        '''
        This function generates atom indices for different types. 
        INPUTS:
            self: [obj]
                class object
        OUTPUTS:
            
        '''
    
        ## GENERATING DICTIONARY OF ATOM INDICES
        self.density_dict = {
                    'GNP': 
                            {
                            'atom_index': self.np_heavy_atom_index,
                            },
                    'LM':
                            {
                            'atom_index': self.lm_heavy_atom_index,
                            },
                    }
                            
        ## GETTING DICTIONARY
        group_dict = {
                'GNP' : 
                    {'group': self.np_groups,
                     'resname': self.ligand_names[0],
                         },
                'LM'  : 
                    {'group': self.lm_groups,
                     'resname': self.lm_res_name,
                         },
                    
                    
                }
        
        ## ADDING OTHER DICTIONARIES
        for group_key in group_dict.keys():
            ## LOOPING THROUGH EACH KEY
            for each_key in group_dict[group_key]['group'].keys():

                ## GETTING ATOM INDICES
                _, atom_index = calc_tools.find_residue_atom_index(traj = traj,
                                                                   residue_name = group_dict[group_key]['resname'],
                                                                   atom_names = group_dict[group_key]['group'][each_key])
                
                ## FLATTENING
                atom_index = calc_tools.flatten_list_of_list(atom_index)
                
                ## STORING THE DICTIONARY
                self.density_dict[group_key + '-' + each_key] = {'atom_index': atom_index[:]}
        
        ## ADDING COUNTERIONS
        _, atom_index = calc_tools.find_residue_atom_index(traj = traj,
                                                           residue_name = 'CL',)
        
        ## FLATTENING
        atom_index = calc_tools.flatten_list_of_list(atom_index)
        
        ## STORING THE DICTIONARY
        self.density_dict['CL'] = {'atom_index': atom_index[:]}
        
        return 
    
    ### FUNCTION TO COMPUTE HISTOGRAM DETAILS
    def compute_histogram_bins(self):
        '''
        The purpose of this function is to generate histogram details, like bins, and so forth
        INPUTS:
            self: [obj]
                class object
        OUTPUTS:
            arange: [tuple]
                tuple of r and z ranges
            bins: [tuple]
                tuple of number of binds for each
        '''
        ## DEFINING RANGE
        arange = (self.r_range, self.z_range)
        
        ## FINDING MAXIMUM NUMBER OF BINS
        bins =  tuple([ int(np.floor( (each_range[1]- each_range[0]) / self.bin_width)) for each_range in arange ])
        
        return arange, bins
    
    ### FUNCTION TO GET DISPLACMENETS
    def compute_displacements(self,
                              traj, 
                              atom_index,
                              periodic = True):
        '''
        This function computes the displacements relative to some reference center. 
        
        INPUTS:
            traj: [obj]
                trajectory from md.load
            atom_index: [list]
                atom indices as a form of a list
            periodic: [logical]
                True if you want PBC
        OUTPUTS:
            displacements: [np.array, shape = (num_frames, num_index, 3)]
                displacements in cartesian coordinates
            polar_array: [np.array, shape = (num_frames, num_index, 3)]
                displacements in polar coordinates (r, theta,  z)
        '''
        ## COMPUTING DISPLACEMENTS
        displacements = compute_displacements(traj = traj, 
                                              reference_centers = self.gold_com, 
                                              atom_index = atom_index, # density_dict['LM']['atom_index'],
                                              periodic = periodic)
        ## GETTING POLAR ARRAY
        polar_array = convert_xyz_array_cart2polar( array = displacements )
        
        return displacements, polar_array
    
    ### FUNCTION TO GENERATE GRID ARRAY
    def compute_num_dist(self,
                        polar_array,
                        ):
        '''
        The purpose of this function is to generate 2D histograms 
        INPUTS:
            self: [obj]
                class object
            polar_array: [np.array, shape = (num_frames, num_index, 3)]
                displacements in polar coordinates (r, theta,  z)
        OUTPUTS:
            grid: [np.array, shape = (num_bin_r, num_bin_z)]
                number of occurances for each bin
        '''
        ## GENERATING HISTOGRAM
        grid, edges = np.histogramdd(np.zeros((1, len(self.bins))), bins=self.bins, range=self.arange, normed=False)
        grid *=0.0 # Start at zero
        
        ## LOOPING THROUGH FRAMES
        for frame in range(self.time_length):
            ## DEFINING CURRENT DISPLACEMENTS
            disp = polar_array[frame][:,[0,-1]] # r and z displacmeents
            ### USING HISTOGRAM FUNCTION TO GET DISPLACEMENTS WITHIN BINS
            hist, edges = np.histogramdd(disp, bins=self.bins, range=self.arange, normed=False)
            ### ADDING TO THE GRID
            grid += hist
            
        return grid
    
    ### FUNCTION TO NORMALIZE HISTOGRAM
    def normalize_histogram(self,
                            grid,
                            normtype="none"):
        '''
        The purpose of this function is to normalize the histograms.
        INPUTS:
            grid: [np.array, shape = (num_bin_r, num_bin_z)]
                number of occurances for each bin
            normtype: [str]
                normalization type
        OUTPUT:
            updated_grid: [np.array, shape = (num_bin_r, num_bin_z)]
                updated grid
        '''
        ## COPYING GRID
        updated_grid = grid.copy()
        
        ## 
        if normtype == "by_time":
            updated_grid /= self.time_length
        else:
            print("Error! Normalization type not found: %s"%(normtype) )
        
        return
    
    ### FUNCTION TO LOOP AND COMPUTE THE GRID
    def compute(self,
                traj):
        '''
        The purpose of this function is to compute the grid information for 
        multiple different types of atom indices.
        INPUTS:
            traj: [traj]
                trajectory object
        OUTPUTS:
            self.density_dict: [dict]
                dictionary that contains all the grid information
        '''
        ## LOOPING THROUGH EACH KEY
        for each_key in self.density_dict:
            ## PRINTING
            print("Computing histograms for key: %s"%(each_key) )
            
            ## GETTING ATOM INDICES
            atom_index = self.density_dict[each_key]['atom_index']
            
            ## COMPUTING DISPLACEMENTS
            displacements, polar_array = self.compute_displacements(
                                                                    traj = traj, 
                                                                    atom_index = atom_index,
                                                                    periodic = True
                                                                    )
            
            ## COMPUTING NUMBER DIST
            grid = self.compute_num_dist(polar_array = polar_array)
            
            ## STORE THE GRID
            self.density_dict[each_key]['grid'] = grid[:]
            
        return
    
### FUNCTION TO PLOT HISTOGRAM
def plot_2d_histogram(grid,
                      r_range,
                      z_range,
                      increments = 1,
                      cmap_type = 'jet',
                      interpolation = 'none',
                      fig = None,
                      ax = None,
                      want_color_bar = True,
                      **kwargs
                      ):
    '''
    The purpose of this function is to plot 2D histogram
    INPUTS:
        grid: [np.array]
            grid as a 2D numpy array
        r_range: [tuple]
            r range
        z_range: [tuple]
            z range
        increments: [int]
            increments of the x, y plot
        cmap_type: [str] 
            type of color map
        interpolation: [str]
            type of interpolation
        **kwargs: variables for imshow
    OUTPUTS:
        
    '''
    ## IMPORT TOOLS
    import matplotlib.pyplot as plt
    
    ## GENERATING FIGURE
    if fig == None or ax == None:
        fig = plt.figure(figsize=(4, 6))
        ax = fig.add_subplot(111)
        ## ADDING X LABELS
        ax.set_xlabel("R (nm)")
        ax.set_ylabel("Z (nm)")
        ## X AND Y TICK LABELS
        ax.set_xticks(np.arange(r_range[0], r_range[1] + increments, increments))
        ax.set_yticks(np.arange(z_range[0], z_range[1] + increments, increments))

    ## PLOTTING HEAT MAP
    h = plt.imshow(X = np.rot90(grid), # Rotating 90 degees due to im.show defaults
                   extent=[r_range[0], r_range[1], z_range[0], z_range[1] ],
                   interpolation = interpolation,
                   cmap=cmap_type,
                   **kwargs)

    ## ADDING COLOR BAR    
    if want_color_bar is True:
        cbar = plt.colorbar(mappable=h, 
                            ax = ax,
                            )
    
    ## ADDING MINOR TICKS
    # cbar.minorticks_on()

    return fig, ax

### FUNCTION TO RUN DENSIITES
def main_compute_nplm_densities(path_to_sim,
                                input_prefix,
                                nplm_densities_input,
                                last_time_ps = 50000,
                                ):
    '''
    This function computes the densities in nanoparticle lipid membrane systems.
    INPUTS:
        path_to_sim: [str]
            path to simulation folder
        input_prefix: [str]
            input prefix for gro, xtc, tpr, and so on
    OUTPUTS:
        densities: [obj]
            densities object
    '''
    ## CONVERTING TRAJECTORY
    output_gro, output_xtc = generate_gro_xtc_with_center_AUNP(path_to_sim = path_to_sim,
                                                               input_prefix = input_prefix,
                                                               output_suffix = None,
                                                               rewrite = False, # CHANGE TO FALSE LATER
                                                               skip_frames=1)
    
    ## LOADING FILES
    traj_data = import_tools.import_traj(directory = path_to_sim,
                                         structure_file = output_gro,
                                         xtc_file = output_xtc,
                                         )

    
    ## UPDATING TRAJECTORY BASED ON TIME
    traj_data.traj = get_traj_from_last_frame(traj = traj_data.traj,
                                              last_time_ps = last_time_ps)
    
    
    ## STORING TRAJ DATA
    nplm_densities_input['traj_data'] = traj_data

    ## GETTING DENSITIES
    densities = compute_nplm_density(**nplm_densities_input)
    
    ## COMPUTTING DENSITIES
    densities.compute(traj = traj_data.traj)
    
    return densities
    

#%%
if __name__ == "__main__":
    
    ## DEFINING PATH TO SIMULATION
    path_sim_parent = r"R:/scratch/nanoparticle_project/simulations"
    path_sim_parent = r"/Volumes/akchew/scratch/nanoparticle_project/simulations"
    sim_parent = r"20200120-US-sims_NPLM_rerun_stampede"
    sim_folder= r"US-1.3_5_0.2-NPLM-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1"
    
    ## DEFINING RELATIVE SIM PATH
    relative_sim_path = r"4_simulations/1.300"
    
    ## DEFINING GRO AND XTC
    gro_file = "nplm_prod_last_5_ns_centered_np.gro"
    xtc_file = "nplm_prod_last_5_ns_centered_np.xtc"
    
    ## DEFINING LAST TIME
    last_time_ps = 50000 # using the last 50 ns of simulation data

    ## DEFINING PATH TO SIMULATION
    path_to_sim = os.path.join(path_sim_parent, 
                            sim_parent,
                            sim_folder,
                            relative_sim_path)
    
    ## DEFINING PREFIX
    input_prefix = "nplm_prod"
    
    
    ## DEFINING INPUTS
    nplm_densities_input = {
            'bin_width': 0.1, # 0.2,
            'r_range': (0, 5),
            'z_range': (-7, 7),
            'lm_res_name': "DOPC",

            }
    
    ## DEFINING FUNCTION INPUT
    func_inputs = {'input_prefix': input_prefix,
                   'path_to_sim': path_to_sim,
                   'nplm_densities_input': nplm_densities_input,
                   'last_time_ps': 50000,
                   }

    ## COMPUTING NPLM DENSITIES
    densities = main_compute_nplm_densities(path_to_sim,
                                input_prefix,
                                nplm_densities_input
                                )

        
    
    #%%
    
    

    
    
    
        #%%
        
    '''
    ## PLOTTTING HISTOGRAMS
    fig, ax = plot_2d_histogram(grid = densities.density_dict['LM']['grid'],
                                r_range = densities.r_range,
                                z_range = densities.z_range,
                                increments = 1,
                                cmap_type = 'jet',
                                interpolation = 'none',
                                )
    '''
    
    