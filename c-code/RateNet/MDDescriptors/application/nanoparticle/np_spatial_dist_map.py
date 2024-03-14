# -*- coding: utf-8 -*-
"""
np_spatial_dist_map.py
The purpose of this script is to compute the spatial distribution maps of solvents 
around a NP surface. The goal is to see which parts of the nanoparticle preferred 
to be solvated by a particular solvent.

ASSUMPTIONS:
    - You have already ran a NP system in mixed-solvent environment and you have frozen 
    the NP in the center of the box. 
    - You have ran trjconv and pbc mol to ensure that the system is not rotating and all 
    the molecules are within the  box.

Author: Alex K. Chew (Created on 09/06/2019)
"""
## IMPORTING MODULES
import os
import numpy as np
import mdtraj as md

## IMPORTING MD DESCRIPTOR TOOLS
import MDDescriptors.core.import_tools as import_tools # Loading trajectory details
import MDDescriptors.core.initialize as initialize # Checks file path

### IMPORTING NANOPARTICLE STRUCTURE CLASS
from MDDescriptors.application.nanoparticle.nanoparticle_structure import nanoparticle_structure

## PLOTTING TOOLS
import MDDescriptors.core.plot_tools as plot_tools

## CALC TOOLS
import MDDescriptors.core.calc_tools as calc_tools # Loading calculation tools


### FUNCTION TO PLOT X OF A TRAJ
def plot_X_over_time( time_array, X, X_label='Volume (nm^3)' ):
    '''
    The purpose of this function is to plot volume over timee for a trajectory.
    INPUTS:
        time_array: [np.array]
            time frames
        X: [np.array]
            an array with same size as time
        X_label: [str]
            label for x
    OUTPUTS:
        fig, ax: figure and axis for the plot
    '''
    ## CREATING FIGURE
    fig, ax = plot_tools.create_plot()
    
    ## ADDING AXIS
    ax.set_xlabel("Time frame")
    ax.set_ylabel(X_label)
    
    ## PLOTTING
    ax.plot( time_array, X, '-', linewidth=2, color='k')
    
    return fig, ax


### FUNCTION TO FIND THE BOX RANGE FOR PROBABILITY DISTRIBUTION FUNCTION
def find_box_range(map_box_length, map_box_increment):
    '''
    The purpose of this function is to take the input data and find the bounds of the box
    INPUTS:
        map_box_length: [float]
            length of the box
        map_box_increment: [float]
            box size increments
    OUTPUTS:
        box_range_dict: [dict]
            dictionary containing box range information:    
                max_bin_num: maximum bin number (integer)
                map_half_box_length: half box length in nm
                plot_axis_range: plotting range if you plotted in Cartesian coordinates, tuple (min, max)
                r_range: Range that we are interested in as a tuple (-half_box, +half_box)
    '''
    ## FINDING MAXIMUM NUMBER OF BINS
    max_bin_num = int(np.floor(map_box_length / map_box_increment))
    ## FINDING HALF BOX LENGTH
    map_half_box_length = map_box_length/2.0
    ## FINDING PLOT AXIS RANGE
    plot_axis_range = (0, max_bin_num) # Plot length for axis in number of bins
    ## FINDING RANGE OF INTEREST
    r_range = (-map_half_box_length, map_half_box_length)
    ## DEFINING DICTIONARY FOR BOX RANGE
    box_range_dict = {
            'max_bin_num': max_bin_num,
            'map_half_box_length': map_half_box_length,
            'r_range': r_range,
            'plot_axis_range': plot_axis_range,
            }
    
    return box_range_dict

### FUNCTION TO FIND THE FIRST ATOM INDEX
def find_first_atom_index(traj, residue_name):
    '''
    The purpose of this script is to find the first atom of all residues given your residue name
    INPUTS:
        traj: trajectory from md.traj
        residue_name: name of your residue
    OUTPUTS:
        first_atom_index: first atom of each residue as a list
    '''
    ## FINDING ALL RESIDUES
    residue_index = [ x.index for x in traj.topology.residues if x.name == residue_name]
    ## FINDING ATOM INDEX OF THE FIRST RESIDUE
    atom_index = [ [ atom.index for atom in traj.topology.residue(res).atoms] for res in residue_index ]
    ## FINDING FIRST ATOM FOR EACH RESIDUE
    first_atom_index = [ atom[0] for atom in atom_index ]
    return first_atom_index

### FUNCTION TO FIND DISPLACEMENTS BETWEEN SOLUTE AND SOLVENT CENTER OF MASSES
def calc_solute_solvent_displacements(traj, 
                                      solute_name, 
                                      solute_center_of_mass, 
                                      solvent_name, 
                                      solvent_center_of_mass,
                                      periodic = True,
                                      verbose = False,
                                      map_type = 'allatom'):
    '''
    This function calculates the displacements between solute and solvent center of masses using md.traj's function of displacement. First, the first atoms of the solute and solvent are found. Then, we copy the trajectory, change the coordinates to match center of masses of solute and solvent, then calculate displacements between solute and solvent.
    INPUTS:
        traj: [obj]
            trajectory from md.traj
        solute_name: [str]
            name of the solute as a string
        solute_center_of_mass: [np.array]
            numpy array containing the solute center of mass
        solvent_name: [str]
            name of the solvent as a string
        solvent_center_of_mass: [np.array]
            numpy array containing solvent center of mass
        verbose: [logical]
            True if you want to print
    OUTPUTS:
        displacements: [np.array]
            displacements as a time frame x number of displacemeny numpy float           
    '''
    ## PRINTING
    if verbose is True:
        print("\n--- CALCULATING SOLUTE(%s) and SOLVENT(%s) DISPLACEMENTS"%(solute_name, solvent_name))
    ## COPYING TRAJECTORY
    copied_traj=traj[:]
    
    ## FINDING FIRST ATOM INDEX FOR ALL SOLUTE AND SOLVENT
    Solute_first_atoms = find_first_atom_index(traj = traj, residue_name = solute_name )
    Solvent_first_atoms = find_first_atom_index(traj = traj, residue_name = solvent_name )
    
    ## CHANGING POSITION OF THE SOLUTE
    copied_traj.xyz[:, Solute_first_atoms[0]] = solute_center_of_mass[:]
    
    ## SELECTING MAP TYPE -- SEEING IF CENTER OF MASS
    if map_type == 'COM':
        ## CHANGING POSITIONS OF SOLVENT 
        copied_traj.xyz[:, Solvent_first_atoms] = solvent_center_of_mass[:]        
        ## CREATING ATOM PAIRS BETWEEN SOLUTE AND SOLVENT COM
        atom_pairs = [ [Solute_first_atoms[0], x] for x in Solvent_first_atoms]
    else: ## ALL OTHER OPTIONS
        #self.map_type == 'allatom' or self.map_type == 'allatomwithsoluteoxygen' or self.map_type == '3channel_oxy':
        ## FINDING ALL ATOMS
        num_solvent, solvent_index = calc_tools.find_total_atoms(traj, solvent_name)
        ## CREATING ATOM PAIRS BETWEEN SOLUTE AND EACH SOLVENT
        atom_pairs = [ [Solute_first_atoms[0], x] for x in solvent_index]
        
    ## FINDING THE DISPLACEMENTS USING MD TRAJ -- Periodic is true
    displacements = md.compute_displacements( traj=copied_traj, atom_pairs = atom_pairs, periodic = periodic)
    
    return displacements

### FUNCTION TO CALCULATE HISTOGRAM INFORMATION
def calc_prob_dist( r_range,
                    max_bin_num,
                    total_frames,
                    displacements, 
                    freq = 1000,
                    verbose = False):
    '''
    The purpose of this function is to take the displacements and find a histogram worth of data.
    INPUTS:
        r_range: [tuple, 2]
            range of x, y, z, e.g. (-3.75, 3.75)
        max_bin_num: [int]
            number of maximum bin
        total_frames: [int]
            total frames used to normalize the bins
        displacements: [np.array]
            numpy vector containing all displacements
        freq: [int]
            Frequency of frames you want to print information
        verbose: [logical]
            True if you want to print
    OUTPUTS:
        prob_dist: [np.array]
            probability distribution histogram
    '''
    ## PRINTING
    
    ## FINDING RANGE BASED ON CENTER OF MASS
    arange = (r_range, r_range, r_range)
    
    ### TESTING NUMPY HISTOGRAM DD
    grid, edges = np.histogramdd(np.zeros((1, 3)), bins=(max_bin_num, max_bin_num, max_bin_num), range=arange, normed=False)
    grid *=0.0 # Start at zero
    
    for frame in range(total_frames):
        ### DEFINING DISPLACEMENTS
        current_displacements = displacements[frame]
        ### USING HISTOGRAM FUNCTION TO GET DISPLACEMENTS WITHIN BINS
        hist, edges = np.histogramdd(current_displacements, bins=(max_bin_num, max_bin_num, max_bin_num), range=arange, normed=False)
        ### ADDING TO THE GRID
        grid += hist
        
        ### PRINTING
        if frame % freq == 0:
            ### CURRENT FRAME
            print("Generating histogram for frame %s out of %s, found total atoms: %s"%(frame, total_frames, np.sum(hist)))
            ''' BELOW IS A WAY TO CHECK
            ### CHECKING
            x_displacements = current_displacements[:,0]; y_displacements = current_displacements[:,1]; z_displacements = current_displacements[:,2]
            ### FINDING NUMBER OF WATER MOLECULES WITHIN A SINGLE FRAME
            count = (x_displacements > arange[0][0]) & (x_displacements < arange[0][1]) & \
            (y_displacements > arange[1][0]) & (y_displacements < arange[1][1]) & \
            (z_displacements > arange[2][0]) & (z_displacements < arange[2][1])
            print("Checking total count: %s"%(np.sum(count)))
            '''
    ### NORMALIZING THE HISTOGRAM
    prob_dist = grid / float(total_frames)

    return prob_dist

### FUNCTION TO FIND ALL ELEMENTS AND DETAILS SO THEY ARE PLOTTABLE
def find_res_coord( traj, residue_name ):
    '''
    The purpose of this function is to get the coordinates and element types of the solute so we can plot it
    INPUTS:
        traj: [obj]
            traj from md.traj
        residue_name: [str]
            name of your residue
    OUTPUTS:
        atom_elements: [list]
            Elements according to your atoms
        atom_names: [list]
            Atom names
        all_atom_coord: [np.array]
            All-atom coords of the residue
    '''
    ### FINDING ALL SOLVENT MOLECULES
    residue_index = [ x.index for x in traj.topology.residues if x.name == residue_name]
    ### FINDING ATOM INDEX OF THE FIRST RESIDUE
    atom_index = [ x.index for x in traj.topology.atoms if x.residue.index in [residue_index[0]] ]
    ### FINDING ATOM ELEMENTS
    atom_elements = [ traj.topology.atom(atom).element.symbol for atom in atom_index ]
    ### FINDING ATOM NAMES -- used for bonding information
    atom_names = [ traj.topology.atom(atom).name for atom in atom_index ]
    ### FINDING COORDINATES -- AVERAGED ACROSS SIMULATION
    all_atom_coord = traj.xyz[:, atom_index] # All frames
    return atom_elements, atom_names, all_atom_coord

####################################################
### CLASS FUNCTION TO CALCULATE SPATIAL DIST MAP ###
####################################################
class calc_np_spatial_dist_map:
    '''
    The purpose of this function is to compute the spatial distribution map 
    of a nanoparticle in solution. We are assuming you have some mixed-solvent environment 
    that contains the NP (frozen). 
    INPUTS:
        traj_data: [obj]
            Data taken from import_traj class
        ligand_names: [list]
            ligand names as a list
        solvent_list: [list]
            solvent names as a list
        itp_file: [str, default=None]
            itp file name of the solute. If None, we will search for it.
        box_length: [float]
            box length that you want the spatial distribution map to be
        box_size_increments: [float]
            box size increment size in nm
        map_type: [str]
            map type that is desired
        verbose: [str]
            True if you want to verbosely print out details
        separated_ligands: [logical]
            True if you have separated ligands (in separate itp files)
        save_disk_space: [logical]
            True if you want to save disk space
    OUTPUTS:
        
        
    FUNCTIONS:
        normalize_histogram: normalizes histogram
        plot_volume_and_box_lengths: [active] function to plot volume and box lengths over time
    '''
    ### INITIALIZING
    def __init__(self, 
                 traj_data, 
                 ligand_names,
                 solvent_list,
                 itp_file = None, 
                 box_length = 7.5, # Box length
                 box_size_increments = 0.1, # 1 A Suggested by the SILCS approach
                 map_type = 'allatom',
                 verbose = False,
                 separated_ligands = False,
                 save_disk_space = True,
                 ):
        ## STORING
        self.verbose = verbose
        self.map_type = map_type
        self.box_length = box_length
        self.box_size_increments = box_size_increments
        self.save_disk_space = save_disk_space
        
        ## PRINTING
        if self.verbose is True:
            print("**** CLASS: %s ****"%( self.__class__.__name__ ))
        
        ## DEFINING TRAJ
        traj = traj_data.traj
        
        ## STORING TIME AND VOLUME
        self.traj_time = traj.time
        self.traj_volume = traj.unitcell_volumes
        self.traj_box_lengths = traj.unitcell_lengths
        
        ### CALCULATING NANOPARTICLE STRUCTURE
        self.structure_np = nanoparticle_structure(traj_data           = traj_data,                # trajectory data
                                                   ligand_names        = ligand_names,        # ligand names
                                                   itp_file            = itp_file,                 # defines the itp file
                                                   structure_types      = None,                     # checks structural types
                                                   separated_ligands    = separated_ligands    # True if you want separated ligands 
                                                   )
        ### CHECK SOLVENT NAMES TO SEE IF THEY EXISTS IN THE TRAJECTORY
        self.solvent_name = [ each_solvent for each_solvent in solvent_list if each_solvent in traj_data.residues.keys() ]
        
        ### AVERAGE VOLUME
        self.average_volume = np.mean(traj_data.traj.unitcell_volumes)
        ### FINDING TOTAL FRAMES
        self.total_frames = len(traj)
        ### FINDING TOTAL RESIDUES
        self.num_solvent_residues = { each_solvent : traj_data.residues[each_solvent] for each_solvent in self.solvent_name }
        ### FINDING TOTAL ATOMS
        self.num_solvent_atoms = { each_solvent : calc_tools.find_total_atoms(traj,each_solvent)[0] for each_solvent in self.solvent_name }
        
        ### GETTING THE BOX LIMITS
        self.box_range_dict = find_box_range( map_box_length = box_length, 
                                              map_box_increment = box_size_increments)
        
        ### FINDING THE BOX CENTER ( CENTER OF ALL GOLD ATOMS )
        self.center_of_mass = np.mean(self.structure_np.gold_geom,axis = 1)
        
        ## DEFINING COM OF SOLVENT
        self.COM_solvent = [[] for i in range(len(self.solvent_name))] # Empty list
        
        ## DEFINING LIGAND NAME (SINGLE FOR NOW)
        lig_name = self.structure_np.ligand_names[0]
        
        ##############################################
        ### COMPUTING SOLUTE-SOLVENT DISPLACEMENTS ###
        ##############################################
        self.displacements = [] # Empty list
        for index, solvent in enumerate(self.solvent_name):
            self.displacements.append( calc_solute_solvent_displacements( traj= traj,
                                                                          solute_name = lig_name,
                                                                          solute_center_of_mass = self.center_of_mass,
                                                                          solvent_name = self.solvent_name[index],
                                                                          solvent_center_of_mass = self.COM_solvent[index],
                                                                          verbose = self.verbose
                                                                          ) )
        
        ###########################
        ### COMPUTING HISTOGRAM ###
        ###########################
        self.num_dist = [] # Empty list
        for index, solvent in enumerate(self.solvent_name):
            ## PRINTING
            if self.verbose is True:
                print("\n--- GENERATING HISOGRAM DATA FOR SOLVENT: %s ---"%(solvent))
            ## GENERATING PROBABILITY DISTRIBUTION
            self.num_dist.append(calc_prob_dist( r_range = self.box_range_dict['r_range'],
                                                  max_bin_num = self.box_range_dict['max_bin_num'],
                                                  total_frames = self.total_frames,
                                                  displacements = self.displacements[index],
                                                  verbose = self.verbose
                                                  ))
        ###########################
        ### NORMALIZE HISTOGRAM ###
        ###########################
        if self.verbose is True:
            print("--- NORMALIZING HISTOGRAM ---")
        self.normalize_histogram()
        
        ##############################
        ### STORING SOLUTE DETAILS ###
        ##############################
        ### FINDING COORDINATES AND ATOM TYPES OF THE SOLUTE TO PLOT
        self.solute_atom_elements, self.solute_atom_names, solute_full_atom_coord = find_res_coord( traj = traj_data.traj, residue_name = lig_name )
        ### FINDING PLOT CENTER
        plot_center = np.array([self.box_range_dict['map_half_box_length']]*3)
        ### MOVING FULL ATOM COORD OF SOLUTE TO CENTER
        translation = plot_center - self.center_of_mass
        solute_full_atom_coord_center =  translation[: ,np.newaxis] + solute_full_atom_coord
        self.solute_atom_coord = solute_full_atom_coord_center[-1] / float(self.box_size_increments) # Last frame
            
        ## DEFINING SOLUTE STRUCTURE USED FOR PLOTTING
        self.solute_structure = self.structure_np.itp_file
        ## ADDING PLOT AXIS RANGE
        self.plot_axis_range = (0, self.box_range_dict['max_bin_num'])
        
        ## CLEARING UP DATA
        if self.save_disk_space is True:
            self.clean_data()
        
        return
    
    ### FUNCTION TO CLEAN DATA    
    def clean_data(self):
        '''
        This function cleans the data that is useless
        '''
        ## REMOVING SOME DATA TO SAVE SPACE
        self.COM_solvent = []; self.COM_solute = []; self.displacements = []
        ## CLEARING UP ALL POSITIONS
    
    ### FUNCTION TO HISTOGRAM BASED ON PROBABILITY DENSITY
    def normalize_histogram(self):
        '''
        The purpose of this function is to normalize the histogram. 
        INPUTS:
            self: class property
        OUTPUTS:
            self.prob_dist_norm: Probability density normalized by the bulk. We calculate this from the probability distribution by:
                prob_dist / box_volume / bulk density (ANALOGOUS TO RADIAL DISTRIBUTION FUNCTION)
            As a result, the probability density should normalize to 1 at far regions away!
        '''
        if self.map_type == 'COM':
            total_num_solvent_points = [ self.num_solvent_residues[solvent_name] for solvent_name in self.solvent_name ]
        elif self.map_type == 'allatom':
            ## GETTING TOTAL NUMBER OF SOLVENT POINTS
            total_num_solvent_points = [ self.num_solvent_atoms[solvent_name] for solvent_name in self.solvent_name ]
        ## RECALCULATING PROBABILITY DISTRIBUTION (I.E. NORMALIZATION)
        self.prob_dist = [ self.num_dist[each_solvent_index]\
                                                   / float((self.box_size_increments**3)) \
                                                   / float((total_num_solvent_points[each_solvent_index]/self.average_volume)) \
                                                   for each_solvent_index in range(len(self.solvent_name)) ]
        return
        
    
    ### FUNCTION TO PLOT VOLUME AND BOX LENGTHS
    def plot_volume_and_box_lengths(self):
        '''
        The purpose of this function is to plot volume and box lengths.
        INPUTS:
            self: [obj]
                class property
        OUTPUTS:
            
        '''
        ## VOLUME
        fig, ax = plot_X_over_time( time_array = self.traj_time,
                                    X = self.traj_volume,
                                    X_label='Volume (nm^3)',)
        
        ## LENGTHS
        fig, ax = plot_X_over_time( time_array = self.traj_time,
                                    X = self.traj_box_lengths[:,0],
                                    X_label='X Length (nm)',)
        
        
        return
        

#%% MAIN SCRIPT
if __name__ == "__main__":
    
    ### DIRECTORY TO WORK ON
    analysis_dir=r"190906-Most_likely_NP_mixed_solvents" # Analysis directory
    
    specific_dir="Mostlikelynp_EAM_300.00_K_2_nmDIAM_C11COOH_CHARMM36jul2017_Trial_1_likelyindex_1_frame_10000-methanol_50_massfrac_300"
    # "Mostlikelynp_EAM_300.00_K_2_nmDIAM_dodecanethiol_CHARMM36jul2017_Trial_1_likelyindex_1_frame_10000-methanol_50_massfrac_300"
    
    analysis_dir=r"190906-Most_likely_NP_mixed_solvents_new_solvs" # Analysis directory
    
    specific_dir="Mostlikelynp_EAM_300.00_K_2_nmDIAM_C11COOH_CHARMM36jul2017_Trial_1_likelyindex_1_frame_10000-tetrahydrofuran_90_massfrac_300"
    # "Mostlikelynp_EAM_300.00_K_2_nmDIAM_dodecanethiol_CHARMM36jul2017_Trial_1_likelyindex_1_frame_10000-methanol_50_massfrac_300"
    
    path2AnalysisDir=os.path.join( r"R:\scratch\nanoparticle_project\simulations", 
                                   analysis_dir,
                                   specific_dir)
    
    ## CHECKING PATH
    path2AnalysisDir = initialize.checkPath2Server(path2AnalysisDir)
    
    ### DEFINING FILE NAMES
    gro_file=r"sam_prod.gro" # Structural file
    xtc_file=r"sam_prod.xtc" # r"mixed_solv_prod_last_90_ns_center_rot_trans_center_prog_rot_trans_center.xtc" # Trajectory
    # xtc_file=r"sam_prod_10ps_timesteps.xtc" # r"mixed_solv_prod_last_90_ns_center_rot_trans_center_prog_rot_trans_center.xtc" # Trajectory
    # itp_file=r"tBuOH.itp" # For binding information
    
    ### LOADING TRAJECTORY
    traj_data = import_tools.import_traj( directory = path2AnalysisDir, # Directory to analysis
                 structure_file = gro_file, # structure file
                  xtc_file = xtc_file, # trajectories
                  )
    
    #%%
    ## RUNNING FILE
    ### DEFINING INPUT DETAILS
    input_details={ 'traj_data'          : traj_data, # Traj data
                    'ligand_names'       : ['DOD','COO'], # Solute of interest tBuOH
                    'solvent_list'       : ['HOH', 'MET', 'THF'] , # Solvent of interest HOH 'HOH' , 'GVLL'
                    'box_length'         : 7.5, # nm box length
                    'box_size_increments': 0.1, # box cell increments
                    'map_type'           : 'allatom', # mapping type: 'COM' or 'allatom'
                    'verbose'            : True,
                    'itp_file'           : 'sam.itp',   
                    'separated_ligands'  : False,
                   }
    
    ## RUNNING 
    np_spat_dist = calc_np_spatial_dist_map( **input_details )
    
    #%%
    # np_spat_dist.plot_volume_and_box_lengths()
    
    
    