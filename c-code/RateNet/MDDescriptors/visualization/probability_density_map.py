# -*- coding: utf-8 -*-
"""
probability_density_map.py
The purpose of this program is to generate probability density maps that can find specific solvent molecules. 
IMPORTANT NOTE: To use this code, we will need gmx trjconv to eliminate problems with molecules moving (and similarly rotating)

Before running the code, you will need to center and run rot+trans fitting:
    ### FOR 90 NS
    gmx trjconv -s mixed_solv_prod.tpr -f mixed_solv_prod_10_ns_whole.xtc -o mixed_solv_prod_last_90_ns.xtc -b 110000 -e 200000
    Response 0
    gmx trjconv -s mixed_solv_prod.tpr -f mixed_solv_prod_last_90_ns.xtc -o mixed_solv_prod_last_90_ns_whole.xtc -pbc mol -center
    Response tBuOH 0
    gmx trjconv -s mixed_solv_prod.tpr -f mixed_solv_prod_last_90_ns_whole.xtc -o mixed_solv_prod_last_90_ns_center_rot_trans_center.xtc -fit rot+trans -center
    Response tBuOH tBuOH 0

Author: Alex K. Chew (Created on 01/08/2018)

INPUTS:
	1. Gro file
	2. Xtc file
	3. Solute
	4. Solvent of interest
	5. Box size of interest
OUTPUTS:
	1. Visualization of the probability distribution as a function of isovalue
	
ALGORITHM:
	1. Load gro and xtc file
	2. Find solute of interest
	3. Find center of mass of the solute
	4. Dimensionalize space around the solute's location
	5. Loop through each frame:
		a. Calculate center of mass of all solvent molecules
		b. Count the number of solvent within each box
		c. Add to counter
	6. After looping, normalize by:
		a. Number of frames
	7. Plotting
		a. Generate plotting functions for the solute
		b. Plot iso-surface of probability distribution
Output to DX file for VMD visualization

REFERENCES:
    mayavi reference: http://docs.enthought.com/mayavi/mayavi/auto/examples.html
    opendx reference: http://opendx.org/index2.php
    python script for opendx conversion(?): https://github.com/MDAnalysis/GridDataFormats/blob/master/gridData/OpenDX.py
    MDAnalysis tool kit for finding densitites: https://www.mdanalysis.org/docs/documentation_pages/analysis/density.html
    Animation python tools: http://zulko.github.io/blog/2014/11/29/data-animations-with-python-and-moviepy/

*** UPDATES ***
180225 - Made the spatial distribution maps mayavi compatible
180226 - Added debugging functions -- corrected spatial distribution function
180227 - Creating class: calc_prob_density_map
180228 - Creating class: plot_prob_density_map
180302 - Fixing plotting mechanism for dual plotting
180307 - Updated classes so they are not imported incorrectly
180328 - Updated class to include 'allatom' and 'COM' options
"""
### IMPORTING MODULES
import numpy as np
import mdtraj as md # to import xtc and run analysis
import MDDescriptors.core.initialize as initialize # Checks file path
import MDDescriptors.core.import_tools as import_tools # Loading trajectory details
import MDDescriptors.core.calc_tools as calc_tools # Loading calculation tools
import MDDescriptors.core.read_write_tools as read_write_tools
import sys

###########################################################
### CLASS FUNCTION TO CALCULATE PROBABILITY DENSITY MAP ###
###########################################################
class calc_prob_density_map:
    '''
    The purpose of this function is to take the trajectory data and calculate a probability density map between the solute and solvent.
    INPUTS:
        traj_data: Data taken from import_traj class
        solute_itp_file: Name of solute ITP file -- should be within the same directory as the trajectory information
        input_kwargs: input arguments, should be a dictionary
            'Solute': Solute name (single)
            'Solvent': List of solvents you are interested in -- will calculate probability distribution for each
            'map_box_size': The length of how large you want your box to be in nm
            'map_box_increment': Increments for the box in nm
            'map_type':
                'COM': center of mass of solute to center of mass solvent. Probability distribution is the probability of finding the COM of the solvent
                'allatom': all solvent atoms are taken into account. Probability distribution is the probability of finding any atom of the solvent
    OUTPUTS:
        ## STORAGE OF INPUT INFO
            self.solute_name: name of the solute (string)
            self.solvent_name: list of the solvents (list)
            self.map_box_size: nanometer length of the desired box in nm (scalar)
            self.map_box_increment: box increment size in nm (scalar)
            
        ## TRAJECTORY INFORMATION
            self.total_frames: total frames for the trajectory
            self.average_volume: ensemble average volume in nm^3
            self.num_solvent_residues: list of total number of residues of each solvent
                e.g. [15, 310] can mean 15 water, and 310 cosolvent
            self.num_solvent_atoms: similar to residues, but for total solvent atoms
            
        ## BOX RANGE
            self.max_bin_num: maximum bin number (integer)
            self.map_half_box_length: half box length in nm
            self.plot_axis_range: plotting range if you plotted in Cartesian coordinates, tuple (min, max)
            self.r_range: Range that we are interested in as a tuple (-half_box, +half_box)
        
        ## CENTER OF MASS
            self.COM_solute: Center of mass of solute per frame (NUMPY: FRAME X CENTER OF MASS) (e.g  numpy (9001, 1, 3))
            self.COM_solute_avg: Average COM of solute per frame
            self.COM_solute_std: Standard deviation of the COM of solute per frame
            self.COM_solvent: List of center of mass of solvent per frame (e.g. [ numpy(9001, 1511, 3) ])
            
        ## DISPLACEMENTS
            self.displacements: List of displacements between each solvent system (e.g. [ numpy(9001, 266, 3])]) <-- FRAMES X NUM_SOLVENT X COORDINATES
        
        ## SOLUTE INFORMATION (FOR PLOTTING LATER)
            self.itp_file_path: Full path to itp file of the solute
            self.solute_structure: ITP information using "extract_itp" class
            self.solute_atom_elements: atom elements (e.g. ['C', 'C', ...])
            self.solute_atom_names: atom names (e.g. ['C1', 'C2', ...])
            self.solute_full_atom_coord: NumPy of the atomic coordinates across all frames (e.g. shape 9001, 15, 3) <-- frames, number of atoms, coordinates
            self.solute_atom_coord: Solute coordinate correctly expanded using the map's box increment. Currently, we use the last frame of the solute
    
        ## FUNCTIONS:
            find_box_range: Finds the box range given the input
            find_solute_COM: Finds center of mass of the solute
            find_solvent_COM: Finds center of mass of the solvent
            find_first_atom_index: Finds first atom index given the trajectory and residue name
            calc_solute_solvent_displacements: Calculates solute and solvent displacements using mdtraj.compute_displacements
            calc_prob_dist: Calculates probability distribution based on the displacements
            find_solute_coord: finds the coordinates of the solute in a given trajectory [ staticmethod ]
            print_summary: Prints summary of what was done
            
        ## ACTIVE FUNCTION:
            calc_prob_density_dist: calculates the probability density distribution. You will need to call this to use it!
        
        ## ALGORITHM:
            - Check solvent names
            - Calculate average volume
            - Find total frames
            - Find number of solvent residues (for printing)
            - Getting box range based on the inputs
    '''
    ### INITIALIZING
    def __init__(self, traj_data, solute_itp_file, **input_kwargs):
        ### PRINTING
        print("**** CLASS: calc_prob_density_map ****")
        
        ### EXTRACTION OF DETAILS
        self.solute_name = input_kwargs['Solute']
        self.solvent_name = input_kwargs['Solvent']
        self.map_box_size = input_kwargs['map_box_size']
        self.map_box_increment = input_kwargs['map_box_increment']
        try:
            self.map_type = input_kwargs['map_type']
        except NameError:
            self.map_type = 'COM' # Default center of mass
        
        ### DEFINING TRAJECTORY INFORMATION
        traj = traj_data.traj
        
        ### CHECK IF SOLUTE EXISTS IN TRAJECTORY
        if self.solute_name not in traj_data.residues.keys():
            print("ERROR! Solute (%s) not available in trajectory. Stopping here to prevent further errors. Check your 'Solute' input!")
            sys.exit()
            
        ### CHECK SOLVENT NAMES TO SEE IF THEY EXISTS IN THE TRAJECTORY
        self.solvent_name = [ each_solvent for each_solvent in self.solvent_name if each_solvent in traj_data.residues.keys() ]
        
        ### AVERAGE VOLUME
        self.average_volume = np.mean(traj_data.traj.unitcell_volumes)
        
        ### FINDING TOTAL FRAMES
        self.total_frames = len(traj)
        
        ### FINDING TOTAL RESIDUES
        self.num_solvent_residues = [traj_data.residues[each_solvent] for each_solvent in self.solvent_name]
        
        ### FINDING TOTAL ATOMS
        self.num_solvent_atoms = [ calc_tools.find_total_atoms(traj,each_solvent)[0] for each_solvent in self.solvent_name ]
        
        ### FINDING BOX LIMITS
        self.find_box_range()
        
        ### FINDING CENTER OF MASSES OF SOLUTE
        print("\n---- CALCULATING CENTER OF MASS ----\n")
        self.COM_solute = self.find_solute_COM( traj=traj, residue_name=self.solute_name)
        self.COM_solute_avg = np.mean(self.COM_solute, axis = 0) # Averaging center of mass
        self.COM_solute_std = np.std(self.COM_solute, axis = 0) # Standard deviation of the COM
        
        ## CHECK IF CENTER OF MASSES NEED TO BE CALCULATED
        self.COM_solvent = [[] for i in range(len(self.solvent_name))] # Empty list
        if self.map_type == 'COM':
            ### FINDING CENTER OF MASS OF SOLVENT
            for index, solvent in enumerate(self.solvent_name):
                self.COM_solvent[index] = self.find_solvent_COM( traj= traj, residue_name= solvent )
                
        
        ### FINDING DISPLACEMENTS FOR EACH SOLVENT
        self.displacements = [] # Empty list
        for index, solvent in enumerate(self.solvent_name):
            self.displacements.append( self.calc_solute_solvent_displacements( traj= traj,
                                                                              solute_name = self.solute_name,
                                                                              solute_center_of_mass = self.COM_solute_avg,
                                                                              solvent_name = self.solvent_name[index],
                                                                              solvent_center_of_mass = self.COM_solvent[index],
                                                                              ) )
        
        ### USING HISTOGRAM METHODS TO CALCULATE PROBABILITY DISTRIBUTION
        self.prob_dist = [] # Empty list
        for index, solvent in enumerate(self.solvent_name):
            self.prob_dist.append(self.calc_prob_dist( 
                                                       solvent_name = self.solvent_name[index],
                                                       displacements = self.displacements[index]
                                                       ))
        
        # -----------------------#
        ### SOLUTE INFORMATION ###
        # -----------------------#
        ### DEFINING SOLUTE ITP FILE
        self.itp_file_path = initialize.checkPath2Server(traj_data.directory + '/' + solute_itp_file)
        
        ### READING ITP FILE
        self.solute_structure = read_write_tools.extract_itp(self.itp_file_path)
        
        ### FINDING COORDINATES AND ATOM TYPES OF THE SOLUTE TO PLOT
        self.solute_atom_elements, self.solute_atom_names, self.solute_full_atom_coord = self.find_solute_coord( traj = traj_data.traj, residue_name = self.solute_name )
        
        # ----------------------------#
        ### MOVING SOLUTE TO CENTER ###
        # ----------------------------#
        ### FINDING PLOT CENTER
        plot_center = np.array([self.map_half_box_length]*3)
        
        ### MOVING FULL ATOM COORD OF SOLUTE TO CENTER
        translation = plot_center - self.COM_solute_avg[0]
        self.solute_full_atom_coord_center =  translation + self.solute_full_atom_coord
        self.solute_atom_coord = self.solute_full_atom_coord_center[-1] / float(self.map_box_increment) # Last frame
        
        ### CREATING DICTIONARY TO STORE MULTIPLE ATOM COORDINATES -- USEFUL FOR VISUALIZING SOLUTES AT DIFFERENT LOCATIONS
        self.solute_atom_coord_dict={
                                    'last_frame': self.solute_atom_coord,                                                               # last frame
                                    'avg': np.mean(self.solute_full_atom_coord_center,axis=0) / float(self.map_box_increment)        # Average of the solute atom positions
                                    }
        
        # self.solute_atom_coord = np.mean(solute_full_atom_coord_center,axis=0) / float(prob_density.map_box_increment) # <-- untick if you want average
        
        ### PRINTING SUMMARY
        self.print_summary()
        
        ## REMOVING SOME DATA TO SAVE SPACE
        self.COM_solvent = []; self.COM_solute = []; self.displacements = []
        
        return

    ### FUNCTION TO PRINT SUMMARY
    def print_summary(self):
        '''
        The purpose of this function is to simply summarize what was done
        INPUTS:
            self: class object
        OUTPUTS:
            Void
        '''
        ### SUMMARY
        print("\n----- SUMMARY -----")
        print("SOLUTE: %s"%(self.solute_name))
        print("SOLVENTS: %s"%(', '.join(self.solvent_name)))
        print("BOX SIZE: %s nm"%(self.map_box_size))
        print("BOX INCREMENT: %s nm"%(self.map_box_increment))
        print("TOTAL FRAMES: %s"%(self.total_frames))
        print("MAPPYING TYPE: %s"%(self.map_type))
      
    ### FUNCTION TO FIND THE BOX RANGE FOR PROBABILITY DISTRIBUTION FUNCTION
    def find_box_range(self):
        '''
        The purpose of this function is to take the input data and find the bounds of the box
        INPUTS:
            self: class object
        OUTPUTS:
            self.max_bin_num: maximum bin number (integer)
            self.map_half_box_length: half box length in nm
            self.plot_axis_range: plotting range if you plotted in Cartesian coordinates, tuple (min, max)
            self.r_range: Range that we are interested in as a tuple (-half_box, +half_box)
        '''
        ## FINDING MAXIMUM NUMBER OF BINS
        self.max_bin_num = int(np.floor(self.map_box_size / self.map_box_increment))
        ## FINDING HALF BOX LENGTH
        self.map_half_box_length = self.map_box_size/2.0
        ## FINDING PLOT AXIS RANGE
        self.plot_axis_range = (0, self.max_bin_num) # Plot length for axis in number of bins
        ## FINDING RANGE OF INTEREST
        self.r_range = (-self.map_half_box_length, self.map_half_box_length)
        return
        
    ### FUNCTION TO FIND THE CENTER OF MASS GIVEN A RESIDUE NAME
    ## ASSUMES SINGLE SOLUTE
    def find_solute_COM( self, traj, residue_name ):
        '''
        This function simply finds the center of mass given the residue name
        INPUTS:
            self: class object
            traj: traj from md.traj
            residue_name: Name of residue as a string
        OUTPUTS:
            center_of_mass: center of mass as a numpy arrays
        '''
        ## FINDING ALL RESIDUES OF THE TYPE
        residue_index = [ x.index for x in traj.topology.residues if x.name == residue_name]
        atom_index = [ x.index for x in traj.topology.atoms if x.residue.index in residue_index ]
        ## FINDING ATOM NAMES
        atom_names = [ traj.topology.atom(current_atom_index).name for current_atom_index in atom_index]
        ## FINDING SOLUTE COM
        center_of_mass = calc_tools.find_center_of_mass(traj=traj, residue_name=residue_name, atom_names=atom_names)
        return center_of_mass

    ### FUNCTION TO FIND CENTER OF MASS OF SOLVENT (ASSUMING THERE IS AT LEAST ONE!)
    def find_solvent_COM(self,traj, residue_name):
        '''
        This function finds the solvent center of mass across all trajectories
        INPUTS:
            self: class object
            traj: trajectory from md.traj
            residue_name: name of solvent as a string
        OUTPUTS:
            center_of_mass: numpy array with each trajectory and the center of mass
        '''
        ## FINDING ALL SOLVENT MOLECULES
        residue_index = [ x.index for x in traj.topology.residues if x.name == residue_name]
        ## FINDING ATOM INDEX OF THE FIRST RESIDUE
        atom_index = [ x.index for x in traj.topology.atoms if x.residue.index in [residue_index[0]] ]
        ## FINDING NAMES JUST FOR THE FIRST RESIDUE
        atom_names = [ traj.topology.atom(current_atom_index).name for current_atom_index in atom_index]
        ## FINDING CENTER OF MASS
        center_of_mass = calc_tools.find_center_of_mass(traj=traj, residue_name=residue_name, atom_names=atom_names)
    
        return center_of_mass

    ### FUNCTION TO FIND THE FIRST ATOM OF EACH RESIDUE
    @staticmethod
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
    def calc_solute_solvent_displacements( self, traj,solute_name, solute_center_of_mass, solvent_name, solvent_center_of_mass ):
        '''
        This function calculates the displacements between solute and solvent center of masses using md.traj's function of displacement. First, the first atoms of the solute and solvent are found. Then, we copy the trajectory, change the coordinates to match center of masses of solute and solvent, then calculate displacements between solute and solvent.
        INPUTS:
            traj: trajectory from md.traj
            solute_name: name of the solute as a string
            solute_center_of_mass: numpy array containing the solute center of mass
            solvent_name: name of the solvent as a string
            solvent_center_of_mass: numpy array containing solvent center of mass
        OUTPUTS:
            displacements: displacements as a time frame x number of displacemeny numpy float           
        '''
        print("--- CALCULATING SOLUTE(%s) and SOLVENT(%s) DISPLACEMENTS"%(solute_name, solvent_name))
        ## COPYING TRAJECTORY
        copied_traj=traj[:]
        
        ## FINDING FIRST ATOM INDEX FOR ALL SOLUTE AND SOLVENT
        Solute_first_atoms = self.find_first_atom_index(traj = traj, residue_name = solute_name )
        Solvent_first_atoms = self.find_first_atom_index(traj = traj, residue_name = solvent_name )
        
        ## CHANGING POSITION OF THE SOLUTE
        copied_traj.xyz[:, Solute_first_atoms] = solute_center_of_mass[:]
        
        if self.map_type == 'COM':
            ## CHANGING POSITIONS OF SOLVENT 
            copied_traj.xyz[:, Solvent_first_atoms] = solvent_center_of_mass[:]        
            ## CREATING ATOM PAIRS BETWEEN SOLUTE AND SOLVENT COM
            atom_pairs = [ [Solute_first_atoms[0], x] for x in Solvent_first_atoms]
        elif self.map_type == 'allatom':
            ## FINDING ALL ATOMS
            num_solvent, solvent_index = calc_tools.find_total_atoms(traj, solvent_name)
            ## CREATING ATOM PAIRS BETWEEN SOLUTE AND EACH SOLVENT
            atom_pairs = [ [Solute_first_atoms[0], x] for x in solvent_index]
            
        ## FINDING THE DISPLACEMENTS USING MD TRAJ -- Periodic is true
        displacements = md.compute_displacements( traj=copied_traj, atom_pairs = atom_pairs, periodic = True)
        
        return displacements
    
    ### FUNCTION TO CALCULATE HISTOGRAM INFORMATION
    def calc_prob_dist( self, solvent_name, displacements, freq = 1000 ):
        '''
        The purpose of this function is to take the displacements and find a histogram worth of data
        INPUTS:
            self: class object
            solvent_name: Name of the solvent
            displacements: numpy vector containing all displacements
            freq: Frequency of frames you want to print information
        OUTPUTS:
        '''
        ## PRINTING
        print("\n--- GENERATING HISOGRAM DATA FOR SOLVENT: %s ---"%(solvent_name))
        ## FINDING RANGE BASED ON CENTER OF MASS
        arange = (self.r_range, self.r_range, self.r_range)
        
        ### TESTING NUMPY HISTOGRAM DD
        grid, edges = np.histogramdd(np.zeros((1, 3)), bins=(self.max_bin_num, self.max_bin_num, self.max_bin_num), range=arange, normed=False)
        grid *=0.0 # Start at zero
        
        for frame in range(self.total_frames):
            ### DEFINING DISPLACEMENTS
            current_displacements = displacements[frame]
            ### USING HISTOGRAM FUNCTION TO GET DISPLACEMENTS WITHIN BINS
            hist, edges = np.histogramdd(current_displacements, bins=(self.max_bin_num, self.max_bin_num, self.max_bin_num), range=arange, normed=False)
            ### ADDING TO THE GRID
            grid += hist
            
            ### PRINTING
            if frame % freq == 0:
                ### CURRENT FRAME
                print("Generating histogram for frame %s out of %s, found total atoms: %s"%(frame, self.total_frames, np.sum(hist)))
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
        PROB_DIST = grid / float(self.total_frames)

        return PROB_DIST
    
    ### FUNCTION TO GET SOLUTE ELEMENT/COORD
    @staticmethod
    def find_solute_coord( traj, residue_name ):
        '''
        The purpose of this function is to get the coordinates and element types of the solute so we can plot it
        INPUTS:
            traj: traj from md.traj
            residue_name: name of your residue
        OUTPUTS:
            atom_elements: Elements according to your atoms
            atom_coord: Coordinates of the solute  (FIRST FRAME)
            (NOTE! THIS IS AVERAGED ACROSS THE ENTIRE SIMULATION) <-- depreciated
            all_atom_coord: Coordinates of all the solute positions
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
    
    ### FUNCTION TO CALCULATE PROBABILITY DENSITY DISTRIBUTION
    def calc_prob_density_dist(self):
        '''
        The purpose of this script is to calculate the probability density distribution. Note that this function is not automatically called since it would be expensive to save all that data (since we already have the probability distribution 3d matrix!!!)
        Instead, use this class by: self.calc_prob_density_dist(), which will create a self variable: self.prob_dens_dist
        This way, we will not have to save additional data!
        INPUTS:
            self: class property
        OUTPUTS:
            self.prob_dens_dist: Probability density normalized by the bulk. We calculate this from the probability distribution by:
                prob_dist / box_volume / bulk density (ANALOGOUS TO RADIAL DISTRIBUTION FUNCTION)
            As a result, the probability density should normalize to 1 at far regions away!
        '''
        ### CALCULATING PROBABILITY DENSITY DISTRIBUTION
        print("*** CALCULATING PROBABILITY DENSITY DISTRIBUTION ***")
        print("*** SAVING INTO prob_dist VARIABLE ***")
        ## SIMPLY PROBABILITY DIST / BIN SIZE / (BULK DENSITY OF SOLVENT)
        if self.map_type == 'COM':
            total_num_solvent_points = [ self.num_solvent_residues[each_solvent_index] for each_solvent_index in range(len(self.solvent_name)) ]
        elif self.map_type == 'allatom':
            total_num_solvent_points = [ self.num_solvent_atoms[each_solvent_index] for each_solvent_index in range(len(self.solvent_name)) ]
        ## RECALCULATING PROBABILITY DISTRIBUTION (I.E. NORMALIZATION)
        self.prob_dist = [ self.prob_dist[each_solvent_index]/float((self.map_box_increment**3))/ \
                          float((total_num_solvent_points[each_solvent_index]/self.average_volume)) \
                          for each_solvent_index in range(len(self.solvent_name)) ]
        return
 
#%% MAIN SCRIPT
if __name__ == "__main__":
    
    ### DIRECTORY TO WORK ON
    analysis_dir=r"180302-Spatial_Mapping" # Analysis directory
    specific_dir="TBA\\TBA_50_GVL" # Directory within analysis_dir r"mdRun_363.15_6_nm_tBuOH_100_WtPercWater_spce_Pure"
    # specific_dir=r"Planar_310.15_ROT_TMMA_10x10_CHARMM36_withGOLP" # Directory within analysis_dir
    path2AnalysisDir=r"R:\scratch\SideProjectHuber\Analysis\\" + analysis_dir + '\\' + specific_dir # PC Side
    
    ## CHECKING PATH
    path2AnalysisDir = initialize.checkPath2Server(path2AnalysisDir)
    
    ### DEFINING FILE NAMES
    gro_file=r"mixed_solv_prod.gro" # Structural file
    xtc_file=r"mixed_solv_prod_last_90_ns_center_rot_trans_center.xtc" # r"mixed_solv_prod_last_90_ns_center_rot_trans_center_prog_rot_trans_center.xtc" # Trajectory
    itp_file=r"tBuOH.itp" # For binding information
    
    ### LOADING TRAJECTORY
    traj_data = import_tools.import_traj( directory = path2AnalysisDir, # Directory to analysis
                 structure_file = gro_file, # structure file
                  xtc_file = xtc_file, # trajectories
                  )
    #%%
    ### DEFINING INPUT DETAILS
    input_details={ 'Solute':             'tBuOH', # Solute of interest tBuOH
                    'Solvent':   ['HOH', 'GVLL'] , # Solvent of interest HOH 'HOH' , 'GVLL'
                    'map_box_size':           3.0, # nm box length in all three dimensions
                    'map_box_increment':      0.1, # box cell increments
                    'map_type':             'allatom', # mapping type: 'COM' or 'allatom'
                   }
            
    ### CALCULATING THE PROBABILITY DENSITY MAP
    prob_density = calc_prob_density_map(traj_data=traj_data,
                                         solute_itp_file = itp_file,
                                         **input_details)
    
    
    #%%
    
    
    def _putline(*args):
        """
        Generate a line to be written to a cube file where 
        the first field is an int and the remaining fields are floats.
        
        params:
            *args: first arg is formatted as int and remaining as floats
        
        returns: formatted string to be written to file with trailing newline
        """
        s = "{0:^ 8d}".format(args[0])
        s += "".join("{0:< 12.6f}".format(arg) for arg in args[1:])
        return s + "\n"
        
    # Function taken from: https://gist.github.com/aditya95sriram/8d1fccbb91dae93c4edf31cd6a22510f
    def write_cube(data, meta, fname):
        """
        Write volumetric data to cube file along
        
        params:
            data: volumetric data consisting real values
            meta: dict containing metadata with following keys
                atoms: list of atoms in the form (mass, [position])
                org: origin
                xvec,yvec,zvec: lattice vector basis
            fname: filename of cubefile (existing files overwritten)
        
        returns: None
        """
        with open(fname, "w") as cube:
            # first two lines are comments
            cube.write(" Cubefile created by cubetools.py\n  source: none\n")
            natm = len(meta['atoms'])
            nx, ny, nz = data.shape
            cube.write(_putline(natm, *meta['org'])) # 3rd line #atoms and origin
            cube.write(_putline(nx, *meta['xvec']))
            cube.write(_putline(ny, *meta['yvec']))
            cube.write(_putline(nz, *meta['zvec']))
            for atom_mass, atom_pos in meta['atoms']:
                cube.write(_putline(atom_mass, *atom_pos)) #skip the newline
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        if (i or j or k) and k%6==0:
                            cube.write("\n")
                        cube.write(" {0: .5E}".format(data[i,j,k]))
    
    
    ## CONVERSION FROM NM TO BOHR
    nm_to_bohr = 18.897161646321 # 1 nm to this  bohr
    
    ## DEFINING META
    meta = { 'atoms': [(1, list(each_atom/nm_to_bohr) ) for each_atom in prob_density.solute_atom_coord], # 
             'org': (0, 0, 0), 
             'xvec': (3/nm_to_bohr, 0, 0),
             'yvec': (0, 3/nm_to_bohr, 0),
             'zvec': (0, 0, 3/nm_to_bohr),
             }
    
    ## WRITING CUBE FILE
    write_cube(data = prob_density.prob_dist[0], 
               meta = meta, 
               fname = "test.cube")
    
    
    
    
    #%%
    
    from MDDescriptors.visualization.cube import write_file, Bohr
    
    #%%
    
    
    ## CONVERSION FROM NM TO BOHR
    nm_to_bohr = 18.897161646321 # 1 nm to this  bohr
    
    ## DEFINING SPACING
    spacing = np.array([ prob_density.map_box_increment ] * 3) / nm_to_bohr
    
    ## RUNNING WRITE FILE
    
    write_file(filename = "test.cube",
               group = None,
               grid_size = prob_density.max_bin_num,
               spacing = spacing,
               scalars = prob_density.prob_dist[0],
               atomic_numbers=None,
               normalize=False)
    
    
    


    #%% PLOTTING PROBABILY DIST
    ### CALCULATING PROBABILITY DENSITIES
    prob_density.calc_prob_density_dist()
    from MDDescriptors.visualization.plot_probability_density_map import plot_prob_density_map, MAP_INPUT
    ### RUNNING PROBABILITY DENSITY MAPS
    prob_density_map = plot_prob_density_map(prob_density = prob_density,
                                             **MAP_INPUT)
    