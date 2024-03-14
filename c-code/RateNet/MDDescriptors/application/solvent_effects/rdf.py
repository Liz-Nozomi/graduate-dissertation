#/usr/bin/python
"""
rdf.py
The purpose of this script is to calculate the radial distribution function given the trajectory.
INPUTS:
    trajectory: traj from md.traj.
    input_details={
                    'Solute'        : 'tBuOH',           # Solute of interest
                    'Solvent'       : ['HOH', 'GVLL'],   # Solvents you want radial distribution functions for
                    'bin_width'     : 0.02,              # Bin width of the radial distribution function
                    'cutoff_radius' : 2.00,              # Radius of cutoff for the RDF (OPTIONAL)
                    'want_oxy_rdf'  : True,              # True if you want oxygen rdfs
                    }
    
OUTPUTS:
    rdf class from "calc_rdf" class function

FUNCTIONS:
    find_solute_atom_names_by_type: finds atom names of a solute
    find_atom_index_from_res_index: finds atom indexes based on residue index
    create_copy_traj_new_distances: creates a copy of trajectories with new solute/solvent positions
    calc_cumulative_num_atoms_vs_r: function that computes cumulative number of atoms with respect to radius
    
CLASSES:
    calc_rdf: calculates rdf of your system

AUTHOR(S):
    Alex K. Chew (alexkchew@gmail.com)
    
** UPDATES **
20180321 - AKC - Added functionality to calculate RDF at each individual oxygens
20180506 - AKC - Added calculation of ensemble volume in the RDF
20181003 - AKC - Adding functionality to calculate RDFs in multiple time steps
20181212 - AKC - Adding functionality to compute mole fractions
20181213 - AKC - Completion of functionality addition for mole fractions
"""
                      
### IMPORTING MODULES
import MDDescriptors.core.import_tools as import_tools # Loading trajectory details
import MDDescriptors.core.calc_tools as calc_tools # Loading calculation tools
from MDDescriptors.core.check_tools import check_exists
import mdtraj as md # Running calculations
import sys
import numpy as np

### FUNCTION TO FIND ATOM NAMES OF THE SOLUTE
def find_solute_atom_names_by_type(solute_atom_names, atom_element = 'O',):
    '''
    The purpose of this script is to find the solute atom names
    INPUTS:
        solute_atom_names: [list]
            list of solute atom names
        atom_element: [str]
            element you are interested in ,e.g. 'O'
    OUTPUTS:
        atom_names: [list]
            Names of the atoms within the solute of interest
    '''
    ### FINDING ALL OF THE ATOM TYPES
    atom_names = [ each_name for each_name in solute_atom_names if atom_element in each_name ] # Returns list of names, e.g. ['O1']
    return atom_names

### FUNCTION TO FIND ALL ATOM INDEXES BASED ON RESIDUE INDEX
def find_atom_index_from_res_index(traj, res_index):
    '''
    This function finds the atom index based on the residue index. 
    INPUTS:
        traj: trajectory from md.load
        res_index: residue index as a form of a list
    OUTPUTS:
        atom_index: atom index as a list of list
            e.g. [[2,3,4] , [5,6,7]] stands for atoms for each residue
    '''
    return [ [atom.index for atom in traj.topology.residue(res).atoms] for res in res_index]

### FUNCTION TO CREATE COPY OF TRAJECTORIES T
def create_copy_traj_new_distances(traj, solute_residue_index, solvent_residue_index,
                                         solute_COM, solvent_COM  ):
    '''
    The purpose of this function is to create trajectories with solute and solvent's center of mass (COM). Once the trajectory is generated, you can run computations on it (e.g. RDFs, distances, etc.)
    INPUTS:
        traj: [md.traj object]
            trajectory file from md.load
        solute_residue_index: [np.array, shape=(num_residues, 1)]
            index of solutes
        solute_COM: [np.array, shape=(time_frames, num_solute_residue)]
            solute center of mass
        solvent_residue_index: [np.array, shape=(num_residues, 1)]
            index of residues
        solvent_COM: [np.array, shape=(time_frames, num_solvent_residue)]
            solvent center of mass
    OUTPUTS:
        copied_traj: [md.traj object]
            copied trajectory with the positions of the first atom of each solute/solvent changed to the center of mass of the molecule
        solute_first_atom_index: [np.array, shape=(num_residues, 1)]
            First atom index of each solute residue, which can be used for computations
        solvent_first_atom_index: [np.array, shape=(num_residues, 1)]
            First atom index of each solvent residue, which can be used for computations
    '''    
    ### COPYING TRAJECTORY
    copied_traj=traj[:]
    ### FINDING ALL ATOM INDEXES OF EACH RESIDUE
    ## SOLUTE
    solute_atom_index  = find_atom_index_from_res_index(traj, res_index = solute_residue_index )
    ## SOLVENT
    solvent_atom_index = find_atom_index_from_res_index(traj, res_index = solvent_residue_index )
    
    ## FINDING FIRST ATOM INDEX OF EACH RESIDUE
    solute_first_atom_index = [ solute_atom_index[each_res][0] for each_res in range(len(solute_atom_index))]
    solvent_first_atom_index = [ solvent_atom_index[each_res][0] for each_res in range(len(solvent_atom_index))]
    
    ## COPYING COM TO TRAJECTORY
    copied_traj.xyz[:, solute_first_atom_index] = solute_COM[:]
    copied_traj.xyz[:, solvent_first_atom_index] = solvent_COM[:]

    return copied_traj, solute_first_atom_index, solvent_first_atom_index


### FUNCTION TO COMPUTE R_RANGE AND BINS
def compute_range_and_bins(r_range = None, bin_width=0.005, n_bins=None ):
    '''
    The purpose of this function is to create range and bins used for other calculations. The idea here is that we need histograms that need a range of values.
    Note: These commands are the same as the ones in MDTraj, RDF function. 
    INPUTS:
        r_range: array-like, shape=(2,), optional, default=(0.0, 1.0)
            Minimum and maximum radii.
        bin_width : float, optional, default=0.005
            Width of the bins in nanometers.
        n_bins : int, optional, default=None
            The number of bins. If specified, this will override the `bin_width`
             parameter.
    OUTPUTS:
        r_range: array-like, 
            range of r values
        n_bins : int, optional, default=None
            The number of bins.
    '''
    if r_range is None:
        r_range = np.array([0.0, 1.0])
    if n_bins is not None:
        n_bins = int(n_bins)
        if n_bins <= 0:
            raise ValueError('`n_bins` must be a positive integer')
    else:
        n_bins = int((r_range[1] - r_range[0]) / bin_width)
    return r_range, n_bins

### FUNCTION TO COMPUTE THE CUMULATIVE ATOMS WITH RESPECT TO R FOR A TRAJECTORY
def calc_cumulative_num_atoms_vs_r( traj, solute_first_atom_index, solvent_first_atom_index,
                                    r_range, n_bins, periodic = True):
    '''
    The purpose of this function is to compute the cumulative number of atoms with respect to radius. The idea here is that we want the number of atoms at an r distance away. Here, we measure cumulative number of atoms.
    NOTE:
        - We are assuming you already computed some center of mass; or you have a reference in mind between the solute and solvent
        
    INPUTS:
        traj: [class]
            trajectory from md.traj
        solute_first_atom_index: [np.array, shape=(num_residues, 1)]
            First atom index of each solute residue, which can be used for computations
        solvent_first_atom_index: [np.array, shape=(num_residues, 1)]
            First atom index of each solvent residue, which can be used for computations
        periodic: [logical, default=True]
            True if you want periodic conditions taken into account
        r_range: array-like, 
            range of r values
        n_bins : int, optional, default=None
            The number of bins.
    OUTPUTS:
        cumulative_atoms_wrt_r: [np.array, shape=(dist, 1)
            cumulative sum of number of atoms with respect to radius, r
            For example, you may get: [0, 0, 1...], this means that at radius_0, you have 0 of the atoms across your trajectory. Then, 0 at radius_2, etc. This is a cumulative sum, so the numbers should always be increasing.
        r: [np.array]
            radius array taken from np.histogram
    '''
    ## COMPUTING DISTANCES BETWEEN ATOM PAIRS
    distances = calc_tools.calc_pair_distances_between_two_atom_index_list( traj = traj,
                                                                            atom_1_index = solute_first_atom_index,
                                                                            atom_2_index = solvent_first_atom_index,
                                                                            periodic = periodic,
                                                                           ) ## OUTPUT SHAPE: ( NUM_FRAMES X NUM_SOLUTE X NUM_SOLVENT )


    ## COMPUTING HISTOGRAM WITH RESPECT TO DISTANCES
    num_atoms, edges = np.histogram(distances, range=r_range, bins=n_bins)
    #   num_atoms returns total number of atoms as a function of r (not taking into account division across the total frames)
    #   edges returns the left-most edge of the bins
    
    ## COMPUTING R VECTOR
    r = 0.5 * (edges[1:] + edges[:-1])

    ## COMPUTING THE CUMULATIVE NUMBER OF ATOMS
    cumulative_atoms_wrt_r = np.cumsum(num_atoms)  # Finds cumulative sums with respect to r
    
    return cumulative_atoms_wrt_r, r


#################################################################
### CLASS FUNCTION TO CALCULATE RADIAL DISTRIBUTION FUNCTIONS ###
#################################################################
class calc_rdf:
    '''
    The purpose of this function is to take the trajectory data and calculate the radial distribution map between the solute and solvent. 
    NOTE: We currently assume you care about center of mass of the solute to solvent. *** WILL UPDATE IN THE FUTURE ***
    INPUTS:
        traj_data: [class]
            Data taken from import_traj class
        solute_name: [list] 
            list of the solutes you are interested in (Note that this script will look for all possible solutes)
        solvent_name: [list] 
            list of solvents you are interested in.
        bin_width: [float] 
            Bin width of the radial distribution function
        cutoff_radius: [float] 
            Radius of cutoff for the RDF (OPTIONAL) [Default: None]
        split_rdf: [int, default = 1]
            Number of times you want to split your RDF and re-calculate all parameters. By default, setting this to 1 will not attempt to truncate your data and calculate the RDF. 
        frames: [list, default=[]]
            frames you want to calculate RDFs for. Useful when debugging to see when the RDF does in fact converge.
            Note: if -1 is included, then the last frame will by used
        want_reverse_frames: [logical, default = False]
            If true, we will do the reverse. For frames, we start at time=0 to time=t. If this is True, then we start at time=-1 to time=t
        want_oxy_rdf: [logical, default=False] 
            True or False if you want oxygen rdfs of the solute
        want_oxy_rdf_all_solvent: [logical, default=False]
            True if you want solute oxygen rdfs to all solvent atom types
        want_solute_all_solvent_rdf: [logical, default = False]
            True if you want the solvents to be considered as a single solvent, then calcualte an RDF for it
        want_composition_with_r: [logical, default=False]
            True if you want composition with respect to r. This will give you mole/mass fraction of water, and other solvents.
        save_disk_space: [logical, default=True] 
            True if you want to save disk space by removing the following variables: [Default: True]
            solute_COM, solvent_COM
    OUTPUTS:
        ### STORING INPUTS
            self.bin_width, self.cutoff_radius, self.want_oxy_rdf, self.save_disk_space
            self.solute_name: [list] 
                updated list of the solutes you are interested in (Note that this script will look for all possible solutes)
            self.solvent_name: [list] 
                updated list of solvents you are interested in.
            self.frames: [list]
                updated frames (-1 are converted to total frames)
            self.split_rdf: [int]
                Number of times to split the trajectory to calculate the RDFs
            self.solvent_masses: [list]
                solvent masses for each solvent
        ### RDF INFORMATION
            self.r_range: [np.array, shape=2]
                r range for RDFs, etc.
            self.n_bins: [int]
                number of bins for the histograms
        ### RESIDUE INDEXES
            self.total_solutes: [list]
                Total number of solute residues
            self.solute_res_index: Residue index for solutes as a list
            self.total_solvent: Total number of solvents as a list
                e.g. [13, 14] <-- first solvent has 13 residues, second solvent has 14 resiudes
            self.solvent_res_index: list of lists for the residue index of each solvent
        ### TRAJECTORY INFORMATION
            self.volume: [float] ensemble volume in nm^3
            self.total_frames: total frames in trajectory
        ### SOLUTE/SOLVENT INFORMATION
            self.solute_atom_names: solute atom names as a lists
            self.solvent_atom_names: solvent atom names as a list of lists 
                (NOTE: You can add as many as you'd like, we will check the trajectory to make sure it makes sense)
            self.solute_COM: Center of mass of the solute
            self.solvent_COM: Center of mass of the solvent
        ### IF YOU WANTED MOLE FRACTION DATA
            self.comp_r_vec: [np.array, shape=(num_bins, 1)]
                radius vector used to plot the mass fraction as a function of radius
            self.comp_solvent_vs_r_by_number: [np.array, shape=(num_solute, num_solvent, r)]
                composition of solvent for all frames at radius r
            self.comp_solvent_vs_r_by_mass: [np.array, shape=(num_solute, num_solvent, r)]
                mass composition of all solvents for all frames
            self.comp_total_num_solvents_with_respect_to_r:  [np.array, shape=(num_solute, r)]
                total number of solvents for each radius r
            self.comp_total_mass_solvents_with_respect_to_r:  [np.array, shape=(num_solute, r)]
                total mass of the solvents for each radius r
            self.comp_water_mole_frac_by_r: [np.array, shape=(r,1)]
                water mole fraction as a function of radius r
            self.comp_water_mass_frac_by_r: [np.array, shape=(r,1)]
                water mass fraction as a function of radius r
        ### RDF DETAILS
            self.rdf_r: list of list containing r values for RDF
            self.rdf_g_r: list of list containing g(r) values for RDF
            ## FOR OXYGEN (if want_oxy_rdf is True)
                self.rdf_oxy_r: list of list containing r values for RDF [ [r1a, r2a, r3a ], [r1b, r2b, r3b]], where a and b are cosolvents
                self.rdf_oxy_g_r: similar to previous, but the g(r)
                self.rdf_oxy_names: names of each r, g_r
            ## FOR OXYGEN WITH ALL SOLVENTS
                self.rdf_oxy_all_solvents: list
                    list of each solute, followed by each solvent key, then each oxygen on the solute
            ## FOR MULTIPLE FRAMES
                self.rdf_frames_r: [list]
                    Large list: [solute][solvent][frame] -> [ frame, r ]
                self.rdf_frames_g_r: [list]
                    Large list: [solute][solvent][frame] -> [ frame, g_r ]
            ## FOR MULTIPLE SPLITTING
                self.rdf_split_data: [dict]:
                    dictionary containing rdf data for multiple splitting information. Useful if you want RDF of a specific 
                        'COM': center of mass data
                        'OXY_RDF': oxygen rdf data
                        Each dictionary is structured as follows: (e.g. self.rdf_split_data['COM'])
                            'g_r'
                            'r'
                                [solute][solvent][split_0]
                                [solute][solvent][split_1]
                    
                            
    FUNCTIONS:
        find_residue_index: function to get the residue index of solute and solvent
        find_solute_atom_names: Finds solute names based on an element (e.g. Oxygen, etc.)
        calc_rdf_solute_atom: calculates rdf based on a solute atom
        print_summary: prints summary of what was done
    '''
    #####################
    ### INITIALIZATION###
    #####################
    def __init__(self, traj_data, 
                 solute_name, 
                 solvent_name, 
                 bin_width, 
                 frames = [], 
                 split_rdf = 1,
                 want_reverse_frames = False, 
                 cutoff_radius = None, 
                 want_oxy_rdf = False, 
                 want_oxy_rdf_all_solvent = False,
                 want_solute_all_solvent_rdf = False, 
                 want_composition_with_r = False,
                 save_disk_space=True):
        ### PRINTING
        print("**** CLASS: %s ****"%(self.__class__.__name__))
        
        ### DEFINING VARIABLES
        self.solute_name = solute_name
        self.solvent_name = solvent_name
        self.bin_width = bin_width
        self.cutoff_radius = cutoff_radius
        self.frames = frames
        self.want_oxy_rdf = want_oxy_rdf
        self.want_oxy_rdf_all_solvent = want_oxy_rdf_all_solvent
        self.want_reverse_frames = want_reverse_frames
        self.want_solute_all_solvent_rdf = want_solute_all_solvent_rdf
        self.want_composition_with_r = want_composition_with_r
        self.save_disk_space = save_disk_space
        self.split_rdf = split_rdf
        
        ### TRAJECTORY
        traj = traj_data.traj
        
        ### FINDING ENSEMBLE AVERAGE VOLUME (USED FOR DENSITIES)
        self.volume = calc_tools.calc_ensemble_vol( traj )
        
        ## COMPUTING BINS
        self.r_range, self.n_bins = compute_range_and_bins( r_range = (0, self.cutoff_radius),
                                                            bin_width = self.bin_width,
                                                           )
        
        ### FINDING TOTAL FRAMES
        self.total_frames = len(traj)
        
        ### CHECK SOLVENT NAMES TO SEE OF THEY EXISTS IN THE TRAJECTORY
        self.solute_name = [ each_solute for each_solute in self.solute_name if each_solute in traj_data.residues.keys() ]
        self.num_solutes = len(self.solute_name)
        
        ### CHECK SOLVENT NAMES TO SEE OF THEY EXISTS IN THE TRAJECTORY
        self.solvent_name = [ each_solvent for each_solvent in self.solvent_name if each_solvent in traj_data.residues.keys() ]
        self.num_solvents = len(self.solvent_name)
        
        ### COMPUTING MASSES
        self.solvent_masses = [ calc_tools.calc_mass_from_residue_name(traj = traj, residue_name = each_solvent) for each_solvent in self.solvent_name]
        
        ## CORRECTING FOR FRAMES
        if len(self.frames) > 0:
            ## CHECKING FRAMES
            self.frames = [ each_frame if each_frame != -1 else self.total_frames for each_frame in self.frames ]
        
        ### CHECK IF SOLUTE EXISTS IN TRAJECTORY
        if len(self.solute_name) == 0 or len(self.solvent_name) == 0:
            print("ERROR! Solute or solvent specified not available in trajectory! Stopping here to prevent further errors.")
            print("Residue names available: ")
            print(traj_data.residues.keys())
            print("Input solute names: ")
            print(solute_name)
            print("Input solvent names: ")
            print(solvent_name)
            sys.exit()
        
        ### FINDING RESIDUE INDEX
        self.find_residue_index(traj=traj)
        
        ### FINDING ATOM NAMES
        self.solute_atom_names =  [ calc_tools.find_atom_names(traj, residue_name=each_name) for each_name in self.solute_name]
        self.solvent_atom_names = [ calc_tools.find_atom_names(traj, residue_name=each_name) for each_name in self.solvent_name]
        
        ### FINDING CENTER OF MASSES
        # self.solute_COM = calc_tools.find_center_of_mass(traj, residue_name = self.solute_name, atom_names = self.solute_atom_names)
        self.solute_COM = [ calc_tools.find_center_of_mass(traj, residue_name=self.solute_name[index], atom_names = self.solute_atom_names[index] ) \
                            for index in range(self.num_solutes)]
        self.solvent_COM = [ calc_tools.find_center_of_mass(traj, residue_name=self.solvent_name[solvent_index], atom_names = self.solvent_atom_names[solvent_index] ) \
                            for solvent_index in range(self.num_solvents)]
        
        ### CALCULATING MOLE FRACTIONS
        if self.want_composition_with_r: 
            self.calc_composition_of_water_vs_r(traj = traj)
            
        
        ### CREATING STORAGE VECTOR FOR RDFS
        self.rdf_r, self.rdf_g_r = [], []
        if self.want_oxy_rdf is True:
            self.rdf_oxy_r = []
            self.rdf_oxy_g_r = []
            self.rdf_oxy_names = [ find_solute_atom_names_by_type(self.solute_atom_names[index], atom_element = 'O') for index in range(self.num_solutes) ]
        if self.want_solute_all_solvent_rdf is True:
            ## PRINTING
            print("Solute to all solvent molecules set to True!")
            print("This script will compute solute-solvent RDFs, where all solvents are treated equally")
            ## GENERATING STORAGE VECTORS
            self.rdf_solute_solvent_r, self.rdf_solute_solvent_g_r = [], []
            ## CONCATENATING ALL SOLVENTS
            solute_all_solvent_res_index = np.concatenate(self.solvent_res_index) # Residue index that is concatenated
            solute_all_solvent_COM = np.concatenate(self.solvent_COM, axis=1) # Solvent index that is concatendated
            
        ### CREATING STORAGE IF YOU WANT PER FRAME RDFs
        if len(frames) > 0:
            self.rdf_frames_r= []
            self.rdf_frames_g_r= []
            
        ### CREATING STORAGE DICTIONARY FOR SPLIT RDF
        if self.split_rdf > 1:
            print("split_rdf function turned on to value %d"%( self.split_rdf ) )
            print("This function now will split the RDFs into %d chunks"%(self.split_rdf) )
            self.rdf_split_data ={ each_key: {      'r': [], 
                                                  'g_r': [] } for each_key in ['COM', 'OXY_RDF'] }
            '''{
            'COM': {}, # CENTER OF MASS
            'oxy_RDF': {}, # OXYGEN RDFS
            }'''
        
        ## LOCATING ATOM INDEXES
        if self.want_oxy_rdf_all_solvent is True:
            ## LOCATING ATOM NAMES
            self.solvent_atom_names = { each_solvent: calc_tools.find_atom_names( traj=traj, 
                                                                                  residue_name=each_solvent) for each_solvent in self.solvent_name }
            ## LOCATING ATOM INDEX FOR EACH NAME
            self.solvent_atom_indexes = { each_solvent: {  atom_name : calc_tools.find_atom_index(traj = traj,
                                                                                                  atom_name = self.solvent_atom_names[each_solvent][atom_index],
                                                                                                  resname = each_solvent
                                                                                                  )
                                                                    for atom_index, atom_name in enumerate(self.solvent_atom_names[each_solvent])
                                                                    } for each_solvent in self.solvent_atom_names
                                                                    }
            ## STORING SOLVENT
            self.rdf_oxy_all_solvents = []
                
            
        ###################################################################
        ################### LOOPING THROUGH EACH SOLUTE ###################
        ###################################################################
        for each_solute in range(self.num_solutes):
            ## GENERATING TEMPORARY VARIABLES
            solute_r= [] 
            solute_g_r = []
            if self.want_oxy_rdf is True:
                solute_oxy_r = []
                solute_oxy_g_r = []
            if len(frames) > 0:
                solute_frames_r = []
                solute_frames_g_r = []
            ## SPLIT RDF STORAGE VARIABLES
            if self.split_rdf > 1:
                solute_split_data = { each_key: {'r':[], 
                                 'g_r': [],}
                                  for each_key in list(self.rdf_split_data.keys() ) }
                # solute_split_g_r = []
            
                
            
            ## DEFINING VARIABLES FOR SOLUTES
            solute_residue_index = self.solute_res_index[each_solute]
            solute_COM = self.solute_COM[each_solute]
            solute_res_name = self.solute_name[each_solute]
            if self.want_oxy_rdf is True or self.want_oxy_rdf_all_solvent is True:
                solute_atom_names = self.rdf_oxy_names[each_solute]
            
            ## LOCATING ATOM INDEXES
            if self.want_oxy_rdf_all_solvent is True:
                rdf_oxy_all_solvents_each_solvent = []

                
            
            ####################################################################
            ################### LOOPING THROUGH EACH SOLVENT ###################
            ####################################################################
            for each_solvent in range(self.num_solvents):
                ## PRINTING
                print("\n--- CALCULATING RDF FOR %s TO %s ---"%( self.solute_name[each_solute], self.solvent_name[each_solvent]   ) )
                ## DEFINING VARIABLES FOR SOLVENTS
                solvent_residue_index = self.solvent_res_index[each_solvent]
                solvent_COM = self.solvent_COM[each_solvent]
                
                ### CALCULATING RDF
                r, g_r, solvent_first_atom_index = self.calc_rdf_com( traj = traj,                                          # Trajectory
                                                                       solute_residue_index     = solute_residue_index,     # Solute residue index
                                                                       solute_COM               = solute_COM,               # Solute center of mass
                                                                       solvent_residue_index    = solvent_residue_index,    # Solvent residue index
                                                                       solvent_COM              = solvent_COM,              # Solvent center of mass
                                                                       bin_width                = self.bin_width,           # Bin width for RDF
                                                                       cutoff_radius            = self.cutoff_radius,       # Cut off radius for RDf 
                                                                       periodic                 = True,                     # Periodic boundary conditions 
                                                                      )
                ## APPENDING R, G(R)
                solute_r.append(r)
                solute_g_r.append(g_r)
                
                ### CREATING STORAGE IF YOU WANT PER FRAME RDFs
                if len(frames) > 0:
                    ### CALCULATING RDF
                    frames_r, frames_g_r, solvent_first_atom_index  =self.calc_rdf_com( traj = traj,                                          # Trajectory
                                                                           solute_residue_index     = solute_residue_index,     # Solute residue index
                                                                           solute_COM               = solute_COM,               # Solute center of mass
                                                                           solvent_residue_index    = solvent_residue_index,    # Solvent residue index
                                                                           solvent_COM              = solvent_COM,              # Solvent center of mass
                                                                           bin_width                = self.bin_width,           # Bin width for RDF
                                                                           cutoff_radius            = self.cutoff_radius,       # Cut off radius for RDf 
                                                                           frames                   = self.frames,              # Frames you are interested in
                                                                           want_reverse_frames      = self.want_reverse_frames, # If you want reverse frames
                                                                           periodic                 = True,                     # Periodic boundary conditions 
                                                                          )
                    ## APPENDING
                    solute_frames_r.append(frames_r)
                    solute_frames_g_r.append(frames_g_r)
                
                
                ### RUNNING COM WITH SPLIT
                if self.split_rdf > 1:
                    ### DEFINING STATIC AND NONSTATIC VARIABLES
                    split_variable_dict = { ## DYNAMIC VARIABLES THAT NEEDS TO  BE SPLITTED
                                            'traj'          : traj,
                                            'solute_COM'    : solute_COM,
                                            'solvent_COM'   : solvent_COM,
                            }
                    static_variable_dict = {
                                            'solute_residue_index'      :   solute_residue_index,
                                            'solvent_residue_index'     :   solvent_residue_index,
                                            'bin_width'                 :   self.bin_width,
                                            'cutoff_radius'             :   self.cutoff_radius,
                                            'periodic'                  :   True

                            }
                    ### CALCULATING RDF
                    rdf_split_output = calc_tools.split_general_functions( input_function = self.calc_rdf_com, 
                                                                     split_variable_dict = split_variable_dict,
                                                                     static_variable_dict = static_variable_dict,
                                                                     num_split = self.split_rdf,
                                                                    )
                    ### STORING ARRAY
                    solute_split_data['COM']['r'].append( [ each_split[0] for each_split in rdf_split_output ])
                    solute_split_data['COM']['g_r'].append( [ each_split[1] for each_split in rdf_split_output ])
                    
                ### RUNNING RDFS FOR OXYGENS [ OPTIONAL ]
                if self.want_oxy_rdf is True:
                    ## COMPUTING RDF FOR OXYGENS
                    oxy_r, oxy_g_r = self.calc_rdf_solute_atom(traj = traj,
                                                               solute_res_name = solute_res_name,
                                                               atom_names = solute_atom_names, 
                                                               solvent_first_atom_index = solvent_first_atom_index,
                                                               solvent_COM = solvent_COM,
                                                               bin_width = self.bin_width,
                                                               cutoff_radius = self.cutoff_radius,
                                                               )
                    ## APPENDING
                    solute_oxy_r.append(oxy_r)
                    solute_oxy_g_r.append(oxy_g_r)
                    
                    ### RUNNING COM WITH SPLIT
                    if self.split_rdf > 1:
                        ### DEFINING STATIC AND NONSTATIC VARIABLES
                        split_variable_dict_oxy = { ## DYNAMIC VARIABLES THAT NEEDS TO  BE SPLITTED
                                                'traj'          : traj,
                                                'solvent_COM'   : solvent_COM,
                                }
                        static_variable_dict_oxy = {
                                                'solvent_first_atom_index'      :   solvent_first_atom_index,
                                                'solute_res_name'               :   solute_res_name,
                                                'bin_width'                 :   self.bin_width,
                                                'cutoff_radius'             :   self.cutoff_radius,
                                                'periodic'                  :   True,
                                                'atom_names'                : solute_atom_names,
    
                                }
                    
                        ### CALCULATING RDF
                        rdf_split_output_oxy = calc_tools.split_general_functions( input_function = self.calc_rdf_solute_atom, 
                                                                         split_variable_dict = split_variable_dict_oxy,
                                                                         static_variable_dict = static_variable_dict_oxy,
                                                                         num_split = self.split_rdf,
                                                                        )
                        ### STORING ARRAY
                        solute_split_data['OXY_RDF']['r'].append( [ each_split[0] for each_split in rdf_split_output_oxy ])
                        solute_split_data['OXY_RDF']['g_r'].append( [ each_split[1] for each_split in rdf_split_output_oxy ])
                    
                ## LOOPING THROUGH RDF 
                if self.want_oxy_rdf_all_solvent is True:
                    ## RDF FOR ALL SOLVENT ATOMS
                    ## PRINTING
                    print("--- Computing oxygen to rdf all solvents ---")                                            
                    ## COMPUTING RDF for each solvent index
                    oxy_all_solvent_rdfs = { self.solvent_name[each_solvent] + '_' + each_combination: self.calc_rdf_solute_atom(traj = traj,
                                                                       solute_res_name = solute_res_name,
                                                                       atom_names = solute_atom_names, 
                                                                       solvent_first_atom_index = self.solvent_atom_indexes[self.solvent_name[each_solvent]][each_combination],
                                                                       solvent_COM = None,
                                                                       bin_width = self.bin_width,
                                                                       cutoff_radius = self.cutoff_radius,
                                                                       )
                                              for each_combination in self.solvent_atom_indexes[self.solvent_name[each_solvent]] }
                    ## STORING
                    rdf_oxy_all_solvents_each_solvent.append(oxy_all_solvent_rdfs)
                    
                        
            ## APPENDING FOR EACH SOLUTE
            self.rdf_oxy_all_solvents.append(rdf_oxy_all_solvents_each_solvent)
            
            ### STORING RDFS FOR EACH SOLUTE
            self.rdf_r.append(solute_r)
            self.rdf_g_r.append(solute_g_r)
            
            ### FOR OXYGENS
            if self.want_oxy_rdf is True:
                self.rdf_oxy_r.append(solute_oxy_r)
                self.rdf_oxy_g_r.append(solute_oxy_g_r)
            ### CREATING STORAGE IF YOU WANT PER FRAME RDFs
            if len(frames) > 0:
                self.rdf_frames_r.append(solute_frames_r)
                self.rdf_frames_g_r.append(solute_frames_g_r)
            ## STORING IF YOU WANT SPLIT RDF
            if self.split_rdf > 1:
                ## COM DATA
                self.rdf_split_data['COM']['r'].append(solute_split_data['COM']['r'])
                self.rdf_split_data['COM']['g_r'].append(solute_split_data['COM']['g_r'])
                ## OXY DATA
                self.rdf_split_data['OXY_RDF']['r'].append(solute_split_data['OXY_RDF']['r'])
                self.rdf_split_data['OXY_RDF']['g_r'].append(solute_split_data['OXY_RDF']['g_r'])
                
            ### FOR SOLUTE-SOLVENT FULL INTERACTIONS
            if self.want_solute_all_solvent_rdf is True:
                ### CALCULATING RDF
                solute_all_solvent_r, solute_all_solvent_g_r, _  =self.calc_rdf_com( traj = traj,                                          # Trajectory
                                                   solute_residue_index     = solute_residue_index,     # Solute residue index
                                                   solute_COM               = solute_COM,               # Solute center of mass
                                                   solvent_residue_index    = solute_all_solvent_res_index,    # Solvent residue index
                                                   solvent_COM              = solute_all_solvent_COM,              # Solvent center of mass
                                                   bin_width                = self.bin_width,           # Bin width for RDF
                                                   cutoff_radius            = self.cutoff_radius,       # Cut off radius for RDf 
                                                   periodic                 = True,                     # Periodic boundary conditions 
                                                  )
                ### STORING
                self.rdf_solute_solvent_r.append(solute_all_solvent_r)
                self.rdf_solute_solvent_g_r.append(solute_all_solvent_g_r)
            
        ### PRINTING SUMMARY
        self.print_summary()
        
        ### SETTING SOME VALUES TO NONE TO SAVE DISK SPACE
        if self.save_disk_space is True:
            self.solute_COM, self.solvent_COM = None, None
        
        return
    
    ### FUNCTION TO PRINT THE SUMMARY
    def print_summary(self):
        '''
        This function simply prints the summary of what was done
        '''
        print("\n----- SUMMARY -----")
        print("Radial distribution functions were calculated!")
        print("SOLUTE: %s"%(', '.join(self.solute_name) ) )
        print("SOLVENTS: %s"%(', '.join(self.solvent_name)))
        print("--- RDF DETAILS ---")
        print("BIN WIDTH: %s nm"%(self.bin_width))
        print("RADIUS OF OMISSION: %.3f nm"%(self.cutoff_radius))
        print("TOTAL FRAMES: %s "%(self.total_frames))
        print("WANT OXYGEN RDFS? ---> %s"%(self.want_oxy_rdf))
        return
    
    ### FUNCTION TO GET RESIDUE INDICES
    def find_residue_index(self, traj):
        '''
        The purpose of this function is to get the residue index for the solute and solvent
        INPUTS:
            self: class object
            traj: trajectory from md.traj
        OUTPUTS:
            self.total_solutes: [list]
                Total number of solute residues
            self.solute_res_index: [list]
                Residue index for solutes as a list
            self.total_solvent: [list]
                Total number of solvents as a list
                    e.g. [13, 14] <-- first solvent has 13 residues, second solvent has 14 resiudes
            self.solvent_res_index: [list]
                Residue index of each solvent as a list
        '''
        ### FINDING RESIDUE INDICES
        ## SOLUTE
        self.total_solutes, self.solute_res_index = calc_tools.find_multiple_residue_index(traj, residue_name_list=self.solute_name)
        ## SOLVENT
        self.total_solvent, self.solvent_res_index = calc_tools.find_multiple_residue_index(traj, residue_name_list=self.solvent_name)
        return
    
    ### FUNCTION TO RUN RDF OF OXYGENS
    @staticmethod
    def calc_rdf_solute_atom(traj, 
                             solute_res_name, 
                             atom_names, 
                             solvent_first_atom_index = None, 
                             solvent_COM = None, 
                             bin_width=0.02, 
                             cutoff_radius=None, 
                             periodic = True ):
        '''
        The purpose of this function is to calculate a RDF based on a solute atom (e.g. all the oxygens, etc.)
        INPUTS:
            self: class property
            traj: traj from md.traj
            solute_res_name: [str]
                solute residue name
            atom_names: atom names within the solute that you are interested in (e.g. ['O1'] ).
                Note: This is taken from the "find_solute_atom_names" function
            solvent_first_atom_index: [list]
                First index of solvents taken from calc_rdf_com
            solvent_COM: [np.array, shape=(time_frame, residue_index)]
                Center of mass of all the solvents
            bin_width: [float]
                width of the bin
            cutoff_radius: [float] 
                radius of cutoff (None by default)
            periodic: [logical, default=True]
                True or false, if you want periodic boundary conditions to be taken into account
        OUTPUTS:
            r, g_r: rdfs for each of the elements in a form of a list
            atom_names: Names of each of the atoms for the element list
        '''        
        ### COPYING TRAJECTORY
        copied_traj=traj[:]
        
        ### COPYING THE SOLVENT CENTER OF MASSES TO TRAJECTORY
        if solvent_COM is not None:
            copied_traj.xyz[:, solvent_first_atom_index] = solvent_COM[:]
        
        ### FINDING ALL ATOM INDEXES FOR EACH OF THE ATOM NAMES
        atom_index = [ atom.index for atom_name in atom_names for atom in traj.topology.atoms if atom.residue.name == solute_res_name and atom.name == atom_name]
        
        ### NOW, CREATING ATOM PAIRS
#        atom_pairs_list =np.array([ [ [atom_index[each_atom_name], interest_atom_indexes] for interest_atom_indexes in solvent_first_atom_index ]
#                      for each_atom_name in range(len(atom_names))])
        
        ## CREATING ATOM PAIR LIST
        atom_pairs_list = [ calc_tools.create_atom_pairs_list( atom_1_index_list = [each_atom],
                                                               atom_2_index_list = solvent_first_atom_index) for each_atom in atom_index ]
            
        ### CREATING EMPTY R AND G_R
        r, g_r = [], []
        
        ### LOOPING THROUGH EACH atom
        for each_atom in range(len(atom_names)):
            ## FINDING ATOM PAIRS
            atom_pairs = atom_pairs_list[each_atom]
            
            ## CALCULATING RDF
            element_r, element_g_r = md.compute_rdf(traj = copied_traj,
                     pairs = atom_pairs,
                     r_range=[0, cutoff_radius], # Cutoff radius
                     bin_width = bin_width,
                     periodic = periodic, # periodic boundary is on
                     )
            ## APPENDING
            r.append(element_r)
            g_r.append(element_g_r)
            
        return r, g_r

    ### FUNCTION TO COMPUTE MOLE FRACTIONS OF WATER AS A FUNCTION OF RADIUS
    def calc_composition_of_water_vs_r(self, traj, water_solvent_name='HOH'):
        '''
        The purpose of this function is to compute the mass fraction of water as a function of radius. The idea here is to get a distribution of the mass fraction of water. 
        In the bulk phase, we suspect the mass fraction to reach the mass fraction of water as a whole. We do the following:
        INPUTS:
            self: [class object]
                self object
            traj: [md.traj]
                trajectory file from md.load
            solute_residue_index: [np.array, shape=(num_residues, 1)]
                index of solutes
            solute_COM: [np.array, shape=(time_frames, num_solute_residue)]
                solute center of mass
            solvent_residue_index: [np.array, shape=(num_residues, 1)]
                index of residues
            solvent_COM: [np.array, shape=(time_frames, num_solvent_residue)]
                solvent center of mass
            water_solvent_name: [str, default='HOH']
                solvent residue name. We will use this to find the water solvent.
        OUTPUTS:
            self.comp_r_vec: [np.array, shape=(num_bins, 1)]
                radius vector used to plot the mass fraction as a function of radius
            self.comp_solvent_vs_r_by_number: [np.array, shape=(num_solute, num_solvent, r)]
                composition of solvent for all frames at radius r
            self.comp_solvent_vs_r_by_mass: [np.array, shape=(num_solute, num_solvent, r)]
                mass composition of all solvents for all frames
            self.comp_total_num_solvents_with_respect_to_r:  [np.array, shape=(num_solute, r)]
                total number of solvents for each radius r
            self.comp_total_mass_solvents_with_respect_to_r:  [np.array, shape=(num_solute, r)]
                total mass of the solvents for each radius r
            self.comp_water_mole_frac_by_r: [np.array, shape=(r,1)]
                water mole fraction as a function of radius r
            self.comp_water_mass_frac_by_r: [np.array, shape=(r,1)]
                water mass fraction as a function of radius r
        ALGORITHM:
            - Compute COM of solute
            - Compute COM of water + organic solvent (all solvents)
            - Compute distances between solute to water
            - Computer distances between solute to cosolvent
        
        '''
        ## PRINTING
        print("*** COMPUTING COMPOSITION OF SOLVENTS WITH RESPECT TO R ***")
        
        ## DEFINING R VECTOR (USED TO STORE R DISTANCES)
        self.comp_r_vec = None
        
        ## DEFINING STORAGE VECTOR
        solute_data = []
        
        ## LOOPING THROUGH EACH SOLUTE
        for each_solute in range(self.num_solutes):
            ## DEFINING STORAGE VECTOR
            solvent_data = []
            ## LOOPING THROUGH EACH SOLVENT
            for each_solvent in range(self.num_solvents):
                ### COPYING TRAJECTORY
                copied_traj, solute_first_atom_index, solvent_first_atom_index = create_copy_traj_new_distances(    
                                                                                                                    traj = traj,
                                                                                                                    solute_residue_index = self.solute_res_index[each_solute],
                                                                                                                    solvent_residue_index = self.solvent_res_index[each_solvent],
                                                                                                                    solute_COM = self.solute_COM[each_solute],
                                                                                                                    solvent_COM = self.solvent_COM[each_solvent],
                                                                                                                )
                
                ### COMPUTING CUMULATIVE ATOMS WITH RESPECT TO R
                cumulative_atoms_wrt_r, r = calc_cumulative_num_atoms_vs_r(
                                                                           traj = copied_traj, 
                                                                           solute_first_atom_index = solute_first_atom_index, 
                                                                           solvent_first_atom_index = solvent_first_atom_index, 
                                                                           periodic = True,
                                                                           r_range = self.r_range, 
                                                                           n_bins = self.n_bins )
                ## SEEING IF R VEC IS NONE
                if self.comp_r_vec is None:
                    self.comp_r_vec = r
                
                ## STORING
                solvent_data.append( cumulative_atoms_wrt_r ) # Escaping extra array index
                
            ## APPENDING TO THE SOLUTE
            solute_data.append(np.array(solvent_data))
        
        ## FINDING TOTAL ATOMS
        self.comp_solvent_vs_r_by_number = solute_data
        
        ## FINDING TOTAL MASSES
        self.comp_solvent_vs_r_by_mass = [ solute_data[each_solute] * np.array(self.solvent_masses)[:, np.newaxis ] for each_solute in range(self.num_solutes)]
        
        ## FINDING THE TOTAL NUMBER OF SOLVENTS
        self.comp_total_num_solvents_with_respect_to_r =[np.sum(self.comp_solvent_vs_r_by_number[each_solute], axis = 0) for each_solute in range(self.num_solutes)]
        self.comp_total_mass_solvents_with_respect_to_r =  [np.sum(self.comp_solvent_vs_r_by_mass[each_solute], axis = 0) for each_solute in range(self.num_solutes)]
        
        ## FINDING WATER INDEX
        water_index = self.solvent_name.index(water_solvent_name)
        
        ## IGNORING ERRORS FROM NUMPY BY DIVISION ERROR (NAN's indicate nothing is there!)
        with np.errstate(divide='ignore'):
            ##  FINDING TOTAL NUMBER OF WATER MOLECULES
            self.comp_water_mole_frac_by_r = [ self.comp_solvent_vs_r_by_number[each_solute][water_index] / self.comp_total_num_solvents_with_respect_to_r for each_solute in range(self.num_solutes) ][0]
            self.comp_water_mass_frac_by_r = [ self.comp_solvent_vs_r_by_mass[each_solute][water_index] / self.comp_total_mass_solvents_with_respect_to_r for each_solute in range(self.num_solutes) ][0]
        
        return

    
    
    ### FUNCTION TO FIND RDF BASED ON CENTER OF MASS
    @staticmethod
    def calc_rdf_com(traj, solute_residue_index, solute_COM, solvent_residue_index, solvent_COM, frames = [], 
                     want_reverse_frames = False, bin_width=0.02, cutoff_radius=None, periodic=True):
        '''
        The purpose of this function is to calculate the radial distribution function based on the center of mass. 
        We will do this by:
            1. transforming the trajectory such that a its first element is at the center of mass location.
            2. calculating the RDF based on the new trajectory
        INPUTS:
            traj: [md.traj]
                trajectory file from md.load
            solute_residue_index: [np.array, shape=(num_residues, 1)]
                index of solutes
            solute_COM: [np.array, shape=(time_frames, num_solute_residue)]
                solute center of mass
            solvent_residue_index: [np.array, shape=(num_residues, 1)]
                index of residues
            solvent_COM: [np.array, shape=(time_frames, num_solvent_residue)]
                solvent center of mass
            bin_width: [float]
                width of the bin
            cutoff_radius: [float] 
                radius of cutoff (None by default)
            periodic: [logical, default=True]
                True/False if you want periodic boundary conditions
            want_reverse_frames: [logical, default=False]
                If True, the times for frames are reversed such that we start counting from t=-1 to frame instead of t=0
        OUTPUTS:
            r:[list]
                radius vector for your radial distribution function
            g_r: [list]
                g(r), RDF
            solvent_first_atom_index: [np.array]
                first atom index of each solvent
        FUNCTIONS:
            find_atom_index_from_res_index: Finds atom index based on residue index
        '''
        ### COPYING TRAJECTORY
        copied_traj=traj[:]
        ### FINDING ALL ATOM INDEXES OF EACH RESIDUE
        ## SOLUTE
        solute_atom_index  = find_atom_index_from_res_index(traj, res_index = solute_residue_index )
        ## SOLVENT
        solvent_atom_index = find_atom_index_from_res_index(traj, res_index = solvent_residue_index )
        
        ## FINDING FIRST ATOM INDEX OF EACH RESIDUE
        solute_first_atom_index = [ solute_atom_index[each_res][0] for each_res in range(len(solute_atom_index))]
        solvent_first_atom_index = [ solvent_atom_index[each_res][0] for each_res in range(len(solvent_atom_index))]
        
        ## COPYING COM TO TRAJECTORY
        copied_traj.xyz[:, solute_first_atom_index] = solute_COM[:]
        copied_traj.xyz[:, solvent_first_atom_index] = solvent_COM[:]
        
        ## GETTING ATOM PAIRS
        atom_pairs = calc_tools.create_atom_pairs_list(solute_first_atom_index,solvent_first_atom_index)
        # DEPRECIATED VERSION: (SLOWER):
        # [ [center_atom_indexes, interest_atom_indexes] for center_atom_indexes in solute_first_atom_index 
                      # for interest_atom_indexes in solvent_first_atom_index ]
        if len(frames) == 0:
            ## CALCULATING RDF
            r, g_r = md.compute_rdf(traj = copied_traj,
                 pairs = atom_pairs,
                 r_range=[0, cutoff_radius], # Cutoff radius
                 bin_width = bin_width,
                 periodic = periodic, # periodic boundary is on
                 )
        else:
            ## PRINTING
            print("FRAMES IS NONZERO, CALCULATING RDFS FOR MULTIPLE FRAMES")
            ## CREATING EMPTY VECTORY FOR R, G(R)
            r, g_r = [], []
            ## LOOPING THROUGH EACH FRAME
            for frame_idx, each_frame in enumerate(frames):
                ## CORRECTING IF FRAME IS -1
                #if each_frame == -1:
                #    each_frame=len(traj) ## GETTING FINAL TRAJECTORY
                ## SEEING IF THE FRAMES ARE RIGHT
                if want_reverse_frames == False:
                    ## PRINTING
                    print("Working on frame index %d out of %d, going up to %d frame"%(frame_idx, len(frames), each_frame) )
                    ## DEFINING CURRENT TRAJECTORY
                    current_traj=copied_traj[:each_frame]
                else:
                    ## PRINTING
                    print("Reverse frames are on, working on frame index %d out of %d, going up from %d to %d frames"%(frame_idx, len(frames), each_frame, len(traj)) )
                    ## DEFINING CURRENT TRAJECTORY
                    current_traj=copied_traj[each_frame:]
                ## RUNNING RDF WITH DIFFERENT LENGTHS
                frame_r, frame_g_r = md.compute_rdf(traj = current_traj,
                     pairs = atom_pairs,
                     r_range=[0, cutoff_radius], # Cutoff radius
                     bin_width = bin_width,
                     periodic = periodic, # periodic boundary is on
                     )
                ## APPENDING
                r.append([each_frame, frame_r])
                g_r.append([each_frame, frame_g_r])
            
        return r, g_r, solvent_first_atom_index


#%% MAIN SCRIPT
if __name__ == "__main__":
    
    ### DIRECTORY TO WORK ON
    analysis_dir=r"181107-PDO_DEHYDRATION_FULL_DATA_300NS" # Analysis directory
    # analysis_dir=r"180316-ACE_PRO_DIO_DMSO"
    # specific_dir="HYDTEST\\HYDTEST_300.00_6_nm_xylitol_10_WtPercWater_spce_dioxane" # Directory within analysis_dir r"mdRun_363.15_6_nm_tBuOH_100_WtPercWater_spce_Pure"
    specific_dir="PDO/mdRun_433.15_6_nm_PDO_10_WtPercWater_spce_dioxane" # Directory within analysis_dir r"mdRun_363.15_6_nm_tBuOH_100_WtPercWater_spce_Pure"
    # specific_dir="PDO/mdRun_433.15_6_nm_PDO_10_WtPercWater_spce_dmso" # Directory within analysis_dir r"mdRun_363.15_6_nm_tBuOH_100_WtPercWater_spce_Pure"
    # specific_dir="PDO/mdRun_433.15_6_nm_PDO_100_WtPercWater_spce_Pure" # Directory within analysis_dir r"mdRun_363.15_6_nm_tBuOH_100_WtPercWater_spce_Pure"
    # specific_dir="ACE/mdRun_433.15_6_nm_ACE_10_WtPercWater_spce_dioxane" # Directory within analysis_dir r"mdRun_363.15_6_nm_tBuOH_100_WtPercWater_spce_Pure"
    # specific_dir=r"Planar_310.15_ROT_TMMA_10x10_CHARMM36_withGOLP" # Directory within analysis_dir
    path2AnalysisDir=r"R:\scratch\SideProjectHuber\Analysis\\" + analysis_dir + '\\' + specific_dir # PC Side
    
    ### DEFINING FILE NAMES
    gro_file=r"mixed_solv_prod.gro" # Structural file
    xtc_file=r"mixed_solv_prod.xtc" # r"mixed_solv_prod_whole_last_50ns.xtc" # r"mixed_solv_prod_last_90_ns_center_rot_trans_center_prog_rot_trans_center.xtc" # Trajectory
    # xtc_file=r"mixed_solv_last_50_ns_whole.xtc"
    xtc_file=r"mixed_solv_prod_10_ns_whole_290000.xtc"
    
    ### LOADING TRAJECTORY
    traj_data = import_tools.import_traj( directory = path2AnalysisDir, # Directory to analysis
                 structure_file = gro_file, # structure file
                  xtc_file = xtc_file, # trajectories
                  )
    
    
    #%%
    ### DEFINING INPUT DATA
    input_details={
                    'solute_name'                   : ['PDO'],           # Solute of interest as a form of a list ['XYL', 'HYD']
                    'solvent_name'                  : ['HOH', 'DIO', 'GVLL', 'dmso'],   # Solvents you want radial distribution functions for
                    'bin_width'                     : 0.02,              # Bin width of the radial distribution function
                    'cutoff_radius'                 : 2.00,              # 2.00 Radius of cutoff for the RDF (OPTIONAL)
                    'frames'                        : []   , # Frames to run, default = [] 1000, 2000
                    'split_rdf'                     :  1, # RDF splitting
                    'want_oxy_rdf'                  : True,              # True if you want oxygen rdfs
                    'want_oxy_rdf_all_solvent'      : True,
                    'want_reverse_frames'           : False,
                    'want_solute_all_solvent_rdf'   : False,         # True if you want the solute-all solvent RDFs
                    'want_composition_with_r'       : False,         # True if you want mole/mass fractions as a function of radius
                    'save_disk_space'               : False,       # False for debugging
                    }    
    ### CALLING RDF CLASS = 
    rdf = calc_rdf(traj_data, **input_details)
    
    #%%
    
    
    #%%
    
    ## PLOTTING
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot( rdf.comp_r_vec, rdf.comp_water_mass_frac_by_r[0])
    
    
    
    
    
    #%%
    
    
    rdf_split_data ={ each_key: {      'r': [], 
                                          'g_r': [] } for each_key in ['COM', 'OXY_RDF'] }
    
    solute_split_data = { each_key: {'r':[], 
                                     'g_r': [],}
                                      for each_key in list(rdf_split_data.keys() ) }
    #%%
    ## TESTING
    traj_split = calc_tools.split_list( alist = traj_data.traj, wanted_parts = 2 )
    
    ## SPLITTING COM
    com_split = calc_tools.split_list( alist = rdf.solute_COM[0], wanted_parts = 2 )
    

    