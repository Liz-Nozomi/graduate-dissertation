lin# -*- coding: utf-8 -*-
"""
nanoparticle_sasa.py
The purpose of this script is to analyze the sasa of a nanoparticle. It would be highly useful to calculate SASA for the terminal groups (and even the inner groups)

INPUTS:
    - gro file
    - itp file (for bonding purposes)
OUTPUTS
    - sasa ratio (avg and std)
FUNCTIONS:
    find_ligand_r_groups: finces specific R group from a large dictionary
CLASSES:
    nanoparticle_sasa: calculates sasa of the ligands on the nanoparticle
    
CREATED ON: 05/10/2018

AUTHOR(S):
    Alex K. Chew (alexkchew@gmail.com)


** UPDATES **
    20180513: Completed draft of nanoparticle sasa
    20180710: Updated draft to include ligand atoms --- also turning off atom pairs to prevent slowdown / memory issues
TODO:
    - Add other groups other than end groups
"""
### IMPORTING MODULES
import MDDescriptors.core.import_tools as import_tools # Loading trajectory details
import numpy as np
import MDDescriptors.core.calc_tools as calc_tools # calc tools
import MDDescriptors.core.read_write_tools as read_write_tools # Reading itp file tool
import mdtraj as md
import sys
import time # Measures time

## MDDescriptor modules
### IMPORTIING CORE ITEMS
from MDDescriptors.core.track_time import print_total_time
from MDDescriptors.core.plot_tools import color_y_axis
### IMPORTING NANOPARTICLE STRUCTURE CLASS
from MDDescriptors.application.nanoparticle.nanoparticle_structure import nanoparticle_structure

### DEFINING GLOBAL VARIABLES
from MDDescriptors.global_vars.plotting_global_vars import LABELS, LINE_STYLE

### IMPORTING TOOLS FOR CALCULATING SASA
from MDDescriptors.geometry.accessible_hydroxyl_fraction import custom_shrake_rupley  ## CUSTOM SHRAKE AND RUPLEY BASED ON BONDI VDW RADII


### FUNCTION TO FIND LIGAND VIA SPECIFIC R GROUP
def find_ligand_r_groups(r_groups_per_ligand, ligand_residue_name):
    '''
    The purpose of this function is to find the ligand R groups for all possible groups
    INPUTS:
        r_groups_per_ligand: [dict]
            r groups per ligand, e.g.
                {'r_group: {'RO1: [ ] },..., 'r_group_2': {'RO1: [ ] },.. }
        ligand_residue_name: [str]
                residue name of the ligand, should be within your r_groups_per_ligand
    OUTPUTS:
        ligand_specific_r_groups: [dict]
            r groups for that specific ligand that you want
    '''
    ## CREATING EMPTY DICTIONARY
    ligand_specific_r_groups = {}
    ## LOOPING THROUGH EACH KEY
    for each_key in r_groups_per_ligand:
        ## TRYING TO SEE IF THERE IS A KEY
        try:
            ## IF AVAILABLE, THEN CREATING A KEY WITHIN SPECIFIC GROUP
            ligand_specific_r_groups[each_key] = r_groups_per_ligand[each_key][ligand_residue_name]
        except KeyError: # Passing otherwise
            pass 
    return ligand_specific_r_groups

########################################
### CLASS TO STUDY NANOPARTICLE SASA ###
########################################
class nanoparticle_sasa:
    '''
    The purpose of this function is to calculate the solvent accessible surface area of the ligands.
    INPUTS:
        traj_data: Data taken from import_traj class
        ligand_names: [list] list of ligand residue names. Note that you can list as much as you'd like, we will check with the trajectory to ensure that it is actually there.
        itp_file: itp file name (note! We will look for the itp file that should be in the same directory as your trajectory)
            if 'match', it will look for all itp files within the trajectory directory and locate all itp files that match your ligand names.
                NOTE: This is currently assuming homogenous ligands -- this will need to be adjusted for multiple ligands
        sasa_cutoff [float] cutoff in nm for finding nearby solvent residues to calculate sasa for
            'optimize': [default] optimizes sasa cutoff by trying different sasa cutoffs
            None: Simply use all molecule sasa
            float: will use this specific cutoff
        separated_ligands: [logical] True if you want to avoid itp file. This assumes that your gro defines ligands separately. (e.g. more than one residue for ligands)
            NOTE: This is the case for the planar SAMs
        job_type: [str] type of sasa you desire
            'end_groups': calculates sasa of the end groups (default)
            'all_atom_ligands': calculates sasa of all ligand atoms
            'r_group': calculates for r groups -- NOTE: you will need to include rgroups within variable "r_group"
        r_group_dict: [dict]
            dictionary filled with r groups for different ligands, e.g.
                {'r_group: {'RO1: [ ] },..., 'r_group_2': {'RO1: [ ] },.. }
            This enables specific sasa calculations
        ## SASA INPUTS
            probe_radius: [float, optional] radius of the probe in nm
            n_sphere_points: [int] number of points representing the surface of each atom. Higher values tend to lead to more accuracy.
        save_disk_space: [logical] True if you want to save disk space by removing the following variables: [Default: True]
            Turning off: terminal_group_distances, terminal_group_min_distance_matrix, dihedrals
        debug_mode: [logical] True if you want debug mode to be on
    OUTPUTS:
        ## INPUT VARIABLES
            self.ligand_names, self.itp_file_name, self.separated_ligands
            
        ## GROUP TYPES
            self.atom_index_each_group_type: [dict] dictionary containing atom index of each group type that you want.
                'end_groups': contains all end group heavy atoms along with its hydrogens 
            self.atom_pairs: [dict] contains all atom pairs information
            self.distances: [dict] contains dictionary of all the distances between atom pairs
        ## SASA
            self.sasa: [dict] dictionary with a list of average sasa values per ligand of each frame
            self.sasa_total_time: [dict] dictionary with the total sasa time it took for each group
            self.sasa_avg_std: [dict] dictionary of each group type with 'avg' and 'std' as keys
        ## DEBUGGING/OPTIMIZATION OUTPUTS
            self.optimal_cutoff_dict: [dict] returns cutoff dictionary for each group type
        ## FOR ALL ATOM LIGANDS
            self.all_atom_ligands_shape: [tuple] Shape of all atom ligands
            self.sasa_all_atom_ligands_each_ligand: [np.array, shape=(num_frames, num_ligands)] sasa of all the atoms for each ligand
    ALGORITHM:
        - define the ligands
        - find the groups of interest
        - find all residues surrounding groups of interest within a cutoff region (speed up calculations!)
        - re-create the trajectory with only the residues of interest
        - calculate a sasa value for each group, then sum them and divide by the number of ligands (sasa / ligand)
    
    FUNCTIONS:
        find_atom_interest_groups: finds groups of interest
        find_atom_pairs_and_distances: finds distances between pairs
        calc_sasa: calculates sasa based on some cutoff
        ## DEBUGGING / OPTIMIZATION SCRIPTS
        debug_calc_sasa_cutoff: calculates sasa for various values of cutoffs
        debug_find_optimal_cutoff: finds optimal cutoff based on some tolerance
        
    ACTIVE FUNCTIONS:
        debug_plot_sasa_cutoff: plots debugging of sasa cutoff
        
    '''
    ### INITIALIZING
    def __init__(self, 
                 traj_data, 
                 ligand_names, 
                 itp_file, 
                 sasa_cutoff = 'optimize', 
                 separated_ligands = False, 
                 group_type = [ 'end_groups' ], 
                 r_group_dict = {}, 
                 probe_radius = 0.14, 
                 n_sphere_points = 960, 
                 save_disk_space = True, 
                 debug_mode = False):
        ### PRINTING
        print("**** CLASS: %s ****"%(self.__class__.__name__))

        
        ## STORING INITIAL VARIABLES
        self.ligand_names           = ligand_names                        # Ligand names
        self.separated_ligands      = separated_ligands              # Used to avoid itp file generation
        self.group_type             = group_type
        self.sasa_cutoff            = sasa_cutoff
        self.probe_radius           = probe_radius
        self.n_sphere_points        = n_sphere_points
        self.debug_mode             = debug_mode
        self.save_disk_space        = True

        ## DEFINING TRAJECTORY
        traj = traj_data.traj # [0:10]
        
        ### CALCULATING NANOPARTICLE STRUCTURE
        self.structure = nanoparticle_structure(traj_data           = traj_data,                # trajectory data
                                                ligand_names        = self.ligand_names,        # ligand names
                                                itp_file            = itp_file,                 # defines the itp file
                                                structure_types      = None,                     # checks structural types
                                                separated_ligands    = self.separated_ligands    # True if you want separated ligands 
                                                )
        ## DEFINING AVAILABLE JOB TYPES
        available_group_type = [ 'end_groups', 'all_atom_ligands', 'r_group' ] 
        
        ## CHECKING IF YOUR GROUP TYPE IS WITHIN THE AVAILABLE GROUPS
        self.group_type = [ each_structure for each_structure in self.group_type if each_structure in available_group_type]
        
        ## STOPPING IF GROUP TYPE IS NOT AVAILABLE
        if len(self.group_type) == 0:
            print("ERROR! No group type is presented!")
            print("Available group types are: %s"%(', '.join(available_group_type)))
            print("Your input group types are: %s"%(', '.join(group_type)))
            print("Stopping here! Please check your inputs!")
            sys.exit()
            
        ## ADDING TO GROUP TYPE IF YOU HAVE R GROUP
        if 'r_group' in self.group_type:
            ### DEFINING ALL R GROUPS
            # NOTE! THIS ONLY TAKES IN ONE LIGAND -- MAY NEED TO FIX LATER FOR MIXED-LIGAND SITUATIONS
            self.r_group = find_ligand_r_groups(r_groups_per_ligand = r_group_dict, ligand_residue_name = self.structure.ligand_names[0])
            ## CHECKING IF LENGTH OF R GROUP IS GREATER THAN ZERO
            if len(self.r_group) > 0:
                print("R groups selected! Adding to the group_type")
                ## REMOVING R_GROUP (GENERALIZED) FOR A NEW ONE
                self.group_type.remove('r_group')
                ## FINDING R GROUP TYPE NAMES
                self.r_group_type_names = [ 'r_group_' + each_key for each_key in self.r_group ]
                self.group_type.extend(self.r_group_type_names)
            else:
                print("Error! R group selected, but no r_group key word presented")
                print("Please check your inputs! There should be r_groups in your input")
                print("The r_group variable should be a dictionary with a label and corresponding atom labels")
                sys.exit()
            
        
        ## RUNNING FUNCTION TO FIND THE GROUPS OF INTEREST
        self.find_atom_interest_groups(traj)
        
        if self.debug_mode is True:
            ## DEBUGGING SASA
            self.debug_calc_sasa_cutoff(traj=traj)
            ## FINDING OPTIMAL CUTOFF
            self.debug_find_optimal_cutoff()
            ## PLOTTING THE SASA CUTOFF
            self.debug_plot_sasa_cutoff()
        else:
            if self.sasa_cutoff == 'optimize':
                ## DEBUGGING SASA
                self.debug_calc_sasa_cutoff(traj=traj)
                ## FINDING OPTIMAL CUTOFF
                self.debug_find_optimal_cutoff()
            
        ## FINDING DISTANCES AND ATOM PAIRS
        self.find_atom_pairs_and_distances(traj, sasa_cutoff = self.sasa_cutoff)            
        ## RUNNING FUNCTION TO CALCULATE SASA FOR EACH GROUP OF INTEREST
        self.calc_sasa(traj, sasa_cutoff = self.sasa_cutoff)
        
        ## DESTROYING VARIABLES TO SAVE SPACE
        if self.save_disk_space == True:
            print("Saving space!")
            self.distances = {}; self.atom_pairs = {};
        return
        
    ### FUNCTION TO FIND THE GROUPS OF INTEREST
    def find_atom_interest_groups(self, traj):
        '''
        The purpose of this function is to find the groups of interest
        INPUTS:
            traj: [md.traj] trajectory from md.traj
        OUTPUTS:
            self.atom_index_each_group_type: [dict] dictionary containing atom index of each group type that you want.
                'end_groups': contains all end group heavy atoms along with its hydrogens 
                'all_atom_ligands': contains all atoms on the ligand
            self.all_atom_ligands_shape: [tuple] Shape of all atom ligands
        '''
        ## DEFINING EMPTY DICTIONARY
        self.atom_index_each_group_type = {}
        ## LOOPING THROUGH GROUP TYPES, FINDING ATOM INDEXES OF GROUPS OF INTEREST
        for each_group_type in self.group_type:
            ## END_GROUPS
            if each_group_type == 'end_groups':
                ## DEFINING ENDGROUP LIST
                end_group_atom_list = []
                ## FINDING ALL END GROUP HEAVY ATOMS FROM THE STRUCTURE (LAST INDEX OF THE HEAVY ATOMS)
                heavy_atoms_index = [ each_ligand[-1] for each_ligand in self.structure.ligand_heavy_atom_index]
                ## APPENDING HEAVY ATOMS
                end_group_atom_list.extend(heavy_atoms_index)
                ## FINDING ALL HYDROGENS BONDED TO THE HEAVY ATOM, BUT NOT A HEAVY ATOM
                for each_heavy_atom in heavy_atoms_index:
                    ## LIGANDS ARE BONDED
                    if self.separated_ligands is False:
                        ## DEFINING SERIAL NUMBER
                        current_serial_index = each_heavy_atom + 1
                        ## FINDING ALL ATOMS THAT ARE ATTACHED
                        bonds_attached = self.structure.itp_file.bonds[np.where(self.structure.itp_file.bonds==current_serial_index)[0],:]
                        ## FINDING ALL ATOM NUMBERS WITHIN ATTACHED ATOMS
                        atoms_attached = bonds_attached[bonds_attached != current_serial_index]
                        ## APPENDING TO LIST, AVOIDING ALL GOLD ATOMS
                        atoms_to_append=[ each_atom - 1 for each_atom in atoms_attached if traj.topology.atom(each_atom-1).element.symbol == 'H' ] # Just hydrogens attached
                        ## APPENDING HYDROGENS
                    else:
                        ## FINDING ALL ATOM INDEXES
                        atoms_index_within_residue=[ each_atom.index for each_atom in traj.topology.atom(each_heavy_atom).residue.atoms ] # Outputs list , e.g. [14, 15, ...] <-- atom numbers for a residue
                        ## FINDING HEAVY ATOM WITHIN THE NEW NOMENCLATURE
                        current_serial_index = atoms_index_within_residue.index(each_heavy_atom) + 1
                        ## FINDING ALL ATOMS THAT ARE ATTACHED
                        bonds_attached = self.structure.itp_file.bonds[np.where(self.structure.itp_file.bonds==current_serial_index)[0],:]
                        ## FINDING ALL ATOM NUMBERS WITHIN ATTACHED ATOMS
                        atoms_attached = [ atoms_index_within_residue[each_atom-1] for each_atom in bonds_attached[bonds_attached != current_serial_index] ]
                        ## APPENDING TO LIST, AVOIDING ALL GOLD ATOMS
                        atoms_to_append=[ each_atom for each_atom in atoms_attached if traj.topology.atom(each_atom).element.symbol == 'H' ] # Just hydrogens attached
                        
                    ## APPENDING ATOMS    
                    end_group_atom_list.extend(atoms_to_append)
                ## AT THE END, SORT AND RE-APPEND TO FULL ATOM INDEX GROUP
                end_group_atom_list.sort()
                ## APPENDING GROUPS
                self.atom_index_each_group_type[each_group_type] = end_group_atom_list
            ## HAEVY ATOMS
            elif each_group_type == 'all_atom_ligands':
                ## DEFINING HEAVY ATOM GROUPS
                all_atom_ligands_atoms = np.array(self.structure.ligand_atom_index_list)
                ## APPENDING GROUPS
                self.atom_index_each_group_type[each_group_type] = np.array(all_atom_ligands_atoms).flatten() ## FLATTENING LIST
                
                ## CREATING CUSTOM VARIABLE THAT IS USED LATER
                self.all_atom_ligands_shape = all_atom_ligands_atoms.shape ## STORING SHAPE
            elif 'r_group' in each_group_type:
                ## CONVERTING GROUP TYPE BACK TO ORIGINAL NAME
                orig_label = each_group_type.replace('r_group_','', 1)
                ## LOOPING THROUGH EACH R GROUP
                atom_names = self.r_group[orig_label]
                ## FINDING ATOM INDICES AS A NUMPY ARRAY
                atom_indices = np.array([idx for lig_idx_list in self.structure.ligand_atom_index_list 
                                    for idx in lig_idx_list 
                                    if traj.topology.atom(idx).name in atom_names])
                ## APPENDING ENTIRE GROUP LIST
                self.atom_index_each_group_type[each_group_type] = atom_indices

        return
                
    ### FUNCTION TO COMPUTE DISTANCES BETWEEN PAIRS
    def find_atom_pairs_and_distances( self, traj, sasa_cutoff = True, periodic=True):
        '''
        The purpose of this function is to get all the atom pairs and distances (for cutoff purposes)
        INPUTS:
            traj: [md.traj] trajectory from md.traj
            sasa_cutoff: [float] cutoff for sasa calculation
                NOTE 1: None type will turn off distance calculations
                NOTE 2: True will turn on distance calculations
            periodic: [logical] True if you want PBCs
        OUTPUTS:
            self.atom_pairs: [dict] contains all atom pairs information
            self.distances: [dict] contains dictionary of all the distances between atom pairs
        '''
        ## CREATING DICTIONARY
        self.distances, self.atom_pairs =  {}, {}
        ## LOOPING THROUGH EACH GROUP TYPE
        for each_group_type in self.group_type:
            if sasa_cutoff is not None or sasa_cutoff is True:
                print("--- CALCULATING DISTANCES/ATOM PAIRS FOR GROUP TYPE: %s ---"%(each_group_type))
                ## KEEPING TRACK OF TIME
                start_time = time.time()
                ## DEFINING ATOM INDEX
                atom_index = self.atom_index_each_group_type[each_group_type]            
                ## PRINTING TOTAL ATOMS
                print("Total atoms for this group type: %d"%(len(atom_index)))
                ## FINDING ALL HEAVY ATOM INDEXES
                heavy_atom_index = [ each_atom_index for each_atom_index in atom_index if traj.topology.atom(each_atom_index).element.symbol != 'H' ]
                ## FINDING ALL OTHER INDEX
                other_atom_index = np.compress(~np.isin(atom_index,heavy_atom_index), atom_index)
                # FINDING ATOM PAIRS
                atom_pairs = np.array([ [heavy_atom, atoms.index] for heavy_atom in heavy_atom_index for atoms in traj.topology.atoms if heavy_atom != atoms.index and atoms.index not in other_atom_index ])
                print("Total pairs: %d"%(len(atom_pairs) ))
                ## COMPUTING DISTANCES
                print("--- COMPUTING DISTANCES FOR ALL THE PAIRS ---")
                distances = md.compute_distances( traj =  traj, atom_pairs = atom_pairs, periodic = periodic)
                print_total_time(start_time, 'Distance calculation time: ')
            else:
                distances = []
                atom_pairs = []
            ## STORING
            self.distances[each_group_type] = distances
            self.atom_pairs[each_group_type] = atom_pairs
        
    ### FUNCTION TO CALCULATE SASA FOR EACH FRAME
    def calc_sasa(self, traj, sasa_cutoff, traj_time_print_frequency= 10000, periodic=True):
        '''
        The purpose of this script is to calculate the sasa for each of the interest groups. Here, we will try to speed up the script by removing atoms within some cutoff.
        You can turn off the speed up by turning "sasa_cutoff" to None.
        INPUTS:
            traj: [md.traj] trajectory from md.traj
            sasa_cutoff: [float] cutoff for sasa calculation
            traj_time_print_frequency: [int] frequency to print output based on trajectory time (ps)
            periodic: [logical] True if you want PBCs
        OUTPUTS:
            self.sasa: [dict] dictionary with a list of average sasa values per ligand
            self.sasa_total_time: [dict] dictionary with the total sasa time it took for each group
            self.sasa_avg_std: [dict] dictionary of each group type with 'avg' and 'std' as keys
            self.sasa_all_atom_ligands_each_ligand: [np.array, shape=(num_frames, num_ligands)] sasa of all the atoms for each ligand
        ALGORITHM:
            - Find all heavy atoms of your groups
            - LOOP THROUGH TRAJECTORY
                - Find all nearby solvents / molecules
                - Truncate the trajectory
                - Calculate SASA for a single trajectory
                - Add up all the SASAs for your selected group
                - Loop until finished
            - After all the trajectories are complete, take an average and standard deviation of the sasa
        NOTES:
            - Distances are calculated for entire trajectories -- can run out of memory! -- should include memory tests
            - Could just include distances as a per frame basis
        '''
        ## FINDING TRAJECTORY TOTAL TIME
        traj_time = traj.time
        self.sasa = {}; self.sasa_total_time = {}; self.sasa_avg_std = {}
        ## CUSTOM STORAGE VARIABLES
        if 'all_atom_ligands' in self.group_type:
            self.sasa_all_atom_ligands_each_ligand = np.zeros( (len(traj),  self.structure.total_ligands) )
        
        ## LOOPING THROUGH EACH GROUP TYPE
        for each_group_type in self.group_type:
            ## CREATING EMPTY LIST FOR SASA
            sasa_list= []
            ## KEEPING TRACK OF TIME
            start_time = time.time()
            ## DEFINING ATOM INDEX
            atom_index = self.atom_index_each_group_type[each_group_type]            
            ## DEFINING DISTANCES AND ATOM PAIRS
            distances = self.distances[each_group_type]
            atom_pairs = self.atom_pairs[each_group_type]
            ## LOOPING THROUGH THE TRAJECTORY
            for index, each_traj in enumerate(traj):                
                ## STORING TIME FOR REACH TRAJ CALCULATION
                if traj_time[index] % traj_time_print_frequency == 0:
                    time_each_traj = time.time()
                ## FINDING ALL ATOM INDEX THAT FALL INTO THE CURRENT DISTANCES
                if sasa_cutoff is not None:
                    ## DEFINING CURRENT DISTANCES
                    current_distance = distances[index]
                    if sasa_cutoff == 'optimize':
                        ## DEFINING CUTOFF BASED ON THE OPTIMIZED VALUES
                        sasa_cutoff = self.optimal_cutoff_dict[each_group_type]
                    
                    atom_index_within_range = atom_pairs[np.where(current_distance < sasa_cutoff), 1][0]
                    ## FINDING ALL ATOMS NEAR THE HEAVY ATOM BASED ON A CUTOFF
                    # APPENDING ATOM INDICES AND SORTING -- FINDING UNIQUE INDEXES
                    all_atom_index = np.unique( np.sort( np.append(atom_index_within_range, atom_index, axis = 0) ) )
                    ## SLICING TRAJECTORY
                    slice_traj = each_traj.atom_slice(atom_indices = all_atom_index, inplace=False )
                    '''
                        SLICED TRAJECTORY WILL HAVE THE ATOM NUMBERS REPLACED.  For example, suppose your atom_index has [ 7, 10, ...]
                        Now, sliced trajectory will have: [0, 1, ...]
                        We can easily create a mapping between full to slice trajectory
                    '''
                    ## CREATING CODE BETWEEN FULL TO SLICE
                    atom_code_full_to_slice = np.array([ [orig_traj_index,slice_traj_index]  for slice_traj_index, orig_traj_index in enumerate(all_atom_index)])
                    
                    ## SEARCHING FOR INDEX FOR YOUR ATOM INDEX
                    atom_index_slice = atom_code_full_to_slice[np.where(np.isin(atom_code_full_to_slice[:,0], atom_index)), 1] ## ATOMS OF INTEREST IN THE NEW SASA
                else:
                    slice_traj = each_traj[:]
                    atom_index_slice = atom_index[:]

                ## CALCULATING SASA
                sasa = custom_shrake_rupley(traj=slice_traj[:],
                     probe_radius=self.probe_radius, # in nms
                     n_sphere_points=self.n_sphere_points, # Larger, the more accurate
                     mode='atom' # Extracted areas are per atom basis
                     ) ## OUTPUT IN NUMBER OF FRAMES X NUMBER OF FEATURES, e.g. shape = 1, 465
                
                ## FINDING SASA FOR THE ATOM INDEX
                group_sasa = sasa[:, atom_index_slice]
                ## CUSTOM SASA FUNCTIONS
                if each_group_type == 'all_atom_ligands':
                    ## RESHAPING HEAVY GROUP BACKBONE
                    reshaped_group_sasa = group_sasa.reshape( 1, self.all_atom_ligands_shape[0], self.all_atom_ligands_shape[1]) # One for each frame
                    ## SUMMING ALL THE SASA
                    reshaped_group_sasa = np.sum(reshaped_group_sasa, axis=2)
                    ## STORING
                    self.sasa_all_atom_ligands_each_ligand[index] = reshaped_group_sasa[0][:] # ESCAPING TIME FRAME
                    
                ## FINDING AVERAGE SASA
                # avg_sasa = np.mean(group_sasa)
                ## FINDING TOTAL SASA
                sum_sasa = np.sum(group_sasa)
                ## APPENDING SASA
                sasa_list.append(sum_sasa)
                ## FINDING TRAJECTORY TIME
                if traj_time[index] % traj_time_print_frequency == 0:
                    print("\n*** WORKING ON TRAJECTORY TIME: %.0f ps out of %.0f ***"%(traj_time[index], traj_time[-1]))
                    if sasa_cutoff is not None:
                        print("CURRENT SASA CUTOFF: %.2f"%(sasa_cutoff) )
                        print("FOUND THIS NUMBER OF ATOMS NEAR HEAVY ATOMS: %d"%(len(atom_index_within_range)))
                        print("------- SLICING TRAJECTORY --------")
                        print("SLICING TRAJECTORY WITH TOTAL ATOMS: %d"%( len(all_atom_index) ) )
                    else:
                        print("NO SLICING OF TRAJECTORY WAS MADE SINCE sasa_cutoff SET TO None.")
                        print("TOTAL ATOMS CONSIDERED IN SASA: %d"%(len(atom_index_slice)) )
                    ## PRINTING AVERAGE SASA
                    print("TOTAL SASA: %.4f nm^2"%(sum_sasa) )
                    ## PRINTING TIME DETAILS
                    print_total_time(time_each_traj)
                    
            ## PRINTING TIME
            total_time = print_total_time(start_time, '\nTOTAL %s GROUP SASA TIME: '%(each_group_type) )
            ## AT THE END, APPEND THE ENTIRE SASA LIST
            self.sasa[each_group_type] = sasa_list    
            self.sasa_total_time[each_group_type] = total_time
            ## FINDING AVERAGE AND STANDARD DEVIATION OF EACH GROUP
            self.sasa_avg_std[each_group_type] = {'avg': np.mean(self.sasa[each_group_type]),
                                                  'std': np.std(self.sasa[each_group_type])}
            
    ### FUNCTION TO DEBUG THE SASA CUTOFF
    def debug_calc_sasa_cutoff(self, traj, traj_time_cutoff = 1000, cutoff_ranges=np.arange(0.3, 1.0, 0.1), tolerance = 0.0001):
        '''
        The purpose of this function is to debug the sasa cutoff to find optimal time constraints prior to running the full sasa.
        NOTE: We will see if the sasa value is reasonably good and stop the debugging script
        INPUTS:
            traj: [md.traj] trajectory from md.traj
            traj_time_cutoff: [float, default = 5,000] time of cutoff for the debugging in picoseconds. 
            cutoff_ranges: [np.array] cutoff points you want to check for SASA value
            show_plot: [logical] True if you want to show plots
            tolerance: [float] value to check between full trajectories and cutoff. If below this value, assume the cutoff is sufficient!
        OUTPUTS:
            plot of sasa value vs. cutoff
        '''
        ## PRINTING
        print("**** OPTIMIZING SASA CUTOFF ****")
        
        ## FINDING TIME
        traj_time = traj.time - traj.time[0] ## ZEROING THE TIME
        
        ## FINDING INDEX WHERE THE FIRST TIME INDEX IS TRUE
        index_of_traj_within_cutoff = np.argwhere(traj_time >= traj_time_cutoff)
        
        if len(index_of_traj_within_cutoff) == 0:
            print("ERROR! Traj time cutoff greater than length of traj")
            print("Supplied time (ps): %d"%(traj_time_cutoff) )
            print("Largest traj time (ps): %d"%( traj_time[-1] ))
            print("Stopping here before we have more errors! Check your inputs on traj_time_cutoff")
            sys.exit()
        else:
            index_traj_shorten = int(index_of_traj_within_cutoff[0])
            print("SHORTENING TRAJECTORY TO: %d"%(traj_time[index_traj_shorten]) )
            print('TRAJ INDEX: %d'%(index_traj_shorten))
        
        ## MAKING TRAJECTORY SMALLER IN TERMS OF INDEX
        traj_shorten = traj[:index_traj_shorten]
        
        ## CREATING EMPTY DICTIONARIES
        self.debug_sasa_values = {}
        self.debug_time_values = {}
        self.debug_cutoff_ranges = {}        
        
        ## CALCULATING DISTANCES
        self.find_atom_pairs_and_distances(traj = traj_shorten, sasa_cutoff = True)
        
        ## RUNNING NONETYPE CALCULATIONS
        self.calc_sasa(traj = traj_shorten, sasa_cutoff = None)
        
        ## APPENDING TO DEBUG SASA VALUES
        self.debug_sasa_values['None_sasa'] = [ np.mean(self.sasa[each_group_type]) for each_group_type in self.group_type ]
        self.debug_sasa_values['None_time'] = [ self.sasa_total_time[each_group_type] for each_group_type in self.group_type ]
        
        ## LOOPING THROUGH EACH GROUP AND STORING
        for index, each_group_type in enumerate(self.group_type):
            ## DEFINING CUTOFF STORAGE
            cutoff_range_storage = cutoff_ranges[:]
            ## LOOPING THROUGH CUTOFF RANGES
            for cutoff_index, each_cutoff in enumerate(cutoff_range_storage):
                ## CALCULATING SASA
                self.calc_sasa(traj = traj_shorten, sasa_cutoff = each_cutoff)

                ## SEEING IF KEY IS THERE
                if each_group_type in self.debug_sasa_values:
                    self.debug_sasa_values[each_group_type].append( self.sasa_avg_std[each_group_type]['avg'])
                    self.debug_time_values[each_group_type].append(self.sasa_total_time[each_group_type])
                else:
                    self.debug_sasa_values[each_group_type] = [ self.sasa_avg_std[each_group_type]['avg'] ]
                    self.debug_time_values[each_group_type] = [ self.sasa_total_time[each_group_type] ]
                    
                ## SEEING IF THE DEBUG VALUE MATCHES SOME THRESHOLD CUTOFF
                sasa_error_from_no_cutoff = self.debug_sasa_values[each_group_type][-1] - self.debug_sasa_values['None_sasa'][index]
                if self.debug_sasa_values[each_group_type][-1] - self.debug_sasa_values['None_sasa'][index] <= tolerance:
                    print("Since sasa value of the current cutoff, %.2f, is sufficiently satisfying tolerance: %.5f"%(each_cutoff, tolerance)  )
                    print("We assume for group type, %s, that this cutoff is sufficient, so stopping here!"%( each_group_type ) )
                    print("Error between sasas: %.5f"%(sasa_error_from_no_cutoff))
                    ## CORRECTING CUTOFF RANGES
                    cutoff_range_storage = cutoff_range_storage[:cutoff_index+1] # Adding one since python counts from zero
                    break
            ## STORING CUTOFF RANGES
            self.debug_cutoff_ranges[each_group_type] = cutoff_range_storage[:]
        return
    ### FUNCTION TO FIND OPTIMAL CUTOFF
    def debug_find_optimal_cutoff(self, tolerance = 0.0001 ):
        '''
        The purpose of this sccript is to take the debugging values and find the optimal cutoff
        INPUTS:
            tolerance: [float] tolerance between the differences of sasa and the "None" cutoff sasa
        OUTPUTS:
            self.optimal_cutoff_dict: [dict] returns cutoff dictionary for each group type
        '''
        ## CREATING EMPTY DICTIONARY TO STORE OPTIMAL CUTOFFS
        self.optimal_cutoff_dict = {}
        for index, each_group_type in enumerate(self.group_type):
            ## LOCATING THE SASA THAT IS OPTIMAL FOR THE GROUP TYPE
            differences = self.debug_sasa_values[each_group_type] - self.debug_sasa_values['None_sasa'][index]
            ## FINDING INDEXES THAT MEET 
            indexes_within_tol = np.argwhere( differences < tolerance ) ## RETURNS [[2, 5, 6, etc.]]
            if len(indexes_within_tol) == 0:
                print('No indices of cutoff radius was found using cutoff ranges: %s'%(self.debug_cutoff_ranges[each_group_type]))
                print('Therefore, we will just use cutoff Nonetype (no cutoff for sasa!)')
                print('If this is not what you want, consider changing the cutoff ranges!')
                self.optimal_cutoff_dict[each_group_type] = None
            else:
                ## DEFINING OPTIMAL AS ONE MORE THAN THE INDEX
                optimal_index = indexes_within_tol[0] + 1
                ## GETTING CUTOFF IN NM
                try: ## SEEING IF WE CAN GET A VALUE HIGHER
                    optimal_cutoff = self.debug_cutoff_ranges[each_group_type][ optimal_index ]
                except:
                    optimal_cutoff = self.debug_cutoff_ranges[each_group_type][ indexes_within_tol[0] ]
                ## STORING CUTOFF
                self.optimal_cutoff_dict[each_group_type] = optimal_cutoff
        return
        
    ### FUNCTION TO PLOT DEBUG SASA
    def debug_plot_sasa_cutoff(self, tolerance = 0.0001):
        '''
        The purpose of this function is to plot the sasa cutoff based on "debug_sasa_cutoff"
        INPUTS:
            tolerance: [float] tolerance between the differences of sasa and the "None" cutoff sasa
        OUTPUTS:
            plot for each group type:
                sasa value and total time vs. cutoff 
        '''
        import matplotlib.pyplot as plt
        ## LOOPING THROUGH EACH GROUP TYPE
        for index, each_group_type in enumerate(self.group_type):
            ## CREATING PLOT
            fig, ax = plt.subplots()
            ## SETTING TITLE
            ax.set_title('SASA optimization for group type: %s'%(each_group_type))
            ## DEFINING X AND Y AXIS
            ax.set_xlabel('SASA cutoff (nm)', **LABELS)
            ax.set_ylabel('Avg SASA value (nm^2)', **LABELS)
            ## PLOTTING SASA VS. CUTOFF
            ax.plot( self.debug_cutoff_ranges[each_group_type], self.debug_sasa_values[each_group_type], color='k', **LINE_STYLE )
            ## PLOTTING NONE TYPE
            ax.axhline( self.debug_sasa_values['None_sasa'][index], linestyle = '--', color='k',
                       label='No cutoff SASA', **LINE_STYLE )
            
                        
            ## PLOTTING TIME VS. CUTOFF
            ax2 = ax.twinx()
            ax2.set_ylabel('Time (s)',color='b', **LABELS)
            ax2.plot( self.debug_cutoff_ranges[each_group_type], self.debug_time_values[each_group_type], color='b', **LINE_STYLE )
            ## PLOTTING NONE TYPE
            ax2.axhline( self.debug_sasa_values['None_time'][index], linestyle = '--', color='b',
                       label='No cutoff time', **LINE_STYLE )
            color_y_axis(ax2, 'b')
            
            ## PLOTTING OPTIMAL TOLERANCE
            optimal_cutoff = self.optimal_cutoff_dict[ each_group_type ]
            ax.axvline( x = optimal_cutoff, linestyle = '--', color='r',
                       label='Optimal SASA cutoff', **LINE_STYLE)
    
            ## PLOTTING LEGEND
            ax.legend()
            
        return


#%% MAIN SCRIPT
if __name__ == "__main__":
    
    ### DIRECTORY TO WORK ON
    '''
    analysis_dir=r"180508-transfer_ligands_2nm_4nm" # Analysis directory
    category_dir="spherical" # category directory
    specific_dir="spherical_310.15_K_2_nmDIAM_butanethiol_CHARMM36_Trial_1" # spherical_310.15_K_2_nmDIAM_octanethiol_CHARMM36" # Directory within analysis_dir r"mdRun_363.15_6_nm_tBuOH_100_WtPercWater_spce_Pure"
    
    analysis_dir=r"180702-Trial_1_spherical_EAM_correction" # Analysis directory
    category_dir="EAM" # category directory
    specific_dir="EAM_310.15_K_2_nmDIAM_butanethiol_CHARMM36_Trial_1" # spherical_310.15_K_2_nmDIAM_octanethiol_CHARMM36" # Directory within analysis_dir r"mdRun_363.15_6_nm_tBuOH_100_WtPercWater_spce_Pure"    
    '''
    
    analysis_dir=r"180928-2nm_RVL_proposal_runs_fixed" # Analysis directory
    category_dir="EAM" # category directory
    specific_dir="EAM_310.15_K_2_nmDIAM_ROT001_CHARMM36_Trial_1" 
    
    
    path2AnalysisDir=r"R:\scratch\nanoparticle_project\analysis\\" + analysis_dir + '\\' + category_dir + '\\' + specific_dir + '\\' # PC Side
    # path2AnalysisDir=r"/Volumes/akchew/scratch/nanoparticle_project/analysis/" + analysis_dir + '/' + category_dir + '/' + specific_dir + '/' # MAC side

    ### DEFINING FILE NAMES
    '''
    gro_file=r"sam_prod.gro" # Structural file
    xtc_file=r"sam_prod_10_ns_whole.xtc" # r"sam_prod_10_ns_whole.xtc" # Trajectory file
    '''    
    gro_file=r"sam_prod_10_ns_whole_no_water_center.gro" # Structural file
    xtc_file=r"sam_prod_10_ns_whole_no_water_center.xtc" # r"sam_prod_10_ns_whole.xtc" # Trajectory file
    
    ### LOADING TRAJECTORY
    traj_data = import_tools.import_traj( directory = path2AnalysisDir, # Directory to analysis
                 structure_file = gro_file, # structure file
                  xtc_file = xtc_file, # trajectories
                  )
    
    #%%
    ## IMPORTING R GROUP
    from MDDescriptors.application.nanoparticle.global_vars import R_GROUPS_PER_LIGAND
    ### DEFINING INPUT DATA
    input_details = {
                        'ligand_names':             ['OCT', 'BUT', 'HED', 'DEC', 'RO1', 'RO2',],       # Name of the ligands of interest
                        'itp_file':                 'sam.itp',                          # ITP FILE
 #                       'itp_file':                 'match',                          # ITP FILE
                        'group_type':               ['r_group'],                     # job type you want
                        'r_group_dict':             R_GROUPS_PER_LIGAND,
                        'separated_ligands':        False   ,                           # True if your gro file already separates the ligands -- removes the need for itp file searching
                        'probe_radius':             0.14    ,                           # in nm, VDW of water
                        'n_sphere_points':          960     ,                           # number of sphere points
                        'save_disk_space':          True    ,                           # Saving space
                        'sasa_cutoff':              None     ,                           # nm sasa cutoff
                        'debug_mode':               False    ,                           # True if you want to debug the script
                        }
    ## RUNNING CLASS
    sasa = nanoparticle_sasa(traj_data, **input_details )
    
