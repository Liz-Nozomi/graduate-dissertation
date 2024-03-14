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
    
CLASSES:
    calc_rdf: calculates rdf of your system

AUTHOR(S):
    Alex K. Chew (alexkchew@gmail.com)
    
** UPDATES **
20180321 - AKC - Added functionality to calculate RDF at each individual oxygens
20180506 - AKC - Added calculation of ensemble volume in the RDF
20181003 - AKC - Adding functionality to calculate RDFs in multiple time steps
"""
                      
### IMPORTING MODULES
import MDDescriptors.core.import_tools as import_tools # Loading trajectory details
import MDDescriptors.core.calc_tools as calc_tools # Loading calculation tools
from MDDescriptors.core.check_tools import check_exists
import mdtraj as md # Running calculations
import sys

### FUNCTION TO FIND TOTAL SOLUTES AND RESIDUES GIVEN A TRAJECTORY
def find_multiple_residue_index( traj, residue_name_list ):
    '''
    The purpose of this function is to find multiple residue indices and total number of residues given a list of residue name list
    INPUTS:
        traj: [md.traj]
            trajectory from md.traj
        residue_name_list: [list]
            residue names in a form of a list that is within your trajectory
    OUTPUTS:
        total_residues: [list]
            total residues of each residue name list
        residue_index: [list]
            list of residue indices
    '''
    # CREATING EMPTY ARRAY TO STORE
    total_residues, residue_index = [], []
    # LOOPING THROUGH EACH POSSIBLE SOLVENT
    for each_solvent_name in residue_name_list:
        ## FINDING TOTAL RESIDUES
        each_solvent_total_residue, each_solvent_residue_index= calc_tools.find_total_residues(traj, resname=each_solvent_name)
        ## STORING
        total_residues.append(each_solvent_total_residue)
        residue_index.append(each_solvent_residue_index)
    return total_residues, residue_index

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
        frames: [list, default=[]]
            frames you want to calculate RDFs for. Useful when debugging to see when the RDF does in fact converge.
            Note: if -1 is included, then the last frame will by used
        want_oxy_rdf: [logical, default=False] 
            True or False if you want oxygen rdfs of the solute
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
        ### RDF DETAILS
            self.rdf_r: list of list containing r values for RDF
            self.rdf_g_r: list of list containing g(r) values for RDF
            ## FOR OXYGEN (if want_oxy_rdf is True)
            self.rdf_oxy_r: list of list containing r values for RDF [ [r1a, r2a, r3a ], [r1b, r2b, r3b]], where a and b are cosolvents
            self.rdf_oxy_g_r: similar to previous, but the g(r)
            self.rdf_oxy_names: names of each r, g_r
    FUNCTIONS:
        find_residue_index: function to get the residue index of solute and solvent
        find_solute_atom_names: Finds solute names based on an element (e.g. Oxygen, etc.)
        calc_rdf_solute_atom: calculates rdf based on a solute atom
        print_summary: prints summary of what was done
    '''
    #####################
    ### INITIALIZATION###
    #####################
    def __init__(self, traj_data, solute_name, solvent_name, bin_width, frames = [], cutoff_radius = None, want_oxy_rdf = None, save_disk_space=True):
        ### PRINTING
        print("**** CLASS: %s ****"%(self.__class__.__name__))
        
        ### DEFINING VARIABLES
        self.solute_name = solute_name
        self.solvent_name = solvent_name
        self.bin_width = bin_width
        self.cutoff_radius = cutoff_radius
        self.frames = frames
        self.want_oxy_rdf = want_oxy_rdf
        self.save_disk_space = save_disk_space
        
        ### TRAJECTORY
        traj = traj_data.traj
        
        ### FINDING ENSEMBLE AVERAGE VOLUME (USED FOR DENSITIES)
        self.volume = calc_tools.calc_ensemble_vol( traj )
        
        ### FINDING TOTAL FRAMES
        self.total_frames = len(traj)
        
        ### CHECK SOLVENT NAMES TO SEE OF THEY EXISTS IN THE TRAJECTORY
        self.solute_name = [ each_solute for each_solute in self.solute_name if each_solute in traj_data.residues.keys() ]
        self.num_solutes = len(self.solute_name)
        
        ### CHECK SOLVENT NAMES TO SEE OF THEY EXISTS IN THE TRAJECTORY
        self.solvent_name = [ each_solvent for each_solvent in self.solvent_name if each_solvent in traj_data.residues.keys() ]
        self.num_solvents = len(self.solvent_name)
        
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
        
        ### CREATING STORAGE VECTOR FOR RDFS
        self.rdf_r, self.rdf_g_r = [], []
        if self.want_oxy_rdf is True:
            self.rdf_oxy_r = []
            self.rdf_oxy_g_r = []
            self.rdf_oxy_names = [ find_solute_atom_names_by_type(self.solute_atom_names[index], atom_element = 'O') for index in range(self.num_solutes) ]
            
        ### CREATING STORAGE IF YOU WANT PER FRAME RDFs
        if len(frames) > 0:
            self.rdf_frames_r= []
            self.rdf_frames_g_r= []
            
        ### LOOPING THROUGH EACH SOLUTE
        for each_solute in range(self.num_solutes):
            ## GENERATING TEMPORARY VARIABLES
            solute_r= [] 
            solute_g_r = []
            if self.want_oxy_rdf is True:
                solute_oxy_r = []
                solute_oxy_g_r = []
            
            ## DEFINING VARIABLES FOR SOLUTES
            solute_residue_index = self.solute_res_index[each_solute]
            solute_COM = self.solute_COM[each_solute]
            solute_res_name = self.solute_name[each_solute]
            solute_atom_names = self.rdf_oxy_names[each_solute]
            
            ### LOOPING THROUGH EACH SOLVENT
            for each_solvent in range(self.num_solvents):
                ## PRINTING
                print("\n--- CALCULATING RDF FOR %s TO %s ---"%( self.solute_name[each_solute], self.solvent_name[each_solvent]   ) )
                ## DEFINING VARIABLES FOR SOLVENTS
                solvent_residue_index = self.solvent_res_index[each_solvent]
                solvent_COM = self.solvent_COM[each_solvent]
                
                ### CALCULATING RDF
                r, g_r, solvent_first_atom_index  =self.calc_rdf_com( traj = traj,                                          # Trajectory
                                                                       solute_residue_index     = solute_residue_index,     # Solute residue index
                                                                       solute_COM               = solute_COM,               # Solute center of mass
                                                                       solvent_residue_index    = solvent_residue_index,    # Solvent residue index
                                                                       solvent_COM              = solvent_COM,              # Solvent center of mass
                                                                       bin_width                = self.bin_width,           # Bin width for RDF
                                                                       cutoff_radius            = self.cutoff_radius,       # Cut off radius for RDf 
                                                                       periodic                 = True,                     # Periodic boundary conditions 
                                                                      )
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
                                                                           periodic                 = True,                     # Periodic boundary conditions 
                                                                          )
                
                
                ## APPENDING R, G(R)
                solute_r.append(r)
                solute_g_r.append(g_r)
                ### RUNNING RDFS FOR OXYGENS [ OPTIONAL ]
                if self.want_oxy_rdf is True:
                    oxy_r, oxy_g_r = self.calc_rdf_solute_atom(traj = traj,
                                                               solute_res_name = solute_res_name,
                                                               atom_names = solute_atom_names, 
                                                               solvent_first_atom_index = solvent_first_atom_index,
                                                               solvent_COM = solvent_COM)
                    ## APPENDING
                    solute_oxy_r.append(oxy_r)
                    solute_oxy_g_r.append(oxy_g_r)
                
            ### STORING RDFS FOR EACH SOLUTE
            self.rdf_r.append(solute_r)
            self.rdf_g_r.append(solute_g_r)
            
            ### FOR OXYGENS
            if self.want_oxy_rdf is True:
                self.rdf_oxy_r.append(solute_oxy_r)
                self.rdf_oxy_g_r.append(solute_oxy_g_r)
            ### CREATING STORAGE IF YOU WANT PER FRAME RDFs
            if len(frames) > 0:
                self.rdf_frames_r.append(frames_r)
                self.rdf_frames_g_r.append(frames_g_r)
                
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
        self.total_solutes, self.solute_res_index = find_multiple_residue_index(traj, residue_name_list=self.solute_name)
        ## SOLVENT
        self.total_solvent, self.solvent_res_index = find_multiple_residue_index(traj, residue_name_list=self.solvent_name)
        return
    
    ### FUNCTION TO RUN RDF OF OXYGENS
    def calc_rdf_solute_atom(self, traj, solute_res_name, atom_names, solvent_first_atom_index, solvent_COM, periodic = True ):
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
            periodic: [logical, default=True]
                True or false, if you want periodic boundary conditions to be taken into account
        OUTPUTS:
            r, g_r: rdfs for each of the elements in a form of a list
            atom_names: Names of each of the atoms for the element list
        '''        
        ### COPYING TRAJECTORY
        copied_traj=traj[:]
        
        ### COPYING THE SOLVENT CENTER OF MASSES TO TRAJECTORY
        copied_traj.xyz[:, solvent_first_atom_index] = solvent_COM[:]
        
        ### FINDING ALL ATOM INDEXES FOR EACH OF THE ATOM NAMES
        atom_index = [ atom.index for atom_name in atom_names for atom in traj.topology.atoms if atom.residue.name == solute_res_name and atom.name == atom_name]
        
        ### NOW, CREATING ATOM PAIRS
        atom_pairs_list =[ [ [atom_index[each_atom_name], interest_atom_indexes] for interest_atom_indexes in solvent_first_atom_index ]
                      for each_atom_name in range(len(atom_names))]
        
        ### CREATING EMPTY R AND G_R
        r, g_r = [], []
        
        ### LOOPING THROUGH EACH atom
        for each_atom in range(len(atom_names)):
            ## FINDING ATOM PAIRS
            atom_pairs = atom_pairs_list[each_atom]
            
            ## CALCULATING RDF
            element_r, element_g_r = md.compute_rdf(traj = copied_traj,
                     pairs = atom_pairs,
                     r_range=[0, self.cutoff_radius], # Cutoff radius
                     bin_width = self.bin_width,
                     periodic = periodic, # periodic boundary is on
                     )
            ## APPENDING
            r.append(element_r)
            g_r.append(element_g_r)
            
        return r, g_r

    
    
    ### FUNCTION TO FIND RDF BASED ON CENTER OF MASS
    @staticmethod
    def calc_rdf_com(traj, solute_residue_index, solute_COM, solvent_residue_index, solvent_COM, frames = [], bin_width=0.02, cutoff_radius=None, periodic=True):
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
        atom_pairs = [ [center_atom_indexes, interest_atom_indexes] for center_atom_indexes in solute_first_atom_index 
                      for interest_atom_indexes in solvent_first_atom_index ]
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
                ## PRINTING
                print("Working on frame index %d out of %d, going up to %d frame"%(frame_idx, len(frames), each_frame) )
                ## DEFINING CURRENT TRAJECTORY
                current_traj=copied_traj[:each_frame]
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
    analysis_dir=r"181002-XYL_testing_HYD_sims_other_solvents" # Analysis directory
    # analysis_dir=r"180316-ACE_PRO_DIO_DMSO"
    specific_dir="HYDTEST\\HYDTEST_300.00_6_nm_xylitol_10_WtPercWater_spce_dioxane" # Directory within analysis_dir r"mdRun_363.15_6_nm_tBuOH_100_WtPercWater_spce_Pure"
    # specific_dir="ACE/mdRun_433.15_6_nm_ACE_10_WtPercWater_spce_dioxane" # Directory within analysis_dir r"mdRun_363.15_6_nm_tBuOH_100_WtPercWater_spce_Pure"
    # specific_dir=r"Planar_310.15_ROT_TMMA_10x10_CHARMM36_withGOLP" # Directory within analysis_dir
    path2AnalysisDir=r"R:\scratch\SideProjectHuber\Analysis\\" + analysis_dir + '\\' + specific_dir # PC Side
    
    ### DEFINING FILE NAMES
    gro_file=r"mixed_solv_prod.gro" # Structural file
    xtc_file=r"mixed_solv_prod_whole_last_50ns.xtc" # r"mixed_solv_prod_last_90_ns_center_rot_trans_center_prog_rot_trans_center.xtc" # Trajectory
    # xtc_file=r"mixed_solv_last_50_ns_whole.xtc"
    
    ### LOADING TRAJECTORY
    traj_data = import_tools.import_traj( directory = path2AnalysisDir, # Directory to analysis
                 structure_file = gro_file, # structure file
                  xtc_file = xtc_file, # trajectories
                  )
    #%%
    ### DEFINING INPUT DATA
    input_details={
                    'solute_name'        : ['XYL', 'HYD'],           # Solute of interest as a form of a list ['XYL', 'HYD']
                    'solvent_name'       : ['HOH', 'DIO', 'GVLL'],   # Solvents you want radial distribution functions for
                    'bin_width'     : 0.02,              # Bin width of the radial distribution function
                    'cutoff_radius' : 2.00,              # Radius of cutoff for the RDF (OPTIONAL)
                    'frames'        : [1000, 2000, 3000, 4000, -1]   , # Frames to run, default = []
                    'want_oxy_rdf'  : True,              # True if you want oxygen rdfs
                    }
    
    ### CALLING RDF CLASS
    rdf = calc_rdf(traj_data, **input_details)
    


    