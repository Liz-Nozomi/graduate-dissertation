# -*- coding: utf-8 -*-
"""
electrostatic_potential.py
The purpose of this code is to compute the potential across the box. This is 
analogous to "gmx potential" in GROMACS. We need a python version of this code
due to the lack of documentation in the original gmx potential code. 

Written by: Alex K. Chew (alexkchew@gmail.com)

"""
### IMPORTING MODULES
import numpy as np
import mdtraj as md
import sys
import glob
import time

## INTEGRATION MODULES
from scipy import integrate

## CUSTOM MODULES
import MDDescriptors.core.import_tools as import_tools # Loading trajectory details
import MDDescriptors.core.calc_tools as calc_tools # calc tools
import MDDescriptors.core.read_write_tools as read_write_tools # Reading itp file tool

############################################################################
### GLOBAL VARIABLES
############################################################################
## DEFINING GLOBAL VARIABLES
EPS0=8.85419E-12 # C / (Vm)
ELC=1.60219E-19 # C / electron

## WATER MODEL CHARGES
WATER_MODEL_CHARGES={
                     'TIP3P':
                         {'O': -0.834,
                          'H':  0.417,},
                     'SPCE':
                         {'O': -0.8476,
                          'H':  0.4238,}
                     }

## COUNTER_ION CHARGES
COUNTER_ION_CHARGES={
        'CL': -1,
        'NA': 1,    
        }

############################################################################
### FUNCTIONS
############################################################################







############################################################################
### CLASSES
############################################################################

### CLASS TO COMPUTE THE POTENTIAL
class calc_electrostatic_potential:
    '''
    The purpose of this code is to compute the electrostatic potential across 
    the box. Here, we will use Gauss's law to compute charges across 
    multiple frames.
    INPUTS:
        traj_data: [obj]
            Data taken from import_traj class
        water_model: [str, default="SPCE"]
            water model used for your molecular simulations
        bin_width: [float, default=0.02 nm]
            bin width to compute the electrostatic potential
        direction: [str, default="z"]
            direction to get the electrostatic potential. Possible options are
             "z", "x", and "y" for now.
        truncate_edges: [float, default=0.000]
            amount of edges you want to truncate in both ends. 
            e.g., a value of 0.2 nm would truncate 0.2 nm in both edges of the perpendicular direction
    OUTPUTS:
        
        
    FUNCTIONS:
        compute_direction_vectors: function that computes the directional vector to integrate from
        calc_box_bounds: calculates the box boundaries to know how far to histogram
    '''
    ## INITIALIZING
    def __init__(self, traj_data, water_model = "SPCE", direction="z" , truncate_edges = 0.000,
                 bin_width=0.02, n_bins=None):
        ## STORING INPUTS
        self.water_model = water_model
        self.bin_width = bin_width
        self.direction = direction
        self.truncate_edges = truncate_edges
        self.n_bins = n_bins
        
        ## DEFINING TRAJ
        traj = traj_data.traj
        
        ## FINDING TOTAL FRAMES
        self.total_frames = traj_data.num_frames
        
        ## COMPUTING DIRECTION INDEX
        self.compute_direction_vectors()
        
        ## DEFINING BOX BOUNDS
        self.calc_box_bounds(traj=traj)
        
        ## FINDING ALL UNIQUE ATOMS
        self.find_all_unique_atoms_per_residue(traj=traj)
        
        ## ASSIGNING CHARGES TO EACH ATOM
        self.assign_charges_unique_atoms()
        
        ## COMPUTING CHARGES
        self.compute_charge_density(traj = traj)
        
        ## COMPUTING ELECTROSTATIC POTENTIAL
        self.compute_electrostatic_potential()
        
        return
        
    ### FUNCTION TO COMPUTE THE DIRECTIONAL VECTOR
    def compute_direction_vectors(self):
        ''' This simply computes what direction we are integrating towards '''
        if self.direction == "z":
            self.direction_perp = 2 # Perpendicular vector
        elif self.direction == "y":
            self.direction_perp = 1
        elif self.direction == "x":
            self.direction_perp = 0
        else:
            print("Error! Direction %s is not a valid option!"%(self.direction) )
            print("Stopping here to prevent subsequent errors!")
            sys.exit()
        ## COMPUTING VECTORS THAT ARE NOT WITHIN THE DIRECTION
        self.direction_parallel=[0,1,2]
        self.direction_parallel.remove( self.direction_perp )
        
    ### FUNCTION TO COMPUTE BOX BOUNDS
    def calc_box_bounds(self, traj):
        '''
        The purpose of this function is to compute the box boundaries using the
        available trajectory and bin size.
        INPUTS:
            self: [obj]
                class object
            traj: [obj]
                trajectory from md.traj
        OUTPUTS:
            self.box_range: [list, size=2]
                box range (min, max) in nm
            self.n_bins: [int]
                number of bins across box range
            self.area_parallel: [float]
                ensemble area parallel to the direction of interest
            self.bin_volume: [float]
                ensemble bin volume
        '''
        ## DEFINING UNIT CELL LENGTHS
        box_perp_lengths = traj.unitcell_lengths[:, self.direction_perp]
        ## COMPUTING MAXIMUM RANGE
        box_perp_max_value = np.max(box_perp_lengths)
        box_perp_min_value = 0 # Default
        ## DEFINING BOX RANGE
        self.box_range = [ box_perp_min_value, box_perp_max_value  ]
        ## SEEING IF TRUNCATION IS NECESSARY
        if self.truncate_edges > 0:
            self.box_range[0]+=self.truncate_edges
            self.box_range[1]-=self.truncate_edges
        
        ## DEFINING NUMBER OF BINS
        if self.n_bins is None:
            self.n_bins =int( (self.box_range[1] - self.box_range[0]) / self.bin_width)
        else:
            print("Since n_bins is specified, we are ignoring any inputs for bin_width!")
        ## COMPUTING AREA OF THE BOX (ENSEMBLE AVERAGE)
        self.area_parallel = np.mean(np.prod(traj_data.traj.unitcell_lengths[:,[self.direction_parallel]],axis=1))
        ## COMPUTING VOLUME
        self.bin_volume = np.mean(traj.unitcell_volumes)/ self.n_bins   # self.area_parallel * self.bin_width
        return
        
    ### FUNCTION TO FIND ALL UNIQUE ATOMS PER RESIDUE
    def find_all_unique_atoms_per_residue(self, traj):
        '''
        The purpose of this function is to find all unique atoms per residue 
        basis. 
        INPUTS:
            self: [obj]
                class object
            traj: [obj]
                trajectory from md.traj
        OUTPUTS:
            self.res_atom_dict: [dict]
                dictionary of the residue and atom names that are unique
        '''
        ## FINDING UNIQUE SET OF RESIDUES
        residues = calc_tools.find_unique_residue_names( traj = traj )
        
        ## FOR EACH RESIDUE, WE WILL COMPUTE A LIST OF ATOM NAMES
        atom_names = [ calc_tools.find_atom_names(traj = traj,
                                       residue_name = each_residue)
                                       for each_residue in residues ]
        
        ## CREATING DICTIONARY
        self.res_atom_dict = {}
        for idx, each_residue in enumerate(residues):
            self.res_atom_dict[each_residue] = atom_names[idx]
        return

    ### FUNCTION TO ASSIGN CHANGES
    def assign_charges_unique_atoms(self):
        '''
        The purpose of this function is to assign charges to each atom type. 
        Here, we will utilize and itp file and global variables to correctly 
        assign each atom the correct charge.
        INPUTS:
            self: [obj]
                class object
        OUTPUTS:
            self.res_atom_charges: [dict]
                dictionary containing charges for each atom in the residue
        '''
        ## CREATING DICTIONARY FOR CHARGES
        self.res_atom_charges = {}
        ## LOOPING THROUGH EACH RESIDUE
        for each_residue in self.res_atom_dict.keys():
            ## CREATING EMPTY DICTIONARY
            self.res_atom_charges[each_residue] = {}
            ## LOOPING THROUGH EACH ATOM
            for each_atom in self.res_atom_dict[each_residue]:
                ## MAKE EXCEPTION FOR HOH
                if each_residue == 'HOH':
                    if 'O' in each_atom:
                        self.res_atom_charges[each_residue][each_atom] = \
                                    WATER_MODEL_CHARGES[self.water_model]['O']
                    elif 'H' in each_atom:
                        self.res_atom_charges[each_residue][each_atom] = \
                                    WATER_MODEL_CHARGES[self.water_model]['H']
                ## OTHERWISE, SEARCH FOR ITP FILE AND LOCATE CHARGES
        return
    
    ### FUNCTION TO USE UNIQUE CHARGES TO COMPUTE CHARGE WITH RESPECT TO R
    def compute_charge_density(self,traj):
        '''
        The purpose of this function is to compute the charge density with 
        respect to a specific perpendicular direction. We will loop through 
        each of the unique atoms and compute a histogram of the distance 
        with respect to some distance
        INPUTS:
            self: [obj]
                class object
        OUTPUTS:
            self.charge_density: [np.array, shape=(n_bins)]
                charge density in C/nm3 with respect to the perpendicular direction
            self.dist_vec: [np.array, shape=(n_bins)]
                distance vector corresponding to the histograms in nm
        '''
        ## STORAGE VECTOR FOR CHARGES
        self.charge_density = np.zeros(self.n_bins) # {}
        
        ## LOOPING THROUGH EACH RESIDUE
        for each_residue in self.res_atom_charges:
            ## LOOPING THROUGH EACH ATOM TYPE
            for each_atom in self.res_atom_charges[each_residue].keys():
                ## DEFINING CURRENT CHARGE
                atom_charge = self.res_atom_charges[each_residue][each_atom]
                ## FINDING INDICES FOR THESE ATOM TYPES
                indices = np.array(calc_tools.find_atom_index( traj = traj,
                                                      atom_name = each_atom,
                                                      resname = each_residue)) ## SHAPE: num_atoms
                ## FINDING TRAJECTORY POSITIONS
                positions = traj.xyz[:,indices,self.direction_perp] ## SHAPE: num_frames, num_atoms
                ## FINDING HISTOGRAM
                hist, edges = np.histogram(positions, 
                                           range=self.box_range,
                                           bins = self.n_bins ) ## SHAPE: Num_bins
                ## PRINTING
                print("Adding charges for %s -- %s, %d atoms, charge = %.3f"%(each_residue, each_atom, len(indices), atom_charge) )
                ## ADDING TO CHARGE DENSITY
                self.charge_density+= hist * atom_charge # Units of elementary charges
                self.edges = edges
        
        ## CORRECTING CHARGE DENSITY BASED ON TOTAL FRAMES AND VOLUME
        self.charge_density = self.charge_density  / (self.total_frames*self.bin_volume) # electron charges/ (nm)^3 * ELC
        ## COMPUTING DISTANCE VECTOR
        self.dist_vec = 0.5 * (self.edges[1:] + self.edges[:-1]) # nm
        return
        
    ### FUNCTION TO COMPUTE ELECTROSTATIC POTENTIAL BY INTEGRATION
    def compute_electrostatic_potential(self):
        '''
        This script computes the electrostatic potential by integrating the 
        charge density twice. The equations for the electrostatic potential
        are defined as follows:
            Electric field
                E_z(z) = int_(-infty)^z ( rho(z) / eps_0 ) dz
                    - E_z(z) [=] V/m
                    - rho(z) charge density with respect to z [=]C/m3
                    - eps_0 dielectric permittivity of vacuum [=] C/(Vm)
            Electrostatic potential
                phi(z) = - int_(-infty)^z ( E_z(z) ) dz
                    - phi(z) [=] V
        To integrate, we will need to specify integration scheme. We can 
        numerically integrate in many ways, but the end result should 
        remain the same. Note that the integrals are cumulative integration. 
        Therefore, any integration script must take the cumulative sum with 
        respect to the first value.
        '''
        ## CONVERTING CHARGE DENSITY TO SI UNITS
        nm2meters = 1E-9
        self.charge_density_SI_units = self.charge_density * ELC # V / nm^3  # * ( 1 / nm2meters )**3 # C / m^3
        self.dist_vec_SI_units = self.dist_vec # nm  # * nm2meters # m
        ## INTEGRATING TO GET ELECTRIC FIELD
        self.E_field = integrate.cumtrapz( self.charge_density_SI_units / (EPS0 * nm2meters ), self.dist_vec_SI_units, initial = 0    ) # V/nm
        # y_int = integrate.cumtrapz(y, x, initial=0)
        ## INTEGRATING TO GET POTENTIAL
        self.potential = - integrate.cumtrapz( self.E_field, self.dist_vec_SI_units, initial = 0    )  # V
        
        return
        
    ### FUNCTION TO PLOT CHARGE DENSITY WITH RESPECT TO SOME DIMENSION
    def plot_charge_density(self):
        ''' This plots the charge density vs. r '''
        ## IMPORTING MODULES
        import matplotlib.pyplot as plt
        from MDDescriptors.core.plot_tools import create_plot
        ## CREATING PLOT
        fig, ax = create_plot()
        ## SETTING AXIS
        ax.set_ylabel("Charge density (elementary charge/nm3)")
        ax.set_xlabel("Distance (nm)")
        ## PLOTTING THE DATA
        ax.plot( self.dist_vec, self.charge_density, color='k', linewidth =2  )
        ## CREATING PLOT
        fig, ax = create_plot()
        ax.set_ylabel("Electric field (V/nm)")
        ax.set_xlabel("Distance (nm)")
        ## PLOTTING THE DATA
        ax.plot( self.dist_vec, self.E_field, color='k', linewidth =2  )
        
        ## PLOT FOR  POTENTIAL
        ## CREATING PLOT
        fig, ax = create_plot()
        ax.set_ylabel("Electrostatic potential (V)")
        ax.set_xlabel("Distance (nm)")
        ## PLOTTING THE DATA
        ax.plot( self.dist_vec, self.potential, color='k', linewidth =2  )
        return
        
        





#%%
############################################################################
### MAIN CODE
############################################################################
if __name__ == "__main__":
    ### DIRECTORY TO WORK ON    
    analysis_dir=r"190114-Expanded_NoSolute_Pure_Sims" # Analysis directory
    category_dir="NoSolute" # category directory
    specific_dir="Expand_8nm_300.00_6_nm_NoSolute_100_WtPercWater_spce_Pure" # spherical_310.15_K_2_nmDIAM_octanethiol_CHARMM36" # Directory within analysis_dir r"mdRun_363.15_6_nm_tBuOH_100_WtPercWater_spce_Pure"    
    path2AnalysisDir=r"R:\scratch\\SideProjectHuber\analysis\\" + analysis_dir + '\\' + category_dir + '\\' + specific_dir + '\\' # PC Side
    
    ### DEFINING FILE NAMES
    gro_file=r"mixed_solv_prod.gro" # Structural file
    # xtc_file=r"sam_prod_10_ns_center.xtc" # r"sam_prod_10_ns_whole.xtc" # Trajectory file
    xtc_file=r"mixed_solv_prod_5_ns_whole.xtc"
    xtc_file="mixed_solv_prod_5_ns_center.xtc"
    ### LOADING TRAJECTORY
    traj_data = import_tools.import_traj( directory = path2AnalysisDir, # Directory to analysis
                 structure_file = gro_file, # structure file
                  xtc_file = xtc_file, # trajectories
                  )
    
    #%%
    # Gromacs code to move the water away from pbc    
    #  gmx trjconv -f mixed_solv_prod_5_ns_whole.xtc -s mixed_solv_prod.tpr -o mixed_solv_prod_5_ns_center.xtc -pbc mol -trans 0 0 6.07377
    ### DEFINING INPUTS
    input_vars= {
            'traj_data': traj_data,
            'bin_width': 0.02,
            'water_model': 'SPCE',
            'direction': 'z',
            'truncate_edges': 0.000,
            'n_bins': 600
            
            }
    
    ### RUNNING POTENTIAL
    potential = calc_electrostatic_potential(**input_vars)
    
    
    #%%
    plt.close('all')
    potential.plot_charge_density()
    
    #%%
    from scipy import integrate
    x = np.linspace(-2, 2, num=20)
    y = x
    y_int = integrate.cumtrapz(y, x, initial=0)
    plt.plot(x, y_int, 'ro', x, y[0] + 0.5 * x**2, 'b-')
    plt.show()