#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
class_template.py
The purpose of this template is to give you some background in generating classes

Created on: 03/27/2018

CLASSES:
    calc_res_densities: class that calculates the residue densities
    
DEFINITIONS:
    print_density: prints the density of the output by calc_res_densities
    
Author(s):
    Your_Name (Your_email@address.com)
    
UPDATES:
    20180327: Updated this file to include classes
    20180328: Completed tutorial
"""
#########################
### IMPORTING MODULES ###
#########################
import numpy as np # Typical way to import a pre-loaded module, in this case NumPy -- great for math!
import MDDescriptors.core.import_tools as import_tools # Alternatively, you can import from your own python script. Here, we are importing MDDescriptors -> Core -> import_tools.py
# Note that periods denote a new directory. You never import something with ".py" attached (It would be very confusing then! Is py a file by itself? -- Thus, no .py, ever.)


# Please import modules necessary for calculation at the beginning! These modules are passed through every class and function, so you do not have to recall it!
# In addition, these modules are always reloaded when you import this script (so your script will not fail as long as the loaded module is here!)
##########################
### GLOBAL DEFINITIONS ###
##########################
# Please make all global variables ALLCAPS
# EXAMPLES:
#   PI=3.1457....
# All variables defined at the beginning of the script is considered a global variable!

##########################################################
### CLASS FUNCTION TO CALCULATE DENSITIES OF A RESIDUE ###
##########################################################
class calc_res_densities:
    '''
    The purpose of this class is to calculate the densities of a residue
    
    INPUTS:
        traj_data: (class) traj data from import_traj
        residue_name: (string) residue name
            e.g. 'HOH' (default)

    OUTPUTS:
        ## INPUTS
            self.residue_name: residue name
        ## TRAJECTORY DETAILS
            self.average_volume: (float) average volume
        ## RESIDUE DETAILS
            self.res_indices: (list) residue index of the solute
            self.total_residues: (int) total residues
        
    FUNCTIONS:
        find_total_residues: finds total residue
        calc_average_density: [staticmethod] finds density
    
    '''
    #####################
    ### INITIALIZATION###
    #####################
    def __init__(self, traj_data, residue_name='HOH'): # <-- INPUT VARIABLES ARE HERE. SELF IS ALWAYS INPUTTED
        
        ### DEFINING VARIABLES
        self.residue_name = residue_name # <-- stores name of the solute into "self.solute_name"
        
        ### DEFINING TRAJECTORY
        traj = traj_data.traj # traj_data should have been generated from the "import_tools" class, and tagged with the trajectory details (i.e. traj)
        
        ### FINDING AVERAGE VOLUME
        self.average_volume = np.mean(traj_data.traj.unitcell_volumes) ## USING NUMPY TO AVERAGE THE VOLUMES
        
        ### NOW, FINDING ALL ATOM NUMBERS WITH A FUNCTION
        self.find_total_residues(traj)
        # RETURNS: 
        #   self.res_indices
        #   self.total_residues
        
        ## LASTLY, FINDING DENSITIES USING A STATIC FUNCTION, AS AN EXAMPLE. Note that you do not need to use static functions, but it was used here for learning purposes.
        self.density = self.calc_average_density(   num_of_residues = self.total_residues, # Total residues
                                                     average_volume  = self.average_volume # Total volume
                                                )

        return # Typically no returns for classes. The class itself is returned along with all its variables
        
    ## EXAMPLE OF A CLASS FUNCTION
    ### FUNCTION TO CALCULATE TOTAL RESIDUES
    def find_total_residues(self, traj):
        '''
        The purpose of this function is to get all the residues
        INPUTS:
            self: class property
        
        OUTPUTS:
            self.res_indices: (list) residue index of the solute
                e.g. [ 1,2,3 ] for solute residue index
            self.total_residues: (int) total residues
                e.g. "2" for 2 residues        
        '''
        self.res_indices = [ res for res in traj.topology.residues if res.name == self.residue_name ]
        self.total_residues = len(self.res_indices)
        return # NOTE, you don't need to return anything since you saved to self!s
        
    
    ## EXAMPLE OF A STATIC FUNCTION
    ### FUNCTION TO TAKE THE TOTAL NUMBER OF RESIDUES / VOLUME
    @staticmethod
    def calc_average_density( num_of_residues, average_volume ):
        '''
        The purpose of this function is to show you what a static function does. Note! There is no "self" property here. This function acts on it's own without the class (thus "static")
        INPUTS:
            num_of_residues: (int) total number of residues
            average_volume: (float) total volume in nm^3
        OUTPUTS:
            density: (float) number of residues divided by volume
        '''
        density =  num_of_residues / average_volume
        return density
    
###################
### DEFINITIONS ###
###################
## EXAMPLE OF A DEFINITION (I.E. FUNCTION)
### FUNCTION TO PRINT THE DENSITY GIVEN THE DENSITY CLASS
def print_density(density):
    '''
    The purpose of this function is to take your density class and print what comes out of it.
    INPUTS:
        density: density from calc_res_densities class
    OUTPUTS:
        void: simply prints the densities
    '''
    print("EXAMPLE OF A DEFINITION")
    print("The density for %s is: %.3f"%(density.residue_name, density.density))
    return    


#%% MAIN SCRIPT
if __name__ == "__main__":
    ## NOTE: Everything under this is run if this script run! If this script is imported, nothing below will run!
    ## NOTE: This serves as a great way of debugging the code to ensure that it runs correctly!
    
    ## DEFINING DIRECTORY TO WORK ON
    tutorial_dir="MDDescriptors/tutorials/0_Density_tBuOH_50_GVLL"
    
    ### DEFINING FILE NAMES
    gro_file=r"mixed_solv_prod.gro" # Structural file
    xtc_file=r"mixed_solv_prod_last_10_ns.xtc" # Trajectory xtc file
    
    ##########################
    ### LOADING TRAJECTORY ###
    ##########################
    traj_data = import_tools.import_traj( directory = tutorial_dir, # Directory to analysis
                 structure_file = gro_file, # structure file
                  xtc_file = xtc_file, # trajectories
                  )
    #%% <-- SPYDER's way of separating portions of the code
    
    ############################
    ### RUNNING PYTHON CLASS ###
    ############################
    
    ## WRITING INPUT DETAILS
    # NOTE: THIS IS A DICTIONARY. DICTIONARIES ARE LIKE LISTS, BUT INSTEAD OF INDEXES, YOU HAVE STRINGS. THUS, input_details['residue_name'] WILL OUTPUT 'tBuOH'
    input_details = {
                        'residue_name' : 'tBuOH' # Solute of interest
                        }
    
    ### RUNNING CLASS
    densities = calc_res_densities(traj_data, **input_details)
    # NOTE: **input_details means to give all the inputs as such that residue_name will be tBuOH. 
    
    ### CHECKING IF YOUR CLASS RAN SUCCESSFULLY
    print("The density for %s is: %.3f"%(densities.residue_name, densities.density))
    
    ### YOU COULD HAVE ALSO USED A FUNCTION
    print_density(densities)
    
    
    
    