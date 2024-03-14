#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preferential_interaction_coefficients.py
The purpose of this script is to calculate preferential interaction coefficients for a given trajectory. The preferential interaction coefficient informs you about the preferential bonding of a solute to solvent

Created on: 03/28/2018

FUNCTIONS:
    calc_kirkwood_integral: calculates the kirkwood buff integral
    calc_excess_coord_num: calculates the excess coordination number

CLASSES:
    calc_pref_interaction_coefficients: class to calculate preferential interaction coefficients (and now excess coordination number)

Author(s):
    Alex K. Chew (alexkchew@gmail.com)
    
** UPDATES **
20180430 - AKC - Added excess coordination number and kirkwood buff integral calculations
20190109 - AKC - Updating script to incorporate multiple solutes -- fixing up to incorporate checking functions
"""


## SYSTEM TOOLS
import numpy as np
import mdtraj as md # Running calculations
import sys

### IMPORTING MODULES
import MDDescriptors.core.import_tools as import_tools # Loading trajectory details
import MDDescriptors.core.calc_tools as calc_tools # Loading calculation tools

## IMPORTING RDF TOOL
from MDDescriptors.geometry.rdf import calc_rdf

## CUSTOM FUNCTIONS
import MDDescriptors.application.solvent_effects.check_functions as check_functions

### FUNCTION TO CALCULATE KIRKWOOD BUFF INTEGRALS
def calc_kirkwood_integral(r, g_r, bin_width):
    '''
    The purpose of this function is to find the Kirkwood integral (KBI) given the RDF. General form of KBIs: \int_0^R [ g_ij (r) - 1 ] * 4 * pi *r^2 dr
    INPUTS:
        r: [np.array] Radius vector
        g_r: [np.array] radial distribution function
        bin_width: [float] Width of the bin -- used for integration
    OUTPUTS:
        kirkwood_int: Value of Kirkwood-Buff Analysis
    '''
    # Defining the y-function
    kirkwood_y_func = (g_r - 1)*4*np.pi*r**2

    # Finding the integral
    kirkwood_int = np.trapz(kirkwood_y_func,r,dx=bin_width) # nm^3 /mol
    return kirkwood_int

### FUNCTION TO CALCULATE EXCESS COORDINATION NUMBERS
def calc_excess_coord_num(traj, r, g_r, bin_width, residue_of_interest):
    '''
    This script takes your RDF and calculates the kirkwood buff integral and excess coordination number. 
    INPUTS:
        traj: traj from md.traj
        r: radius vector for your radial distribution function
        g_r: g(r), RDF
        residue_of_interest: Residue you are interested in the bulk phase (e.g. 'HOH', etc.)
    OUTPUTS:
        excess_coord_num: Excess Coordination Number
        kirkwood_int: Kirkwood buff integral
    '''
    # Finding ensemble volume
    vol = calc_tools.calc_ensemble_vol(traj=traj) # In nanometers
    # Finding number of residues and their indexes
    num_residues, _ = calc_tools.find_total_residues( traj=traj, resname=residue_of_interest)
    # Finding number density
    numDensity = num_residues / float(vol) # Number / nm^3 <-- Makes sure we have a floating number for divisions!
    # Now, finding Kirkwood-integral
    kirkwood_int = calc_kirkwood_integral( r, g_r, bin_width) # nm^3 / mol
    # Finding excess coordination number
    excess_coord_num = numDensity * kirkwood_int
    return excess_coord_num, kirkwood_int

#################################################################
### CLASS FUNCTION TO CALCULATE RADIAL DISTRIBUTION FUNCTIONS ###
#################################################################
class calc_pref_interaction_coefficients:
    '''
    The purpose of this class is to calculate preferential interaction coefficients defined by:
            G_23(r,t) = n_3(r,t) - n_1(r,t) * (n_3 - n_3(r,t) / (n_1 - n_1(r,t)) )
    This tells us how much excess the number of cosolvent molecules within a local domain around our solute compared to the bulk. 
    Obviously, we would like to see the excess number of water molecules relative to the bulk as well. Perhaps we can tweak this equation to find that. 
    INPUTS:
        traj_data: Data taken from import_traj class
        solute_name: [str] name of the solute (assuming single solute)
        solvent_name: [list] list of solvents you are interested in. NOTE: There must be at least two solvents! If not, then we will not continue the script.
            Preferential interaction coefficient for a pure system is simply zero.
        bin_width: [float] Bin width of the radial distribution function
        cutoff_radius: [float] Radius of cutoff for the RDF (OPTIONAL) [Default: None]
        want_oxy_pref_int_coeff: [logical] True if you want preferential interaction coefficients of just the oxygens
        tolerance: [float] tolerance when you say the RDF has reached equilibrium [Default: 0.015]
        traj_num_split: [int] number of times you want to split your trajectory to average and standard deviation [Default: 2]
        save_disk_space: [logical] True if you want to save memory space, will turn off the following variables:
            self.rdf.solute_COM, self.rdf.solvent_COM, self.input_split_vars
    OUTPUTS:
        ## INITIAL VARIABLES ARE STORED
            self.solute_name, self.solvent_name, self.bin_width, self.cutoff_radius, self.want_oxy_pref_int_coeff, self.tolerance
        
        ## SOLVENT INFORMATION
            self.total_solvents: [list] total solvents of each solvent
            self.solvent_residue_index: [list] list of residue index of each solvent
            self.water_index: [int] index in solvent_name of water
            self.cosolvent_index: [int] index in solvent_name of cosolvent
            self.total_water: [int] total number of water
            self.total_cosolvent:[int] total number of cosolvent
            
        ## RDF INFORMATION
            self.equil_radius: [float] equilibration radius in nms
            
        ## VARIABLES
            self.input_static_vars: [dict] includes all static variables used (and changed)
            self.input_split_vars: [dict] includes all splitting variables (by split, I mean split for splitted trajectories)
            
        ## RESULTS
            self.pref_int_coeff: [dict] contains preferential interation coefficient results (Note, this is average and standard deviationed)
                    'gamma_23': gamma_23 as we would expect from:
                        G_23(r,t) = n_3(r,t) - n_1(r,t) * (n_3 - n_3(r,t) / (n_1 - n_1(r,t)) )
                    'ngamma_23': negative of gamma_23 (useful to have opposite conventions)
                    'n_1_local_avg': average value of water around the local domain
                    'n_1_bulk_avg': average value of water around the bulk domain
                    'n_3_local_avg': average value of cosolvent in the local domain
                    'n_3_bulk_avg': average value of cosolvent in the bulk domain
            
        ## PREF INT FOR EACH OXYGEN
            self.rdf.rdf_oxy_names: [list] list of each oxygen name
            self.solute_atom_cutoff_radius: [list] list of cutoff radii for the solute
            self.pref_int_coeff_solute_atom: [dict] dictionary of output data as a list
        
        ## EXCESS COORDINATION NUMBER
            self.excess_coord_num: [list] list of excess coordination numbers for each solvent
            self.excess_coord_num_oxy: [list] list of each oxygen's excess coordination number for each solvent
        
    FUNCTIONS:
        calc_cutoff_radius: calculates cutoff radius based on radial distribution function given r, g(r)
        find_distances_btn_solute_solvent: calculates distances between solute and solvent for each frame
        find_distances_btn_solute_solvent: finds the distances between solute and solvent
        calc_pref_int_coeff: calculates preferential interaction coefficient between the solute and solvents
        calc_pref_int_coeff_solute_atom: calculates preferential interaction coefficents for each solute atom
        
    ALGORITHM:
        - Check which solvents are available in the system
        - Find total number of water molecules / cosolvent molecules and their residue indexes
        - Find the RDF between the solute and pure water (assumed to be the one that we use preferential interaction coefficients on)
        - Calculate the RDF cutoff between solute and pure water
        - Go through each frame and calculate:
                G_23(r,t) = n_3(r,t) - n_1(r,t) * (n_3 - n_3(r,t) / (n_1 - n_1(r,t)) )
        - At the end, get an ensemble average of G_23(r,t)
    '''
    #####################
    ### INITIALIZATION###
    #####################
    def __init__(self, traj_data, solute_name, solvent_name, bin_width, job_type=['pref_int'], cutoff_radius = None, want_oxy_pref_int_coeff = False, tolerance = 0.015,
                 traj_num_split=2, save_disk_space = True):
        ### PRINTING
        print("**** CLASS: %s ****"%(self.__class__.__name__))
        
        ## STORING INITIAL VARIABLES
        self.solute_name = solute_name
        self.solvent_name = solvent_name
        self.bin_width = bin_width
        self.cutoff_radius = cutoff_radius
        self.want_oxy_pref_int_coeff = want_oxy_pref_int_coeff
        self.tolerance = tolerance
        self.traj_num_split = traj_num_split
        self.save_disk_space = save_disk_space
        
        ## CHECKING JOB TYPES
        available_job_types = ['pref_int','excess_coord']
        ## FINDING CORRECT JOB TYPES (SCREENING ALL INNCORRECT JOB TYPES)
        self.job_types = [ each_structure for each_structure in job_type if each_structure in available_job_types]
        print("Final job type list: %s"%(', '.join(self.job_types) ))
        
        ### TRAJECTORY
        traj = traj_data.traj
        
        ### CHECING IF SOLUTE AND SOLVENT NAMES ARE WITHIN THE DIRECTORIES
        self.solute_name, self.solvent_name = check_functions.check_solute_solvent_names( traj_data = traj_data,
                                                                                          solute_name = self.solute_name,
                                                                                          solvent_name = self.solvent_name,
                                                                                         )
        
        '''
        ### CHECK IF SOLUTE EXISTS IN TRAJECTORY
        if self.solute_name not in traj_data.residues.keys():
            print("ERROR! Solute (%s) not available in trajectory. Stopping here to prevent further errors. Check your 'Solute' input!")
            sys.exit()
            
        ### CHECK SOLVENT NAMES TO SEE OF THEY EXISTS IN THE TRAJECTORY
        self.solvent_name = [ each_solvent for each_solvent in self.solvent_name if each_solvent in traj_data.residues.keys() ]
            
        ### CHECKING IF SOLVENT LIST IS CORRECT
        if len(self.solvent_name) != 2 or 'HOH' not in self.solvent_name:
            print("ERROR! We do not have two solvents (only %d is available) or 'HOH' is not defined in the system"%(len(self.solvent_name)))
            if 'HOH' in self.solvent_name and len(self.solvent_name)==1:
                print("Since only water is found, preferential interaction coefficient is obviously zero!")    
            else:
                print("Stopping here to prevent further errors!")
        '''                

        if (len(self.solvent_name) == 2 and 'HOH' in self.solvent_name) or 'excess_coord' in self.job_types:
            ## DEFINING RDF INPUT
            rdf_input={
                        'solute_name'       : self.solute_name,                     # Solute of interest
                        'solvent_name'      : self.solvent_name,                    # Solvents you want radial distribution functions for
                        'bin_width'         : self.bin_width,                       # Bin width of the radial distribution function
                        'cutoff_radius'     : self.cutoff_radius,                   # Radius of cutoff for the RDF (OPTIONAL)
                        'want_oxy_rdf'      : self.want_oxy_pref_int_coeff,         # True if you want oxygen rdfs
                        }
        
            
            ## CALCULATING RDF
            self.rdf = calc_rdf(traj_data, save_disk_space = False, **rdf_input )
            
        # if len(self.solvent_name) == 2 and 'HOH' in self.solvent_name:
        ## RUNNING ONLY IF WE HAVE SATISFACTORY DETAILS
        if 'HOH' in self.solvent_name:
            ## ONLY RUN FOR PREFERENTIAL INTERACTION COEFFICIENT
            if 'pref_int' in self.job_types:
                
                ### FINDING WHICH SOLVENT IS WATER AND WHICH IS COSOLVENT
                self.water_index = self.solvent_name.index('HOH')
                
                ## FINDING TOTAL SOLVENTS
                self.total_water = self.rdf.total_solvent[self.water_index]
                
                ## CREATING EMPTY LISTS FOR EACH SOLUTE
                self.pref_int_coeff = []
                self.pref_int_coeff_solute_atom = []
                self.equil_radius = []     
            
            ## IF COSOLVENT IS PRESENT
            if len(self.solvent_name) == 2:
                ## LOCATING COSOLVENT INFORMATION
                self.cosolvent_index = [index for index, each_solvent in enumerate(self.solvent_name) if index != self.water_index][0]
                self.total_cosolvent = self.rdf.total_solvent[self.cosolvent_index]
                
            #############################################
            ### PREFERENTIAL INTERACTION COEFFICIENTS ###
            #############################################
           
            ### LOOPING THROUGH EACH SOLUTE
            for solute_index in range(len(self.solute_name)):
                
                ### FINDING CUTOFF RADIUS
                ## INDEX OF EQUILIBRATION BASED ON WATER
                self.equil_radius.append(self.calc_cutoff_radius( g_r =  self.rdf.rdf_g_r[solute_index][self.water_index],
                                                               r =  self.rdf.rdf_r[solute_index][self.water_index]
                                                              ))
                
                ## RUNNING PREFERENTIAL INTERACTION COEFFICIENT ONLY IF COSOLVENT IS PRESENT
                if len(self.solvent_name) == 2:
                    
                    ### DEFINING INPUT VARIABLES
                    self.input_static_vars = { 'solute_res_index'        : self.rdf.solute_res_index[solute_index],                        # Residue index for the solute
                                          'water_res_index'         : self.rdf.solvent_res_index[self.water_index],     # Residue index for the water
                                          'cosolvent_res_index'     : self.rdf.solvent_res_index[self.cosolvent_index], # Residue index for the cosolvent
                                          'cutoff'                  : self.equil_radius[solute_index],                                # Cutoff radius for the RDF
                                          'total_water'             : self.total_water,                                 # Total number of water molecules
                                          'total_cosolvent'         : self.total_cosolvent,                             # Total number of cosolvents
                                         }
                    ### DEFINING SPLITTING VARIABLES
                    self.input_split_vars = { 'COM_Solute'   : self.rdf.solute_COM[solute_index],                          # Solute center of mass
                                              'COM_Solvent'  : self.rdf.solvent_COM[self.water_index],       # Solvent center of mass
                                              'COM_Cosolvent': self.rdf.solvent_COM[self.cosolvent_index]    # Cosolvent center of mass
                                            }
                    
                    ### CALCULATING PREFERENTIAL INTERACTION COEFFICIENT
                    output = calc_tools.split_traj_for_avg_std( 
                                                                 traj = traj,                               # Trajectory
                                                                 num_split = self.traj_num_split,           # Number of times to split trajectory
                                                                 input_function = self.calc_pref_int_coeff, # Function to calculate preferential interaction coefficient
                                                                 split_variables_dict = self.input_split_vars,   # Input variables that will need to be split (dictionary)
                                                                 **self.input_static_vars                        # Static variables (dictionary)
                                                                )
                    ### CALCULATING THE AVERAGE AND STANDARD DEVIATION AND APPENDING
                    self.pref_int_coeff.append( calc_tools.calc_avg_std(output) )
                    
                    ### CALCULATING OTHER QUANTITIES
                    if want_oxy_pref_int_coeff is True:
                        self.calc_pref_int_coeff_solute_atom(traj = traj, solute_index = solute_index)
                
            ##################################
            ### EXCESS COORDINATION NUMBER ###
            ##################################
            if 'excess_coord' in self.job_types and len(self.solvent_name) == 2:
                print("\n---WORKING ON EXCESS COORDINATION NUMBER---")
                ## RUNNING EXCESS COORDINATION NUMBER
                self.calc_excess_coord_num(traj)
        
        ### CLEARING SPACE
        if self.save_disk_space is True:
            self.rdf.solute_COM, self.rdf.solvent_COM, self.input_split_vars = [], [], []
        
    ### FUNCTION TO FIND THE CUTOFF RADIUS
    def calc_cutoff_radius(self, g_r, r):
        '''
        The purpose of this function is to get the cutoff radius based off a radial distribution function.
        INPUTS:
            g_r: [np.array] g(r) for radial distribution function
            r: [np.array] r for RDF, in nms
            solvent_name: [str] which solvent rdf do you want to base it on?
        OUTPUTS:
            equil_radius: [float] radius in nm of the cutoff
        '''
        ## GETTING EQUILIBRIUM POINT
        equil_index = calc_tools.find_equilibrium_point(g_r, self.tolerance)
        equil_radius = r[equil_index]
        return equil_radius
        
    ## FUNCTION TO FIND SOLUTE TO SOLVENT DISTANCES ACROSS THE TRAJECTORY
    def find_distances_btn_solute_solvent(self, traj, COM_Solute, COM_Solvent, COM_Cosolvent, solute_res_index, water_res_index, cosolvent_res_index, periodic=True):
        '''
        The purpose of this functions is to find the center of mass distances between solute to solvent, and solute to cosolvent.
        This algoritm copies the trajectory and recapitulates the positions of the first atoms of residues as the COM.
        INPUTS:
            traj: trajectory from md.traj
            COM_Solute: [np.array, shape=(time_frame, num_atoms, xyz_positions)] Center of mass of solute matrix
            COM_Solvent: [np.array, shape=(time_frame, num_atoms, xyz_positions)] Center of mass of solvent matrix
            COM_Cosolvent: [np.array, shape=(time_frame, num_atoms, xyz_positions)] Center of mass of cosolvent matrix
            solute_res_index: [list] list of all the solutes you are interested in
            water_res_index: [list] list of all the water residues you are interested in
            cosolvent_res_index: [list] list of all the cosolvent residues you are interested in
        OUTPUTS:
            dist_solute_solvent: [np.array] distances from solute to solvent COM
            dist_solute_cosolvent: [np.array] distances from solutes to cosolvent COM
        '''
        # Now, replacing the first atom of each residue as positions of COMs, then calculating distances
        copied_traj = traj[:]
        
        # Finding 1st atom of each residue
        solute_atom_1_index = [ traj.topology.residue(res_index).atom(0).index for res_index in solute_res_index ]
        water_atom_1_index = [ traj.topology.residue(res_index).atom(0).index for res_index in water_res_index ]
        cosolvent_atom_1_index = [ traj.topology.residue(res_index).atom(0).index for res_index in cosolvent_res_index ]
        
        # Editing trajectory according to the center of masses
        copied_traj.xyz[:, solute_atom_1_index] = COM_Solute[:]
        copied_traj.xyz[:, water_atom_1_index] = COM_Solvent[:]
        copied_traj.xyz[:, cosolvent_atom_1_index] = COM_Cosolvent[:]
        
        # Now, creating atom pairs between center of interest to solvent / cosolvent
        atom_pairs_solvent = [ [center_atom_indexes, interest_atom_indexes] for center_atom_indexes in solute_atom_1_index 
                          for interest_atom_indexes in water_atom_1_index ]
        atom_pairs_cosolvent = [ [center_atom_indexes, interest_atom_indexes] for center_atom_indexes in solute_atom_1_index 
                          for interest_atom_indexes in cosolvent_atom_1_index ]
        
        # Running md.distances to compute how far a solute is to a solvent
        dist_solute_solvent = md.compute_distances(traj=copied_traj, atom_pairs = atom_pairs_solvent, periodic = periodic)
        dist_solute_cosolvent = md.compute_distances(traj=copied_traj, atom_pairs = atom_pairs_cosolvent, periodic = periodic)
    
        return dist_solute_solvent, dist_solute_cosolvent
    
    ## FUNCTION TO CALCULATE PREFERENTIAL INTERACTION COEFFICIENT
    def calc_pref_int_coeff(self, traj, COM_Solute, COM_Solvent, COM_Cosolvent, solute_res_index, water_res_index, cosolvent_res_index, total_water, total_cosolvent, cutoff, ):
        '''
        The purpose of this function is to calculate the preferential interaction coefficient given distances between solute and solvent/cosolvents.
        The cutoff is typically when the distance at which the RDF goes to 1.
        INPUTS:
            traj: trajectory from md.traj
            COM_Solute: [np.array, shape=(time_frame, num_atoms, xyz_positions)] Center of mass of solute matrix
            COM_Solvent: [np.array, shape=(time_frame, num_atoms, xyz_positions)] Center of mass of solvent matrix
            COM_Cosolvent: [np.array, shape=(time_frame, num_atoms, xyz_positions)] Center of mass of cosolvent matrix
            solute_res_index: [list] list of all the solutes you are interested in
            water_res_index: [list] list of all the water residues you are interested in
            cosolvent_res_index: [list] list of all the cosolvent residues you are interested in
            total_water: [int] total number of water molecules
            total_cosolvent: [int] total number of cosolvent molecules
            cutoff: [float] cutoff radius in nm
        OUTPUTS:
            pref_int_dict: [dict] contains preferential interaction coefficient results as a dictionary
                'gamma_23': gamma_23 as we would expect from:
                    G_23(r,t) = n_3(r,t) - n_1(r,t) * (n_3 - n_3(r,t) / (n_1 - n_1(r,t)) )
                'ngamma_23': negative of gamma_23 (useful to have opposite conventions)
                'n_1_local_avg': average value of water around the local domain
                'n_1_bulk_avg': average value of water around the bulk domain
                'n_3_local_avg': average value of cosolvent in the local domain
                'n_3_bulk_avg': average value of cosolvent in the bulk domain
        '''        
        ### FINDING DISTANCE BETWEEN SOLUTE AND SOLVENT
        dist_solute_solvent, dist_solute_cosolvent = self.find_distances_btn_solute_solvent( traj = traj,
                                                                                             COM_Solute = COM_Solute,
                                                                                             COM_Solvent = COM_Solvent,
                                                                                             COM_Cosolvent = COM_Cosolvent,
                                                                                             solute_res_index = solute_res_index,
                                                                                             water_res_index = water_res_index,
                                                                                             cosolvent_res_index = cosolvent_res_index,
                                                                                             )
        ## FINDING LOCAL REGIONS
        n_1_local = np.sum(dist_solute_solvent < cutoff, axis=1).astype('float') # 1 x N frames
        n_3_local = np.sum(dist_solute_cosolvent < cutoff, axis=1).astype('float') # 1 x N frames
        ## FINDING BULK REGIONS
        n_1_bulk = (total_water - n_1_local).astype('float')
        n_3_bulk = (total_cosolvent - n_3_local).astype('float')
        ## CALCULATING GAMMA
        gamma_23 = np.mean( n_3_local - n_1_local * ( n_3_bulk / n_1_bulk  ) )
        #### OTHER RESULTS ####
        ## NEGATIVE GAMMA
        ngamma_23 = -gamma_23
        ## AVERAGE VALUES OF n_1, etc.
        n_1_local_avg = np.mean(n_1_local)
        n_1_bulk_avg = np.mean(n_1_bulk)
        n_3_local_avg = np.mean(n_3_local)
        n_3_bulk_avg = np.mean(n_3_bulk)
        ## DICTIONARY TO STORE ALL THE VALUES
        pref_int_dict = {
                'gamma_23': gamma_23,
                'ngamma_23': ngamma_23,
                'n_1_local_avg': n_1_local_avg,
                'n_1_bulk_avg': n_1_bulk_avg,
                'n_3_local_avg': n_3_local_avg,
                'n_3_bulk_avg': n_3_bulk_avg,
                }
        return pref_int_dict
        
    ### FUNCTION TO CALCULATE PREFERENTIAL INTERACTION COEFFICIENTS OF EACH OXYGEN OF THE SOLUTE
    def calc_pref_int_coeff_solute_atom(self, traj, solute_index):
        '''
        The purpose of this function is to calculate a preferential interaction coefficient based on a solute atom (e.g. all the oxygens, etc.)
        INPUTS:
            self: class property
            traj: trajectory from md.traj
            solute_index: [int]
                solute index that you are interested in, e.g. 0 
        OUTPUTS:
            self.solute_atom_cutoff_radius: [list] list of cutoff radii for the solute
            self.pref_int_coeff_solute_atom: [list] list of dictionaries:
                    dictionary of output data as a list
                    e.g. 'O1': {'gamma_23' ...}
        '''
        ## FINDING THE CUTOFF RADIUS FOR EACH OXYGEN
        self.solute_atom_cutoff_radius = [ self.calc_cutoff_radius( g_r =  self.rdf.rdf_oxy_g_r[solute_index][self.water_index][each_atom],
                                                       r =  self.rdf.rdf_oxy_r[solute_index][self.water_index][each_atom]
                                                      ) for each_atom in range(len(self.rdf.rdf_oxy_names[solute_index])) ]
        
        ## CREATING EMPTY DICTIONARY
        pref_int_coeff_solute_atom={}
            
        ## LOOPING THROUGH EACH ATOM
        for index, each_atom in enumerate(self.rdf.rdf_oxy_names[solute_index]):
            print("\nWORKING ON PREF INT FOR ATOM %s FOR SOLUTE %s"%(each_atom, self.solute_name[solute_index]))
            ## FINDING ATOM INDEX
            atom_index = calc_tools.find_atom_index(traj = traj,
                                                    resname = self.solute_name[solute_index],
                                                    atom_name = each_atom
                                                    )
            ## FINDING POSITION OF THAT ATOM THROUGHOUT THE TRAJECTORY
            atom_position = traj.xyz[:, atom_index]
            
            ## UPDATING THE INPUTS AND RUNNING PREFERENTIAL INTERACTION COEFFICIENTS AGAIN
            self.input_static_vars['cutoff'] = self.solute_atom_cutoff_radius[index]    ## CHANGING CUTOFF RADIUS BASED ON THE ATOM
            self.input_split_vars['COM_Solute'] = atom_position                         ## CHANGING THE SOLUTE POSITIONS TO ALL THE ATOMIC POSITIONS OF A SPECIFIC SOLUTE ATOM
            
            ### CALCULATING PREFERENTIAL INTERACTION COEFFICIENT
            output = calc_tools.split_traj_for_avg_std( 
                                                     traj = traj,                               # Trajectory
                                                     num_split = self.traj_num_split,           # Number of times to split trajectory
                                                     input_function = self.calc_pref_int_coeff, # Function to calculate preferential interaction coefficient
                                                     split_variables_dict = self.input_split_vars,   # Input variables that will need to be split (dictionary)
                                                     **self.input_static_vars                        # Static variables (dictionary)
                                                    )
            pref_int_coeff = calc_tools.calc_avg_std(output)
            ## STORING
            pref_int_coeff_solute_atom[each_atom] = pref_int_coeff
        
        
        ## STORING INTO LIST
        self.pref_int_coeff_solute_atom.append( pref_int_coeff_solute_atom )
            
        return
    
    ### FUNCTION TO CALCULATE EXCESS COORDINATION NUMBER
    def calc_excess_coord_num(self, traj):
        '''
        The purpose of this function is to calculate the excess coordination number. Excess coordination number is defined as:
            Nij = rho_j * Gij, where rho_ij is the number density of the solvent, Gij is the kirkwood-buff integral, and Nij is how much solvent is in excess in the vicinity with respect to the bulk
            Gij = integral from 0 to R ( RDF - 1) * 4*pi*r^2 dr
        This function will take advantage of the fact that we have calculated RDFs for this class, thus enabling us to quickly find excess coordination numbers
        INPUTS:
            self: class property
            traj: trajectory from md.traj
        OUTPUTS:
            self.excess_coord_num: [list] list of excess coordination numbers for each solvent
            self.excess_coord_num_oxy: [list] list of each oxygen's excess coordination number for each solvent
        ALGORITHM:
            - look for all possible rdfs
            - run excess coordination number for each rdf
            - store the excess coordination number
        '''
        ## CREATING EMPTY LIST
        self.excess_coord_num=[]; self.excess_coord_num_oxy = []
        ## LOOPING THROUGH EACH SOLUTE OF INTEREST
        for solute_index, solute_name in enumerate(self.solute_name):
            ## CREATING EMPTY LIST
            excess_coord_num_solute = []
            excess_coord_num_oxy_solute = []
            
            ## LOOPING THROUGH EACH SOLVENT SYSTEM
            for solvent_index, solvent_name in enumerate(self.rdf.solvent_name):
                print("WORKING ON EXCESS COORDINATION FOR SOLUTE-SOLVENT: %s-%s"%(solute_name, solvent_name))
                ## DEFINING R, G(R) FOR RADIAL DISTRIBUTION FUNCTION
                r = self.rdf.rdf_r[solute_index][solvent_index]
                g_r = self.rdf.rdf_g_r[solute_index][solvent_index]
                ## FINDING EXCESS COORDINATION NUMBER
                excess_coord_num, _ = calc_excess_coord_num(traj, r = r, g_r = g_r, bin_width=self.bin_width, residue_of_interest=solvent_name )
                ## STORING EXCESS COORD NUM
                excess_coord_num_solute.append( excess_coord_num )

                ## CHECKING IF YOU HAVE OXYGEN INTERACTION COEFFICINTS
                if self.want_oxy_pref_int_coeff is True:
                    ## CREATING EMPTY TEMPORARY LIST
                    excess_coord_num_oxy = []
                    ## RUNNING ANALYSIS FOR EACH OXYGEN
                    for oxygen_index, oxygen_name in enumerate(self.rdf.rdf_oxy_names):
                        print("WORKING ON EXCESS COORDINATION FOR ATOM %s FOR SOLUTE %s"%(oxygen_name, self.solute_name))
                        ## DEFINING R, G(R) FOR RADIAL DISTRIBUTION FUNCTION
                        r = self.rdf.rdf_oxy_r[solute_index][solvent_index][oxygen_index]
                        g_r = self.rdf.rdf_oxy_g_r[solute_index][solvent_index][oxygen_index]
                        ## FINDING EXCESS COORDINATION NUMBER
                        excess_coord_num, _ = calc_excess_coord_num(traj, r = r, g_r = g_r, bin_width=self.bin_width, residue_of_interest=solvent_name )
                        ## STORING EACH OXYGEN COORDINATION NUMBER
                        excess_coord_num_oxy.append(excess_coord_num)
                    ## STORING TO SELF
                    excess_coord_num_oxy_solute.append(excess_coord_num_oxy)
            ## STORING FOR EACH SOLUTE
            self.excess_coord_num.append(excess_coord_num_solute)
            self.excess_coord_num_oxy.append(excess_coord_num_oxy_solute)
            
#%% MAIN SCRIPT
if __name__ == "__main__":
    
    ### DIRECTORY TO WORK ON
    analysis_dir=r"181107-PDO_DEHYDRATION_FULL_DATA_300NS" # Analysis directory
    # analysis_dir=r"180316-ACE_PRO_DIO_DMSO"
    specific_dir="PDO\\mdRun_433.15_6_nm_PDO_10_WtPercWater_spce_dioxane" # Directory within analysis_dir r"mdRun_363.15_6_nm_tBuOH_100_WtPercWater_spce_Pure"
    # specific_dir="ACE/mdRun_433.15_6_nm_ACE_10_WtPercWater_spce_dioxane" # Directory within analysis_dir r"mdRun_363.15_6_nm_tBuOH_100_WtPercWater_spce_Pure"
    # specific_dir=r"Planar_310.15_ROT_TMMA_10x10_CHARMM36_withGOLP" # Directory within analysis_dir
    path2AnalysisDir=r"R:\scratch\SideProjectHuber\Analysis\\" + analysis_dir + '\\' + specific_dir # PC Side
    
    ### DEFINING FILE NAMES
    gro_file=r"mixed_solv_prod.gro" # Structural file
    xtc_file=r"mixed_solv_prod_10_ns_whole_290000.xtc" # r"mixed_solv_prod_last_90_ns_center_rot_trans_center_prog_rot_trans_center.xtc" # Trajectory
    # xtc_file=r"mixed_solv_last_50_ns_whole.xtc"
    
    ### LOADING TRAJECTORY
    traj_data = import_tools.import_traj( directory = path2AnalysisDir, # Directory to analysis
                 structure_file = gro_file, # structure file
                  xtc_file = xtc_file, # trajectories
                  )
    
    #%%
    
    ### DEFINING INPUT DATA
    input_details={
                    'solute_name'               : ['PDO'],                    # Solute of interest
                    'solvent_name'              : ['HOH', 'DIO', 'GVLL'],   # Solvents you want radial distribution functions for
                    'bin_width'                 : 0.02,                     # Bin width of the radial distribution function
                    'cutoff_radius'             : 2.00,                     # Radius of cutoff for the RDF (OPTIONAL)
                    'want_oxy_pref_int_coeff'   : True,                     # True if you want preferential interaction coefficient of just the oxygens
                    'traj_num_split'            : 2,                        # Number of times to split the trajectory for averaging and standard deviation
                    'job_type'                  : ['pref_int', 'excess_coord']   # job types that you want
                    }

    ## RUNNING CLASS
    pref_int = calc_pref_interaction_coefficients(traj_data, **input_details)
    
    
    