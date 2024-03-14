# -*- coding: utf-8 -*-
"""
h_bond_debug.py
The purpose of this script is to debug the hydrogen bond parameters script

CREATED ON: 01/16/2020

AUTHOR(S):
    Bradley C. Dallin (brad.dallin@gmail.com)
    
** UPDATES **

TODO:
"""
## IMPORT OS
import os
## IMPORT NUMPY
import numpy as np  # Used to do math functions
## MDBUILDERS FUNCTIONS
from MDBuilder.core.check_tools import check_server_path
## IMPORT TRAJECTORY FUNCTION
from MDDescriptors.application.sams.trajectory import load_md_traj
## IMPORT WILLARD-CHANDER HEIGHT FUNCTION
from MDDescriptors.application.sams.willard_chandler import compute_willad_chandler_height
## IMPORT HBOND FUNCTIONS
from MDDescriptors.application.sams.h_bond import compute_hbond_triplets
## FUNCTION TO SAVE AND LOAD PICKLE FILES
from MDDescriptors.application.sams.pickle_functions import load_pkl, save_pkl
## IMPORTING TRACKING TIME
from MDDescriptors.core.track_time import track_time

##############################################################################
# HBOND FUNCTIONS
##############################################################################
### FUNCTION TO COMPUTE HBOND TRIPLETS
def compute_num_hbonds( sim_working_dir = "None",
                        input_prefix    = "None",
                        rewrite         = False,
                        **kwargs ):
    r'''
    Function loads gromacs trajectory and computes triplet angle distribution
    '''
    if sim_working_dir  == "None" \
        or input_prefix == "None":
        print( "ERROR: missing inputs. Check inputs and try again" )
        return 1
    else:
        path_pkl = os.path.join( sim_working_dir, r"output_files", input_prefix + "_num_hbonds_{:s}.pkl" )
        path_pkl_triplets = os.path.join( sim_working_dir, r"output_files", input_prefix + "_hbond_triplets.pkl" )
        if rewrite is not True and os.path.exists( path_pkl_triplets ):
            ## LOAD MD TRAJECTORY
            traj = load_md_traj( path_traj    = sim_working_dir,
                                 input_prefix = input_prefix,
                                 **kwargs )
            ## LOAD TRIPLET DATA
            triplets = load_pkl( path_pkl_triplets )
        else:
            ## PREPARE DIRECTORY FOR ANALYSIS
            path_to_sim = check_server_path( sim_working_dir )
            ## LOAD MD TRAJECTORY
            traj = load_md_traj( path_traj    = path_to_sim,
                                 input_prefix = input_prefix,
                                 **kwargs )
            ## GET HBOND TRIPLETS
            triplets = compute_hbond_triplets( sim_working_dir = path_to_sim,
                                               input_prefix    = input_prefix,
                                               rewrite         = rewrite,
                                               **kwargs )
        ## COMPUTE NUM HBONDS
        hbonds = num_hbonds( traj = traj,
                             triplet_list = triplets,
                             **kwargs )
        ## SAVE PICKLE FILE
        for key, data in hbonds.items():
            save_pkl( data, path_pkl.format( key ) )

## FUNCTION TO COMPUTE HBOND TYPES
def num_hbonds( traj,
                triplet_list,
                hbond_types = [ "sam-sam", 
                                "sam-water", 
                                "water-water" ],
                water_residues = [ "SOL", "HOH" ],
                print_freq = 100,
                **kwargs ):
    R'''
    Computes hydrogen bonding states given triplets and molecule indices, outputs
    from compute_hbonds function
    '''
    ## TRACKING TIME
    timer = track_time()
    ## LISTS OF WATER AND SAM INDICES
    atoms = [ atom for atom in traj.topology.atoms if atom.element.symbol in set(( "N", "O" )) ]
    water_indices = np.array([ atom.index for atom in atoms if atom.residue.name in water_residues ])
#    head_group_indices = np.array([ atom.index for atom in atoms if atom.residue.name not in water_residues + [ "MET", "CL" ] ])        
    ## SET UP TOTAL COUNTERS
    n_hbonds = 0.
    n_hbonds_sam = 0.
    n_hbonds_water = 0.
    n_hbonds_sam_sam = 0.
    n_hbonds_sam_water = 0.
    n_hbonds_water_water = 0.
    ## CREATE EMPTY ARRAYS
    N = np.arange( 0, 10, 1 )
    histo_all = np.zeros( 10 )
    histo_sam = np.zeros( 10 )
    histo_water = np.zeros( 10 )
    histo_sam_sam = np.zeros( 10 )
    histo_sam_water = np.zeros( 10 )
    histo_water_water = np.zeros( 10 )
    ## LOOP THROUGH TRIPLET LIST FRAME BY FRAME
    for ii in range(traj.n_frames):
        ## PRINTING 
        if ii % print_freq == 0:
            print("====> Working on frame %d"%(ii))            
        ## GET CURRENT TRIPLETS
        target_atoms = triplet_list[ii][0]
        triplets = triplet_list[ii][1]
        ## SET UP COUNTER
        hbond = {}
        for hb in hbond_types:
            hbond[hb] = np.empty( shape = ( 0, 2 ) )            
        ## LOOP THROUGH TRIPLETS TO SEPARATE INTO HBOND TYPES
        for donor, acceptor in triplets[:,[0,2]]:
            ## DETERMINE IF DONOR OR ACCEPTOR IN BOX
            if donor in target_atoms or acceptor in target_atoms:
                ## ASSIGN DONOR TYPE
                if traj.topology.atom(donor).residue.name not in water_residues:
                    donor_type = "sam"
                else:
                    donor_type = "water"
                ## ASSIGN ACCEPTOR TYPE
                if traj.topology.atom(acceptor).residue.name not in water_residues:
                    acceptor_type = "sam"
                else:
                    acceptor_type = "water"
                ## COMBINE STRING
                hb_type = donor_type + '-' + acceptor_type
                ## WATER-SAM IS EQUIVALENT TO SAM-WATER
                if hb_type == "water-sam":
                    hb_type = "sam-water"
                ## APPEND TO DICTIONARY
                hbond[hb_type] = np.vstack(( hbond[hb_type], np.array([ donor, acceptor ]) ))        
        ## LOOP THROUGH TARGET ATOMS TO COUNT HBONDS
        n_waters = 0.
        n_head_groups = 0.
        for residue in traj.topology.residues:
            if residue.name not in water_residues + [ "MET", "CL" ]:
                residue_atoms = [ atom.element.symbol for atom in residue.atoms ]
                if "N" in residue_atoms or "O" in residue_atoms:
                    n_head_groups += 1.
        # TO DO: ADD MORE GENERAL WAY TO GET N HEAD GROUPS. NOW IT IS ASSUMED 
        # EACH LIGAND AS A HEAD GROUP AND IS COUNTED
        tmp_all = 0.
        tmp_sam = 0.
        tmp_water = 0.
        tmp_sam_sam = 0.
        tmp_sam_water = 0.
        tmp_water_water = 0.
        for ta in target_atoms:
            ## COUNT N_WATER
            if ta in water_indices:
                n_waters += 1.
            ## COUNT N_HEAD_GROUPS
#            if ta in head_group_indices:
#                n_head_groups += 1.            
            ## SAM-SAM DONORS AND ACCEPTORS
            n_donors_sam_sam = np.sum( hbond['sam-sam'][:,0] == ta )
            n_acceptors_sam_sam = np.sum( hbond['sam-sam'][:,1] == ta )
            sam_sam = n_donors_sam_sam + n_acceptors_sam_sam
            tmp_sam_sam += sam_sam
            histo_sam_sam[sam_sam] += 1.
            ## SAM-WATER DONORS AND ACCEPTORS
            n_donors_sam_water = np.sum( hbond['sam-water'][:,0] == ta )
            n_acceptors_sam_water = np.sum( hbond['sam-water'][:,1] == ta )
            sam_water = n_donors_sam_water + n_acceptors_sam_water
            tmp_sam_water += sam_water
            histo_sam_water[sam_water] += 1.
            ## WATER-WATER DONORS AND ACCEPTORS
            n_donors_water_water = np.sum( hbond['water-water'][:,0] == ta )
            n_acceptors_water_water = np.sum( hbond['water-water'][:,1] == ta )
            water_water = n_donors_water_water + n_acceptors_water_water
            tmp_water_water += water_water
            histo_water_water[water_water] += 1.
            ## ALL DONORS AND ACCEPTORS
            n_donors = n_donors_sam_sam + n_donors_sam_water + n_donors_water_water
            n_acceptors = n_acceptors_sam_sam + n_acceptors_sam_water + n_acceptors_water_water
            n_all = n_donors + n_acceptors
            tmp_all += n_all
            histo_all[n_all] += 1.
            ## ALL SAM DONORS AND ACCEPTORS
            n_donors_sam = n_donors_sam_sam + n_donors_sam_water
            n_acceptors_sam = n_acceptors_sam_sam + n_acceptors_sam_water
            sam = n_donors_sam + n_acceptors_sam
            tmp_sam += sam
            histo_sam[sam] += 1.
            ## ALL WATER DONORS AND ACCEPTORS
            n_donors_water = n_donors_sam_water + n_donors_water_water
            n_acceptors_water = n_acceptors_sam_water + n_acceptors_water_water
            water = n_donors_water + n_acceptors_water
            tmp_water += water
            histo_water[water] += 1.
        ## NORMALIZE BY NUMBER OF HBONDING MOLECULES
#        n_hbonds += tmp_all / np.max([ n_waters + n_head_groups, 1 ]) # avoids division by zero error
        n_hbonds += tmp_all / np.max([ n_waters, 1 ]) # avoids division by zero error
        n_hbonds_sam += tmp_sam / np.max([ n_head_groups, 1 ]) 
        n_hbonds_water += tmp_water / np.max([ n_waters, 1 ])
        n_hbonds_sam_sam += tmp_sam_sam / np.max([ n_head_groups, 1 ]) 
#        n_hbonds_sam_water += tmp_sam_water / np.max([ n_head_groups, 1 ]) 
        n_hbonds_sam_water += tmp_sam_water / np.max([ n_waters, 1 ]) 
        n_hbonds_water_water += tmp_water_water / np.max([ n_waters, 1 ])
    ## NORMALIZE BY NUMBER OF FRAMES
    n_hbonds /= traj.n_frames
    n_hbonds_sam /= traj.n_frames
    n_hbonds_water /= traj.n_frames
    n_hbonds_sam_sam /= traj.n_frames
    n_hbonds_sam_water /= traj.n_frames
    n_hbonds_water_water /= traj.n_frames
    ## NORMALIZE HISTOGRAMS
    histo_all = histo_all / np.trapz( histo_all, x = N )
    histo_all = np.array([ N, histo_all ]).transpose()
    histo_sam = histo_sam / np.trapz( histo_sam, x = N )
    histo_sam = np.array([ N, histo_sam ]).transpose()
    histo_water = histo_water / np.trapz( histo_water, x = N )
    histo_water = np.array([ N, histo_water ]).transpose()
    histo_sam_sam = histo_sam_sam / np.trapz( histo_sam_sam, x = N )
    histo_sam_sam = np.array([ N, histo_sam_sam ]).transpose()
    histo_sam_water = histo_sam_water / np.trapz( histo_sam_water, x = N )
    histo_sam_water = np.array([ N, histo_sam_water ]).transpose()
    histo_water_water = histo_water_water / np.trapz( histo_water_water, x = N )
    histo_water_water = np.array([ N, histo_water_water ]).transpose()
    ## OUTPUTTING TIME
    timer.time_elasped()
    ## RETURN RESULTS
    return { "all"               : n_hbonds,
             "sam"               : n_hbonds_sam,
             "water"             : n_hbonds_water,
             "sam-sam"           : n_hbonds_sam_sam,
             "sam-water"         : n_hbonds_sam_water,
             "water-water"       : n_hbonds_water_water,
             "histo_all"         : histo_all,
             "histo_sam"         : histo_sam,
             "histo_water"       : histo_water,
             "histo_sam-sam"     : histo_sam_sam,
             "histo_sam-water"   : histo_sam_water,
             "histo_water-water" : histo_water_water }
