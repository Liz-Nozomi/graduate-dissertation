"""
sam_analysis.py
this is the main script for SAM analysis

CREATED ON: 02/18/2020

AUTHOR(S):
    Bradley C. Dallin (brad.dallin@gmail.com)
    
** UPDATES **

TODO:

"""
##############################################################################
## IMPORTING MODULES
##############################################################################
import os, sys
import pickle
import numpy as np
import mdtraj as md

## MDBUILDERS FUNCTIONS
from MDBuilder.core.check_tools import check_server_path

## IMPORTING COMMANDS 
from MDDescriptors.traj_tools.trjconv_commands import convert_with_trjconv
from MDDescriptors.core.read_write_tools import extract_itp     # Loading itp reading tool

##############################################################################
## FUNCTIONS AND CLASSES
##############################################################################
### FUNCTION TO CHECK IF YOU ARE RUNNING ON SPYDER
def check_spyder():
    ''' This function checks if you are running on spyder '''
    if any('SPYDER' in name for name in os.environ):
        return True
    else:
        return False

## FUNCTION TO FIND ITP FILES IN TARGET DIRECTORY
def find_itp_files( directory = '.', extension = 'itp' ):
    r'''
    '''
    itp_files = []
    extension = extension.lower()
    
    for dirpath, dirnames, files in os.walk( directory ):
        for name in files:
            if extension in name:
                itp_files.append( os.path.join( dirpath, name ) )
    
    return itp_files

## FUNCTION TO ADD BONDS TO TRAJECTORY FROM ITP DATA
def add_bonds_to_traj( traj, itp_data = [] ):
    r'''
    Function takes data from itp files to add bonds to trajectory.topology object
    '''
    if not itp_data:
        print( "no itp data loaded! Load itp data." )
    else:
        for itp in itp_data:
            ## DOUBLE CHECK THAT MDTRAJ LOADS IN CORRECT RESIDUE
            res_atoms = [ [ atom for atom in residue.atoms ] for residue in traj.topology.residues if residue.name == itp.residue_name ]
            for res in res_atoms:
                for a1, a2 in itp.bonds:
                    traj.topology.add_bond( res[a1-1], res[a2-1] )

    return traj

## FUNCTION TO ADD BONDS TO WATER MOLECULES
def add_water_bonds( traj ):
    r'''
    Function to add water bonds to traj
    '''
    ## GATHER WATER GROUPS
    res_atoms = [ [ atom for atom in residue.atoms ] for residue in traj.topology.residues if residue.name == "SOL" ]
    ## LOOP THROUGH WATER GROUPS
    for res in res_atoms:
        h1 = 0
        ## LOOP THROUGH MOLECULE
        for atom in res:
            if atom.element.symbol == "O":
                oxygen = atom
            elif atom.element.symbol == "H" and h1 < 1:
                hydrogen1 = atom
                h1 = 1
            elif atom.element.symbol == "H" and h1 > 0:
                hydrogen2 = atom
        ## BIND WATER
        traj.topology.add_bond( oxygen, hydrogen1 )
        traj.topology.add_bond( oxygen, hydrogen2 )
            
    return traj

## FUNCTION TO GATHER BOND INFO FROM ITP FILES
def gather_itp_data( path_data ):
    r'''
    Function reads itp files and loads itp data
    '''        
    ## ITP FILES IN WD/LOOKS FOR THOSE CREATED BY MDBuilder
    itp_files = []
    all_itp_files = find_itp_files( directory = path_data, extension = 'itp' )
    for itp in all_itp_files:
        with open( itp ) as f:
            line = f.readline()
            if "MDBuilder" in line:
                itp_files.append(itp)
    del all_itp_files
    
    ## EXTRACT ITP DATA
    itp_data = [ extract_itp( itp_file ) for itp_file in itp_files if itp_file != '' ]
    
    ## RETURN ITP DATA
    return itp_data    

## FUNCTION TO LOAD TRAJECTORY
def load_md_traj( path_traj,
                  input_prefix,
                  make_whole = True,
                  add_bonds_from_itp = False,
                  add_water_bonds_to_traj = True,
                  remove_dummy_atoms = True,
                  want_rewrite = False,
                  standard_names = False,
                  **kwargs ):
    r'''
    Function to load md trajectory (gromacs only)
    ''' 
    ## CONVERTING TRAJECTORY
    trjconv_conversion = convert_with_trjconv( wd = path_traj )
    
    ## LOAD FULL TRAJECTORY
    full_pdb, full_xtc = trjconv_conversion.generate_pdb_from_gro( input_prefix, make_whole = make_whole, rewrite = want_rewrite )
    traj = md.load( full_xtc,
                    top = full_pdb,
                    standard_names = standard_names )
    
    if add_bonds_from_itp is True:
        print( " --- ADDING BONDS FROM ITP DATA ---" )
        ## GATHER ITP FILE DATA
        itp_data = gather_itp_data( path_traj )
        
        ## UPDATE TRAJ WITH ITP DATA/ADDS BONDS
        traj = add_bonds_to_traj( traj, itp_data )

    if add_water_bonds_to_traj is True:
        print( " --- ADDING BONDS TO WATER MOLECULES ---" )
        traj = add_water_bonds( traj )

    if remove_dummy_atoms is True:
        print( " --- REMOVING DUMMY ATOMS FROM TRAJECTORY ---" )
        ## IDENTIFY DUMMY ATOM INDICES
        atom_indices = np.array([ atom.index for atom in traj.topology.atoms
                                  if atom.element.symbol != "VS" ])

        ## REDUCE TRAJECTORY BY REMOVING DUMMY ATOMS (EX. VIRTUAL SITE OF TIP4P)
        traj.atom_slice( atom_indices, inplace = True )
    
    ## PRINT TRAJ INFO
    traj_info( traj )
    
    return traj

## FUNCTION TO PRINT TRAJ INFO
def traj_info( traj ):
    r'''
    '''
    print( "\n---  TRAJECTORY INFO ---" )
    print( "%-12s %6d" % ( "# Frames:", traj.n_frames ) )
    list_of_residues = set( [ residue.name for residue in traj.topology.residues ] )
    print( "%-12s %6d" % ( "# Residues:", len(list_of_residues) ) )
    print( "%-12s %6d" % ( "# Atoms:", len( list(traj.topology.atoms) ) ) )
    for residue in list_of_residues:
        n_residue = len([ ii for ii in traj.topology.residues if ii.name == residue ])
        print( "%-12s %6d" % ( "# " + residue + ":", n_residue ) )

#%%
##############################################################################
## MAIN SCRIPT
##############################################################################
if __name__ == "__main__":
    ## GROMACS OUTPUT
    input_prefix = "sam_prod"
    
    ## MAIN DIRECTORY
    main_dir = r"R:\simulations\polar_sams\unbiased\sample2"
    
    ## TEST DIRECTORY
    test_directory = "sam_single_12x12_300K_dodecanethiol_tip4p_nvt_CHARMM36"

    ## LOAD TRAJECTORY
    path_traj = check_server_path( os.path.join( main_dir, test_directory ) )
    traj =  load_md_traj( path_traj,
                          input_prefix,
                          make_whole = True,
                          add_bonds_from_itp = True,
                          add_water_bonds_to_traj = True,
                          remove_dummy_atoms = True,
                          want_rewrite = False,
                          standard_names = False )
    

            