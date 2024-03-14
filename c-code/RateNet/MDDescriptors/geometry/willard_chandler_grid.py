#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
willard_chandler_grid.py
This script generates a willard chandler grid. 

FUNCTIONS:
    willard_chandler: 
        class function that computes willard-chandler surfaces
    load_datafile: 
        loads database
    compute_compatible_mesh_params: 
        computes compatible mesh parameters
    generate_grid_in_box: 
        generates grid in the box
    density_map:
        function to generate density maps
    wc_interface:
        computes willard chandler interface
"""

## IMPORTING FUNCTIONS
from datetime import datetime
import os
import numpy as np

## IMPORTING PARALLEL
from MDDescriptors.core.parallel import parallel

## IMPORTING SCIPY FUNCTIONS
from scipy.stats import gaussian_kde
from scipy.spatial import cKDTree
import platform

## DEFINING MARTCHING CUBES
from skimage import measure
try:
    marching_cubes = measure.marching_cubes
except AttributeError:
    marching_cubes =  measure.marching_cubes_lewiner

try:
    import matplotlib.pyplot as plt
    import scipy.interpolate as interpolate
    _PYPLOT_AVAILABLE = True
except ImportError:
    plt.plot = None
    msg = ( 'plt.plot not available (requires display), '
            'will only generate image files' )
    _PYPLOT_AVAILABLE = False

## DEFINING WILLARD CHANDLER VARIABLES
WC_DEFAULTS={
        'alpha' : 0.24, # Describes the width. This is hte same value used in the WC paper.
        'contour': 25.6, # 80% of the bulk water density. 
                         #    contour = 16. # 0.5 bulk <-- used in the WC paper
                         #    contour = 20.8 # 0.65 bulk
        'mesh': 0.1, # Grid mesh
        }

##############################################################################
# willard_chandler class
##############################################################################
class willard_chandler:
    R'''
    This class function computes willard-chandler surfaces given the trajectory. 
    Good reference for WC surfaces using MDAnalysis:
        https://marcello-sega.github.io/pytim/WillardChandler.html
    INPUTS:
        alpha: [float]
            xi value used to describe the width Gaussian functions
        mesh: [float]
            mesh size in nanometers
    '''
    def __init__( self, alpha = 0.24, mesh = 0.1 ):
        ### PRINTING
        print("**** CLASS: %s ****"%(self.__class__.__name__)) 
        
        if not _PYPLOT_AVAILABLE:
            print( 'Display not available, figures will be generated'
                   'as image files only' )
        ## STORING DEFAULT VARIABLES
        self.alpha = alpha
        self.mesh = mesh
        
    def compute( self, traj ):
        R'''
        
        INPUT:
            traj: [mdtraj.trajectory] 
                trajectory imported from mdtraj
            periodic: [bool] 
                include pbc (default is True)
        '''
        print("--- Calculating density field for %s simulations windows ---" % (str(traj.time.size)) )
        
        n_frames = traj.time.size
        avg_df, avg_spacing = density_field( traj, 0, alpha = self.alpha, mesh = self.mesh )
        
        ## LOOPING THROUGH FRAMES
        for frame in range(n_frames-1):
            df, spacing = density_field( traj, frame+1, alpha = self.alpha, mesh = self.mesh )
            avg_df += df
            avg_spacing += spacing

        avg_df /= n_frames
        avg_spacing /= n_frames

        return avg_df

## GENERATING COMPATIBLE MESH PARAMETETRS
def compute_compatible_mesh_params(mesh, box):
    """ 
    Given a target mesh size and a box, return the number of grid elements
    and spacing in each direction, which are commensurate with the box
    INPUTS:
        mesh: [float]
            mesh size in nm
    OUTPUTS:
        box: [list]
            list of box array sizes in nm        
    """
    n = np.array([np.ceil(b / mesh) for b in box])
    d = box / n
    return n, d

def generate_grid_in_box(box, npoints, order='zxy'):
    """ 
    Generate an homogenous grid of npoints^3 points that spans the
    complete box.
    INPUTS:
        box: [list]
            the simulation box edges
        npoints: [list]
            the number of points along each direction
    OUTPUTS:
        grid: [np.array]
            grid that has been resizes to fit the box. 
    """
    xyz = []
    for i in range(3):
        xyz.append(np.linspace(0., box[i] - box[i] / npoints[i], npoints[i]))
        
    if order == 'zyx':
        z, y, x = np.meshgrid(xyz[0], xyz[1], xyz[2], indexing='ij')
    else:
        x, y, z = np.meshgrid(xyz[2], xyz[1], xyz[0], indexing='ij')

    grid = np.append(x.reshape(-1, 1), y.reshape(-1, 1), axis=1)
    grid = np.append(grid, z.reshape(-1, 1), axis=1)
    return grid.T

### FUNCTION TO GENERATE DENSITY MAPS
def density_map(pos, grid, sigma, box):
    '''
    This function generates density maps. The bw_method is the type of method 
    used as a parameter in the gaussian_kde function. 
    INPUTS:
        pos: [np,array, shape=(N,3)]
            positions
        grid: DEPRECIATED INPUT
        sigma: [float]
            sigma vlaues
        box: [list]
            the simulation box edges
    OUTPUTS:
        kernel: 
            kernel outputs
        values.std(ddof=1):
            values with a standrad deviation of 1
    '''
    values = np.vstack([pos[::, 0], pos[::, 1], pos[::, 2]])
    kernel = gaussian_kde_pbc(values, bw_method=sigma / values.std(ddof=1))
    kernel.box = box
    kernel.sigma = sigma
    return kernel, values.std(ddof=1)

class gaussian_kde_pbc(gaussian_kde):
    '''
    These functions evaluate PBC fast. See reference:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
    '''
    # note that here "points" are those on the grid

    def evaluate_pbc_fast(self, points):
        grid = points
        pos = self.pos
        box = self.box
        d = self.sigma * 2.5
        results = np.zeros(grid.shape[1], dtype=float)
        gridT = grid[::-1].T[:]
        tree = cKDTree(gridT, boxsize=box)
        # the indices of grid elements within distane d from each of the pos
        scale = 2. * self.sigma**2
        indlist = tree.query_ball_point(pos, d)
        for n, ind in enumerate(indlist):
            dr = gridT[ind, :] - pos[n]
            cond = np.where(dr > box / 2.)
            dr[cond] -= box[cond[1]]
            cond = np.where(dr < -box / 2.)
            dr[cond] += box[cond[1]]
            dens = np.exp(-np.sum(dr * dr, axis=1) / scale)
            results[ind] += dens

        return results
   
### FUNCTION TO GENERATE DENSITY FIELD
def density_field(traj, 
                  frame, 
                  alpha = 0.24, 
                  mesh = 0.1,
                  residue_list = ['HOH', 'MET']):
    R'''
    This function creates a density field given the trajectory.
    INPUTS:
        traj: [obj]
            trajectory object
        alpha: [float]
            alpha balue used in WC interface
        mesh: [float]
            mesh increment value
        residue_list: [list]
            list of resiudes to check for density field, default is water and methanol
    OUTPUTS:
        volume: 
            volume
        spacing:
            spacing
    '''
    box = traj.unitcell_lengths[ frame, : ]
    ndx_solv = np.array( [ [ atom.index for atom in residue.atoms if 'H' not in atom.name ] for residue in traj.topology.residues if residue.name in residue_list ] ).flatten()
    pos = traj.xyz[ frame, ndx_solv, : ]
    
    ## FINDING COMPATIBLE MESH PARAMETERS
    ngrid, spacing = compute_compatible_mesh_params( mesh, box )
    grid = generate_grid_in_box( box, ngrid, order = 'xyz' )
    kernel, _ = density_map( pos, grid, alpha, box )
    kernel.pos = pos.copy()
    density_field = ( 2 * np.pi * alpha**2 )**(-1.5) * kernel.evaluate_pbc_fast(grid)
    
    # Thomas Lewiner, Helio Lopes, Antonio Wilson Vieira and Geovan
    # Tavares. l. Journal of Graphics Tools 8(2) pp. 1-15
    # (december 2003). DOI: 10.1080/10867651.2003.10487582
    volume = density_field.reshape( tuple( np.array( ngrid[::-1] ).astype(int) ) )    
            
    return volume, spacing

### FUNCTION TO COMPUTE THE WILLARD-HANDLER INTERFACE
def wc_interface( density_field, spacing, contour = 16. ):
    R'''
    Returns (z,y,x)
    '''
    verts, faces, normals, values = marching_cubes( density_field, level = contour, spacing = tuple( spacing ) )
    # note that len(normals) == len(verts): they are normals
    # at the vertices, and not normals of the faces
    triangulated_surface = [verts, faces, normals]
    
    return triangulated_surface[0]





##############################################################################

### FUNCTION TO LOAD WC DATAFILE
def load_datafile(path_to_file):
    '''
    The purpose of this function is to load the WC data file
    INPUTS:
        path_to_file: [str]
            path to the data file
    OUTPUTS:
        data: [np.array, shape = (n_points, (x,y,z,value))]
            data from the wc interface
    '''
    ## OPENING DATA FILE
    with open( path_to_file ) as raw_data:
        data = raw_data.readlines()
    
    ## TRANSFORMING DATA TO NUMPY ARRAY
    data = np.array( [ [ float(el) for el in line.split(',') ] for line in data[2:] ] )
    
    return data


### FUNCTION TO CREATE WC GRID
def create_grid(traj, 
                out_path, 
                wcdatfilename, 
                wcpdbfilename, 
                alpha = 0.24, 
                mesh = 0.2, 
                contour = 16., 
                write_pdb = True, 
                n_procs = 28,
                want_rewrite = False,
                last_frame = 1000,
                verbose = False):
    r"""
    This script creates the grid used for willard chandler interface. 
    INPUTS:
        traj: [traj object]
            trajectory information
        out_path: [str]
            path to output
        wcdatfilename: [str]
            willard chandler data file name, which stores the xyz positions of the interface
        wcpdbfilename: [str]
            willard chandler PDB file, visualized with PDB, etc. on VMD.
        alpha: [float]
            alpha value for WC interface
        contour: [float]
            contour value for WC interface, default=16 for half the bulk number density of water
        write_pdb: [logical]
            True if you want to write a PDB file
        n_procs: [int]
            number of processors desired
        want_rewrite: [logical]
            True if you want to rewrite
        last_frame: [int]
            Last frame to use
        verbose: [logical]
            True if you want to print out details
    OUTPUTS:
        data: [np.array, shape=(N,3)]
            data for the WC interface contained in x, y, z positions in space
    """
    ## DEFINING PATH TO WC FILE
    path_wc_dat_file = os.path.join(out_path, wcdatfilename)
    path_pdb_file = os.path.join( out_path, wcpdbfilename)
    ## SEEING IF YOU WANT TO RUN THE WC GRIDDING
    if os.path.isfile(path_wc_dat_file) is False or want_rewrite is True:
        ## FINDING THE LAST 1000 FRAMES
        end = 0
        if traj.time.size > last_frame:
            end = traj.time.size - last_frame
            
        ## DEFINING INPUTS FOR WILLARD-CHANDLER INTERFACE
        kwargs = { 'alpha': alpha, 'mesh': mesh  }
        
        _, avg_spacing = density_field( traj, 0, alpha = alpha, mesh = mesh )

        ## TRACKING TIME
        start = datetime.now()
        ## GENERATING WILLARD CHANDLER SURFACE VIA PARALLEL
        wc = parallel( traj[end:], willard_chandler, kwargs, n_procs = n_procs, )
        time_elapsed = datetime.now() - start
        if verbose is True:
            print( 'Time elapsed (hh:mm:ss.ms) {}\n'.format(time_elapsed) )
        avg_density_field = wc.results
        
        ## CREATING WILLARD CHANDLER INTERFACE
        data = wc_interface( avg_density_field, avg_spacing, contour = contour )
        
        ## PRINTING TO FILE
        if verbose is True:
            print( '--- Outputs written to %s' %(out_path) )
        with open( path_wc_dat_file, 'w+' ) as outfile:
            outfile.write( '# x y z\n\n' )                      
            for line in data:
                outfile.write( '{:0.3f},{:0.3f},{:0.3f}\n'.format( line[2], line[1], line[0] ) ) # Outputting in x y z format
    
    ## LOADING THE DATA
    data = load_datafile(path_to_file = path_wc_dat_file)

    ## FUNCTION TO WRITE THE PDB FILE OF PROBES
    if write_pdb is True or os.path.isfile(path_pdb_file) is False or want_rewrite is True:
        # WRITE PDB FILE WITH PROBES   
        if verbose is True:         
            print( '--- Grid PDB file written to %s' %( path_pdb_file) )
        with open( path_pdb_file, 'w+' ) as pdbfile:
            pdbfile.write( 'TITLE     frame t=1.000 in water\n' )
            pdbfile.write( 'REMARK    THIS IS A SIMULATION BOX\n' )
            pdbfile.write( 'CRYST1{:9.3f}{:9.3f}{:9.3f}{:>7s}{:>7s}{:>7s} P 1           1\n'.format( traj.unitcell_lengths[0,0]*10, 
                                                                                                     traj.unitcell_lengths[0,1]*10, 
                                                                                                     traj.unitcell_lengths[0,2]*10, 
                                                                                                     '90.00', '90.00', '90.00' ) )
            pdbfile.write( 'MODEL        1\n' )
            for ndx, coord in enumerate( data ):
                line = "{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}\n".format( \
                        'ATOM', ndx+1, 'C', '', 'SUR', '', 1, '', coord[0]*10, coord[1]*10, coord[2]*10, 1.00, 1.00, '', '' )
                pdbfile.write( line )
                
            pdbfile.write( 'TER\n' )
            pdbfile.write( 'ENDMDL\n' )
        
    return data    


