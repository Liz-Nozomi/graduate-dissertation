# -*- coding: utf-8 -*-
# NOCW
# This file is part of the hydrophobicity project

R"""
get_geometry.py
"""

##############################################################################
# Imports
##############################################################################

import numpy as np
import mdtraj as md
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



__all__ = [ 'geometry' ]

##############################################################################
# geometry class
##############################################################################
class geometry:
    R'''
    '''
    def __init__( self, ref_ndx, ref_out = [], visualize_ellipse = False, visualize_frame = False ):
        R'''
        '''
        self.ref_ndx = ref_ndx
        self.ref_out = ref_out
        self.visualize_ellipse = visualize_ellipse
        self.visualize_frame = visualize_frame
    
    def center_mol( self, traj ):
        R'''
        '''
        if self.ref_out:
            traj = traj.atom_slice( self.ref_out )
        
        if self.visualize_ellipse:
            coords = traj.xyz[ 0, self.ref_ndx, : ]
            mol_masses = np.array([ traj.topology.atom(idx).element.mass for idx in self.ref_ndx ])
            mol_mass = mol_masses.sum()
            com = np.sum( coords * mol_masses[np.newaxis,:,np.newaxis], axis=1 ) / mol_mass        
            plot_ellipsoid( coords - com, limits = [ [ -1.5, 1.5 ], [ -1.5, 1.5 ], [ -1.5, 1.5 ] ] )  
        else: 
            for ff in range( traj.n_frames ):
                if self.visualize_frame:
                    plot_frame( traj, ff, self.ref_ndx )
                    
                coords = traj.xyz[ ff, self.ref_ndx, : ]
                mol_masses = np.array([ traj.topology.atom(idx).element.mass for idx in self.ref_ndx ])
                mol_mass = mol_masses.sum()
                com = np.sum( coords * mol_masses[np.newaxis,:,np.newaxis], axis=1 ) / mol_mass    
                
                # SVD on coords
                U, D, V = np.linalg.svd( coords - com )
                
                # rotate coords
                traj.xyz[ff,:,:] = np.dot( traj.xyz[ff,:,:] - com, np.linalg.inv(V) )
                                
                # center coords
                traj.xyz[ff,:,:] += 0.5 * traj.unitcell_lengths[ff,:]
                
                # adjust pbc
                x = traj.xyz[ff,:,0] 
                y = traj.xyz[ff,:,1] 
                z = traj.xyz[ff,:,2]
                x[x>traj.unitcell_lengths[ff,0]] -= traj.unitcell_lengths[ff,0]
                x[x<0.] += traj.unitcell_lengths[ff,0]
                y[y>traj.unitcell_lengths[ff,1]] -= traj.unitcell_lengths[ff,1]
                y[y<0.] += traj.unitcell_lengths[ff,1]
                z[z>traj.unitcell_lengths[ff,2]] -= traj.unitcell_lengths[ff,2]
                z[z<0.] += traj.unitcell_lengths[ff,2]
                traj.xyz[ff,:,0] = x 
                traj.xyz[ff,:,1] = y
                traj.xyz[ff,:,2] = z
                
                if self.visualize_frame:
                    plot_frame( traj, ff, self.ref_ndx )                    
            
        return traj

def plot_frame( traj, frame, ref_ndx ):
    R'''
    '''
    other_ndx = [ atom.index for atom in traj.topology.atoms if atom.index not in ref_ndx ]
    coords = traj.xyz[frame,ref_ndx,:]
    other = traj.xyz[frame,other_ndx,:]    
    
    fig = plt.figure()
    ax = fig.add_subplot( 111, projection = '3d' )
    
    # plot ref atoms
    ax.scatter( coords[:,0], coords[:,1], coords[:,2], c = 'b', marker = 'o' )
    
    # plot rest
    ax.scatter( other[:1000,0], other[:1000,1], other[:1000,2], c = 'r', marker = '.' )

    ax.set_xlim([ 0, traj.unitcell_lengths[frame,0] ])
    ax.set_ylim([ 0, traj.unitcell_lengths[frame,1] ])
    ax.set_zlim([ 0, traj.unitcell_lengths[frame,2] ])
    ax.set_xlabel( 'X Label' )
    ax.set_ylabel( 'Y Label' )
    ax.set_zlabel( 'Z Label' )     

        
def plot_ellipsoid( coords, limits ):
    """
    Modify the unit circle and basis vector by applying a matrix.
    Visualize the effect of the matrix in 2D.

    Parameters
    ----------
    matrix : array-like
        2D matrix to apply to the unit circle.
    vectorsCol : HEX color code
        Color of the basis vectors

    Returns:

    fig : instance of matplotlib.figure.Figure
        The figure containing modified unit circle and basis vectors.
    """
    # SVD on coords
    U, D, V = np.linalg.svd( coords )

    # Sphere coords
    theta = np.linspace( 0.0, 2.0 * np.pi, 100 )
    phi = np.linspace( 0.0, np.pi, 100 )
    
    # scale ellipsoid
    x = D[0] * np.outer( np.cos( theta ), np.sin( phi ) )
    y = D[1] * np.outer( np.sin( theta ), np.sin( phi ) )
    z = D[2] * np.outer( np.ones_like( theta ), np.cos( phi ) )
    
    # rotate ligand ellipsoid
    for i in range( len( x ) ):
        for j in range( len( x ) ):
            [ x[ i, j ], y[ i, j ], z[ i, j ] ] = np.dot( [ x[ i, j ], y[ i, j ], z[ i, j ] ], V )
    
    # Vectors
    u1 = [ D[0] * V[0,0], D[0] * V[0,1], D[0] * V[0,2] ]
    v1 = [ D[1] * V[1,0], D[1] * V[1,1], D[1] * V[1,2] ]
    w1 = [ D[2] * V[2,0], D[2] * V[2,1], D[2] * V[2,2] ]

    fig = plt.figure()
    ax = fig.add_subplot( 111, projection = '3d' )
    
    # plot ligand
    ax.scatter( coords[:,0], coords[:,1], coords[:,2], c = 'r', marker = 'o' )

    plot_vectors( ax, [ u1, v1, w1 ] )

    # plot
    ax.plot_wireframe( x, y, z, rstride = 4, cstride = 4, color = 'b', alpha = 0.2 )

    ax.set_xlim([ limits[0][0], limits[0][1] ])
    ax.set_ylim([ limits[1][0], limits[1][1] ])
    ax.set_zlim([ limits[2][0], limits[2][1] ])
    ax.set_xlabel( 'X Label' )
    ax.set_ylabel( 'Y Label' )
    ax.set_zlabel( 'Z Label' )           

def plot_vectors( ax, vecs, cols = [ 'black', 'black', 'black' ], alpha = 1 ):
    """
    Plot set of vectors.

    Parameters
    ----------
    vecs : array-like
        Coordinates of the vectors to plot. Each vectors is in an array. For
        instance: [[1, 3, 2], [2, 2, 1]] can be used to plot 2 vectors.
    cols : array-like
        Colors of the vectors. For instance: ['red', 'blue'] will display the
        first vector in red and the second in blue.
    alpha : float
        Opacity of vectors

    Returns:

    fig : instance of matplotlib.figure.Figure
        The figure of the vectors
    """
    for i in range( len( vecs ) ):
        x = np.concatenate( [ [0,0,0], vecs[i] ] )
        ax.quiver( [ x[0] ],
                   [ x[1] ],
                   [ x[2] ],
                   [ x[3] ],
                   [ x[4] ],
                   [ x[5] ],
                   color = cols[i],
                   alpha = alpha )
            