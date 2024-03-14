# -*- coding: utf-8 -*-
# NOCW
# This file is part of the hydrophobicity project

R"""
The :class:`analysis_tool.voronoi` module contains tools to characterize Voronoi cells
of a system.
"""

##############################################################################
# Imports
##############################################################################

import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

#import itertools
import logging

logger = logging.getLogger(__name__)

try: 
    from scipy.spatial import Voronoi, voronoi_plot_2d
    _SCIPY_AVAILABLE = True
except ImportError:
    msg = ( 'scipy.spatial.Voronoi not available (requires scipy 0.12+), '
            'i.e hydrophobicity.voronoi not available.' )
    logger.warning(msg)
    _SCIPY_AVAILABLE = False
    
try:
    import matplotlib.pyplot as plt
    _PYPLOT_AVAILABLE = True
except ImportError:
    plt.plot = None
    msg = ( 'plt.plot not available (requires display), '
            'will only generate image files' )
    logger.warning(msg)
    _PYPLOT_AVAILABLE = False

__all__ = [ 'voronoi' ]

##############################################################################
# Voronoi tesselation class
##############################################################################
class voronoi:
    R'''Computes the Voronoi tessellation of a 2D or 3D system using qhull. This
    uses :class"`scipy.spatial.Voronoi`, accounting for pbc.
    
    qhull does not support pbc the box is expanded to include the periodic images.
    '''
    def __init__( self, out_path = None, periodic = True, dim = 2, cavity_center = np.array([]), cavity_dimensions = np.array([]), bin_width = 0.02, color_range = [], plot = False ):
        '''
        periodic: [bool] pbc true or false
        bin_width: [float] width of color bins
        color_range: [list] [min,max] of color range, or areas
        '''
        ### PRINTING
        print("**** CLASS: %s ****"%(self.__class__.__name__)) 

        if not _SCIPY_AVAILABLE:
            raise RuntimeError( 'Class cannot be run without SciPy' )
            
        if not _PYPLOT_AVAILABLE:
            print( 'Display not available, figures will be generated'
                   'as image files only' )
        
        if not out_path:
            raise RuntimeError( 'Output path not specified' )           
        
        self.out_path = out_path
        self.periodic = periodic
        self.dim = dim
        self.cavity_center = cavity_center
        self.cavity_dimensions = cavity_dimensions
        self.bin_width = bin_width
        self.color_range = color_range # color_range = [ 0.14, 0.32 ]
        self.plot = plot
            
    def compute( self, traj, tail_groups = [], out_name = "voronoi_diagram" ):
        R''' compute function calculates the voronoi tessallation of a SAM surface
        Cell area is calculated using the shoelace method (Gauss' area theory).
        See: https://en.wikipedia.org/wiki/Shoelace_formula
        
        INPUT
        -----
        traj: MDTraj trajectory
        tail_groups: list with atoms in tail groups, if empty last heavy atom is used
        
        OUTPUT
        ------
        Generates a voronoi diagram with the points colored based on the size of the area
        vor: [object]: output object from scipy.spatial.Voronoi
        sorted_positions [np.array, shape=(n_atoms,2)] array containing the positions of the tail atoms, sorted by area
        sorted_areas [np.array, shape=(n_atoms,1)] array containing the sorted area of each atom
        '''        
        solvents = [ 'HOH', 'MET', "CL" ]
        # DETERMINE HEAVY ATOMS OF THE SURFACE (THIS ASSUMES WE WANT TO IGNORE HYDROGENS)
        positions = np.copy( traj.xyz[:] )
#        positions = traj.xyz[:]
        if not tail_groups:
            ligand_ndx = [ [ atom.index for atom in residue.atoms if 'H' not in atom.name ] for residue in traj.topology.residues if residue.name not in solvents ]
            tail_group_label = 0
        else:
            ligand_ndx = [ [ atom.index for atom in residue.atoms ] for residue in traj.topology.residues if residue.name not in solvents ]
            atom_names = [ [ atom.name for atom in residue.atoms ] for residue in traj.topology.residues if residue.name not in solvents ]
            tail_group_label = []
            for ll, tail_group in enumerate( tail_groups ):
                for ndx in range(len(atom_names)):
                    if all( elem in atom_names[ndx] for elem in tail_group ):
                        tail_group_label.append( ll )
                        group_ndx = [ ligand_ndx[ndx][ii] for ii, name in enumerate( atom_names[ndx] ) if name in tail_group ]
                        tmp_pos = positions[ :, group_ndx, : ]
                        atom_mass = np.array([ traj.topology.atom(atom_ind).element.mass for atom_ind in group_ndx ])
                        positions[:, ligand_ndx[ndx][-1], : ] = ( tmp_pos * atom_mass[np.newaxis,:,np.newaxis] ).sum(axis=1) / atom_mass.sum()

        tail_group_ndx = [ ligand[-1] for ligand in ligand_ndx ]
        com_positions = positions[ :, tail_group_ndx, :2 ]

        if self.periodic:
            com_positions = self.replicate_images( com_positions, traj.unitcell_lengths[:,:2] )
            tail_group_label = np.tile( tail_group_label, 3**self.dim )
        
        vor = Voronoi( com_positions.mean(axis=0) ) # assumes surfaces are restained, i.e., only first frame is used
        
        ## SELECT POINTS AND CELLS ONLY IN THE SIMULATION BOX
        in_box = np.flatnonzero( np.logical_and( (vor.points[:,0] > 0.) * (vor.points[:,0] < traj.unitcell_lengths[:,0].mean()), 
                                                 (vor.points[:,1] > 0.) * (vor.points[:,1] < traj.unitcell_lengths[:,1].mean()) ) )
        
        region_ndx = vor.point_region[in_box]
        labels = tail_group_label[in_box]
        
        ## CALCULATE THE AREA OF EACH CELL USING THE SHOELACE METHOD
        areas = []
        for ndx in region_ndx:
            region = vor.regions[ndx]
            x = vor.vertices[region,0]
            y = vor.vertices[region,1]
            shift_up = np.arange( -len(x)+1, 1 )
            shift_down = np.arange( -1, len(x)-1 )
            areas.append( np.abs( np.sum( 0.5 * x * ( y[shift_up] - y[shift_down] ) ) ) )

        ## SORT AREAS, POSITIONS, AND LABELS IN INCREASING AREA. MAKES SIMPLER TO PLOT
        sorted_tails = []
        sorted_positions = []
        sorted_areas = []
        for a_ndx in np.argsort( areas ):
            sorted_tails.append( labels[a_ndx] )
            sorted_positions.append([ vor.points[in_box[a_ndx],0], vor.points[in_box[a_ndx],1] ])
            sorted_areas.append( areas[a_ndx] )

        ## CALCULATE THE AREA OF THE GROUPS UNDER THE CAVITY
        if self.cavity_center.any() and self.cavity_dimensions.any():
            center_2d = self.cavity_center[:2]
            x1_cavity = np.round( center_2d[0] - 0.5 * self.cavity_dimensions[0], decimals = 8 )
            x2_cavity = np.round( center_2d[0] + 0.5 * self.cavity_dimensions[0], decimals = 8 )
            y1_cavity = np.round( center_2d[1] - 0.5 * self.cavity_dimensions[1], decimals = 8 )
            y2_cavity = np.round( center_2d[1] + 0.5 * self.cavity_dimensions[1], decimals = 8 )
            
            if self.plot:
                ## PLOT A VORONOI DIAGRAM
                fig = voronoi_plot_2d( vor, show_points = True, show_vertices = False, line_colors = 'k', line_width = 2,
                                       line_alpha = 1.0, point_size = 16 )           
    
                plt.plot( [x1_cavity, x2_cavity], [y1_cavity, y1_cavity], 'c--', linewidth = 1.5 )
                plt.plot( [x1_cavity, x2_cavity], [y2_cavity, y2_cavity], 'c--', linewidth = 1.5 )
                plt.plot( [x1_cavity, x1_cavity], [y1_cavity, y2_cavity], 'c--', linewidth = 1.5 )
                plt.plot( [x2_cavity, x2_cavity], [y1_cavity, y2_cavity], 'c--', linewidth = 1.5 )
                plt.xlim([ x1_cavity-0.5, x2_cavity+0.5 ])
                plt.ylim([ y1_cavity-0.5, y2_cavity+0.5 ])
#                plt.xlim([ -1.*traj.unitcell_lengths.mean(axis=1)[0], 2.*traj.unitcell_lengths.mean(axis=1)[0] ])
#                plt.ylim([ -1.*traj.unitcell_lengths.mean(axis=1)[1], 2.*traj.unitcell_lengths.mean(axis=1)[1] ])                
                color = [ 'blue', 'red' ]
                marker = [ 's', 'o' ]
                
            x_verts = []
            y_verts = []
            areas = np.zeros( np.max(tail_group_label)+2 )
            for jj, ndx in enumerate( vor.point_region ):
                label = tail_group_label[jj]
                region = vor.regions[ndx]
                x = vor.vertices[region,0]
                y = vor.vertices[region,1]
                lx = len(x)
                x_vert = []
                y_vert = []
                for ii in range(lx):
                    ## DETERMINE IF VERTEX IS UNDER CAVITY
                    if x[ii] > x1_cavity and x[ii] < x2_cavity and y[ii] > y1_cavity and y[ii] < y2_cavity:
                        x_vert.append( x[ii] )
                        y_vert.append( y[ii] )                     
                        
                        ## LINEAR INTERPOLATE POINTS ALONG THE BOX PERIMETER
                        if x[(ii-1) % lx] < x1_cavity:
                            interp = ( y[ii] - y[(ii-1) % lx] ) * ( x1_cavity - x[(ii-1) % lx] ) / ( x[ii] - x[(ii-1) % lx] ) + y[(ii-1) % lx]
                            if interp > y1_cavity and interp < y2_cavity:
                                x_vert.append( x1_cavity )
                                y_vert.append( interp )

                        if x[(ii-1) % lx] > x2_cavity:
                            interp = ( y[ii] - y[(ii-1) % lx] ) * ( x2_cavity - x[(ii-1) % lx] ) / ( x[ii] - x[(ii-1) % lx] ) + y[(ii-1) % lx]
                            if interp > y1_cavity and interp < y2_cavity:
                                x_vert.append( x2_cavity )
                                y_vert.append( interp )

                        if y[(ii-1) % lx] < y1_cavity:
                            interp = ( x[ii] - x[(ii-1) % lx] ) * ( y1_cavity - y[(ii-1) % lx] ) / ( y[ii] - y[(ii-1) % lx] ) + x[(ii-1) % lx]
                            if interp > x1_cavity and interp < x2_cavity:
                                x_vert.append( interp )
                                y_vert.append( y1_cavity )

                        if y[(ii-1) % lx] > y2_cavity:
                            interp = ( x[ii] - x[(ii-1) % lx] ) * ( y2_cavity - y[(ii-1) % lx] ) / ( y[ii] - y[(ii-1) % lx] ) + x[(ii-1) % lx]
                            if interp > x1_cavity and interp < x2_cavity:
                                x_vert.append( interp )
                                y_vert.append( y2_cavity )
                                                      
                        if x[(ii+1) % lx] < x1_cavity:
                            interp = ( y[ii] - y[(ii+1) % lx] ) * ( x1_cavity - x[(ii+1) % lx] ) / ( x[ii] - x[(ii+1) % lx] ) + y[(ii+1) % lx]
                            if interp > y1_cavity and interp < y2_cavity:
                                x_vert.append( x1_cavity )
                                y_vert.append( interp )
                        
                        if x[(ii+1) % lx] > x2_cavity:
                            interp = ( y[ii] - y[(ii+1) % lx] ) * ( x2_cavity - x[(ii+1) % lx] ) / ( x[ii] - x[(ii+1) % lx] ) + y[(ii+1) % lx]
                            if interp > y1_cavity and interp < y2_cavity:
                                x_vert.append( x2_cavity )
                                y_vert.append( interp )
                        
                        if y[(ii+1) % lx] < y1_cavity:
                            interp = ( x[ii] - x[(ii+1) % lx] ) * ( y1_cavity - y[(ii+1) % lx] ) / ( y[ii] - y[(ii+1) % lx] ) + x[(ii+1) % lx]
                            if interp > x1_cavity and interp < x2_cavity:
                                x_vert.append( interp )
                                y_vert.append( y1_cavity )
                    
                        if y[(ii+1) % lx] > y2_cavity:
                            interp = ( x[ii] - x[(ii+1) % lx] ) * ( y2_cavity - y[(ii+1) % lx] ) / ( y[ii] - y[(ii+1) % lx] ) + x[(ii+1) % lx]
                            if interp > x1_cavity and interp < x2_cavity:
                                x_vert.append( interp )
                                y_vert.append( y2_cavity )
                    
                    ## DETERMINE IF EDGE INTERSECTS TWICE (DO NO VERTICES LIE UNDER CAVITY, BUT PART OF POLYGON DOES)            
                    ## LINEAR INTERPOLATE POINTS ALONG THE BOX PERIMETER
                    if (x[ii] < x1_cavity and y[ii] > y1_cavity and y[ii] < y2_cavity) and \
                       (x[(ii-1) % lx] > x1_cavity and x[(ii-1) % lx] < x2_cavity and y[(ii-1) % lx] < y1_cavity):
                        interp_y = ( y[ii] - y[(ii-1) % lx] ) * ( x1_cavity - x[(ii-1) % lx] ) / ( x[ii] - x[(ii-1) % lx] ) + y[(ii-1) % lx]
                        if interp_y > y1_cavity and interp_y < y2_cavity:
                            x_vert.append( x1_cavity )                        
                            y_vert.append( interp_y )
                        
                        interp_x = ( x[ii] - x[(ii-1) % lx] ) * ( y1_cavity - y[(ii-1) % lx] ) / ( y[ii] - y[(ii-1) % lx] ) + x[(ii-1) % lx]
                        if interp_x > x1_cavity and interp_x < x2_cavity:
                            x_vert.append( interp_x )
                            y_vert.append( y1_cavity )
                        
                    if (x[ii] < x1_cavity and y[ii] > y1_cavity and y[ii] < y2_cavity) and \
                       (x[(ii-1) % lx] > x1_cavity and x[(ii-1) % lx] < x2_cavity and y[(ii-1) % lx] > y2_cavity):
                        interp_y = ( y[ii] - y[(ii-1) % lx] ) * ( x1_cavity - x[(ii-1) % lx] ) / ( x[ii] - x[(ii-1) % lx] ) + y[(ii-1) % lx]
                        if interp_y > y1_cavity and interp_y < y2_cavity:
                            x_vert.append( x1_cavity )                        
                            y_vert.append( interp_y )
                        
                        interp_x = ( x[ii] - x[(ii-1) % lx] ) * ( y2_cavity - y[(ii-1) % lx] ) / ( y[ii] - y[(ii-1) % lx] ) + x[(ii-1) % lx]
                        if interp_x > x1_cavity and interp_x < x2_cavity:
                            x_vert.append( interp_x )
                            y_vert.append( y2_cavity )
                        
                    if (x[ii] > x2_cavity and y[ii] > y1_cavity and y[ii] < y2_cavity) and \
                       (x[(ii-1) % lx] > x1_cavity and x[(ii-1) % lx] < x2_cavity and y[(ii-1) % lx] < y1_cavity):
                        interp_y = ( y[ii] - y[(ii-1) % lx] ) * ( x2_cavity - x[(ii-1) % lx] ) / ( x[ii] - x[(ii-1) % lx] ) + y[(ii-1) % lx]
                        if interp_y > y1_cavity and interp_y < y2_cavity:
                            x_vert.append( x2_cavity )                        
                            y_vert.append( interp_y )
                        
                        interp_x = ( x[ii] - x[(ii-1) % lx] ) * ( y1_cavity - y[(ii-1) % lx] ) / ( y[ii] - y[(ii-1) % lx] ) + x[(ii-1) % lx]
                        if interp_x > x1_cavity and interp_x < x2_cavity:
                            x_vert.append( interp_x )
                            y_vert.append( y1_cavity )
                        
                    if (x[ii] > x2_cavity and y[ii] > y1_cavity and y[ii] < y2_cavity) and \
                       (x[(ii-1) % lx] > x1_cavity and x[(ii-1) % lx] < x2_cavity and y[(ii-1) % lx] > y2_cavity):
                        interp_y = ( y[ii] - y[(ii-1) % lx] ) * ( x2_cavity - x[(ii-1) % lx] ) / ( x[ii] - x[(ii-1) % lx] ) + y[(ii-1) % lx]
                        if interp_y > y1_cavity and interp_y < y2_cavity:
                            x_vert.append( x2_cavity )                        
                            y_vert.append( interp_y )
                        
                        interp_x = ( x[ii] - x[(ii-1) % lx] ) * ( y2_cavity - y[(ii-1) % lx] ) / ( y[ii] - y[(ii-1) % lx] ) + x[(ii-1) % lx]
                        if interp_x > x1_cavity and interp_x < x2_cavity:
                            x_vert.append( interp_x )
                            y_vert.append( y2_cavity )
                        
                    if (x[ii] < x1_cavity and y[ii] > y1_cavity and y[ii] < y2_cavity) and \
                       (x[(ii+1) % lx] > x1_cavity and x[(ii+1) % lx] < x2_cavity and y[(ii+1) % lx] < y1_cavity):
                        interp_y = ( y[ii] - y[(ii+1) % lx] ) * ( x1_cavity - x[(ii+1) % lx] ) / ( x[ii] - x[(ii+1) % lx] ) + y[(ii+1) % lx]
                        if interp_y > y1_cavity and interp_y < y2_cavity:
                            x_vert.append( x1_cavity )                        
                            y_vert.append( interp_y )
                        
                        interp_x = ( x[ii] - x[(ii+1) % lx] ) * ( y1_cavity - y[(ii+1) % lx] ) / ( y[ii] - y[(ii+1) % lx] ) + x[(ii+1) % lx]
                        if interp_x > x1_cavity and interp_x < x2_cavity:
                            x_vert.append( interp_x )
                            y_vert.append( y1_cavity )
                        
                    if (x[ii] < x1_cavity and y[ii] > y1_cavity and y[ii] < y2_cavity) and \
                       (x[(ii+1) % lx] > x1_cavity and x[(ii+1) % lx] < x2_cavity and y[(ii+1) % lx] > y2_cavity):
                        interp_y = ( y[ii] - y[(ii+1) % lx] ) * ( x1_cavity - x[(ii+1) % lx] ) / ( x[ii] - x[(ii+1) % lx] ) + y[(ii+1) % lx]
                        if interp_y > y1_cavity and interp_y < y2_cavity:
                            x_vert.append( x1_cavity )                        
                            y_vert.append( interp_y )
                        
                        interp_x = ( x[ii] - x[(ii+1) % lx] ) * ( y2_cavity - y[(ii+1) % lx] ) / ( y[ii] - y[(ii+1) % lx] ) + x[(ii+1) % lx]
                        if interp_x > x1_cavity and interp_x < x2_cavity:
                            x_vert.append( interp_x )
                            y_vert.append( y2_cavity )
                        
                    if (x[ii] > x2_cavity and y[ii] > y1_cavity and y[ii] < y2_cavity) and \
                       (x[(ii+1) % lx] > x1_cavity and x[(ii+1) % lx] < x2_cavity and y[(ii+1) % lx] < y1_cavity):
                        interp_y = ( y[ii] - y[(ii+1) % lx] ) * ( x2_cavity - x[(ii+1) % lx] ) / ( x[ii] - x[(ii+1) % lx] ) + y[(ii+1) % lx]
                        if interp_y > y1_cavity and interp_y < y2_cavity:
                            x_vert.append( x2_cavity )                        
                            y_vert.append( interp_y )
                        
                        interp_x = ( x[ii] - x[(ii+1) % lx] ) * ( y1_cavity - y[(ii+1) % lx] ) / ( y[ii] - y[(ii+1) % lx] ) + x[(ii+1) % lx]
                        if interp_x > x1_cavity and interp_x < x2_cavity:
                            x_vert.append( interp_x )
                            y_vert.append( y1_cavity )
                        
                    if (x[ii] > x2_cavity and y[ii] > y1_cavity and y[ii] < y2_cavity) and \
                       (x[(ii+1) % lx] > x1_cavity and x[(ii+1) % lx] < x2_cavity and y[(ii+1) % lx] > y2_cavity):
                        interp_y = ( y[ii] - y[(ii+1) % lx] ) * ( x2_cavity - x[(ii+1) % lx] ) / ( x[ii] - x[(ii+1) % lx] ) + y[(ii+1) % lx]
                        if interp_y > y1_cavity and interp_y < y2_cavity:
                            x_vert.append( x2_cavity )                        
                            y_vert.append( interp_y )
                        
                        interp_x = ( x[ii] - x[(ii+1) % lx] ) * ( y2_cavity - y[(ii+1) % lx] ) / ( y[ii] - y[(ii+1) % lx] ) + x[(ii+1) % lx]
                        if interp_x > x1_cavity and interp_x < x2_cavity:
                            x_vert.append( interp_x )
                            y_vert.append( y2_cavity )
                    
                ## PUT CORNERS IN NEAREST POLYGON
                if x_vert:
                    min_xx = np.round( np.min(x_vert), decimals = 8 )
                    min_yy = np.round( np.min(y_vert), decimals = 8 )
                    max_xx = np.round( np.max(x_vert), decimals = 8 )
                    max_yy = np.round( np.max(y_vert), decimals = 8 )
                    
                    if min_xx == x1_cavity and min_yy == y1_cavity:
                        point = Point( x1_cavity, y1_cavity )
                        polygon = Polygon([ (xi, yi) for xi, yi in zip( x, y ) ])
                        if polygon.contains(point):
                            x_vert.append( x1_cavity )
                            y_vert.append( y1_cavity )
                        
                    if min_xx == x1_cavity and max_yy == y2_cavity: 
                        point = Point( x1_cavity, y2_cavity )
                        polygon = Polygon([ (xi, yi) for xi, yi in zip( x, y ) ])
                        if polygon.contains(point):
                            x_vert.append( x1_cavity )
                            y_vert.append( y2_cavity )
                        
                    if max_xx == x2_cavity and min_yy == y1_cavity:
                        point = Point( x2_cavity, y1_cavity )
                        polygon = Polygon([ (xi, yi) for xi, yi in zip( x, y ) ])
                        if polygon.contains(point):
                            x_vert.append( x2_cavity )
                            y_vert.append( y1_cavity )
                        
                    if max_xx == x2_cavity and max_yy == y2_cavity:
                        point = Point( x2_cavity, y2_cavity )
                        polygon = Polygon([ (xi, yi) for xi, yi in zip( x, y ) ])
                        if polygon.contains(point):
                            x_vert.append( x2_cavity )
                            y_vert.append( y2_cavity )
                
                    ## SORT VERTICES COUNTERCLOCKWISE USING POLAR ANGLE APPROACH
                    x_vert, y_vert = np.array(x_vert), np.array(y_vert)
                    sorted_idx = np.argsort([ np.arctan2( y_vert[nn]-y_vert.mean(), x_vert[nn]-x_vert.mean() ) for nn in range(len(x_vert)) ])
                    x_sorted, y_sorted = x_vert[sorted_idx], y_vert[sorted_idx]
                    x_verts.append( x_sorted )
                    y_verts.append( y_sorted )
                    ## CALCULATE AREA OF POLYGONS USING THE SHOELACE METHOD
                    shift_up = np.arange( -len(x_sorted)+1, 1 )
                    shift_down = np.arange( -1, len(x_sorted)-1 )
                    area = np.abs( np.sum( 0.5 * x_sorted * ( y_sorted[shift_up] - y_sorted[shift_down] ) ) )
                    areas[0] += area
                    areas[label+1] += area
            
                    if self.plot:                    
                        ## PLOT POINT ON DIAGRAM
                        plt.scatter( x_sorted, y_sorted, marker = marker[label], color = color[label], facecolors = 'None', s = 160 )

#        color = [ 'grey', 'blue', 'red', 'green', 'orange' ]
#        marker = [ 's', 'o', '^', 'v', '<', '*' ]
#        for ii in range(len(x_verts)):
#            for jj in range(len(x_verts[ii])):          
#                plt.scatter( x_verts[ii][jj], y_verts[ii][jj], marker = marker[jj], color = color[jj], facecolors = 'None', s = 160 )
#                plt.pause(0.5)
            
        print( "Total area: {:.3f} nm^2".format( areas[0] ) )
        for label in range(len(tail_groups)):
            print( "End group {:d} area: {:.3f} nm^2 ({:.1f}%)".format( label+1, areas[label+1], 100. * areas[label+1] / areas[0] ) )
        
        return vor, np.array(sorted_positions), np.array(sorted_areas), sorted_tails, areas

    ## COMPUTE THE BUFFER PARTICLES
    @staticmethod
    def replicate_images( positions, box ):
        R'''Adds nearest image particles to box
        
        INPUT:
            positions: [np.array; size = (n_frames, n_atoms, 2 or 3)] array containing atom positions
            box: [np.array; size = (n_frames, 2 or 3)] simution box vectors
            
        OUTPUT:
            new_positions: [ np.array; size = ( n_frames, n_atoms*(9 or 27), 2 or 3)] array containing positions of real and image atoms
        '''
        ## DETERMINING DIMENSIONS OF DIAGRAM
        if positions.shape[2] < 3:
            if box.shape[1] > 2:
                raise RuntimeError( '  BOX SIZE AND POSITIONS ARE NOT THE SAME DIMENSIONS' )
                
            print( '  Calculating 2D Voronoi diagram\n')
            is_2d = True
        else:
            if positions.shape[2] > 2:
                raise RuntimeError( '  BOX SIZE AND POSITIONS ARE NOT THE SAME DIMENSIONS' )
                
            print( '  Calculating 3D Voronoi diagram\n')
            is_2d = False
        
        ## UPDATE POSITIONS TO ACCOUNT FOR ALL SIDES
        shape = list(positions.shape)
        shape[1] = 0
        new_positions = np.empty( shape = shape, dtype = float )
        for x in [-1,0,1]:
            new_x = positions[...,0] + x*box[:,np.newaxis,0]
            for y in [-1,0,1]:
                new_y = positions[...,1] + y*box[:,np.newaxis,1]
                if is_2d:
                    new_xy = np.stack( ( new_x, new_y ), axis=2 )
                    new_positions = np.hstack(( new_positions, new_xy ))
                else:
                    for z in [-1,0,1]:
                        new_z = positions[...,2] + z*box[:,np.newaxis,2]
                        new_xyz = np.stack( ( new_x, new_y, new_z ), axis=2 )
                        new_positions = np.hstack(( new_positions, new_xyz ))
                        
        return new_positions
    
    ## PLOT THE VOROROI DIAGRAM
    def plot( self, traj, vor, positions, areas, color_range = [] ):
        '''
        '''
        x_max = np.mean(traj.unitcell_lengths[:,0])
        y_max = np.mean(traj.unitcell_lengths[:,1])
        ## PLOT A VORONOI DIAGRAM
        fig = voronoi_plot_2d( vor, show_points = False, show_vertices = False, line_colors = 'k', line_width = 2,
                               line_alpha = 1.0, point_size = 16 )
        
        ## CREATE A SCATTER PLOT OF EACH VORONOI CELL WITH THE POINT COLORED BASED ON THE CELL AREA
        color = plt.cm.coolwarm( np.linspace( 0, 1, int(1./self.bin_width) ) )
        norm_area = [ ( area - color_range[0] ) / ( color_range[1] - color_range[0] ) for area in areas ] # normalize where min = 0 and max = 1
        put_bins = [ np.floor( a / self.bin_width ).astype('int') for a in norm_area ]
        for ndx, area in enumerate( areas ):
            plt.scatter( positions[ndx,0], positions[ndx,1], marker = 'o', 
                          s = 60, color = color[put_bins[ndx]] )
            
        plt.xlim([ 0, x_max ])
        plt.ylim([ 0, y_max ])
        
        if self.cavity_center.any() and self.cavity_dimensions.any():
            print( 'plotting cavity' )
            center_2d = self.cavity_center[:2]
            plt.scatter( center_2d[0], center_2d[1], marker = 's', s = 60, color = 'c' )
            x1_cavity = center_2d[0] - 0.5 * self.cavity_dimensions[0]
            x2_cavity = center_2d[0] + 0.5 * self.cavity_dimensions[0]
            y1_cavity = center_2d[1] - 0.5 * self.cavity_dimensions[1]
            y2_cavity = center_2d[1] + 0.5 * self.cavity_dimensions[1]
            
            ## ADJUST FOR PBC
            if self.periodic:
                if x1_cavity < 0.0:
                    x1_cavity += x_max
                if x2_cavity > x_max:
                    x2_cavity -= x_max
                if y1_cavity < 0.0:
                    y1_cavity += y_max
                if y2_cavity > y_max:
                    y2_cavity -= y_max
                
            if x1_cavity > x2_cavity:
                plt.plot( [x1_cavity, x_max], [y1_cavity, y1_cavity], 'c--', linewidth = 1.5 )
                plt.plot( [0.0, x2_cavity], [y1_cavity, y1_cavity], 'c--', linewidth = 1.5 )
                plt.plot( [x1_cavity, x_max], [y2_cavity, y2_cavity], 'c--', linewidth = 1.5 )
                plt.plot( [0.0, x2_cavity], [y2_cavity, y2_cavity], 'c--', linewidth = 1.5 )
            else:
                plt.plot( [x1_cavity, x2_cavity], [y1_cavity, y1_cavity], 'c--', linewidth = 1.5 )
                plt.plot( [x1_cavity, x2_cavity], [y2_cavity, y2_cavity], 'c--', linewidth = 1.5 )
                
            if y1_cavity > y2_cavity:
                plt.plot( [x1_cavity, x1_cavity], [y1_cavity, y_max], 'c--', linewidth = 1.5 )
                plt.plot( [x1_cavity, x1_cavity], [0.0, y2_cavity], 'c--', linewidth = 1.5 )
                plt.plot( [x2_cavity, x2_cavity], [y1_cavity, y_max], 'c--', linewidth = 1.5 )
                plt.plot( [x2_cavity, x2_cavity], [0.0, y2_cavity], 'c--', linewidth = 1.5 )
            else:
                plt.plot( [x1_cavity, x1_cavity], [y1_cavity, y2_cavity], 'c--', linewidth = 1.5 )
                plt.plot( [x2_cavity, x2_cavity], [y1_cavity, y2_cavity], 'c--', linewidth = 1.5 )
        
        plt.tight_layout()
#        plt.savefig( self.out_path + 'voronoi_diagram.png', format='png', dpi=1000, bbox_inches='tight' )
#        plt.savefig( self.out_path + 'voronoi_diagram.svg', format='svg', dpi=1000, bbox_inches='tight' )
