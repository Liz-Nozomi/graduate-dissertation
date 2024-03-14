# -*- coding: utf-8 -*-
"""
willard_chandler.py
The purpose of this script is to create a willard-chandler interface code that 
is interpretable. In other words, if you are debugging things like the width 
of the distribution and so on, we would like to be able to generate WC-interfaces 
for those variations. More importantly, we would like to be able to visualize 
these effects in a frame-by-frame basis. 

Written by: Alex K. Chew (12/12/2019)
Using code from Brad C. Dallin (Thanks Brad!)
"""
## IMPORTING MODULES
import os
import mdtraj as md
import numpy as np

## IMPORTING GLOBAL VARIABLES
from MDDescriptors.surface.willard_chandler_global_vars import WC_DEFAULTS

## GAUSSIAN FUNCTIONS
from scipy.stats import gaussian_kde
from scipy.spatial import cKDTree

## MARCHING CUBES
from skimage.measure import marching_cubes_lewiner as marching_cubes

### FUNCTION TO PLOT THE SCALAR FIELD AND POINTS
def mlab_plot_density_field(density_field,
                            grid,
                            num_grid_pts,
                            interface_points = None,
                            pos = None, 
                            size=(400, 350)
                            ):
    '''
    The purpose of this function is to plot the mayavi.
    INPUTS:
        pos: [np.array, shape=(num, 3)]
            Atomic positions that you want to print out. If None, none of 
            the positions are printed.
        density_field: [np.array, shape=(num_grid_points)]
            density field output
        grid: [np.array, shape = (3, num_points)]
            grid positions in x, y, z
        size: [tuple]
            size of the figure in pixels
    OUTPUTS:
        figure: [obj]
            Figure object
    '''
    ## IMPORTING MLAB
    from mayavi import mlab
    # from mayavi import mlab
    ## CLEARING FIGURE
    try:
        mlab.close()
    except AttributeError:
        pass
    
    # PLOTTING WITH MAYAVI
    figure = mlab.figure('DensityPlot',
                         bgcolor = (.5, .5, .5),
                         size = size,)
    
    ## RESHAPE DENSITY
    density = density_field.reshape(num_grid_pts)
    
    ## DEFINING X, Y, Z VECTORS
    xi = grid[0].reshape(num_grid_pts)
    yi = grid[1].reshape(num_grid_pts)
    zi = grid[2].reshape(num_grid_pts)

    ## PLOTTING DOTS
    if pos is not None:
        mlab.points3d(pos[:,0],
                      pos[:,1],
                      pos[:,2],
                      scale_factor=0.05, #.05
                      opacity=1, # 0.1
                      color=(0, 0, 0), # black
                      ) # size_array
    
    ## PLOTTING INTERFACE
    if interface_points is not None:
        mlab.points3d(interface_points[:,0],
                      interface_points[:,1],
                      interface_points[:,2],
                      scale_factor=0.05, #.05
                      opacity=1, # 0.1
                      color=(1, 0, 0),
                      ) # size_array
    
    ## PLOTTING SCALAR FIELD
    scalar_field = mlab.pipeline.scalar_field(xi, 
                                              yi, 
                                              zi, 
                                              density,
                                              opacity = 1)

    ## GETTING MIN AND MAX
    dens_min = density.min()
    dens_max = density.max()
    
    ## GETTING VOLUME
    vol = mlab.pipeline.volume(scalar_field, 
                               vmin=dens_min, 
                               vmax=dens_min + .5*(dens_max-dens_min),
                               )
    
    ## ADDING COLOR BAR
    mlab.colorbar(object = vol,
                  orientation = 'vertical')
    
    ## ADDING OUTLINE
    mlab.outline()
    
    ## DRAWING AXIS
    mlab.axes()
    # mlab.show() # <-- freezes the mlab system for python 3.6
    
    ## STORING FIGURE
    '''
    PRINT_FIGURE_INFO = {
                            'size': (1920,1200), # Size of the image in pixels width x height
                        }

    
    ## SAVING FIGURE
    mlab.savefig(filename="Debug_planar_SAM.png",
                 figure = figure,
                 magnification = 5,
                 **PRINT_FIGURE_INFO
                 )

    ## PLOTTING DENSITY FIELD
    mlab_plot_density_field(density_field = density_field,
                            interface_points = interface_points,
                            grid = grid,
                            pos = None, 
                            )
    '''
    return figure

##########################
### WC INTERFACE CLASS ###
##########################
class wc_interface:
    '''
    The purpose of this class is to generate the Willard-Chandler interface.
    ASSUMPTIONS:
        - You have a NVT ensemble, so box does not change
        - You do not have atoms appearing and being destroyed (normal for MD simulations)
    INPUTS:
        traj: [md.traj]
            trajectory object
        sigma: [float]
            sigma value used to denote the molecular diameter (xi in text)
        mesh: [list]
            list of mesh increments in x, y, z
        residue_list: [list]
            list of water residues
        verbose: [logical]
            True if you want to print out details
        print_freq: [int]
            freuqency to print density field results
        norm: [logical]
            True if you want to normalize by dividing number of frames. 
            If not, we just sum all the densities -- useful for parallelizing!
    OUTPUTS:
        ## STORED INPUTS
        self.sigma: [float]
            sigma value used to denote the molecular diameter (xi in text)
        self.mesh: [list]
            list of mesh increments in x, y, z
        self.residue_list: [list]
            list of water residues
        ## INITIALIZATION
        self.box: [list]
            list of box vectors (L_x, L_y, L_z)
        self.ndx_solv: [np.array, shape=(num_atoms)]
            index of all the solvents
        self.num_grid_pts: [np.array, shape = 3]
            number of grid points in the x, y, z dimensions
        self.spacing: [np.array, shape = 3]
            spacing between grid points in x, y, z dimensions
        self.grid: [np.array, shape = (3, num_grid_pts)]
            x, y, z positions of all the grid points
        
    FUNCTIONS:
        compute_density_single_frame:
            computes single frame density across all grid points
        compute:
            main function that finds average density across the entire 
            interface
        compute_compatible_mesh_params: 
            computes mesh parameters based on the box
        create_grid_of_points: [staticmethod]
            creates grid points based on number of grids in x, y, z
        compute_gaussian_kde_pbc: [staticmethod]
            coomputes gaussian kernals with periodic boundary conditions
        compute_density_field:
            function that does the density field calculation
        find_heavy_atoms_of_residues: [staticmethod]
            function that gets all the heavy atoms of residues
        generate_wc_interface: [staticmethod]
            general function to create the willard-chandler interface
        get_wc_interface_points: 
            practical function using the variables in this class
       mlab_plot_density_field: [staticmethod]
           method for plotting the density field as a visualization
    '''
    ## INITIALIZING
    def __init__( self, 
                  traj,
                  sigma = 0.24, 
                  mesh = [ 0.1, 0.1, 0.1 ],
                  residue_list = ['HOH'],
                  verbose = True, 
                  print_freq = 100,
                  norm = True,
                  ):
        
        ## STORING sigma AND MESH
        self.sigma = sigma
        self.mesh = mesh
        self.residue_list = residue_list
        self.verbose = verbose
        self.print_freq = print_freq
        self.norm = norm
        
        ##############################################
        ### INITIALIZING THE GRIDDING AND INDEXING ###
        ##############################################
        ## GETTING BOX LENGTHS
        self.box = traj.unitcell_lengths[ 0, : ] # ASSUME BOX DOES NOT CHANGE!
        
        ## GETTING INDEX
        self.ndx_solv = self.find_heavy_atoms_of_residues(traj = traj)
        
        ## COMPUTING MESH PARAMETERS
        self.num_grid_pts, self.spacing = self.compute_compatible_mesh_params(mesh = self.mesh,
                                                                              box  =  self.box)
        
        ## CREATING GRID POINTS
        self.grid = self.create_grid_of_points(box = self.box,
                                               num_grid_pts = self.num_grid_pts)

        return
    
    ### FUNCTION TO COMPUTE DENSITY FOR A SINGLE FRAME
    def compute_density_single_frame(self, 
                                     traj, 
                                     frame = 0,
                                     want_plot = False):
        '''
        The purpose of this function is to compute the density field for a single 
        frame. 
        INPUTS:
            traj: [md.traj]
                trajectory object
            frame: [int]
                frame that you are interested in
            want_plot: [list]
                True if you want the plot the density
        OUTPUTS:
            density_field: [np.array, shape=(num_grid_points)]
                density at each grid point
        '''        
        ## GETTING POSITIONS OF ALL WATER HEAVY ATOMS
        pos = traj.xyz[ frame, self.ndx_solv, : ]
        
        ## GETTING DENSITIES
        density_field = self.compute_density_field(grid = self.grid,
                                                   pos = pos,
                                                   )
        
        ## PLOTTING
        if want_plot is True:     
            ## PLOTTING DENSITY FIELD
            figure = mlab_plot_density_field(density_field = density_field,
                                              interface_points = None,
                                              grid = self.grid,
                                              pos = None, 
                                              )
        return density_field
    
    ### FUNCTION TO COMPUTE FOR ALL FRAMES
    def compute(self, 
                traj,
                frames = [],
                ):
        '''
        The purpose of this function is to compute density fields for all frames. 
        INPUTS:
            traj: [md.traj]
                trajectory object
            frames: [list]
                list of frames to compute for trajectory
            print_rate: [int]
                print rate, default = 100 frames. 
        OUTPUTS:
            avg_density_field: [np.array, shape=(num_grid_points)]
                average density field
        '''
        ## LOADING FRAMES TO TRAJECTORY
        if len(frames)>0:
            traj = traj[tuple(frames)]
        ## DEFINING TOTAL TRAJECTORY SIZE
        total_traj_size = traj.time.size
        if self.verbose is True:
            if len(frames)==0:
                print("--- Calculating density field for %s simulations windows ---" % (str(total_traj_size)) )
                
        ## LOOPING THROUGH EACH TRAJECTORY FRAME
        for frame in np.arange(0, total_traj_size):
            if frame == 0:
                ## GETTING DENSITY FIELD FOR FIRST FRAME
                avg_density_field = self.compute_density_single_frame(traj = traj,
                                                                      frame = 0)
            else:
                ## COMPUTING DENSITY FIELD
                density_field = self.compute_density_single_frame(traj = traj,
                                                                  frame = frame)
                ## ADDING TO TOTAL DENSITY
                avg_density_field += density_field
            
            ## PRINTING 
            if traj.time[frame] % self.print_freq == 0:
                print("====> Working on frame %d"%(traj.time[frame]))
                    
        ## NORMALIZING BY FRAMES
        if self.norm is True:
            avg_density_field /= total_traj_size
        
        return avg_density_field
    
    ## GENERATING COMPATIBLE MESH PARAMETETRS
    @staticmethod
    def compute_compatible_mesh_params(mesh, box):
        """ 
        The purpose of this function is to determine the number of grid points 
        for the mesh in x, y, and z dimensions. The mesh size and box size 
        are taken into account.
        INPUTS:
            mesh: [np.array, shape = 3]
                mesh in x, y, z dimensions
            box: [np.array, shape = 3]
        OUTPUTS:
            num_grid_pts: [np.array, shape=3]
                number of grid points in x, y, z dimensions
            spacing: [np.array, shape=3]
                spacing in x, y, z dimensions
        """
        ## GETTING THE UPPER BOUND OF THE GRID POINTS
        num_grid_pts = np.ceil(box/mesh).astype('int')
        ## GETTING SPACING BETWEEN GRID POINTS
        spacing = box / num_grid_pts
        return num_grid_pts, spacing
    
    ### FUNCTION TO CREATE GRID POINTS
    @staticmethod
    def create_grid_of_points(box,num_grid_pts):
        '''
        The purpose of this function is to create a grid of points given the box 
        details. 
        INPUTS:
            box: [np.array, shape = 3]
                the simulation box edges
            num_grid_pts: [np.array, shape = 3]
                number of grid points
        OUTPUTS:
            grid: [np.array, shape = (3, num_points)]
                grid points in x, y, z positions
        '''    
        ## CREATING XYZ POINTS OF THE GRID
        xyz = np.array([ np.linspace(0, box[each_axis], int(num_grid_pts[each_axis]), 
                                     endpoint = False )
                         for each_axis in range(len(num_grid_pts))])
        '''
        NOTE1: If endpoint is false, it will not include the last value. This is 
        important for PBC concerns
        NOTE2: XYZ was generated slightly differently in Brad's code. He included a 
        subtraction of  box[i] / num_grid_pts[i] (equivalent to spacing). This is no 
        longer necessary with endpoint set to false
        '''
        ## CREATING MESHGRID OF POINTS
        x, y, z = np.meshgrid( xyz[0], xyz[1], xyz[2], indexing = "ij" ) 
        ## Indexing "xy" messes up the order!
        
        ## CONCATENATING ALL POINTS
        grid = np.concatenate( (x.reshape(1, -1),
                                y.reshape(1, -1),
                                z.reshape(1, -1)),
                                axis = 0  
                              ) ## SHAPE: 3, NUM_POINTS
        return grid
    
    ### FUNCTION TO GET DENSITIES
    @staticmethod
    def compute_gaussian_kde_pbc(grid,
                                 pos,
                                 sigma,
                                 box,
                                 cutoff_factor = 2.5):
        '''
        This function computes the Gaussian KDE values using Gaussian distributions. 
        This takes into account periodic boundary conditions
        INPUTS:
            grid: [np.array, shape = (3, num_points)]
                grid points in, x,y,z dimensions.
            pos: [np.array, shape = (N_atoms, 3)]
                positions of the water
            sigma: [float]
                standard deviation of the points
            box: [np.array, shape = 3]
                box size in x, y, z dimension
            cutoff_factor: [float]
                cutoff of standard deviations. By default, this is 2.5, which 
                is 2.5 standard deviations (~98.75% of population). Decreasing 
                this value will decrease the accuracy. Increasing this value will 
                make it difficult to compute nearest neighbors.
        OUTPUTS:
            dist: [np.array]
                normal density for the grid of points. Note that this distribution 
                is NOT normalized. It is just the exponential:
                    np.exp( -sum(delta r) / scale )
                You will need to normalize this afterwards.
        '''
            
        ## GETTING KC TREE: FINDS ALL OF DISTANCE R WITHIN X -- QUICK NN LOOK UP
        tree = cKDTree(data = grid.T, 
                       boxsize=box)
        
        ## DEFINING THE SCALE (VARIANCE)
        scale = 2. * sigma**2
        
        ## GETTING RADIUS (2.5 sigma gives you radius cutoff of 98.75%)
        d = sigma*cutoff_factor
        # 2.5 standard deviations truncation, equivalent to ~99% of the population
        # If you increased STD 
        
        ## GETTING THE INDICES LIST FOR ALL POINTS AROUND THE POSITIONS (MULTIPROCESSED)
        indlist = tree.query_ball_point(pos, r = d, n_jobs = -1)
        
        ## DEFINING RESULTS (SHAPE = NUM_POINTS)
        dist = np.zeros(grid.shape[1], dtype=float)
        
        ## LOOPING THROUGH THE LIST
        for n, ind in enumerate(indlist):
            ## GETTING DIFFERENCE OF R
            dr = grid.T[ind, :] - pos[n]
            ## POINTS GREATER THAN L / 2
            cond = np.where(dr > box / 2.)
            dr[cond] -= box[cond[1]]
            
            ## POINTS LESS THAN -L / 2
            cond = np.where(dr < -box / 2.)
            dr[cond] += box[cond[1]]
            
            ## DEFINING THE GAUSSIAN FUNCTION
            dens = np.exp(-np.sum(dr * dr, axis=1) / scale)
            dist[ind] += dens
        
        return dist
    
    ### FUNCTION TO GENERATE DENSITY FIELD
    def compute_density_field(self, grid, pos):
        '''
        The purpose of this function is to compute the density field
        INPUTS:
            grid: [np.array, shape = (3, num_points)]
                grid points in, x,y,z dimensions.
            pos: [np.array, shape = (N_atoms, 3)]
                positions of the water
                
            self.sigma: [float]
                standard deviation of the points
        OUTPUTS:
            density_field: [np.array, shape=num_grid_points]
                density field as a function of grid points
        '''
        ## GETTING DISTRIBUTION
        dist = self.compute_gaussian_kde_pbc(grid = grid,
                                             pos = pos,
                                             sigma = self.sigma,
                                             box = self.box,
                                             )
    
        ## GETTING DENSITY FIELD (NORMALIZED)
        density_field = ( 2 * np.pi * self.sigma**2 )**(-1.5) * dist
        return density_field
    
    ### FUNCTION TO FIND INDEXES
    @staticmethod
    def find_heavy_atoms_of_residues(traj, residue_list = ['HOH']):
        '''
        The purpose of this function is to find heavy atom indices. These 
        are the indexes used to generate gaussian distributions and generate
        a willard-chandler interface
        '''
        ## GETTING ATOM INDEX
        ndx_solv = np.array( [ [ atom.index for atom in residue.atoms if 'H' not in atom.name ] 
                                 for residue in traj.topology.residues if residue.name in residue_list ] ).flatten()
        
        return ndx_solv
    
    ### FUNCTION TO COMPUTE THE WILLARD-HANDLER INTERFACE
    @staticmethod
    def generate_wc_interface( density_field_reshaped, spacing, contour = 16. ):
        '''
        The purpose of this function is to compute the contour of the WC interface. 
        INPUTS:
            density_field_reshaped: [np.array, shape = (num_grid_x, num_grid_y, num_grid_y) ]
                density values as a function of grid point
            spacing: [np.array]
                spacing in the mesh grid
            contour: [float]
                c value in the WC interface. 16 would be half the bulk. 
        OUTPUTS:
            verts: [np.array, shape = (num_atoms, 3)]
                points of the marching cubes
        '''
        ## USING MARCHING CUBES
        verts, faces, normals, values = marching_cubes( density_field_reshaped, 
                                                        level = contour, 
                                                        spacing = tuple( spacing ) )        
        return verts
    
    ### FUNCTION TO RESHAPE AND GENERATE WC INTERFACE
    def get_wc_interface_points(self, density_field, contour = 16.):
        '''
        The purpose of this function is to get the wc interface. 
        INPUTS:
            density_field: [np.array, shape = (num_grid_pts)]
                density values as a function of grid point
            spacing: [np.array]
                spacing in the mesh grid
            contour: [float]
                c value in the WC interface. 16 would be half the bulk. 
                If None, then we will define the contour level to be origin.
        OUTPUTS:
            interface_points: [np.array, shape = (num_atoms, 3)]
                interface of points in x, y,z positions
        '''
        ## GETTING WC INTERFACE
        if contour is not None:
            interface_points = self.generate_wc_interface(density_field_reshaped = density_field.reshape(self.num_grid_pts),
                                                          spacing = self.spacing,
                                                          contour = contour, # contour
                                                          )
        else:
            print("Since contour is set to None, outputting interface at 0,0,0.")
            print("The None flag should be used to debug the WC interface!")
            interface_points=np.array([[0,0,0]])
        
        return interface_points
    



#%% RUN FOR TESTING PURPOSES 
if __name__ == "__main__":
    
    ## DEFINING sigma, CONTOUR, AND MESH
    sigma = WC_DEFAULTS['sigma']
    contour = WC_DEFAULTS['contour']
    ## DEFINING MESH
    mesh = WC_DEFAULTS['mesh']
    mesh = [0.1, 0.1, 0.05]
    meshsize=0.1
    mesh = [meshsize]*3
    
    #%%
    ##########################
    ### LOADING TRAJECTORY ###
    ##########################
    ## DEFINING MAIN SIMULATION
    main_sim=r"S:\np_hydrophobicity_project\simulations\191210-annealing_try4"
    ## DEFINING SIM NAME
    sim_name=r"FrozenGoldPlanar_300.00_K_dodecanethiol_10x10_CHARMM36jul2017_intffGold_Trial_1-50000_ps"
    ## DEFINING WORKING DIRECTORY
    wd = os.path.join(main_sim, sim_name)
    ## DEFINING GRO AND XTC
    gro_file = r"sam_prod-0_1000-watO_grid.gro"
    xtc_file = r"sam_prod-0_1000-watO_grid.xtc"
    ## DEFINING PATHS
    path_gro = os.path.join(wd, gro_file)
    path_xtc = os.path.join(wd, xtc_file)
    
    ## PRINTING
    print("Loading trajectory")
    print(" --> XTC file: %s"%(path_xtc) )
    print(" --> GRO file: %s"%(path_gro) )
    ## LOADING TRAJECTORY
    traj = md.load(path_xtc, top = path_gro)

    #%%
    
    ## DEFINING INTERFACE
    interface = wc_interface(traj = traj,
                             mesh = mesh,
                             sigma = sigma,
                             residue_list = ['HOH'],                         
                             )
    
    ## GETTING DENSITY FIELD
    avg_density_field = interface.compute(traj = traj[0:10])
    
    #%%
    ## GETTING WC INTERFACE
    interface_points = interface.get_wc_interface_points(density_field = avg_density_field, 
                                                         contour = contour)
    
    #%%
    
    ## PLOTTING
    interface.mlab_plot_density_field(density_field = avg_density_field,
                                      interface_points = interface_points,
                                      num_grid_pts = interface.num_grid_pts,
                                      grid = interface.grid,
                                      pos = None, 
                                      )
