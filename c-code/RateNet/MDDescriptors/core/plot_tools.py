# -*- coding: utf-8 -*-
"""
plot_tools.py
This script contains all functions that have plotting (e.g. saving figures, etc.)

USAGE: 
    import plot_tools
    import MDDescriptors.core.plot_tools as plot_tools

GLOBAL VARIABLES:
    LABELS: 
        label for plots
    LINE_STYLE: 
        dictionary for line style
    ATOM_DICT: 
        atom dictionary for plotting purposes
    COLOR_CODE_DICT:
        color code for mayavi
    
FUNCTIONS:
    set_mpl_defaults: 
        sets mpl defaults so you have publishable plots
    create_plot: 
        creates a general plot
    save_fig_png: 
        saves current figure as a png
    store_figure: 
        stores figure as any format
    cm2inch: 
        function that converts cm to inches
    create_fig_based_on_cm: 
        function that creates figure  based on input cms
    color_y_axis: 
        changes color of y-axis
    create_3d_axis_plot: 
        creates 3D axis plot
    get_cmap: 
        generates a color map for plot with a lot of colors
    plot_solute_atom_index: 
        plots solute with atom index
    plot_3d_molecule: 
        plots 3D molecule
    adjust_x_y_ticks:
        adjusts x and y ticks if necessary
    create_subplots:
        create subplots with shared axis
    plot_mayavi_nanoparticle:
        example of plotting nanoparticle in mayavi
    update_fig_size:
        update figure size
    plot_3d_points_with_scalar:
        uses mayavi to plot 3d points with scalar values
    create_cmap_with_white_zero:
        function that creates a cmap with white areas around the zero values
    
Authors:
    Alex K. Chew (alexkchew@gmail.com)

** UPDATES **
20180511 - AKC - added color_y_axis function
20180622 - AKC - creates 3D axis plot
20181003 - AKC - adding get_cmap function to get colors

USAGE:
    import MDDescriptors.core.plot_tools as plot_funcs
    ## SETTING DEFAULTS
    plot_funcs.set_mpl_defaults()
"""
## IMPORTING IMPORTANT MODULES
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

### GLOBAL VARIABLES
ATOM_RESOLUTION = 1000 # 50 # Resolution of atoms

## DEFINING DEFAULT FIGURE SIZES
FIGURE_SIZES_DICT_CM = {
        '1_col': (8.55, 8.55),
        '1_col_landscape': (8.55, 6.4125), # 4:3 ratio
        '1_col_portrait' : (6.4125, 8.55),
        '2_col': (17.1, 17.1), 
        }

## DEFINING GLOBAL PLOT PROPERTIES
LABELS = {
            'fontname': "Arial",
            'fontsize': 16
            }
## DEFINING LINE STYLE
LINE_STYLE={
            "linewidth": 1.6, # width of lines
            }

### DEFINING ATOM AND BOND TYPES
ATOM_DICT = { 
        'C': {'color': 'black','size':4},
        'O': {'color': 'red','size':4},
        'F': {'color': 'cyan','size':4},
        'H': {'color': 'gray','size':3},
        'N': {'color': 'blue','size':1},
        'Au': {'color': 'orange','size':6},
        'VS': {'color': 'orange','size':6}, # Au virtual site
        'S': {'color': 'yellow','size':5},
        'P': {'color': 'white','size':5},
        } 

### DEFINING COLOR CODES
COLOR_CODE_DICT = {
        'yellow' : [1, 1, 0],
        'orange' : [1.0, 0.5, 0.25],
        'magenta': [1, 0, 1],
        'cyan'   : [0, 1, 1],
        'red'    : [1, 0, 0],
        'green'  : [0, 1, 0],
        'blue'   : [0, 0, 1],
        'white'  : [1, 1, 1],
        'black'  : [0, 0, 0],
        'gray'   : [0.5, 0.5, 0.5],
        }

### FUNCTION TO SET MPL DEFAULTS
def set_mpl_defaults():
    ''' 
    This function sets all default parameters 
    # https://matplotlib.org/tutorials/introductory/customizing.html
    '''
    import matplotlib as mpl
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['axes.linewidth'] = 1
    mpl.rcParams['axes.labelsize'] = 10
    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelsize'] = 8
    ## EDITING TICKS
    mpl.rcParams['xtick.major.width'] = 1.0
    mpl.rcParams['ytick.major.width'] = 1.0
    
    ## FONT SIZE
    mpl.rcParams['legend.fontsize'] = 8
    
    ## CHANGING FONT
    mpl.rcParams['font.sans-serif'] = "Arial"
    mpl.rcParams['font.family'] = "sans-serif"
    ## DEFINING FONT
    font = {'size'   : 10}
    mpl.rc('font', **font)
    return

### FUNCTION TO CREATE PLOT
def create_plot():
    '''
    The purpose of this function is to generate a figure.
    Inputs:
        fontSize: Size of font for x and y labels
        fontName: Name of the font
    Output:
        fig: Figure to print
        ax: Axes to plot on
    '''
    ## CREATING PLOT
    fig = plt.figure() 
    ax = fig.add_subplot(111)
    return fig, ax

## FUNCTION TO CONVERT FIGURE SIZE
def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

## FUNCTION TO CREATE FIG BASED ON CM
def create_fig_based_on_cm(fig_size_cm = (16.8, 16.8)):
    ''' 
    The purpose of this function is to generate a figure based on centimeters 
    INPUTS:
        fig_size_cm: [tuple]
            figure size in centimeters 
    OUTPUTS:
        fig, ax: 
            figure and axis
    '''
    ## FINDING FIGURE SIZE
    if fig_size_cm is not None:
        figsize=cm2inch( *fig_size_cm )
    ## CREATING FIGURE
    fig = plt.figure(figsize = figsize) 
    ax = fig.add_subplot(111)
    return fig, ax

### FUNCTION TO GENERATE AXIS
def create_subplots(num_rows = 2, 
                    num_cols = 1, 
                    figsize = FIGURE_SIZES_DICT_CM['1_col_landscape'],
                    wspace = 0,
                    hspace = 0):
    '''
    The purpose of this function is to generate subplots with a shared axis. 
    INPUTS:
        num_rows: [int]
            number of rows
        num_cols: [int]
            number of columns
        figsize: [tuple, size=2]
            figure size in height x width
        wspace: [float]
            width space across columns
        hspace: [float]
            height space across rows
    OUTPUTS:
        fig, axs:
            figure and axis
    '''

    ## GETTING TOTAL SUB PLOTS
    total_subplots = num_rows * num_cols

    ## DIVIDING THE FIGURE SIZE TO MATCH
    figsize = (figsize[0] / total_subplots, figsize[1] / total_subplots)
    
    ## GENERATING FIGURE
    fig = plt.figure(figsize = figsize)
    gs1 = gridspec.GridSpec(num_rows, num_cols)
    
    ## SPACING BETWEEN AXIS
    gs1.update(wspace=0, hspace=0)
    
    ## GENERATING AXS
    axs = [ plt.subplot(gs1[i]) for i in range(total_subplots) ]
    
    return fig, axs



### FUNCTION TO UPDATE FIG SIZE
def update_fig_size(fig,
                    fig_size_cm = (6,5),
                    tight_layout = True):
    '''
    This function updates the figure sizes based on input cm in x, y
    INPUTS:
        fig: [obj]
            figure object
        fig_size_cm: [tuple]
            figure size in width and height
        tight_layout: [logical]
            True if you want tight layout
    OUTPUTS:
        fig: [obj]
            updated figure
    '''
    ## GETTING FIGURE SIZE IN INCHES    
    fig_size_inches = cm2inch(fig_size_cm)    
    ## CHANGING FIGURE SIZE
    fig.set_size_inches(*fig_size_inches, forward=True)
    if tight_layout is True:
        fig.tight_layout()
    return fig

### FUNCTION TO SAVE FIGURE AS A PNG
def save_fig_png(fig, label, save_fig = True, dpi=600, bbox_inches = 'tight'):
    '''
    The purpose of this function is to save figure as a png
    INPUTS:
        fig: Figure
        label: The label you want to save your figure in
        save_fig: True or False. If True, then save the figure
        dpi: [OPTIONAL, default = 600] resolution of image
        bbox_inches: [OPTIONAL, default='tight'] how tight you want your image to be
    OUTPUTS:
        png figure
    '''
    if save_fig is True:
        label_png = label + '.png'
        ## SAVING FIGURE
        fig.savefig( label_png, format='png', dpi=dpi, bbox_inches=bbox_inches)
        ## PRINTING
        print("EXPORTING TO: %s"%(label_png))
    return

### FUNCTION THAT DEALS WITH SAVING FIGURE
def store_figure(fig, path, fig_extension = 'png', save_fig=False, dpi=1200, bbox_inches = 'tight'):
    '''
    The purpose of this function is to store a figure.
    INPUTS:
        fig: [object]
            figure object
        path: [str]
            path to location you want to save the figure (without extension)
        fig_extension:
    OUTPUTS:
        void
    '''
    ## STORING FIGURE
    if save_fig is True:
        ## DEFINING FIGURE NAME
        fig_name =  path + '.' + fig_extension
        print("Printing figure: %s"%(fig_name) )
        fig.savefig( fig_name, 
                     format=fig_extension, 
                     dpi = dpi,    
                     bbox_inches = bbox_inches,
                     )
    return


### FUNCTION TO CHANGE THE COLOR OF Y AXIS
def color_y_axis(ax, color):
    """Color your axes."""
    for t in ax.get_yticklabels():
        t.set_color(color)
    return None

### FUNCTION TO CREATE 3D PLOTS
def create_3d_axis_plot():
    '''
    The purpose of this function is to create a 3D axis plot
    INPUTS:
        void
    OUTPUTS:
        fig, ax: figure and axis of the plot
    '''
    ## IMPORTING TOOLS
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D # For 3D axes
    ## CREATING FIGURE
    fig = plt.figure(); ax = fig.add_subplot(111, projection='3d') # , aspect='equal'
    return fig, ax

### FUNCTION TO CREATE 3D PLOTS WITH XYZ COORDINATES
def create_3d_axis_plot_labels(labels=['x', 'y', 'z']):
    '''
    The purpose of this function is to create a 3D axis plot with x, y, z
    INPUTS:
        labels: [list, size=3] list of labels for the 3D axis
    OUTPUTS:
        fig, ax
    '''
    ## CREATING FIGURE
    fig, ax = create_3d_axis_plot()
    
    ## SETTING LABELS
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    
    return fig, ax
    
### FUNCTION TO GET CMAP
def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.
    This function is useful to generate colors between red and purple without having to specify the specific colors
    USAGE:
        ## GENERATE CMAP
        cmap = get_cmap(  len(self_assembly_coord.gold_facet_groups) )
        ## SPECIFYING THE COLOR WITHIN A FOR LOOP
        for ...
            current_group_color = cmap(idx) # colors[idx]
            run plotting functions
    '''
    ## IMPORTING FUNCTIONS
    import matplotlib.pyplot as plt
    return plt.cm.get_cmap(name, n + 1)
    
### FUNCTION TO VISUALIZE SOLUTE ATOM INDEX
def plot_solute_atom_index( positions, atom_elements, atom_names, bond_atomnames = None, atom_dict = ATOM_DICT ):
    '''
    The purpose of this function is to plot the solute atom index
    INPUTS:
        positions: [np.array, shape=(N_atoms, 3)]
            positions in cartesian coordinates
        atom_elements: [np.array, shape=(N_atoms, 1)]
            atom elements of each position
        atom_names: [list, shape=(N_atoms, 1)]
            atom names, e.g. 'O1', etc.
        bond_atomnames: [list, optional]
            bond atomnames 
        atom_dict: [dict]
            dictionary of color and size for the atom
    OUTPUTS:
        plot with solute atom index
        fig, ax for plot
    '''
    ## CREATING FIGURE
    fig, ax = create_3d_axis_plot_labels()
    
    ## FINDING UNIQUE ATOM LIST
    unique_atom_list = list(set(atom_elements))
    
    ## LOOPING THROUGH EACH UNIQUE ATOM LIST
    for current_atom_type in unique_atom_list:
        ## FINDING ALL ELEMENTS THAT MATCH THE CRITERIA
        index_matched = np.array([ index for index, atom in enumerate(atom_elements) if atom == current_atom_type ])
        ## DEFINING CURRENT ATOM DICT
        current_atom_dict = { your_key: atom_dict[current_atom_type][your_key] for your_key in ['color'] }
        ## ADDING TO PLOT
        ax.scatter( positions[index_matched, 0], positions[index_matched, 1], positions[index_matched, 2], s = 50, **current_atom_dict )
    ## ADDING ANNOTATION
    for idx, each_label in enumerate(atom_names):
        ax.text( positions[idx][0], positions[idx][1], positions[idx][2], each_label )
        
    ## ADDING BONDS
    if bond_atomnames is not None:
        for current_bond in bond_atomnames:
            ## FINDING COORDINATES FOR EACH ATOM
            bond_coordinates = np.array( [ positions[index] for index in [ atom_names.index(eachAtom) for eachAtom in current_bond ] ] )
            ## PLOTTING
            ax.plot( bond_coordinates[:,0], bond_coordinates[:,1], bond_coordinates[:,2], color='k' )
            
    return fig, ax


### FUNCTION TO PLOT ATOM AND BONDS USING MATPLOTLIB
def plot_3d_molecule( atom_positions, atom_symbols, bonds, atom_names = None,
                         ATOM_PLOT_INFO={
                                            'H':
                                                { "color"   :"gray",
                                                  "s"       : 50,
                                                        },
                                            'C':
                                                { "color": "black",
                                                  "s"       : 100,
                                                 },
                                            'O':
                                                { "color": "red",
                                                  "s"       : 100,
                                                 },
                                            },
                        BOND_PLOT_INFO = {
                                'color': 'black',
                                'linestyle': '-',
                                'linewidth': 2,
                                }
            ):
    '''
    The purpose of this function is to plot in 3D the molecule of your interest.
    INPUTS:
        atom_positions: [np.array, shape=(num_atoms, 3)]
            atomic positions in Cartesian coordinates
        atom_symbols: [np.array, shape=(num_atoms, 1)]
            atomic symbols (e.g. 'H', 'O', etc.)
        bonds: [np.array, shape=(num_bonds, 2)]
            bonds between atoms , e.g. [1,2] means that atoms 1 and 2 are bonded
        atom_names: [np.array, shape=(num_atoms), default=None]
            Optional, will print atom names if you supply it
        ATOM_PLOT_INFO: [dict]
            dictionary for atom types
        BOND_PLOT_INFO: [dict]
            dictionary for bond information type
    OUTPUTS:
        fig, ax:
            figure and axis
    '''
    ## CREATING 3D AXIS
    fig, ax = create_3d_axis_plot_labels(labels=['x', 'y', 'z'])
    
    ## GENERATING SIZES
    sizes = [ ATOM_PLOT_INFO[each_symbol]['s'] for each_symbol in atom_symbols ]
    
    ## GENERATING ARRAY OF COLORS
    color_array = [ ATOM_PLOT_INFO[each_symbol]['color'] for each_symbol in atom_symbols ]
    
    ## SETTING X, Y, Z
    X = atom_positions[:,0]
    Y = atom_positions[:,1]
    Z = atom_positions[:,2]
    
    ## FINDING LIMITS
    minX, maxX = np.min(X), np.max(X)
    minY, maxY = np.min(Y), np.max(Y)
    minZ, maxZ = np.min(Z), np.max(Z)
    
    ## FINDING LARGEST INTERVAL / 2
    largest_interval = np.max( [maxX - minX,
                               maxY - minY,
                               maxZ - minZ,]
                              ) / 2.0
    
    ## CREATING NEW AXIS LIMITS
    xlimits = [ np.mean( [minX,maxX] ) - largest_interval, np.mean( [minX,maxX] ) + largest_interval ]
    ylimits = [ np.mean( [minY,maxY] ) - largest_interval, np.mean( [minY,maxY] ) + largest_interval ]
    zlimits = [ np.mean( [minZ,maxZ] ) - largest_interval, np.mean( [minZ,maxZ]) + largest_interval ]
    
    
    ## SETTING AXIS LIMITS
    ax.set_xlim( xlimits  )
    ax.set_ylim( ylimits  )
    ax.set_zlim( zlimits  )
    
    ## PLOTTING ATOM
    ax.scatter( X, Y, Z, s = sizes , color= color_array  )  # scatter = 
    
    ## DRAWLING BONDS FOR LINES
    for current_bond in bonds:
        ## FINDING COORDINATES
        atom_coord_1 = atom_positions[current_bond[0]] 
        atom_coord_2 = atom_positions[current_bond[1]] 
        ## PLOTTING BOND
        ax.plot( [ atom_coord_1[0], atom_coord_2[0] ] , # x
                 [ atom_coord_1[1], atom_coord_2[1] ], # y
                 [ atom_coord_1[2], atom_coord_2[2] ], # z
                 **BOND_PLOT_INFO
                 )
#            
    ## ADDING LABELS
    if atom_names is not None:
        for x,y,z,i in zip(X,Y,Z,atom_names):
            ax.text(x,y,z,i)
    
    return fig, ax

### FUNCTION TO ADJUST X AND Y TICK
def adjust_x_y_ticks(ax, 
                     x_axis_labels = None,
                     y_axis_labels = None,
                     ax_x_lims = None, 
                     ax_y_lims = None,
                     ):
    '''
    The purpose of this function is to adjust x and y ticks based on limits 
    INPUTS:
        ax: [obj]
            axis
        x_axis_labels: [tuple, 3]
            low, max, increments for x-axis
        y_axis_labels: [tuple, 3]
            low, max, increments for y-axis
        ax_x_lims: [tuple, 2]
            x axis limits
        ax_y_lims: [tuple, 2]
            y axis limits
    OUTPUTS:
        void
    '''
    ## SETTING X AND Y LIMS
    if ax_x_lims is not None:
        ax.set_xlim(*ax_x_lims)
    if ax_y_lims is not None:
        ax.set_ylim(*ax_y_lims)
    
    ## SETTING X TICKS AND Y TICKS
    if x_axis_labels is not None:
        ax.set_xticks(np.arange(x_axis_labels[0], x_axis_labels[1] + x_axis_labels[2], x_axis_labels[2]))
    if y_axis_labels is not None:
        ax.set_yticks(np.arange(y_axis_labels[0], y_axis_labels[1] + y_axis_labels[2], y_axis_labels[2]))
    return ax

### FUNCTION TO PLOT OVERLAPPING GRID POINTS
def plot_intersecting_points(grid,
                             avg_neighbor_array = None,
                             ):
    '''
    The purpose of this function is to plot the intersecting points and 
    the corresponding ligands.
    INPUTS:
        traj: [obj]
            trajectory array
        grid: [np.array]
            array of grid points
        avg_neighbor_array: [np.array]
            number of neighbors for array
        
    OUTPUTS:
        
    '''
    ## IMPORTING MLAB
    from mayavi import mlab
    ## CLOSING ALL MLAB FIGURES
    mlab.clf()
    # PLOTTING WITH MAYAVI
    figure = mlab.figure('Scatter plot',
                         bgcolor = (.5, .5, .5))

#    if avg_neighbor_array is None:
#        avg_neighbor_array = np.ones(len(grid))
#
    if avg_neighbor_array is None:
        points = mlab.points3d(grid[:,0],
                               grid[:,1],
                               grid[:,2],
                               figure = figure,
                               )
    else:

        ## PLOTTING GRID POINTS
        points = mlab.points3d(grid[:,0],
                               grid[:,1],
                               grid[:,2],
                               avg_neighbor_array,
                               figure = figure,
                               )
    
    ## ADDING COLOR  BAR
    mlab.colorbar(object = points)
    
    return figure

### FUNCTION TO ADD ATOMS TO STRUCTURE
def plot_mayavi_atoms(traj,
                      atom_index,
                      frame = 0,
                      figure = None,
                      dict_atoms = ATOM_DICT,
                      dict_colors = COLOR_CODE_DICT,
                      desired_atom_colors = None):
    '''
    The purpose of this function is to add mayavi atoms to the figure
    INPUTS:
        traj: [obj]
            trajectory array
        atom_index: [int]
            atom indices that you care about
        frame: [int]
            frame you are interested in printing
        desired_atom_colors: [str]
            desired atom colors of all the atoms, e.g. 'red'
    OUTPUTS:
        figure: [obj]
            mayavi figure object
    '''
    ## IMPORTING MLAB
    from mayavi import mlab
    
    ## GETTING NUMPY
    atom_index = np.array(atom_index)
    
    ## FINDING ALL THE ELEMENTS IN TRAJECTORY
    element_list = [ traj.topology.atom(each_atom).element.symbol for each_atom in atom_index ]
    ##  GETTING UNIQUE ATOM LIST
    unique_atom_list = list(set(element_list))
    
    ## DEFINING FRAME
    ### DRAWING ATOMS
    for current_atom_type in unique_atom_list:
        ## PRINTING
        print("Plotting atom type: %s"%(current_atom_type) )
        ## FINDING ALL ELEMENTS THAT MATCH THE CRITERIA
        index_matched = np.array([ index for index, atom in enumerate(element_list) if atom == current_atom_type ])
        
        ## GETTING COLOR
        marker_dict = dict_atoms[current_atom_type].copy()
        
        ## GETTING COLOR
        if desired_atom_colors is not None:
            marker_dict['color'] = tuple(dict_colors[desired_atom_colors])
        else:
            marker_dict['color'] = tuple(dict_colors[marker_dict['color']])
        
        ### CREATING SHAPES
        shape_atom_size = np.ones( len(index_matched))
        
        ### REMOVING SIZE FROM DICTIONARY
        marker_dict.pop('size', None)
        
        ## DEFINING POSITIONS
        x_pos = traj.xyz[frame, atom_index[index_matched], 0]
        y_pos = traj.xyz[frame, atom_index[index_matched], 1]
        z_pos = traj.xyz[frame, atom_index[index_matched], 2]
        
        ## PLOTTING POINTS
        mlab.points3d(
                    x_pos, 
                    y_pos, 
                    z_pos,
                    shape_atom_size,
                    figure = figure,
                    scale_factor=.25, 
                    opacity=1.0,
                    mode = "sphere", # "sphere",
                    **marker_dict,
                     )
    return figure

### FUNCTION TO PLOT MAYAVI NANOPARTICLE
def plot_mayavi_nanoparticle(traj_data,
                             frame = 0 ):
    '''
    This function plots mayavi nanoparticle.
    INPUTS:
        traj_data: [obj]
            trajectory object
        frame: [int]
            frame you want to plot
    OUTPUTS:
        fig:
            figure for mayavi nanoparticle
    '''
    from MDDescriptors.application.nanoparticle.core.find_ligand_residue_names import get_atom_indices_of_ligands_in_traj
    
    ## GETTING ATOM INDICES AND LIGAND NAME
    ligand_names, atom_index = get_atom_indices_of_ligands_in_traj( traj = traj_data.traj )
    
    ## GETTING GOLD INDEX    
    au_index = [atom.index for atom in traj_data.topology.atoms if atom.name == 'Au' or atom.name == 'BAu']
    
    ## FIGURE FROM 
    fig = plot_mayavi_atoms(traj = traj_data.traj,
                              atom_index = atom_index,
                              frame = frame,
                              figure = None,
                              dict_atoms = ATOM_DICT,
                              dict_colors = COLOR_CODE_DICT)
    
    ## PLOTTING GOLD FIGURE
    fig = plot_mayavi_atoms(traj = traj_data.traj,
                                       atom_index = au_index,
                                       frame = frame,
                                       figure = fig,
                                       dict_atoms = ATOM_DICT,
                                       dict_colors = COLOR_CODE_DICT)
    return fig

### FUNCTION TO PLOT POINTS
def plot_3d_points_with_scalar(xyz_coordinates, 
                               scalar_array = None,
                               points_dict={
                                       'scale_factor': 0.5,
                                       'scale_mode': 'none',
                                       'opacity': 1.0,
                                       'mode': 'sphere',
                                       'colormap': 'blue-red',
                                       'vmin': 9,
                                       'vmax': 11,
                                       },
                                figure = None):
    '''
    The purpose of this function is to plot the 3d set of points with scalar
    values to indicate differences between the values
    INPUTS:
        xyz_coordinates: [np.array]
            xyz coordinate to plot
        scalar_array: [np.array]
            scalar values for each coordinate. If None, then we will assume 
            all the same colors
        points_dict: [dict]
            dicitonary for the scalars
        figure: [obj, default = None]
            mayavi figure. If None, we will create a new figure
    OUTPUTS:
        figure: [obj]
            mayavi figure
    '''
    ## IMPORTING MLAB
    from mayavi import mlab
    
    if scalar_array is None:
        scalar_array = np.ones(len(xyz_coordinates))

    if figure is None:
        ## CLOSING MLAB
        try:
            mlab.close()
        except AttributeError:
            pass
        ## CREATING FIGURE
        figure = mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
    
    ## PLOTTING POINTS
    mlab.points3d(
                    xyz_coordinates[:,0], 
                    xyz_coordinates[:,1], 
                    xyz_coordinates[:,2], 
                    scalar_array,
                    figure = figure,
                    **points_dict
                    )
    return figure

### FUNCTION TO CREATE NEW MAP FOR ZERO
def create_cmap_with_white_zero(cmap = plt.cm.jet,
                                n = 100,
                                perc_zero = 0.10):
    '''
    This function creates a cmap with white around the zero values. The 
    idea is to generate color maps that are white around zero, so you don't 
    see dark colors (e.g. found in jet)
    INPUTS:
        cmap: [obj]
            cmap object
        n: [int]
            number of points to use for the cmap. The higher this value is, 
            the greater the output cmap resolution. 
        perc_zero: [float]
            value between 0 and 1, percentage of the points that you want to be 
            zero. For instance, if this is 0.10, this means that the lower 
            10% of the color map will be set to zero.
    OUTPUTS:
        tmap: [obj]
            new map object to use for color mapping
    '''
    import matplotlib
    
    ## FINDING MAX
    total_pts = n / (1 - perc_zero) # M = N / (1-r)
    
    ## FINDING TOTAL POINTS FOR ZERO
    n_zero_pts= int(total_pts * perc_zero)
    
    ## FINDING CMAP
    current_colors = cmap(np.linspace(0, 1, n))
    
    ## GETTING WHITES
    white_colors = np.ones((n_zero_pts, 4))
    
    ## STACKING
    colors = np.vstack((white_colors, current_colors))
    
    ## GETTING TMAP
    tmap = matplotlib.colors.LinearSegmentedColormap.from_list('map_white', colors)
    return tmap
