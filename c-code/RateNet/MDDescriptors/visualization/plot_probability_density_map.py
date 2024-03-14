# -*- coding: utf-8 -*-
"""
plot_probability_density_map.py
The purpose of this script is to plot the probability density maps given the data from "probability_density_map.py"



Created on: 03/07/2018

Author(s):
    Alex K. Chew (alexkchew@gmail.com)

UPDATES:
    2018-03-28: Changed probability density mapping to remove all buttons (unnecessary)
    2018-04-24: Added functionality for probability density maps to take multiple inputs for vmin, vmax, contours, etc. This is useful for multiple solvent systems, which can have differences in magnitude for the contours.


"""

### IMPORTING MAYAVI -- NOTE SYS required to prevent mayavi from removing all print commands
# REFERENCE: https://github.com/enthought/mayavi/issues/503
import numpy as np

### FIXING ENVIRONMENTAL VALUES FOR MAYAVI
import os
os.environ["QT_API"] ="pyqt5"  #  "pyqt" #  'pyside2' # 

import sys
stream = sys.stdout
from mayavi import mlab
# from mayavi.core.ui.api import MlabSceneModel, SceneEditor # Used to create scenes
# from traitsui.api import View, Item, HSplit, Group
# from traits.api import HasTraits, Instance, Button, on_trait_change # Used for buttons
sys.stdout = stream

## GLOBAL VARIABES
from MDDescriptors.core.plot_tools import ATOM_DICT

### IMPORTING CLASS THAT RAN THIS
from MDDescriptors.visualization.probability_density_map import calc_prob_density_map
import MDDescriptors.core.plot_tools as plot_tools
### GETTING SCREENSIZE
import ctypes
user32 = ctypes.windll.user32
SCREENSIZE = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

## IMPORTING LOADING FUNCTIONS
from MDDescriptors.traj_tools.multi_traj import load_multi_traj_pickle, multi_traj_analysis

## LOADING PICKLE INFORMATION
from MDBuilder.core.pickle_funcs import store_class_pickle, load_class_pickle

### GLOBAL VARIABLES
ATOM_RESOLUTION = 1000 # 50 # Resolution of atoms
ANIMATION_DELAY = 500 # Delay for animations

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

#### BONDING PROPERTIES
### DICTIONARY FOR BONDS
BOND_DICT = {
        'color': (0, 0, 0),
        'line_width': 20,
        }

### INPUT FOR MAPPING DETAILS
MAP_INPUT ={ 'COLOR_CODE_DICT': COLOR_CODE_DICT,
             'ATOM_DICT'     : ATOM_DICT,
             'BOND_DICT'     : BOND_DICT,
           }

###
PRINT_FIGURE_INFO = {
                        'size': (1920,1200), # Size of the image in pixels width x height
                    }


#######################################################
### CLASS FUNCTION TO PLOT PROBABILITY DENSITY MAPS ###
#######################################################
class plot_prob_density_map: # HasTraits
    '''
    The purpose of this class is to take the data from the "prob_density" class and plot it accordingly. 
    INPUTS:
        prob_density: Probability density from "calc_prob_density_map" class
        iso_map_dict: [dict] dictionary of the types of isomaps you want. Each entry should have "contours", "vmin", and "vmax". Tentatively, we have two keys within dictionary: "HOH" and "COSOLVENT".
            The reason for this dictionary is because some isosurfaces vary in magnitude. (e.g. water vs. cosolvent)
        combine_plot: True if you want to combine the solvent plots into a single one
        want_water_only: True if you only want the water plot
        kwargs: other arguments
            'color_code_dict': dictionary for the color codes (e.g. white is [1, 1, 1])
            'atom_dict': How you want your atoms to look like
            'bond_dict': How you want your bonds to look like
        opacity: [float]
            opacity of the isosurfaces
        skip_bonds: [logical]
            True if you want to skip bond drawing
    OUTPUTS:
        ## DICTIONARY ITEMS
            self.dict_atoms: Dictionary containing all atom plotting information
            self.dict_bonds: Dictionary containing all bond plotting information
            self.dict_colors: Dictionary containing all the colors and their relationships (since mayavi does not have color compatibility)
    BUTTONS:
        button1_inc // button2_inc : Increases contour level
        button1_dec // button2_dec : Decreases contour levels
        button1_print // button2_print: Prints the image
        
    FUNCTIONS:
        _setup1 // _setup2: Activated once scene is created
        plot_mayavi_solute_structure: plots structure of the solute
        main_isosurface_plot: main function that plots the isosurfaces
            calc_contours_prob_dist: calculates the contours required for a probability distribution function (using linspace)
            plot_isosurface: Plots the isosurface
            add_color_bar_isosurface: Adds the color bar for the isosurface
            add_outline: Adds black outline to the figure
            add_planar_contour: Adds a planar contour that can help with visualization
            edit_mayavi_fig: Changes the background coloring
    ACTIVE FUNCTIONS:
        plot_solute_index: plots solute index in Cartesian coordinates
  
    '''
    ### INITIALIZING
    def __init__(self, 
                 prob_density, 
                 opacity = 0.3,
                 iso_map_dict = None, 
                 combine_plot=False, 
                 want_color_bar = True,
                 want_water_only=False,
                 skip_bonds = False,
                 **MAP_INPUT):
        ### STORING INFORMATION
        self.opacity = opacity
        self.dict_atoms = MAP_INPUT['ATOM_DICT']
        self.dict_bonds = MAP_INPUT['BOND_DICT']
        self.dict_colors = MAP_INPUT['COLOR_CODE_DICT']
        self.prob_density = prob_density
        self.iso_map_dict = iso_map_dict
        self.want_color_bar = want_color_bar
        self.skip_bonds = skip_bonds
        
        ## COLOR BAR LOCATIONS
        self.color_bar_locations=[
                                np.array( [0.10, 0.15 ]),
                                np.array( [0.85, 0.15 ]),
                                ]
        self.color_map_variations=[
                            'blue-red',
                            'Reds',
                            'Blues'
                            ,]
        
        ### FINDING NUMBER OF SCENES TO SET UP
        self.total_figures = len(prob_density.solvent_name)
        
        ## TURNING RUN FIGURE ON
        run_figure = True
        
        ### STORING SCENES
        self.figures = [] #  [ mlab.figure() for each_figure in range(self.total_figures)]
        
        if want_water_only is True:
            self.total_figures = 1 # one figure for water
        
        if combine_plot is False: # or self.total_figures == 1:
        
            ## LOOPING THROUGH EACH POSSIBLE FIGURE
            for index in range(self.total_figures):
                if want_water_only is True:
                    if self.prob_density.solvent_name[index] == 'HOH':
                        run_figure = True
                    else:
                        run_figure = False
                
                if run_figure is True:
                    ## DEFINING ISOSURFACE PARAMETERS
                    if self.prob_density.solvent_name[index] == 'HOH':
                        self.contours, vmin, vmax, nb_labels = self.iso_map_dict['HOH']['contours'], self.iso_map_dict['HOH']['vmin'], \
                                                                self.iso_map_dict['HOH']['vmax'], self.iso_map_dict['HOH']['nb_labels'],
                    else:
                        self.contours, vmin, vmax, nb_labels= self.iso_map_dict['COSOLVENT']['contours'], self.iso_map_dict['COSOLVENT']['vmin'], \
                                                                self.iso_map_dict['COSOLVENT']['vmax'], self.iso_map_dict['COSOLVENT']['nb_labels'],
                    
                    ### CREATING FIGURE
                    if self.total_figures > 1:
                        each_figure = mlab.figure(size=(SCREENSIZE[0]/float(self.total_figures),SCREENSIZE[1]) )
                    else:
                        each_figure = mlab.figure(size=(SCREENSIZE[0]/2.0,SCREENSIZE[1]) )
                    ### CHANGING THE COLORS
                    self.edit_mayavi_fig(each_figure)
                    
                    ### DEFINING PROBABILITY DENSITY DISTRIBUTION
                    PROB_DIST = self.prob_density.prob_dist[index]
                    
                    ### PLOTTING EACH SURFACE
                    self.main_isosurface_plot(each_figure, PROB_DIST, vmin = vmin, vmax = vmax, nb_labels = nb_labels)
                    
                    ### ADDING TITLE
                    self.title_text = mlab.title("%s--%s"%(self.prob_density.solute_name, self.prob_density.solvent_name[index]), figure=each_figure )
                    
                    ### SAVING FIGURE CLASS
                    self.figures.append(each_figure)
        ## COMBINING PLOTS
        else:
            ## FIXING COLOR MAP BY REMOVING THE FIRST INDEX (BLUE RED)
            try:
                self.color_map_variations.remove('blue-red')
            except:
                pass
            ## CREATING FIGURE
            each_figure = mlab.figure(size=(SCREENSIZE[0],SCREENSIZE[1]) )
            
            ### CHANGING THE COLORS
            self.edit_mayavi_fig(each_figure)
            
            ### DEFINING PROBABILITY DENSITY DISTRIBUTION
            PROB_DIST_FULL = self.prob_density.prob_dist
            
            ## LOOPING THROUGH EACH PROBABILITY DENSITY DISTRIBUTION
            for index, PROB_DIST in enumerate(PROB_DIST_FULL):
                if self.prob_density.solvent_name[index] == 'HOH':
                    self.contours, vmin, vmax, nb_labels = self.iso_map_dict['HOH']['contours'], self.iso_map_dict['HOH']['vmin'], \
                                                            self.iso_map_dict['HOH']['vmax'], self.iso_map_dict['HOH']['nb_labels'],
                else:
                    self.contours, vmin, vmax, nb_labels= self.iso_map_dict['COSOLVENT']['contours'], self.iso_map_dict['COSOLVENT']['vmin'], \
                                                            self.iso_map_dict['COSOLVENT']['vmax'], self.iso_map_dict['COSOLVENT']['nb_labels'],
                ### PLOTTING EACH SURFACE
                self.main_isosurface_plot(each_figure, PROB_DIST, index, vmin = vmin, vmax = vmax,  nb_labels = nb_labels)
            ## STORING IFGURE
            self.figures.append(each_figure)
                
            
    ### DEFINING MAIN ISOSURFACE PLOT
    def main_isosurface_plot(self, figure, PROB_DIST, index = 0, contour_level=50, vmin = 1, vmax = 3, nb_labels = 10):
        '''
        The purpose of this plot is to plot everything with the current scene
        INPUTS:
            scene: scene for your figure
            vmin: [float] minimum value for the color bar
            vmax: [float] maximum value for the color bars
            nb_labels: [int] number of labels for the color bar
            PROB_DIST: Probability distribution
        # REFERENCE: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html#mayavi.mlab.contour3d
        # CASE STUDIES FOR VISUALIZATION IN 3D: http://docs.enthought.com/mayavi/mayavi/mlab_case_studies.html
        # COLOR MAP REF: http://docs.enthought.com/mayavi/mayavi/mlab_changing_object_looks.html
        '''
        print("WITHIN MAIN ISOSURFACE PLOT")
        ### CLEARING FIGURE
        # mlab.clf(figure=scene.mayavi_scene)     
        if index == 0:
            ### PLOTTING THE SOLUTE
            self.plot_mayavi_solute_structure(figure)
        ## FINDING CONTOURS
        contours = self.calc_contours_prob_dist(PROB_DIST, contour_level)                
        ## PLOTTING ISOSURFACE
        iso_obj = self.plot_isosurface(PROB_DIST, contours, figure, 
                                       opacity = self.opacity,
                                       index = index, 
                                       vmin = vmin, 
                                       vmax = vmax)
        
        ## ADDING COLOR  BAR
        if self.want_color_bar == True:
            self.color_bar_obj = self.add_color_bar_isosurface(iso_obj, index = index, nb_labels = nb_labels)
        
        ## ADDING OUTLINE
        self.outline_obj = self.add_outline()
        ## ADDING CONTOURS
        self.planar_contour_obj = self.add_planar_contour(PROB_DIST=PROB_DIST,plane_orientation='x_axes',slice_index=0)
        
        ## TURNING OFF VISISBILITIES
        self.planar_contour_obj.visible=False
        self.outline_obj.visible=False
        ## TURNING PARALLEL PROJECTION
        # figure.scene.parallel_projection = True
        

        return # iso_obj, color_bar_obj, outline_obj, planar_contour_obj
    
    ### FUNCTION TO PLOT THE SOLUTE STRUCTURE
    def plot_mayavi_solute_structure(self, scene):
        '''
        The purpose of this function is to plot the solute structure using the mayavi module
        INPUTS:
            self: class object
            scene: which scene to populate
        OUTPUTS:
            PLOT OF THE SOLUTE
        '''
        ### PRINTING
        print("*** Plotting solute ***")
        unique_atom_list = list(set(self.prob_density.solute_atom_elements))
        
        ### DRAWING ATOMS
        for current_atom_type in unique_atom_list:
            ## PRINTING
            print("PLOTTING ATOMTYPE: %s"%(current_atom_type))
            ## FINDING ALL ELEMENTS THAT MATCH THE CRITERIA
            index_matched = [ index for index, atom in enumerate(self.prob_density.solute_atom_elements) if atom == current_atom_type ]
            ## FINDING DICTIONARY ELEMENT
            try:
                marker_dict = self.dict_atoms[current_atom_type].copy() # COPYING THE DICTIONARY (PREVENTS OVERWRITING)
                ### CHECK IF THE COLOR IS A STRING
                if type(marker_dict['color']) is str:
                    marker_dict['color'] = tuple( self.dict_colors[marker_dict['color']] )
                ### CREATING SHAPES
                shape_atom_size = np.ones( len(index_matched))* marker_dict['size']
                ### REMOVING SIZE FROM DICTIONARY
                marker_dict.pop('size', None)
            except Exception:
                pass
            ### ADDING TO PLOT
            ## MAKING EDITS TO THE COORDINATES
            # self.prob_density.solute_atom_coord = self.prob_density.solute_atom_coord_dict['avg']
            ## PLOTTING POINTS
            mlab.points3d(
                        self.prob_density.solute_atom_coord[index_matched,0], 
                        self.prob_density.solute_atom_coord[index_matched,1], 
                        self.prob_density.solute_atom_coord[index_matched,2], 
                        shape_atom_size,
                        figure = scene,
                        scale_factor=.25, resolution = ATOM_RESOLUTION,**marker_dict
                           ) 
            
        ### DRAWING BONDS
        if self.skip_bonds is False:
            ## DEFINING COORDINATES
            solute_atom_coord = self.prob_density.solute_atom_coord
            
            ## DEFINING BONDS INDEX
            bonds_index = self.prob_density.solute_structure.bonds - 1 # -1 to resolve issue with python starting with 0
            
            ## FINDING BONDING COORDINATES
            bond_coordinates_array = solute_atom_coord[bonds_index]
            
            #### DRAWING BONDS
            for bond_idx,current_bond in enumerate(self.prob_density.solute_structure.bonds_atomname):
                ## DO NOT DRAW GOLD BONDS
                if 'Au' not in current_bond:
                    ## FINDING COORDINATES FOR EACH ATOM
                    bond_coordinates = bond_coordinates_array[bond_idx]
                    # np.array( [ self.prob_density.solute_atom_coord[index] \
                                                  # for index in [ self.prob_density.solute_structure.atom_atomname.index(eachAtom) for eachAtom in current_bond ] ] )
                    
                    mlab.plot3d(  
                             bond_coordinates[:,0], # x
                             bond_coordinates[:,1], # y
                             bond_coordinates[:,2], # z    
                             figure =  scene,
                             **self.dict_bonds
                             )
                ## PRINTING
                if bond_idx % 100 == 0:
                    print("Plotting bond index: %d of %d, current bond: %s"%(bond_idx, len(self.prob_density.solute_structure.bonds_atomname), '-'.join(current_bond)))
        else:
            print("Skipping drawing of bonds!")
        return 
    
    ### FUNCTION TO FIND NUMBER OF CONTOURS BASED ON MAX AND MIN
    def calc_contours_prob_dist(self, PROB_DIST, num_contours = 3):
        '''
        The purpose of this function is to calculate the contour levels for the probability distribution function
        INPUTS:
            self: class object
            PROB_DIST: probability distribution, numpy 3D array (e.g. 30 x 30 x 30)
        OUTPUTS:
            contours: np array that includes all the contours that are desired
        REFERENCE:
            https://stackoverflow.com/questions/37146143/mayavi-contour-3d
        '''
        if self.contours is None:
            max_value = np.max(PROB_DIST)
            min_value = np.min(PROB_DIST)
            ## CALCULATING UPPER CONTOURS
            lower_contours =  np.linspace(min_value,0.25, num=0)                    
            upper_contours =  np.arange(1.5, 3+0.1, 0.1)
            
            contours = np.concatenate( (lower_contours,upper_contours), axis=0 )
            # contours = np.concatenate( (contours,np.array([1])), axis=0 )
        else:
            contours = self.contours
        return contours
    
    ### FUNCTION TO PLOT ISOSURFACE
    def plot_isosurface(self, PROB_DIST, contours, scene, opacity = 0.3, index = 0, vmin = 1, vmax = 3):
        '''
        The purpose of this function is to plot the isosurface given the probability distribution
        INPUTS:
            self: class object
            PROB_DIST: probability distribution, numpy 3D array (e.g. 30 x 30 x 30)
            figure: Figure you are plotting on
            opacity: [float]
                opacity of the isosurface
        OUTPUTS:
            plot of isosurface
        # COLOR MAP OPTIONS --- http://docs.enthought.com/mayavi/mayavi/mlab_changing_object_looks.html
            'blue-red': Blue to red
            'black-white': black to white
            'cool', 'summer' winter spectral gray bone autumn prism gist_rainbow
        '''
        obj = mlab.contour3d(PROB_DIST, 
                             contours=contours.tolist(), 
                             opacity=opacity, # 0.10
                             transparent=True, 
                             colormap =self.color_map_variations[index], 
                             vmin=vmin, 
                             vmax=vmax,
                             figure = scene,
                             )
        '''
        Oranges
        TraitError: The 'colormap' trait of an IsoSurfaceFactory instance must be 'Accent' or 'Blues' or 'BrBG' or 'BuGn' or 'BuPu' 
        or 'CMRmap' or 'Dark2' or 'GnBu' or 'Greens' or 'Greys' or 'OrRd' or 'Oranges' or 'PRGn' or 'Paired' or 'Pastel1' or 
        'Pastel2' or 'PiYG' or 'PuBu' or 'PuBuGn' or 'PuOr' or 'PuRd' or 'Purples' or 'RdBu' or 'RdGy' or 'RdPu' or 'RdYlBu' or 'RdYlGn' or
        'Reds' or 'Set1' or 'Set2' or 'Set3' or 'Spectral' or 'Vega10' or 'Vega20' or 'Vega20b' or 'Vega20c' or 'Wistia' or 'YlGn' or 'YlGnBu' or 
        'YlOrBr' or 'YlOrRd' or 'afmhot' or 'autumn' or 'binary' or 'black-white' or 'blue-red' or 'bone' or 'brg' or 'bwr' or 'cool' or 'coolwarm' or 
        'copper' or 'cubehelix' or 'file' or 'flag' or 'gist_earth' or 'gist_gray' or 'gist_heat' or 'gist_ncar' or 'gist_rainbow' or 'gist_stern' or 'gist_yarg' or 
        'gnuplot' or 'gnuplot2' or 'gray' or 'hot' or 'hsv' or 'inferno' or 'jet' or 'magma' or 'nipy_spectral' or 'ocean' or 'pink' or 'plasma' or 'prism' or 
        'rainbow' or 'seismic' or 'spectral' or 'spring' or 'summer' or 'terrain' or 'viridis' or 'winter', but a value of 'white-red' <class 'str'> was specified.
        '''

        
        return obj
    
    ### FUNCTION TO ADD COLOR BAR TO SURFACE
    def add_color_bar_isosurface(self, iso_obj, orientation='vertical', index = 0, label_fmt='%.2f', nb_labels = 10):
        '''
        INPUTS:
            iso_obj: Object outputted from isosurface
            orientation: 'vertical' or 'horizontal'
            label_fmt: The format of the color bar
            nb_labels: Number of labels on color bar
        OUTPUTS:
            VOID -- color bar will be added to figure
        '''
        
        colorbar  = mlab.colorbar(iso_obj, 
                                  orientation=orientation, 
                                  label_fmt=label_fmt, 
                                  nb_labels = nb_labels)
        colorbar.scalar_bar_representation.position = self.color_bar_locations[index]
        ''' COMMANDS TO CHANGE COLORBAR POSITIONS
        colorbar.scalar_bar_representation.position = [0.1, 0.9]
        colorbar.scalar_bar_representation.position2 = [0.8, 0.05]'''
        return colorbar
    
    ### FUNCTION TO ADD OUTLINE
    def add_outline(self, line_width = 3):
        '''
        This function adds an outline box
        INPUTS:
            self: class object
            prob_density: class object of the probability density
            line_width: line width of the outline
        OUTPUTS:
            VOID -- outline will be added to the figure
        '''
        obj = mlab.outline(extent=self.prob_density.plot_axis_range*3, line_width=line_width) # Shows the outline
        return obj
    
    ### FUNCTION TO ADD CONTOUR PLOT
    def add_planar_contour(self, PROB_DIST, plane_orientation='x_axes',slice_index=0, transparent=False):
        '''
        The purpose of this function is to add a contour plot for a given orientation. By default, the slice index starts at zero. But it does not have to be the case, change by:
            obj.ipw.slice_index=1
        INPUTS:
            self: class object
            scene: scene of mayavi figure
            PROB_DIST: Probability distribution function that you are using
            plane_orientation: orientation of the plane
            slice_index: Which slice do you want
            transparent: True or False -- whether or not you want transparency
            opacity: Float -- transparency depends on this 
        '''
        obj=mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(PROB_DIST),
                                # figure = scene.mayavi_scene,
                                plane_orientation=plane_orientation,
                                slice_index=slice_index,
                                transparent = transparent,
                            )
        return obj
    ### FUNCTION TO EDIT FIGURE
    @staticmethod
    def edit_mayavi_fig( figure, foreground_color=(0,0,0) , background_color =(1,1,1) ):
        '''
        The purpose of this function is to create a mayavi figure
        INPUTS:
            engine: engine you want to run with
            foreground_color: Color of the text (default black) -- Should be a tuple of 3 values (e.g. (0,0,0) )
            background_color: Color of the background (default white) -- Should be a tuple of 3 values (e.g. (1,1,1) )
        OUTPUTS:
            Mayavi figure
        '''
        figure.scene.foreground=foreground_color
        figure.scene.background=background_color

    ### FUNCTION TO PLOT SOLUTE ATOM INDEX
    def plot_solute_index(self):
        ''' This function plots the solute index in cartesian coordinates  '''
        fig, ax = plot_tools.plot_solute_atom_index(positions = self.prob_density.solute_atom_coord, 
                                            atom_elements = self.prob_density.solute_atom_elements, 
                                            atom_names = self.prob_density.solute_atom_names, 
                                            bond_atomnames = self.prob_density.solute_structure.bonds_atomname,
                                            atom_dict = ATOM_DICT )
        return fig, ax

### FUNCTION TO UPDATE IMAGE OF PROB DENSITY MAP
class update_image_combined_image:
    '''
    The purpose of this function is to update an image based on previous viewing positions. 
    The idea here is that we want to revive previous images -- potentially as a way of recreating videos 
    in a quick manner.
    INPUTS:
        prob_density_map: [obj]
            probability density map object
        input_pickle_file_name: [str]
            input pickle file name -- file name you want to store your viewing information
        pickle_path: [str]
            pickle path to store information about your image
    OUTPUTS:
        self.prob_density_map: [obj]
            probability density map
        self.pickle_path: [str]
            stored pickle path
        self.input_pickle_file_name: [str]
            stored pickle file name
        self.figure: [obj]
            figure object
        self.cam: [obj]
            camera for object
    FUNCTIONS:
        zoom: zooms camera
        reload_view: reloads view from a pickle file
        save_view_data: saves view into pickle file so you can view it quickly
    '''
    ## INITIALIZING
    def __init__(self, prob_density_map, input_pickle_file_name, pickle_path, ):
        ## STORING PROBABILITY DENSITY MAP
        self.prob_density_map = prob_density_map
        self.pickle_path = pickle_path
        self.input_pickle_file_name = input_pickle_file_name
        
        ## DEFINING FIGURE
        self.figure = self.prob_density_map.figures[0]
        
        ## DEFINING CAMERA
        self.cam = self.figure.scene.camera
        
    ## ZOOMING FUNCTION
    def zoom(self, zoom_value = 3.5):
        ''' The purpose of this function is to zoom into image '''
        self.cam.zoom(zoom_value)
        self.figure.render()            
        return
    
    
    ## FUNCTION TO RELOAD DATA
    def reload_view(self):
        '''
        The purpose of this function is to reload view angles based on viewing data. 
        If no reloading data is found, then mention it. 
        INPUTS:
            self: [obj]
                class object
        OUTPUTS:
            reloaded image
        '''
        ## DEFINING RELOADING
        self.reload_view_logical = False
        
        ## ZOOMING
        # self.zoom()
        
        ## LOADING THE PICKLE AND STORING
        self.full_view_dict = load_class_pickle(pickle_path = self.pickle_path)
        
        if self.input_pickle_file_name in self.full_view_dict.keys():
            ## CHANGING LOGICAL 
            self.reload_view_logical = True
            ## CHANGING VIEW IMAGE
            mlab.view(*self.full_view_dict[self.input_pickle_file_name]['view'], 
                      roll = self.full_view_dict[self.input_pickle_file_name]['roll'], 
                      reset_roll = True, 
                      figure = self.figure)
            
            ## CHANGING VIEW ANGLE
            self.figure.scene.camera.view_up = self.full_view_dict[self.input_pickle_file_name]['viewup']
            
            ## RENDERING IMAGE    
            self.figure.scene.camera.compute_view_plane_normal()
            self.figure.render()
        else:
            print("No reloading was done! No data on: %s"%(self.input_pickle_file_name) )
        return
                
    ## FUNCTION TO SAVE VIEW DATA
    def save_view_data(self, want_overwrite = False) :
        '''
        The purpose of this function is to save view data.
        INPUTS:
            self: [obj]
                class object
        OUTPUTS:
            pickle file with the viewing data
        '''
        ## STORING
        self.want_overwrite = want_overwrite
        ## GETTING VIEW ANGLE
        view = mlab.view( figure =  self.figure )
        ## FINDING ROLL
        roll = mlab.roll( figure =  self.figure  )
        ## FINDING VIEW UP
        view_up = self.figure.scene.camera.view_up

        ## DEFINING VIEW AND ROLL DICTIONARY
        view_and_roll_values = {
                        'view': view,
                        'roll': roll,
                        'viewup': view_up,
                        }
        ## SEEING IF EVEN A PICKLE EXISTS EXISTS
        if os.path.isfile(self.pickle_path) is not True:
            store_class_pickle( my_class = view_and_roll_values, output_path = self.pickle_path  ) # view_dict
        ## LOADING FILE
        full_view_dict = load_class_pickle(pickle_path = self.pickle_path)
        
        ## SEEING IF NEW VIEW EXISTS
        if self.input_pickle_file_name in full_view_dict.keys():

            
            ## SEE IF YOU WANT TO OVERWRITE
            if self.want_overwrite is True:
                full_view_dict.update({self.input_pickle_file_name: view_and_roll_values})
                ## STORING DICTIONARY
                store_class_pickle( my_class = full_view_dict, output_path = self.pickle_path  )
        else:
            ## ADDING TO DICTIONARY
            full_view_dict[self.input_pickle_file_name] = view_and_roll_values
            ## STORING DICTIONARY
            store_class_pickle( my_class = full_view_dict, output_path = self.pickle_path  )
            
        return


#%% MAIN SCRIPT
if __name__ == "__main__":

    ## DEFINING CLASS
    Descriptor_class = calc_prob_density_map
    
    ## DEFINING DATE
    Date='190924-GAUCHEPDO'
    # Date='190623'
    # '190924-GAUCHEPDO'
    # '190325-FINAL' # '180516'
    
    ## DEFINING DESIRED DIRECTORY
    residue_name='PDO'
    # residue_name='THD'
    mass_perc='10' # 100 10
    mass_perc='100' # 100 10
    # '50' '5' '85' '90'
    # cosolvent_residue_name='GVL_L'
    # cosolvent_residue_name='N-methyl-2-pyrrolidone'
    # cosolvent_residue_name='tetrahydrofuran'
    # cosolvent_residue_name='dmso'
    # cosolvent_residue_name='GVL_L'
    cosolvent_residue_name='Pure'
    # cosolvent_residue_name='dioxane'
    # cosolvent_residue_name='tetramethylenesulfoxide'
    # cosolvent_residue_name='tetrahydrofuran'
    # cosolvent_residue_name='N-methyl-2-pyrrolidone'
    Pickle_loading_file=r"Mostlikely_433.15_6_nm_" + residue_name + "_" + mass_perc + "_WtPercWater_spce_" + cosolvent_residue_name
    # Pickle_loading_file=r"mdRun_403.15_6_nm_XYL_10_WtPercWater_spce_dioxane"
    # Pickle_loading_file=r"mdRun_433.15_6_nm_ACE_100_WtPercWater_spce_Pure"
    
    '''
    'mdRun_433.15_6_nm_PDO_10_WtPercWater_spce_N-methyl-2-pyrrolidone'
    'mdRun_433.15_6_nm_PDO_10_WtPercWater_spce_NN-dimethylacetamide'
    'mdRun_433.15_6_nm_PDO_10_WtPercWater_spce_NN-dimethylformamide'
    'mdRun_433.15_6_nm_PDO_10_WtPercWater_spce_urea'
    
    'mdRun_433.15_6_nm_ACE_10_WtPercWater_spce_dioxane',
    'mdRun_433.15_6_nm_ACE_10_WtPercWater_spce_dmso',
    'mdRun_433.15_6_nm_PDO_10_WtPercWater_spce_dioxane',
    'mdRun_433.15_6_nm_PDO_10_WtPercWater_spce_dmso',
    'mdRun_433.15_6_nm_PRO_10_WtPercWater_spce_dioxane',
    'mdRun_433.15_6_nm_PRO_10_WtPercWater_spce_dmso',
    '''
    ### EXTRACTING THE DATA
    prob_density = load_multi_traj_pickle(Date, Descriptor_class, Pickle_loading_file )
    
    ### CALCULATING PROBABILITY DENSITIES
    prob_density.calc_prob_density_dist()
    
    ''' FOR SUBTRACTION TO WATER
    Pickle_loading_file=r"mdRun_433.15_6_nm_PDO_100_WtPercWater_spce_Pure"
    
    ### EXTRACTING THE DATA
    prob_density_2 = load_multi_traj_pickle(Date, Descriptor_class, Pickle_loading_file )
    
    ### CALCULATING PROBABILITY DENSITIES
    prob_density_2.calc_prob_density_dist()
    
    prob_density.prob_dist[0] = prob_density.prob_dist[0] - prob_density_2.prob_dist[0]
    '''
    ### DEFINING INPUTS
    # contours = np.arange(1.3, 3+0.1, 0.1)
    # contours = np.arange(1.3, 3+0.1, 0.1)
    # contours = np.arange(0.5, 2, 0.1)
    
    iso_map_dict={
            "HOH":{
                    "contours": np.arange(1.5, 3+0.1, 0.3),   # np.arange(1.3, 3+0.1, 0.1),
                    "vmin": 1.5, # 0
                    "vmax": 3,
                    'nb_labels': 6, # Number of labels
                    },
            "COSOLVENT":{
                    "contours": np.arange(1.3, 1.5+0.1, 0.1), # np.arange(1.3, 3+0.1, 0.1)
                    "vmin":  1.3, # 0
                    "vmax": 1.5,
                    'nb_labels': 8,
                    }                
            }
    mlab.close(all=True)
    ### RUNNING PROBABILITY DENSITY MAPS
    prob_density_map = plot_prob_density_map(prob_density = prob_density,
                                             iso_map_dict = iso_map_dict,
                                             combine_plot = True,
                                             want_water_only = False,
                                             want_color_bar = True,
                                             **MAP_INPUT
                                             )
    # np.max(prob_density.prob_dist[1])
    # np.max(prob_density.prob_dist[0])
    # prob_density_map.figures[0].scene.parallel_projection = False
    ## PLOTTING SOLUTE INDEX
    # prob_density_map.plot_solute_index()
    
    #%%
    
            
    ## LOADING PICKLE INFORMATION
    from MDBuilder.core.pickle_funcs import store_class_pickle, load_class_pickle
    PICKLE_PATH_SPATIAL_DIST=r"R:\scratch\SideProjectHuber\Scripts\AnalysisScripts\output_spatial_dist_functions\spatial_dist_map_info.pickle"
    SPATIAL_DIST_IMAGE_STORAGE_PATH = r"R:\scratch\SideProjectHuber\Scripts\AnalysisScripts\output_spatial_dist_functions\output_images_spatial_dist_fxns"
    ### RELOADING IMAGE
    updated_image = update_image_combined_image( prob_density_map = prob_density_map, 
                                                 input_pickle_file_name = Pickle_loading_file, 
                                                 pickle_path = PICKLE_PATH_SPATIAL_DIST)
    #%%
    ### ZOOMING
    updated_image.zoom()
    
    #%%
    
    ### UPDATING IMAGE
    updated_image.reload_view()
    
    #%%
    ### STORING IMAGE
    updated_image.save_view_data(want_overwrite=True)
    
    #%%
    
    ### STORING SPECIFIC IMAGE
    fig_name = Pickle_loading_file + '_front' + '.eps'
    mlab.savefig( fig_name, figure= prob_density_map.figures[0], **PRINT_FIGURE_INFO )

    #%%
    ### CROSS IMAGE
    fig_name = Pickle_loading_file + '_side' + '.eps'
    mlab.savefig( fig_name, figure= prob_density_map.figures[0], **PRINT_FIGURE_INFO )
    
    # unload gz: gunzip *.gz
    
    
    #%%
    
    ### STORING SPECIFIC IMAGE
    fig_name = Pickle_loading_file + '_front' + '.png'
    mlab.savefig( fig_name, figure= prob_density_map.figures[0], **PRINT_FIGURE_INFO )

    #%%
    ### CROSS IMAGE
    fig_name = Pickle_loading_file + '_side' + '.png'
    mlab.savefig( fig_name, figure= prob_density_map.figures[0], **PRINT_FIGURE_INFO )
    
    
    #%%
    @mlab.animate(delay=ANIMATION_DELAY, ui=False)
    def animate_rotate(figure, azimuth_rotation=10, saved_plots=False, name='', output_directory ='.'):
        '''
        The purpose of this function is to animate and rotate the image across the axis 
        and output pngs for each frame. 
        INPUTS:
            figure: [obj]
                figure object
            azimuth_rotation: [float]
                azimuth rotation increment between 0 and 360 -- smaller increments mean more images
            saved_plots: [logical, default=False]
                True if you want to save the plots as png
            name: [str, default='']
                name of your saving plot
            output_directory: [str, default='.']
                output directory for your plots saving
        OUTPUTS:
            
        '''
        ## COMPUTING NORMAL CAMERA VIEW
        figure.scene.camera.compute_view_plane_normal()
        
        setpoint_rotation = 360 # Full rotation
        counter=azimuth_rotation
        index=0
        if saved_plots is True:
            ## DISABLING RENDERING
            figure.scene.disable_render = True
        ## LOOPING THROUGH EACH ROTATION        
        while counter <= setpoint_rotation:
            
            ## SAVING PLOTS                
            if saved_plots is True:
                ## PRINTING
                print("Counter on: %d, %.1f of %d" %(index, counter, setpoint_rotation) )
                output_file = os.path.join( output_directory, name + '_' + str(index) + '.png' )
                mlab.savefig( filename = output_file, figure= figure, **PRINT_FIGURE_INFO )
            
            ## ROTATING
            figure.scene.camera.azimuth(azimuth_rotation); 
            ## RENDERING
            if saved_plots is False:
                figure.render()
            ## UPDATING COUNTER AND ROTATION
            counter+=azimuth_rotation; index+=1
            yield
        if saved_plots is True:
            ## REENABLING RENDER
            figure.scene.disable_render = False
        return
            
    ### ANIMATING IMAGE AND STORING
    anim = animate_rotate(
                          figure = updated_image.figure, 
                          azimuth_rotation = 10,
                          saved_plots=True, 
                          name= Pickle_loading_file,
                          output_directory = SPATIAL_DIST_IMAGE_STORAGE_PATH)
    
    #%%
    
    ## CREATING MOVIE
    basename = "Mostlikely_433.15_6_nm_PDO_10_WtPercWater_spce_dmso"
    storage_dir = SPATIAL_DIST_IMAGE_STORAGE_PATH
    ## COMBINING PATH
    storage_path = os.path.join( storage_dir, basename  )
    
    os.system("ffmpeg -r 1 -i img%01d.png -vcodec mpeg4 -y movie.mp4")
    
#     os.system("ffmpeg -r 1 -i %s%1d.png -vcodec mpeg4 -y movie.mp4"%(storage_path))
    
    
    
    
    
       
    
        
        
       
        
    #%%
        
    ## STORING IMAGE IF NECESSARY
        
        
    
    ## CHANGING VIEW IMAGE
    mlab.view(*full_view_dict[Pickle_loading_file]['view'], 
              roll = full_view_dict[Pickle_loading_file]['roll'], 
              reset_roll = True, 
              figure = fig)
    
    ## CHANGING VIEW ANGLE
    fig.scene.camera.view_up = full_view_dict[Pickle_loading_file]['viewup']
    
    ## RENDERING IMAGE    
    fig.scene.camera.compute_view_plane_normal()
    fig.render()
    
    #%%
    
    ### FUNCTION TO GET POSITION AND EVERYTHING
    def get_camera_info( fig ):
        ''' This function gets position and so on for figure '''
        cam_position = fig.scene.camera.position
        cam_focal_point = fig.scene.camera.focal_point
        cam_view_angle = fig.scene.camera.view_angle
        cam_view_up = fig.scene.camera.view_up
        cam_clipping_range = fig.scene.camera.clipping_range
        
        return cam_position, cam_focal_point, cam_view_angle, cam_view_up, cam_clipping_range
    
    cam_position, cam_focal_point, cam_view_angle, cam_view_up, cam_clipping_range = get_camera_info(fig)
    
    #%%
    args = get_camera_info(fig)
    
    #%%
    
    ### FUNCTION TO UPDATE POSITIONS AND EVERYTHING
    def update_camera_info( fig, cam_position, cam_focal_point, cam_view_angle, cam_view_up, cam_clipping_range ):
        ''' This function updates camera information'''
        fig.scene.camera.position = cam_position
        fig.scene.camera.focal_point = cam_focal_point
        fig.scene.camera.view_angle = cam_view_angle
        fig.scene.camera.view_up = cam_view_up
        fig.scene.camera.clipping_range = cam_clipping_range
        fig.scene.camera.compute_view_plane_normal()
        fig.render()
        return
    ## UPDATING FIGURE
    update_camera_info( fig, cam_position, cam_focal_point, cam_view_angle, cam_view_up, cam_clipping_range)
    
    
    
    
    
    #%%
    

    
    ## GETTING VIEW ANGLE
    view = mlab.view( figure =  fig )
    ## FINDING ROLL
    roll = mlab.roll( figure =  fig )
    ## FINDING VIEW UP
    view_up = fig.scene.camera.view_up
    
    ## CREATING DICTIONARY
    view_dict={Pickle_loading_file: {
                    'view': view,
                    'roll': roll,
                    'viewup': view_up,
                    }
               }
            
    ## DEFINING PATH OF INTEREST
    
    want_overwrite = True
    
    ## STORING DICTIONARY INTO PICKLE
    ## SEEING IF EXISTS
    if os.path.isfile(PICKLE_PATH_SPATIAL_DIST) is not True:
        store_class_pickle( my_class = view_dict, output_path = PICKLE_PATH_SPATIAL_DIST  )
    else:
        ## LOADING THE PICKLE AND STORING
        full_view_dict = load_class_pickle(pickle_path = PICKLE_PATH_SPATIAL_DIST)
    
    ## SEEING IF NEW VIEW EXISTS
    if Pickle_loading_file in full_view_dict.keys():
        ## DEFINING VIEW AND ROLL DICTIONARY
        view_and_roll_values = view_dict[Pickle_loading_file]
        
        ## SEE IF YOU WANT TO OVERWRITE
        if want_overwrite is True:
            full_view_dict.update({Pickle_loading_file: view_and_roll_values})
            ## STORING DICTIONARY
            store_class_pickle( my_class = full_view_dict, output_path = PICKLE_PATH_SPATIAL_DIST  )
    else:
        ## ADDING TO DICTIONARY
        full_view_dict[Pickle_loading_file] = view_and_roll_values
        ## STORING DICTIONARY
        store_class_pickle( my_class = full_view_dict, output_path = PICKLE_PATH_SPATIAL_DIST  )
    
    
    
    
        ## DO NOTHING
    
    



    #%%
    ## CHANGING VIEW ANGLE
    mlab.view(*view, roll = roll, reset_roll = False, figure = fig)
    # mlab.view(figure = fig)
    
    #%% ANIMATIONS

    #%% fig.scene.camera.view_up array([0.5151978 , 0.32698131, 0.79224646])
    
    #%%
    
    ## GETTING FOCAL POINT
    distance = full_view_dict[Pickle_loading_file]['view'][2]
    focal_point = full_view_dict[Pickle_loading_file]['view'][3]
    
    ## MOVING CAMERA
    mlab.view( distance = distance, focalpoint = focal_point, figure = fig )
    
    # mlab.yaw(5)
    #%%
    
    def _putline(*args):
        """
        Generate a line to be written to a cube file where 
        the first field is an int and the remaining fields are floats.
        
        params:
            *args: first arg is formatted as int and remaining as floats
        
        returns: formatted string to be written to file with trailing newline
        """
        s = "{0:^ 8d}".format(args[0])
        s += "".join("{0:< 12.6f}".format(arg) for arg in args[1:])
        return s + "\n"
        
    # Function taken from: https://gist.github.com/aditya95sriram/8d1fccbb91dae93c4edf31cd6a22510f
    def write_cube(data, meta, fname):
        """
        Write volumetric data to cube file along
        
        params:
            data: volumetric data consisting real values
            meta: dict containing metadata with following keys
                atoms: list of atoms in the form (mass, [position])
                org: origin
                xvec,yvec,zvec: lattice vector basis
            fname: filename of cubefile (existing files overwritten)
        
        returns: None
        """
        with open(fname, "w") as cube:
            # first two lines are comments
            cube.write(" Cubefile created by cubetools.py\n  source: none\n")
            natm = len(meta['atoms'])
            nx, ny, nz = data.shape
            cube.write(_putline(natm, *meta['org'])) # 3rd line #atoms and origin
            cube.write(_putline(nx, *meta['xvec']))
            cube.write(_putline(ny, *meta['yvec']))
            cube.write(_putline(nz, *meta['zvec']))
            for atom_mass, atom_pos in meta['atoms']:
                cube.write(_putline(atom_mass, *atom_pos)) #skip the newline
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        if (i or j or k) and k%6==0:
                            cube.write("\n")
                        cube.write(" {0: .5E}".format(data[i,j,k]))
    
    
    ## CONVERSION FROM NM TO BOHR
    nm_to_bohr = 18.897161646321 # 1 nm to this  bohr
    
    ## DEFINING ORIGIN
    origin = tuple(prob_density.COM_solute_avg[0])
    
    ## DEFINING META
    meta = { 'atoms': [(1, list(each_atom) ) for each_atom in prob_density.solute_atom_coord], # 
             'org': origin, 
             'xvec': (3, 0, 0),
             'yvec': (0, 3, 0),
             'zvec': (0, 0, 3),
             }
    
    ## WRITING CUBE FILE
    write_cube(data = prob_density.prob_dist[1], 
               meta = meta, 
               fname = "test_2.cube")
    
    
    
    #%%
    
    ### FIRST SCENE
    fig_name = Pickle_loading_file + '_' + prob_density_map.prob_density.solvent_name[0] + '.png'
    mlab.savefig( fig_name, figure= prob_density_map.figures[0], **PRINT_FIGURE_INFO )
    
    ### SECOND SCENE
    fig_name = Pickle_loading_file + '_' + prob_density_map.prob_density.solvent_name[1] + '.png'
    mlab.savefig( fig_name, figure= prob_density_map.figures[1], **PRINT_FIGURE_INFO )
    #%%    
    ## FINDING VIEW ANGLE
    
    def rotate_90(figure):
        view_angles = mlab.view(figure=figure)
        ## ADDING 90 TO AZIMUTH
        view_angles_corrected = list(view_angles)
        view_angles_corrected[0]  = view_angles_corrected[0] + 90
        ## CONVERTING TO SEt
        view_angles_corrected = tuple(view_angles_corrected[:])
    
        ## CHANGING VIEW ANGLE
        mlab.view( *view_angles_corrected, figure = figure )
        
#    rotate_90(figure =prob_density_map.figures[0] )
#    rotate_90(figure =prob_density_map.figures[1] )
#    
    
    mlab.yaw(degrees = 90, figure = prob_density_map.figures[0])
    
    
    #%%
    ### FIRST SCENE (CROSS)
    fig_name = Pickle_loading_file + '_' + prob_density_map.prob_density.solvent_name[0] + '_cross.png'
    mlab.savefig( fig_name, figure= prob_density_map.figures[0], **PRINT_FIGURE_INFO )
    
    ### SECOND SCENE (CROSS)
    fig_name = Pickle_loading_file + '_' + prob_density_map.prob_density.solvent_name[1] + '_cross.png'
    mlab.savefig( fig_name, figure= prob_density_map.figures[1], **PRINT_FIGURE_INFO )
    
    
    #%%
    
    import numpy as np
    import math
    
    def rotation_matrix(axis, theta):
        """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        axis = np.asarray(axis)
        theta = np.asarray(theta)
        axis = axis/math.sqrt(np.dot(axis, axis))
        a = math.cos(theta/2.0)
        b, c, d = -axis*math.sin(theta/2.0)
        aa, bb, cc, dd = a*a, b*b, c*c, d*d
        bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
        return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                         [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                         [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
    
    
    az = 90
    el = -75
    
    x = np.cos(np.deg2rad(el))*np.cos(np.deg2rad(az))
    y = np.cos(np.deg2rad(el))*np.sin(np.deg2rad(az))
    z = np.sin(np.deg2rad(el))
    
    # So your viewing vector in x,y coordinates on unit sphere
    v = [x,y,z]
    
    # Since you want to rotate about the y axis from this viewing angle, we just increase the
    # elevation angle by 90 degrees to obtain our axis of rotation
    
    az2 = az
    el2 = el+90
    
    x = np.cos(np.deg2rad(el2))*np.cos(np.deg2rad(az2))
    y = np.cos(np.deg2rad(el2))*np.sin(np.deg2rad(az2))
    z = np.sin(np.deg2rad(el2))
    
    axis = [x,y,z]
    
    # Now to rotate about the y axis from this viewing angle we use the rodrigues formula
    # We compute our new viewing vector, lets say we rotate by 45 degrees
    theta = 45
    newv = np.dot(rotation_matrix(axis,np.deg2rad(theta)), v)
    
    #Get azimuth and elevation for new viewing vector
    az_new = np.rad2deg(np.arctan(newv[1]/newv[0]))
    el_new = np.rad2deg(np.arcsin(newv[2]))
    
    
    
    #%%
    
    
    ## DESIGNATING VECTORS
    ATOM_INDEX_TO_VISUALIZE = {
            'PDO':
                [ [ 'C2', 'O2' ],
                  [ 'C3', 'O1' ]]

            }
    
    ## GETTING STRUCTURE COORDINATES
    solute_atom_elements = prob_density_map.prob_density.solute_atom_elements
    solute_positions = prob_density_map.prob_density.solute_atom_coord
    solute_atom_names = prob_density_map.prob_density.solute_atom_names
    bond_atomnames = prob_density_map.prob_density.solute_structure.bonds_atomname
        
    ## FINDING VECTORS
    atom_index_visualize = ATOM_INDEX_TO_VISUALIZE['PDO']
    
    ## FINDING VECTORS
    vectors = np.zeros( (2,3) )
    for idx, atom_indices in enumerate(atom_index_visualize):
        ## FINDING ATOM INDEX
        atom_index = [ solute_atom_names.index(eachAtom) for eachAtom in atom_indices] 
        ## FINDING EACH POSITION
        atom_positions = solute_positions[atom_index, :]
        ## FINDING VECTOR
        vectors[idx,:] = atom_positions[1,:] - atom_positions[0,:]
        # vectors = np.append(vectors, atom_positions[1,:] - atom_positions[0,:] )
        # vectors.append( atom_positions[1,:] - atom_positions[0,:] )
        
    ## FINDING CROSS PRODUCT
    cross_prod = np.cross( a= vectors[0,:], b = vectors[1,:] )
    
    ## FINDING UNIT VECTOR
    cross_prod_unit_vec = cross_prod / np.linalg.norm(cross_prod)
    
    ## FINDING ELEVATION
    elev = np.degrees(np.arccos( cross_prod_unit_vec[2])) # 0 to 180
    ## FINDING AZIMUTH
    azim = np.degrees( np.arccos( cross_prod_unit_vec[0] / np.cos( np.radians(elev) ) ) ) # 0 to 180
    
    ## CORRECTIONS FOR ELEVATION AN AZIMUTH AND ELEVATION
    
    ## CHANGING AZIMUTH
#    prob_density_map.figures[0].get()['camera'].azimuth = azim
#    prob_density_map.figures[0].get()['camera'].elevation = elev
    
    
    mlab.view( azimuth = azim, elevation = elev, figure = prob_density_map.figures[0]  )
    mlab.view( azimuth = azim, elevation = elev, figure = prob_density_map.figures[1]  )
    
    mlab.view( *mlab.view(figure=prob_density_map.figures[0]), figure = prob_density_map.figures[1],  )
    

    

    ## FIGURE DEFINITION
    # test = prob_density_map.figures[0]
    # test3 = test2.get()['camera']
    # test3.azimuth(30)
    
    
    #%%
    
    ### FIRST SCENE (CROSS)
    fig_name = Pickle_loading_file + '_' + prob_density_map.prob_density.solvent_name[0] + '_cross.png'
    mlab.savefig( fig_name, figure= prob_density_map.figures[0], **PRINT_FIGURE_INFO )
    

    
    #%%
    #%%
    

    
    
    #%%
    
    # prob_density_map.figures[0].scene.save_png(fig_name, **PRINT_FIGURE_INFO)
    # test.scene.save_png
    
    ### SAVING FIGURES
    
    
    
    
    #%%
    
    ### CONFIGURING PLOTS
    f = prob_density_map.scene1.mlab.gcf(prob_density_map.scene1.engine)
    camera = f.scene.camera
    camera.yaw(45)
    
    #%%
    # Store the information
    view = mlab.view()
    roll = mlab.roll()
    
    # Reposition the camera
    mlab.view(*view)
    mlab.roll(roll)

    #%%
    f = prob_density_map.scene2.mlab.gcf()
    camera = f.scene.camera
    camera.yaw(45)

    # prob_density_map.scene2.camera.azimuth(500)
    #  prob_density_map.scene2.render()


    
    #%%
    @mlab.animate(delay=ANIMATION_DELAY, ui=True)
    def animate_contour(prob_density_map, prob_density):
        f = mlab.gcf()
        while 1:
            for each_line in range(0,prob_density.max_bin_num+1):
                for contour_line in prob_density_map.contour_line:
                    contour_line.ipw.slice_index = each_line
                    f.scene.render()
                yield
    
    anim = animate_contour(prob_density_map, prob_density)
    
    '''
    test=prob_density_map.iso_obj.contour
    test.auto_contours = True
    prob_density_map.iso_obj.contour.number_of_contours = 5
    
    import ctypes
    
    user32 = ctypes.windll.user32
    
    screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    
    screensize
    Out[73]: (1920, 1200)
    '''