#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_spatial_heterogeniety.py

This script is designed to analyze the spatial heterogeniety observed for 
the mu distribution maps. The idea would be that we want to quantify how 
spatially heterogenous a surface is. We could do this in a number of ways:
    - compute the sphericity of the surface
    - use a clustering algorithm to count the distinct groups

This script will try to go through many different approaches to see which 
method works best in terms of getting quantifying spatially heterogenous 
surfaces. 

Written by: Alex K. Chew (06/12/2020)

"""
import os
import numpy as np
import matplotlib.pyplot as plt
import glob

## CLUSTERING ALGORITHM
import hdbscan

## IMPORTING SCIPY
from scipy.spatial.distance import pdist, squareform

## IMPORTING SKLEARN MODULES
from sklearn.cluster import DBSCAN
import pandas as pd

## IMPORTING GLOBAL VARIABLES
from MDDescriptors.application.np_hydrophobicity.global_vars import \
    PARENT_SIM_PATH, PATH_DICT, RELABEL_DICT, MU_PICKLE, LIGAND_COLOR_DICT, \
    PURE_WATER_SIM_DICT, PREFIX_SUFFIX_DICT, DEFAULT_WC_ANALYSIS, GRID_LOC, GRID_OUTFILE

# from MDDescriptors.application.np_hydrophobicity.publish_images import load_mu_values_for_multiple_ligands

## LOADING DAT FILE FUNCTION
from MDDescriptors.surface.core_functions import load_datafile

## PLOTTING SCATTER MU VALUES
from MDDescriptors.application.np_hydrophobicity.analyze_mu_distribution import plot_scatter_mu_values

## IMPORTING 3D AXIS
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 

## IMPORTING PLOT TOOLS
import MDDescriptors.core.plot_tools as plot_tools

## LOADING MODULES
from MDDescriptors.application.np_hydrophobicity.analyze_mu_distribution import \
    extract_hydration_maps, compute_histogram_for_mu, plot_histogram_data

## DEFAULTS
plot_tools.set_mpl_defaults()

## FIGURE SIZE
FIGURE_SIZE=plot_tools.FIGURE_SIZES_DICT_CM['1_col_landscape']

## DEFINING FIGPATH
PATH_FIG="/Users/alex/Box Sync/VanLehnGroup/2.Research Documents/Alex_RVL_Meetings/20200622/images/np_hydrophobicity_clustering"

## DEFINING MAIN HYDROPHOBICITY PATH
MAIN_HYDRO_DIR  = PARENT_SIM_PATH

### FUNCTION TO CREATE 3D PLOT
def create_3d_plot():
    ''' Function creates 3D plots '''
    ## CREATING PLOT
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ## ADDING LABELS
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    return fig, ax

### FUNCTION TO PLOT THE GRID POINTS
def plot_grid_points(grid_xyz,
                     scatter_style = {
                             's'         : 400,
                             'linewidth' : 1,
                             },
                     ):
    '''
    The purpose of this function is to plot the grid points
    INPUTS:
        grid_xyz: [np.array, (N, 3)]
            grid points in x, y, z cartersian coordinates
        scatter_style: [dict]
            dictionary for the scatter style
    OUTPUTS:
        fig, ax: figure and axis for gridding
    '''        
    ## CREATING FIGURE
    fig, ax = create_3d_plot()
    
    ## SCATTER PLOT
    ax.scatter(grid_xyz[:,0],
               grid_xyz[:,1],
               grid_xyz[:,2],
               **scatter_style)
    
    return fig, ax

### FUNCTION TO PLOT POINTS
def plot_3d_points_with_scalar(xyz_coordinates, 
                               scalar_array,
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
            scalar values for each coordinate
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

### FUNCTION TO PLOT CLUSTERED LABELS
def plot_3d_points_clusters(xyz_coordinates,
                            labels,
                            title = None,
                            close_all_figs = True):
    '''
    This function plots the clustering based on input labels and xyz coordinates
    INPUTS:
        xyz_coordinates: [np.array]
            xyz coordinates of the points
        labels: [np.array]
            labels for the clustering
        close_all_figs: [logical]
            True if you want to close all figures before re-creating
    OUTPUTS:
        figure: [obj]
            figure 
    '''
    ## IMPORTING MLAB
    from mayavi import mlab

    if close_all_figs is True:
    
        ## CLOSING MLAB
        try:
            mlab.close()
        except AttributeError:
            pass
    
    ## CREATING FIGURE
    mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
    
    ## GETTING UNIQUE LABELS
    unique_labels = np.unique(labels)
    
    ## COMPUTING CMAP
    colors_obj = plot_tools.get_cmap(len(unique_labels)-1)
    
    ## DEFINING FIGURE
    figure = None
    
    ## LOOPING
    for i, each_label in enumerate(unique_labels):
        ## GETTING INDEX
        idx_labels = np.where(labels == each_label)
        
        ## DEFINING COLORS
        if each_label == -1:
            color = (0.5, 0.5, 0.5)
        else:
            color = colors_obj(i)[0:3]
        
        ## PLOTTING POINTS
        figure = plot_3d_points_with_scalar(xyz_coordinates = xyz_coordinates[idx_labels], 
                                            scalar_array = labels[idx_labels],
                                            figure = figure,
                                            points_dict={
                                                   'scale_factor': 0.4,
                                                   'scale_mode': 'none',
                                                   'opacity': 1.0,
                                                   'mode': 'sphere',
                                                   'color': color,
                                               })
    
    ## ADDING TITLE
    if title is not None:
        mlab.title(title)
    
    return figure

### CLASS FUNCTION TO CLUSTER VALUES
class cluster_mu_values:
    '''
    The purpose of this class object is to compute cluster of mu values
    INPUTS:
        mu_values: [np.array]
            clustering mu value array
        grid_values: [np.array, shape= (N,3)]
            grid values corresponding to the mu values
    OUTPUTS:
        
    '''
    ## INITIALIZING
    def __init__(self,
                 mu_values,
                 grid_values):
        ## STORING
        self.mu_values = mu_values
        self.grid_values = grid_values
        
        return
    
    ## RUNNING DIVIDE INDEX ALGORITHM
    def divide_mu_values_by_index(self,
                         cutoff = 11):
        '''
        The purpose of this function is to find all hydrophobic clusters. 
        This will look for all mu values greater than 11 and less than 11. 
        INPUTS:
            cutoff: [float]
                cutoff for mu values in kT
        OUTPUTS:
            idx_below: [np.array]
                indexes below cutoff mu values
            idx_above: [np.array]
                indexes above cutoff mu values
        '''
        ## STORING CUTOFF
        self.cutoff = cutoff
        idx_below = np.where(self.mu_values < self.cutoff)
        idx_above = np.where(self.mu_values >= self.cutoff)
        
        return idx_below, idx_above

### FUNCTION FOR HDBSCAN CLUSTERING
def compute_hdbscan_cluster(grid_values,
                        min_samples = 4,
                        verbose = True
                        ):
    '''
    This function runs the hdbscan algorithm given the gride values.
    INPUTS:
        grid_values: [np.array]
            grid values to run the hdbscan clustering
        min_samples: [int]
            number of minimum samples for clustering
    OUTPUTS:
        hdbscan_cluster: [obj]
            hdbscan object
        labels: [np.array]
            labels for each object
    '''
    if len(grid_values) >= min_samples:
        ## RUNNING CLUSTERING
        hdbscan_cluster = hdbscan.HDBSCAN(algorithm='best', 
                                          alpha=1.0, 
                                          approx_min_span_tree = True,
                                          gen_min_span_tree = False,
                                          leaf_size = 40,
                                          min_cluster_size = min_samples,
                                          metric='euclidean').fit(grid_values)
    
        ## GETTING LABELS
        labels = hdbscan_cluster.labels_
        
        ## PRINTING TOTAL NUMBER OF CLUSTERS
        num_clusters = len(np.unique(labels))-1
        
    else:
        print("Since number of grid points above %.2f kT cutoff is lower than min samples, setting cluster to None")
        print("Number of grid values: %d"%(len(grid_values)))
        hdbscan_cluster = None
        labels = None
        num_clusters = 0
        
    if verbose is True:
        print("Total number of clusters: %d"%( num_clusters ) )
    
    return hdbscan_cluster, labels, num_clusters

## DEFINING MAIN CLUSTERING
def main_mu_clustering(path_to_sim,
                               relative_grid_path = os.path.join(DEFAULT_WC_ANALYSIS,
                                                                 GRID_LOC,
                                                                 GRID_OUTFILE,),
                               wc_folder = DEFAULT_WC_ANALYSIS,
                               mu_pickle = MU_PICKLE,
                               cutoff = 11.25,
                               min_samples = 4,
                               eps = None,
                               clustering_type = "dbscan",
                               ):
    '''
    The purpose of this function is to cluster mu values based on some cutoff. 
    Since mu is continuous, we would like to first select mu values with 
    large values, such as mu greater than 11 kT. Then, we want to see 
    how many clusters are formed from that.
    INPUTS:
        path_to_sim: [str]
            path to simulation
        relative_grid_path: [str]
            relative path to grid
        wc_folder: [str]
            WC folder of analysis
        mu_pickle: [str]
            pickle of mu values
        cutoff: [int]
            cutoff for mu values
        min_samples: [int]
            minimum samples used for hdbscan/dbscan
        eps: [float]
            value used for dbscan
        clustering_type: [str]
            dbscan or hdbscan. If hdbscan, eps is ignored.
    OUTPUTS:
        cluster_obj: [obj]
            clustering results
        labels: [list]
            list of labels
        clustering: [obj]
            clustering object. Note that this contains the grid and mu
            values used for the clustering, e.g.
            
                grid_values: [np.array]
                    grid values used for the clustering
                mu_values: [np.array]
                    mu values with N grids
                    
            Access these values by "clustering.grid_values"
        num_clusters: [int]
            total number of clusters
        idx_above: [np.array]
            indexes for mu values above
        idx_below: [np.array]
            index values for below
    '''
    ## PATH TO GRID
    path_to_grid = os.path.join(path_to_sim,
                                relative_grid_path)
    
    ## LOADING GRID
    grid_values = load_datafile(path_to_grid)
    
    ## PATH TO MU
    parent_to_mu_pickle = os.path.join(path_to_sim,
                                       wc_folder)
    
    ## LOADING MU VALUES
    hydration_map = extract_hydration_maps()
    mu_values = hydration_map.load_mu_values(main_sim_list = [parent_to_mu_pickle],
                                             pickle_name = mu_pickle)[0]
    
    
    ## RUNNING CLUSTERING
    clustering = cluster_mu_values(mu_values = mu_values,
                                   grid_values = grid_values)
    
    ## DIVIDING
    idx_below, idx_above = clustering.divide_mu_values_by_index(cutoff = cutoff)
    
    ## PRINTING
    print("Computing cluster with type: %s"%(clustering_type))
    
    ## GETTING CLUSTER
    if clustering_type == "hdbscan":
        cluster_obj, labels, num_clusters = compute_hdbscan_cluster(grid_values[idx_above],
                                                                    min_samples = min_samples,
                                                                    verbose = True)
    elif clustering_type == "dbscan":
        cluster_obj, labels, num_clusters = compute_dbscan_cluster(grid_values[idx_above],
                                                          eps = eps,
                                                          min_samples = min_samples,
                                                          verbose = True)
    else:
        print("Error! %s is not an available clustering type")
        available_clusters=  [ "dbscan", "hdbscan"]
        print("Available clusters:", available_clusters)
        
    
    return cluster_obj, labels, clustering, num_clusters, idx_above, idx_below


### FUNCTION TO COMPUTE CLUSTER
def compute_dbscan_cluster(grid_values,
                           eps = 0.5,
                           min_samples = 3,
                           verbose = True):
    '''
    This function computes DBSCAN to compute clusters
    INPUTS:
        grid_values: [np.array]
            grid values to run the hdbscan clustering
        eps: [float]
            size of cutoff
        min_samples: [int]
            number of minimum samples for clustering
            
    OUTPUTS:
        db: [obj]
            dbscan object
        labels: [np.array]
            labels for each object
        num_clusters: [int]
            number of clusters
    '''    
    if len(grid_values) >= min_samples:
        ## COMPUTING DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(grid_values)
        
        ## GETTING LABELS
        labels = db.labels_
        
        ## PRINTING TOTAL NUMBER OF CLUSTERS
        num_clusters = len(np.unique(labels))-1
    
    else:
        print("Since number of grid points above %.2f kT cutoff is lower than min samples, setting cluster to None")
        print("Number of grid values: %d"%(len(grid_values)))
        db = None
        labels = None
        num_clusters = 0
    
    if verbose is True:
        print("Total number of clusters: %d"%( num_clusters ) )
    
    return db, labels, num_clusters

### FUNCTION TO COMPUTE CLUSTERS FOR VARYING ARRAY
def vary_parameters_dbscan(grid_values,
                           eps_array = np.arange(0.1, 2, 0.1),
                           min_sample_array = np.arange(2, 10, 1)):
    '''
    This function varies the parameters for dbscan and outputs a dataframe.
    INPUTS:
        grid_values: [np.array]
            grid values to run the hdbscan clustering
    OUTPUTS:
        df: [pd.DataFrame]
            dataframe containing all the clusters
             eps  min_samples  num_clusters
        0    0.1            2           956
        1    0.1            3           882
    '''

    ## CREATING STORAGE DICT
    storage_dict = []
    
    ## LOOPING THROUGH EACH ONE
    for eps in eps_array:
        for min_samples in min_sample_array:
            ## COMPUTING CLUSTERS
            db, labels, num_clusters = compute_dbscan_cluster(grid_values = grid_values,
                                                               eps = eps,
                                                               min_samples = min_samples,
                                                               verbose = False)
            
            ## PRINTING
            print("Computing clusters for eps (%.2f) and min_samples (%d) --> %d"%(eps, min_samples, num_clusters ))
            
            ## STORING
            storage_dict.append(
                    {'eps': eps,
                     'min_samples': min_samples,
                     'num_clusters': num_clusters}
                    )
    ## CREATING DATAFRAME
    df = pd.DataFrame(storage_dict)
    
    return df

### FUNCTION TO PLOT VARYING DBSCAN FACTORS
def plot_vary_parameters_dbscan(df,
                                x_key = "eps",
                                y_key = "num_clusters",
                                fixed_key = "min_samples",
                                fig_size_cm = FIGURE_SIZE):
    '''
    This function plots the varying parameters for dbscan
    INPUTS:
        df: [pd.dataframe]
            dataframe containing varying parameters
        x_key: [str]
            key in dataframe that will be the x value
        y_key: [str]
            key in dataframe that will be the y value
        fixed_key: [str]
            fixed key in minimum samples
        fig_size_cm: [tuple]
            figure size in centimeters
    OUTPUTS:
        fig, ax: 
            figure and axis
    '''    
    ## CREATING FIGURE
    fig, ax = plot_tools.create_fig_based_on_cm(fig_size_cm)
    
    ## MAKING LABELS
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    
    ## GETTING UNIQUE LABELS
    unique_labels = np.unique(df[fixed_key])
    
    ## GETTING COLORS
    colors = plot_tools.get_cmap(len(unique_labels))
    
    ## LOOPING AND GETTING X Y
    for label_idx, each_label in enumerate(unique_labels):
        ## GETTING ALL EQUIVALENT MATCHING
        current_df = df.loc[df[fixed_key] == each_label]
        ## GETTING X Y
        x = current_df[x_key]
        y = current_df[y_key]
        color = colors(label_idx)
        ## PLOTTING
        ax.plot(x, y, color = color, label = str(each_label) )
        
    ## ADDING LEGEND
    ax.legend()
    
    ## TIGHT LAYOUT
    fig.tight_layout()
    return fig, ax

### FUNCTION TO VARY HDBSCAN CLUSTERING ALGORITHM
def vary_parameters_hdbscan(grid_values,
                            min_sample_array = np.arange(2, 10, 1)):
    '''
    This function varies the parameters for hdbscan algorithm.
    INPUTS:
        grid_values: [np.array]
            grid values to run the hdbscan clustering
        min_sample_array: [np.array]
            minimum sample array to vary
    OUTPUTS:
        df: [pd.DataFrame]
            dataframe containing all the clusters
    '''
    ## CREATING STORAGE DICT
    storage_dict = []
    
    ## LOOPING THROUGH EACH ONE
    for min_samples in min_sample_array:
        
        ## GETTING CLUSTER
        hdbscan_cluster, labels, num_clusters = compute_hdbscan_cluster(grid_values,
                                                                        min_samples = int(min_samples),
                                                                        verbose = True)
        
        
        ## PRINTING
        print("Computing clusters for min_samples (%d) --> %d"%( min_samples, num_clusters ))
        
        ## STORING
        storage_dict.append(
                {'min_samples': min_samples,
                 'num_clusters': num_clusters}
                )
    ## CREATING DATAFRAME
    df = pd.DataFrame(storage_dict)
    
    return df

### FUNCTION TO PLOT VARYING DBSCAN FACTORS
def plot_vary_parameters_hdbscan(df,
                                x_key = "min_samples",
                                y_key = "num_clusters",
                                fig_size_cm = FIGURE_SIZE):
    '''
    This function plots the varying parameters for dbscan
    INPUTS:
        df: [pd.dataframe]
            dataframe containing varying parameters
        x_key: [str]
            key in dataframe that will be the x value
        y_key: [str]
            key in dataframe that will be the y value
        fixed_key: [str]
            fixed key in minimum samples
        fig_size_cm: [tuple]
            figure size in centimeters
    OUTPUTS:
        fig, ax: 
            figure and axis
    '''    
    ## CREATING FIGURE
    fig, ax = plot_tools.create_fig_based_on_cm(fig_size_cm)
    
    ## MAKING LABELS
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    
    ## GETTING X Y
    x = df[x_key]
    y = df[y_key]
    
    ## PLOTTING
    ax.plot(x, y, color = 'k')
    
    ## TIGHT LAYOUT
    fig.tight_layout()
    return fig, ax

### CLASS FUNCTION TO ANALYZE CLUSTERING ALGORITHM
class debug_clustering_algorithm:
    '''
    This function is designed to debug the clustering algorithm. 
    The idea would be to vary min samples and cutoff for dbscan, etc.
    INPUTS:
        
    OUTPUTS:
        
    '''
    ## INITIALIZING
    def __init__(self,
                 ):
        
        
        return
    
    ## FUNCTION TO GET DISTANCE BETWEEN GRID POINTS
    @staticmethod
    def get_dist_btn_grid_values(grid_values):
        '''
        This function is designed to get distance matrix between grid points.
        INPUTS:
            grid_values: [np.array]
                grid values in N x 3
        OUTPUTS:
            distances: [np.array, shape = (N, N)]
                distances between pairs of grid values
            distances_upper_triangular: [np.array, shape = M]
                distances occurring in the upper triangular of paired distances. 
                This is useful when trying to plot the different distances
        '''
        ## COMPUTING DISTANCES
        distances = squareform(pdist(grid_values, 'euclidean'))
        
        ## GETTING UPPER TRIANGULAR DISTANCE
        upper_triangular = np.triu_indices(len(distances), k = 1)
        distances_upper_triangular = distances[upper_triangular]
        return distances, distances_upper_triangular
    
    ## FUNCTION TO GENERATE DISTANCE HISTOGRAM
    def compute_dist_histogram(self,
                               grid_values,
                               bin_width = 0.02,
                               r_range = None):
        '''
        This function computes the distance histogram. The bins assume 
        min and max of distances.
        INPUTS:
            grid_values: [np.array, M]
                grid values
        OUTPUTS:
            x: [np.array]
                bins that are averaged between bins
            g_r: [np.array]
                normalized histogram, same as radial distribution function
            hist: [np.array]
                histogram output
        '''
        
        ## COMPUTING DISTANCES
        distances, distances_upper_triangular = self.get_dist_btn_grid_values(grid_values)
        
        ## DEFINING DISTANCE TO USE
        dist = distances_upper_triangular[:]
        
        ## DEFINING R RANGE
        if r_range is None:
            r_range = np.array([dist.min(), dist.max()])

        ## DEFINING BINS
        bins = np.arange(r_range[0], r_range[1], bin_width)
        
        ## GETTING HISTOGRAM OF DISTANCES
        hist, bin_edges = np.histogram(dist, bins = bins, density=False)
        
        ## DEFINING X
        x = (bin_edges[1:] + bin_edges[:-1])/2
        ## GETTING VOLUME            
        V = (4 / 3) * np.pi * (np.power(bin_edges[1:], 3) - np.power(bin_edges[:-1], 3))
        
        ## FINDING NORM
        norm = len(distances) * V
        
        ## NORMALIZING
        g_r = hist.astype(np.float64) / norm  # From int64.
        
        return x, g_r, hist
    
    ## FUNCTION TO GENERATE DISTANCE PDF
    @staticmethod
    def plot_distance_pdf(x,
                          hist,
                          fig_size_cm = FIGURE_SIZE):
        ''' This plots density function ofr distances '''
        ## CREATING FIGURE
        fig, ax = plot_tools.create_fig_based_on_cm(fig_size_cm)
        
        ## AXIS
        ax.set_xlabel("Distance (nm)")
        ax.set_ylabel("PDF")
        
        ## PLOTTING
        ax.plot(x, hist, color ='k')
    
        ## TIGHT LAYOUT
        fig.tight_layout()
        
        return fig, ax
    
#%% MAIN FUNCTION
if __name__ == "__main__":
    
    ## DEFINING KEY
    sim_key = "GNP_unsaturated"
    # 'GNP_unsaturated'
    # "GNP"
    
    ## DEFINING PATH TO SIM
    parent_sim_path=os.path.join(PARENT_SIM_PATH,
                             PATH_DICT[sim_key],# GNP # GNP_unsaturated
                             )
    ## LOOKING INTO LIST
    sim_list = glob.glob(parent_sim_path + "/*")
    
    ## DEFINNIG PATH TO SIM
    if sim_key == "GNP":
        path_to_sim = sim_list[3]
    else:
        path_to_sim = sim_list[-2]
    
    ## DEFINING CLUSTERING INPUTS
    clustering_inputs = {
            'path_to_sim' : path_to_sim,
            'cutoff': 11.25,  # 11,
            'min_samples': 6,
            'eps': 0.40,
            "clustering_type": "dbscan",
            }
    
    ## RUNNING CLUSTERING ALGORITHM
    cluster_obj, labels, clustering, num_clusters, idx_above, idx_below = main_mu_clustering(**clustering_inputs)
    
    ## DEFINING BASENAME
    sim_basename = os.path.basename(path_to_sim)
    
    ## DEFINING GRID_VALUES
    grid_values = clustering.grid_values
    
    #%% PLOTTING GRID  
    ## DEFINING PREFIX
    ## IMPORTING
    from mayavi import mlab
    ## PLOTTING POINTS
    figure = plot_3d_points_with_scalar(xyz_coordinates = clustering.grid_values, 
                                        scalar_array = clustering.mu_values,
                                        points_dict={
                                               'scale_factor': 0.5,
                                               'scale_mode': 'none',
                                               'opacity': 1.0,
                                               'mode': 'sphere',
                                               'colormap': 'blue-red',
                                               'vmin': 9,
                                               'vmax': 11
                                               },
                                        figure = None)
    
    fig_name = "1-%s-sat_grids"%(sim_basename)
    ## SAVING IMAGE
    mlab.savefig(os.path.join(PATH_FIG, fig_name + ".png"))
    
    #%% DEBUG CLUSTERING
    
    ## GETTING DEBUG ALGORITHM
    debug_cluster = debug_clustering_algorithm()
    
    ''' METHOD FOR GETTING DISTANCE PDFs
    ## GETTING DISTANCES
    distances, distances_upper_triangular = debug_cluster.get_dist_btn_grid_values(grid_values = grid_values[idx_above])
    
    ## DEFINING DISTANCES FOR HISTOGRAM
    x, g_r, hist = debug_cluster.compute_dist_histogram(grid_values = grid_values,
                                                   bin_width = 0.02,
                                                   r_range = (0.05,1))
    
    ## GETTING DISTANCE PDF
    fig, ax = debug_cluster.plot_distance_pdf(x = x, hist = g_r)
    
    '''
    ## COMPUTING DATAFRAME FOR DIFFERENT PARAMETERS
    df = vary_parameters_dbscan(grid_values[idx_above],
                                eps_array = np.arange(0.1, 0.55, 0.05),
                                min_sample_array = np.arange(2, 10, 1))
    

    
    #%% PLOTTING FOR DIFFERENT EPS VALUES

    ## PLOTTING VARY PARAMETERS
    fig, ax = plot_vary_parameters_dbscan(df,
                                    x_key = "eps",
                                    y_key = "num_clusters",
                                    fixed_key = "min_samples",
                                    fig_size_cm = FIGURE_SIZE)
    
    figure_name = '2-%s-plot_vary_params'%(sim_basename)
    plot_tools.store_figure(fig = fig, 
                 path = os.path.join(PATH_FIG,
                                     figure_name), 
                 fig_extension = 'png', 
                 save_fig=True,)
    
    
    #%% DBSCAN CLUSTERING
    
    ## COMPUTING CLUSTERS
    db, labels, num_clusters = compute_dbscan_cluster(grid_values[idx_above],
                                                      eps = 0.4,
                                                      min_samples = 6,
                                                      verbose = True)
    
    ## GETTING FIGURE FOR CLUSTERS
    figure = plot_3d_points_clusters(xyz_coordinates = clustering.grid_values[idx_above],
                                     labels = labels,
                                     title = "Clusters: %d"%(num_clusters))
    
    
    fig_name = "3-%s-result_eps_0.4_minsamples_6"%(sim_basename)
    ## SAVING IMAGE
    mlab.savefig(os.path.join(PATH_FIG, fig_name + ".png"))
    
    #%% VARYING HDBSCAN CLUSTERING
    
    ## COMPUTING HDBSCAN
    df_hdbscan = vary_parameters_hdbscan(grid_values[idx_above],
                                         min_sample_array = np.arange(2, 10, 1))
    
    ## PLOTTING
    fig, ax = plot_vary_parameters_hdbscan(df = df_hdbscan,
                                           x_key = "min_samples",
                                           y_key = "num_clusters",
                                           fig_size_cm = FIGURE_SIZE )
    
    figure_name = '4-%s-hdbscan_vary'%(sim_basename)
    plot_tools.store_figure(fig = fig, 
                 path = os.path.join(PATH_FIG,
                                     figure_name), 
                 fig_extension = 'png', 
                 save_fig=True,)
    
    
    
    #%% HDBSCAN CLUSTERING
    
    ## DEFINING PARAMETERS
    min_samples = 3
    
    ## COMPUTING CLUSTERS
    db, labels, num_clusters = compute_hdbscan_cluster(grid_values[idx_above],
                                                       min_samples = min_samples,
                                                       verbose = True)
    
    ## GETTING FIGURE FOR CLUSTERS
    figure = plot_3d_points_clusters(xyz_coordinates = clustering.grid_values[idx_above],
                                     labels = labels,
                                     title = "Clusters: %d"%(num_clusters))
    
    fig_name = "5-%s-result_hdbscan_%d"%(sim_basename, min_samples)
    ## SAVING IMAGE
    mlab.savefig(os.path.join(PATH_FIG, fig_name + ".png"))
    
                #%%
    ## DEFINING FIGURE NAME
    fig_name = prefix + "Mu_values"
    ## SAVING IMAGE
    mlab.savefig(os.path.join(PATH_FIG, fig_name + ".png"))
    
    
    
    #%%
    
    

    #%%
    
    
    ## DEFINING LOCATION
    path_list = {
                    'Unsaturated': {
                                    'path_dict_key': 'GNP_unsaturated',
                                    'prefix_key': 'GNP',
                                    'yticks': np.arange(0, 2.5, 0.5),
                                    'ylim': [-0.1, 2.25],
                                    'ligands': ['C11double67OH'],
                                        },
                    'Saturated': {
                                    'path_dict_key': 'GNP',
                                    'prefix_key': 'GNP',
                                    'yticks': np.arange(0, 2.5, 0.5),
                                    'ylim': [-0.1, 2.25],
                                    'ligands': ['C11OH'],
                                        },
                    'Branched': {
                                    'path_dict_key': 'GNP_branched',
                                    'prefix_key': 'GNP',
                                    'yticks': np.arange(0, 2.5, 0.5),
                                    'ylim': [-0.1, 2.25],
                                    'ligands': ['C11branch6OH'],
                                        },
                }
                    
    ## DEFINING GRID DETAILS
    wc_analysis=DEFAULT_WC_ANALYSIS
    mu_pickle = MU_PICKLE

    ## LOADING MU VALUES
    storage_dict = load_mu_values_for_multiple_ligands(path_list = path_list,
                                                       ligands = [],
                                                       main_sim_dir=MAIN_HYDRO_DIR,
                                                       want_mu_array = True,
                                                       want_stats = False,
                                                       want_grid = True,
                                                       )
    
    #%%
    
    ## DEFINING MU VALUES AND GRID VALUES
    mu_values = storage_dict['Unsaturated']['C11double67OH']['mu']
    grid_values = storage_dict['Unsaturated']['C11double67OH']['grid']

    mu_values = storage_dict['Saturated']['C11OH']['mu']
    grid_values = storage_dict['Saturated']['C11OH']['grid']
    
    
    ## LOOPING
    for each_key in storage_dict:
        for each_lig in storage_dict[each_key]:
            ## DEFINING MU AND GRID
            mu_values = storage_dict[each_key][each_lig]['mu']
            grid_values = storage_dict[each_key][each_lig]['grid']
            
            ## RUNNING CLUSTERING
            clustering = cluster_mu_values(mu_values = mu_values,
                                           grid_values = grid_values)
            
            ## DIVIDING
            idx_below, idx_above = clustering.divide_mu_values_by_index(cutoff = 11)
            
            ## GETTING CLUSTER
            hdbscan_cluster, labels, num_clusters = clustering.run_hdbscan_cluster(grid_values[idx_above],
                                                                     min_samples = 4,
                                                                     verbose = True)
            
            
            ## DEFINING PREFIX
            prefix = "%s-%s-"%(each_key,each_lig)
            ## PLOTTING POINTS
            figure = plot_3d_points_with_scalar(xyz_coordinates = grid_values, 
                                                scalar_array = mu_values,
                                                points_dict={
                                                       'scale_factor': 0.5,
                                                       'scale_mode': 'none',
                                                       'opacity': 1.0,
                                                       'mode': 'sphere',
                                                       'colormap': 'blue-red',
                                                       'vmin': 9,
                                                       'vmax': 11
                                                       },
                                                figure = None)
                        
            ## DEFINING FIGURE NAME
            fig_name = prefix + "Mu_values"
            ## SAVING IMAGE
            mlab.savefig(os.path.join(PATH_FIG, fig_name + ".png"))
            
            
            ## GETTING FIGURE FOR CLUSTERS
            figure = plot_3d_points_clusters(xyz_coordinates = clustering.grid_values[idx_above],
                                             labels = labels,
                                             title = "Clusters: %d"%(num_clusters))
            
            fig_name = prefix + 'Clusters'
            ## SAVING IMAGE
            mlab.savefig(os.path.join(PATH_FIG, fig_name + ".png"))
        
            ## CLUSTERTINB ELOW
            ## DIVIDING
            idx_below, idx_above = clustering.divide_mu_values_by_index(cutoff = 10)
            
            ## GETTING CLUSTER
            hdbscan_cluster, labels, num_clusters = clustering.run_hdbscan_cluster(grid_values[idx_below],
                                                                     min_samples = 4,
                                                                     verbose = True)
            
            ## GETTING FIGURE FOR CLUSTERS
            figure = plot_3d_points_clusters(xyz_coordinates = clustering.grid_values[idx_below],
                                             labels = labels,
                                             title = "Clusters: %d"%(num_clusters))
            
            fig_name = prefix + 'Clusters_BELOW'
            ## SAVING IMAGE
            mlab.savefig(os.path.join(PATH_FIG, fig_name + ".png"))
            

    
    #%% PLOTTING GRID VALUES
    
    fig, ax = plot_grid_points(grid_values,
                               scatter_style = {
                                       's'         : 1,
                                       'linewidth' : 0.5,
                                       'color'     : 'k',
                                       })
    
    #%% PLOTTING MU DISTRIBUTION IN 3D

    ## PLOTTING POINTS
    figure = plot_3d_points_with_scalar(xyz_coordinates = grid_values, 
                                        scalar_array = mu_values,
                                        points_dict={
                                               'scale_factor': 0.5,
                                               'scale_mode': 'none',
                                               'opacity': 1.0,
                                               'mode': 'sphere',
                                               'colormap': 'blue-red',
                                               'vmin': 9,
                                               'vmax': 11
                                               })
    ## SETTING VIEW              
    # mlab.view(azimuth=0, elevation=0, distance = 20, focalpoint=np.mean(grid_values,axis=0), figure = figure)
    ## DEFINING VIEW FROM SATURATED C11OH
    view = ((-170.8918413945595,
             60.77512687457703,
             19.999999999999996,
             np.array([3.6231738 , 3.59103177, 3.53831992])),
                -35.43910393663796)
    
    '''
    (mlab.view(figure=figure), mlab.roll(figure=figure))
    '''
    mlab.view(*view[0], figure=figure)
    mlab.roll(view[1], figure=figure)
    ## DEFINING FIGURE NAME
    fig_name = "1_C11OH_sat_girds"
    ## SAVING IMAGE
    mlab.savefig(os.path.join(PATH_FIG, fig_name + ".png"))
    
    #%% PLOTTING MU DISTRIBUTION
    fig, ax = plot_scatter_mu_values(mu_values,
                                     fig_size = FIGURE_SIZE)
    
    
    ## SETTING AXIS
    figure_name = '2_mu_no_order'
    plot_tools.store_figure(fig = fig, 
                 path = os.path.join(PATH_FIG,
                                     figure_name), 
                 fig_extension = 'png', 
                 save_fig=True,)
    
    #%% PLOTTING MU DISTRIBUTION THAT IS SORTED
    fig, ax = plot_scatter_mu_values(np.sort(mu_values),
                                     fig_size = FIGURE_SIZE)
    
    ## GETTING ORDER
    figure_name = '3_mu_with_order'
    plot_tools.store_figure(fig = fig, 
                 path = os.path.join(PATH_FIG,
                                     figure_name), 
                 fig_extension = 'png', 
                 save_fig=True,)
    
                 
    #%% IMAGE 4
    
    ## GETTING ALL MU VALUES AT A SPECIFIC VALUE
    cutoff = 11 # kT
    idx_above = np.where(mu_values >= cutoff)
    
    ## PLOTTING POINTS
    figure = plot_3d_points_with_scalar(xyz_coordinates = grid_values[idx_above], 
                                        scalar_array = mu_values[idx_above],
                                        points_dict={
                                               'scale_factor': 0.4,
                                               'scale_mode': 'none',
                                               'opacity': 1.0,
                                               'mode': 'sphere',
                                               'colormap': 'blue-red',
                                               'vmin': 9,
                                               'vmax': 11
                                               })
                    
    mlab.view(*view[0], figure=figure)
    mlab.roll(view[1], figure=figure)
    ## DEFINING FIGURE NAME
    fig_name = "4_shrink_3d_grids"
    ## SAVING IMAGE
    mlab.savefig(os.path.join(PATH_FIG, fig_name + ".png"))
    
    
    #%% 
    
    min_samples = 4
    hdbscan_cluster = hdbscan.HDBSCAN(algorithm='best', 
                                      alpha=1.0, 
                                      approx_min_span_tree = True,
                                      gen_min_span_tree = False,
                                      leaf_size = 40,
                                      min_cluster_size = min_samples,
                                      metric='euclidean').fit(grid_values[idx_above])
    
    ## GETTING LABELS
    labels = hdbscan_cluster.labels_
    
    
    ## PRINTING TOTAL NUMBER OF CLUSTERS
    print("Total number of clusters: %d"%( len(np.unique(labels))-1 ) )
    
    #%% PLOTTING
    
    ## GETTING FIGURE
    figure = plot_3d_points_clusters(xyz_coordinates = grid_values[idx_above],
                                     labels = labels)
    
    ## CHANGING VIEW    
    mlab.view(*view[0], figure=figure)
    mlab.roll(view[1], figure=figure)
    ## DEFINING FIGURE NAME
    fig_name = "5_hdbscan_clustered"
    ## SAVING IMAGE
    mlab.savefig(os.path.join(PATH_FIG, fig_name + ".png"))
    

    #%% COMPUTING DIFFERENCES IN MU VALUE AS A GRID MATRIX

    
    ## GETTING MATRIX WITH MU
    def compute_mu_similarity_matrix(mu_values):
        '''
        The purpose of this function is to compute the similarity between 
        mu matrices. The idea would be that we want to distinguish our data 
        based on the mu values that are inputted. 
        INPUTS:
            mu_values : [np.array, shape = N grid points]
                mu values outputted from density fluctuation calculations
                
        OUTPUTS:
            mu_similarity_matrix: [np.array, shape = (N,N)]
                similarity matrix of mu values
                
        ## SIMILARITY FOR DISTANCES
        ## DISTANCES BETWEEN THE GRID VALUES
        distances = squareform(pdist(grid_values, 'euclidean'))
            
        '''
        ## EXPANDING TO NEXT AXIS
        mu_values_expanded = np.expand_dims(mu_values, 1)
        
        ## SUBTRACTING AND TAKING ABSOLUTE VALUE
        mu_similarity_matrix = np.abs(mu_values - mu_values_expanded)
        return mu_similarity_matrix
    
    ## COMPUTING SIMILARITY OF MU
    mu_similarity_matrix = compute_mu_similarity_matrix(mu_values = mu_values)
        
    
    #%% NORMALIZING

    from sklearn.preprocessing import MinMaxScaler
    
    ## NORMALIZE DISTANCE ARRAY
    distance_scalar = MinMaxScaler()
    distances_norm = distance_scalar.fit_transform(distances)
    
    ## NORMALIZE MU DISTRIBUTION ARRAY
    mu_scalar = MinMaxScaler()
    mu_similarity_matrix_norm = mu_scalar.fit_transform(mu_similarity_matrix)
    
    ## GETTING AVERAGE MU
    combined_similarity = mu_similarity_matrix_norm[:]
    combined_similarity = np.average( (distances_norm, mu_similarity_matrix_norm), axis = 0, weights = [0.5, 0.5])
    
    #%% HDBSCAN
    
    ## IMPORTING HDBSCAN
    ## USAGE INFORMATION: http://hdbscan.readthedocs.io/en/latest/basic_hdbscan.html
    import hdbscan
    min_samples = 4
    metric = 'precomputed'
    hdbscan_cluster = hdbscan.HDBSCAN(algorithm='best', 
                                      alpha=1.0, 
                                      approx_min_span_tree = True,
                                      gen_min_span_tree = False,
                                      leaf_size = 40,
                                      min_cluster_size = min_samples,
                                      metric=metric).fit(combined_similarity)
    
    ## GETTING LABELS
    labels = hdbscan_cluster.labels_
    
    #%% USING DBSCAN
    
    ## IMPORTING SKLEARN MODULES
    from sklearn.cluster import DBSCAN
    
    eps = 0.04 # kT units
    min_samples = 10
    ## RUNNING DBSCAN
    # db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(mu_similarity_matrix)
    db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(combined_similarity)
    
    ## GETTING LABELS
    labels = db.labels_

    #%% PLOTTING REGIONS THAT ARE CLUSTERED
    
    ## CLOSING MLAB
    try:
        mlab.close()
    except AttributeError:
        pass
    
    ## CREATING FIGURE
    mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
    
    ## PLOTTING POINTS
    pts = mlab.points3d(
                    grid_values[:,0], 
                    grid_values[:,1],   
                    grid_values[:,2], 
                labels,
                scale_factor=0.5, 
                scale_mode = 'none',
                opacity=1.0,
                mode = "sphere", # "sphere",
                colormap='blue-red',
                 )

    
    
    
    
    
    
    
    
    