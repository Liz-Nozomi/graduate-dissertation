# -*- coding: utf-8 -*-
"""
pca_debug.py
"""
##############################################################################
## IMPORTING MODULES
##############################################################################
## IMPORT OS
import os
## IMPORT RANDOM
import random
## IMPORT NUMPY
import numpy as np
## IMPORT PCA AND REGRESSION FUNCTIONS FROM SKLEARN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
## FUNCTION TO SAVE AND LOAD PICKLE FILES
from MDDescriptors.application.sams.pickle_functions import load_pkl
## PLOTTING WITH MATPLOTLIB
import matplotlib.pyplot as plt
## PLOT DATA FUNCTION
from MDDescriptors.application.sams.sam_create_plots import set_plot_defaults
##############################################################################
## FUNCTIONS
##############################################################################
def plot_parity( x, y,
                 title = "",
                 path_fig = None,
                 colors = [],
                 **kwargs ):
    r"""
    """
    ## SET PLOT DEFAULT
    plot_details = set_plot_defaults("jacs")
        
    ## CREATING LINE PLOT
    fig, ax = plt.subplots()
    fig.subplots_adjust( left = 0.15, bottom = 0.16, right = 0.99, top = 0.97 )
    # SET PLOT TITLE USING RMSE
    plt.title( "RMSE: {:.2f}".format( RMSE ) )
    # SET X AND Y LABELS
    ax.set_xlabel( "$\mu_{indus}$" )
    ax.set_ylabel( "$\mu_{predicted}$" )
    ## PLOT X-Y LINE
    plt.plot( [ 0, 130 ], [ 0, 130 ],
              linestyle = ":",
              linewidth = 1.5,
              color = "darkgray" )
    if len(colors) > 1:
        ## PLOT DATA
        for xi, yi, ci in zip( x, y, colors ):
            plt.scatter( xi, yi,
                         marker = "s",
                         color = ci )
    else:
        ## PLOT DATA
        plt.scatter( x, y,
                     marker = "s",
                     color = "dimgray" )
    ## SET X AND Y TICKS
    x_diff = 20
    x_min = 20
    x_max = 120
    x_lower = np.floor( x_min - 0.5*x_diff )
    x_upper = np.ceil( x_max + 0.5*x_diff )
    ax.set_xlim( x_lower, x_upper )
    ax.set_xticks( np.arange( x_min, x_max + x_diff, x_diff ), minor = False )       # sets major ticks
    ax.set_xticks( np.arange( x_lower, x_max + 1.5*x_diff, x_diff ), minor = True )  # sets minor ticks
    
    y_diff = 20
    y_min = 20
    y_max = 120
    y_lower = np.floor( y_min - 0.5*y_diff )
    y_upper = np.ceil( y_max + 0.5*y_diff )
    ax.set_ylim( y_lower, y_upper )
    ax.set_yticks( np.arange( y_min, y_max + y_diff, y_diff ), minor = False )       # sets major ticks
    ax.set_yticks( np.arange( y_lower, y_max + 1.5*y_diff, y_diff ), minor = True )  # sets minor ticks
    
    ## SET FIGURE SIZE AND LAYOUT
    fig.set_size_inches( plot_details.width, plot_details.height )
    fig.tight_layout()
    if path_fig is not None:
        print( "FIGURE SAVED TO: %s" % path_fig )
        fig.savefig( path_fig, dpi = 300, facecolor = 'w', edgecolor = 'w' )
        
## FUNCTION TO PLOT LINE
def plot_line( x, y,
               path_fig = None,
               **kwargs ):
    r"""
    """
    ## SET PLOT DEFAULT
    plot_details = set_plot_defaults("jacs")
        
    ## CREATING LINE PLOT
    fig, ax = plt.subplots()
    fig.subplots_adjust( left = 0.15, bottom = 0.16, right = 0.99, top = 0.97 )
    # SET X AND Y LABELS
    ax.set_xlabel( "Num. Components" )
    ax.set_ylabel( "RMSE (kT)" )
    ## PLOT DATA
    plt.plot( x, y,
              linestyle = "-",
              linewidth = 1.5,
              color = "dimgray",
              marker = "s" )
    ## SET X AND Y TICKS
    x_diff = 2
    x_min = 0
    x_max = 16
    x_lower = np.floor( x_min - 0.5*x_diff )
    x_upper = np.ceil( x_max + 0.5*x_diff )
    ax.set_xlim( x_lower, x_upper )
    ax.set_xticks( np.arange( x_min, x_max + x_diff, x_diff ), minor = False )       # sets major ticks
    ax.set_xticks( np.arange( x_lower, x_max + 1.5*x_diff, x_diff ), minor = True )  # sets minor ticks
    
    y_diff = 1
    y_min = 2
    y_max = 12
    y_lower = np.round( y_min - 0.5*y_diff, decimals = 2 )
    y_upper = np.round( y_max + 0.5*y_diff, decimals = 2 )
    ax.set_ylim( y_lower, y_upper )
    ax.set_yticks( np.arange( y_min, y_max + y_diff, y_diff ), minor = False )       # sets major ticks
    ax.set_yticks( np.arange( y_lower, y_max + 1.5*y_diff, y_diff ), minor = True )  # sets minor ticks
    
    ## SET FIGURE SIZE AND LAYOUT
    fig.set_size_inches( plot_details.width, plot_details.height )
    fig.tight_layout()
    if path_fig is not None:
        print( "FIGURE SAVED TO: %s" % path_fig )
        fig.savefig( path_fig, dpi = 300, facecolor = 'w', edgecolor = 'w' )

## PCA WITH EIGEN DECOMPOSITION
class pca():
    ## INITIALIZE CLASS
    def __init__( self, X ):
        r'''
        Parameters
        ----------
        X : Unnormalized data for PCA
        
        Returns
        -------
        None.

        '''
        ## STORE DATA IN CLASS
        self.X = X
        ## GET DETAILS ABOUT X
        self.n, self.m = self.X.shape
        ## COMPUTE COLUMN AVERAGE
        self.mu_ = X.mean(axis=0)
        ## COMPUTE COLUMN STDEV
        self.std_ = X.std(axis=0)
        ## COMPUTE PCA OF RESCALED DATA
        self.X_pca = self.fit_transform( X )
    
    ## FUNCTION TO NORMALIZE DATA
    def normalize_data( self, X ):
        r'''
        Normalizes X such that mean = 0 and var = 1
        '''
        ## CREATE PLACEHOLDER ZEROS
        X_rescaled = np.zeros_like( X )
        ## CREATE MATRIX OF STDEV
        stdev_mat = np.tile( self.std_, reps = ( X.shape[0], 1 ) )
        ## MASK OUT ZERO STDEV (PREVENT DIVISION BY ZERO WARNING)
        mask = stdev_mat > 0.
        ## RESCALE X DATA
        X_rescaled[mask] = ( X - self.mu_ )[mask] / stdev_mat[mask]
        ## RETURN RESULT
        return X_rescaled
        
    ## FUNCTION TO COMPUTE PCA OF DATA
    def fit_transform( self, X ):
        r'''
        Computes PCA of rescaled data
        '''
        ## NORMALIZE DATA SUCH THAT MEAN = 0 AND VAR = 1
        self.X_rescaled_ = self.normalize_data( X )
        ## COMPUTE COVARIANCE MATRIX
        cov_mat = np.dot( self.X_rescaled_.transpose(), self.X_rescaled_ ) / ( self.n - 1. )
        ## EIGEN DECOMPOSITION
        eigen_vals, eigen_vecs = np.linalg.eig( cov_mat )
        ## ONLY CONSIDER THE REAL PARTS
        self.eigen_vals = np.real( eigen_vals )
        self.eigen_vecs = np.real( eigen_vecs )
        ## PROJECT X ONTO PC SPACE
        X_pca = -1. * np.dot( self.X_rescaled_, self.eigen_vecs[:,:np.min([self.n,self.m])] )
        ## RETURN RESULTS
        return X_pca
        
    
    
#%%
##############################################################################
## MAIN SCRIPT
##############################################################################
if __name__ == "__main__":
    ## REWRITE
    save_fig = False
    plot_title = None
    subtract_normal = False
    ## INPUT DIRECTORY
    input_dir = r"C:\Users\bdallin\Box Sync\univ_of_wisc\manuscripts\mixed_polar_sams\figure_pca\raw_data"
    ## OUTPUT DIRECTORY
    output_dir = r"C:\Users\bdallin\Box Sync\univ_of_wisc\manuscripts\mixed_polar_sams\figure_pca"
    out_name = "pca_regression_prediction_random"
    path_png = os.path.join( output_dir, out_name + r".png" )
    path_svg = os.path.join( output_dir, out_name + r".svg" )
    ## FILES
    X_file = os.path.join( input_dir, r"triplet_distribution_pca_input.pkl" )
    y_file = os.path.join( input_dir, r"hydration_fe_pca_input.pkl" )
    ## LOAD X & Y DATA
    X_raw = load_pkl( X_file )
    y_raw = load_pkl( y_file )
    ## COMPILE LIST OF SHARED KEYS
    keys = [ key for key in X_raw.keys() if key in y_raw.keys() ]
    ## LOOP THROUGH KEYS TO GATHER DATA FOR PCA
    X = []
    y = []
    for ii, key in enumerate(keys):
            X.append( X_raw[key][:,1] )
            y.append( y_raw[key] )
    ## TRANSFORM INTO COLUMN MATRICES
    N, K = len(keys), len(X[0])
    X = np.array(X).reshape((N,K))
    y = np.array(y).reshape((N,1))
    ## STORE DATA INDICES
    indices = np.arange(N).astype("int")

    ## PCA
    pca = pca( X )
    X_rescaled = pca.X_rescaled_
    X_pca = pca.X_pca
    
    ## COMPARE WITH SKLEARN PCA    
    ## INITIALIZE CLASSES
    scaler    = StandardScaler()        # rescaling class
    pca_test  = PCA( n_components = np.min([ N, K ]) ) # pca class
    regressor = LinearRegression()      # linear regression class
    ## RESCALE TRAINING AND TEST DATA SETS
    X_rescaled_test = scaler.fit_transform( X )
    ## PERFORM PCA ON TRAINING SET
    X_pca_test = pca_test.fit_transform( X_rescaled )

