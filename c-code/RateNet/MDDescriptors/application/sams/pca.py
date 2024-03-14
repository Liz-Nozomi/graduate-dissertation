# -*- coding: utf-8 -*-
"""
pca.py
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
    colors = []
    for ii, key in enumerate(keys):
            X.append( X_raw[key][:,1] )
            y.append( y_raw[key] )
            if 'CONH2' in key:
                colors.append( "darkseagreen" )
            elif 'NH2' in key:
                colors.append( "slateblue" )
            elif 'OH' in key:
                colors.append( "tomato" )
            else:
                colors.append( "dimgrey" )
    ## TRANSFORM INTO COLUMN MATRICES
    N, K = len(keys), len(X[0])
    X = np.array(X).reshape((N,K))
    y = np.array(y).reshape((N,1))
    ## STORE DATA INDICES
    indices = np.arange(N).astype("int")
    # ## K PREDICTS REAL
    # indices_k = []
    # for ii, key in zip(indices,keys):
    #     if "k" in key:
    #         indices_k.append(ii)
    # testing_chunks = [ np.array(indices_k) ]  
    # y_orig = y[indices_k]          
    
    # N-FOLD CROSS VALIDATING LOOP (RANDOMIZE AND SHUFFLE)
    n_fold = 5
    ## SHUFFLE INDICES
    random.shuffle( indices )
    y_orig = y[indices]
    colors = [ colors[ii] for ii in indices ]
    ## SPLIT INDICES INTO N_FOLD LISTS
    k, m = divmod( len(indices), n_fold )
    testing_chunks = [ indices[ii*k + min( ii, m ):(ii + 1)*k + min( ii+1, m)] for ii in range(n_fold) ]
    
    # LOOP THROUGH INDICES
    y_predict = np.empty( shape = (0,1) )
    for testing_indices in testing_chunks:
        training_indices = np.array([ ii for ii in indices if ii not in testing_indices ])
        ## SPLIT DATA INTO TRAINING AND TESTING DATA
        X_train = X[training_indices,:]
        X_test  = X[testing_indices,:]
        y_train = y[training_indices,:]
        y_test  = y[testing_indices,:]    
        ## INITIALIZE CLASSES
        scaler    = StandardScaler()        # rescaling class
        pca       = PCA( n_components = 8 ) # pca class
        regressor = LinearRegression()      # linear regression class
        ## RESCALE TRAINING AND TEST DATA SETS
        X_train_rescaled = scaler.fit_transform( X_train )
        X_test_rescaled  = scaler.transform( X_test )
        ## PERFORM PCA ON TRAINING SET
        X_train_pca = pca.fit_transform( X_train_rescaled )
        ## PROJECT PCA ON TEST SET
        X_test_pca = pca.transform( X_test_rescaled )
        # ## PRINT TRANSFORMATION DETAILS
        # print( "\n           PCA INFO")
        # print( "--------------------------------" )
        # print("Original shape:", X_train_rescaled.shape)
        # print("Reduced shape: ", X_train_pca.shape)
        # ## CONVERT RATIOS TO PERCENTAGES
        # var = 100 * pca.explained_variance_ratio_
        # ## PRINT OUT DETAILS
        # print( "% CONTRIBUTION BY EACH COMPONENT" )
        # for ii, v in enumerate(var):
        #     print( "{:5s}{:5.1f}%".format( 'PC' + str(ii+1) + ':', v ) )
        # print( "--------------------------------" )
        ## TRAIN LINEAR REGRESSION MODEL ON TRAINING PCA DATA
        regressor.fit( X_train_pca, y_train )
        ## PREDICT Y FROM REGRESSED PCA DATA
        y_predict = np.vstack(( y_predict, regressor.predict( X_test_pca ) ))
        
    ## COMPUTE MSE & RMSE
    MSE = metrics.mean_squared_error( y_orig, y_predict )
    RMSE = np.sqrt(metrics.mean_squared_error( y_orig, y_predict ))
    print( "MSE:", MSE )
    print( "RMSE:", RMSE )
    ## PARITY PLOT
    plot_parity( y_orig, y_predict,
                  title = RMSE,
                  colors = colors,
                  path_fig = path_png )
    plot_parity( y_orig, y_predict,
                  title = RMSE,
                  colors = colors,
                  path_fig = path_svg )
    # ## CHECK RMSE VS N_COMPONENTS
    # RMSE = []
    # n_components = list(range(1,15+1))
    # for n in n_components:
    #     ## LOOP THROUGH INDICES
    #     y_predict = np.empty( shape = (0,1) )
    #     for testing_indices in testing_chunks:
    #         training_indices = np.array([ ii for ii in indices if ii not in testing_indices ])
    #         ## SPLIT DATA INTO TRAINING AND TESTING DATA
    #         X_train = X[training_indices,:]
    #         X_test  = X[testing_indices,:]
    #         y_train = y[training_indices,:]
    #         y_test  = y[testing_indices,:]    
    #         ## INITIALIZE CLASSES
    #         scaler    = StandardScaler()        # rescaling class
    #         pca       = PCA( n_components = n ) # pca class
    #         regressor = LinearRegression()      # linear regression class
    #         ## RESCALE TRAINING AND TEST DATA SETS
    #         X_train_rescaled = scaler.fit_transform( X_train )
    #         X_test_rescaled  = scaler.transform( X_test )
    #         ## PERFORM PCA ON TRAINING SET
    #         X_train_pca = pca.fit_transform( X_train_rescaled )
    #         ## PROJECT PCA ON TEST SET
    #         X_test_pca = pca.transform( X_test_rescaled )
    #         # ## PRINT TRANSFORMATION DETAILS
    #         # print( "\n           PCA INFO")
    #         # print( "--------------------------------" )
    #         # print("Original shape:", X_train_rescaled.shape)
    #         # print("Reduced shape: ", X_train_pca.shape)
    #         # ## CONVERT RATIOS TO PERCENTAGES
    #         # var = 100 * pca.explained_variance_ratio_
    #         # ## PRINT OUT DETAILS
    #         # print( "% CONTRIBUTION BY EACH COMPONENT" )
    #         # for ii, v in enumerate(var):
    #         #     print( "{:5s}{:5.1f}%".format( 'PC' + str(ii+1) + ':', v ) )
    #         # print( "--------------------------------" )
    #         ## TRAIN LINEAR REGRESSION MODEL ON TRAINING PCA DATA
    #         regressor.fit( X_train_pca, y_train )
    #         ## PREDICT Y FROM REGRESSED PCA DATA
    #         y_predict = np.vstack(( y_predict, regressor.predict( X_test_pca ) ))          
    #     ## COMPUTE RMSE
    #     RMSE.append( np.sqrt(metrics.mean_squared_error( y_orig, y_predict )) )

    # ## PLOTTING
    # path_png = None # os.path.join( output_dir, r"pca_regression_validation_random.png" )
    # path_svg = None # os.path.join( output_dir, r"pca_regression_validation_random.svg" )
    # plot_line( n_components, np.array(RMSE),
    #           path_fig = path_png )
    # plot_line( n_components, np.array(RMSE),
    #           path_fig = path_svg )
