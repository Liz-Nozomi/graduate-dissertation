# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 10:19:00 2020

@author: bdallin
"""

    if check_spyder() is True:
        ## PLOT INTERFACIAL RDF FIGURE
        path_fig = os.path.join( output_dir, r"interfacial_rdf.png" )        
        plot_data( path_fig, data, "interfacial_rdf", 
                   x_label = r"r (nm)",
                   x_ticks = np.arange( 0, 2, 0.2 ),
                   y_label = r"g(r)", 
                   y_ticks = np.arange( 0, 6, 1 ),
                   line_plot = True, 
                   savefig = save_fig )
        
        ## PLOT DENSITY PROFILE FIGURE
        path_fig = os.path.join( output_dir, r"density_profile.png" )        
        plot_data( path_fig, data, "density_profile", 
                   x_label = r"Distance from interface (nm)",
                   x_ticks = np.arange( -0.4, 0.8, 0.2 ),
                   y_label = r"Density $(\rho/\rho_{bulk})$",
                   y_ticks = np.arange( 0, 1.6, 0.2 ),
                   line_plot = True, 
                   savefig = save_fig )

        ## PLOT TRIPLET ANGLE FIGURE
        path_fig = os.path.join( output_dir, r"triplet_angle_distribution.png" )        
        plot_data( path_fig, data, "triplet_angle_distribution", 
                   x_label = r"$\theta$ (degrees)",
                   x_ticks = np.arange( 0, 180, 20 ),
                   y_label = r"$\it{p}(\theta)$", 
                   y_ticks = np.arange( 0, 0.016, 0.004 ),
                   line_plot = True, 
                   savefig = save_fig )

        ## PLOT HBONDS AVERAGE
        path_fig = os.path.join( output_dir, r"hbonds_average.png" )        
        plot_bar( path_fig, 
                  data,
                  "hbonds_average",
                  y_label = r"Num. hbonds per molecule",
                  y_ticks = np.arange( 0, 4.0, 1.0 ), 
                  savefig = False )

        ## PLOT HBONDS TOTAL
        path_fig = os.path.join( output_dir, r"hbonds_total.png" )        
        plot_bar( path_fig, 
                  data,
                  "hbonds_total",
                  y_label = r"Num. hbonds",
                  y_ticks = np.arange( 0, 4.0e3, 1.0e3 ), 
                  savefig = False )   

        ## PLOT HBONDS DISTRIBUTION
        path_fig = os.path.join( output_dir, r"hbonds_distribution.png" )        
        plot_hbond_dist( path_fig, data, "hbonds_distribution", 
                         x_label = r"Num. hbonds per molecule",
                         x_ticks = np.arange( 0, 10, 2 ),
                         y_label = r"$\it{p}(\it{N})$", 
                         y_ticks = np.arange( 0, 1.0, 0.2 ),
                         savefig = False )