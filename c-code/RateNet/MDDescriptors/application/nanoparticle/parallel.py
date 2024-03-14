##############################################################################
# hydrophobicity: A Python Library for analysis of an water-soft material 
#                 interface
#
# Author: Bradley C. Dallin
# email: bdallin@wisc.edu
# Edited by AKC
##############################################################################

##############################################################################
# Imports
##############################################################################

#import sys
import multiprocessing
import numpy as np

__all__ = [ 'parallel' ]

##############################################################################
# Monolayer info class
##############################################################################
def worker( arg ):
    R''' '''
    obj, traj = arg
    return obj.compute( traj )

class parallel:
    R'''
    This class parallelizes functions
    INPUTS:
        traj: [traj]
            trajectory information
        func: [function]
            function to run trajectory on
        args: [dict]
            dictionary of arguments
        average: [logical]
            True if you care about the average ensemble value of a result. 
            If false, the reuslts will just summed up, which is typical 
            for distributions.
    OUTPUTS:
        self.split_traj: [list]
            lits of split trajectories
        self.results: [list or value]
            results outputted for your function
    '''
    def __init__( self, traj, func, args, n_procs = -1, average = True ):
        R''' '''
        ### PRINTING
        print("**** CLASS: %s ****"%(self.__class__.__name__))
        
        ## COMPUTING NUBMER OF PROCESSORS
        if n_procs < 0:
            n_procs = multiprocessing.cpu_count()
        print( "\n--- Running %s on %s cores ---\n" %( func.__name__, str(n_procs) ) )
        
        if traj.time.size < n_procs:
            n_procs = traj.time.size
        
        ## SPLITTING TRAJECTORIES
        traj_list = self.split_traj( traj, n_procs )
        
        ## DELETE ORIGINAL TRAJECTORY REDUCE MEMORY
        del traj 
        
        ## DEFINING OBJECT LIST        
        object_list = self.create_obj_list( func, n_procs, traj_list, args )
        ## CREATING POOL OF WORKERS
        pool = multiprocessing.Pool( processes = n_procs )
        
        ## GENERATE RESULTS LIST
        result_list = pool.map( worker, ( ( obj, trj ) for obj, trj in zip( object_list, traj_list ) ) )
        
        ## closing pools and joins the information
        pool.close()
        pool.join()
        
        ## EITHER AVERAGING OR PRINTING THE RESULTS
        if average: # average outputs, typically what you want for ensemble averages
            self.results = result_list[0] 
            for result in result_list[1:]:
                self.results += result
            self.results /= len(result_list)
        else: # sum outputs, typically what you want for distributions
            self.results = result_list[0] 
            for result in result_list[1:]:
                self.results += result
        return
    
    @staticmethod
    def split_traj( traj, n_procs ):
        R'''
        The purpose of this function is to split the trajectory for multiple 
        processors.
        INPUTS:
            traj: [obj]
                trajectory object from md.load
            n_procs: [int]
                number of processors to divide your object
        OUTPUTS:
            traj_list: [list]
                list of trajectories that are evenly divided
        '''
        traj_list = []
        ## SEE IF NUMBER OF PROCESSORS IS GREATER THAN ONE
        if n_procs > 1:
            ## SPLIT TRAJ INTO N PROCESSES
            len_split = len(traj) // n_procs
            remainder = len(traj) % n_procs
            splits = []
            ## LOOPING THROUGH EACH PROCESSORS AND SPLITTING
            for n in range( n_procs ):
                if n < remainder:
                    splits.append( len_split + 1 )
                else:
                    splits.append( len_split )
            
            ## LOOPING AND STORING THE SPLITS
            current = 0
            for split in splits:
                traj_list.append( traj[current:current+split] )
                current += split
        else:
            traj_list.append( traj )
            
        return traj_list
    
    @staticmethod
    def create_obj_list( func, n_procs, traj_list, args ):
        R'''
        This function creates an object list and runs the function across different 
        processors.
        INPUTS:
            func: [function]
                function that you are trying to run
            n_procs: [int]
                number of processors to run the function on
            traj_list: [list]
                list of trajectories that has been split
            args: [dict]
                dictionary of arguments for the function
        OUTPUTS:
            object_list: [list]
                list of results from the functions
        '''
        ## CREATES A LIST OF CLASS OBJECTS
        object_list = [ func( **args ) for ii in range( 0, n_procs ) ]        
        ## CHECKING IF OBJECT AND TRAJ LIST MATCHES
        if len(object_list) != len(traj_list):
            raise RuntimeError( '\n  ERROR! More objects created than split trajectories' )
        
        return object_list
  