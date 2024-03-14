# -*- coding: utf-8 -*-
"""
trjconv_commands.py
The purpose of this function is to run trjconv functions within python. 
The idea is to mainstream the code so it is almost entirely in python. 
In addition, this would be a good way of keeping track of trajectory files. 



Written by: Alex K. Chew (01/31/2020)


References:
    https://stackoverflow.com/questions/4256107/running-bash-commands-in-python

TO TEST:
    Go to path of this file:
        cd /home/akchew/bin/pythonfiles/modules/MDDescriptors/traj_tools
    Run the following:
        python3.6 trjconv_commands.py
    
    This should output the correct inputs

GLOBAL VARIABLES    
TRJCONV_SUFFIX_DICT:
    dictionary containing suffix for specific trjconvs
    
FUNCTIONS:
    trjconv_generate_center_pbc_mol:
        function to generate centered pbc mol trajectories
    generate_gro_xtc_with_center_AUNP:
        generates gro and xtc with a centered AUNP
    remake_ndx:
        function that remakes the index file
    run_bash_command:
        function that runs the bash command
    generate_gromacs_command:
        function that generates the GROMACS commands
    convert_str_to_input:
        function that converts the commands into byte format

## USAGE
from MDDescriptors.traj_tools.trjconv_commands import *

    
"""
import os
import shutil
import subprocess

## CUSTOM MODULES
from MDDescriptors.core.check_tools import check_path, check_spyder
## IMPORTING TOPOLOGY TOOLS
from MDDescriptors.core.read_write_tools import top_file

###############################################################################
### GLOBAL VARIABLES 
###############################################################################

## DEFINING DEFAULT GMX COMMAND
GMX_COMMAND_PREFIX="gmx"

### DEFINING SUFFIX DICTIONARY
TRJCONV_SUFFIX_DICT = {
        'center_pbc_mol': '-center_pbc_mol',
        'rot_trans': '-rot_trans',
        'rmsf': 'rmsf',
        'water_heavy_atoms': '-water_only_heavy_atoms',
        }

###############################################################################
### FUNCTIONS
###############################################################################

### FUNCTION TO CONVERT STRING TO GROMACS INPUT
def convert_str_to_input(my_string):
    '''This function converts all enters into \n and then converts to bytes'''
    new_string = "\n".join(my_string.split("<br />"))
    new_string_bytes = bytes(new_string,'utf8')
    return new_string_bytes

### FUNCTION TO RUN GROMACS COMMAND
def generate_gromacs_command(executable = 'gmx',
                             arguments = [],
                             input_files = {},
                             output_files = {}):
    '''
    The purpose of this function is to generate the gromacs command based on 
    your inputs
    INPUTS:
        executable: [str]
            executible run flag
        arguments: [list]
            list of arguments. Note that the first one should be the one you are running, e.g.
                solvate, trjconv, etc.
        input_files: [dict]
            dictionary of input files. The flags should match those from 
            GROMACS
        output_files: [dict]
            dictionary of output files. The flags should again match those 
            from GROMACS
    The idea is to replicate the idea from:
        solvate = gmx.commandline_operation('gmx',
                                            arguments=['solvate', '-box', '5', '5', '5'],
                                            input_files={'-cs': structurefile},
                                            output_files={'-p': topfile,
                                                          '-o': structurefile,
                                                          }
    OUTPUTS:
        gromacs_command: [str]
            string of gromacs command
            e.g. 'gmx solvate -box 5 5 5 -cs structure.gro -p structure.top -o structure.gro'
    '''
    
    ## INCLUDING EACH ARGUMENT
    argument_inputs = ' '.join(arguments)
    
    ## INCLUDING ALL INPUTS
    input_files_string = ' '.join([ ' '.join([each_key, input_files[each_key]]) for each_key in input_files.keys()] )
    output_files_string = ' '.join([ ' '.join([each_key, output_files[each_key]]) for each_key in output_files.keys()] )
    
    ## PRINTING COMMAND
    gromacs_command = ' '.join([executable, argument_inputs, input_files_string, output_files_string])
    
    return gromacs_command
    
### FUNCTION TO RUN COMMAND
def run_bash_command(command,
                     wd,
                     string_input = None,
                     path_check = None,
                     rewrite = True,
                     want_stderr = False,
                     shell = False):
    '''
    The purpose of this function is to run the bash command
    INPUTS:
        command: [str]
            string of command
        string_input: [str]
            input string, should be separated by returns
        path_check: [str]
            str to check the path. If exists, then we won't run the bash command. 
            Otherwise, run the bash. This is overwritten by "rewrite" variable.
        shell: [logical]
            False by default. True if you are having problems with the command splitting.
            This is an issue when you are using selections from gmx distance, etc.
            Set this to True if your command has some string input, e.g. 'com of group X...'
    OUTPUTS:
        p: [obj]
            processor
    '''
    ## RUNNING NDX COMMAND
    if check_spyder() is False:
        ## CHECKING PATH
        if path_check is not None:
            file_exists = os.path.isfile(path_check)
        else:
            file_exists = False
        
        ## SEEING IF YOU WANT TO RUN THE COMMAND            
        if file_exists is False or rewrite is True:
            ## RUNNING COMMAND
            if want_stderr is True:
                print_err = subprocess.PIPE
            else:
                print_err = None
            ## RUNNING BASH COMMAND
            if shell is True:
                send_command = command
            else:
                # SPLIT AND LET PROCESS FIGURE IT OUT
                send_command = command.split()
            ## RUNNING SUBPROCESS
            p = subprocess.Popen( send_command, # .split()
                                  stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE,
                                  stderr=print_err,
                                  cwd = wd,
                                  shell = shell
                                  )
            ## ADDING INPUTS IF NECESSARY
            if string_input is not None:
                ## ADDING INPUTS
                (output, err) = p.communicate(input=convert_str_to_input(string_input))
            else:
                (output, err) = p.communicate()
                ## WAITING FOR COMMAND TO FINISH
                p.wait()
            return p, err
        else:
            print("Since %s exists, continuing!"%(os.path.basename(path_check)) ) 

### FUNCTION TO REMAKE THE INDEX FILE
def remake_ndx( path_wd, 
                system_file, 
                component_ndx_file, 
                output_ndx_file,
                rewrite = False ):
    r'''
    function uses tools from convert_with_trjconv to update index file to include 
    peptide features
    '''    
    ### GENERATING GROMACS COMMAND FOR SYSTEM INDEX
    path_system_file = os.path.join( path_wd, system_file )
    path_output_ndx = os.path.join( path_wd, output_ndx_file )
    path_ndx = os.path.join( path_wd, component_ndx_file )
    _SYSTEM_EXT = os.path.splitext( path_system_file )[-1]
    
    ## CHECKING IF SPYDER
    if check_spyder() is False:
        path_system_ndx = path_system_file
        if _SYSTEM_EXT in [ ".tpr", ".gro" ]:
            system_ndx = "tmp.ndx"
            path_system_ndx = os.path.join( path_wd, system_ndx )
            ndx_command = generate_gromacs_command( "gmx",
                                                    arguments=['make_ndx'],
                                                    input_files={'-f': system_file},
                                                    output_files={'-o': system_ndx}
                                                   )
            
            ## INPUTTING
            ndx_inputs=\
                """
                q
                """
        
            ## RUNNING COMMAND
            run_bash_command(command = ndx_command,
                             wd = path_wd,
                             string_input = ndx_inputs,
                             path_check = path_system_ndx,
                             rewrite = rewrite,
                             )
        
        with open( path_output_ndx, 'wb' ) as wfd:
            for f in [ path_system_ndx, path_ndx ]:
                with open( f, 'rb' ) as fd:
                    shutil.copyfileobj( fd, wfd )
        
        ## REMOVE TEMP FILE
        os.remove( path_system_ndx )
    
    return output_ndx_file

#########################################
### CLASS FUNCTION TO CONVERT TRJCONV ###
#########################################
class convert_with_trjconv:
    '''
    The purpose of this function is to convert with trjconv.
    INPUTS:
        wd: [str]
            working directory
    OUTPUTS:
        self.wd: [str]
            working directory
        
    FUNCTIONS:
        compute_com_distances:
            uses gmx distance to compute distances between two groups
        generate_gro_xtc_specific_selection:
            function that generates gro and xtc files based on a selection
        generate_center_pbc_mol:
            function to center and generate periodic boundaries with centering
        generate_water_heavy_atoms:
            function that generates heavy water atoms only -- useful for WC interface
        compute_gmx_rmsf:
            function that computes rmsf using GROMACS
        generate_rotation_translational_fitting:
            function that generates rotational and translational fititng
        gmx_check_traj_length:
            function that checks the trajectory length
        generate_pdb_from_gro:
            function that generates pdb from gro file
        
    '''
    def __init__(self,
                 wd):
        ## STORING
        self.wd = wd
        
        return
    
    ### FUNCTION THAT GENERATES HEAVY ATOMS ONLY
    def generate_heavy_atoms_only(self,
                                  input_prefix,
                                  first_frame = 0,
                                  last_frame = -1,
                                  pbc = ['-pbc', 'mol'],
                                  rewrite = False,
                                  gro_output_time_ps = 0,):
        '''
        The purpose of this function is to generate only heavy atoms of a trajectory.
        INPUTS:
            input_prefix: [str]
                input prefix for trajectories
            first_frame: [float]
                first frame in ps
            last_frame: [float]
                last frame in ps
            pbc: [list]
                list of periodic boundary condition application
            rewrite: [logical]
                True if you want to rewrite
        OUTPUTS:
            output_gro_file, output_xtc_file, output_ndx_file:
                gro, xtc, and index file for the desired selection
        '''
        output_prefix = '-'.join( [input_prefix , str(first_frame), str(last_frame) ] + pbc )
        output_gro_file = output_prefix + ".gro"
        output_xtc_file = output_prefix + ".xtc"
        output_ndx_file = output_prefix + ".ndx"
        
        ##################
        ### INDEX FILE ###
        ##################
        ## DEFINING NO HYDROGEN NAME
        no_hydrogen_name = r"no_hydrogens"
        
        ## GENERATING INDEX FILE
        ndx_inputs =\
        """
        keep 0
        ! a H*
        name 1 %s
        q
        """%(no_hydrogen_name)
        ### GENERATING GROMACS COMMAND FOR INDEX
        ndx_command = generate_gromacs_command(GMX_COMMAND_PREFIX,
                                               arguments=['make_ndx'],
                                               input_files={'-f': input_prefix + '.tpr',
                                                            },
                                               output_files={'-o': output_ndx_file,
                                                              }
                                            )
        ## RUNNING COMMAND
        run_bash_command(command = ndx_command,
                         wd = self.wd,
                         string_input = ndx_inputs,
                         path_check = os.path.join(self.wd, output_ndx_file),
                         rewrite= rewrite,
                         )
        
        
        ################
        ### XTC FILE ###
        ################
        xtc_args = ['trjconv'] + pbc
        
        if first_frame != 0:
            xtc_args += ['-b', str(first_frame)]
        
        if last_frame != -1:
            xtc_args += ['-e', str(last_frame)]
        
        xtc_command = generate_gromacs_command(GMX_COMMAND_PREFIX,
                                               arguments = xtc_args,
                                               input_files={'-s': input_prefix + '.tpr',
                                                            '-f': input_prefix + '.xtc',
                                                            '-n': output_ndx_file,},
                                               output_files={'-o': output_xtc_file,
                                                              }
                                               )
        ## INPUTS FOR XTC
        output_selection = \
        """
        %s
        """%(no_hydrogen_name)
        
        ## RUNNING COMMAND
        run_bash_command(command = xtc_command,
                         wd = self.wd,
                         string_input = output_selection,
                         path_check = os.path.join(self.wd, output_xtc_file),
                         rewrite = rewrite,
                         )
        
        ################
        ### GRO FILE ###
        ################
        
        ## FINDING PREVIOUS TIME
        previous_time=gro_output_time_ps - 10
        
        gro_args = xtc_args + [ '-dump', str(gro_output_time_ps) ]
        ### GENERATING GROMACS COMMAND FOR INDEX
        gro_command = generate_gromacs_command(GMX_COMMAND_PREFIX,
                                               arguments = gro_args,
                                               input_files={'-s': input_prefix + '.tpr',
                                                            '-f': input_prefix + '.xtc',
                                                            '-n': output_ndx_file,},
                                               output_files={'-o': output_gro_file,
                                                              }
                                            )
        
        ## RUNNING COMMAND
        run_bash_command(command = gro_command,
                         wd = self.wd,
                         string_input = output_selection,
                         path_check = os.path.join(self.wd, output_gro_file),
                         rewrite= rewrite,
                         )
        
        return output_gro_file, output_xtc_file, output_ndx_file
    
    ### FUNCTION THAT SIMPLY OUTPUTS A SELECTION
    def generate_gro_xtc_specific_selection(self,
                                            input_prefix,
                                            selection = 'non-Water',
                                            gro_output_time_ps = 0,
                                            rewrite = False):
        '''
        The purpose of this script is to generate gro and xtc files for a 
        desired selection. Note that we assume periodic boundary conditions (pbc mol)
        is desired. The output suffix is dependent on the selection you want, e.g. 
        input_prefix + '-' + selection
        INPUTS:
            input_prefix: [str]
                input prefix for trajectories
            rewrite: [logical]
                True if you want to rewrite
            gro_output_time_ps: [float]
                time in picoseconds that you want to output 
        OUTPUTS:
            output_gro_file, output_xtc_file, output_ndx_file:
                gro, xtc, and index file for the desired selection

        
        '''
        ## DEFINING OUTPUT
        output_prefix = '-'.join( [input_prefix , selection] )
        output_gro_file = output_prefix + ".gro"
        output_xtc_file = output_prefix + ".xtc"
        output_ndx_file = output_prefix + ".ndx"
        
        ##################
        ### INDEX FILE ###
        ##################
        ## GENERATING INDEX FILE
        ndx_inputs =\
        """
        q
        """
        ### GENERATING GROMACS COMMAND FOR INDEX
        ndx_command = generate_gromacs_command(GMX_COMMAND_PREFIX,
                                               arguments=['make_ndx'],
                                               input_files={'-f': input_prefix + '.tpr',
                                                            },
                                               output_files={'-o': output_ndx_file,
                                                              }
                                            )
        ## RUNNING COMMAND
        run_bash_command(command = ndx_command,
                         wd = self.wd,
                         string_input = ndx_inputs,
                         path_check = os.path.join(self.wd, output_ndx_file),
                         rewrite= rewrite,
                         )
        
        
        ################
        ### XTC FILE ###
        ################
        xtc_args = ['trjconv', '-pbc', 'mol']
        xtc_command = generate_gromacs_command(GMX_COMMAND_PREFIX,
                                               arguments = xtc_args,
                                               input_files={'-s': input_prefix + '.tpr',
                                                            '-f': input_prefix + '.xtc',
                                                            '-n': output_ndx_file,},
                                               output_files={'-o': output_xtc_file,
                                                              }
                                               )
        ## INPUTS FOR XTC
        output_selection = \
        """
        %s
        """%(selection)
        
        ## RUNNING COMMAND
        run_bash_command(command = xtc_command,
                         wd = self.wd,
                         string_input = output_selection,
                         path_check = os.path.join(self.wd, output_xtc_file),
                         rewrite = rewrite,
                         )
        
        ################
        ### GRO FILE ###
        ################
        
        ## FINDING PREVIOUS TIME
        previous_time=gro_output_time_ps - 10
        
        gro_args = xtc_args + [ '-dump', str(gro_output_time_ps), '-b', str(previous_time)  ]
        ### GENERATING GROMACS COMMAND FOR INDEX
        gro_command = generate_gromacs_command(GMX_COMMAND_PREFIX,
                                               arguments = gro_args,
                                               input_files={'-s': input_prefix + '.tpr',
                                                            '-f': input_prefix + '.xtc',
                                                            '-n': output_ndx_file,},
                                               output_files={'-o': output_gro_file,
                                                              }
                                            )
    
        ## RUNNING COMMAND
        run_bash_command(command = gro_command,
                         wd = self.wd,
                         string_input = output_selection,
                         path_check = os.path.join(self.wd, output_gro_file),
                         rewrite= rewrite,
                         )
        
        return output_gro_file, output_xtc_file, output_ndx_file
        
    ### FUNCTION TO COMPUTE COM DISTANCES
    def compute_com_distances(self,
                              input_prefix,
                              group_1_resname= 'DOPC',
                              group_2_resname= 'AUNP',
                              rewrite = False,):
        '''
        The purpose of this function is to compute the center of mass distances 
        between two groups. 
        INPUTS:
            input_prefix: [str]
                input prefix for trajectories
            group_1_resname: [str]
                resname for group 1
            group_2_resname: [str]
                resname for group 2
            rewrite: [logical]
                True if you want to rewrite
        OUTPUTS:
            output_xvg_file: [str]
                xvg file used to compute COM distances
            output_ndx_file: [str]
                ndx file used to compute index
        '''
        ## DEFINING OUTPUT
        output_prefix = '-'.join( [input_prefix , group_1_resname, group_2_resname] )
        ## DEFINING OUTPUT NDX
        output_ndx_file = output_prefix + ".ndx"
        output_xvg_file = output_prefix + ".xvg"
        
        ##################
        ### INDEX FILE ###
        ##################
        ## GENERATING INDEX FILE
        ndx_inputs =\
        """
        keep 0
        r %s
        r %s
        q
        """%(group_1_resname, group_2_resname)
        ### GENERATING GROMACS COMMAND FOR INDEX
        ndx_command = generate_gromacs_command(GMX_COMMAND_PREFIX,
                                               arguments=['make_ndx'],
                                               input_files={'-f': input_prefix + '.tpr',
                                                            },
                                               output_files={'-o': output_ndx_file,
                                                              }
                                            )
        ## RUNNING COMMAND
        run_bash_command(command = ndx_command,
                         wd = self.wd,
                         string_input = ndx_inputs,
                         path_check = os.path.join(self.wd, output_ndx_file),
                         rewrite= rewrite,
                         )
        ###########################
        ### RUNNING GMX DISANCE ###
        ###########################
        '''
        gmx distance -s nplm_prod.tpr -f nplm_prod.xtc -oxyz push_summary.xvg -select 'com of group DOPC plus com of group AUNP' -n push.ndx
        '''
        selection = "'com of group %s plus com of group %s\'"%(group_1_resname, group_2_resname)
        
        ## RUNNING DISTANCE
        xvg_args = ['distance', '-select', selection]
        xvg_command = generate_gromacs_command(GMX_COMMAND_PREFIX,
                                               arguments = xvg_args,
                                               input_files={'-s': input_prefix + '.tpr',
                                                            '-f': input_prefix + '.xtc',
                                                            '-n': output_ndx_file,},
                                               output_files={'-oxyz': output_xvg_file,
                                                              }
                                               )
        ## RUNNING COMMAND
        run_bash_command(command = xvg_command,
                         wd = self.wd,
                         string_input = None,
                         path_check = os.path.join(self.wd, output_xvg_file),
                         rewrite = rewrite,
                         shell = True
                         )
        return output_xvg_file, output_ndx_file

    ### FUNCTIONS TO GENERATE CENTER WITH PERIODIC BOUNDARY MOLE
    def generate_center_pbc_mol(self,
                                input_tpr_file,
                                input_xtc_file,
                                index_file = None,
                                output_suffix = None,
                                center_residue_name = 'AUNP',
                                output_name = 'System',
                                rewrite = False,
                                want_water = True,
                                water_residue_name = "SOL",
                                skip_frames = 1,
                                keep_ndx_components = False,
                                ndx_components_file = None,
                                first_frame = None,
                                last_frame = None
                                ):
        '''
        The purpose of this function is to center a residue and generate a trajectory 
        with periodic boundaries
        INPUTS:
            input_tpr_file: [str]
                input tpr file
            input_xtc_file: [str]
                input xtc file
            index_file: [str]
                index file to generate with the residue name
            output_suffix: [str]
                output suffix
            center_residue_name: [str]
                center residue name
            output_name: [str]
                output name
            rewrite: [logical]
                True if you want to rewrite all files
            want_water: [logical, default = True]
                True if you want water. Otherwise, we will only find "non-Waters"
            water_residue_name: [str]
                water residue name
            skip_frames: [int]
                frames to skip
            first_frame: [float]
                first frame in ps, default None means to use all the first frames
            last_frame: [float]
                last frame in ps, default None means use all the times
        OUTPUTS:
            gro and xtc with the trajectory centered and periodic boundary 
            conditions applied with pbc mol
        '''
        if output_suffix is None:
            output_suffix=TRJCONV_SUFFIX_DICT['center_pbc_mol']
        ## GETTING PREFIX
        prefix = os.path.splitext(input_xtc_file)[0]
        
        ## DEFINING NEW PREFIX
        new_prefix = prefix + output_suffix
        
        ## GETTING NEW INDEX FILE
        if index_file is None:
            index_file = new_prefix + ".ndx"
        
        ## GETTING OUTPUT XTC
        output_tpr_file = new_prefix + ".tpr"
        output_xtc_file = new_prefix + ".xtc"
        output_gro_file = new_prefix + ".gro"
        
        ## DEFINING PATH
        path_index_file = os.path.join(self.wd, index_file)
        path_xtc_file = os.path.join(self.wd, output_xtc_file)
        path_gro_file = os.path.join(self.wd, output_gro_file)
        path_tpr_file = os.path.join(self.wd, output_tpr_file)
        
        ## INPUTTING
        if want_water is True:
            if keep_ndx_components is True:
                out_index_file = index_file
                index_file = prefix + ".ndx"
                path_index_file = os.path.join(self.wd, index_file)
                ndx_inputs=\
                    """
                    keep 0
                    r %s
                    q
                    """%(center_residue_name)
                
            else:
                ndx_inputs=\
                    """
                    keep 0
                    r %s
                    q
                    """%(center_residue_name)
        else:
            ## REDEFINING OUTPUT
            output_name = "non-Water"
            if keep_ndx_components is True:
                out_index_file = index_file
                index_file = prefix + ".ndx"
                path_index_file = os.path.join(self.wd, index_file)
                ndx_inputs=\
                    """
                    keep 0
                    ! r %s
                    name 1 %s
                    q
                    """%(water_residue_name,
                    output_name)
                
            else:
                ndx_inputs=\
                    """
                    keep 0
                    r %s
                    ! r %s
                    name 2 %s
                    q
                    """%(center_residue_name,
                    water_residue_name,
                    output_name)

        ### GENERATING GROMACS COMMAND FOR INDEX
        ndx_command = generate_gromacs_command(GMX_COMMAND_PREFIX,
                                               arguments=['make_ndx'],
                                               input_files={'-f': input_tpr_file},
                                               output_files={'-o': index_file,
                                                              }
                                               )
    
        ## RUNNING COMMAND
        run_bash_command(command = ndx_command,
                         wd = self.wd,
                         string_input = ndx_inputs,
                         path_check = path_index_file,
                         rewrite = rewrite,
                         )

        ## REMAKING INDEX FILE
        if keep_ndx_components is True:
            index_file = remake_ndx( self.wd, 
                                     index_file, 
                                     ndx_components_file, 
                                     output_ndx_file = out_index_file,
                                     rewrite = rewrite )
        
        ###########################
        ### GENERATING XTC FILE ###
        ###########################
        
        ## DEFINING XTC INPUTS
        xtc_args = ['trjconv', '-center', '-pbc', 'mol', '-skip', str(skip_frames)]
        
        if first_frame is not None:
            print("Starting trajectory in %.3f ps"%(first_frame) )
            xtc_args = xtc_args + [ "-b", str(first_frame) ]
        if last_frame is not None:
            print("Ending trajectory in %.3f ps"%(last_frame) )
            xtc_args = xtc_args + [ "-e", str(last_frame) ]
        
        ### GENERATING GROMACS COMMAND FOR INDEX
        xtc_command = generate_gromacs_command(GMX_COMMAND_PREFIX,
                                               arguments=xtc_args,
                                               input_files={'-s': input_tpr_file,
                                                            '-f': input_xtc_file,
                                                            '-n': index_file,
                                                            },
                                               output_files={'-o': output_xtc_file,
                                                              }
                                            )
        
        ## INPUTS FOR XTC
        center_inputs = \
        """%s
        %s
        """%(center_residue_name,
             output_name,
             )
        
        ## RUNNING COMMAND
        run_bash_command(command = xtc_command,
                         wd = self.wd,
                         string_input = center_inputs,
                         path_check = path_xtc_file,
                         rewrite = rewrite,
                         )
    
        ###########################
        ### GENERATING GRO FILE ###
        ###########################
        
        ### GENERATING GROMACS COMMAND FOR INDEX
        gro_command = generate_gromacs_command(GMX_COMMAND_PREFIX,
                                               arguments=['trjconv', '-dump', '0', '-pbc', 'mol', '-center'],
                                               input_files={'-s': input_tpr_file,
                                                            '-f': input_xtc_file,
                                                            '-n': index_file,},
                                               output_files={'-o': output_gro_file,
                                                              }
                                            )
    
        ## RUNNING COMMAND
        run_bash_command(command = gro_command,
                         wd = self.wd,
                         string_input = center_inputs,
                         path_check = path_gro_file,
                         rewrite= rewrite,
                         )
            
        ###########################
        ### GENERATING TPR FILE ###
        ###########################
        
        ### GENERATING GROMACS COMMAND FOR INDEX
        # gmx convert-tpr -s sam_prod.tpr -o sam_prod-center_pbc_mol.tpr
        tpr_command = generate_gromacs_command(GMX_COMMAND_PREFIX,
                                               arguments=['convert-tpr'],
                                               input_files={'-s': input_tpr_file,
                                                            '-n': index_file,},
                                               output_files={'-o': output_tpr_file,
                                                              }
                                            )
        
        ## DEFINING OUTPUTS
        ## INPUTS FOR XTC
        tpr_inputs = \
        """%s
        """%(output_name,
             )
    
        ## RUNNING COMMAND
        run_bash_command(command = tpr_command,
                         wd = self.wd,
                         string_input = tpr_inputs,
                         path_check = path_tpr_file,
                         rewrite= rewrite,
                         )

        ## REMAKE INDEX TO EXCLUDE WATER, IF WANT_WATER IS FALSE
        if want_water is False and ndx_components_file is not None:
            index_file = remake_ndx( self.wd, 
                                     output_gro_file, 
                                     ndx_components_file, 
                                     output_ndx_file = index_file,
                                     rewrite = True )        
        
        return output_gro_file, output_xtc_file, output_tpr_file, index_file
    
    
    ### FUNCTION TO GENERATE PURE WATER OXYGEN ATOM SIMULATIONS
    def generate_water_heavy_atoms(self,
                                   input_prefix,
                                   output_suffix = TRJCONV_SUFFIX_DICT['water_heavy_atoms'],
                                   water_residue_name = 'SOL',
                                   center_residue_name = 'AUNP',
                                   only_last_ns = False,           # add gmx check option to find total frames
                                   rewrite = False,
                                   first_frame = None,
                                   last_frame = None
                                   ):
        '''
        The purpose of this function is to generate trajectories with only water heavy atoms.
        INPUTS:
            input_prefix: [str]
                input prefix for trajectories
            output_prefix: [str]
                output prefix for trajectories
            water_residue_name: [str]
                water residue name
            center_residue_name: [str]
                center residue name. If None, no centering will be done
            only_last_ns: [logical]
                True if you only want the last 1 ns of the trajectory
            rewrite: [logical]
                True if you want to rewrite the files
            first_frame: [float]
                first frame desired in ps. If None, all frames will be used. The 'only_last_ns' flag takes precedence over this if True.
            last_frame: [float]
                last frame desired in ps. If None, all frames will be used. The 'only_last_ns' flag takes precedence over this if True.
                
        OUTPUTS:
            output_gro_file: [str]
                gro file output name
            output_xtc_file: [str]
                xtc file output name
        '''
        ## DEFINING OUTPUT
        output_prefix = input_prefix + output_suffix
        output_gro_file = output_prefix + ".gro"
        output_xtc_file = output_prefix + ".xtc"
        output_ndx_file = output_prefix + ".ndx"
        output_tpr_file = output_prefix + ".tpr"
        
        ## GETTING THE INDEX FILE
        index_res_name = "r %s & a O*"%(water_residue_name)
        renamed_group = "water_heavy_atoms"
        ## GENERATING INDEX FILE
        ndx_inputs=\
            """
            keep 0
            keep 1
            %s
            name 0 %s
            r %s
            q
            """%(index_res_name,
                 renamed_group,
                 center_residue_name)
        
        ### GENERATING GROMACS COMMAND FOR INDEX
        ndx_command = generate_gromacs_command(GMX_COMMAND_PREFIX,
                                               arguments=['make_ndx'],
                                               input_files={'-f': input_prefix + '.tpr',
                                                            },
                                               output_files={'-o': output_ndx_file,
                                                              }
                                            )
        
        ## RUNNING COMMAND
        run_bash_command(command = ndx_command,
                         wd = self.wd,
                         string_input = ndx_inputs,
                         path_check = os.path.join(self.wd, output_ndx_file),
                         rewrite= rewrite,
                         )
        
        ## DEFINING GROMACS COMMAND
        if center_residue_name == "None":
            cmd_args = ['trjconv', '-pbc', 'mol']
        else:
            cmd_args = ['trjconv', '-pbc', 'mol', '-center' ]
        
        ## DEFAULT TPR COMMAND
        tpr_cmd = ['convert-tpr']
        
        ## SEEING IF ONLY LAST NS IS DESIRED
        if only_last_ns is True and check_spyder() is False:
            traj_length = self.gmx_check_traj_length( input_prefix,
                                                      rewrite = rewrite, )
            begin_time = int( traj_length - 1000 ) # assumes units of ps
            xtc_args = cmd_args + [ "-b", str(begin_time) ]
            tpr_cmd = ['convert-tpr', "-until", "1000" ]
            
        elif only_last_ns is False and (first_frame is not None or last_frame is not None):
            ## DEFINING BEGIN AND END TIME
            xtc_args = cmd_args[:]
            if first_frame is not None:
                print("Starting trajectory in %.3f ps"%(first_frame) )
                xtc_args = xtc_args + [ "-b", str(first_frame) ]
            if last_frame is not None:
                print("Ending trajectory in %.3f ps"%(last_frame) )
                xtc_args = xtc_args + [ "-e", str(last_frame) ]

        else:
            xtc_args = cmd_args

        xtc_command = generate_gromacs_command(GMX_COMMAND_PREFIX,
                                       arguments = xtc_args,
                                       input_files={'-s': input_prefix + '.tpr',
                                                    '-f': input_prefix + '.xtc',
                                                    '-n': output_ndx_file,},
                                       output_files={'-o': output_xtc_file,
                                                      }
                                    )
        
        ## INPUTS FOR XTC
        center_inputs = \
        """
        %s
        %s
        """%(center_residue_name,
             renamed_group,
             )
        

        ## RUNNING COMMAND
        run_bash_command(command = xtc_command,
                         wd = self.wd,
                         string_input = center_inputs,
                         path_check = os.path.join(self.wd, output_xtc_file),
                         rewrite = rewrite,
                         )
        
        ################
        ### GRO FILE ###
        ################
        
        
        gro_args = cmd_args + [ '-dump', '0' ]
        ### GENERATING GROMACS COMMAND FOR INDEX
        gro_command = generate_gromacs_command(GMX_COMMAND_PREFIX,
                                               arguments = gro_args,
                                               input_files={'-s': input_prefix + '.tpr',
                                                            '-f': input_prefix + '.xtc',
                                                            '-n': output_ndx_file,},
                                               output_files={'-o': output_gro_file,
                                                              }
                                            )
    
        ## RUNNING COMMAND
        run_bash_command(command = gro_command,
                         wd = self.wd,
                         string_input = center_inputs,
                         path_check = os.path.join(self.wd, output_gro_file),
                         rewrite= rewrite,
                         )
        
        
        ################
        ### TPR FILE ###
        ################
        tpr_inputs = \
        """
        %s
        """%(
             renamed_group,
             )
        
        ## GENERATING COMMAND
        tpr_command = generate_gromacs_command(GMX_COMMAND_PREFIX,
                                               arguments = tpr_cmd,
                                               input_files={'-s': input_prefix + '.tpr',
                                                            '-n': output_ndx_file,},
                                               output_files={'-o': output_tpr_file,
                                                              }
                                            )
        
        ## RUNNING COMMAND
        run_bash_command(command = tpr_command,
                         wd = self.wd,
                         string_input = tpr_inputs,
                         path_check = os.path.join(self.wd, output_tpr_file),
                         rewrite= rewrite,
                         )
        
        
        return output_gro_file, output_xtc_file
    
    
    ### FUNCTION TO COMPUTE RMSF
    def compute_gmx_rmsf(self,
                         input_gro_file,
                         input_tpr_file,
                         input_xtc_file,
                         index_file = None,
                         input_top_file = None,
                         rmsf_residue_name = "LIG",
                         output_name = "non-Water",
                         top_comment_out = [],
                         input_mdp_file = None,
                         output_prefix = TRJCONV_SUFFIX_DICT['rmsf'],
                         rewrite = False,
                         ):
        '''
        The purpose of this function is to compute gromacs RMSF command. Note that 
        we will need to make corrections for RMSF if spring constants are present.
        INPUTS:
            input_tpr_file: [str]
                input tpr file
            input_xtc_file: [str]
                input xtc file
            index_file: [str]
                index file to generate with the residue name
            input_top_file: [str]
                topology file
            rmsf_residue_name: [str]
                residue name to compute RMSF for
            top_comment_out: [list]
                list of topologies to comment out for
            input_mdp_file: [str]
                MDP string
            output_prefix: [str]
                output prefix for all RMSF calculations
            rewrite: [logical]
                True if you want to rewrite
        OUTPUTS:
            This code will output a rmsf_output.xvg that will account for RMSF 
            of all ligands.
        ALGORITHM:
            - If top_comment_out is > 1 in length, then we will need to re-create 
            the .tpr file
            - Use the RMSF code to generate a PDB
            - Use RMS function to compute RMSF based on a reference
        '''        
        ## DFEINING PDB
        output_pdb_file = output_prefix + ".pdb"
        if input_top_file is not None:
            output_top_file = output_prefix + ".top"
            output_tpr_file = output_prefix + ".tpr"
        output_rmsf_file = output_prefix + "_per_atom.xvg"
        output_rmsf_vs_time_file= output_prefix + "_per_time.xvg"
        output_ndx_file = output_prefix + ".ndx"
        
        ## EDITING TOPOLOGY FILE
        if len(top_comment_out) > 0:
            
            ## CORRECTING TOPOLOGY FILE
            top = top_file(os.path.join(self.wd, input_top_file))
            
            ## UPDATING LINES BY COMMENTING
            updated_lines = top.comment_out_top_files(top_file_lines = top.top_file_lines,
                                                      comment_out_list = top_comment_out )
            ## WRITING TOPOLOGY
            top.write_top(output_path = os.path.join(self.wd, output_top_file),
                          top_file_lines = updated_lines)
            
        else:
            ## SIMPLY REDEFINE OUTPUT TOP
            output_top_file = input_top_file
        
        
        ## RECREATING TPR FILE
        if input_mdp_file is not None:
            # gmx grompp -f nvt_double_prod_gmx5_charmm36_frozen_gold.mdp -c sam_prod.gro -p sam_copy.top -o sam_prod_rmsf.tpr
            tpr_command = generate_gromacs_command(GMX_COMMAND_PREFIX,
                                                   arguments=['grompp'],
                                                   input_files={'-f': input_mdp_file,
                                                                '-n': index_file,
                                                                '-c': input_gro_file,
                                                                '-p': output_top_file,
                                                                },
                                                   output_files={'-o': output_tpr_file,
                                                                  }
                                                )
            ## RUNNING COMMAND
            run_bash_command(command = tpr_command,
                             wd = self.wd,
                             string_input = None,
                             path_check = os.path.join(self.wd, output_tpr_file),
                             rewrite= rewrite,
                             )
            
        
        ## GETTING RESIDUE NAMES
        res_input = ' & '.join([ 'r %s'%(each_name) for each_name in rmsf_residue_name ])
        renamed_value = "LIGAND"
        ## GENERATING INDEX FILE
        ndx_inputs=\
            """
            keep 0
            keep 1
            %s
            name 0 %s
            q
            """%(res_input,
                 renamed_value)
        
        ### GENERATING GROMACS COMMAND FOR INDEX
        ndx_command = generate_gromacs_command(GMX_COMMAND_PREFIX,
                                               arguments=['make_ndx'],
                                               input_files={'-f': output_tpr_file,
                                                            },
                                               output_files={'-o': output_ndx_file,
                                                              }
                                            )
        
        ## RUNNING COMMAND
        run_bash_command(command = ndx_command,
                         wd = self.wd,
                         string_input = ndx_inputs,
                         path_check = os.path.join(self.wd, output_ndx_file),
                         rewrite= rewrite,
                         )
        
        ### COMPUTING RMSF
        rmsf_command = generate_gromacs_command(GMX_COMMAND_PREFIX,
                                               arguments=['rmsf'],
                                               input_files={'-s': output_tpr_file,
                                                            '-f': input_xtc_file,
                                                            '-n': output_ndx_file,
                                                            },
                                               output_files={'-ox': output_pdb_file,
                                                             '-o': output_rmsf_file,
                                                              }
                                               )
        
        ## GETTING RMSF INPUTS
        rmsf_inputs=\
            """
            %s
            """%(renamed_value)
        
        ## RUNNING COMMAND
        run_bash_command(command = rmsf_command,
                         wd = self.wd,
                         string_input = rmsf_inputs,
                         path_check = os.path.join(self.wd, output_rmsf_file),
                         rewrite= rewrite,
                         )
        
        ## RUNNING RMS
        rms_command = generate_gromacs_command(GMX_COMMAND_PREFIX,
                                               arguments=['rms'],
                                               input_files={'-s': output_pdb_file,
                                                            '-f': input_xtc_file,
                                                            '-n': output_ndx_file,
                                                            },
                                               output_files={'-o': output_rmsf_vs_time_file,
                                                              }
                                               )
        
        ## GETTING RMS INPUTS
        rms_inputs=\
            """
            %s
            %s
            """%(renamed_value,
                 renamed_value)
        
        ## RUNNING COMMAND
        run_bash_command(command = rms_command,
                         wd = self.wd,
                         string_input = rms_inputs,
                         path_check = os.path.join(self.wd, output_rmsf_vs_time_file),
                         rewrite= rewrite,
                         )
        
        return output_pdb_file, output_rmsf_file, output_rmsf_vs_time_file
        
    
    ### FUNCTION TO GENERATE ROTATION AND TRANSLATION DEGREES OF FREEDOM
    def generate_rotation_translational_fitting(self,
                                input_tpr_file,
                                input_xtc_file,
                                index_file = None,
                                output_suffix = TRJCONV_SUFFIX_DICT['rot_trans'],
                                center_residue_name = 'AUNP',
                                output_name = 'System',
                                rewrite = False,
                                want_water = True,
                                water_residue_name = "SOL",
                                keep_ndx_components = False,
                                ndx_components_file = None, ):
        '''
        The purpose of this function is to generate rotational and translational 
        by fitting protocol. Note that we assume that you have centered the 
        model.
        Note: still need to fix this for correct rotational and translational fitting. (issue with spring constants...)
        '''
        ## RUNNING CENTERING ALGORITHM
        output_gro_file, output_xtc_file, output_tpr_file, index_file = self.generate_center_pbc_mol(input_tpr_file = input_tpr_file,
                                                                                                     input_xtc_file = input_xtc_file,
                                                                                                     index_file = index_file,
                                                                                                     output_suffix = TRJCONV_SUFFIX_DICT['center_pbc_mol'],
                                                                                                     center_residue_name = center_residue_name,
                                                                                                     output_name = output_name,
                                                                                                     rewrite = rewrite,
                                                                                                     want_water= want_water,
                                                                                                     water_residue_name = water_residue_name,
                                                                                                     keep_ndx_components = keep_ndx_components,
                                                                                                     ndx_components_file = ndx_components_file,
                                                                                                    )
        
        ## GETTING PREFIX
        prefix = os.path.splitext(output_xtc_file)[0]
        
        ## DEFINING NEW PREFIX
        new_prefix = prefix + output_suffix


        if keep_ndx_components is True:
            out_index_file = new_prefix + ".ndx"
            path_index_file = os.path.join( self.wd, index_file )
            ndx_inputs=\
                """
                keep 0
                q
                """

            ### GENERATING GROMACS COMMAND FOR INDEX
            ndx_command = generate_gromacs_command(GMX_COMMAND_PREFIX,
                                                   arguments=['make_ndx'],
                                                   input_files={'-f': output_tpr_file},
                                                   output_files={'-o': index_file,
                                                                  }
                                                   )
        
            ## RUNNING COMMAND
            run_bash_command(command = ndx_command,
                             wd = self.wd,
                             string_input = ndx_inputs,
                             path_check = path_index_file,
                             rewrite = rewrite,
                             )
            
            index_file = remake_ndx( self.wd, 
                                     index_file, 
                                     ndx_components_file, 
                                     output_ndx_file = out_index_file,
                                     rewrite = rewrite )
        
        ## DEFINING GROMACS COMMAND
        path_out_xtc = os.path.join(self.wd, new_prefix + ".xtc")
        gromacs_command = generate_gromacs_command(GMX_COMMAND_PREFIX,
                                               arguments=['trjconv', '-center', '-fit', 'rot+trans'],
                                               input_files={'-s': output_tpr_file,
                                                            '-f': output_xtc_file,
                                                            '-n': index_file,},
                                               output_files={'-o': new_prefix + ".xtc",
                                                              }
                                            )
        ## DEFINING INPUTS
        inputs_rot_trans = \
        """
        %s
        %s
        System
        """%(center_residue_name,
            center_residue_name)
        
        ## RUNNING COMMAND
        run_bash_command(command = gromacs_command,
                         wd = self.wd,
                         string_input = inputs_rot_trans,
                         path_check = path_out_xtc,
                         rewrite= rewrite,
                         )
        
        ## DEFINING GROMACS COMMAND
        path_out_gro = os.path.join(self.wd, new_prefix + ".gro")
        gromacs_command = generate_gromacs_command(GMX_COMMAND_PREFIX,
                                               arguments=['trjconv', '-dump', '0', '-center', '-fit', 'rot+trans'],
                                               input_files={'-s': output_tpr_file,
                                                            '-f': output_xtc_file,
                                                            '-n': index_file,},
                                               output_files={'-o': new_prefix + ".gro",
                                                              }
                                            )
        ## DEFINING INPUTS
        inputs_rot_trans = \
        """
        %s
        %s
        System
        """%(center_residue_name,
            center_residue_name)
        
        ## RUNNING COMMAND
        run_bash_command(command = gromacs_command,
                         wd = self.wd,
                         string_input = inputs_rot_trans,
                         path_check = path_out_gro,
                         rewrite= rewrite,
                         )    
    
        return path_out_xtc, path_out_gro, index_file, os.path.join( self.wd, output_tpr_file )
    
    ### FUNCTION THAT CHECKS THE TRAJ LENGTH    
    def gmx_check_traj_length( self, file_prefix, rewrite = False ):
        r'''
        This function determines the final frame using gmx check and the .cpt file
        NOTE: ENSURE CHECKPOINT FILE EXISTS AND HAS SAME FILE PREFIX AS GRO AND XTC
        '''
        ### GENERATING GROMACS COMMAND FOR INDEX
        cmd = generate_gromacs_command( GMX_COMMAND_PREFIX,
                                        arguments=['check'],
                                        input_files={'-f': file_prefix + ".cpt"},
                                       )
                
        ## RUNNING COMMAND
        output = run_bash_command( command = cmd,
                                   wd = self.wd,
                                   want_stderr = True,
                                   rewrite = rewrite
                                  )[1]
        
        traj_length = -1
        output = output.decode('utf-8')
        for line in output.split('\n'):
            if "Last frame" in line:
                traj_length = float(line.split()[-1])
                
        return traj_length

    ### FUNCTION THAT GENERATES PDB FROM GRO
    def generate_pdb_from_gro( self, file_prefix, make_whole = True, rewrite = False ):
        r'''
        Function to create a pdb file from a gro
        '''               
        output_pdb = os.path.join(self.wd, file_prefix + '.pdb')
        output_xtc = os.path.join(self.wd, file_prefix + '.xtc')
        
        ### GENERATING GROMACS COMMAND FOR INDEX
        cmd_args = ['trjconv']
        if make_whole is True:
            cmd_args += ['-pbc', 'mol']

        ### GENERATING GROMACS COMMAND FOR INDEX
        gro_command = generate_gromacs_command(GMX_COMMAND_PREFIX,
                                               arguments = cmd_args,
                                               input_files={'-s': file_prefix + '.tpr',
                                                            '-f': file_prefix + '.xtc'},
                                               output_files={'-o': 'tmp.xtc'}
                                               )
        ## DEFINING INPUTS
        inputs_pdb = \
        """
        System
        """
        
        ## RUNNING COMMAND
        if check_spyder() is False and os.path.exists( output_pdb ) is False:
            tmp_xtc = os.path.join(self.wd, 'tmp.xtc')
            run_bash_command( command = gro_command,
                              wd = self.wd,
                              string_input = inputs_pdb,
                              path_check = tmp_xtc,
                              rewrite = rewrite,
                             )
            
            if os.path.exists( tmp_xtc ) is True:
                os.remove( output_xtc )
                os.rename( tmp_xtc, output_xtc )            

        gro_command = generate_gromacs_command(GMX_COMMAND_PREFIX,
                                               arguments = cmd_args,
                                               input_files={'-s': file_prefix + '.tpr',
                                                            '-f': file_prefix + '.gro'},
                                               output_files={'-o': file_prefix + '.pdb'}
                                               )
        
        ## RUNNING COMMAND
        run_bash_command(command = gro_command,
                         wd = self.wd,
                         string_input = inputs_pdb,
                         path_check = output_pdb,
                         rewrite = rewrite,
                         )
            
        return output_pdb, output_xtc

### FUNCTION TO CENTER GOLD NP
def generate_gro_xtc_with_center_AUNP(path_to_sim,
                                      input_prefix,
                                      output_suffix,
                                      rewrite = False,
                                      skip_frames=1,
                                      center_residue_name = "AUNP",
                                      **args):
    '''
    This generates a gro and xtc with gold nanoparticle centered.
    INPUTS:
        path_to_sim: [str]
            path to sims
        input_prefix: [str]
            input prefix
        output_suffix: [str]
            output suffix
        rewrite: [logical]
            True/False if you want to overwrite
        skip_frames: [int]
            skipping frames
        center_residue_name: [str]
            center residue name
        **args:
            arguments used for cnetering            
    OUTPUTS:
        output_gro: [str]
            output gro file
        output_xtc: [str]
            output xtc file
    '''
    ## CONVERTING TRAJECTORY
    trjconv_conversion = convert_with_trjconv(wd = path_to_sim)

    ## GENERATING CENTER + PBC MOL
    output_gro, output_xtc, _, _ = trjconv_conversion.generate_center_pbc_mol(input_tpr_file = input_prefix + ".tpr",
                                                                               input_xtc_file = input_prefix + ".xtc",
                                                                               center_residue_name = center_residue_name,
                                                                               output_name = 'System',
                                                                               rewrite = rewrite,
                                                                               want_water = False,
                                                                               water_residue_name = "SOL",
                                                                               output_suffix = output_suffix,
                                                                               skip_frames = skip_frames,
                                                                               **args)
    return output_gro, output_xtc


#%% RUN FOR TESTING PURPOSES 
if __name__ == "__main__":
    
                                     
    #%%
    
    
    ## RUNNING COMMAND
    run_bash_command(command = command,
                     wd = wd
                     )
    
    
    
    # ''' DEBUGGING FOR CENTERING
    ## DEFINING PATH
    path_of_interest = "/home/akchew/scratch/nanoparticle_project/simulations/20200120-US-sims_NPLM_rerun_stampede/US-1.3_5_0.2-NPLM-DOPC_196-300.00-5_0.0005_2000-EAM_2_ROT012_1/4_simulations/1.300"
    # "/home/akchew/scratch/nanoparticle_project/simulations/20200212-Debugging_GNP_spring_constants_heavy_atoms/MostlikelynpNVTspr_1000-EAM_300.00_K_2_nmDIAM_C11OH_CHARMM36jul2017_Trial_1_likelyindex_1"
    # "/home/akchew/scratch/nanoparticle_project/simulations/20200129-Debugging_GNP_spring_constants_heavy_atoms/MostlikelynpNVTspr_100-EAM_300.00_K_2_nmDIAM_C11NH3_CHARMM36jul2017_Trial_1_likelyindex_1"
    
    ### FUNCTION TO CONVERT PATHS 
    def convert_paths(input_path):
        '''
        The purpose of this function is to convert the path from home to 
        server and so on. 
        
        '''
        import sys
        import getpass
    
        ## CHECKING THE USER NAME
        user_name = getpass.getuser() # Outputs $USER, e.g. akchew, or bdallin
        
        ## IDENTIFYING PATHS
        if input_path.startswith('/home'):
            current_location="server"
        elif input_path.startswith(r'R:'):
            current_location="windows"
        elif input_path.startswith(r'/Volumes'):
            current_location="mac"
            
        ## IDENTIFYING RUN LOCATION
        if sys.prefix.startswith(r"C:"):
            run_location = "windows"
        elif sys.prefix.startswith(r"/Users/"):
            run_location = "mac"
        else:
            run_location = ""
        
        ## UPDATING PATHS
        if run_location == "windows":
            if current_location == "server":
                input_path = input_path.replace('/home/' + user_name,r'R:')
        elif run_location == "mac":
            if current_location == "server":
                input_path = input_path.replace('/home/',r'/Volumes')
        
        ## PRINTING
        print("Current path: %s"%(current_location) )
        print("Run location: %s"%(run_location) )
        return input_path
        
    
    ## CHECKING PATH
    path_of_interest = convert_paths(path_of_interest)

    ## DEFINING PREFIX
    # input_prefix="sam_prod"
    input_prefix="nplm_prod"
    
    ## DEFINING OUTPUT PREFIX
    output_prefix="sam_prod-center_pbc_mol"
    # "sam_prod_center_whole"
    
    ## DEFINING GRO AND XT
    tpr_file = input_prefix + ".tpr"
    xtc_file = input_prefix + ".xtc"  

    ## USING TRJCONV
    trjconv_output = convert_with_trjconv(wd = path_of_interest)
    
    
    ## GENERATING GRO AND XTC + CENTERING
    output_gro, output_xtc = generate_gro_xtc_with_center_AUNP(path_of_interest,
                                                               input_prefix = input_prefix,
                                                               output_suffix = None,
                                                               rewrite = False,
                                                               skip_frames=10)
    
    
    ''' DEBUGGING HEAVY ATOMS
    func_inputs = {
            'input_prefix': input_prefix,
            }

    ## GENERATING RMSF
    trjconv_output.generate_water_heavy_atoms(**func_inputs)
    '''    

    ''' RMSE DEBUGGING
    ## DEFINING INPUTS
    func_inputs = {
            'input_gro_file': input_prefix + ".gro",
            'input_tpr_file': output_prefix + ".tpr",
            'input_xtc_file': output_prefix + ".xtc",
            'index_file': output_prefix + ".ndx",
            'input_top_file': "sam.top",
            'top_comment_out': [ "lig_posre.itp", 
                                 "sulfur_posre.itp",
                                 "gold_posre.itp"],
            'input_mdp_file': "nvt_double_prod_gmx5_charmm36_frozen_gold.mdp",
            'rmsf_residue_name': ['HUN'],
            'rewrite': True
            }

    ## GENERATING RMSF
    trjconv_output.compute_gmx_rmsf(**func_inputs)
    '''
    

        
        # func_inputs['top_comment_out']
            
        
    



    # gmx rmsf -s sam_prod_rmsf.tpr -f sam_prod-center_pbc_mol.xtc -ox rmsf_pdb.pdb -n sam_prod-center_pbc_mol.ndx
    # gmx rms -s rmsf_pdb.pdb  -f sam_prod-center_pbc_mol.xtc
#    ## CONVERTING TRJCONV
#    trjconv_generate_center_pbc_mol(wd = path_of_interest,
#                                    tpr_file = tpr_file,
#                                    xtc_file = xtc_file,
#                                    index_file = output_prefix + ".ndx",
#                                    output_prefix = output_prefix,
#                                    center_residue_name = 'AUNP',
#                                    output_name = 'System',
#                                    rewrite = True,
#                                    )

    # '''
