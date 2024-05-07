#!/bin/bash 
# loop_traj_debugging_spring_constants.sh

#SBATCH -p compute
#SBATCH -t 1000:00:00
#SBATCH -J loop_traj_debugging_spring_constants.py 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1              # total number of mpi tasks requested
#SBATCH --mail-user=akchew@wisc.edu
#SBATCH --mail-type=all  # email me when the job starts

# INSTRUCTIONS:
# The -t command specifies how long this simulation will run for (in HOURS:MINUTES:SECONDS). Try to estimate this as best you can
# as the simulation will temrinate after this time.
# The -J flag species the name of the simulation
# the --mail-user command will send you email when the job runs / terminates
# Do not change the other flags, really.

# SUBMIT THIS SCRIPT using the command sbatch thisscriptname

PythonScriptName="loop_traj_debugging_spring_constants.py"
ScriptName=${PythonScriptName%.py}
# stdbuf -oL 
python3.6 -u ${PythonScriptName} 
# > ${ScriptName}.log