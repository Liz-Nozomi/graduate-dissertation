# WELCOME TO **MDDESCRIPTORS**!

The goal of this project is to develop MDDescriptors that we can share with each other and potentially develop a module that can be shared publically. We will update this document as new scripts are uploaded. If you make major changes (i.e. creating directories, etc.), please help us keep track by updating this file. This project was developed by students in the Van Lehn group.

# FOR NEW INCOMING CONTRIBUTORS
Please start by adding yourself to the *CONTRIBUTOR LIST* (assuming you will contribute to the scripts). We would like to fully acknowledge authors who have worked on this project. For beginner coders who want to work in developing python modules, try to familiarize yourself with some of the references below:

- PYTHON CODING: https://www.lynda.com/Python-training-tutorials/415-0.html
- PYTHON MODULE CREATIONS: https://packaging.python.org/tutorials/distributing-packages/
- GIT: https://services.github.com/on-demand/downloads/github-git-cheat-sheet.pdf

# CONTRIBUTOR LIST
- Alex K. Chew (alexkchew@gmail.com)

# USAGE
We will have MDDescriptors within the directory of your analysis directory. Note, MDDescriptors can actually be placed in within a directory outside your analysis folder. To do this, you will need to work on fixing the python paths (analogous to using import numpy without having the numpy in the same directory that you are working in). Please see the [Wikipedia page](https://sites.google.com/a/wisc.edu/vanlehngroup/procedures-codes/24-dealing-with-python-paths) on this.

The main idea is that we will use GIT to allow multiple users to update and keep track of changes to the files. Import MDDescriptors by using the command below in your python script:
```
	import MDDescriptors as mdd
```
# DIRECTORY STRUCTURE:
- application: folder containing all application-specific scripts
- core: contains important scripts that can be used by applications
- geometry: contains all code pertaining to geometry
- global_vars: contains global variables used in general
- publishing_tools: contains files for future publication of MDDescriptors. Note that these files will have to be moved to a directory prior to "MDDescriptors" so we can upload our module on the python database.
- templates: contains template scripts for code development
- traj_tools: important folder containing the code that can loop through trajectories and run multiple analysis
- tutorials: contains tutorial examples
- visualization: scripts that contain visualization tools

# UPDATES
- 2018-03-01: (Alex K. Chew) Created initial folder structure and generated publishing_tools
- 2018-06-28: (Alex K. Chew) Updating READMe
