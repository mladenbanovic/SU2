#!/usr/bin/env python

from __future__ import division, print_function, absolute_import
from optparse import OptionParser    # use a parser for configuration
import SU2                # imports SU2 python tools
import pysu2
import pysu2ad            # imports the SU2 AD wrapped module
import os
import shutil

# -------------------------------------------------------------------
#  Primal (Objective function(s)) 
# -------------------------------------------------------------------
def primal(options, config, mpi_comm):
  # Initialize the corresponding driver of SU2, this includes solver preprocessing
  try:
    SU2Driver = pysu2.CSinglezoneDriver(options.filename, options.nZone, mpi_comm)
  except TypeError as exception:
    print('A TypeError occured in pysu2.CDriver : ',exception)
    if options.with_MPI == True:
      print('ERROR : You are trying to initialize MPI with a serial build of the wrapper. Please, remove the --parallel option that is incompatible with a serial build.')
    else:
      print('ERROR : You are trying to launch a computation without initializing MPI but the wrapper has been built in parallel. Please add the --parallel option in order to initialize MPI for the wrapper.')
    return
  
    # Launch the solver for the entire computation
  SU2Driver.StartSolver()

  # Postprocess the solver and exit cleanly
  SU2Driver.Postprocessing()

  if SU2Driver != None:
    del SU2Driver

  # history filename
  plot_format = config.get('TABULAR_FORMAT', 'CSV')
  plot_extension = SU2.io.get_extension(plot_format)
  history_filename = config['CONV_FILENAME'] + plot_extension
  
  # Read the objective values from history file
  aerodynamics = SU2.io.read_aerodynamics(history_filename, config.NZONES)
  funcs = SU2.util.ordered_bunch()
  for key in SU2.io.historyOutFields:
    if key in aerodynamics:
      funcs[key] = aerodynamics[key]
  #print (funcs)
      
  def_objs = config['OPT_OBJECTIVE']
  objectives = def_objs.keys()
  func_vals_sum = 0.
  for i_obj,this_obj in enumerate(objectives):
    scale = def_objs[this_obj]['SCALE']
    global_factor = float(config['OPT_GRADIENT_FACTOR'])
    sign  = SU2.io.get_objectiveSign(this_obj)
    
    func_vals_sum += funcs[this_obj] * sign * scale * global_factor
  
  return func_vals_sum

# -------------------------------------------------------------------
#  Adjoint
# -------------------------------------------------------------------
def adjoint(options, config, mpi_comm):
  config['MATH_PROBLEM']  = 'DISCRETE_ADJOINT'
  
  dumpFilename = 'config_CFD_AD.cfg'
  config.dump(dumpFilename)
  
  #RESTART TO SOLUTION
  restart  = config.RESTART_FILENAME
  solution = config.SOLUTION_FILENAME
  if os.path.exists(restart):
    shutil.move( restart , solution )
  
  # Initialize the corresponding driver of SU2, this includes solver preprocessing
  try:
    SU2DriverAD = pysu2ad.CDiscAdjSinglezoneDriver(dumpFilename, options.nZone, mpi_comm);
  except TypeError as exception:
    print('A TypeError occured in pysu2ad.CDriver : ',exception)
    if options.with_MPI == True:
      print('ERROR : You are trying to initialize MPI with a serial build of the wrapper. Please, remove the --parallel option that is incompatible with a serial build.')
    else:
      print('ERROR : You are trying to launch a computation without initializing MPI but the wrapper has been built in parallel. Please add the --parallel option in order to initialize MPI for the wrapper.')
    return
  
  # Launch the solver for the entire computation
  SU2DriverAD.StartSolver()

  # Postprocess the solver and exit cleanly
  SU2DriverAD.Postprocessing()
  
  if SU2DriverAD != None:
    del SU2DriverAD
    
  # Prepare for gradient projection
  restart  = config.RESTART_ADJ_FILENAME
  solution = config.SOLUTION_ADJ_FILENAME  

  # add suffix
  func_name = config.OBJECTIVE_FUNCTION
  suffix    = SU2.io.get_adjointSuffix(func_name)
  restart   = SU2.io.add_suffix(restart,suffix)
  solution  = SU2.io.add_suffix(solution,suffix)

  if os.path.exists(restart):
    shutil.move( restart , solution )
    
  # Initialize the corresponding driver of SU2, this includes solver preprocessing
  try:
    SU2GradientProjectionAD = pysu2ad.CGradientProjection(dumpFilename, mpi_comm)
  except TypeError as exception:
    print('A TypeError occured in pysu2ad.CDriver : ',exception)
    if options.with_MPI == True:
      print('ERROR : You are trying to initialize MPI with a serial build of the wrapper. Please, remove the --parallel option that is incompatible with a serial build.')
    else:
      print('ERROR : You are trying to launch a computation without initializing MPI but the wrapper has been built in parallel. Please add the --parallel option in order to initialize MPI for the wrapper.')
    return
  
  SU2GradientProjectionAD.Run()
  
  # read gradients
  grad_filename  = config.GRAD_OBJFUNC_FILENAME
  raw_gradients = SU2.io.read_gradients(grad_filename)
  return raw_gradients

# -------------------------------------------------------------------
#  Main 
# -------------------------------------------------------------------

#### Used for test-case: Tutorials/design/Inviscid_2D_Unconstrained_NACA0012

def main():

  # Command line options
  parser=OptionParser()
  parser.add_option("-f", "--file", dest="filename", help="Read config from FILE", metavar="FILE")
  parser.add_option("--nZone", dest="nZone", default=1, help="Define the number of ZONES", metavar="NZONE")
  parser.add_option("--parallel", action="store_true",
                    help="Specify if we need to initialize MPI", dest="with_MPI", default=False)

  (options, args) = parser.parse_args()
  options.nZone = int( options.nZone )

  if options.filename == None:
    raise Exception("No config file provided. Use -f flag")
  
  # This is a Python implementation of the config file, should be done with pywrapper in the future
  config = SU2.io.Config(options.filename)

  if options.with_MPI == True:
    from mpi4py import MPI      # use mpi4py for parallel run (also valid for serial)
    comm = MPI.COMM_WORLD
  else:
    comm = 0 

  # PRIMAL
  objective_values = primal(options, config, comm)
  
  gradients = adjoint(options, config, comm)
  
  print (objective_values)
  print (gradients)

# -------------------------------------------------------------------
#  Run Main Program
# -------------------------------------------------------------------

# this is only accessed if running from command prompt
if __name__ == '__main__':
    main()  
