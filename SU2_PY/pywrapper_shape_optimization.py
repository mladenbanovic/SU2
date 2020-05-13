#!/usr/bin/env python

from __future__ import division, print_function, absolute_import
from optparse import OptionParser    # use a parser for configuration
import SU2                # imports SU2 python tools
import pysu2
import pysu2ad            # imports the SU2 AD wrapped module
import os
import shutil
import copy
from numpy import linalg
import sys

class OptHandle(object):
  def __init__(self, options, config, mpi_comm):
    self.options = options
    self.config = config
    self.mpi_comm = mpi_comm
    
    # Initial set of design parameters
    self.x0 = copy.deepcopy(config['DV_VALUE_NEW'])
    self.numberOfDesignParameters = len(self.x0)
    
    self.currentIteration = 0
    self.maxIter      = int (config.OPT_ITERATIONS)                      # number of opt iterations
    bound_upper       = float (config.OPT_BOUND_UPPER)                   # variable bound to be scaled by the line search
    bound_lower       = float (config.OPT_BOUND_LOWER)                   # variable bound to be scaled by the line search
    relax_factor      = float (config.OPT_RELAX_FACTOR)                  # line search scale
    gradient_factor   = float (config.OPT_GRADIENT_FACTOR)               # objective function and gradient scale
 
    self.accu = float (config.OPT_ACCURACY) * gradient_factor            # optimizer accuracy
    
    xb_low = [float(bound_lower)/float(relax_factor)]*self.numberOfDesignParameters      # lower dv bound it includes the line search acceleration factor
    xb_up  = [float(bound_upper)/float(relax_factor)]*self.numberOfDesignParameters      # upper dv bound it includes the line search acceleration fa
    self.xbounds = list(zip(xb_low, xb_up)) # design bounds
    
    # prescale x0
    dv_size = config['DEFINITION_DV']['SIZE']
    dv_scales = config['DEFINITION_DV']['SCALE']
    k = 0
    for i, dv_scl in enumerate(dv_scales):
      for j in range(dv_size[i]):
        self.x0[k] =self.x0[k]/dv_scl;
        k = k + 1
    print (k)
    
    # scale accuracy
    obj = config['OPT_OBJECTIVE']
    obj_scale = []
    for this_obj in obj.keys():
      obj_scale = obj_scale + [obj[this_obj]['SCALE']]
      
    # Only scale the accuracy for single-objective problems: 
    if len(obj.keys())==1:
      self.accu = self.accu*obj_scale[0]
      
    # scale accuracy
    self.eps = 1.0e-04
    
    # optimizer summary
    sys.stdout.write('Sequential Least SQuares Programming (SLSQP) parameters:\n')
    sys.stdout.write('Number of design variables: ' + str(self.numberOfDesignParameters) + '\n' )
    sys.stdout.write('Objective function scaling factor: ' + str(obj_scale) + '\n')
    sys.stdout.write('Maximum number of iterations: ' + str(self.maxIter) + '\n')
    sys.stdout.write('Requested accuracy: ' + str(self.accu) + '\n')
    sys.stdout.write('Initial guess for the independent variable(s): ' + str(self.x0) + '\n')
    sys.stdout.write('Lower and upper bound for each independent variable: ' + str(self.xbounds) + '\n\n')

  def update_mesh(self):
    deform_todo = not self.config['DV_VALUE_NEW'] == self.config['DV_VALUE_OLD']
    if deform_todo:
      # for some reason, mesh deformation is done always in the initial mesh
      mesh_name = self.config['MESH_FILENAME']
      suffix = 'initial'
      initial_meshname_suffixed = SU2.io.add_suffix( mesh_name , suffix )
      # copy the initial mesh file over the current one to get ready for the mesh deformation
      if os.path.exists(initial_meshname_suffixed):
        shutil.copy( initial_meshname_suffixed , mesh_name )
      
      # setup output mesh name
      suffix = 'deform'
      mesh_name = self.config['MESH_FILENAME']
      meshname_suffixed = SU2.io.add_suffix( mesh_name , suffix )
      self.config['MESH_OUT_FILENAME'] = meshname_suffixed
      
      dumpFilename = 'config.cfg'
      self.config.dump(dumpFilename)
      try:
        SU2MeshDeformation = pysu2.CMeshDeformation(dumpFilename, self.mpi_comm)
      except TypeError as exception:
        print('A TypeError occured in pysu2.CDriver : ',exception)
        if self.options.with_MPI == True:
          print('ERROR : You are trying to initialize MPI with a serial build of the wrapper. Please, remove the --parallel option that is incompatible with a serial build.')
        else:
          print('ERROR : You are trying to launch a computation without initializing MPI but the wrapper has been built in parallel. Please add the --parallel option in order to initialize MPI for the wrapper.')
        return
      
      SU2MeshDeformation.Run()
      print ("SU2MeshDeformation successfully evaluated")
      # once the deformation is finished, rename the result
      if os.path.exists(meshname_suffixed):
        shutil.move( meshname_suffixed , mesh_name )
      # update DV_VALUE_OLD
  #     config.update({ 'MESH_FILENAME' : config['MESH_OUT_FILENAME'] , 
  #                     'DV_VALUE_OLD'  : config['DV_VALUE_NEW']      })
      self.config.update({'DV_VALUE_OLD'  : self.config['DV_VALUE_NEW']})
      
    return
    
  # -------------------------------------------------------------------
  #  Primal (Objective function(s)) 
  # -------------------------------------------------------------------
  def primal(self, design_parameters):
    self.config['MATH_PROBLEM']  = 'DIRECT'
    self.x0 = copy.deepcopy(design_parameters)
    self.config.unpack_dvs(design_parameters)
    # update mesh if design is changed
    dumpFilename = 'config.cfg'
    if(self.currentIteration > 0):
      self.update_mesh()
      self.config['RESTART_SOL'] = 'YES'
      self.config.dump(dumpFilename)
      
    # Initialize the corresponding driver of SU2, this includes solver preprocessing
    try:
      if(self.currentIteration > 0):
        SU2Driver = pysu2.CSinglezoneDriver(dumpFilename, self.options.nZone, self.mpi_comm)
      else:
        SU2Driver = pysu2.CSinglezoneDriver(self.options.filename, self.options.nZone, self.mpi_comm)
    except TypeError as exception:
      print('A TypeError occured in pysu2.CDriver : ',exception)
      if self.options.with_MPI == True:
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
    plot_format = self.config.get('TABULAR_FORMAT', 'CSV')
    plot_extension = SU2.io.get_extension(plot_format)
    history_filename = self.config['CONV_FILENAME'] + plot_extension
    
    # Read the objective values from history file
    aerodynamics = SU2.io.read_aerodynamics(history_filename, self.config.NZONES)
    funcs = SU2.util.ordered_bunch()
    for key in SU2.io.historyOutFields:
      if key in aerodynamics:
        funcs[key] = aerodynamics[key]
    #print (funcs)
        
    def_objs = self.config['OPT_OBJECTIVE']
    objectives = def_objs.keys()
    func_vals_sum = 0.
    for i_obj,this_obj in enumerate(objectives):
      scale = def_objs[this_obj]['SCALE']
      global_factor = float(self.config['OPT_GRADIENT_FACTOR'])
      sign  = SU2.io.get_objectiveSign(this_obj)
      
      func_vals_sum += funcs[this_obj] * sign * scale * global_factor
      
    self.currentIteration += 1
    
    return func_vals_sum
  
  # -------------------------------------------------------------------
  #  Adjoint
  # -------------------------------------------------------------------
  def adjoint(self, design_parameters):
    self.config['MATH_PROBLEM']  = 'DISCRETE_ADJOINT'
    
    ### check if design parameters are changed, if so, run primal beforehand
    ### TODO
    ###
    
    if(self.currentIteration > 1):
      self.config['RESTART_SOL'] = 'YES'
    
    dumpFilename = 'config.cfg'
    self.config.dump(dumpFilename)
    
    #RESTART TO SOLUTION
    restart  = self.config.RESTART_FILENAME
    solution = self.config.SOLUTION_FILENAME
    if os.path.exists(restart):
      shutil.move( restart , solution )
    
    # Initialize the corresponding driver of SU2, this includes solver preprocessing
    try:
      SU2DriverAD = pysu2ad.CDiscAdjSinglezoneDriver(dumpFilename, self.options.nZone, self.mpi_comm);
    except TypeError as exception:
      print('A TypeError occured in pysu2ad.CDriver : ',exception)
      if self.options.with_MPI == True:
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
    restart  = self.config.RESTART_ADJ_FILENAME
    solution = self.config.SOLUTION_ADJ_FILENAME  
  
    # add suffix
    func_name = self.config.OBJECTIVE_FUNCTION
    suffix    = SU2.io.get_adjointSuffix(func_name)
    restart   = SU2.io.add_suffix(restart,suffix)
    solution  = SU2.io.add_suffix(solution,suffix)
  
    if os.path.exists(restart):
      shutil.move( restart , solution )
      
    # Initialize the corresponding driver of SU2, this includes solver preprocessing
    try:
      SU2GradientProjectionAD = pysu2ad.CGradientProjection(dumpFilename, self.mpi_comm)
    except TypeError as exception:
      print('A TypeError occured in pysu2ad.CDriver : ',exception)
      if self.options.with_MPI == True:
        print('ERROR : You are trying to initialize MPI with a serial build of the wrapper. Please, remove the --parallel option that is incompatible with a serial build.')
      else:
        print('ERROR : You are trying to launch a computation without initializing MPI but the wrapper has been built in parallel. Please add the --parallel option in order to initialize MPI for the wrapper.')
      return
    
    SU2GradientProjectionAD.Run()
    
    # read gradients
    grad_filename  = self.config.GRAD_OBJFUNC_FILENAME
    raw_gradients = SU2.io.read_gradients(grad_filename)
    #print(raw_gradients)
    
    #scale gradients according to config
    ############ ADJOINT SCALING
    dv_scales = self.config['DEFINITION_DV']['SCALE']
    dv_size   = self.config['DEFINITION_DV']['SIZE']
    
    def_objs = self.config['OPT_OBJECTIVE']
    this_obj = def_objs.keys()[0]
    scale = def_objs[this_obj]['SCALE']
    global_factor = float(self.config['OPT_GRADIENT_FACTOR'])
    sign  = SU2.io.get_objectiveSign(this_obj)
    
    gradients = [0.0]*self.numberOfDesignParameters
    k = 0
    for i_dv,dv_scl in enumerate(dv_scales):
      for i_grd in range(dv_size[i_dv]):
        gradients[k] = raw_gradients[k] * sign * scale * global_factor / dv_scl
        k = k + 1
    
    return gradients

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
    
  optHandle = OptHandle(options, config, comm)
    
  current_iteration = 0
    
  designparams = copy.deepcopy(config['DV_VALUE_NEW'])

  # PRIMAL
  # iteration 0
  #objective_values = primal(designparams, options, config, comm, current_iteration)
  objective_values = optHandle.primal(designparams)
  
  #gradients = adjoint(options, config, comm, current_iteration)
  gradients = optHandle.adjoint(designparams)
  print("%5i %5i % 16.6E % 16.6E" % (current_iteration,current_iteration,
                                               objective_values,linalg.norm(gradients)))
  
  #print (objective_values)
  #print (gradients)
  
  current_iteration += 1
  

  
  #propose new values of design parameters
  config_read = SU2.io.Config("iteration2designs.cfg")
  myx = config_read['DV_VALUE_NEW']
  print (myx)
  
  objective_values = optHandle.primal(myx)
  
  gradients = optHandle.adjoint(myx)
  
  #print (objective_values)
  #print (gradients)
  print("%5i %5i % 16.6E % 16.6E" % (current_iteration,current_iteration,
                                               objective_values,linalg.norm(gradients)))
  
  current_iteration += 1
  
  #propose new values of design parameters
  config_read = SU2.io.Config("iteration3designs.cfg")
  myx = config_read['DV_VALUE_NEW']
  print (myx)
  
  #objective_values = primal(myx, options, config, comm, current_iteration)
  
  #gradients = adjoint(options, config, comm, current_iteration)
  
  #print (objective_values)
  #print (gradients)
  print("%5i %5i % 16.6E % 16.6E" % (current_iteration,current_iteration,
                                               objective_values,linalg.norm(gradients)))
  
  current_iteration += 1
  
  #config.unpack_dvs(x)

# -------------------------------------------------------------------
#  Run Main Program
# -------------------------------------------------------------------

# this is only accessed if running from command prompt
if __name__ == '__main__':
    main()  
