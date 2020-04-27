/*!
 * \file CMeshDeformation.hpp
 * \brief Header the Gradient Projection Code (used in SU2_DOT).
 * \author M. Banovic
 * \version 7.0.3 "Blackbird"
 *
 * SU2 Project Website: https://su2code.github.io
 *
 * The SU2 Project is maintained by the SU2 Foundation
 * (http://su2foundation.org)
 *
 * Copyright 2012-2020, SU2 Contributors (cf. AUTHORS.md)
 *
 * SU2 is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * SU2 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with SU2. If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "../../Common/include/mpi_structure.hpp"
#include "../../Common/include/omp_structure.hpp"

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cmath>

#include "../../SU2_CFD/include/solvers/CSolver.hpp"
#include "../../SU2_CFD/include/output/CMeshOutput.hpp"
#include "../../Common/include/geometry/CPhysicalGeometry.hpp"
#include "../../Common/include/CConfig.hpp"
#include "../../Common/include/grid_movement_structure.hpp"

using namespace std;

class CMeshDeformation {
public:

  /*!
   * \brief Constructor of the class.
   * \param[in] confFile - Configuration file name.
   * \param[in] MPICommunicator - MPI communicator for SU2.
   */
  CMeshDeformation(const char* confFile,
                   const SU2_Comm MPICommunicator);

  /*!
   * \brief Destructor of the class.
   */
  ~CMeshDeformation();

  /*!
   * \brief Execute mesh deformation
   */
  void Run();

private:
  char* config_file_name;
};

