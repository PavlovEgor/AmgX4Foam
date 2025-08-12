/*---------------------------------------------------------------------------*\
-------------------------------------------------------------------------------
    Copyright (C) 2025 Cineca
-------------------------------------------------------------------------------
License
    This file is part of foamExternalSolvers.

    foamExternalSolvers is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    foamExternalSolvers is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with foamExternalSolvers. If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>

#ifndef CUDA_GLOBAL
#define CUDA_GLOBAL

#define NUM_THREADS_PER_BLOCK 128
#define CUDA_MEM_ALIGN_BYTES 16

int checkLastCudaError
(
    const char* const file,
    const int line
);
#define CHECK_LAST_CUDA_ERROR() checkLastCudaError(__FILE__, __LINE__)

template <typename T>
int checkCudaError
(
    T err,
    const char* const func,
    const char* const file,
    const int line
);
#define CHECK_CUDA_ERROR(val) checkCudaError((val), #val, __FILE__, __LINE__)

namespace cudaKernels
{
    void printDeviceDouble(const double* value, int length);
    void printDeviceInteger(const int* value, int length);
}

#endif

// ************************************************************************* //
