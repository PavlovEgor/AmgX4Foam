/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2016 OpenFOAM Foundation
    Copyright (C) 2017-2022 OpenCFD Ltd.
    Copyright (C) 2022-2023 Cineca
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cusparse.h>

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

template<typename T>
int checkCusparse(T f, const char* const func, const char* const file, const int line)
{
    const cusparseStatus_t __s = f;

    if(__s != CUSPARSE_STATUS_SUCCESS)
    {
        std::cerr << "CUDA status in: " << f << " [" << __FILE__ << ","
            << __LINE__ << "], " << std::endl;
        exit(1);
    }
    return static_cast<int>(__s);
}

#define CHECK_CUSPARSE(val) checkCusparse((val), #val, __FILE__, __LINE__)

#endif
