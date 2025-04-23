/*---------------------------------------------------------------------------*\
-------------------------------------------------------------------------------
    Copyright (C) 2025 CINECA
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

#include <global.cuh>

template <typename T>
int checkCudaError
(
    T err,
    const char* const func,
    const char* const file,
    const int line
)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
	abort();
    }
    return static_cast<int>(err);
}

int checkLastCudaError
(
    const char* const file,
    const int line
)
{
    cudaError_t err(cudaGetLastError());
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
	abort();
    }
    return static_cast<int>(err);
}

template int checkCudaError<cudaError_t>(cudaError_t err, const char* const func,
                                          const char* const file, const int line);

__global__
void cudaPrintDeviceDouble(const double* value, int length)
{
    for(int i=0; i<length; ++i) printf("%23.16e ", value[i]);
    printf("\n");
}

void cudaKernels::printDeviceDouble(const double* value, int length)
{
    cudaPrintDeviceDouble<<<1, 1>>>(value, length);
    cudaDeviceSynchronize();
    return;
}

__global__
void cudaPrintDeviceInteger(const int* value, int length)
{
    for(int i=0; i<length; ++i) printf("%d ", value[i]);
    printf("\n");
}

void cudaKernels::printDeviceInteger(const int* value, int length)
{
    cudaPrintDeviceInteger<<<1, 1>>>(value, length);
    cudaDeviceSynchronize();
    return;
}

// ************************************************************************* //
