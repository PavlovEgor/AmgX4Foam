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

// ************************************************************************* //

#include "cudaCsrMatrixExecutor.H"
#include "global.cuh"
#include <cub/cub.cuh>

// * * * * * * * * * * * * * * * * CUDA Kernels  * * * * * * * * * * * * * * //

__global__
void cudaInitializeValueD
(
    const int   nCells,
    const double * const diag,
          double * valuesTmp
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < nCells)
    {
        valuesTmp[i] = diag[i];
    }
}

__global__
void cudaInitializeValueUL
(
    const int   nCells,
    const int   nIntFaces,
    const double * const upper,
    const double * const lower,
          double * valuesTmp
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < nIntFaces)
    {
        valuesTmp[nCells + i] = upper[i];
        valuesTmp[nCells + nIntFaces + i] = lower[i];
    }
}

__global__
void cudaInitializeValueExt
(
    const int   nCells,
    const int   nIntFaces,
    const int   nnzExt,
    const double * const extValues,
          double * valuesTmp
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < nnzExt)
    {
        valuesTmp[nCells + 2*nIntFaces + i] = extValues[i];
    }
}

__global__
void cudaApplyPermutation 
(
    const int      length,
    const int      blockLen,
    const int    * const permArray,
    const double * const srcArray,
          double * dstArray
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < length)
    {
        dstArray[i] = srcArray[permArray[i / blockLen] + i % blockLen];
    }
} 
//NOTA: this function (when csrAdressing will be joined back to csrMatrix) will 
//      become e template on the array type to be used both for adressing and 
//      values permutaiton


// * * * * * * * * * * * * * *  Wrapper functions * * * * * * * * * * * * * * //

void Foam::cudaCsrMatrixExecutor::initializeValue
(
    const label    nCells,
    const label    nIntFaces,
    const scalar * const diag,
    const scalar * const upper,
    const scalar * const lower,
          scalar * valuesTmp    
) const
{
    label numBlocks;
    
    // Initialize valuesTmp = [(diag), (upper), (lower)]
    numBlocks = (nCells + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
    cudaInitializeValueD<<<numBlocks, NUM_THREADS_PER_BLOCK>>>
    (
        nCells,
        diag,
        valuesTmp
    );
    
    numBlocks = (nIntFaces + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
    cudaInitializeValueUL<<<numBlocks, NUM_THREADS_PER_BLOCK>>>
    (
        nCells,
        nIntFaces,
        upper,
        lower,
        valuesTmp
    );

    cudaDeviceSynchronize();

    CHECK_LAST_CUDA_ERROR();
    return;
}


void Foam::cudaCsrMatrixExecutor::initializeValueExt
(
    const label    nCells,
    const label    nIntFaces,
    const label    nnzExt,
    const scalar * const diag,
    const scalar * const upper,
    const scalar * const lower,
    const scalar * const extValue,
          scalar * valuesTmp
) const
{
    // Initialize valuesTmp = [(diag), (upper), (lower), (extValues)]
    initializeValue
    (
        nCells,
        nIntFaces,
        diag,
        upper,
        lower,
        valuesTmp
    );

    int numBlocks = (nnzExt + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
    cudaInitializeValueExt<<<numBlocks, NUM_THREADS_PER_BLOCK>>>
    (
        nCells,
        nIntFaces,
        nnzExt,
        extValue,
        valuesTmp
    );

    cudaDeviceSynchronize();

    CHECK_LAST_CUDA_ERROR();
    return;
}


void Foam::cudaCsrMatrixExecutor::applyValuePermutation
(
    const label    totNnz,
    const label  * const ldu2csr,
    const scalar * const valuesTmp,
          scalar * values,
    const label    nBlocks
) const
{
    int blockLen = nBlocks*nBlocks;
    int numBlocks = (totNnz + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
    cudaApplyPermutation<<<numBlocks, NUM_THREADS_PER_BLOCK>>>
    (
        totNnz,
        blockLen,
        ldu2csr,
        valuesTmp,
        values
    );

    cudaDeviceSynchronize();

    CHECK_LAST_CUDA_ERROR();
    return;
}

// * * * * * * * * * * * * * Explicit instantiations  * * * * * * * * * * *  //

// ************************************************************************* //
