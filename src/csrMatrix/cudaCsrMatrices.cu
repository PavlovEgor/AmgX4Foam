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

#include "deviceCsrMatrix.C"
#include "deviceField.H"
#include "kernels.H"
#include "global.cuh"
#include <cub/cub.cuh>

// * * * * * * * * * * * * * * * * CUDA Kernels  * * * * * * * * * * * * * * //

__global__
void cudaInitializeSequence
(
    const int   totNnz,
          int * tmpPerm
)
{
    int iNnz = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize tmpPerm = [0, 1, ... totNnz-1]
    if(iNnz < totNnz)
    {
        tmpPerm[iNnz] = iNnz;
    }
}

__global__
void cudaInitializeSequencePair
(
    const int   nnz,
          int * rowInd,
          int * colInd
)
{
    int iNnz = blockIdx.x * blockDim.x + threadIdx.x;

    if(iNnz < nnz)
    {
        rowInd[iNnz] = iNnz;
        colInd[iNnz] = iNnz;
    }
}

__global__
void cudaInitializeAddr
(
    const int   nCells,
    const int   nInternalFaces,
    const int * const owner,
    const int * const neighbour,
          int * rowIndTmp,
          int * colIndTmp
)
{
    int iNnz = blockIdx.x * blockDim.x + threadIdx.x;

    if(iNnz < nInternalFaces)
    {
        rowIndTmp[nCells + iNnz] = owner[iNnz];
        colIndTmp[nCells + iNnz] = neighbour[iNnz];

        rowIndTmp[nCells + nInternalFaces + iNnz] = neighbour[iNnz];
        colIndTmp[nCells + nInternalFaces + iNnz] = owner[iNnz];
    }
}


__global__
void cudaInitializeAddrExt
(
    const int   nCells,
    const int   nInternalFaces,
    const int   nnzExt,
    const int * const extRows,
    const int * const extCols,
          int * rowIndTmp,
          int * colIndTmp
)
{
    int iNnz = blockIdx.x * blockDim.x + threadIdx.x;

    if(iNnz < nnzExt)
    {
        rowIndTmp[nCells + 2*nInternalFaces + iNnz] = extRows[iNnz];
        colIndTmp[nCells + 2*nInternalFaces + iNnz] = extCols[iNnz];
    }
}


__global__
void cudaLocToGlobD
(
    const int   nRows,
    const int   diagIndexGlobal,
          int * colIndicesGlobal
)
{
    int iNnz = blockIdx.x * blockDim.x + threadIdx.x;

    if(iNnz < nRows)
    {
        colIndicesGlobal[iNnz] += diagIndexGlobal;
    }
}


__global__
void cudaLocToGlobON
(
    const int   nRows,
    const int   nIntFaces,
    const int   lowOffGlobal,
    const int   uppOffGlobal,
          int * colIndicesGlobal
)
{
    int iNnz = blockIdx.x * blockDim.x + threadIdx.x;

    if(iNnz < nIntFaces)
    {
        colIndicesGlobal[nRows + iNnz] += uppOffGlobal;
        colIndicesGlobal[nRows + nIntFaces + iNnz] += lowOffGlobal;
    }
}


__global__
void cudaComputeNNZ
(
    const int   totNnz,
    const int * const rowInd,
          int * nnz
)
{
    int iNnz = blockIdx.x * blockDim.x + threadIdx.x;

    if(iNnz<totNnz)
    {
        atomicAdd(&nnz[rowInd[iNnz]], 1);
    }
}

template<typename T>
__global__
void cudaApplyPermutation
(
    const int   length,
    const int * const permArray,
    const T   * const srcArray,
          T   *       dstArray
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < length)
    {
        dstArray[i] = srcArray[permArray[i]];
    }
}

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


// * * * * * * * * * * * * * *  Wrapper functions * * * * * * * * * * * * * * //

void Foam::deviceCsrMatrix::initializeAddressing
(
    const int   nInternalFaces,
    const int   totNnz,
    const int * const owner,
    const int * const neighbour,
          int * tmpPerm,
          int * rowIndTmp,
          int * colIndTmp,
    const int   nCells
)
{
    // Initialize tmpPerm = [0, 1, ... totNnz-1]
    int numBlocks = (totNnz + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
    cudaInitializeSequence<<<numBlocks, NUM_THREADS_PER_BLOCK>>>
    (
        totNnz,
        tmpPerm
    );

    // Initialize: rowindicesTmp = [0, ... nCells-1, (owner), (neighbour)]
    //             colindicesTmp = [0, ... nCells-1, (neighbour), (owner)]
    if(nCells > 0)
    {
        numBlocks = nCells / NUM_THREADS_PER_BLOCK + 1;
        cudaInitializeSequencePair<<<numBlocks, NUM_THREADS_PER_BLOCK>>>
        (
            nCells,
            rowIndTmp,
            colIndTmp
        );
    }

    numBlocks = (nInternalFaces + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
    cudaInitializeAddr<<<numBlocks, NUM_THREADS_PER_BLOCK>>>
    (
        nCells,
        nInternalFaces,
        owner,
        neighbour,
        rowIndTmp,
        colIndTmp
    );

    cudaDeviceSynchronize();

    CHECK_LAST_CUDA_ERROR();
    return;
}


void Foam::deviceCsrMatrix::initializeAddressingExt
(
    const int   nInternalFaces,
    const int   nnzExt,
    const int   totNnz,
    const int * const owner,
    const int * const neighbour,
    const int * const extRows,
    const int * const extCols,
          int * tmpPerm,
          int * rowIndTmp,
          int * colIndTmp,
    const int   nCells
)
{
    // Initialize tmpPerm = [0, 1, ... totNnz-1]
    // Initialize: rowindicesTmp = [0, ... nCells-1, (owner), (neighbour), (extrows)]
    //             colindicesTmp = [0, ... nCells-1, (neighbour), (owner), (extcols)]
    initializeAddressing
    (
        nInternalFaces,
        totNnz,
        owner,
        neighbour,
        tmpPerm,
        rowIndTmp,
        colIndTmp,
        nCells
    );

    int numBlocks = (nnzExt + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
    cudaInitializeAddrExt<<<numBlocks, NUM_THREADS_PER_BLOCK>>>
    (
        nCells,
        nInternalFaces,
        nnzExt,
        extRows,
        extCols,
        rowIndTmp,
        colIndTmp
    );

    cudaDeviceSynchronize();

    CHECK_LAST_CUDA_ERROR();
    return;
}


void Foam::deviceCsrMatrix::computeSorting
(
    const int   totNnz,
          int * tmpPerm,
          int * rowIndTmp,
          int * rowInd,
          int * ldu2csr
)
{
    cub::DoubleBuffer<int> d_keys(rowIndTmp, rowInd);
    cub::DoubleBuffer<int> d_values(tmpPerm, ldu2csr);

    // Determine temporary device storage requirements for sort pairs
    void * tempStorage = NULL;
    size_t tempStorageBytes = 0;
    cub::DeviceRadixSort::SortPairs(tempStorage, tempStorageBytes, d_keys, d_values, totNnz);
    // Allocate temporary storage for exclusive sort pairs
    cudaMalloc(&tempStorage, tempStorageBytes);
    // Run radix sort pairs
    cub::DeviceRadixSort::SortPairs(tempStorage, tempStorageBytes, d_keys, d_values, totNnz);

    rowInd = d_keys.Current();
    ldu2csr = d_values.Current();

    cudaFree(tempStorage);

    CHECK_LAST_CUDA_ERROR();
    return;
}


void Foam::deviceCsrMatrix::localToGlobalColIndices
(
    const int nIntFaces,
    const int lowOffGlobal,
    const int uppOffGlobal,
    int *colIndicesGlobal,
    const int nRows,
    const int diagIndexGlobal
)
{
    int numBlocks;
    
    if(nRows > 0)
    {
        numBlocks = (nRows + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
        cudaLocToGlobD<<<numBlocks, NUM_THREADS_PER_BLOCK>>>
        (
            nRows,
            diagIndexGlobal,
            colIndicesGlobal
        );
    }

    numBlocks = (nIntFaces + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
    cudaLocToGlobON<<<numBlocks, NUM_THREADS_PER_BLOCK>>>
    (
        nRows,
        nIntFaces,
        lowOffGlobal,
        uppOffGlobal,
        colIndicesGlobal
    );

    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

    return;
}


void Foam::deviceCsrMatrix::applyAddressingPermutation
(
    const int   nCells,
    const int   totNnz,
    const int * const ldu2csr,
    const int * const colIndTmp,
    const int * const rowInd,
          int * colInd,
          int * ownStart
)
{
    Foam::deviceField<Foam::label> nnz(nCells + 1, Foam::Zero);
    // int * nnz;
    // cudaMalloc((void **)&nnz, (nCells + 1)*sizeof(int));
    // cudaMemset((void **)&nnz, 0, (nCells + 1)*sizeof(int));

    int numBlocks = (totNnz + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
    cudaComputeNNZ<<<numBlocks, NUM_THREADS_PER_BLOCK>>>
    (
        totNnz,
        rowInd,
        nnz.data()
    );
    cudaDeviceSynchronize();

    // Determine temporary device storage requirements for exclusive prefix scan
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, nnz.data(), ownStart, nCells + 1);

    // Allocate temporary storage for exclusive prefix scan
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Run exclusive prefix min-scan
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, nnz.data(), ownStart, nCells + 1);

    cudaApplyPermutation<int><<<numBlocks, NUM_THREADS_PER_BLOCK>>>
    (
        totNnz,
        ldu2csr,
        colIndTmp,
        colInd
    );

    cudaDeviceSynchronize();

    // cudaFree(nnz);
    cudaFree(d_temp_storage);

    CHECK_LAST_CUDA_ERROR();
    return;
}


void Foam::deviceCsrMatrix::initializeValue
(
    const int   nIntFaces,
    const double * const upper,
    const double * const lower,
          double * valuesTmp,
    const int   nCells,
    const double * const diag
)
{
    int numBlocks;
    
    // Initialize valuesTmp = [(diag), (upper), (lower)]
    if(nCells > 0)
    {
        numBlocks = (nCells + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
        cudaInitializeValueD<<<numBlocks, NUM_THREADS_PER_BLOCK>>>
        (
            nCells,
            diag,
            valuesTmp
        );
    }
    
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


void Foam::deviceCsrMatrix::initializeValueExt
(
    const int   nIntFaces,
    const int   nnzExt,
    const double * const upper,
    const double * const lower,
    const double * const extValue,
          double * valuesTmp,
    const int   nCells,
    const double * const diag

)
{
    // Initialize valuesTmp = [(diag), (upper), (lower), (extValues)]
    initializeValue
    (
        nIntFaces,
        upper,
        lower,
        valuesTmp,
        nCells,
        diag
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


void Foam::deviceCsrMatrix::applyValuePermutation
(
    const int   totNnz,
    const int * const ldu2csr,
    const double * const valuesTmp,
          double * values
)
{
    int numBlocks = (totNnz + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
    cudaApplyPermutation<double><<<numBlocks, NUM_THREADS_PER_BLOCK>>>
    (
        totNnz,
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
