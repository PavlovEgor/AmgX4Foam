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

#include "scalar.H"
#include "csrMatrix.H"
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

template<class Type>
Type* Foam::cudaCsrMatrixExecutor::alloc
(
    Foam::label size
) const
{
    void* ptr;
    int err = CHECK_CUDA_ERROR(cudaMalloc((void**)&ptr, size*sizeof(Type)));
    if (err != 0)
    {
        FatalErrorInFunction << "ERROR: cudaMalloc returned " << err << abort(FatalError);
    }
    return static_cast<Type*>(ptr);
}

template<class Type>
const Type* Foam::cudaCsrMatrixExecutor::copyFromFoam
(
    Foam::label size,
	const Type* hostPtr
) const
{
	void* ptr;
    label err = CHECK_CUDA_ERROR(cudaMalloc((void**)&ptr, (size_t) size*sizeof(Type)));
    if (err != 0)
    {
        FatalErrorInFunction << "ERROR: cudaMalloc returned " << err << abort(FatalError);
    }

    err = CHECK_CUDA_ERROR(cudaMemcpy(ptr, hostPtr, (size_t) size*sizeof(Type), cudaMemcpyHostToDevice));
    if (err != 0)
    {
        FatalErrorInFunction << "ERROR: cudaMemcpy returned " << err << abort(FatalError);
    }

    return static_cast<const Type*>(ptr);
}

template<class Type>
void Foam::cudaCsrMatrixExecutor::clear(Type* ptr) const
{
    if (ptr)
    {
        int err = CHECK_CUDA_ERROR(cudaFree(ptr));
    }
}

template<class Type>
void Foam::cudaCsrMatrixExecutor::clear(const Type* ptr) const
{
    if (ptr)
    {
        int err = CHECK_CUDA_ERROR(cudaFree((Type*) ptr));
    }
}


void Foam::cudaCsrMatrixExecutor::initializeAddressing
(
    const Foam::label   nCells,
    const Foam::label   nInternalFaces,
    const Foam::label   totNnz,
    const Foam::label * const owner,
    const Foam::label * const neighbour,
          Foam::label * tmpPerm,
          Foam::label * rowIndTmp,
          Foam::label * colIndTmp
) const
{
    // Initialize tmpPerm = [0, 1, ... totNnz-1]
    label numBlocks = (totNnz + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
    cudaInitializeSequence<<<numBlocks, NUM_THREADS_PER_BLOCK>>>
    (
        totNnz,
        tmpPerm
    );

    // Initialize: rowindicesTmp = [0, ... nCells-1, (owner), (neighbour)]
    //             colindicesTmp = [0, ... nCells-1, (neighbour), (owner)]
    numBlocks = (nCells + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK + 1;
    cudaInitializeSequencePair<<<numBlocks, NUM_THREADS_PER_BLOCK>>>
    (
        nCells,
        rowIndTmp,
        colIndTmp
    );

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

void Foam::cudaCsrMatrixExecutor::initializeAddressingExt
(
    const label   nCells,
    const label   nInternalFaces,
    const label   nnzExt,
    const label   totNnz,
    const label * const owner,
    const label * const neighbour,
    const label * const extRows,
    const label * const extCols,
          label * tmpPerm,
          label * rowIndTmp,
          label * colIndTmp
) const
{
    this->initializeAddressing
    (
        nCells,
        nInternalFaces,
        totNnz,
        owner,
        neighbour,
        tmpPerm,
        rowIndTmp,
        colIndTmp
    );

    label numBlocks = (nnzExt + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
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

void Foam::cudaCsrMatrixExecutor::computeSorting
(
    const label   totNnz,
          label * tmpPerm,
          label * rowIndTmp,
          label * rowInd,
          label * ldu2csr
) const
{   
    cub::DoubleBuffer<label> d_keys(rowIndTmp, rowInd);
    cub::DoubleBuffer<label> d_values(tmpPerm, ldu2csr);

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

void Foam::cudaCsrMatrixExecutor::localToGlobalColIndices
(
    const label nRows,
    const label nIntFaces,
    const label diagIndexGlobal,
    const label lowOffGlobal,
    const label uppOffGlobal,
    label *colIndicesGlobal
) const
{
    label numBlocks;

    numBlocks = (nRows + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
    cudaLocToGlobD<<<numBlocks, NUM_THREADS_PER_BLOCK>>>
    (
        nRows,
        diagIndexGlobal,
        colIndicesGlobal
    );

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

void Foam::cudaCsrMatrixExecutor::applyAddressingPermutation
(
    const label   nCells, //NOTE: it is not used but is need for the cuda kernel
    const label   totNnz,
    const label * const ldu2csr,
    const label * const colIndTmp,
    const label * const rowInd,
          label * colInd,
          label * ownStart
) const
{
    //Foam::deviceField<Foam::label> nnz(nCells + 1, Foam::Zero);
    label * nnz;
    cudaMalloc((void **)&nnz, (nCells + 1)*sizeof(label));
    cudaMemset((void *)nnz, 0, (nCells + 1)*sizeof(label));

    label numBlocks = (totNnz + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
    cudaComputeNNZ<<<numBlocks, NUM_THREADS_PER_BLOCK>>>
    (
        totNnz,
        rowInd,
        nnz
    );
    cudaDeviceSynchronize();

    // Determine temporary device storage requirements for exclusive prefix scan
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, nnz, ownStart, nCells + 1);

    // Allocate temporary storage for exclusive prefix scan
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Run exclusive prefix min-scan
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, nnz, ownStart, nCells + 1);

    cudaApplyPermutation<int><<<numBlocks, NUM_THREADS_PER_BLOCK>>>
    (
        totNnz,
        ldu2csr,
        colIndTmp,
        colInd
    );
    cudaDeviceSynchronize();

    cudaFree(nnz);
    cudaFree(d_temp_storage);

    CHECK_LAST_CUDA_ERROR();
    return;
}


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

#define makecudaCsrMatrixExecutor(Type)                                   \
    template Type* Foam::cudaCsrMatrixExecutor::alloc<Type>               \
    (                                                                         \
        Foam::label size                                                      \
    ) const;                                                                  \
    template const Type* Foam::cudaCsrMatrixExecutor::copyFromFoam<Type>  \
    (                                                                         \
        Foam::label size,                                                     \
        const Type* hostPtr                                                   \
    ) const;                                                                  \
    template void  Foam::cudaCsrMatrixExecutor::clear<Type>               \
    (                                                                         \
        Type* ptr                                                             \
    ) const;                                                                  \
    template void  Foam::cudaCsrMatrixExecutor::clear<Type>               \
    (                                                                         \
        const Type* ptr                                                       \
    ) const;

makecudaCsrMatrixExecutor(Foam::label)
makecudaCsrMatrixExecutor(Foam::scalar)

// ************************************************************************* //
