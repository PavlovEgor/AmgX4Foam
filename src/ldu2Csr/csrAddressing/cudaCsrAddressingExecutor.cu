/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2016-2022 OpenCFD Ltd.
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

#include "cudaCsrAddressingExecutor.H"

#include "label.H"
#include "csrAddressing.H"
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


// * * * * * * * * * * * * Public Member Functions * * * * * * * * * * * * * //

template<class Type>
Type* Foam::cudaCsrAddressingExecutor::alloc
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
const Type* Foam::cudaCsrAddressingExecutor::copyFromFoam
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
void Foam::cudaCsrAddressingExecutor::clear(Type* ptr) const
{
    if (ptr)
    {
        int err = CHECK_CUDA_ERROR(cudaFree(ptr));
    }
}

template<class Type>
void Foam::cudaCsrAddressingExecutor::clear(const Type* ptr) const
{
    if (ptr)
    {
        int err = CHECK_CUDA_ERROR(cudaFree((Type*) ptr));
    }
}


void Foam::cudaCsrAddressingExecutor::initializeAddressing
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

void Foam::cudaCsrAddressingExecutor::initializeAddressingExt
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

void Foam::cudaCsrAddressingExecutor::computeSorting
(
    const label   totNnz,
          label * tmpPerm,
          label * rowIndTmp,
          label * rowInd,
          label * ldu2csr
) const
{
    label * rowIndHost = new label[totNnz];
    cudaMemcpy((void *) rowIndHost, (const void *) rowIndTmp, sizeof(label)*totNnz, cudaMemcpyDeviceToHost);
    Info << "-> rowIndTmp array: ";
    for(int i=0; i<10; ++i) Info << rowIndHost[i] << " ";
    Info << nl;
    label check = 0;
    for(int i=0; i<totNnz; ++i) check += rowIndHost[i];
    Info << "-> sum of rowINdTmp = " << check << nl;
    
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

void Foam::cudaCsrAddressingExecutor::localToGlobalColIndices
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

void Foam::cudaCsrAddressingExecutor::applyAddressingPermutation
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

    cudaDeviceSynchronize();

    //cudaMemset((void **)&ownStart, 0, (nCells + 1)*sizeof(label));

    label numBlocks = (totNnz + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;

    Info << "--> totNnz = " << totNnz << nl;
    Info << "--> thread per block = " << NUM_THREADS_PER_BLOCK << ", numBlocks = " << numBlocks << nl;

    cudaComputeNNZ<<<numBlocks, NUM_THREADS_PER_BLOCK>>>
    (
        totNnz,
        rowInd,
        nnz
    );
    cudaDeviceSynchronize();

    label * nnzHost = new label[nCells+1];
    cudaMemcpy((void *) nnzHost, (const void *) nnz, sizeof(label)*(nCells+1), cudaMemcpyDeviceToHost);
    Info << "-> nnz array: ";
    for(int i=0; i<10; ++i) Info << nnzHost[i] << " ";
    Info << nl;

    // Determine temporary device storage requirements for exclusive prefix scan
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, nnz, ownStart, nCells + 1);

    // Allocate temporary storage for exclusive prefix scan
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Run exclusive prefix min-scan
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, nnz, ownStart, nCells + 1);

    /*label * ownStartHost = new label[nCells+1];
    label * rowIndHost = new label[totNnz];
    cudaMemcpy((void *) rowIndHost, (const void *) rowInd, sizeof(label)*totNnz, cudaMemcpyDeviceToHost);
    ownStartHost[0] = 0;
    label curRow = 0;
    for(label i=0; i<totNnz; ++i)
    {
        if(curRow < rowIndHost[i])
        {
            if(rowIndHost[i] > nCells) fprintf(stderr, "-> row index too high\n");
            ownStartHost[rowIndHost[i]] = i;
        }
        curRow = rowIndHost[i];
    }
    ownStartHost[nCells] = totNnz;
    cudaMemcpy((void *) ownStart, (const void *) ownStartHost, sizeof(label)*(nCells+1), cudaMemcpyHostToDevice);*/

    cudaApplyPermutation<int><<<numBlocks, NUM_THREADS_PER_BLOCK>>>
    (
        totNnz,
        ldu2csr,
        colIndTmp,
        colInd
    );

    label * colIndHost = new label[totNnz];
    cudaMemcpy((void *) colIndHost, (const void *) colInd, sizeof(label)*totNnz, cudaMemcpyDeviceToHost);
    label check = 0;
    for(int i=0; i<totNnz; ++i) check += colIndHost[i];
    Info << "-> sum of colIndTmp = " << check << nl;

    cudaDeviceSynchronize();

    // cudaFree(nnz);
    cudaFree(d_temp_storage);

    CHECK_LAST_CUDA_ERROR();
    return;
}

// * * * * * * * * * * * * * Explicit instantiations  * * * * * * * * * * * //

#define makecudaCsrAddressingExecutor(Type)                                   \
    template Type* Foam::cudaCsrAddressingExecutor::alloc<Type>               \
    (                                                                         \
        Foam::label size                                                      \
    ) const;                                                                  \
    template const Type* Foam::cudaCsrAddressingExecutor::copyFromFoam<Type>  \
    (                                                                         \
        Foam::label size,                                                     \
        const Type* hostPtr                                                   \
    ) const;                                                                  \
    template void  Foam::cudaCsrAddressingExecutor::clear<Type>               \
    (                                                                         \
        Type* ptr                                                             \
    ) const;                                                                  \
    template void  Foam::cudaCsrAddressingExecutor::clear<Type>               \
    (                                                                         \
        const Type* ptr                                                       \
    ) const;

makecudaCsrAddressingExecutor(Foam::label)
makecudaCsrAddressingExecutor(Foam::scalar)

