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

#include "csrAddressing.H"
#include "global.cuh"

// * * * * * * * * * * * * * * * * CUDA Kernels  * * * * * * * * * * * * * * //

//__global__
//void cudaComputeNormFactor
//(
//    const int size,
//    const double * const Apsi,
//    const double * const avgPsiSumA,
//    const double * const source,
//    double * const normFactor
//)
//{
//    __shared__ double temp[NUM_THREADS_PER_BLOCK];
//    int celli    = blockIdx.x * blockDim.x + threadIdx.x;
//    int celliLoc = threadIdx.x;
//    temp[celliLoc] = 0.0;
//    if (celli < size)
//    {
//        temp[celliLoc] =  std::abs(Apsi[celli]   - avgPsiSumA[celli]) +
//                          std::abs(source[celli] - avgPsiSumA[celli]);
//
//        __syncthreads();
//
//        if (celliLoc == 0)
//        {
//            double sum = 0;
//            for (int iThread = 0; iThread < NUM_THREADS_PER_BLOCK; ++iThread)
//            {
//                sum += temp[iThread];
//            }
//            atomicAdd(normFactor, sum);
//        }
//    }
//}
//
//template<int nComps>
//__global__
//void cudaComputeH
//(
//    const int nInternalFaces,
//    const int* const uPtr,
//    const int* const lPtr,
//    const double* const lowerPtr,
//    const double* const upperPtr,
//    const double* const psiPtr,
//          double* const HpsiPtr
//)
//{
//    int facei = blockIdx.x * blockDim.x + threadIdx.x;
//    if (facei < nInternalFaces)
//    {
//        for (int cmpt = 0; cmpt < nComps; ++cmpt)
//        {
//            atomicAdd(&HpsiPtr[cmpt + nComps*uPtr[facei]], -lowerPtr[facei]*psiPtr[cmpt + nComps*lPtr[facei]]);
//            atomicAdd(&HpsiPtr[cmpt + nComps*lPtr[facei]], -upperPtr[facei]*psiPtr[cmpt + nComps*uPtr[facei]]);
//        }
//    }
//}

// * * * * * * * * * * * * Public Member Functions * * * * * * * * * * * * * //

template<class Type>
Type* Foam::cudaCsrAddressingExecutor::alloc
(
    int size
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
void Foam::cudaCsrAddressingExecutor::clear(Type* ptr) const
{
    if (ptr)
    {
        int err = CHECK_CUDA_ERROR(cudaFree(ptr));
    }
}
// * * * * * * * * * * * * * Explicit instantiations  * * * * * * * * * * * //

#define makecudaCsrAddressingExecutor(Type)                             \
    template Type* Foam::cudaCsrAddressingExecutor::alloc<Type>         \
    (                                                                   \
        int size                                                   \
    ) const;                                                            \
    template void  Foam::cudaCsrAddressingExecutor::clear<Type>         \
    (                                                                   \
        Type* ptr                                                       \
    ) const;

makecudaCsrAddressingExecutor(int)
makecudaCsrAddressingExecutor(double)

// ************************************************************************* //

