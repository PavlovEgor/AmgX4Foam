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

#include "cpuCsrAddressingExecutor.H"
#include "csrAddressing.H"
#include <cmath>
#include <bits/stdc++.h>

// * * * * * * * * * * * * Public Member Functions * * * * * * * * * * * * * //


template<class Type>
Type* Foam::cpuCsrAddressingExecutor::alloc
(
    Foam::label size
) const
{
    Type* ptr = new Type[size];
	return ptr;
}

template<class Type>
const Type* Foam::cpuCsrAddressingExecutor::copyFromFoam
(
    Foam::label size,
	const Type* hostPtr
) const
{
    const Type* ptr = hostPtr;
	return ptr;
}

template<class Type>
void Foam::cpuCsrAddressingExecutor::clear(Type* ptr) const
{
    delete ptr;
}

void Foam::cpuCsrAddressingExecutor::initializeAddressing
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
    for (label i = 0; i < totNnz; ++i)
    {
        tmpPerm[i] = i;
    }

    // Initialize: rowIndecesTmp = [0, ... totNnz-1, (owner), (neighbour)]
    //              colIndecesTmp = [0, ... totNnz-1, (neighbour), (owner)]
    //              valuesTmp = [(diag), (upper), (lower)]

    for(label i=0; i<nCells; ++i)
    {
        rowIndTmp[i] = i;
        colIndTmp[i] = i;
    }

    for(label i=0; i<nInternalFaces; ++i)
    {
        rowIndTmp[nCells + i] = owner[i];
        colIndTmp[nCells + i] = neighbour[i];

        rowIndTmp[nCells + nInternalFaces + i] = neighbour[i];
        colIndTmp[nCells + nInternalFaces + i] = owner[i];
    }

    return;
}

void Foam::cpuCsrAddressingExecutor::initializeAddressingExt
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

    for(int i=0; i<nnzExt; ++i)
    {
        rowIndTmp[nCells + 2*nInternalFaces + i] = extRows[i];
        colIndTmp[nCells + 2*nInternalFaces + i] = extCols[i];
    }

    return;
}

void Foam::cpuCsrAddressingExecutor::computeSorting
(
    const label   totNnz,
    const label * const tmpPerm,
    const label * const rowIndTmp,
          label * rowInd,
          label * ldu2csr
) const
{
    std::pair<label, label> *pairTmp = new std::pair<label, label>[totNnz];
    for(label i=0; i<totNnz; ++i)
    {
        pairTmp[i].first = rowIndTmp[i];
        pairTmp[i].second = tmpPerm[i];
    }
    std::vector< std::pair<label,label>> pairVect(pairTmp, pairTmp+totNnz);

    // Find the permutation vector
    std::sort(pairVect.begin(), pairVect.end());

    for(label i=0; i<totNnz; ++i)
    {
        rowInd[i] = pairVect[i].first;
        ldu2csr[i] = pairVect[i].second;
    }
}


void Foam::cpuCsrAddressingExecutor::localToGlobalColIndices
(
    const label nRows,
    const label nIntFaces,
    const label diagIndexGlobal,
    const label lowOffGlobal,
    const label uppOffGlobal,
    label *colIndicesGlobal
) const
{
    for(label i=0; i<nRows; ++i)
    {
        colIndicesGlobal[i] += diagIndexGlobal;
    }

    for(label i=0; i<nIntFaces; ++i)
    {
        colIndicesGlobal[nRows + i] += uppOffGlobal;
        colIndicesGlobal[nRows + nIntFaces + i] += lowOffGlobal;
    }
}


void Foam::cpuCsrAddressingExecutor::applyAddressingPermutation
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
    label curRow = 0;

    ownStart[0] = 0;
    for(label i=0; i<totNnz; ++i)
    {
        colInd[i] = colIndTmp[ldu2csr[i]];
        if(curRow < rowInd[i])
        {
            ownStart[rowInd[i]] = i;
        }
        curRow = rowInd[i];
    }

    ownStart[nCells] = totNnz;
}

// * * * * * * * * * * * * * Explicit instantiations  * * * * * * * * * * * //

#define makecpuCsrAddressingExecutor(Type)                                    \
    template Type* Foam::cpuCsrAddressingExecutor::alloc<Type>                \
    (                                                                         \
        Foam::label size                                                      \
    ) const;                                                                  \
    template const Type* Foam::cpuCsrAddressingExecutor::copyFromFoam<Type>   \
    (                                                                         \
        Foam::label size,                                                     \
        const Type* hostPtr                                                   \
    ) const;                                                                  \
    template void  Foam::cpuCsrAddressingExecutor::clear<Type>                \
    (                                                                         \
        Type* ptr                                                             \
    ) const;

makecpuCsrAddressingExecutor(Foam::label)
makecpuCsrAddressingExecutor(Foam::scalar)
// ************************************************************************* //

