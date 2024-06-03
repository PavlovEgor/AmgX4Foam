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

#include "cpuCsrMatrixExecutor.H"
#include <cmath>
#include <bits/stdc++.h>

// * * * * * * * * * * * * * * * * Member functions * * * * * * * * * * * * * //

template<class Type>
Type* Foam::cpuCsrMatrixExecutor::alloc
(
    Foam::label size
) const
{
    Type* ptr = new Type[size];
	return ptr;
}

template<class Type>
Type* Foam::cpuCsrMatrixExecutor::alloc
(
    Foam::label size,
    Type value
) const
{
    Type* ptr = new Type[size];
    for(Foam::label i=0; i < size; ++i)  ptr[i] = value;
	return ptr;
}

template<class Type>
const Type* Foam::cpuCsrMatrixExecutor::copyFromFoam
(
    Foam::label size,
	const Type* hostPtr
) const
{
    const Type* ptr = hostPtr;
	return ptr;
}

template<class Type>
void Foam::cpuCsrMatrixExecutor::clear(Type* ptr) const
{
    delete ptr;
}

template<class Type>
void Foam::cpuCsrMatrixExecutor::clear(const Type* ptr) const
{
}

void Foam::cpuCsrMatrixExecutor::initializeSequence
(
    const label len,
          label * vect
) const
{
    // Initialize vect = [0, 1, ... len-1]
    for(label i = 0; i < len; ++i) vect[i] = i;
}

void Foam::cpuCsrMatrixExecutor::initializeAddressing
(
    const label   nConsRows,
    const label   nConsIntFaces,
    const label   nRows,
    const label   nInternalFaces,
    const label * const owner,
    const label * const neighbour,
          label * rowIndTmp,
          label * colIndTmp,
    const label   rowsDispl, //default =0
    const label   intFacesDispl //default =0
) const
{
    // Initialize: rowIndecesTmp = [0, ... nConsRows, (owner), (neighbour)]
    //             colIndecesTmp = [0, ... nRows1, .. 0 ... nRowsN, (neighbour), (owner)]
    for(label i=0; i<nRows; ++i) colIndTmp[rowsDispl + i] = i;

    for(label i=0; i<nInternalFaces; ++i)
    {
        rowIndTmp[nConsRows + intFacesDispl + i] = owner[i];
        colIndTmp[nConsRows + intFacesDispl + i] = neighbour[i];

        rowIndTmp[nConsRows + nConsIntFaces + intFacesDispl + i] = neighbour[i];
        colIndTmp[nConsRows + nConsIntFaces + intFacesDispl + i] = owner[i];
    }

    return;
}

void Foam::cpuCsrMatrixExecutor::initializeAddressingExt
(
    const label   nConsRows,
    const label   nConsIntFaces,
    const label   nRows,
    const label   nInternalFaces,
    const label   nnzExt,
    const label * const owner,
    const label * const neighbour,
    const label * const extRows,
    const label * const extCols,
          label * rowIndTmp,
          label * colIndTmp,
    const label   rowsDispl, // default = 0
    const label   intFacesDispl, // default = 0
    const label   extNnzDispl // default = 0
) const
{
    this->initializeAddressing
    (
        nConsRows,
        nConsIntFaces,
        nRows,
        nInternalFaces,
        owner,
        neighbour,
        rowIndTmp,
        colIndTmp,
        rowsDispl,
        intFacesDispl
    );

    for(int i=0; i<nnzExt; ++i)
    {
        rowIndTmp[nConsRows + 2*nConsIntFaces + extNnzDispl + i] = extRows[i];
        colIndTmp[nConsRows + 2*nConsIntFaces + extNnzDispl + i] = extCols[i];
    }

    return;
}

void Foam::cpuCsrMatrixExecutor::computeSorting
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


void Foam::cpuCsrMatrixExecutor::localToGlobalColIndices
(
    const label   nConsRows,
    const label   nConsIntFaces,
    const label   nRows,
    const label   nIntFaces,
    const label   diagIndexGlobal,
    const label   lowOffGlobal,
    const label   uppOffGlobal,
          label * colIndicesGlobal,
    const label   rowDispl, // default = 0
    const label   intFacesDispl // default = 0
) const
{
    for(label i=0; i<nRows; ++i)
    {
        colIndicesGlobal[rowDispl + i] += diagIndexGlobal;
    }

    for(label i=0; i<nIntFaces; ++i)
    {
        colIndicesGlobal[nConsRows + intFacesDispl + i] += uppOffGlobal;
        colIndicesGlobal[nConsRows + nConsIntFaces + intFacesDispl + i] += lowOffGlobal;
    }
}


void Foam::cpuCsrMatrixExecutor::localToConsRowIndex
(
    const label nConsRows,
    const label nConsIntFaces,
    const label nIntFaces,
    const label nExtNz,
    const label intFacesDipl,
    const label extDispl,
    const label offset,
          label * rowIndices
) const
{
    for(label i=0; i<nIntFaces; ++i)
    {
        rowIndices[nConsRows + intFacesDipl + i] += offset;
        rowIndices[nConsRows + nConsIntFaces + intFacesDipl + i] += offset;
    }

    for(label i=0; i<nExtNz; ++i)
    {
        rowIndices[nConsRows + 2 * nConsIntFaces + extDispl + i] += offset;
    }
}


void Foam::cpuCsrMatrixExecutor::applyAddressingPermutation
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


void Foam::cpuCsrMatrixExecutor::initializeValue
(
    const label   nConsRows,
    const label   nConsIntFaces,
    const label   nRows,
    const label   nIntFaces,
    const double * const diag,
    const double * const upper,
    const double * const lower,
          double * valuesTmp,
    const label   rowsDisp, // default = 0
    const label   intFacesDisp // default = 0
) const
{
    for(label i=0; i<nRows; ++i)
    {
        valuesTmp[rowsDisp + i] = diag[i];
    }

    for(label i=0; i<nIntFaces; ++i)
    {
        valuesTmp[nConsRows + intFacesDisp + i] = upper[i];
        valuesTmp[nConsRows + nConsIntFaces + intFacesDisp + i] = lower[i];
    }
}


void Foam::cpuCsrMatrixExecutor::initializeValueExt
(
    const label nConsRows,
    const label nConsIntFaces,
    const label nCells,
    const label nIntFaces,
    const label nnzExt,
    const double * const diag,
    const double * const upper,
    const double * const lower,
    const double * const extValue,
          double * valuesTmp,
    const label rowsDisp, // default = 0
    const label intFacesDisp, // default = 0
    const label extValDisp // default = 0
) const
{
    // Initialize valuesTmp = [(diag), (upper), (lower), (extValues)]

    initializeValue
    (
        nConsRows,
        nConsIntFaces,
        nCells,
        nIntFaces,
        diag,
        upper,
        lower,
        valuesTmp,
        rowsDisp,
        intFacesDisp
    );

    for(label i=0; i<nnzExt; ++i)
    {
        valuesTmp[nConsRows + 2*nConsIntFaces + extValDisp + i] = extValue[i];
    }
}

void Foam::cpuCsrMatrixExecutor::applyValuePermutation
(
    const label    totNnz,
    const label *  const ldu2csr,
    const scalar * const valuesTmp,
          scalar * values,
    const label    nBlocks
) const
{
    label blockLen = nBlocks * nBlocks;
    
    for(label i=0; i<totNnz; ++i)
    {
        for(label j=0; j<blockLen; ++j)
        {
            values[i*blockLen + j] = valuesTmp[ldu2csr[i]*blockLen + j];
        }
    }
}

// * * * * * * * * * * * *  Public Member Functions * * * * * * * * * * * *  //

// * * * * * * * * * * * * * Explicit instantiations  * * * * * * * * * * *  //

#define makecpuCsrMatrixExecutor(Type)                                    \
    template Type* Foam::cpuCsrMatrixExecutor::alloc<Type>                \
    (                                                                     \
        Foam::label size                                                  \
    ) const;                                                              \
    template Type* Foam::cpuCsrMatrixExecutor::alloc<Type>                \
    (                                                                     \
        Foam::label size,                                                 \
        Type value                                                        \
    ) const;                                                              \
    template const Type* Foam::cpuCsrMatrixExecutor::copyFromFoam<Type>   \
    (                                                                     \
        Foam::label size,                                                 \
        const Type* hostPtr                                               \
    ) const;                                                              \
    template void  Foam::cpuCsrMatrixExecutor::clear<Type>                \
    (                                                                     \
        Type* ptr                                                         \
    ) const;                                                              \
    template void  Foam::cpuCsrMatrixExecutor::clear<Type>                \
    (                                                                     \
        const Type* ptr                                                   \
    ) const;

makecpuCsrMatrixExecutor(Foam::label)
makecpuCsrMatrixExecutor(Foam::scalar)

// ************************************************************************* //
