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

#include "csrAdressing.C"
#include <cmath>
#include <bits/stdc++.h>

// * * * * * * * * * * * * * * * * CPU Kernels  * * * * * * * * * * * * * * //

inline void Foam::csrAdressing::initializeAddressing
(
    const int   nCells,
    const int   nInternalFaces,
    const int   totNnz,
    const int * const owner,
    const int * const neighbour,
          int * tmpPerm,
          int * rowIndTmp,
          int * colIndTmp
)
{
    // Initialize tmpPerm = [0, 1, ... totNnz-1]
    for (int i = 0; i < totNnz; ++i)
    {
        tmpPerm[i] = i;
    }

    // Initialize: rowIndecesTmp = [0, ... totNnz-1, (owner), (neighbour)]
    //              colIndecesTmp = [0, ... totNnz-1, (neighbour), (owner)]
    //              valuesTmp = [(diag), (upper), (lower)]

    for(int i=0; i<nCells; ++i)
    {
        rowIndTmp[i] = i;
        colIndTmp[i] = i;
    }

    for(int i=0; i<nInternalFaces; ++i)
    {
        rowIndTmp[nCells + i] = owner[i];
        colIndTmp[nCells + i] = neighbour[i];

        rowIndTmp[nCells + nInternalFaces + i] = neighbour[i];
        colIndTmp[nCells + nInternalFaces + i] = owner[i];
    }

    return;
}


inline void Foam::csrAdressing::initializeAddressingExt
(
    const int   nCells,
    const int   nInternalFaces,
    const int   nnzExt,
    const int   totNnz,
    const int * const owner,
    const int * const neighbour,
    const int * const extRows,
    const int * const extCols,
          int * tmpPerm,
          int * rowIndTmp,
          int * colIndTmp
)
{
    initializeAddressing
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

inline void Foam::csrAdressing::computeSorting
(
    const int   totNnz,
    const int * const tmpPerm,
    const int * const rowIndTmp,
          int * rowInd,
          int * ldu2csr
)
{
    std::pair<int, int> *pairTmp = new std::pair<int, int>[totNnz];
    for(int i=0; i<totNnz; ++i)
    {
        pairTmp[i].first = rowIndTmp[i];
        pairTmp[i].second = tmpPerm[i];
    }
    std::vector< std::pair<int,int>> pairVect(pairTmp, pairTmp+totNnz);

    // Find the permutation vector
    std::sort(pairVect.begin(), pairVect.end());

    for(int i=0; i<totNnz; ++i)
    {
        rowInd[i] = pairVect[i].first;
        ldu2csr[i] = pairVect[i].second;
    }
}


inline void Foam::csrAdressing::localToGlobalColIndices
(
    const int nRows,
    const int nIntFaces,
    const int diagIndexGlobal,
    const int lowOffGlobal,
    const int uppOffGlobal,
    int *colIndicesGlobal
)
{
    for(int i=0; i<nRows; ++i)
    {
        colIndicesGlobal[i] += diagIndexGlobal;
    }

    for(int i=0; i<nIntFaces; ++i)
    {
        colIndicesGlobal[nRows + i] += uppOffGlobal;
        colIndicesGlobal[nRows + nIntFaces + i] += lowOffGlobal;
    }
}


inline void Foam::csrAdressing::applyAddressingPermutation
(
    const int   nCells, //NOTE: it is not used but is need for the cuda kernel
    const int   totNnz,
    const int * const ldu2csr,
    const int * const colIndTmp,
    const int * const rowInd,
          int * colInd,
          int * ownStart
)
{
    int curRow = 0;

    for(int i=0; i<totNnz; ++i)
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

// * * * * * * * * * * * *  Public Member Functions * * * * * * * * * * * *  //

// * * * * * * * * * * * * * Explicit instantiations  * * * * * * * * * * *  //

// ************************************************************************* //
