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
    const int   nConsRows,
    const int   nConsIntFaces,
    const int   nRows,
    const int   nInternalFaces,
    const int * const owner,
    const int * const neighbour,
          int * tmpPerm,
          int * rowIndTmp,
          int * colIndTmp,
    const int   rowDispl, //default =0
    const int   intFacesDispl // default = 0
)
{
    // Initialize: rowIndecesTmp = [0, ... totNnz-1, (owner), (neighbour)]
    //             colIndecesTmp = [0, ... totNnz-1, (neighbour), (owner)]
    for(int i=0; i<nRows; ++i)
    {
        colIndTmp[rowDispl + i] = i;
    }

    for(int i=0; i<nConsRows; ++i) rowIndTmp[i] = i;

    for(int i=0; i<nInternalFaces; ++i)
    {
        rowIndTmp[nConsRows + intFacesDispl + i] = owner[i];
        colIndTmp[nConsRows + intFacesDispl + i] = neighbour[i];

        rowIndTmp[nConsRows + nConsIntFaces + intFacesDispl + i] = neighbour[i];
        colIndTmp[nConsRows + nConsIntFaces + intFacesDispl + i] = owner[i];
    }

    return;
}


inline void Foam::csrAdressing::initializeAddressingExt
(
    const int   nConsRows,
    const int   nConsintFaces,
    const int   nRows,
    const int   nInternalFaces,
    const int   nnzExt,
    const int * const owner,
    const int * const neighbour,
    const int * const extRows,
    const int * const extCols,
          int * tmpPerm,
          int * rowIndTmp,
          int * colIndTmp,
    const int   rowsDispl, // default = 0
    const int   intFacesDispl, // default = 0
    const int   extNnzDispl // default = 0
)
{
    initializeAddressing
    (
        nConsRows,
        nConsintFaces,
        nRows,
        nInternalFaces,
        owner,
        neighbour,
        tmpPerm,
        rowIndTmp,
        colIndTmp,
        rowsDispl,
        intFacesDispl
    );

    for(int i=0; i<nnzExt; ++i)
    {
        rowIndTmp[nConsRows + 2*nConsintFaces + extNnzDispl + i] = extRows[i];
        colIndTmp[nConsRows + 2*nConsintFaces + extNnzDispl + i] = extCols[i];
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
    const int nConsRows,
    const int nConsIntFaces,
    const int nRows,
    const int nIntFaces,
    const int diagIndexGlobal,
    const int lowOffGlobal,
    const int uppOffGlobal,
    int *colIndicesGlobal,
    const int rowDispl, //default = 0
    const int intFacesDispl //default = 0
)
{   
    for(int i=0; i<nRows; ++i)
    {
        colIndicesGlobal[rowDispl + i] += diagIndexGlobal;
    }

    for(int i=0; i<nIntFaces; ++i)
    {
        colIndicesGlobal[nConsRows + intFacesDispl + i] += uppOffGlobal;
        colIndicesGlobal[nConsRows + nConsIntFaces + intFacesDispl + i] += lowOffGlobal;
    }
}

inline void Foam::csrAdressing::localToConsRowIndex
(
    const int nConsRows,
    const int nConsIntFaces,
    const int nIntFaces,
    const int nExtNz,
    const int intFacesDipl,
    const int extDispl,
    const int offset,
          int * rowIndices
)
{
    for(int i=0; i<nIntFaces; ++i)
    {
        rowIndices[nConsRows + intFacesDipl + i] += offset;
        rowIndices[nConsRows + nConsIntFaces + intFacesDipl + i] += offset;
    }

    for(int i=0; i<nExtNz; ++i)
    {
        rowIndices[nConsRows + 2 * nConsIntFaces + extDispl + i] += offset;
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
