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

#include "csrMatrix.C"
#include <cmath>
#include <bits/stdc++.h>

// * * * * * * * * * * * * * * * * CPU Kernels  * * * * * * * * * * * * * * //

inline void Foam::csrMatrix::initializeValue
(
    const int   nConsRows,
    const int   nConsIntFaces,
    const int   nRows,
    const int   nIntFaces,
    const double * const diag,
    const double * const upper,
    const double * const lower,
          double * valuesTmp,
    const int   rowsDisp, // default = 0
    const int   intFacesDisp // default = 0
)
{
    for(int i=0; i<nRows; ++i)
    {
        valuesTmp[rowsDisp + i] = diag[i];
    }

    for(int i=0; i<nIntFaces; ++i)
    {
        valuesTmp[nConsRows + intFacesDisp + i] = upper[i];
        valuesTmp[nConsRows + nConsIntFaces + intFacesDisp + i] = lower[i];
    }
}


inline void Foam::csrMatrix::initializeValueExt
(
    const int nConsRows,
    const int nConsIntFaces,
    const int nCells,
    const int nIntFaces,
    const int nnzExt,
    const double * const diag,
    const double * const upper,
    const double * const lower,
    const double * const extValue,
          double * valuesTmp,
    const int rowsDisp, // default = 0
    const int intFacesDisp, // default = 0
    const int extValDisp // defaul = 0
)
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

    for(int i=0; i<nnzExt; ++i)
    {
        valuesTmp[nConsRows + 2*nConsIntFaces + extValDisp + i] = extValue[i];
    }
}

inline void Foam::csrMatrix::applyValuePermutation
(
    const int   totNnz,
    const int * const ldu2csr,
    const double * const valuesTmp,
          double * values
)
{
    for(int i=0; i<totNnz; ++i)
    {
        values[i] = valuesTmp[ldu2csr[i]];
    }
}

// * * * * * * * * * * * *  Public Member Functions * * * * * * * * * * * *  //

// * * * * * * * * * * * * * Explicit instantiations  * * * * * * * * * * *  //

// ************************************************************************* //
