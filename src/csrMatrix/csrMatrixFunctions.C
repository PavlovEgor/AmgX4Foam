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

// * * * * * * * * * * * * * * * * * Kernels * * * * * * * * * * * * * * * * //

inline void Foam::csrMatrix::initializeSequence
(
    const label len,
          label * vect
)const 
{
    std::visit
	([
        len,
        vect
	 ]
	 (const auto& exec){ exec.initializeSequence
                            (
                                len,
                                vect
					        );
                       },
     csrMatExec_);
}


inline void Foam::csrMatrix::initializeAddressing
(
    const label   nConsRows,
    const label   nConsIntFaces,
    const label * const owner,
    const label * const neighbour,
          label * rowIndTmp,
          label * colIndTmp
)
{
    std::visit
	([
        nConsRows,
        nConsIntFaces,
        &owner,
        &neighbour,
        &rowIndTmp,
        &colIndTmp
	 ]
	 (const auto& exec){ exec.initializeAddressing
                            (
                                nConsRows,
                                nConsIntFaces,
                                owner,
                                neighbour,
                                rowIndTmp,
                                colIndTmp
					        );
                       },
     csrMatExec_);
}


inline void Foam::csrMatrix::initializeAddressingExt
(
    const label   nConsRows,
    const label   nConsIntFaces,
    const label   nConsExtNz,
    const label * const owner,
    const label * const neighbour,
    const label * const extRows,
    const label * const extCols,
          label * rowIndTmp,
          label * colIndTmp
)
{
    std::visit
	([
        nConsRows,
        nConsIntFaces,
        nConsExtNz,
        &owner,
        &neighbour,
        &extRows,
        &extCols,
        &rowIndTmp,
        &colIndTmp
	 ]
	 (const auto& exec){ exec.initializeAddressingExt
                           (
                                nConsRows,
                                nConsIntFaces,
                                nConsExtNz,
                                owner,
                                neighbour,
                                extRows,
                                extCols,
                                rowIndTmp,
                                colIndTmp
					       );
                       },
     csrMatExec_);
}

inline void Foam::csrMatrix::computeSorting
(
    const label   totNnz,
    const label * const tmpPerm,
    const label * const rowIndTmp,
          label * rowInd,
          label * ldu2csr
)
{
    std::visit
	([
        totNnz,
        &tmpPerm,
        &rowIndTmp,
        &rowInd,
        &ldu2csr
	 ]
	 (const auto& exec){ exec.computeSorting
                           (
                               totNnz,
                               tmpPerm,
                               rowIndTmp,
                               rowInd,
                               ldu2csr
					       );
                       },
     csrMatExec_);
}


inline void Foam::csrMatrix::localToGlobalColIndices
(
    const label   nConsRows,
    const label   nConsIntFaces,
    const label   nRows,
    const label   nIntFaces,
    const label   diagIndexGlobal,
    const label   lowOffGlobal,
    const label   uppOffGlobal,
          label * colIndicesGlobal,
    const label   rowsDispl,
    const label   intFacesDispl
)
{
    std::visit
	([
        nConsRows,
        nConsIntFaces,
        nRows,
        nIntFaces,
        diagIndexGlobal,
        lowOffGlobal,
        uppOffGlobal,
        &colIndicesGlobal,
        rowsDispl,
        intFacesDispl
	 ]
	 (const auto& exec){ exec.localToGlobalColIndices
                            (
                                nConsRows,
                                nConsIntFaces,
                                nRows,
                                nIntFaces,
                                diagIndexGlobal,
                                lowOffGlobal,
                                uppOffGlobal,
                                colIndicesGlobal,
                                rowsDispl,
                                intFacesDispl
					        );
                       },
     csrMatExec_);
}


inline void Foam::csrMatrix::localToConsRowIndex
(
    const label nConsRows,
    const label nConsIntFaces,
    const label nIntFaces,
    const label nExtNz,
    const label intFacesDipl,
    const label extDispl,
    const label offset,
          label * rowIndices
)
{
    std::visit
	([
        nConsRows,
        nConsIntFaces,
        nIntFaces,
        nExtNz,
        intFacesDipl,
        extDispl,
        offset,
        &rowIndices
	 ]
	 (const auto& exec){ exec.localToConsRowIndex
                            (
                                nConsRows,
                                nConsIntFaces,
                                nIntFaces,
                                nExtNz,
                                intFacesDipl,
                                extDispl,
                                offset,
                                rowIndices
					        );
                       },
     csrMatExec_);
}


inline void Foam::csrMatrix::applyAddressingPermutation
(
    const label   nCells, //NOTE: it is not used but is need for the cuda kernel
    const label   totNnz,
    const label * const ldu2csr,
    const label * const colIndTmp,
    const label * const rowInd,
          label * colInd,
          label * ownStart
)
{
    std::visit
	([
        nCells, //NOTE: it is not used but is need for the cuda kernel
        totNnz,
        &ldu2csr,
        &colIndTmp,
        &rowInd,
        &colInd,
        &ownStart
	 ]
	 (const auto& exec){ exec.applyAddressingPermutation
                           (
                               nCells, //NOTE: it is not used but is need for the cuda kernel
                               totNnz,
                               ldu2csr,
                               colIndTmp,
                               rowInd,
                               colInd,
                               ownStart
					       );
                       },
     csrMatExec_);
}

inline void Foam::csrMatrix::initializeValue
(
    const label   nConsRows,
    const label   nConsIntFaces,
    const double * const diag,
    const double * const upper,
    const double * const lower,
          double * valuesTmp                
)
{
    std::visit
	([
        nConsRows,
        nConsIntFaces,
        &diag,
        &upper,
        &lower,
        &valuesTmp
	 ]
	 (const auto& exec){ exec.initializeValue
                           (
                                nConsRows,
                                nConsIntFaces,
                                diag,
                                upper,
                                lower,
                                valuesTmp
					       );
                       },
     csrMatExec_);
}


inline void Foam::csrMatrix::initializeValueExt
(
    const label nConsRows,
    const label nConsIntFaces,
    const label nConsExtNz,
    const double * const diag,
    const double * const upper,
    const double * const lower,
    const double * const extValue,
          double * valuesTmp
)
{
    std::visit
	([
        nConsRows,
        nConsIntFaces,
        nConsExtNz,
        &diag,
        &upper,
        &lower,
        &extValue,
        &valuesTmp
	 ]
	 (const auto& exec){ exec.initializeValueExt
                           (
                                nConsRows,
                                nConsIntFaces,
                                nConsExtNz,
                                diag,
                                upper,
                                lower,
                                extValue,
                                valuesTmp
					       );
                       },
     csrMatExec_);
}


inline void Foam::csrMatrix::applyValuePermutation
(
    const label totNnz,
    const label  * const ldu2csr,
    const scalar * const valuesTmp,
          scalar * values,
    const label  nBlocks
)
{
    std::visit
	([
        totNnz,
        &ldu2csr,
        &valuesTmp,
        &values,
        nBlocks
	 ]
	 (const auto& exec){ exec.applyValuePermutation
                            (
                                totNnz,
                                ldu2csr,
                                valuesTmp,
                                values,
                                nBlocks
					        );
                       },
     csrMatExec_);
}

// * * * * * * * * * * * *  Public Member Functions * * * * * * * * * * * *  //

// * * * * * * * * * * * * * Explicit instantiations  * * * * * * * * * * *  //

// ************************************************************************* //
