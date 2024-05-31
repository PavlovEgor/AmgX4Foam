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

inline void Foam::csrMatrix::initializeAddressing
(
    const label   nCells,
    const label   nInternalFaces,
    const label   totNnz,
    const label * const owner,
    const label * const neighbour,
          label * tmpPerm,
          label * rowIndTmp,
          label * colIndTmp
)
{
    std::visit
	([
        nCells,
        nInternalFaces,
        totNnz,
        &owner,
        &neighbour,
        &tmpPerm,
        &rowIndTmp,
        &colIndTmp
	 ]
	 (const auto& exec){ exec.initializeAddressing
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
                       },
     csrMatExec_);
}


inline void Foam::csrMatrix::initializeAddressingExt
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
)
{
    std::visit
	([
        nCells,
        nInternalFaces,
        nnzExt,
        totNnz,
        &owner,
        &neighbour,
        &extRows,
        &extCols,
        &tmpPerm,
        &rowIndTmp,
        &colIndTmp
	 ]
	 (const auto& exec){ exec.initializeAddressingExt
                           (
                               nCells,
                               nInternalFaces,
                               nnzExt,
                               totNnz,
                               owner,
                               neighbour,
                               extRows,
                               extCols,
                               tmpPerm,
                               rowIndTmp,
                               colIndTmp
					       );
                       },
     csrMatExec_);
}

inline void Foam::csrMatrix::computeSorting
(
    const label   totNnz,
          label * tmpPerm,
          label * rowIndTmp,
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
    const label nRows,
    const label nIntFaces,
    const label diagIndexGlobal,
    const label lowOffGlobal,
    const label uppOffGlobal,
    label *colIndicesGlobal
)
{
    std::visit
	([
        nRows,
        nIntFaces,
        diagIndexGlobal,
        lowOffGlobal,
        uppOffGlobal,
        &colIndicesGlobal
	 ]
	 (const auto& exec){ exec.localToGlobalColIndices
                           (
                               nRows,
                               nIntFaces,
                               diagIndexGlobal,
                               lowOffGlobal,
                               uppOffGlobal,
                               colIndicesGlobal
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
    const label   nCells,
    const label   nIntFaces,
    const scalar * const diag,
    const scalar * const upper,
    const scalar * const lower,
            scalar * valuesTmp                  
)
{
    std::visit
	([
        nCells,
        nIntFaces,
        &diag,
        &upper,
        &lower,
        &valuesTmp 
	 ]
	 (const auto& exec){ exec.initializeValue
                           (
                                nCells,
                                nIntFaces,
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
    const label nCells,
    const label nIntFaces,
    const label nnzExt,
    const scalar * const diag,
    const scalar * const upper,
    const scalar * const lower,
    const scalar * const extValue,
          scalar * valuesTmp
)
{
    std::visit
	([
        nCells,
        nIntFaces,
        nnzExt,
        &diag,
        &upper,
        &lower,
        &extValue,
        &valuesTmp 
	 ]
	 (const auto& exec){ exec.initializeValueExt
                           (
                                nCells,
                                nIntFaces,
                                nnzExt,
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
          scalar * values
)
{
    std::visit
	([
        totNnz,
        &ldu2csr,
        &valuesTmp,
        &values
	 ]
	 (const auto& exec){ exec.applyValuePermutation
                            (
                                totNnz,
                                ldu2csr,
                                valuesTmp,
                                values
					        );
                       },
     csrMatExec_);
}

// * * * * * * * * * * * *  Public Member Functions * * * * * * * * * * * *  //

// * * * * * * * * * * * * * Explicit instantiations  * * * * * * * * * * *  //

// ************************************************************************* //
