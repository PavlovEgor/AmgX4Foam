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


Class
    cpuCsrMatrixExecutor

Description
    cpu visitor of the csrMatrix class

SourceFiles
    cpuCsrMatrixExecutor.H
    cpuCsrMatrixExecutor.C

\*---------------------------------------------------------------------------*/


#ifndef cpu_csrMatrix_Executor_H
#define cpu_csrMatrix_Executor_H

#include "scalar.H"
#include "label.H"
#include "List.H"

namespace Foam
{

// Forward Declarations
class csrMatrix;

class cpuCsrMatrixExecutor
{
public:

    cpuCsrMatrixExecutor() {};

    ~cpuCsrMatrixExecutor() {};

    template<class Type>
    Type* alloc(Foam::label size) const;

    template<class Type>
    Type* allocZero(Foam::label size) const;

    template<class Type>
    Type* copyFromFoam(Foam::label size, const Type* hostPtr) const;

    template<class Type>
    void clear(Type* ptr) const;

    template<class Type>
    void clear(const Type* ptr) const;

    template<class Type>
    void concatenate(label globSize, List<List<Type>> lst, Type * ptr) const;

    void initializeSequence
    (
    const label len,
          label * vect
    ) const;

    void initializeAddressing
    (
        const label   nConsRows,
        const label   nConsIntFaces,
        const label * const owner,
        const label * const neighbour,
              label * rowIndTmp,
              label * colIndTmp
    ) const;

    void initializeAddressingExt
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
    ) const;

    void computeSorting
    (
        const label   totNnz,
        const label * const tmpPerm,
        const label * const rowIndTmp,
              label * rowInd,
              label * ldu2csr
    ) const;

    void localToGlobalColIndices
    (
        const label   nConsRows,
        const label   nConsIntFaces,
        const label   nRows,
        const label   nIntFaces,
        const label   diagIndexGlobal,
        const label   lowOffGlobal,
        const label   uppOffGlobal,
              label * colIndicesGlobal,
        const label   rowDispl = 0,
        const label   intFacesDispl = 0
    ) const;

    void localToConsRowIndex
    (
        const label nConsRows,
        const label nConsIntFaces,
        const label nIntFaces,
        const label nExtNz,
        const label intFacesDipl,
        const label extDispl,
        const label offset,
                label * rowIndices
    ) const;

    void applyAddressingPermutation
    (
        const label   nCells,
        const label   totNnz,
        const label * const ldu2csr,
        const label * const colIndTmp,
        const label * const rowInd,
              label * colInd,
              label * ownStart
    ) const;

    void initializeValue
    (
        const label   nConsRows,
        const label   nConsIntFaces,
        const double * const diag,
        const double * const upper,
        const double * const lower,
              double * valuesTmp                  
    ) const;

    void initializeValueExt
    (
        const label nConsRows,
        const label nConsIntFaces,
        const label nConsExtNz,
        const double * const diag,
        const double * const upper,
        const double * const lower,
        const double * const extValue,
              double * valuesTmp
    ) const;

    void applyValuePermutation
    (
        const label    totNnz,
        const label  * const ldu2csr,
        const scalar * const valuesTmp,
              scalar * values,
        const label    nBlocks = 1
    ) const;
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //

