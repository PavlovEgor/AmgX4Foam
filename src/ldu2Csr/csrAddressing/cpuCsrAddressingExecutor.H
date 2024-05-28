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
    cpulduMatrixExecutor

Description
    cpu visitor of the devicelduMatrix class

SourceFiles
    cpulduMatrixExecutor.H
    cpulduMatrixExecutor.C
*/

#ifndef cpu_CsrAddressing_Executor_H
#define cpu_CsrAddressing_Executor_H

#include "label.H"

namespace Foam
{

// Forward Declarations
class csrAddressing;

class cpuCsrAddressingExecutor
{
public:

    cpuCsrAddressingExecutor() {};

    ~cpuCsrAddressingExecutor() {};

    template<class Type>
    Type* alloc(Foam::label size) const;

    template<class Type>
    const Type* copyFromFoam(Foam::label size, const Type* hostPtr) const;

    template<class Type>
    void clear(Type* ptr) const;

    template<class Type>
    void clear(const Type* ptr) const;

    void initializeAddressing
    (
        const label   nCells,
        const label   nInternalFaces,
        const label   totNnz,
        const label * const owner,
        const label * const neighbour,
              label * tmpPerm,
              label * rowIndTmp,
              label * colIndTmp
    ) const;

    void initializeAddressingExt
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
    ) const;

    void computeSorting
    (
        const label   totNnz,
              label * tmpPerm,
              label * rowIndTmp,
              label * rowInd,
              label * ldu2csr
    ) const;

    void localToGlobalColIndices
    (
        const label   nRows,
        const label   nIntFaces,
        const label   diagIndexGlobal,
        const label   lowOffGlobal,
        const label   uppOffGlobal,
              label * colIndicesGlobal
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

};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //

