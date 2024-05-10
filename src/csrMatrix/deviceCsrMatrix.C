/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2019-2021 OpenCFD Ltd.
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

#include "deviceCsrMatrix.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

/* namespace Foam
{
    defineTypeNameAndDebug(deviceCsrMatrix, 1);
} */


/* // const Foam::scalar Foam::deviceCsrMatrix::defaultTolerance = 1e-6;
const Foam::Enum
<
    Foam::deviceCsrMatrix::normTypes
>
Foam::deviceCsrMatrixnormTypesNames_
({
    { normTypes::NO_NORM, "none" },
    { normTypes::DEFAULT_NORM, "default" },
    { normTypes::L1_SCALED_NORM, "L1_scaled" },
}); */


// * * * * * * * * * * * * * * * Constructors * * * * * * * * * * * * * * * //

Foam::deviceCsrMatrix::deviceCsrMatrix()
:
    valuesPtr_(nullptr),
    ownerStartPtr_(nullptr),
    colIndicesPtr_(nullptr),
    ldu2csrPerm_(nullptr)
{}

Foam::deviceCsrMatrix::deviceCsrMatrix(const deviceCsrMatrix& A)
:
    valuesPtr_(nullptr),
    ownerStartPtr_(nullptr),
    colIndicesPtr_(nullptr),
    ldu2csrPerm_(nullptr)
{
    if (A.valuesPtr_)
    {
        valuesPtr_ = new deviceField<Foam::scalar>(*(A.valuesPtr_));
    }

    if (A.ownerStartPtr_)
    {
        ownerStartPtr_ = new deviceField<Foam::label>(*(A.ownerStartPtr_));
    }

    if (A.colIndicesPtr_)
    {
        colIndicesPtr_ = new deviceField<Foam::label>(*(A.colIndicesPtr_));
    }

    if (A.ldu2csrPerm_)
    {
        ldu2csrPerm_ = new deviceField<Foam::label>(*(A.ldu2csrPerm_));
    }
}


Foam::deviceCsrMatrix::deviceCsrMatrix(deviceCsrMatrix& A, bool reuse)
:
    valuesPtr_(nullptr),
    ownerStartPtr_(nullptr),
    colIndicesPtr_(nullptr),
    ldu2csrPerm_(nullptr)
{
    if (reuse)
    {
        if (A.valuesPtr_)
        {
            valuesPtr_ = A.valuesPtr_;
            A.valuesPtr_ = nullptr;
        }

        if (A.ownerStartPtr_)
        {
            ownerStartPtr_ = A.ownerStartPtr_;
            A.ownerStartPtr_ = nullptr;
        }

        if (A.colIndicesPtr_)
        {
            colIndicesPtr_ = A.colIndicesPtr_;
            A.colIndicesPtr_ = nullptr;
        }

        if (A.ldu2csrPerm_)
        {
            ldu2csrPerm_ = A.ldu2csrPerm_;
            A.ldu2csrPerm_ = nullptr;
        }
    }
    else
    {
        if (A.valuesPtr_)
        {
            valuesPtr_ = new deviceField<Foam::scalar>(*(A.valuesPtr_));
        }

        if (A.ownerStartPtr_)
        {
            ownerStartPtr_ = new deviceField<Foam::label>(*(A.ownerStartPtr_));
        }

        if (A.colIndicesPtr_)
        {
            colIndicesPtr_ = new deviceField<Foam::label>(*(A.colIndicesPtr_));
        }

        if (A.ldu2csrPerm_)
        {
            ldu2csrPerm_ = new deviceField<Foam::label>(*(A.ldu2csrPerm_));
        }
    }
}

// * * * * * * * * * * * *  Public Member Functions * * * * * * * * * * * *  //

void Foam::deviceCsrMatrix::finalize()
{
    if (valuesPtr_)
    {
        delete valuesPtr_;
    }

    if (ownerStartPtr_)
    {
        // delete ownerStartPtr_;
    }

    if (colIndicesPtr_)
    {
        // delete colIndicesPtr_;
    }

    if (ldu2csrPerm_)
    {
        delete ldu2csrPerm_;
    }
}


const Foam::deviceField<Foam::scalar>& Foam::deviceCsrMatrix::values() const
{
    if (!valuesPtr_)
    {
        FatalErrorInFunction
            << "valuesPtr_ unallocated"
            << abort(FatalError);
    }

    return *valuesPtr_;
}


// * * * * * * * * * * * * * * * * Operations * * * * * * * * * * * * * * * //

//- Deallocate useless addressing pointer
void Foam::deviceCsrMatrix::clearAddressing()
{
    if (ownerStartPtr_)
    {
        delete ownerStartPtr_;
        // cudaFree(ownerStartPtr_);
    }

    if (colIndicesPtr_)
    {
        delete colIndicesPtr_;
    }
}


//- Find permutation array and new addressing vectors (no interface)
void Foam::deviceCsrMatrix::computePermutation(const devicelduMatrix& lduMatrix)
{
    const label nCells = lduMatrix.mesh().nCells();
    const label nIntFaces = lduMatrix.mesh().nInternalFaces();
    const label totNnz = nCells + 2*nIntFaces;

    const deviceField<Foam::label>& own = lduMatrix.mesh().owner();
    const deviceField<Foam::label>& neigh = lduMatrix.mesh().neighbour();

    ownerStartPtr_ = new deviceField<Foam::label>(nCells+1, Foam::Zero);
    ldu2csrPerm_ = new deviceField<Foam::label>(totNnz);
    colIndicesPtr_ = new deviceField<Foam::label>(totNnz);

    deviceField<Foam::label> rowIndices(totNnz);
    deviceField<Foam::label> tmpPerm(totNnz);
    deviceField<Foam::label> rowindicesTmp(totNnz);
    deviceField<Foam::label> colindicesTmp(totNnz);

    // Initialize: tmpPerm = [0, 1, ... totNnz-1]
    //             rowindicesTmp = [0, ... nCells-1, (owner), (neighbour)]
    //             colindicesTmp = [0, ... nCells-1, (neighbour), (owner)]
    initializeAddressing
    (
        nIntFaces,
        totNnz,
        own.cdata(),
        neigh.cdata(),
        tmpPerm.data(),
        rowindicesTmp.data(),
        colindicesTmp.data(),
        nCells
    );

    // Compute sorting to obtain permutation
    computeSorting
    (
        totNnz,
        tmpPerm.data(),
        rowindicesTmp.data(),
        rowIndices.data(),
        ldu2csrPerm_->data()
    );

    // Apply permutation vector to find colIndices + compute ownerStart
    applyAddressingPermutation
    (
        nCells,
        totNnz,
        ldu2csrPerm_->cdata(),
        colindicesTmp.cdata(),
        rowIndices.cdata(),
        colIndicesPtr_->data(),
        ownerStartPtr_->data()
    );
}

//- Find permutation array and new addressing vectors
void Foam::deviceCsrMatrix::computePermutation
(
    const devicelduMatrix& lduMatrix,
    const label diagIndexGlobal,
    const label lowOffGlobal,
    const label uppOffGlobal,
    const deviceField<label>& extRows,
    const deviceField<label>& extCols
)
{
    const label nCells = lduMatrix.mesh().nCells();
    const label nIntFaces = lduMatrix.mesh().nInternalFaces();
    const label nnzExt = extRows.size();
    const label totNnz = nCells + 2*nIntFaces + nnzExt;

    const deviceField<Foam::label>& own = lduMatrix.mesh().owner();
    const deviceField<Foam::label>& neigh = lduMatrix.mesh().neighbour();

    ownerStartPtr_ = new deviceField<Foam::label>(nCells+1, Foam::Zero);
    ldu2csrPerm_ = new deviceField<Foam::label>(totNnz);
    colIndicesPtr_ = new deviceField<Foam::label>(totNnz);

    deviceField<Foam::label> rowIndices(totNnz);
    deviceField<Foam::label> tmpPerm(totNnz);
    deviceField<Foam::label> rowindicesTmp(totNnz);
    deviceField<Foam::label> colindicesTmp(totNnz);

    // Initialize: tmpPerm = [0, 1, ... totNnz-1]
    //             rowindicesTmp = [0, ... nCells-1, (owner), (neighbour), (extrows)]
    //             colindicesTmp = [0, ... nCells-1, (neighbour), (owner), (extcols)]
    initializeAddressingExt
    (
        nIntFaces,
        nnzExt,
        totNnz,
        own.cdata(),
        neigh.cdata(),
        extRows.cdata(),
        extCols.cdata(),
        tmpPerm.data(),
        rowindicesTmp.data(),
        colindicesTmp.data(),
        nCells
    );

    // Compute sorting to obtain permutation
    computeSorting
    (
        totNnz,
        tmpPerm.data(),
        rowindicesTmp.data(),
        rowIndices.data(),
        ldu2csrPerm_->data()
    );

    // Make column indices from local to global
    localToGlobalColIndices
    (
        nIntFaces,
        lowOffGlobal,
        uppOffGlobal,
        colindicesTmp.data(),
        nCells,
        diagIndexGlobal
    );

    // Apply permutation vector to find colIndices + compute ownerStart
    applyAddressingPermutation
    (
        nCells,
        totNnz,
        ldu2csrPerm_->cdata(),
        colindicesTmp.cdata(),
        rowIndices.cdata(),
        colIndicesPtr_->data(),
        ownerStartPtr_->data()
    );
}

//- Apply permutation to LDU values (no permutation)
void Foam::deviceCsrMatrix::applyPermutation(const devicelduMatrix& lduMatrix)
{
    // Verify that the permutation has already been computed
    if(!ldu2csrPerm_)
    {
        computePermutation(lduMatrix);
    }

    label nIntFaces = lduMatrix.mesh().nInternalFaces();
    label nCells = lduMatrix.mesh().nCells();
    label totNnz = nCells + 2*nIntFaces;

    const deviceField<Foam::scalar>& diag = lduMatrix.diag();
    const deviceField<Foam::scalar>& upper = lduMatrix.upper();
    const deviceField<Foam::scalar>& lower = lduMatrix.lower();

    if(!valuesPtr_)
    {
        valuesPtr_ = new deviceField<Foam::scalar>(totNnz);
    }

    // Initialize valuesTmp = [(diag), (upper), (lower)]
    deviceField<Foam::scalar> valuesTmp(totNnz);

    initializeValue
    (
        nIntFaces,
        upper.cdata(),
        lower.cdata(),
        valuesTmp.data(),
        nCells,
        diag.cdata()
    );

    // Apply permutation
    applyValuePermutation
    (
        totNnz,
        ldu2csrPerm_->cdata(),
        valuesTmp.cdata(),
        valuesPtr_->data()
    );
}

//- Apply permutation from LDU to CSR considering the interface values
void Foam::deviceCsrMatrix:: applyPermutation
(
    const devicelduMatrix& lduMatrix,
    const label diagIndexGlobal,
    const label lowOffGlobal,
    const label uppOffGlobal,
    const deviceField<label>& extRows,
    const deviceField<label>& extCols,
    const deviceField<scalar>& extVals
)
{
    // Verify that the permutation has already been computed
    if(!ldu2csrPerm_)
    {
        computePermutation(
            lduMatrix,
            diagIndexGlobal,
            lowOffGlobal,
            uppOffGlobal,
            extRows,
            extCols
        );
    }

    label nIntFaces = lduMatrix.mesh().nInternalFaces();
    label nCells = lduMatrix.mesh().nCells();
    label nnzExt = extVals.size();
    label totNnz = nCells + 2*nIntFaces + nnzExt;

    const deviceField<Foam::scalar>& diag = lduMatrix.diag();
    const deviceField<Foam::scalar>& upper = lduMatrix.upper();
    const deviceField<Foam::scalar>& lower = lduMatrix.lower();

    if(!valuesPtr_)
    {
        valuesPtr_ = new deviceField<Foam::scalar>(totNnz);
    }

    // Initialize valuesTmp = [(diag), (upper), (lower), (extValues)]
    deviceField<Foam::scalar> valuesTmp(totNnz);

    initializeValueExt
    (
        nIntFaces,
        nnzExt,
        upper.cdata(),
        lower.cdata(),
        extVals.cdata(),
        valuesTmp.data(),
        nCells,
        diag.cdata()
    );

    // Apply permutation
    applyValuePermutation
    (
        totNnz,
        ldu2csrPerm_->cdata(),
        valuesTmp.cdata(),
        valuesPtr_->data()
    );
}

//- Apply permutation from LDU to CSR considering the interface values
void Foam::deviceCsrMatrix:: applyPermutation
(
    const devicelduMatrix& lduMatrix,
    const deviceField<scalar>& extVals
)
{
    label nIntFaces = lduMatrix.mesh().nInternalFaces();
    label nCells = lduMatrix.mesh().nCells();
    label nnzExt = extVals.size();
    label totNnz = nCells + 2*nIntFaces + nnzExt;

    const deviceField<Foam::scalar>& diag = lduMatrix.diag();
    const deviceField<Foam::scalar>& upper = lduMatrix.upper();
    const deviceField<Foam::scalar>& lower = lduMatrix.lower();

    if(!valuesPtr_)
    {
        valuesPtr_ = new deviceField<Foam::scalar>(totNnz);
    }

    // Initialize valuesTmp = [(diag), (upper), (lower), (extValues)]
    deviceField<Foam::scalar> valuesTmp(totNnz);

    initializeValueExt
    (
        nIntFaces,
        nnzExt,
        upper.cdata(),
        lower.cdata(),
        extVals.cdata(),
        valuesTmp.data(),
        nCells,
        diag.cdata()
    );

    // Apply permutation
    applyValuePermutation
    (
        totNnz,
        ldu2csrPerm_->cdata(),
        valuesTmp.cdata(),
        valuesPtr_->data()
    );
}



// * * * * * * * * * * * * * Explicit instantiations  * * * * * * * * * * * //

// ************************************************************************* //
