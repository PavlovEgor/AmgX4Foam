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

#include "csrAdressing.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //


// * * * * * * * * * * * * * * * Constructors * * * * * * * * * * * * * * * //

Foam::csrAdressing::csrAdressing()
:
    ownerStartPtr_(nullptr),
    colIndicesPtr_(nullptr),
    ldu2csrPerm_(nullptr)
{}

Foam::csrAdressing::csrAdressing(const csrAdressing& A)
:
    ownerStartPtr_(nullptr),
    colIndicesPtr_(nullptr),
    ldu2csrPerm_(nullptr)
{
    if (A.ownerStartPtr_)
    {
        ownerStartPtr_ = new labelList(*(A.ownerStartPtr_));
    }

    if (A.colIndicesPtr_)
    {
        colIndicesPtr_ = new labelList(*(A.colIndicesPtr_));
    }

    if (A.ldu2csrPerm_)
    {
        ldu2csrPerm_ = new labelList(*(A.ldu2csrPerm_));
    }
}


Foam::csrAdressing::csrAdressing(csrAdressing& A, bool reuse)
:
    ownerStartPtr_(nullptr),
    colIndicesPtr_(nullptr),
    ldu2csrPerm_(nullptr)
{
    if (reuse)
    {
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
        if (A.ownerStartPtr_)
        {
            ownerStartPtr_ = new labelList(*(A.ownerStartPtr_));
        }

        if (A.colIndicesPtr_)
        {
            colIndicesPtr_ = new labelList(*(A.colIndicesPtr_));
        }

        if (A.ldu2csrPerm_)
        {
            ldu2csrPerm_ = new labelList(*(A.ldu2csrPerm_));
        }
    }
}

// * * * * * * * * * * * *  Public Member Functions * * * * * * * * * * * *  //

void Foam::csrAdressing::finalizeAdressing()
{
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


// * * * * * * * * * * * * * * * * Operations * * * * * * * * * * * * * * * //

//- Deallocate useless addressing pointer
void Foam::csrAdressing::clearAddressing()
{
    if (ownerStartPtr_)
    {
        delete ownerStartPtr_;
    }

    if (colIndicesPtr_)
    {
        delete colIndicesPtr_;
    }
}


//- Find permutation array and new addressing vectors (no interface)
void Foam::csrAdressing::computePermutation(const lduAddressing * addr)
{
    const labelList& own = addr->lowerAddr();
    const labelList& neigh = addr->upperAddr();
    
    const label nCells = addr->size();
    const label nIntFaces = own.size();
    const label totNnz = nCells + 2*nIntFaces;

    ownerStartPtr_ = new labelList(nCells+1, Foam::Zero);
    ldu2csrPerm_ = new labelList(totNnz);
    colIndicesPtr_ = new labelList(totNnz);

    labelList rowIndices(totNnz);
    labelList tmpPerm(totNnz);
    labelList rowindicesTmp(totNnz);
    labelList colindicesTmp(totNnz);

    // Initialize: tmpPerm = [0, 1, ... totNnz-1]
    //             rowindicesTmp = [0, ... nCells-1, (owner), (neighbour)]
    //             colindicesTmp = [0, ... nCells-1, (neighbour), (owner)]
    initializeAddressing
    (
        nCells,
        nIntFaces,
        totNnz,
        own.cdata(),
        neigh.cdata(),
        tmpPerm.data(),
        rowindicesTmp.data(),
        colindicesTmp.data()
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
void Foam::csrAdressing::computePermutation
(
    const lduAddressing * addr,
    const label diagIndexGlobal,
    const label lowOffGlobal,
    const label uppOffGlobal,
    const labelList& extRows,
    const labelList& extCols
)
{
    const labelList& own = addr->lowerAddr();
    const labelList& neigh = addr->upperAddr();
    
    const label nCells = addr->size();
    const label nIntFaces = own.size();
    const label nnzExt = extRows.size();
    const label totNnz = nCells + 2*nIntFaces + nnzExt;

    ownerStartPtr_ = new labelList(nCells+1, Foam::Zero);
    ldu2csrPerm_ = new labelList(totNnz);
    colIndicesPtr_ = new labelList(totNnz);

    labelList rowIndices(totNnz);
    labelList tmpPerm(totNnz);
    labelList rowindicesTmp(totNnz);
    labelList colindicesTmp(totNnz);

    // Initialize: tmpPerm = [0, 1, ... totNnz-1]
    //             rowindicesTmp = [0, ... nCells-1, (owner), (neighbour), (extrows)]
    //             colindicesTmp = [0, ... nCells-1, (neighbour), (owner), (extcols)]
    initializeAddressingExt
    (
        nCells,
        nIntFaces,
        nnzExt,
        totNnz,
        own.cdata(),
        neigh.cdata(),
        extRows.cdata(),
        extCols.cdata(),
        tmpPerm.data(),
        rowindicesTmp.data(),
        colindicesTmp.data()
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
        nCells,
        nIntFaces,
        diagIndexGlobal,
        lowOffGlobal,
        uppOffGlobal,
        colindicesTmp.data()        
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

// ************************************************************************* //
