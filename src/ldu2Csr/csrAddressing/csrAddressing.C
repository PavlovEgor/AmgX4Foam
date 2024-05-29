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

#include "csrAddressing.H"

#include "globalIndex.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

//Foam::csrAddressingExecutor Foam::csrAddressing::csrAddrExec_ = cudaCsrAddressingExecutor();

// * * * * * * * * * * * * * * * Constructors * * * * * * * * * * * * * * * //

Foam::csrAddressing::csrAddressing(word mode)
:
    ownerStartPtr_(nullptr),
    colIndicesPtr_(nullptr),
    ldu2csrPerm_(nullptr)
{
    if (mode.starts_with("d"))
    {
        csrAddrExec_ = cudaCsrAddressingExecutor();
	}
    else if (mode.starts_with("h"))
    {
        csrAddrExec_ = cpuCsrAddressingExecutor();
	}
    else
    {
        FatalErrorInFunction
            << "'" << mode << "' is not a valid AMGx execution mode"
            << exit(FatalError);
    }
}

//Foam::csrAddressing::csrAddressing(const csrAddressing& A)
//:
//    ownerStartPtr_(nullptr),
//    colIndicesPtr_(nullptr),
//    ldu2csrPerm_(nullptr)
//{
//    if (A.ownerStartPtr_)
//    {
//        ownerStartPtr_ = new labelList(*(A.ownerStartPtr_));
//    }
//
//    if (A.colIndicesPtr_)
//    {
//        colIndicesPtr_ = new labelList(*(A.colIndicesPtr_));
//    }
//
//    if (A.ldu2csrPerm_)
//    {
//        ldu2csrPerm_ = new labelList(*(A.ldu2csrPerm_));
//    }
//}
//
//
//Foam::csrAddressing::csrAddressing(csrAddressing& A, bool reuse)
//:
//    ownerStartPtr_(nullptr),
//    colIndicesPtr_(nullptr),
//    ldu2csrPerm_(nullptr)
//{
//    if (reuse)
//    {
//        if (A.ownerStartPtr_)
//        {
//            ownerStartPtr_ = A.ownerStartPtr_;
//            A.ownerStartPtr_ = nullptr;
//        }
//
//        if (A.colIndicesPtr_)
//        {
//            colIndicesPtr_ = A.colIndicesPtr_;
//            A.colIndicesPtr_ = nullptr;
//        }
//
//        if (A.ldu2csrPerm_)
//        {
//            ldu2csrPerm_ = A.ldu2csrPerm_;
//            A.ldu2csrPerm_ = nullptr;
//        }
//    }
//    else
//    {
//        if (A.ownerStartPtr_)
//        {
//            ownerStartPtr_ = new labelList(*(A.ownerStartPtr_));
//        }
//
//        if (A.colIndicesPtr_)
//        {
//            colIndicesPtr_ = new labelList(*(A.colIndicesPtr_));
//        }
//
//        if (A.ldu2csrPerm_)
//        {
//            ldu2csrPerm_ = new labelList(*(A.ldu2csrPerm_));
//        }
//    }
//}

// * * * * * * * * * * * *  Public Member Functions * * * * * * * * * * * *  //

void Foam::csrAddressing::finalizeAdressing()
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
        //delete ldu2csrPerm_;
        std::visit([this](const auto& exec)
               {exec.template clear<label>(this->ldu2csrPerm_); }, csrAddrExec_);
    }
}


// * * * * * * * * * * * * * * * * Operations * * * * * * * * * * * * * * * //

//- Deallocate useless addressing pointer
void Foam::csrAddressing::clearAddressing()
{
    if (ownerStartPtr_)
    {
        //delete ownerStartPtr_;
        std::visit([this](const auto& exec)
               {exec.template clear<label>(this->ownerStartPtr_); }, csrAddrExec_);
    }

    if (colIndicesPtr_)
    {
        //delete colIndicesPtr_;
        std::visit([this](const auto& exec)
               {exec.template clear<label>(this->colIndicesPtr_); }, csrAddrExec_);
    }
}

//- Find permutation array and new addressing vectors (no interface)
void Foam::csrAddressing::computePermutation(const lduAddressing * addr)
{
	const label* own = nullptr;
	const label* neigh = nullptr;

	const label* hostOwn = addr->lowerAddr().cdata();
	label ownSize = addr->lowerAddr().size();
	const label* hostNeigh = addr->upperAddr().cdata();
	label neighSize = addr->upperAddr().size();

	std::visit([&hostOwn, &own, ownSize](const auto& exec)
               { own = exec.template copyFromFoam<label>(ownSize,hostOwn); },
               csrAddrExec_);
	std::visit([&hostNeigh, &neigh, neighSize](const auto& exec)
               { neigh = exec.template copyFromFoam<label>(neighSize,hostNeigh); },
               csrAddrExec_);
    //const labelList& own = addr->lowerAddr();
    //const labelList& neigh = addr->upperAddr();
    
    const label nCells = addr->size();
    const label nIntFaces = ownSize;
    const label totNnz = nCells + 2*nIntFaces;

    nOwnerStart_ = nCells+1;
    nLocalNz_ = totNnz;

    //ownerStartPtr_ = new label[nCells+1];
    //ldu2csrPerm_ = new label[totNnz];
    //colIndicesPtr_ = new label[totNnz];

    std::visit([this, nCells](const auto& exec)
               { this->ownerStartPtr_ = exec.template alloc<label>(nCells+1); },
               csrAddrExec_);
    std::visit([this, totNnz](const auto& exec)
               { this->colIndicesPtr_ = exec.template alloc<label>(totNnz); },
               csrAddrExec_);
    std::visit([this, totNnz](const auto& exec)
               { this->ldu2csrPerm_ = exec.template alloc<label>(totNnz); },
               csrAddrExec_);

    label* rowIndices = nullptr;
    label* tmpPerm = nullptr;
	label* rowindicesTmp = nullptr;
	label* colindicesTmp = nullptr;
    std::visit([&rowIndices, totNnz](const auto& exec)
               { rowIndices = exec.template alloc<label>(totNnz); },
               csrAddrExec_);
    std::visit([&tmpPerm, totNnz](const auto& exec)
               { tmpPerm = exec.template alloc<label>(totNnz); },
               csrAddrExec_);
    std::visit([&rowindicesTmp, totNnz](const auto& exec)
               { rowindicesTmp = exec.template alloc<label>(totNnz); },
               csrAddrExec_);
    std::visit([&colindicesTmp, totNnz](const auto& exec)
               { colindicesTmp = exec.template alloc<label>(totNnz); },
               csrAddrExec_);
    //labelList tmpPerm(totNnz);
    //labelList rowindicesTmp(totNnz);
    //labelList colindicesTmp(totNnz);

    // Initialize: tmpPerm = [0, 1, ... totNnz-1]
    //             rowindicesTmp = [0, ... nCells-1, (owner), (neighbour)]
    //             colindicesTmp = [0, ... nCells-1, (neighbour), (owner)]
    initializeAddressing
    (
        nCells,
        nIntFaces,
        totNnz,
        own,
        neigh,
        tmpPerm,
        rowindicesTmp,
        colindicesTmp
    );

    // Compute sorting to obtain permutation
    computeSorting
    (
        totNnz,
        tmpPerm,
        rowindicesTmp,
        rowIndices,
        ldu2csrPerm_
    );

    // Apply permutation vector to find colIndices + compute ownerStart
    applyAddressingPermutation
    (
        nCells,
        totNnz,
        ldu2csrPerm_,
        colindicesTmp,
        rowIndices,
        colIndicesPtr_,
        ownerStartPtr_
    );
    std::visit([rowIndices](const auto& exec)
               {exec.template clear<label>(rowIndices); }, csrAddrExec_);
    std::visit([tmpPerm](const auto& exec)
               {exec.template clear<label>(tmpPerm); }, csrAddrExec_);
    std::visit([rowindicesTmp](const auto& exec)
               {exec.template clear<label>(rowindicesTmp); }, csrAddrExec_);
    std::visit([colindicesTmp](const auto& exec)
               {exec.template clear<label>(colindicesTmp); }, csrAddrExec_);
    std::visit([own](const auto& exec)
               {exec.template clear<label>(own); }, csrAddrExec_);
    std::visit([neigh](const auto& exec)
               {exec.template clear<label>(neigh); }, csrAddrExec_);
}


//- Find permutation array and new addressing vectors
void Foam::csrAddressing::computePermutation
(
    const lduAddressing& addr,
    const lduInterfacePtrsList& interfaces,
          label& nnzExt
)
{
//    const labelList& own = addr.lowerAddr();
//    const labelList& neigh = addr.upperAddr();

	const label* own = nullptr;
	const label* neigh = nullptr;

	const label* hostOwn = addr.lowerAddr().cdata();
	label ownSize = addr.lowerAddr().size();
	const label* hostNeigh = addr.upperAddr().cdata();
	label neighSize = addr.upperAddr().size();

	std::visit([&hostOwn, &own, ownSize](const auto& exec)
               { own = exec.template copyFromFoam<label>(ownSize,hostOwn); },
               csrAddrExec_);
	std::visit([&hostNeigh, &neigh, neighSize](const auto& exec)
               { neigh = exec.template copyFromFoam<label>(neighSize,hostNeigh); },
               csrAddrExec_);

	const label nCells = addr.size();
    const label nIntFaces = ownSize;

    const globalIndex globalNumbering(nCells);

    const label diagIndexGlobal = globalNumbering.toGlobal(0);
    const label lowOffGlobal = globalNumbering.toGlobal(own[0]) - own[0];
    const label uppOffGlobal = globalNumbering.toGlobal(neigh[0]) - neigh[0];

    labelList globalCells
    (
        identity
        (
            globalNumbering.localSize(),
            globalNumbering.localStart()
        )
    );

    // Connections to neighbouring processors
    const label nReq = Pstream::nRequests(); //Operation useless if the mesh is steady

    nnzExt = 0;

    // Initialise transfer of global cells
    forAll(interfaces, patchi)
    {
        if (interfaces.set(patchi))
        {
            nnzExt += addr.patchAddr(patchi).size();

            interfaces[patchi].initInternalFieldTransfer
            (
                Pstream::commsTypes::nonBlocking,
                globalCells
            );
        }
    }

    if (Pstream::parRun())
    {
        Pstream::waitRequests(nReq);
    }

    labelField extRows(nnzExt, Foam::Zero);
    labelField extCols(nnzExt, Foam::Zero);


    nnzExt = 0;
    
    forAll(interfaces, patchi)
    {
        if (interfaces.set(patchi))
        {
            // Processor-local values
            const labelUList& faceCells = addr.patchAddr(patchi);
            const label len = faceCells.size();

            labelList nbrCells
            (
                interfaces[patchi].internalFieldTransfer
                (
                    Pstream::commsTypes::nonBlocking,
                    globalCells
                )
            );

            if (faceCells.size() != nbrCells.size())
            {
                FatalErrorInFunction
                    << "Mismatch in interface sizes (AMI?)" << nl
                    << "Have " << faceCells.size() << " != "
                    << nbrCells.size() << nl
                    << exit(FatalError);
            }

            SubList<label>(extRows, len, nnzExt) = faceCells;
            SubList<label>(extCols, len, nnzExt) = nbrCells;
            nnzExt += len;
        }
    }

    const label* extDevRows = nullptr;
    const label* extDevCols = nullptr;
    const label* extRowsPtr = extRows.cdata();
    const label* extColsPtr = extCols.cdata();
	std::visit([&extRowsPtr, &extDevRows, nnzExt](const auto& exec)
               { extDevRows = exec.template copyFromFoam<label>(nnzExt,extRowsPtr); },
               csrAddrExec_);
	std::visit([&extColsPtr, &extDevCols, nnzExt](const auto& exec)
               { extDevCols = exec.template copyFromFoam<label>(nnzExt,extColsPtr); },
               csrAddrExec_);

    const label totNnz = nCells + 2*nIntFaces + nnzExt;

    //ownerStartPtr_ = new labelList(nCells+1, Foam::Zero);
    //ldu2csrPerm_ = new labelList(totNnz);
    //colIndicesPtr_ = new labelList(totNnz);
    //ownerStartPtr_ = new label[nCells+1];
    //ldu2csrPerm_ = new label[totNnz];
    //colIndicesPtr_ = new label[totNnz];

    nOwnerStart_ = nCells+1;
    nLocalNz_ = totNnz;
    std::visit([this, nCells](const auto& exec)
               { this->ownerStartPtr_ = exec.template alloc<label>(nCells+1); },
               csrAddrExec_);
    std::visit([this, totNnz](const auto& exec)
               { this->colIndicesPtr_ = exec.template alloc<label>(totNnz); },
               csrAddrExec_);
    std::visit([this, totNnz](const auto& exec)
               { this->ldu2csrPerm_ = exec.template alloc<label>(totNnz); },
               csrAddrExec_);

    label* rowIndices = nullptr;
    label* tmpPerm = nullptr;
	label* rowindicesTmp = nullptr;
	label* colindicesTmp = nullptr;

    std::visit([&rowIndices, totNnz](const auto& exec)
               { rowIndices = exec.template alloc<label>(totNnz); },
               csrAddrExec_);
    std::visit([&tmpPerm, totNnz](const auto& exec)
               { tmpPerm = exec.template alloc<label>(totNnz); },
               csrAddrExec_);
    std::visit([&rowindicesTmp, totNnz](const auto& exec)
               { rowindicesTmp = exec.template alloc<label>(totNnz); },
               csrAddrExec_);
    std::visit([&colindicesTmp, totNnz](const auto& exec)
               { colindicesTmp = exec.template alloc<label>(totNnz); },
               csrAddrExec_);

    //labelList rowIndices(totNnz);
    //labelList tmpPerm(totNnz);
    //labelList rowindicesTmp(totNnz);
    //labelList colindicesTmp(totNnz);

    // Initialize: tmpPerm = [0, 1, ... totNnz-1]
    //             rowindicesTmp = [0, ... nCells-1, (owner), (neighbour), (extrows)]
    //             colindicesTmp = [0, ... nCells-1, (neighbour), (owner), (extcols)]
    initializeAddressingExt
    (
        nCells,
        nIntFaces,
        nnzExt,
        totNnz,
        own,
        neigh,
        extDevRows,
        extDevCols,
        tmpPerm,
        rowindicesTmp,
        colindicesTmp
    );

    // Compute sorting to obtain permutation
    computeSorting
    (
        totNnz,
        tmpPerm,
        rowindicesTmp,
        rowIndices,
        ldu2csrPerm_
    );

    // Make column indices from local to global
    localToGlobalColIndices
    (
        nCells,
        nIntFaces,
        diagIndexGlobal,
        lowOffGlobal,
        uppOffGlobal,
        colindicesTmp
    );

    // Apply permutation vector to find colIndices + compute ownerStart
    applyAddressingPermutation
    (
        nCells,
        totNnz,
        ldu2csrPerm_,
        colindicesTmp,
        rowIndices,
        colIndicesPtr_,
        ownerStartPtr_
    );
    std::visit([rowIndices](const auto& exec)
               {exec.template clear<label>(rowIndices); }, csrAddrExec_);
    std::visit([tmpPerm](const auto& exec)
               {exec.template clear<label>(tmpPerm); }, csrAddrExec_);
    std::visit([rowindicesTmp](const auto& exec)
               {exec.template clear<label>(rowindicesTmp); }, csrAddrExec_);
    std::visit([colindicesTmp](const auto& exec)
               {exec.template clear<label>(colindicesTmp); }, csrAddrExec_);
    std::visit([own](const auto& exec)
               {exec.template clear<label>(own); }, csrAddrExec_);
    std::visit([neigh](const auto& exec)
               {exec.template clear<label>(neigh); }, csrAddrExec_);
    std::visit([extDevRows](const auto& exec)
               {exec.template clear<label>(extDevRows); }, csrAddrExec_);
    std::visit([extDevCols](const auto& exec)
               {exec.template clear<label>(extDevCols); }, csrAddrExec_);
}

// ************************************************************************* //
