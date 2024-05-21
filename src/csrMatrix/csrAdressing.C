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

#include "globalIndex.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //


// * * * * * * * * * * * * * * * Constructors * * * * * * * * * * * * * * * //

Foam::csrAdressing::csrAdressing()
:
    ownerStartPtr_(nullptr),
    colIndicesPtr_(nullptr),
    ldu2csrPerm_(nullptr),
    rowsConsDispPtr_(nullptr),
    intFacesConsDispPtr_(nullptr),
    extNzConsDispPtr_(nullptr)
{}

Foam::csrAdressing::csrAdressing(const csrAdressing& A)
:
    ownerStartPtr_(nullptr),
    colIndicesPtr_(nullptr),
    ldu2csrPerm_(nullptr),
    rowsConsDispPtr_(nullptr),
    intFacesConsDispPtr_(nullptr),
    extNzConsDispPtr_(nullptr)
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

    if (A.rowsConsDispPtr_)
    {
        rowsConsDispPtr_ = new labelList(*(A.rowsConsDispPtr_));
    }

    if (A.intFacesConsDispPtr_)
    {
        intFacesConsDispPtr_ = new labelList(*(A.intFacesConsDispPtr_));
    }

    if (A.extNzConsDispPtr_)
    {
        extNzConsDispPtr_ = new labelList(*(A.extNzConsDispPtr_));
    }
}


Foam::csrAdressing::csrAdressing(csrAdressing& A, bool reuse)
:
    ownerStartPtr_(nullptr),
    colIndicesPtr_(nullptr),
    ldu2csrPerm_(nullptr),
    rowsConsDispPtr_(nullptr),
    intFacesConsDispPtr_(nullptr),
    extNzConsDispPtr_(nullptr)
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

        if (A.rowsConsDispPtr_)
        {
            rowsConsDispPtr_ = A.rowsConsDispPtr_;
            A.rowsConsDispPtr_ = nullptr;
        }

        if (A.intFacesConsDispPtr_)
        {
            intFacesConsDispPtr_ = A.intFacesConsDispPtr_;
            A.intFacesConsDispPtr_ = nullptr;
        }

        if (A.extNzConsDispPtr_)
        {
            extNzConsDispPtr_ = A.extNzConsDispPtr_;
            A.extNzConsDispPtr_ = nullptr;
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

        if (A.rowsConsDispPtr_)
        {
            rowsConsDispPtr_ = new labelList(*(A.rowsConsDispPtr_));
        }

        if (A.intFacesConsDispPtr_)
        {
            intFacesConsDispPtr_ = new labelList(*(A.intFacesConsDispPtr_));
        }

        if (A.extNzConsDispPtr_)
        {
            extNzConsDispPtr_ = new labelList(*(A.extNzConsDispPtr_));
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

    if (rowsConsDispPtr_)
    {
        delete rowsConsDispPtr_;
    }

    if (intFacesConsDispPtr_)
    {
        delete intFacesConsDispPtr_;
    }

    if (extNzConsDispPtr_)
    {
        delete extNzConsDispPtr_;
    }
}


// * * * * * * * * * * * * * * * * Operations * * * * * * * * * * * * * * * //

//- Initialize matrix comunications for consolidation
void Foam::csrAdressing::initializeComms(label commId, bool gpuProc)
{
    gpuWorld_ = commId;

    gpuProc_ = gpuProc;

    Pout << "---> this process talks with a GPU: " << gpuProc_ << nl;

    gpuWorldSize_ = Pstream::nProcs(gpuWorld_);
    myGpuWorldRank_ = Pstream::myProcNo(gpuWorld_);

    if (gpuWorldSize_ > 1)
    {
        consolidationStatus_ = ConsolidationStatus::necessary;
        Info << "the consolidation is necessary" << nl;
    }
    else
    {
        consolidationStatus_ = ConsolidationStatus::notNecessary;
        Info << "the consolidation is not necessary" << nl;
    }
}

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


//- Deallocate useless addressing pointer
void Foam::csrAdressing::initializeConsolidation
(
    const label nLocalRows,
    const label diagIndexGlobal,
    const label lowOffGlobal,
    const label uppOffGlobal,
    const labelList& own,
    const labelList& neigh,
    const labelList& extRows,
    const labelList& extCols,
          label& nConsTotNz,
          labelList* consDiagOffGlob, 
          labelList* consLowOffGlob, 
          labelList* consUppOffGlob,
          List<labelList>& ownLst,
          List<labelList>& neighLst,
          List<labelList>& extRowsLst,
          List<labelList>& extColsLst
)
{
    const label nLocalIntFaces = own.size();
    const label nLocalExtNz = extRows.size();
    
    rowsConsDispPtr_ = new labelList(gpuWorldSize_ + 1, Foam::Zero);
    rowsConsDispPtr_->data()[myGpuWorldRank_ + 1] = nLocalRows;
    Pstream::gatherList(*rowsConsDispPtr_, UPstream::msgType(), gpuWorld_);

    intFacesConsDispPtr_ = new labelList(gpuWorldSize_ + 1, Foam::Zero);
    intFacesConsDispPtr_->data()[myGpuWorldRank_ + 1] = nLocalIntFaces;
    Pstream::gatherList(*intFacesConsDispPtr_, UPstream::msgType(), gpuWorld_);

    extNzConsDispPtr_ = new labelList(gpuWorldSize_ + 1, Foam::Zero);
    extNzConsDispPtr_->data()[myGpuWorldRank_ + 1] = nLocalExtNz;
    Pstream::gatherList(*extNzConsDispPtr_, UPstream::msgType(), gpuWorld_);

    for(label i=0; i<gpuWorldSize_; ++i)
    {
        rowsConsDispPtr_->data()[i+1] += rowsConsDispPtr_->cdata()[i];
        intFacesConsDispPtr_->data()[i+1] += intFacesConsDispPtr_->cdata()[i];
        extNzConsDispPtr_->data()[i+1] += extNzConsDispPtr_->cdata()[i];
    }

    nConsRows_ = rowsConsDispPtr_->last();
    nConsIntFaces_ = intFacesConsDispPtr_->last();
    nConsExtNz_ = extNzConsDispPtr_->last();
    nConsTotNz = nConsRows_ + 2*nConsIntFaces_ + nConsExtNz_;

    ownLst[myGpuWorldRank_] = own;
    Pstream::gatherList(ownLst, UPstream::msgType(), gpuWorld_);
    
    neighLst[myGpuWorldRank_] = neigh;
    Pstream::gatherList(neighLst, UPstream::msgType(), gpuWorld_);
    
    extRowsLst[myGpuWorldRank_] = extRows;
    Pstream::gatherList(extRowsLst, UPstream::msgType(), gpuWorld_);

    extColsLst[myGpuWorldRank_] = extCols;
    Pstream::gatherList(extColsLst, UPstream::msgType(), gpuWorld_);

    consDiagOffGlob = new labelList(gpuWorldSize_);
    consDiagOffGlob[myGpuWorldRank_] = diagIndexGlobal;

    consLowOffGlob = new labelList(gpuWorldSize_);
    consLowOffGlob[myGpuWorldRank_] = lowOffGlobal;

    consUppOffGlob = new labelList(gpuWorldSize_);
    consUppOffGlob[myGpuWorldRank_] = uppOffGlobal;

    consolidationStatus_ = ConsolidationStatus::initialized;

    return;
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
    labelList rowIndicesTmp(totNnz);
    labelList colIndicesTmp(totNnz);

    // Initialize: tmpPerm = [0, 1, ... totNnz-1]
    //             rowIndicesTmp = [0, ... nCells-1, (owner), (neighbour)]
    //             colIndicesTmp = [0, ... nCells-1, (neighbour), (owner)]
    initializeSequence(totNnz, tmpPerm.data());

    initializeSequence(nCells, rowIndicesTmp.data());
    initializeSequence(nCells, colIndicesTmp.data());
    
    initializeAddressing
    (
        nCells,
        nIntFaces,
        nIntFaces,
        own.cdata(),
        neigh.cdata(),
        tmpPerm.data(),
        rowIndicesTmp.data(),
        colIndicesTmp.data()
    );

    // Compute sorting to obtain permutation
    computeSorting
    (
        totNnz,
        tmpPerm.data(),
        rowIndicesTmp.data(),
        rowIndices.data(),
        ldu2csrPerm_->data()
    );

    // Apply permutation vector to find colIndices + compute ownerStart
    applyAddressingPermutation
    (
        nCells,
        totNnz,
        ldu2csrPerm_->cdata(),
        colIndicesTmp.cdata(),
        rowIndices.cdata(),
        colIndicesPtr_->data(),
        ownerStartPtr_->data()
    );
}


//- Find permutation array and new addressing vectors
void Foam::csrAdressing::computePermutation
(
    const lduAddressing& addr,
    const lduInterfacePtrsList& interfaces,
          label& nnzExt
)
{
    const labelList& own = addr.lowerAddr();
    const labelList& neigh = addr.upperAddr();

    const label nCells = addr.size();
    const label nIntFaces = own.size();

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

    label totNnz;
    List<labelList> ownLst(gpuWorldSize_);
    List<labelList> neighLst(gpuWorldSize_);
    List<labelList> extColsLst(gpuWorldSize_);
    List<labelList> extRowsLst(gpuWorldSize_);
    labelList* consDiagOffGlob = nullptr;
    labelList* consLowOffGlob = nullptr;
    labelList* consUppOffGlob = nullptr;

    if(consolidationStatus_ == ConsolidationStatus::necessary)
    {
        initializeConsolidation(nCells, diagIndexGlobal, lowOffGlobal, uppOffGlobal,
                                own, neigh, extRows, extCols, totNnz,
                                consDiagOffGlob, consLowOffGlob, consUppOffGlob,
                                ownLst, neighLst, extRowsLst, extColsLst);
    }
    else
    {
        totNnz = nCells + 2*nIntFaces + nnzExt;
        nConsRows_ = nCells;
    }

    if(gpuProc_)
    {
        ownerStartPtr_ = new labelList(nConsRows_+1, Foam::Zero);
        ldu2csrPerm_ = new labelList(totNnz);
        colIndicesPtr_ = new labelList(totNnz);

        labelList rowIndices(totNnz, Zero);
        labelList tmpPerm(totNnz);
        labelList rowIndicesTmp(totNnz);
        labelList colIndicesTmp(totNnz);

        // Initialize: tmpPerm = [0, 1, ... totNnz-1]
        //             rowIndicesTmp = [0, ... nCells-1, (owner), (neighbour), (extrows)]
        //             colIndicesTmp = [0, ... nCells-1, (neighbour), (owner), (extcols)]
        initializeSequence(totNnz, tmpPerm.data());

        initializeSequence(nConsRows_, rowIndicesTmp.data());
        initializeSequence(nConsRows_, colIndicesTmp.data());

        if(consolidationStatus_ == ConsolidationStatus::initialized)
        {
            for(label i=0; i<gpuWorldSize_; ++i)
            {
                initializeAddressingExt
                (
                    nConsRows_,
                    nConsIntFaces_,
                    intFacesConsDispPtr_->cdata()[i+1] - intFacesConsDispPtr_->cdata()[i],
                    extNzConsDispPtr_->cdata()[i+1] - extNzConsDispPtr_->cdata()[i],
                    ownLst[i].cdata(),
                    neighLst[i].cdata(),
                    extRowsLst[i].cdata(),
                    extColsLst[i].cdata(),
                    tmpPerm.data(),
                    rowIndicesTmp.data(),
                    colIndicesTmp.data()
                );

                localToConsRowIndex
                (
                    nConsRows_,
                    nConsIntFaces_,
                    intFacesConsDispPtr_->cdata()[i+1] - intFacesConsDispPtr_->cdata()[i],
                    extNzConsDispPtr_->cdata()[i+1] - extNzConsDispPtr_->cdata()[i],
                    intFacesConsDispPtr_->cdata()[i],
                    extNzConsDispPtr_->cdata()[i],
                    rowsConsDispPtr_->cdata()[i],
                    rowIndicesTmp.data()
                );
            }            
        }
        else
        {
            initializeAddressingExt
            (
                nCells,
                nIntFaces,
                nIntFaces,
                nnzExt,
                own.cdata(),
                neigh.cdata(),
                extRows.cdata(),
                extCols.cdata(),
                tmpPerm.data(),
                rowIndicesTmp.data(),
                colIndicesTmp.data()
            );
        }
        

        // Compute sorting to obtain permutation
        computeSorting
        (
            totNnz,
            tmpPerm.data(),
            rowIndicesTmp.data(),
            rowIndices.data(),
            ldu2csrPerm_->data()
        );

        // Make column indices from local to global
        if(consolidationStatus_ == ConsolidationStatus::initialized)
        {
            for(label i=0; i<gpuWorldSize_; ++i)
            {
                localToGlobalColIndices
                (
                    nConsRows_,
                    rowsConsDispPtr_->cdata()[i+1] - rowsConsDispPtr_->cdata()[i],
                    intFacesConsDispPtr_->cdata()[i+1] - intFacesConsDispPtr_->cdata()[i],
                    consDiagOffGlob->cdata()[i],
                    consLowOffGlob->cdata()[i],
                    consUppOffGlob->cdata()[i],
                    colIndicesTmp.data(),
                    rowsConsDispPtr_->cdata()[i],
                    intFacesConsDispPtr_->cdata()[i]
                );
            }
        }
        else
        {
            localToGlobalColIndices
            (
                nConsRows_,
                nCells,
                nIntFaces,
                diagIndexGlobal,
                lowOffGlobal,
                uppOffGlobal,
                colIndicesTmp.data()        
            );
        }

        // Apply permutation vector to find colIndices + compute ownerStart
        applyAddressingPermutation
        (
            nConsRows_,
            totNnz,
            ldu2csrPerm_->cdata(),
            colIndicesTmp.cdata(),
            rowIndices.cdata(),
            colIndicesPtr_->data(),
            ownerStartPtr_->data()
        );
    }
    
    if(consDiagOffGlob) delete consDiagOffGlob;
    if(consLowOffGlob) delete consLowOffGlob;
    if(consUppOffGlob) delete consDiagOffGlob;

}

// ************************************************************************* //
