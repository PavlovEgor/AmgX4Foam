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

#include "PstreamGlobals.H"
#include "csrMatrix.H"

#include "globalIndex.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

// * * * * * * * * * * * * * * * Constructors * * * * * * * * * * * * * * * //

Foam::csrMatrix::csrMatrix(word mode)
:
    ownerStartPtr_(nullptr),
    colIndicesPtr_(nullptr),
    ldu2csrPerm_(nullptr),
    valuesPtr_(nullptr),
    rowsConsDispPtr_(nullptr),
    intFacesConsDispPtr_(nullptr),
    extNzConsDispPtr_(nullptr)
{
    if (mode.starts_with("h"))
    {
        csrMatExec_ = cpuCsrMatrixExecutor();
	}    
    else if (mode.starts_with("d"))
    {
#ifdef have_cuda
        csrMatExec_ = cudaCsrMatrixExecutor();
#else
        FatalErrorInFunction
            << "'" << mode << "' is not a available, CUDA version not compiled"
            << exit(FatalError);
#endif
	}
    else
    {
        FatalErrorInFunction
            << "'" << mode << "' is not a valid AMGx execution mode"
            << exit(FatalError);
    }
}

//Foam::csrMatrix::csrMatrix(const csrMatrix& A)
//:
//    csrMatrix(A),
//    valuesPtr_(nullptr)
//{
//    if (A.valuesPtr_)
//    {
//        valuesPtr_ = new scalarField(*(A.valuesPtr_));
//    }
//}
//
//
//Foam::csrMatrix::csrMatrix(csrMatrix& A, bool reuse)
//:
//    csrMatrix(A, reuse),
//    valuesPtr_(nullptr)
//{
//    if (reuse)
//    {
//        if (A.valuesPtr_)
//        {
//            valuesPtr_ = A.valuesPtr_;
//            A.valuesPtr_ = nullptr;
//        }
//    }
//    else
//    {
//        if (A.valuesPtr_)
//        {
//            valuesPtr_ = new scalarField(*(A.valuesPtr_));
//        }
//    }
//}

// * * * * * * * * * * * *  Public Member Functions * * * * * * * * * * * *  //

void Foam::csrMatrix::finalize()
{
    // NOTA: Implementare controllo con buleano o invalidazione del puntatore
    //       per gestire bene la finalizzazione
    
    if (ownerStartPtr_)
    {
        // delete ownerStartPtr_;
        std::visit([this](const auto& exec)
               {exec.template clear<label>(this->ownerStartPtr_); }, csrMatExec_);
    }

    if (colIndicesPtr_)
    {
        // delete colIndicesPtr_;
        std::visit([this](const auto& exec)
               {exec.template clear<label>(this->colIndicesPtr_); }, csrMatExec_);
    }

    if (ldu2csrPerm_)
    {
        //delete ldu2csrPerm_;
        std::visit([this](const auto& exec)
               {exec.template clear<label>(this->ldu2csrPerm_); }, csrMatExec_);
    }

    if (valuesPtr_)
    {
        // delete valuesPtr_;
        std::visit([this](const auto& exec)
                {exec.template clear<scalar>(this->valuesPtr_); }, csrMatExec_);
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


const Foam::scalar* Foam::csrMatrix::values() const
{
    if (!valuesPtr_)
    {
        FatalErrorInFunction
            << "valuesPtr_ unallocated"
            << abort(FatalError);
    }

    return valuesPtr_;
}


// * * * * * * * * * * * * * * * * Operations * * * * * * * * * * * * * * * //

//- Deallocate useless addressing pointer
void Foam::csrMatrix::clearAddressing()
{
    if (ownerStartPtr_)
    {
        //delete ownerStartPtr_;
        std::visit([this](const auto& exec)
               {exec.template clear<label>(this->ownerStartPtr_); }, csrMatExec_);

        ownerStartPtr_ = nullptr;
    }

    if (colIndicesPtr_)
    {
        //delete colIndicesPtr_;
        std::visit([this](const auto& exec)
               {exec.template clear<label>(this->colIndicesPtr_); }, csrMatExec_);
        
        colIndicesPtr_ = nullptr;
    }
}


//- Initialize matrix comunications for consolidation
void Foam::csrMatrix::initializeComms(label commId, bool gpuProc)
{
    gpuWorld_ = commId;

    gpuProc_ = gpuProc;

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


//- Initialize consolidation
void Foam::csrMatrix::initializeConsolidation
(
    const label nLocalRows,
    const label nLocalIntFaces,
    const label nLocalExtNz,
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
    // rowsConsDispPtr_ = new labelList(gpuWorldSize_ + 1, Foam::Zero);
    // intFacesConsDispPtr_ = new labelList(gpuWorldSize_ + 1, Foam::Zero);
    // extNzConsDispPtr_ = new labelList(gpuWorldSize_ + 1, Foam::Zero);
    std::visit([this](const auto& exec)
               { this->rowsConsDispPtr_ = exec.template allocZero<label>(this->gpuWorldSize_+1); },
               csrMatExec_);

    std::visit([this](const auto& exec)
               { this->intFacesConsDispPtr_ = exec.template allocZero<label>(this->gpuWorldSize_+1); },
               csrMatExec_);

    std::visit([this](const auto& exec)
               { this->extNzConsDispPtr_ = exec.template allocZero<label>(this->gpuWorldSize_+1); },
               csrMatExec_);

    labelList rowsConsDispTmp(gpuWorldSize_);
    rowsConsDispTmp.data()[myGpuWorldRank_] = nLocalRows;
    Pstream::allGatherList(rowsConsDispTmp, UPstream::msgType(), gpuWorld_);
    
    labelList intFacesConsDispTmp(gpuWorldSize_);
    intFacesConsDispTmp.data()[myGpuWorldRank_] = nLocalIntFaces;
    Pstream::allGatherList(intFacesConsDispTmp, UPstream::msgType(), gpuWorld_);
    
    labelList extNzConsDispTmp(gpuWorldSize_);
    extNzConsDispTmp.data()[myGpuWorldRank_] = nLocalExtNz;
    Pstream::allGatherList(extNzConsDispTmp, UPstream::msgType(), gpuWorld_);

    UPstream::barrier(UPstream::worldComm);

    for(label i=0; i<gpuWorldSize_; ++i)
    {
        rowsConsDispPtr_[i+1] = rowsConsDispPtr_[i] + rowsConsDispTmp.cdata()[i];
        intFacesConsDispPtr_[i+1] = intFacesConsDispPtr_[i] + intFacesConsDispTmp.cdata()[i];
        extNzConsDispPtr_[i+1] = extNzConsDispPtr_[i] + extNzConsDispTmp.cdata()[i];
    }

    nConsRows_ = rowsConsDispPtr_[gpuWorldSize_];
    nConsIntFaces_ = intFacesConsDispPtr_[gpuWorldSize_];
    nConsExtNz_ = extNzConsDispPtr_[gpuWorldSize_];
    nConsTotNz = nConsRows_ + 2*nConsIntFaces_ + nConsExtNz_;

    //- Consolidation of LDU adressing
    ownLst[myGpuWorldRank_] = own;
    Pstream::gatherList(ownLst, UPstream::msgType(), gpuWorld_);
    // MPI_Gather((void *) own, nLocalIntFaces, MPI_INT,
    //            (void *) ownLst, nLocalIntFaces, MPI_INT, 0,
    //            PstreamGlobals::MPICommunicators_[gpuWorld_]);
    
    neighLst[myGpuWorldRank_] = neigh;
    Pstream::gatherList(neighLst, UPstream::msgType(), gpuWorld_);
    // MPI_Gather((void *) neigh, nLocalIntFaces, MPI_INT,
    //            (void *) neighLst, nLocalIntFaces, MPI_INT, 0,
    //            PstreamGlobals::MPICommunicators_[gpuWorld_]);
    
    extRowsLst[myGpuWorldRank_] = extRows;
    Pstream::gatherList(extRowsLst, UPstream::msgType(), gpuWorld_);
    // MPI_Gather((void *) extRows, nLocalIntFaces, MPI_INT,
    //            (void *) extRowsLst, nLocalIntFaces, MPI_INT, 0,
    //            PstreamGlobals::MPICommunicators_[gpuWorld_]);

    extColsLst[myGpuWorldRank_] = extCols;
    Pstream::gatherList(extColsLst, UPstream::msgType(), gpuWorld_);
    // MPI_Gather((void *) extCols, nLocalIntFaces, MPI_INT,
    //            (void *) extColsLst, nLocalIntFaces, MPI_INT, 0,
    //            PstreamGlobals::MPICommunicators_[gpuWorld_]);

    //- Exchange of local to global offset
    //consDiagOffGlob = new labelList(gpuWorldSize_, Foam::Zero);
    consDiagOffGlob->data()[myGpuWorldRank_] = diagIndexGlobal;
    Pstream::gatherList(*consDiagOffGlob, UPstream::msgType(), gpuWorld_);

    //consLowOffGlob = new labelList(gpuWorldSize_);
    consLowOffGlob->data()[myGpuWorldRank_] = lowOffGlobal;
    Pstream::gatherList(*consLowOffGlob, UPstream::msgType(), gpuWorld_);

    //consUppOffGlob = new labelList(gpuWorldSize_);
    consUppOffGlob->data()[myGpuWorldRank_] = uppOffGlobal;
    Pstream::gatherList(*consUppOffGlob, UPstream::msgType(), gpuWorld_);

    consolidationStatus_ = ConsolidationStatus::initialized;

    UPstream::barrier(gpuWorld_);

    return;
}


void Foam::csrMatrix::initializeValuesConsolidation
(
    const label nLocalRows,
    const label nLocalIntFaces,
    const label nLocalExtVals,
    const scalarField& diag,
    const scalarField& upper,
    const scalarField& lower,
    const scalarField& extVal,
    List<scalarField>& diagLst,
    List<scalarField>& upperLst,
    List<scalarField>& lowerLst,
    List<scalarField>& extValLst
)
{
    diagLst[myGpuWorldRank_] = diag;
    Pstream::gatherList(diagLst, UPstream::msgType(), gpuWorld_);
    // MPI_Gather((void *) diag, nLocalRows, MPI_REAL,
    //            (void *) diagLst.cdata(), nLocalRows, MPI_REAL, 0,
    //            PstreamGlobals::MPICommunicators_[gpuWorld_]);
    
    upperLst[myGpuWorldRank_] = upper;
    Pstream::gatherList(upperLst, UPstream::msgType(), gpuWorld_);
    // MPI_Gather((void *) upper, nLocalIntFaces, MPI_REAL,
    //            (void *) upperLst.cdata(), nLocalIntFaces, MPI_REAL, 0,
    //            PstreamGlobals::MPICommunicators_[gpuWorld_]);

    lowerLst[myGpuWorldRank_] = lower;
    Pstream::gatherList(lowerLst, UPstream::msgType(), gpuWorld_);
    // MPI_Gather((void *) lower, nLocalIntFaces, MPI_REAL,
    //            (void *) lowerLst.cdata(), nLocalIntFaces, MPI_REAL, 0,
    //            PstreamGlobals::MPICommunicators_[gpuWorld_]);

    extValLst[myGpuWorldRank_] = extVal;
    Pstream::gatherList(extValLst, UPstream::msgType(), gpuWorld_);
    // MPI_Gather((void *) extVal, nLocalExtVals, MPI_REAL,
    //            (void *) extValLst.cdata(), nLocalExtVals, MPI_REAL, 0,
    //            PstreamGlobals::MPICommunicators_[gpuWorld_]);

    Pstream::barrier(gpuWorld_);
}


//- Find permutation array and new addressing vectors (no interface)
void Foam::csrMatrix::computePermutation(const lduAddressing * addr)
{
	const label* own = nullptr;
	const label* neigh = nullptr;

	const label* hostOwn = addr->lowerAddr().cdata();
	label ownSize = addr->lowerAddr().size();
	const label* hostNeigh = addr->upperAddr().cdata();
	label neighSize = addr->upperAddr().size();

	std::visit([&hostOwn, &own, ownSize](const auto& exec)
               { own = exec.template copyFromFoam<label>(ownSize,hostOwn); },
               csrMatExec_);
	std::visit([&hostNeigh, &neigh, neighSize](const auto& exec)
               { neigh = exec.template copyFromFoam<label>(neighSize,hostNeigh); },
               csrMatExec_);
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
               csrMatExec_);
    std::visit([this, totNnz](const auto& exec)
               { this->colIndicesPtr_ = exec.template alloc<label>(totNnz); },
               csrMatExec_);
    std::visit([this, totNnz](const auto& exec)
               { this->ldu2csrPerm_ = exec.template alloc<label>(totNnz); },
               csrMatExec_);

    label* rowIndices = nullptr;
    label* tmpPerm = nullptr;
	label* rowIndicesTmp = nullptr;
	label* colindicesTmp = nullptr;
    std::visit([&rowIndices, totNnz](const auto& exec)
               { rowIndices = exec.template alloc<label>(totNnz); },
               csrMatExec_);
    std::visit([&tmpPerm, totNnz](const auto& exec)
               { tmpPerm = exec.template alloc<label>(totNnz); },
               csrMatExec_);
    std::visit([&rowIndicesTmp, totNnz](const auto& exec)
               { rowIndicesTmp = exec.template alloc<label>(totNnz); },
               csrMatExec_);
    std::visit([&colindicesTmp, totNnz](const auto& exec)
               { colindicesTmp = exec.template alloc<label>(totNnz); },
               csrMatExec_);
    //labelList tmpPerm(totNnz);
    //labelList rowIndicesTmp(totNnz);
    //labelList colindicesTmp(totNnz);

    // Initialize: tmpPerm = [0, 1, ... totNnz-1]
    //             rowIndicesTmp = [0, ... nCells-1, (owner), (neighbour)]
    //             colindicesTmp = [0, ... nCells-1, (neighbour), (owner)]
    initializeSequence(totNnz, tmpPerm);
    initializeSequence(nCells, rowIndicesTmp);
    initializeAddressing
    (
        nCells,
        nIntFaces,
        nCells,
        nIntFaces,
        own,
        neigh,
        rowIndicesTmp,
        colindicesTmp
    );

    // Compute sorting to obtain permutation
    computeSorting
    (
        totNnz,
        tmpPerm,
        rowIndicesTmp,
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
    
    hasPermutation_ = true;
    
    std::visit([rowIndices](const auto& exec)
               {exec.template clear<label>(rowIndices); }, csrMatExec_);
    std::visit([tmpPerm](const auto& exec)
               {exec.template clear<label>(tmpPerm); }, csrMatExec_);
    std::visit([rowIndicesTmp](const auto& exec)
               {exec.template clear<label>(rowIndicesTmp); }, csrMatExec_);
    std::visit([colindicesTmp](const auto& exec)
               {exec.template clear<label>(colindicesTmp); }, csrMatExec_);
    std::visit([own](const auto& exec)
               {exec.template clear<label>(own); }, csrMatExec_);
    std::visit([neigh](const auto& exec)
               {exec.template clear<label>(neigh); }, csrMatExec_);
}


//- Find permutation array and new addressing vectors
void Foam::csrMatrix::computePermutation
(
    const lduAddressing& addr,
    const lduInterfacePtrsList& interfaces,
          label& nnzExt
)
{
    const labelList& hostOwn = addr.lowerAddr();
    const labelList& hostNeigh = addr.upperAddr();
    const label* own = nullptr;
	const label* neigh = nullptr;

	// const label* hostOwn = addr.lowerAddr().cdata();
	// const label* hostNeigh = addr.upperAddr().cdata();

	const label nIntFaces = addr.lowerAddr().size();
    const label nCells = addr.size();

	std::visit([&hostOwn, &own, nIntFaces](const auto& exec)
               { own = exec.template copyFromFoam<label>(nIntFaces,hostOwn.cdata()); },
               csrMatExec_);
	std::visit([&hostNeigh, &neigh, nIntFaces](const auto& exec)
               { neigh = exec.template copyFromFoam<label>(nIntFaces,hostNeigh.cdata()); },
               csrMatExec_);

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

    labelField foamExtRows(nnzExt, Foam::Zero);
    labelField foamExtCols(nnzExt, Foam::Zero);

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

            SubList<label>(foamExtRows, len, nnzExt) = faceCells;
            SubList<label>(foamExtCols, len, nnzExt) = nbrCells;
            nnzExt += len;
        }
    }

    const label* extRows = nullptr;
    const label* extCols = nullptr;
	std::visit([&foamExtRows, &extRows, nnzExt](const auto& exec)
               { extRows = exec.template copyFromFoam<label>(nnzExt,foamExtRows.cdata()); },
               csrMatExec_);
	std::visit([&foamExtCols, &extCols, nnzExt](const auto& exec)
               { extCols = exec.template copyFromFoam<label>(nnzExt,foamExtCols.cdata()); },
               csrMatExec_);

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
        consDiagOffGlob = new labelList(gpuWorldSize_);
        consLowOffGlob = new labelList(gpuWorldSize_);
        consUppOffGlob = new labelList(gpuWorldSize_);
        
        initializeConsolidation(nCells, nIntFaces, nnzExt, diagIndexGlobal, lowOffGlobal, uppOffGlobal,
                                hostOwn, hostNeigh, foamExtRows, foamExtCols, totNnz,
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
        nOwnerStart_ = nConsRows_ + 1;
        nLocalNz_ = totNnz;
        // ownerStartPtr_ = new labelList(nConsRows_+1, Foam::Zero);
        // ldu2csrPerm_ = new labelList(totNnz);
        // colIndicesPtr_ = new labelList(totNnz);
        std::visit([this](const auto& exec)
               { this->ownerStartPtr_ = exec.template allocZero<label>(this->nOwnerStart_); },
               csrMatExec_);
        std::visit([this, totNnz](const auto& exec)
               { this->ldu2csrPerm_ = exec.template alloc<label>(totNnz); },
               csrMatExec_);
        std::visit([this, totNnz](const auto& exec)
               { this->colIndicesPtr_ = exec.template alloc<label>(totNnz); },
               csrMatExec_);

        // labelList rowIndices(totNnz, Zero);
        // labelList tmpPerm(totNnz);
        // labelList rowIndicesTmp(totNnz);
        // labelList colIndicesTmp(totNnz);
        label* rowIndices = nullptr;
        label* tmpPerm = nullptr;
        label* rowIndicesTmp = nullptr;
        label* colIndicesTmp = nullptr;
        std::visit([&rowIndices, totNnz](const auto& exec)
               { rowIndices = exec.template allocZero<label>(totNnz); },
               csrMatExec_);
        std::visit([&tmpPerm, totNnz](const auto& exec)
               { tmpPerm = exec.template alloc<label>(totNnz); },
               csrMatExec_);
        std::visit([&rowIndicesTmp, totNnz](const auto& exec)
               { rowIndicesTmp = exec.template alloc<label>(totNnz); },
               csrMatExec_);
        std::visit([&colIndicesTmp, totNnz](const auto& exec)
               { colIndicesTmp = exec.template alloc<label>(totNnz); },
               csrMatExec_);

        // Initialize: tmpPerm = [0, 1, ... totNnz-1]
        //             rowIndicesTmp = [0, ... nCconsRow-1, (owner), (neighbour), (extrows)]
        //             colIndicesTmp = [(0, ... nRows0-1), ... (0, ... nRowsN-1), (neighbour), (owner), (extcols)]
        initializeSequence(totNnz, tmpPerm);
        initializeSequence(nConsRows_, rowIndicesTmp);
        
        if(consolidationStatus_ == ConsolidationStatus::initialized)
        {            
            for(label i=0; i<gpuWorldSize_; ++i)
            {
                initializeSequence(rowsConsDispPtr_[i+1] - rowsConsDispPtr_[i],
                                   &colIndicesTmp[rowsConsDispPtr_[i]]);

                initializeAddressingExt
                (
                    nConsRows_,
                    nConsIntFaces_,
                    rowsConsDispPtr_[i+1] - rowsConsDispPtr_[i],
                    intFacesConsDispPtr_[i+1] - intFacesConsDispPtr_[i],
                    extNzConsDispPtr_[i+1] - extNzConsDispPtr_[i],
                    ownLst[i].cdata(),
                    neighLst[i].cdata(),
                    extRowsLst[i].cdata(),
                    extColsLst[i].cdata(),
                    rowIndicesTmp,
                    colIndicesTmp,
                    rowsConsDispPtr_[i],
                    intFacesConsDispPtr_[i],
                    extNzConsDispPtr_[i]
                );

                localToConsRowIndex
                (
                    nConsRows_,
                    nConsIntFaces_,
                    intFacesConsDispPtr_[i+1] - intFacesConsDispPtr_[i],
                    extNzConsDispPtr_[i+1] - extNzConsDispPtr_[i],
                    intFacesConsDispPtr_[i],
                    extNzConsDispPtr_[i],
                    rowsConsDispPtr_[i],
                    rowIndicesTmp
                );
            }
        }
        else
        {
            initializeAddressingExt
            (
                nCells,
                nIntFaces,
                nCells,
                nIntFaces,
                nnzExt,
                own,
                neigh,
                extRows,
                extCols,
                rowIndicesTmp,
                colIndicesTmp
            );
        }       

        // Compute sorting to obtain permutation
        computeSorting
        (
            totNnz,
            tmpPerm,
            rowIndicesTmp,
            rowIndices,
            ldu2csrPerm_
        );

        // Make column indices from local to global
        if(consolidationStatus_ == ConsolidationStatus::initialized)
        {            
            for(label i=0; i<gpuWorldSize_; ++i)
            {
                localToGlobalColIndices
                (
                    nConsRows_,
                    nConsIntFaces_,
                    rowsConsDispPtr_[i+1] - rowsConsDispPtr_[i],
                    intFacesConsDispPtr_[i+1] - intFacesConsDispPtr_[i],
                    consDiagOffGlob->cdata()[i],
                    consLowOffGlob->cdata()[i],
                    consUppOffGlob->cdata()[i],
                    colIndicesTmp,
                    rowsConsDispPtr_[i],
                    intFacesConsDispPtr_[i]
                );
            }
        }
        else
        {
            localToGlobalColIndices
            (
                nConsRows_,
                nIntFaces,
                nCells,
                nIntFaces,
                diagIndexGlobal,
                lowOffGlobal,
                uppOffGlobal,
                colIndicesTmp        
            );
        }

        // Apply permutation vector to find colIndices + compute ownerStart
        applyAddressingPermutation
        (
            nConsRows_,
            totNnz,
            ldu2csrPerm_,
            colIndicesTmp,
            rowIndices,
            colIndicesPtr_,
            ownerStartPtr_
        );
    }
    
    UPstream::barrier(gpuWorld_);

    hasPermutation_ = true;
    
    if(consDiagOffGlob) delete consDiagOffGlob;
    if(consLowOffGlob) delete consLowOffGlob;
    if(consUppOffGlob) delete consUppOffGlob;
}


//- Apply permutation to LDU values (no permutation)
void Foam::csrMatrix::applyPermutation(const lduMatrix& lduMatrix)
{
    // Verify that the permutation has already been computed
    if(!ldu2csrPerm_)
    {
        computePermutation(&(lduMatrix.lduAddr()));
    }

    const scalar * foamDiag = lduMatrix.diag().cdata();
    const scalar * foamUpper = lduMatrix.upper().cdata();
    const scalar * foamLower = lduMatrix.lower().cdata();

    label nCells = lduMatrix.diag().size();
    label nIntFaces = lduMatrix.upper().size();
    label totNnz = nCells + 2*nIntFaces;

    const scalar * diag = nullptr;
    const scalar * upper = nullptr;
    const scalar * lower = nullptr;

    std::visit([&foamDiag, &diag, nCells](const auto& exec)
               { diag = exec.template copyFromFoam<scalar>(nCells, foamDiag); },
               csrMatExec_);
    std::visit([&foamUpper, &upper, nIntFaces](const auto& exec)
               { upper = exec.template copyFromFoam<scalar>(nIntFaces, foamUpper); },
               csrMatExec_);
    std::visit([&foamLower, &lower, nIntFaces](const auto& exec)
               { lower = exec.template copyFromFoam<scalar>(nIntFaces, foamLower); },
               csrMatExec_);

    if(!valuesPtr_)
    {
        // valuesPtr_ = new scalarField(totNnz);
        std::visit([this, totNnz](const auto& exec)
               { this->valuesPtr_ = exec.template alloc<scalar>(totNnz); },
               csrMatExec_);
    }

    // Initialize valuesTmp = [(diag), (upper), (lower)]
    // scalarField valuesTmp(totNnz);
    scalar* valuesTmp = nullptr;
    std::visit([&valuesTmp, totNnz](const auto& exec)
               { valuesTmp = exec.template alloc<scalar>(totNnz); },
               csrMatExec_);

    initializeValue
    (
        nCells,
        nIntFaces,
        nCells,
        nIntFaces,
        diag,
        upper,
        lower,
        valuesTmp
    );

    // Apply permutation
    applyValuePermutation
    (
        totNnz,
        ldu2csrPerm_,
        valuesTmp,
        valuesPtr_
    );

    std::visit([valuesTmp](const auto& exec)
               {exec.template clear<scalar>(valuesTmp); },
               csrMatExec_);
}


//- Apply permutation from LDU to CSR considering the interface values
void Foam::csrMatrix:: applyPermutation
(
    const lduMatrix& lduMatrix,
    const FieldField<Field, scalar> interfaceBouCoeffs,
          label& nGlobalCells
)
{
    label nnzExt = 0;
    const lduInterfacePtrsList& interfaces(lduMatrix.mesh().interfaces());

    // Verify that the permutation has already been computed
    if(!hasPermutation())
    {
        computePermutation
        (
            lduMatrix.lduAddr(),
            interfaces,
            nnzExt
        );
    }
    else
    {
        forAll(interfaces, patchi)
        {
            if (interfaces.set(patchi)) nnzExt += interfaceBouCoeffs[patchi].size();
        }
    }

    scalarField foamExtVals(nnzExt, Foam::Zero);

    nnzExt = 0;

    forAll(interfaces, patchi)
    {
        if (interfaces.set(patchi))
        {
            //- Processor-local values
            const scalarField& bCoeffs = interfaceBouCoeffs[patchi];
            const label len = bCoeffs.size();

            SubList<scalar>(foamExtVals, len, nnzExt) = bCoeffs;
            nnzExt += len;
        }
    }

    foamExtVals.negate();

    const scalarField& foamDiag = lduMatrix.diag();
    const scalarField& foamUpper = lduMatrix.upper();
    const scalarField& foamLower = lduMatrix.lower();

    label nIntFaces = lduMatrix.upper().size();
    label nCells = lduMatrix.diag().size();
    label totNnz;

    const scalar * diag = nullptr;
    const scalar * upper = nullptr;
    const scalar * lower = nullptr;
    const scalar * extVals = nullptr;

    std::visit([&foamDiag, &diag, nCells](const auto& exec)
               { diag = exec.template copyFromFoam<scalar>(nCells, foamDiag.cdata()); },
               csrMatExec_);
    std::visit([&foamUpper, &upper, nIntFaces](const auto& exec)
               { upper = exec.template copyFromFoam<scalar>(nIntFaces, foamUpper.cdata()); },
               csrMatExec_);
    std::visit([&foamLower, &lower, nIntFaces](const auto& exec)
               { lower = exec.template copyFromFoam<scalar>(nIntFaces, foamLower.cdata()); },
               csrMatExec_);
    std::visit([&foamExtVals, &extVals, nnzExt](const auto& exec)
               { extVals = exec.template copyFromFoam<scalar>(nnzExt, foamExtVals.cdata()); },
               csrMatExec_);

    //- Compute global number of equations
    nGlobalCells = returnReduce(nCells, sumOp<label>());

    List<scalarField> diagLst(gpuWorldSize_);
    List<scalarField> lowerLst(gpuWorldSize_);
    List<scalarField> upperLst(gpuWorldSize_);
    List<scalarField> extValLst(gpuWorldSize_);

    if(consolidationStatus_ == ConsolidationStatus::initialized)
    {
        initializeValuesConsolidation(nCells, nIntFaces, nnzExt,
                                      foamDiag, foamUpper, foamLower, foamExtVals,
                                      diagLst, upperLst, lowerLst, extValLst);
        
        totNnz = nConsRows_ + 2*nConsIntFaces_ + nConsExtNz_;
    }
    else
    {
        totNnz = nCells + 2*nIntFaces + nnzExt;
    }

    if(gpuProc_)
    {
        if(!valuesPtr_)
        {
            // valuesPtr_ = new scalarField(totNnz);
            std::visit([this, totNnz](const auto& exec)
                        { this->valuesPtr_ = exec.template alloc<scalar>(totNnz); },
                        csrMatExec_);
        }

        // Initialize valuesTmp = [(diag), (upper), (lower), (extValues)]
        // scalarField valuesTmp(totNnz);
        scalar* valuesTmp = nullptr;
        std::visit([&valuesTmp, totNnz](const auto& exec)
                   { valuesTmp = exec.template alloc<scalar>(totNnz); },
                    csrMatExec_);

        if(consolidationStatus_ == ConsolidationStatus::initialized)
        {
            for(label i=0; i<gpuWorldSize_; ++i)
            {
                initializeValueExt
                (
                    nConsRows_,
                    nConsIntFaces_,
                    rowsConsDispPtr_[i+1] - rowsConsDispPtr_[i],
                    intFacesConsDispPtr_[i+1] - intFacesConsDispPtr_[i],
                    extNzConsDispPtr_[i+1] - extNzConsDispPtr_[i],
                    diagLst[i].cdata(),
                    upperLst[i].cdata(),
                    lowerLst[i].cdata(),
                    extValLst[i].cdata(),
                    valuesTmp,
                    rowsConsDispPtr_[i],
                    intFacesConsDispPtr_[i],
                    extNzConsDispPtr_[i]
                );
            }
        }
        else
        {
            initializeValueExt
            (
                nCells,
                nIntFaces,
                nCells,
                nIntFaces,
                nnzExt,
                diag,
                upper,
                lower,
                extVals,
                valuesTmp
            );
        }
        
        // Apply permutation
        applyValuePermutation
        (
            totNnz,
            ldu2csrPerm_,
            valuesTmp,
            valuesPtr_
        );
    }
}


// * * * * * * * * * * * * * Explicit instantiations  * * * * * * * * * * * //

// ************************************************************************* //
