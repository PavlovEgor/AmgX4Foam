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
#include "global.cuh"

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
#ifdef have_cuda    
    else if (mode.starts_with("d"))
    {
        csrMatExec_ = cudaCsrMatrixExecutor();
	}
#endif
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
    
    if (hasPermutation_)
    {
        clearAddressing();
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

    if(psiCons_)
    {
        cudaFree(psiCons_);
    }

    if(rhsCons_)
    {
        cudaFree(rhsCons_);
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
        Info << "The consolidation is necessary" << nl;
    }
    else
    {
        consolidationStatus_ = ConsolidationStatus::notNecessary;
        // Info << "The consolidation is not necessary" << nl;
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
          label * &ownCons,
          label * &neighCons,
          label * &extRowsCons,
          label * &extColsCons
)
{   
    rowsConsDispPtr_ = new labelList(gpuWorldSize_ + 1, Foam::Zero);
    intFacesConsDispPtr_ = new labelList(gpuWorldSize_ + 1, Foam::Zero);
    extNzConsDispPtr_ = new labelList(gpuWorldSize_ + 1, Foam::Zero);

    labelList rowsPerProc(gpuWorldSize_);
    rowsPerProc.data()[myGpuWorldRank_] = nLocalRows;
    Pstream::allGatherList(rowsPerProc, UPstream::msgType(), gpuWorld_);
    
    labelList intFacesPerProc(gpuWorldSize_);
    intFacesPerProc.data()[myGpuWorldRank_] = nLocalIntFaces;
    Pstream::allGatherList(intFacesPerProc, UPstream::msgType(), gpuWorld_);
    
    labelList extNzPerProc(gpuWorldSize_);
    extNzPerProc.data()[myGpuWorldRank_] = nLocalExtNz;
    Pstream::allGatherList(extNzPerProc, UPstream::msgType(), gpuWorld_);

    UPstream::barrier(UPstream::worldComm);

    for(label i=0; i<gpuWorldSize_; ++i)
    {
        rowsConsDispPtr_->data()[i+1] = rowsConsDispPtr_->cdata()[i] + rowsPerProc.cdata()[i];
        intFacesConsDispPtr_->data()[i+1] = intFacesConsDispPtr_->cdata()[i] + intFacesPerProc.cdata()[i];
        extNzConsDispPtr_->data()[i+1] = extNzConsDispPtr_->cdata()[i] + extNzPerProc.cdata()[i];
    }

    nConsRows_ = rowsConsDispPtr_->last(); // [gpuWorldSize_];
    nConsIntFaces_ = intFacesConsDispPtr_->last(); // [gpuWorldSize_];
    nConsExtNz_ = extNzConsDispPtr_->last(); // [gpuWorldSize_];
    nConsTotNz = nConsRows_ + 2*nConsIntFaces_ + nConsExtNz_;

    //- Consolidation of LDU adressing
    List<labelList> ownLst(gpuWorldSize_);
    ownLst[myGpuWorldRank_] = own;
    Pstream::gatherList(ownLst, UPstream::msgType(), gpuWorld_);
        
    List<labelList> neighLst(gpuWorldSize_);
    neighLst[myGpuWorldRank_] = neigh;
    Pstream::gatherList(neighLst, UPstream::msgType(), gpuWorld_);
    
    List<labelList> extRowsLst(gpuWorldSize_);
    extRowsLst[myGpuWorldRank_] = extRows;
    Pstream::gatherList(extRowsLst, UPstream::msgType(), gpuWorld_);

    List<labelList> extColsLst(gpuWorldSize_);
    extColsLst[myGpuWorldRank_] = extCols;
    Pstream::gatherList(extColsLst, UPstream::msgType(), gpuWorld_);

    UPstream::barrier(gpuWorld_);

    if(gpuProc_)
    {
        std::visit([this, &ownCons](const auto& exec)
                { ownCons = exec.template alloc<label>(this->nConsIntFaces_); },
                csrMatExec_);
        std::visit([this, &ownLst, &ownCons](const auto& exec)
                { exec.template concatenate<label>(this->nConsIntFaces_, ownLst, ownCons); },
                csrMatExec_);

        std::visit([this, &neighCons](const auto& exec)
                { neighCons = exec.template alloc<label>(this->nConsIntFaces_); },
                csrMatExec_);
        std::visit([this, &neighLst, &neighCons](const auto& exec)
                { exec.template concatenate<label>(this->nConsIntFaces_, neighLst, neighCons); },
                csrMatExec_);

        std::visit([this, &extRowsCons](const auto& exec)
                { extRowsCons = exec.template alloc<label>(this->nConsExtNz_); },
                csrMatExec_);
        std::visit([this, &extRowsLst, &extRowsCons](const auto& exec)
                { exec.template concatenate<label>(this->nConsExtNz_, extRowsLst, extRowsCons); },
                csrMatExec_);

        std::visit([this, &extColsCons](const auto& exec)
                { extColsCons = exec.template alloc<label>(this->nConsExtNz_); },
                csrMatExec_);
        std::visit([this, &extColsLst, &extColsCons](const auto& exec)
                { exec.template concatenate<label>(this->nConsExtNz_, extColsLst, extColsCons); },
                csrMatExec_);
    }

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
    
    if(gpuProc_)
    {

        std::visit([this](const auto& exec)
               { this->psiCons_ = exec.template allocZero<scalar>(nBlocks_*nConsRows_);
                 this->rhsCons_ = exec.template allocZero<scalar>(nBlocks_*nConsRows_); },
               this->csrMatExec_);
        //cudaMalloc((void**) &psiCons_, sizeof(scalar)*nConsRows_*nBlocks_);
        //cudaMalloc((void**) &rhsCons_, sizeof(scalar)*nConsRows_*nBlocks_);

        cudaIpcGetMemHandle(&psiConsHandle_, psiCons_);
        cudaIpcGetMemHandle(&rhsConsHandle_, rhsCons_);
    }

    Pstream::broadcast((char*) &psiConsHandle_, sizeof(cudaIpcMemHandle_t), gpuWorld_, Pstream::masterNo());
    Pstream::broadcast((char*) &rhsConsHandle_, sizeof(cudaIpcMemHandle_t), gpuWorld_, Pstream::masterNo());
    
    if(!gpuProc_)
    {
        cudaIpcOpenMemHandle((void**) &psiCons_, psiConsHandle_, cudaIpcMemLazyEnablePeerAccess );
        cudaIpcOpenMemHandle((void**) &rhsCons_, rhsConsHandle_, cudaIpcMemLazyEnablePeerAccess );
    }
    return;
}


void Foam::csrMatrix::initializeValuesConsolidation
(
    const scalarField& diag,
    const scalarField& upper,
    const scalarField& lower,
    const scalarField& extVal,
    scalar*& diagCons,
    scalar*& upperCons,
    scalar*& lowerCons,
    scalar*& extValCons
)
{
    List<List<scalar>> diagLst(gpuWorldSize_);
    diagLst[myGpuWorldRank_] = diag;
    Pstream::gatherList(diagLst, UPstream::msgType(), gpuWorld_);
    
    List<List<scalar>> upperLst(gpuWorldSize_);
    upperLst[myGpuWorldRank_] = upper;
    Pstream::gatherList(upperLst, UPstream::msgType(), gpuWorld_);

    List<List<scalar>> lowerLst(gpuWorldSize_);
    lowerLst[myGpuWorldRank_] = lower;
    Pstream::gatherList(lowerLst, UPstream::msgType(), gpuWorld_);

    List<List<scalar>> extValLst(gpuWorldSize_);
    extValLst[myGpuWorldRank_] = extVal;
    Pstream::gatherList(extValLst, UPstream::msgType(), gpuWorld_);

    if(gpuProc_)
    {
        std::visit([this, &diagCons](const auto& exec)
                    { diagCons = exec.template alloc<scalar>(this->nConsRows_); },
                    csrMatExec_);
        std::visit([this, &diagLst, &diagCons](const auto& exec)
                    { exec.template concatenate<scalar>(this->nConsRows_, diagLst, diagCons); },
                    csrMatExec_);

        std::visit([this, &upperCons](const auto& exec)
                    { upperCons = exec.template alloc<scalar>(this->nConsIntFaces_); },
                    csrMatExec_);
        std::visit([this, &upperLst, &upperCons](const auto& exec)
                    { exec.template concatenate<scalar>(this->nConsIntFaces_, upperLst, upperCons); },
                    csrMatExec_);

        std::visit([this, &lowerCons](const auto& exec)
                    { lowerCons = exec.template alloc<scalar>(this->nConsIntFaces_); },
                    csrMatExec_);
        std::visit([this, &lowerLst, &lowerCons](const auto& exec)
                    { exec.template concatenate<scalar>(this->nConsIntFaces_, lowerLst, lowerCons); },
                    csrMatExec_);

        std::visit([this, &extValCons](const auto& exec)
                    { extValCons = exec.template alloc<scalar>(this->nConsExtNz_); },
                    csrMatExec_);
        std::visit([this, &extValLst, &extValCons](const auto& exec)
                    { exec.template concatenate<scalar>(this->nConsExtNz_, extValLst, extValCons); },
                    csrMatExec_);
    }
    

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
    nConsRows_ = nCells;

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
	label* colIndicesTmp = nullptr;
    std::visit([&rowIndices, totNnz](const auto& exec)
               { rowIndices = exec.template alloc<label>(totNnz); },
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
    //labelList tmpPerm(totNnz);
    //labelList rowIndicesTmp(totNnz);
    //labelList colIndicesTmp(totNnz);

    // Initialize: tmpPerm = [0, 1, ... totNnz-1]
    //             rowIndicesTmp = [0, ... nCells-1, (owner), (neighbour)]
    //             colIndicesTmp = [0, ... nCells-1, (neighbour), (owner)]
    initializeSequence(totNnz, tmpPerm);
    initializeSequence(nCells, rowIndicesTmp);
    initializeSequence(nCells, colIndicesTmp);

    initializeAddressing
    (
        nCells,
        nIntFaces,
        own,
        neigh,
        rowIndicesTmp,
        colIndicesTmp
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
        colIndicesTmp,
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
    std::visit([colIndicesTmp](const auto& exec)
               {exec.template clear<label>(colIndicesTmp); }, csrMatExec_);
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

	const label nIntFaces = addr.lowerAddr().size();
    const label nCells = addr.size();

    const globalIndex globalNumbering(nCells);

    const label diagIndexGlobal = globalNumbering.toGlobal(0);
    const label lowOffGlobal = globalNumbering.toGlobal(hostOwn.cdata()[0]) - hostOwn.cdata()[0];
    const label uppOffGlobal = globalNumbering.toGlobal(hostNeigh.cdata()[0]) - hostNeigh.cdata()[0];

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

    label totNnz;
    const label* own = nullptr;
	const label* neigh = nullptr;
    const label* extRows = nullptr;
    const label* extCols = nullptr;
    label* ownCons = nullptr;
	label* neighCons = nullptr;
    label* extRowsCons = nullptr;
    label* extColsCons = nullptr;
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
                                ownCons, neighCons, extRowsCons, extColsCons);
    }
    else
    {
        totNnz = nCells + 2*nIntFaces + nnzExt;
        nConsRows_ = nCells;
        nConsIntFaces_ = nIntFaces;
        nConsExtNz_ = nnzExt;

        std::visit([&hostOwn, &own, nIntFaces](const auto& exec)
                   { own = exec.template copyFromFoam<label>(nIntFaces,hostOwn.cdata()); },
                   csrMatExec_);
        std::visit([&hostNeigh, &neigh, nIntFaces](const auto& exec)
                   { neigh = exec.template copyFromFoam<label>(nIntFaces,hostNeigh.cdata()); },
                   csrMatExec_);
        std::visit([&foamExtRows, &extRows, nnzExt](const auto& exec)
                   { extRows = exec.template copyFromFoam<label>(nnzExt,foamExtRows.cdata()); },
                   csrMatExec_);
        std::visit([&foamExtCols, &extCols, nnzExt](const auto& exec)
                   { extCols = exec.template copyFromFoam<label>(nnzExt,foamExtCols.cdata()); },
                   csrMatExec_);
    }

    if(gpuProc_)
    {
        nOwnerStart_ = nConsRows_ + 1;
        nLocalNz_ = totNnz;
        std::visit([this](const auto& exec)
               { this->ownerStartPtr_ = exec.template allocZero<label>(this->nOwnerStart_); },
               csrMatExec_);
        std::visit([this, totNnz](const auto& exec)
               { this->ldu2csrPerm_ = exec.template alloc<label>(totNnz); },
               csrMatExec_);
        std::visit([this, totNnz](const auto& exec)
               { this->colIndicesPtr_ = exec.template alloc<label>(totNnz); },
               csrMatExec_);

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
            initializeAddressingExt
            (
                nConsRows_,
                nConsIntFaces_,
                nConsExtNz_,
                ownCons,
                neighCons,
                extRowsCons,
                extColsCons,
                rowIndicesTmp,
                colIndicesTmp
            );

            for(label i=0; i<gpuWorldSize_; ++i)
            {
                initializeSequence
                (
                    rowsConsDispPtr_->cdata()[i+1] - rowsConsDispPtr_->cdata()[i],
                    &colIndicesTmp[rowsConsDispPtr_->cdata()[i]]
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
                    rowIndicesTmp
                );
            }
        }
        else
        {
            initializeSequence(nConsRows_, colIndicesTmp);

            initializeAddressingExt
            (
                nConsRows_,
                nConsIntFaces_,
                nConsExtNz_,
                own,
                neigh,
                extRows,
                extCols,
                rowIndicesTmp,
                colIndicesTmp
            );
        }

        //- Compute sorting to obtain permutation
        computeSorting
        (
            totNnz,
            tmpPerm,
            rowIndicesTmp,
            rowIndices,
            ldu2csrPerm_
        );

        //- Make column indices from local to global
        if(consolidationStatus_ == ConsolidationStatus::initialized)
        {            
            for(label i=0; i<gpuWorldSize_; ++i)
            {
                localToGlobalColIndices
                (
                    nConsRows_,
                    nConsIntFaces_,
                    rowsConsDispPtr_->cdata()[i+1] - rowsConsDispPtr_->cdata()[i],
                    intFacesConsDispPtr_->cdata()[i+1] - intFacesConsDispPtr_->cdata()[i],
                    consDiagOffGlob->cdata()[i],
                    consLowOffGlob->cdata()[i],
                    consUppOffGlob->cdata()[i],
                    colIndicesTmp,
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
                nIntFaces,
                nCells,
                nIntFaces,
                diagIndexGlobal,
                lowOffGlobal,
                uppOffGlobal,
                colIndicesTmp        
            );
        }

        //- Apply permutation vector to find colIndices + compute ownerStart
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

        std::visit([rowIndices](const auto& exec)
                    {exec.template clear<label>(rowIndices); }, csrMatExec_);
        std::visit([tmpPerm](const auto& exec)
                    {exec.template clear<label>(tmpPerm); }, csrMatExec_);
        std::visit([rowIndicesTmp](const auto& exec)
                    {exec.template clear<label>(rowIndicesTmp); }, csrMatExec_);
        std::visit([colIndicesTmp](const auto& exec)
                    {exec.template clear<label>(colIndicesTmp); }, csrMatExec_);
        if(consolidationStatus_ == ConsolidationStatus::initialized)
        {
            std::visit([ownCons](const auto& exec)
                        {exec.template clear<label>(ownCons); }, csrMatExec_);
            std::visit([neighCons](const auto& exec)
                        {exec.template clear<label>(neighCons); }, csrMatExec_);
            std::visit([extRowsCons](const auto& exec)
                        {exec.template clear<label>(extRowsCons); }, csrMatExec_);
            std::visit([extColsCons](const auto& exec)
                        {exec.template clear<label>(extColsCons); }, csrMatExec_);
        }
        else
        {
            std::visit([own](const auto& exec)
                        {exec.template clear<label>(own); }, csrMatExec_);
            std::visit([neigh](const auto& exec)
                        {exec.template clear<label>(neigh); }, csrMatExec_);
            std::visit([extRows](const auto& exec)
                        {exec.template clear<label>(extRows); }, csrMatExec_);
            std::visit([extCols](const auto& exec)
                        {exec.template clear<label>(extCols); }, csrMatExec_);
        }
        
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

    std::visit([diag](const auto& exec)
                {exec.template clear<scalar>(diag); }, csrMatExec_);
    std::visit([upper](const auto& exec)
                {exec.template clear<scalar>(upper); }, csrMatExec_);
    std::visit([lower](const auto& exec)
                {exec.template clear<scalar>(lower); }, csrMatExec_);
    std::visit([valuesTmp](const auto& exec)
                {exec.template clear<scalar>(valuesTmp); }, csrMatExec_);
}


//- Apply permutation from LDU to CSR considering the interface values
void Foam::csrMatrix::applyPermutation
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

    //- Compute global number of equations
    nGlobalCells = returnReduce(nCells, sumOp<label>());

    const scalar * diag = nullptr;
    const scalar * upper = nullptr;
    const scalar * lower = nullptr;
    const scalar * extVals = nullptr;
    scalar * diagCons = nullptr;
    scalar * upperCons = nullptr;
    scalar * lowerCons = nullptr;
    scalar * extValsCons = nullptr;

    if(consolidationStatus_ == ConsolidationStatus::initialized)
    {
        initializeValuesConsolidation(foamDiag, foamUpper, foamLower, foamExtVals,
                                      diagCons, upperCons, lowerCons, extValsCons);

        totNnz = nConsRows_ + 2*nConsIntFaces_ + nConsExtNz_;
    }
    else
    {
        totNnz = nCells + 2*nIntFaces + nnzExt;

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
            initializeValueExt
            (
                nConsRows_,
                nConsIntFaces_,
                nConsExtNz_,
                diagCons,
                upperCons,
                lowerCons,
                extValsCons,
                valuesTmp
            );
        }
        else
        {
            initializeValueExt
            (
                nConsRows_,
                nConsIntFaces_,
                nConsExtNz_,
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

        std::visit([valuesTmp](const auto& exec)
                        {exec.template clear<scalar>(valuesTmp); }, csrMatExec_);
        
        if(consolidationStatus_ == ConsolidationStatus::initialized)
        {
            std::visit([diagCons](const auto& exec)
                        {exec.template clear<scalar>(diagCons); }, csrMatExec_);
            std::visit([upperCons](const auto& exec)
                        {exec.template clear<scalar>(upperCons); }, csrMatExec_);
            std::visit([lowerCons](const auto& exec)
                        {exec.template clear<scalar>(lowerCons); }, csrMatExec_);
            std::visit([extValsCons](const auto& exec)
                        {exec.template clear<scalar>(extValsCons); }, csrMatExec_);
        }
        else
        {
            std::visit([diag](const auto& exec)
                        {exec.template clear<scalar>(diag); }, csrMatExec_);
            std::visit([upper](const auto& exec)
                        {exec.template clear<scalar>(upper); }, csrMatExec_);
            std::visit([lower](const auto& exec)
                        {exec.template clear<scalar>(lower); }, csrMatExec_);
            std::visit([extVals](const auto& exec)
                        {exec.template clear<scalar>(extVals); }, csrMatExec_);
        }
    }
}


void Foam::csrMatrix::createConsVect
(
    const scalarField& rhs,
          scalarField& psi
)
{
    if(isConsolidated())
    {
        label consDispl = rowsConsDispPtr_->cdata()[myGpuWorldRank_];
        cudaMemcpy(&psiCons_[consDispl], psi.cdata(), psi.size()*sizeof(scalar), cudaMemcpyHostToDevice );
        cudaMemcpy(&rhsCons_[consDispl], rhs.cdata(), rhs.size()*sizeof(scalar), cudaMemcpyHostToDevice );
    }
    else
    {
        rhsCons_ = const_cast<scalar*>(rhs.cdata());
        psiCons_ = psi.data();
    }
}


void Foam::csrMatrix::distributeSolution(scalarField& psi)
{
    if(isConsolidated())
    {
        label consDispl = rowsConsDispPtr_->cdata()[myGpuWorldRank_];
        cudaMemcpy(psi.data(), &psiCons_[consDispl], psi.size()*sizeof(scalar), cudaMemcpyDeviceToHost);
    }
}
// * * * * * * * * * * * * * Explicit instantiations  * * * * * * * * * * * //

// ************************************************************************* //
