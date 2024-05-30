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

#include "csrMatrix.H"

#include "globalIndex.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

// * * * * * * * * * * * * * * * Constructors * * * * * * * * * * * * * * * //

Foam::csrMatrix::csrMatrix(word mode)
:
    ownerStartPtr_(nullptr),
    colIndicesPtr_(nullptr),
    ldu2csrPerm_(nullptr),
    valuesPtr_(nullptr)
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
               {exec.template clear<label>(this->ldu2csrPerm_); }, csrMatExec_);
    }

    if (valuesPtr_)
    {
        // delete valuesPtr_;
        std::visit([this](const auto& exec)
                {exec.template clear<scalar>(this->valuesPtr_); }, csrMatExec_);
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
    }

    if (colIndicesPtr_)
    {
        //delete colIndicesPtr_;
        std::visit([this](const auto& exec)
               {exec.template clear<label>(this->colIndicesPtr_); }, csrMatExec_);
    }
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
	label* rowindicesTmp = nullptr;
	label* colindicesTmp = nullptr;
    std::visit([&rowIndices, totNnz](const auto& exec)
               { rowIndices = exec.template alloc<label>(totNnz); },
               csrMatExec_);
    std::visit([&tmpPerm, totNnz](const auto& exec)
               { tmpPerm = exec.template alloc<label>(totNnz); },
               csrMatExec_);
    std::visit([&rowindicesTmp, totNnz](const auto& exec)
               { rowindicesTmp = exec.template alloc<label>(totNnz); },
               csrMatExec_);
    std::visit([&colindicesTmp, totNnz](const auto& exec)
               { colindicesTmp = exec.template alloc<label>(totNnz); },
               csrMatExec_);
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
               {exec.template clear<label>(rowIndices); }, csrMatExec_);
    std::visit([tmpPerm](const auto& exec)
               {exec.template clear<label>(tmpPerm); }, csrMatExec_);
    std::visit([rowindicesTmp](const auto& exec)
               {exec.template clear<label>(rowindicesTmp); }, csrMatExec_);
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
               csrMatExec_);
	std::visit([&hostNeigh, &neigh, neighSize](const auto& exec)
               { neigh = exec.template copyFromFoam<label>(neighSize,hostNeigh); },
               csrMatExec_);

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
               csrMatExec_);
	std::visit([&extColsPtr, &extDevCols, nnzExt](const auto& exec)
               { extDevCols = exec.template copyFromFoam<label>(nnzExt,extColsPtr); },
               csrMatExec_);

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
               csrMatExec_);
    std::visit([this, totNnz](const auto& exec)
               { this->colIndicesPtr_ = exec.template alloc<label>(totNnz); },
               csrMatExec_);
    std::visit([this, totNnz](const auto& exec)
               { this->ldu2csrPerm_ = exec.template alloc<label>(totNnz); },
               csrMatExec_);

    label* rowIndices = nullptr;
    label* tmpPerm = nullptr;
	label* rowindicesTmp = nullptr;
	label* colindicesTmp = nullptr;

    std::visit([&rowIndices, totNnz](const auto& exec)
               { rowIndices = exec.template alloc<label>(totNnz); },
               csrMatExec_);
    std::visit([&tmpPerm, totNnz](const auto& exec)
               { tmpPerm = exec.template alloc<label>(totNnz); },
               csrMatExec_);
    std::visit([&rowindicesTmp, totNnz](const auto& exec)
               { rowindicesTmp = exec.template alloc<label>(totNnz); },
               csrMatExec_);
    std::visit([&colindicesTmp, totNnz](const auto& exec)
               { colindicesTmp = exec.template alloc<label>(totNnz); },
               csrMatExec_);

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
               {exec.template clear<label>(rowIndices); }, csrMatExec_);
    std::visit([tmpPerm](const auto& exec)
               {exec.template clear<label>(tmpPerm); }, csrMatExec_);
    std::visit([rowindicesTmp](const auto& exec)
               {exec.template clear<label>(rowindicesTmp); }, csrMatExec_);
    std::visit([colindicesTmp](const auto& exec)
               {exec.template clear<label>(colindicesTmp); }, csrMatExec_);
    std::visit([own](const auto& exec)
               {exec.template clear<label>(own); }, csrMatExec_);
    std::visit([neigh](const auto& exec)
               {exec.template clear<label>(neigh); }, csrMatExec_);
    std::visit([extDevRows](const auto& exec)
               {exec.template clear<label>(extDevRows); }, csrMatExec_);
    std::visit([extDevCols](const auto& exec)
               {exec.template clear<label>(extDevCols); }, csrMatExec_);
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
    if(!ldu2csrPerm_)
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

    const scalar * foamDiag = lduMatrix.diag().cdata();
    const scalar * foamUpper = lduMatrix.upper().cdata();
    const scalar * foamLower = lduMatrix.lower().cdata();

    label nCells = lduMatrix.diag().size();
    label nIntFaces = lduMatrix.upper().size();
    label totNnz = nCells + 2*nIntFaces + nnzExt;

    const scalar * diag = nullptr;
    const scalar * upper = nullptr;
    const scalar * lower = nullptr;
    const scalar * extVals = nullptr;

    std::visit([&foamDiag, &diag, nCells](const auto& exec)
               { diag = exec.template copyFromFoam<scalar>(nCells, foamDiag); },
               csrMatExec_);
    std::visit([&foamUpper, &upper, nIntFaces](const auto& exec)
               { upper = exec.template copyFromFoam<scalar>(nIntFaces, foamUpper); },
               csrMatExec_);
    std::visit([&foamLower, &lower, nIntFaces](const auto& exec)
               { lower = exec.template copyFromFoam<scalar>(nIntFaces, foamLower); },
               csrMatExec_);
    std::visit([&foamExtVals, &extVals, nnzExt](const auto& exec)
               { extVals = exec.template copyFromFoam<scalar>(nnzExt, foamExtVals.cdata()); },
               csrMatExec_);

    //- Compute global number of equations
    nGlobalCells = returnReduce(nCells, sumOp<label>());

    if(!valuesPtr_)
    {
        // valuesPtr_ = new scalarField(totNnz);
        std::visit([this, totNnz](const auto& exec)
               { this->valuesPtr_ = exec.template alloc<scalar>(totNnz); },
               csrMatExec_);
    }

    //- Initialize valuesTmp = [(diag), (upper), (lower), (extValues)]
    // scalarField valuesTmp(totNnz);
    scalar* valuesTmp = nullptr;
    std::visit([&valuesTmp, totNnz](const auto& exec)
               { valuesTmp = exec.template alloc<scalar>(totNnz); },
               csrMatExec_);

    initializeValueExt
    (
        nCells,
        nIntFaces,
        nnzExt,
        diag,
        upper,
        lower,
        extVals,
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
    std::visit([diag](const auto& exec)
               {exec.template clear<scalar>(diag); },
               csrMatExec_);
    std::visit([upper](const auto& exec)
               {exec.template clear<scalar>(upper); },
               csrMatExec_);
    std::visit([lower](const auto& exec)
               {exec.template clear<scalar>(lower); },
               csrMatExec_);
    std::visit([extVals](const auto& exec)
               {exec.template clear<scalar>(extVals); },
               csrMatExec_);
}


// * * * * * * * * * * * * * Explicit instantiations  * * * * * * * * * * * //

// ************************************************************************* //
