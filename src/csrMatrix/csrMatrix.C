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

void Foam::csrMatrix::initializeValuesConsolidation
(
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
    
    upperLst[myGpuWorldRank_] = upper;
    Pstream::gatherList(upperLst, UPstream::msgType(), gpuWorld_);

    lowerLst[myGpuWorldRank_] = lower;
    Pstream::gatherList(lowerLst, UPstream::msgType(), gpuWorld_);

    extValLst[myGpuWorldRank_] = extVal;
    Pstream::gatherList(extValLst, UPstream::msgType(), gpuWorld_);

    Pstream::barrier(gpuWorld_);
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
        diag.cdata(),
        upper.cdata(),
        lower.cdata(),
        valuesTmp.data()
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

    const scalar * foamDiag = lduMatrix.diag().cdata();
    const scalar * foamUpper = lduMatrix.upper().cdata();
    const scalar * foamLower = lduMatrix.lower().cdata();

    label nIntFaces = upper.size();
    label nCells = diag.size();
    label totNnz;

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

    List<scalarField> diagLst(gpuWorldSize_);
    List<scalarField> lowerLst(gpuWorldSize_);
    List<scalarField> upperLst(gpuWorldSize_);
    List<scalarField> extValLst(gpuWorldSize_);

    if(consolidationStatus_ == ConsolidationStatus::initialized)
    {
        initializeValuesConsolidation(diag, upper, lower, extVals,
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
            valuesPtr_ = new scalarField(totNnz);
        }

        // Initialize valuesTmp = [(diag), (upper), (lower), (extValues)]
        scalarField valuesTmp(totNnz);

        if(consolidationStatus_ == ConsolidationStatus::initialized)
        {
            for(label i=0; i<gpuWorldSize_; ++i)
            {
                initializeValueExt
                (
                    nConsRows_,
                    nConsIntFaces_,
                    rowsConsDispPtr_->cdata()[i+1] - rowsConsDispPtr_->cdata()[i],
                    intFacesConsDispPtr_->cdata()[i+1] - intFacesConsDispPtr_->cdata()[i],
                    extNzConsDispPtr_->cdata()[i+1] - extNzConsDispPtr_->cdata()[i],
                    diagLst[i].cdata(),
                    upperLst[i].cdata(),
                    lowerLst[i].cdata(),
                    extValLst[i].cdata(),
                    valuesTmp.data(),
                    rowsConsDispPtr_->cdata()[i],
                    intFacesConsDispPtr_->cdata()[i],
                    extNzConsDispPtr_->cdata()[i]
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
                diag.cdata(),
                upper.cdata(),
                lower.cdata(),
                extVals.cdata(),
                valuesTmp.data()
            );
        }
        
        // Apply permutation
        applyValuePermutation
        (
            totNnz,
            ldu2csrPerm_->cdata(),
            valuesTmp.cdata(),
            valuesPtr_->data()
        );
    }
}


// * * * * * * * * * * * * * Explicit instantiations  * * * * * * * * * * * //

// ************************************************************************* //
