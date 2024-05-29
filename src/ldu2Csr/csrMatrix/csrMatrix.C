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

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

#ifdef have_cuda
Foam::csrMatrixExecutor Foam::csrMatrix::csrMatExec_ = cudaCsrMatrixExecutor();
#else
Foam::csrMatrixExecutor Foam::csrMatrix::csrMatExec_ = cpuCsrMatrixExecutor();
#endif

// * * * * * * * * * * * * * * * Constructors * * * * * * * * * * * * * * * //

Foam::csrMatrix::csrMatrix(word mode)
:
    csrAddressing(mode),
    valuesPtr_(nullptr)
{}

//Foam::csrMatrix::csrMatrix(const csrMatrix& A)
//:
//    csrAddressing(A),
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
//    csrAddressing(A, reuse),
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
    if (valuesPtr_)
    {
        // delete valuesPtr_;
        std::visit([this](const auto& exec){exec.template clear<scalar>(this->valuesPtr_); }, csrAddrExec_);
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
               csrAddrExec_);
    std::visit([&foamUpper, &upper, nIntFaces](const auto& exec)
               { upper = exec.template copyFromFoam<scalar>(nIntFaces, foamUpper); },
               csrAddrExec_);
    std::visit([&foamLower, &lower, nIntFaces](const auto& exec)
               { lower = exec.template copyFromFoam<scalar>(nIntFaces, foamLower); },
               csrAddrExec_);

    if(!valuesPtr_)
    {
        // valuesPtr_ = new scalarField(totNnz);
        std::visit([this, totNnz](const auto& exec)
               { this->valuesPtr_ = exec.template alloc<scalar>(totNnz); },
               csrAddrExec_);
    }

    // Initialize valuesTmp = [(diag), (upper), (lower)]
    // scalarField valuesTmp(totNnz);
    scalar* valuesTmp = nullptr;
    std::visit([&valuesTmp, totNnz](const auto& exec)
               { valuesTmp = exec.template alloc<scalar>(totNnz); },
               csrAddrExec_);

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
               csrAddrExec_);
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
    label totNnz = nCells + 2*nIntFaces;

    const scalar * diag = nullptr;
    const scalar * upper = nullptr;
    const scalar * lower = nullptr;
    const scalar * extVals = nullptr;

    std::visit([&foamDiag, &diag, nCells](const auto& exec)
               { diag = exec.template copyFromFoam<scalar>(nCells, foamDiag); },
               csrAddrExec_);
    std::visit([&foamUpper, &upper, nIntFaces](const auto& exec)
               { upper = exec.template copyFromFoam<scalar>(nIntFaces, foamUpper); },
               csrAddrExec_);
    std::visit([&foamLower, &lower, nIntFaces](const auto& exec)
               { lower = exec.template copyFromFoam<scalar>(nIntFaces, foamLower); },
               csrAddrExec_);
    std::visit([&foamExtVals, &extVals, nnzExt](const auto& exec)
               { extVals = exec.template copyFromFoam<scalar>(nnzExt, foamExtVals.cdata()); },
               csrAddrExec_);

    //- Compute global number of equations
    nGlobalCells = returnReduce(nCells, sumOp<label>());

    if(!valuesPtr_)
    {
        // valuesPtr_ = new scalarField(totNnz);
        std::visit([this, totNnz](const auto& exec)
               { this->valuesPtr_ = exec.template alloc<scalar>(totNnz); },
               csrAddrExec_);
    }

    // Initialize valuesTmp = [(diag), (upper), (lower), (extValues)]
    // scalarField valuesTmp(totNnz);
    scalar* valuesTmp = nullptr;
    std::visit([&valuesTmp, totNnz](const auto& exec)
               { valuesTmp = exec.template alloc<scalar>(totNnz); },
               csrAddrExec_);

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
               csrAddrExec_);
}


// * * * * * * * * * * * * * Explicit instantiations  * * * * * * * * * * * //

// ************************************************************************* //
