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



// * * * * * * * * * * * * * * * Constructors * * * * * * * * * * * * * * * //

Foam::csrMatrix::csrMatrix()
:
    csrAddressing(),
    valuesPtr_(nullptr)
{}

Foam::csrMatrix::csrMatrix(const csrMatrix& A)
:
    csrAddressing(A),
    valuesPtr_(nullptr)
{
    if (A.valuesPtr_)
    {
        valuesPtr_ = new scalarField(*(A.valuesPtr_));
    }
}


Foam::csrMatrix::csrMatrix(csrMatrix& A, bool reuse)
:
    csrAddressing(A, reuse),
    valuesPtr_(nullptr)
{
    if (reuse)
    {
        if (A.valuesPtr_)
        {
            valuesPtr_ = A.valuesPtr_;
            A.valuesPtr_ = nullptr;
        }
    }
    else
    {
        if (A.valuesPtr_)
        {
            valuesPtr_ = new scalarField(*(A.valuesPtr_));
        }
    }
}

// * * * * * * * * * * * *  Public Member Functions * * * * * * * * * * * *  //

void Foam::csrMatrix::finalize()
{
    if (valuesPtr_)
    {
        delete valuesPtr_;
    }
}


const Foam::scalarField& Foam::csrMatrix::values() const
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

//- Apply permutation to LDU values (no permutation)
void Foam::csrMatrix::applyPermutation(const lduMatrix& lduMatrix)
{
    // Verify that the permutation has already been computed
    if(!ldu2csrPerm_)
    {
        computePermutation(&(lduMatrix.lduAddr()));
    }

    const scalarField& diag = lduMatrix.diag();
    const scalarField& upper = lduMatrix.upper();
    const scalarField& lower = lduMatrix.lower();

    label nCells = diag.size();
    label nIntFaces = upper.size();
    label totNnz = nCells + 2*nIntFaces;

    if(!valuesPtr_)
    {
        valuesPtr_ = new scalarField(totNnz);
    }

    // Initialize valuesTmp = [(diag), (upper), (lower)]
    scalarField valuesTmp(totNnz);

    initializeValue
    (
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
        ldu2csrPerm_->cdata(),
        valuesTmp.cdata(),
        valuesPtr_->data()
    );
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

    scalarField extVals(nnzExt, Foam::Zero);

    nnzExt = 0;

    forAll(interfaces, patchi)
    {
        if (interfaces.set(patchi))
        {
            //- Processor-local values
            const scalarField& bCoeffs = interfaceBouCoeffs[patchi];
            const label len = bCoeffs.size();

            SubList<scalar>(extVals, len, nnzExt) = bCoeffs;
            nnzExt += len;
        }
    }

    extVals.negate();

    const scalarField& diag = lduMatrix.diag();
    const scalarField& upper = lduMatrix.upper();
    const scalarField& lower = lduMatrix.lower();

    label nIntFaces = upper.size();
    label nCells = diag.size();
    label totNnz = nCells + 2*nIntFaces + nnzExt;

    //- Compute global number of equations
    nGlobalCells = returnReduce(nCells, sumOp<label>());

    if(!valuesPtr_)
    {
        valuesPtr_ = new scalarField(totNnz);
    }

    // Initialize valuesTmp = [(diag), (upper), (lower), (extValues)]
    scalarField valuesTmp(totNnz);

    initializeValueExt
    (
        nCells,
        nIntFaces,
        nnzExt,
        diag.cdata(),
        upper.cdata(),
        lower.cdata(),
        extVals.cdata(),
        valuesTmp.data()
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
