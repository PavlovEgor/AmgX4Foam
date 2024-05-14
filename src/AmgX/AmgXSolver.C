/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2016-2022 OpenCFD Ltd.
    Copyright (C) 2022-2023 Cineca
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

#include "AmgXSolver.H"
#include "direction.H"
#include "AmgXLinearSolverContext.H"
#include "linearSolverContextTable.H"
#include "csrMatrix.H"

#include "globalIndex.H"

// #include <iostream>
// #include <fstream>

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(AmgXSolver, 0);

    lduMatrix::solver::addsymMatrixConstructorToTable<AmgXSolver>
        addAmgXSolverSymMatrixConstructorToTable_;

    lduMatrix::solver::addasymMatrixConstructorToTable<AmgXSolver>
        addAmgXSolverAsymMatrixConstructorToTable_;
}

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::AmgXSolver::AmgXSolver
(
    const word& fieldName,
    const lduMatrix& matrix,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const FieldField<Field, scalar>& interfaceIntCoeffs,
    const lduInterfaceFieldPtrsList& interfaces,
    const dictionary& solverControls
)
:
    lduMatrix::solver
    (
        fieldName,
        matrix,
        interfaceBouCoeffs,
        interfaceIntCoeffs,
        interfaces,
        solverControls
    ),
    eqName_(fieldName)
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

//- construct the matrix for AmgX
/*void Foam::AmgXSolver::buildAndApplyMatrixPermutation
(
    deviceCsrMatrix* csrMatrix,
    label& nRowsGlobal
) const
{
    const lduInterfacePtrsList interfaces(this->matrix_.mesh().openFoamMesh().interfaces());

    // Local degrees-of-freedom i.e. number of local rows
    const label nLocalRows = this->matrix_.mesh().nCells();
    label nRowsLocal = nLocalRows;
    nRowsGlobal = returnReduce(nRowsLocal, sumOp<label>());

    // Number of internal faces (connectivity)
    const label nIntFaces = this->matrix_.mesh().nInternalFaces();

    const globalIndex globalNumbering(nLocalRows);

    const label diagIndexGlobal = globalNumbering.toGlobal(0);
    const label firstLowerInd = this->matrix_.mesh().openFoamMesh().owner()[0]; //lower.cdata()[0];
    label lowOffGlobal = globalNumbering.toGlobal(firstLowerInd) - firstLowerInd;
    const label firstUpperInd = this->matrix_.mesh().openFoamMesh().neighbour()[0]; // upper.cdata()[0];
    label uppOffGlobal = globalNumbering.toGlobal(firstUpperInd) - firstUpperInd;

    labelList globalCells
    (
        identity
        (
            globalNumbering.localSize(),
            globalNumbering.localStart()
        )
    );

    // Connections to neighbouring processors
    const label nReq = Pstream::nRequests();

    label nProcValues = 0;

    // Initialise transfer of global cells
    forAll(interfaces, patchi)
    {
        if (interfaces.set(patchi))
        {
            nProcValues += interfaces[patchi].faceCells().size(); //lduAddr.patchAddr(patchi).size();

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

    deviceField<label> procRows(nProcValues, 0);
    deviceField<label> procCols(nProcValues, 0);
    deviceField<scalar> procVals(nProcValues, Foam::Zero);
    nProcValues = 0;

    forAll(interfaces, patchi)
    {
        if (interfaces.set(patchi))
        {
            // Processor-local values
            const label len = interfaces[patchi].faceCells().size();
            const deviceField<label> faceCells(len, interfaces[patchi].faceCells().cdata());
            const deviceField<scalar>& bCoeffs = this->interfaceBouCoeffs_[patchi];

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

            procRows.copy(faceCells, nProcValues, 0);
            procCols.copyIn(nbrCells, len, nProcValues);
            procVals.copy(bCoeffs, nProcValues, 0);

            nProcValues += len;
        }
    }

    procVals.negate();  // Change sign for entire field (see previous note)

    csrMatrix->applyPermutation
    (
        this->matrix_,
        diagIndexGlobal,
        lowOffGlobal,
        uppOffGlobal,
        procRows,
        procCols,
        procVals
    );

    DebugInfo<< "Converted LDU matrix to CSR format" << nl;

}


//- construct the matrix for AmgX
void Foam::AmgXSolver::applyMatrixPermutation
(
    deviceCsrMatrix* csrMatrix,
    label& nRowsGlobal
) const
{
    const UPtrList<const devicelduInterfaceField>& interfaces = this->interfaces_;

    // Local degrees-of-freedom i.e. number of local rows
    const label nLocalRows = this->matrix_.mesh().nCells();
    label nRowsLocal = nLocalRows;
    nRowsGlobal = returnReduce(nRowsLocal, sumOp<label>());

    //- Number of internal faces (connectivity)
    const label nIntFaces = this->matrix_.mesh().nInternalFaces();

    //- Connections to neighbouring processors
    const label nReq = Pstream::nRequests();

    label nProcValues = 0;

    // Initialise transfer of global cells
    forAll(interfaces, patchi)
    {
        if (interfaces.set(patchi)) nProcValues += this->interfaceBouCoeffs_[patchi].size();
    }

    if (Pstream::parRun())
    {
        Pstream::waitRequests(nReq);
    }

    deviceField<scalar> procVals(nProcValues, Foam::Zero);
    nProcValues = 0;

    forAll(interfaces, patchi)
    {
        if (interfaces.set(patchi))
        {
            //- Processor-local values
            const deviceField<scalar>& bCoeffs = this->interfaceBouCoeffs_[patchi];
            const label len = bCoeffs.size();

            procVals.copy(bCoeffs, nProcValues, 0);

            nProcValues += len;
        }
    }

    procVals.negate();  // Change sign for entire field (see previous note)

    csrMatrix->applyPermutation
    (
        this->matrix_,
        procVals
    );

    DebugInfo<< "Converted LDU matrix values to CSR format" << nl;

}*/


Foam::solverPerformance Foam::AmgXSolver::solve
(
    solveScalarField& psi,
    const scalarField& source,
    const direction cmpt
) const
{
    solverPerformance solverPerf
    (
        "AmgX", //lduMatrix::preconditioner::getName(controlDict_) + typeName,
        this->fieldName_
    );

    const fvMesh& fvm = dynamicCast<const fvMesh>(this->matrix_.mesh().thisDb());
    
    label nCells = psi.size();

    const linearSolverContextTable<AmgXLinearSolverContext<csrMatrix>>& contexts =
        linearSolverContextTable<AmgXLinearSolverContext<csrMatrix>>::New(fvm);

    AmgXLinearSolverContext<csrMatrix>& ctx = contexts.getContext(eqName_);

    if (!ctx.loaded())
    {
        FatalErrorInFunction
            << "Could not initialize AMGx" << nl << abort(FatalError);
    }

    ctx.performance = solverPerf;

    AmgXWrapper& amgx = ctx.amgx_;
    
    csrMatrix& Amat = ctx.Amat_;

    label nGlobalCells;

    if(!Pstream::parRun())
    {
        nGlobalCells = nCells;
        Amat.applyPermutation(this->matrix_);
    }
    /*else
    {
        if(!Amat.hasPermutation()) buildAndApplyMatrixPermutation(&Amat, nGlobalCells);
        else applyMatrixPermutation(&Amat, nGlobalCells);
    }*/

    label nnz = Amat.values().size();

    //- Print matrix converted to check
    /*string fileName = "csrMatrix-cpu";
    std::ofstream outFile(fileName, std::ios_base::app);
    outFile << "ownerStart:" << nl;
    for(int i=0; i< nCells; ++i) outFile << Amat.ownerStart().cdata()[i] << nl;
    outFile << nl << "colIndeces:" << nl;
    for(int i=0; i< nnz; ++i) outFile << Amat.colIndices().cdata()[i] << nl;
    outFile << nl << "colIndeces:" << nl;
    for(int i=0; i< nnz; ++i) outFile << Amat.values().cdata()[i] << nl;
    outFile.close();*/

    if(!ctx.initialized())
    {
        Info<< "Initializing AmgX Linear Solver " << eqName_ << nl;

        amgx.setOperator(nGlobalCells, &Amat);

        // Amat.clearAddressing();

        ctx.initialized() = true;
    }
    else
    {
        amgx.updateOperator(&Amat);
    }

    amgx.solve(psi.data(), source.cdata(), &Amat);

    scalarField iNorm(1, 0.0);
    amgx.getResidual(0, iNorm);
    ctx.performance.initialResidual() = iNorm[0];

    label nIters = 0;
    amgx.getIters(nIters);
    ctx.performance.nIterations() = nIters;

    scalarField fNorm(1, 0.0);
    amgx.getResidual(nIters, fNorm);
    ctx.performance.finalResidual() = fNorm[0];

    return ctx.performance;
    
}

// * * * * * * * * * * * * * Explicit instantiations  * * * * * * * * * * * //



// ************************************************************************* //
