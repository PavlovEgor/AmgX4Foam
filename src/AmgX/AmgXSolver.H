/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2012 OpenFOAM Foundation
    Copyright (C) 2019 OpenCFD Ltd.
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

Class
    Foam::AmgXSolver

Group
    grpLduMatrixSolvers

Description
    Interface to the external package AmgX (https://github.com/NVIDIA/AMGX).

SourceFiles
    AmgXSolver.C

\*---------------------------------------------------------------------------*/

#ifndef Foam_AmgXSolver_H
#define Foam_AmgXSolver_H

#include "lduMatrix.H"
#include "csrMatrix.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                             Class AmgXSolver Declaration
\*---------------------------------------------------------------------------*/

class AmgXSolver
:
    public lduMatrix::solver
{
    // Private Member Data

        //- Equation name for solver context
        word eqName_;        

        //- Cached preconditioner
        // mutable autoPtr<lduMatrix::preconditioner> preconPtr_;


    // Private Member Functions

        //- No copy construct
        AmgXSolver(const AmgXSolver&) = delete;

        //- No copy assignment
        void operator=(const AmgXSolver&) = delete;

        //- Compute matrix addressing and value permutation for AmgX use
        void buildAndApplyMatrixPermutation
        (
            csrMatrix& csrMatrix,
            label& nRowsGlobal
        ) const;

        //- Compute matrix value permutation for AmgX use
        void applyMatrixPermutation
        (
            csrMatrix& csrMatrix,
            label& nRowsGlobal
        ) const;


public:

    //- Runtime type information
    TypeName("AmgX");


    // Constructors

        //- Construct from matrix components and solver controls
        AmgXSolver
        (
            const word& fieldName,
            const lduMatrix& matrix,
            const FieldField<Field, scalar>& interfaceBouCoeffs,
            const FieldField<Field, scalar>& interfaceIntCoeffs,
            const lduInterfaceFieldPtrsList& interfaces,
            const dictionary& solverControls
        );


    //- Destructor
        virtual ~AmgXSolver() = default;


    // Member Functions

        //- Solve the matrix with this solver
        virtual solverPerformance solve
        (
            solveScalarField& psi,
            const scalarField& source,
            const direction cmpt=0
        ) const;

};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

#endif

// ************************************************************************* //
