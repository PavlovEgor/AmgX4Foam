/*---------------------------------------------------------------------------*\
-------------------------------------------------------------------------------
    Copyright (C) 2011-2012 OpenFOAM Foundation
    Copyright (C) 2019 OpenCFD Ltd.
    Copyright (C) 2025 CINECA
-------------------------------------------------------------------------------
License
    This file is part of foamExternalSolvers.

    foamExternalSolvers is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    foamExternalSolvers is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with foamExternalSolvers. If not, see <http://www.gnu.org/licenses/>.

Class
    Foam::AmgXSolver

Group
    grpLduMatrixSolvers

Description
    Foam solver that interacts with the AmgX wrapper.

SourceFiles
    AmgXSolver.H
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


    // Private Member Functions

        //- No copy construct
        AmgXSolver(const AmgXSolver&) = delete;

        //- No copy assignment
        void operator=(const AmgXSolver&) = delete;


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
