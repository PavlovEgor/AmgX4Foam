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
    Foam::deviceAmgX

Group
    grpLduMatrixSolvers

Description
    Interface to the external package AmgX (https://github.com/NVIDIA/AMGX).

SourceFiles
    deviceAmgX.C

\*---------------------------------------------------------------------------*/

#ifdef have_amgx

#ifndef device_AmgX_H
#define device_AmgX_H

#include "devicelduMatrix.H"
#include "deviceCsrMatrix.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                             Class deviceAmgX Declaration
\*---------------------------------------------------------------------------*/

class deviceAmgX
:
    public devicelduMatrix::solver
{
    // Private Data

        //- Equation name for solver context
        word eqName_;

        //- Matrix format decided for AmgX
        static word matType_;
        

    // Private Member Data

        //- Cached preconditioner
        // mutable autoPtr<lduMatrix::preconditioner> preconPtr_;


    // Private Member Functions

        //- No copy construct
        deviceAmgX(const deviceAmgX&) = delete;

        //- No copy assignment
        void operator=(const deviceAmgX&) = delete;

        //- Compute matrix addressing and value permutation for AmgX use
        void buildAndApplyMatrixPermutation
        (
            deviceCsrMatrix* csrMatrix,
            label& nRowsGlobal
        ) const;

        //- Compute matrix value permutation for AmgX use
        void applyMatrixPermutation
        (
            deviceCsrMatrix* csrMatrix,
            label& nRowsGlobal
        ) const;

        //- Tamplate function for solver call
        template <class matrix>
        solverPerformance solve
        (
            solverPerformance solPerf,
            deviceField<scalar>& psi,
            const deviceField<scalar>& source
        ) const;


public:

    //- Runtime type information
    TypeName("AmgX");


    // Constructors

        //- Construct from matrix components and solver controls
        deviceAmgX
        (
            const word& fieldName,
            const devicelduMatrix& matrix,
            const FieldField<deviceField, scalar>& interfaceBouCoeffs,
            const FieldField<deviceField, scalar>& interfaceIntCoeffs,
            const UPtrList<const devicelduInterfaceField>& interfaces,
            const dictionary& solverControls
        );


    //- Destructor
        virtual ~deviceAmgX() = default;


    // Member Functions

        //- Solve the matrix with this solver
        virtual solverPerformance solve
        (
            deviceField<Foam::scalar>& psi,
            const deviceField<Foam::scalar>& source,
            const direction cmpt=0
        ) const;

};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
