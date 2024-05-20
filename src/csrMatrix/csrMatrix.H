/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2016-2023 OpenCFD Ltd.
    Copyright (C) 2023      Cineca
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
    Foam::csrMatrix

Description
    Single block csrMatrix class.

SourceFiles
    csrMatrix.H
    csrMatrix.C

\*---------------------------------------------------------------------------*/

#ifndef Foam_csrMatrix_H
#define Foam_csrMatrix_H

#include "Enum.H"
#include "scalar.H"
#include "lduMatrix.H"
#include "csrAdressing.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Forward Declarations
class csrMatrix;


/*---------------------------------------------------------------------------*\
                           Class csrMatrix Declaration
\*---------------------------------------------------------------------------*/

class csrMatrix
:
    public csrAdressing
{
    // Private Data

        //- Coefficients
        scalarField *valuesPtr_;


    // Private methods

        void initializeValuesConsolidation
        (
            const scalarField& diag,
            const scalarField& upper,
            const scalarField& lower,
            const scalarField& extVal,
            List<scalarField>& diagLst,
            List<scalarField>& upperLst,
            List<scalarField>& lowerLst,
            List<scalarField>& extValLst
        );
           
        void initializeValue
        (
            const int   nConsRows,
            const int   nConsIntFaces,
            const int   nRows,
            const int   nIntFaces,
            const double * const diag,
            const double * const upper,
            const double * const lower,
                double * valuesTmp,
            const int   rowsDisp = 0,
            const int   intFacesDisp = 0              
        );

        void initializeValueExt
        (
            const int nConsRows,
            const int nConsIntFaces,
            const int nCells,
            const int nIntFaces,
            const int nnzExt,
            const double * const diag,
            const double * const upper,
            const double * const lower,
            const double * const extValue,
                    double * valuesTmp,
            const int rowsDisp = 0,
            const int intFacesDisp = 0,
            const int extValDisp = 0
        );

        void applyValuePermutation
        (
            const int      totNnz,
            const int    * const ldu2csr,
            const double * const valuesTmp,
                  double * values
        );

public:

    //- Public Types

    //- Static Data

        // Declare name of the class and its debug switch
        ClassName("csrMatrix");

    //- Constructors

        //- Construct default.
        //  The coefficients are initialized empty.
        csrMatrix();

        //- Construct as copy
        csrMatrix(const csrMatrix& A);

        //- Construct as copy or re-use as specified.
        csrMatrix(csrMatrix& A, bool reuse);

        //- Construct given an LDU addressed mesh and an Istream
        //- from which the coefficients are read
        // lduMatrix(const lduMesh&, Istream&);


    //- Destructor
        ~csrMatrix() = default;


    //- Member Functions

        //- Access to coefficients
            virtual const scalarField& values() const;

            bool hasValues() const noexcept
            {
                return (valuesPtr_);
            }

        //- Operations

            //- Apply permutation from LDU to CSR
            void applyPermutation(const lduMatrix& lduMatrix);

            //- Apply permutation from LDU to CSR considering the interface values
            void applyPermutation
            (
                const lduMatrix& lduMatrix,
                const FieldField<Field, scalar> interfaceBouCoeffs,
                      label& nGlobalCells
            );

            //- Deallocate useless addressing pointer
            void finalize();

};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
