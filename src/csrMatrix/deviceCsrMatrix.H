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
    Foam::deviceCsrMatrix

Description
    deviceCsrMatrix is a general matrix class in which the coefficients are
    stored in a single array.

    Addressing arrays must be supplied for the number of values per row and
    for teh values column index.

SourceFiles
    deviceCsrMatrix.H
    deviceCsrMatrix.C

\*---------------------------------------------------------------------------*/

#ifndef Foam_deviceCsrMatrix_H
#define Foam_deviceCsrMatrix_H

#include "Enum.H"
#include "scalar.H"
#include "devicelduMatrix.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Forward Declarations
class deviceCsrMatrix;


/*---------------------------------------------------------------------------*\
                           Class deviceCsrMatrix Declaration
\*---------------------------------------------------------------------------*/

class deviceCsrMatrix
{
    // Private Data

    // Private methods

        //- Computation of the permutatrion array
        virtual void computePermutation(const devicelduMatrix& lduMatrix);

        //- Computation of the permutatrion array
        virtual void computePermutation
        (
            const devicelduMatrix& lduMatrix,
            const label diagIndexGlobal,
            const label lowOffGlobal,
            const label uppOffGlobal,
            const deviceField<label>& extRows,
            const deviceField<label>& extCols
        );

protected:
    //- Protected Data

        //- Coefficients
        deviceField<Foam::scalar> *valuesPtr_;

        //- Numbers of non-zeros per row
        deviceField<Foam::label> *ownerStartPtr_;
        // int * ownerStartPtr_;

        //- Column indices array
        deviceField<Foam::label> *colIndicesPtr_;

        //- Permutation ldu2csr array
        deviceField<Foam::label> *ldu2csrPerm_;

    //- Protected functions
        void initializeAddressing
        (
            const int   nInternalFaces,
            const int   totNnz,
            const int * const owner,
            const int * const neighbour,
                  int * tmpPerm,
                  int * rowIndTmp,
                  int * colIndTmp,
            const int   nCells = 0
        );

        void initializeAddressingExt
        (
            const int   nInternalFaces,
            const int   nnzExt,
            const int   totNnz,
            const int * const owner,
            const int * const neighbour,
            const int * const extRows,
            const int * const extCols,
                  int * tmpPerm,
                  int * rowIndTmp,
                  int * colIndTmp,
            const int   nCells = 0
        );

        void computeSorting
        (
            const int   totNnz,
                int * tmpPerm,
                int * rowIndTmp,
                int * rowInd,
                int * ldu2csr
        );

        void localToGlobalColIndices
        (
            
            const int   nIntFaces,
            const int   lowOffGlobal,
            const int   uppOffGlobal,
                  int * colIndicesGlobal,
            const int   nRows = 0,
            const int   diagIndexGlobal = 0

        );

        void applyAddressingPermutation
        (
            const int   nCells,
            const int   totNnz,
            const int * const ldu2csr,
            const int * const colIndTmp,
            const int * const rowInd,
                  int * colInd,
                  int * ownStart
        );

        void initializeValue
        (
            
            const int   nIntFaces,
            const double * const upper,
            const double * const lower,
                  double * valuesTmp,
                  const int   nCells = 0,
                  const double * const diag = nullptr
        );

        void initializeValueExt
        (
            const int      nIntFaces,
            const int      nnzExt,
            const double * const upper,
            const double * const lower,
            const double * const extValue,
                  double * valuesTmp,
            const int      nCells = 0,
            const double * const diag = nullptr
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
        ClassName("deviceCsrMatrix");

    //- Constructors

        //- Construct default.
        //  The coefficients are initialized empty.
        deviceCsrMatrix();

        //- Construct as copy
        deviceCsrMatrix(const deviceCsrMatrix& A);

        //- Construct as copy or re-use as specified.
        deviceCsrMatrix(deviceCsrMatrix& A, bool reuse);

        //- Construct given an LDU addressed mesh and an Istream
        //- from which the coefficients are read
        // devicelduMatrix(const lduMesh&, Istream&);


    //- Destructor
        ~deviceCsrMatrix() = default;


    //- Member Functions

        //- Access to addressing

            //- Return vector of the ownerStart
            const deviceField<label>& ownerStart() const noexcept
            {
                return *ownerStartPtr_;
            }

            //- Return the column index
            const deviceField<label>& colIndices() const noexcept
            {
                return *colIndicesPtr_;
            }

            //- Return the column index
            const deviceField<label>& ldu2csrPerm() const noexcept
            {
                return *ldu2csrPerm_;
            }

            //- Return true if the permutation has been computed
            bool hasPermutation() const noexcept
            {
                return (ldu2csrPerm_);
            }

        //- Access to coefficients
            const deviceField<scalar>& values() const;

            virtual const scalar* diagValuesPtr() const
            {
                return nullptr;
            }

            bool hasValues() const noexcept
            {
                return (valuesPtr_);
            }

        //- Operations

            //- Apply permutation from LDU to CSR
            virtual void applyPermutation(const devicelduMatrix& lduMatrix);

            //- Apply permutation from LDU to CSR considering the interface values
            virtual void applyPermutation
            (
                const devicelduMatrix& lduMatrix,
                const label diagIndexGlobal,
                const label lowOffGlobal,
                const label uppOffGlobal,
                const deviceField<label>& extRows,
                const deviceField<label>& extCols,
                const deviceField<scalar>& extVals
            );

            //- Apply permutation from LDU to CSR considering the interface values
            virtual void applyPermutation
            (
                const devicelduMatrix& lduMatrix,
                const deviceField<scalar>& extVals
            );

            //- Deallocate useless addressing pointer
            void clearAddressing();

            //- Deallocate useless addressing pointer
            void finalize();

};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
