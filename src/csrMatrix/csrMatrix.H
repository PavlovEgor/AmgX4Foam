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
#include "csrMatrixExecutor.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Forward Declarations
class csrMatrix;


/*---------------------------------------------------------------------------*\
                           Class csrMatrix Declaration
\*---------------------------------------------------------------------------*/

class csrMatrix
{
    //- Private Data

    //- Private methods

protected:
    //- Protected Data

        //- Data executor
        csrMatrixExecutor csrMatExec_;

        //- Number of blocks
        label nBlocks_ = 1;

        //- Number of local rows
        label nOwnerStart_ = 0;

        //- Number of local Nz
        label nLocalNz_ = 0;

        //- Numbers of non-zeros per row
        label* ownerStartPtr_;

        //- Column indices array
        label* colIndicesPtr_;

        //- Permutation ldu2csr array
        label* ldu2csrPerm_;

        //- Coefficients
        scalar* valuesPtr_;

    //- Protected functions
        void initializeAddressing
        (
            const label   nCells,
            const label   nInternalFaces,
            const label   totNnz,
            const label * const owner,
            const label * const neighbour,
                  label * tmpPerm,
                  label * rowIndTmp,
                  label * colIndTmp
        );

        void initializeAddressingExt
        (
            const label   nCells,
            const label   nInternalFaces,
            const label   nnzExt,
            const label   totNnz,
            const label * const owner,
            const label * const neighbour,
            const label * const extRows,
            const label * const extCols,
                  label * tmpPerm,
                  label * rowIndTmp,
                  label * colIndTmp
        );


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
            const label totNnz,
            const label  * const ldu2csr,
            const scalar * const valuesTmp,
                  scalar * values
        );

        //- Computation of the permutatrion array
        void computePermutation(const lduAddressing * addr);

        //- Computation of the permutatrion array
        void computePermutation
        (
            const lduAddressing& addr,
            const lduInterfacePtrsList& interfaces,
                  label& nExtNz
        );

public:

    //- Public Types

    //- Static Data

        //- Declare name of the class and its debug switch
        ClassName("csrMatrix");

    //- Constructors

        //- Construct default.
        //- The coefficients are initialized empty.
        csrMatrix(word mode);

//        //- Construct as copy
//        csrMatrix(const csrMatrix& A);
//
//        //- Construct as copy or re-use as specified.
//        csrMatrix(csrMatrix& A, bool reuse);

        //- Construct given an LDU addressed mesh and an Istream
        //- from which the coefficients are read
        // lduMatrix(const lduMesh&, Istream&);


    //- Destructor
        ~csrMatrix() = default;


    //- Member Functions

        //- Access to addressing

            //- Number of blocks
            label nBlocks() const
            {
                return nBlocks_;
            }

            //- Number of local rows
            label nOwnerStart() const
            {
                return nOwnerStart_;
            }
            //- Number of blocks
            label nLocalNz() const
            {
                return nLocalNz_;
            }

            //- Return vector of the ownerStart
            const label* ownerStart() const
            {
                return ownerStartPtr_;
            }

            //- Return the column index
            const label* colIndices() const
            {
                return colIndicesPtr_;
            }

            //- Return the column index
            const label* ldu2csrPerm() const
            {
                return ldu2csrPerm_;
            }

            //- Return true if the permutation has been computed
            bool hasPermutation() const noexcept
            {
                return (ldu2csrPerm_);
            }

        //- Access to coefficients
            const scalar* values() const;

            bool hasValues() const noexcept
            {
                return (valuesPtr_);
            }

        //- Operations

            //- Set number of blocks
            void setNblocks(label n)
            {
                nBlocks_ = n;
            }

            //- Deallocate useless addressing pointer
            void clearAddressing();

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
