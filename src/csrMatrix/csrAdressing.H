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
    Foam::csrAdressing

Description
    csrAdressing is a base class for csr matrices and contains methods to 
    compute permutation from other data structures to csr.
    In CSR format the coefficients are stored in a single array.
    Addressing arrays must be supplied for the number of values per row and
    for teh values column index.

SourceFiles
    csrAdressing.H
    csrAdressing.C

\*---------------------------------------------------------------------------*/

#ifndef Foam_csrAdressing_H
#define Foam_csrAdressing_H

#include "Enum.H"
#include "scalar.H"
#include "lduMatrix.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Forward Declarations
class csrAdressing;


/*---------------------------------------------------------------------------*\
                           Class csrAdressing Declaration
\*---------------------------------------------------------------------------*/

class csrAdressing
{
    // Private Data

    // Private methods


protected:
    //- Protected Data

        //- Number of blocks
        label nBlocks_ = 1;

        //- Numbers of non-zeros per row
        labelList *ownerStartPtr_;

        //- Column indices array
        labelList *colIndicesPtr_;

        //- Permutation ldu2csr array
        labelList *ldu2csrPerm_;

    //- Protected functions
        void initializeAddressing
        (
            const int   nCells,
            const int   nInternalFaces,
            const int   totNnz,
            const int * const owner,
            const int * const neighbour,
                  int * tmpPerm,
                  int * rowIndTmp,
                  int * colIndTmp
        );

        void initializeAddressingExt
        (
            const int   nCells,
            const int   nInternalFaces,
            const int   nnzExt,
            const int   totNnz,
            const int * const owner,
            const int * const neighbour,
            const int * const extRows,
            const int * const extCols,
                  int * tmpPerm,
                  int * rowIndTmp,
                  int * colIndTmp
        );

        void computeSorting
        (
            const int   totNnz,
            const int * const tmpPerm,
            const int * const rowIndTmp,
                    int * rowInd,
                    int * ldu2csr
        );

        void localToGlobalColIndices
        (
            const int   nRows,
            const int   nIntFaces,
            const int   diagIndexGlobal,
            const int   lowOffGlobal,
            const int   uppOffGlobal,
                  int * colIndicesGlobal
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

        // Declare name of the class and its debug switch
        ClassName("csrAdressing");

    //- Constructors

        //- Construct default.
        //  The coefficients are initialized empty.
        csrAdressing();

        //- Construct as copy
        csrAdressing(const csrAdressing& A);

        //- Construct as copy or re-use as specified.
        csrAdressing(csrAdressing& A, bool reuse);

        //- Construct given an LDU addressed mesh and an Istream
        //- from which the coefficients are read
        // lduMatrix(const lduMesh&, Istream&);


    //- Destructor
        ~csrAdressing() = default;


    //- Member Functions

        //- Access to addressing

            //- Number of blocks
            const label& nBlocks() const
            {
                return nBlocks_;
            }

            //- Return vector of the ownerStart
            const labelList& ownerStart() const noexcept
            {
                return *ownerStartPtr_;
            }

            //- Return the column index
            const labelList& colIndices() const noexcept
            {
                return *colIndicesPtr_;
            }

            //- Return the column index
            const labelList& ldu2csrPerm() const noexcept
            {
                return *ldu2csrPerm_;
            }

            //- Return true if the permutation has been computed
            bool hasPermutation() const noexcept
            {
                return (ldu2csrPerm_);
            }

        //- Access to coefficients
            virtual const scalarField& values() const = 0;


        //- Operations

            //- Set number of blocks
            void setNblocks(label n)
            {
                nBlocks_ = n;
            }

            //- Deallocate useless addressing pointer
            void clearAddressing();

            //- Deallocate useless addressing pointer
            void finalizeAdressing();

};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
