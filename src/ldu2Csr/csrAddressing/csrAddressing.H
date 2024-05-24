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
    Foam::csrAddressing

Description
    csrAddressing is a base class for csr matrices and contains methods to
    compute permutation from other data structures to csr.
    In CSR format the coefficients are stored in a single array.
    Addressing arrays must be supplied for the number of values per row and
    for teh values column index.

SourceFiles
    csrAddressing.H
    csrAddressing.C

\*---------------------------------------------------------------------------*/

#ifndef Foam_csrAddressing_H
#define Foam_csrAddressing_H

#include "Enum.H"
#include "scalar.H"
#include "lduMatrix.H"
#include "csrAddressingExecutor.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Forward Declarations
class csrAddressing;


/*---------------------------------------------------------------------------*\
                           Class csrAddressing Declaration
\*---------------------------------------------------------------------------*/

class csrAddressing
{
    // Private Data

    // Private methods

    static csrAddressingExecutor csrAddrExec_;

protected:
    //- Protected Data

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

        void computeSorting
        (
            const label   totNnz,
            const label * const tmpPerm,
            const label * const rowIndTmp,
                    label * rowInd,
                    label * ldu2csr
        );

        void localToGlobalColIndices
        (
            const label   nRows,
            const label   nIntFaces,
            const label   diagIndexGlobal,
            const label   lowOffGlobal,
            const label   uppOffGlobal,
                  label * colIndicesGlobal
        );

        void applyAddressingPermutation
        (
            const label   nCells,
            const label   totNnz,
            const label * const ldu2csr,
            const label * const colIndTmp,
            const label * const rowInd,
                  label * colInd,
                  label * ownStart
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
        ClassName("csrAddressing");

    //- Constructors

        //- Construct default.
        //  The coefficients are initialized empty.
        csrAddressing();

//        //- Construct as copy
//        csrAddressing(const csrAddressing& A);
//
//        //- Construct as copy or re-use as specified.
//        csrAddressing(csrAddressing& A, bool reuse);

        //- Construct given an LDU addressed mesh and an Istream
        //- from which the coefficients are read
        // lduMatrix(const lduMesh&, Istream&);


    //- Destructor
        ~csrAddressing() = default;


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
