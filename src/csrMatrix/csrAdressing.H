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

/* Enumeration for the status of matrix consolidation for the solver.*/
enum class ConsolidationStatus
{
    uninitialised,
    necessary,
    notNecessary,
    initialized,
    consolidated
};


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

        //- Rows diplacement between consolidated processes
        labelList * rowsConsDispPtr_;

        //- Internal faces displcament between consolidated processes
        labelList * intFacesConsDispPtr_;
    
        //- External values displcament between consolidated processes
        labelList * extNzConsDispPtr_;

        //- Total number of rows in the consolidated matrix
        label nConsRows_;

        //- Total number of internal faces in the consolidated matrix
        label nConsIntFaces_;
        
        //- Total number of external values in the consolidated matrix
        label nConsExtNz_;

        //- Total number of non-zero values in the consolidated matrix
        // label nConsTotNz_;

        //- A flag indicating the type of consolidation applied
        ConsolidationStatus consolidationStatus_ = ConsolidationStatus::uninitialised;

        //- Index of the MPI communicator processes that talk to a GPU
        label gpuWorld_;

        //- Size of gpuWorld
        label gpuWorldSize_;

        //- Rank in gpuWorld
        label myGpuWorldRank_;

        //- Indication if the process talk directly to GPU
        bool gpuProc_ = false;

    //- Protected functions
        void initializeConsolidation
        (
            const label nLocalRows,
            const label diagIndexGlobal,
            const label lowOffGlobal,
            const label uppOffGlobal,
            const labelList& own,
            const labelList& neigh,
            const labelList& extRows,
            const labelList& extCols,
                  label& nConsTotNz,
                  labelList* consDiagOffGlob, 
                  labelList* consLowOffGlob, 
                  labelList* consUppOffGlob,
                  List<labelList>& ownLst,
                  List<labelList>& neighLst,
                  List<labelList>& extRowsLst,
                  List<labelList>& extColsLst
        );

        void initializeAddressing
        (
            const int   nConsRows,
            const int   nConsIntFaces,
            const int   nRows,
            const int   nInternalFaces,
            const int   totNnz,
            const int * const owner,
            const int * const neighbour,
                  int * tmpPerm,
                  int * rowIndTmp,
                  int * colIndTmp,
            const int   rowsDispl = 0,
            const int   intFacesDispl = 0
        );

        void initializeAddressingExt
        (
            const int   nConsRows,
            const int   nConsIntFaces,
            const int   nRows,
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
            const int   rowsDispl =0,
            const int   intFacesDispl = 0,
            const int   extNnzDispl = 0
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
            const int   nConsRows,
            const int   nConsIntFaces,
            const int   nRows,
            const int   nIntFaces,
            const int   diagIndexGlobal,
            const int   lowOffGlobal,
            const int   uppOffGlobal,
                  int * colIndicesGlobal,
            const int rowDispl = 0,
            const int intFacesDispl = 0
        );

        void localToConsRowIndex
        (
            const int nConsRows,
            const int nConsIntFaces,
            const int nIntFaces,
            const int nExtNz,
            const int intFacesDipl,
            const int extDispl,
            const int offset,
                  int * rowIndices
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

            //- true if the matrix is consolidated
            bool isConsolidated() const
            {
                return consolidationStatus_ == ConsolidationStatus::initialized;
            }

            //- Retrun rows displacement in consolidated matrix
            label nConsRows() const
            {
                return nConsRows_;
            }

            const labelList& rowsConsDisp() const
            {
                return *rowsConsDispPtr_;
            }

        //- Access to coefficients
            virtual const scalarField& values() const = 0;


        //- Operations

            //- Set number of blocks
            void setNblocks(label n)
            {
                nBlocks_ = n;
            }

            //- Initialize matrix comunications for consolidation
            void initializeComms(label commId, bool gpuProc);

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
