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

        //- Total number of rows in the consolidated matrix
        label nConsRows_;

        //- Total number of internal faces in the consolidated matrix
        label nConsIntFaces_;
        
        //- Total number of external values in the consolidated matrix
        label nConsExtNz_;

        //- Total number of non-zero values in the consolidated matrix
        // label nConsTotNz_;

        //- Numbers of non-zeros per row
        label* ownerStartPtr_;

        //- Column indices array
        label* colIndicesPtr_;

        //- Permutation ldu2csr array
        label* ldu2csrPerm_;

        //- Coefficients
        scalar* valuesPtr_;

        //- Rows diplacement between consolidated processes
        label * rowsConsDispPtr_;

        //- Internal faces displcament between consolidated processes
        label * intFacesConsDispPtr_;
    
        //- External values displcament between consolidated processes
        label * extNzConsDispPtr_;

        //- Flag for computed permutation
        bool       hasPermutation_ = false;

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
            const label nLocalIntFaces,
            const label nLocalExtNz,
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

        void initializeSequence
        (
            const label len,
                  label * vect
        ) const;

        void initializeAddressing
        (
            const label   nConsRows,
            const label   nConsIntFaces,
            const label   nInternalFaces,
            const label * const owner,
            const label * const neighbour,
                  label * rowIndTmp,
                  label * colIndTmp,
            const label   intFacesDispl = 0
        );

        void initializeAddressingExt
        (
            const label   nConsRows,
            const label   nConsIntFaces,
            const label   nInternalFaces,
            const label   nnzExt,
            const label * const owner,
            const label * const neighbour,
            const label * const extRows,
            const label * const extCols,
                  label * rowIndTmp,
                  label * colIndTmp,
            const label   intFacesDispl = 0,
            const label   extNnzDispl = 0
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
            const label   nConsRows,
            const label   nConsIntFaces,
            const label   nRows,
            const label   nIntFaces,
            const label   diagIndexGlobal,
            const label   lowOffGlobal,
            const label   uppOffGlobal,
                  label * colIndicesGlobal,
            const label   rowDispl = 0,
            const label   intFacesDispl = 0
        );

        void localToConsRowIndex
        (
            const label nConsRows,
            const label nConsIntFaces,
            const label nIntFaces,
            const label nExtNz,
            const label intFacesDipl,
            const label extDispl,
            const label offset,
                  label * rowIndices
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

        void initializeValuesConsolidation
        (
            const label nLocalRows,
            const label nLocalIntFaces,
            const label nLocalExtVals,
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
            const label   nConsRows,
            const label   nConsIntFaces,
            const label   nRows,
            const label   nIntFaces,
            const double * const diag,
            const double * const upper,
            const double * const lower,
                  double * valuesTmp,
            const label   rowsDisp = 0,
            const label   intFacesDisp = 0              
        );

        void initializeValueExt
        (
            const label nConsRows,
            const label nConsIntFaces,
            const label nCells,
            const label nIntFaces,
            const label nnzExt,
            const double * const diag,
            const double * const upper,
            const double * const lower,
            const double * const extValue,
                  double * valuesTmp,
            const label rowsDisp = 0,
            const label intFacesDisp = 0,
            const label extValDisp = 0
        );

        void applyValuePermutation
        (
            const label totNnz,
            const label  * const ldu2csr,
            const scalar * const valuesTmp,
                  scalar * values,
            const label    nBlocks = 1
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

            //- Retrun rows displacement in consolidated matrix
            label nConsRows() const
            {
                return nConsRows_;
            }

            const label* rowsConsDisp() const
            {
                return rowsConsDispPtr_;
            }

            //- Return true if the permutation has been computed
            bool hasPermutation() const noexcept
            {
                return hasPermutation_;
            }

            bool isConsolidated() const
            {
                return consolidationStatus_ == ConsolidationStatus::initialized;
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

            //- Initialize matrix comunications for consolidation
            void initializeComms(label commId, bool gpuProc);

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
