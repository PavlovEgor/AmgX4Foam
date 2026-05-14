/*---------------------------------------------------------------------------*\
-------------------------------------------------------------------------------
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
    Foam::AmgXLinearSolverContext

Description
    A class containing an AmgX solver and a matrix (in a suitable format for AmgX) 
    for each equation to be solved.

SourceFiles
    AmgXLinearSolverContext.H
    AmgXLinearSolverContext.C

\*---------------------------------------------------------------------------*/

#ifndef AmgXFoamLinearSolverContext_H
#define AmgXFoamLinearSolverContext_H

#include "linearSolverContext.H"
#include "uncollatedFileOperation.H"
#include "AmgXWrapper.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Forward Declarations
template <class matrix>
class AmgXLinearSolverContext;

/*---------------------------------------------------------------------------*\
                     Class AmgXLinearSolverContext Declaration
\*---------------------------------------------------------------------------*/

template <class matrix>
class AmgXLinearSolverContext
:
    public linearSolverContext
{
    // Private Data
    bool loaded_;
    bool updated_;

    // Private member functions
    //- Generic get value from dict
    word get(const dictionary* dict, const word k);

    //- Concatenation of AmgX setting in a configuration string
    string writeConfigurationString(const dictionary& configDict);

public:
    //- Runtime information
    TypeName("AmgXLinearSolverContext");

    //- Public Members
    AmgXWrapper amgx_;
    matrix Amat_;

    //- Flag to cache the matrix coefficients
    bool updateMatrixCoefficients_;
    static const word packageName;

    //- Constructors
    //- Constructor with external .json file
    AmgXLinearSolverContext
    (
        const word eqName,
        const word solverName,
        const fileName& optionsFile,
        const dictionary solverDict
    );

    //- Constructor with configuration in fvSolution dictionary
    AmgXLinearSolverContext
    (
        const word eqName,
        const word solverName,
        const dictionary solverDict
    );

    //- Destructor
    virtual ~AmgXLinearSolverContext();

    // Member Functions

    //- Update configuration
    void updateConfig(const dictionary& solverDict);

    // Context states
    bool loaded() const;
    bool updated() const;
    bool& updated();
    bool doUpdateMatrixCoefficients() const;
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

#endif

// ************************************************************************* //
