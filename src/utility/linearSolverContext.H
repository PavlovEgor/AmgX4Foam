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
    Foam::linearSolverContext

Description
    Abstract base class for a linear solver context.

SourceFiles
    linearSolverContext.H

\*---------------------------------------------------------------------------*/

#ifndef foamLinearSolverContext_H
#define foamLinearSolverContext_H

#include "solverPerformance.H"


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                     Class linearSolverContext Declaration
\*---------------------------------------------------------------------------*/

class linearSolverContext
{
    // Private Data

        bool init_;

public:

    // Public Members

        solverPerformance performance_;

        const word eqName_;

    // Constructors

        //- Default construct
        linearSolverContext(const word eqName)
        :
            init_(false),
            eqName_(eqName)
        {}

        //- SolverPerf inintialization construct
        linearSolverContext(const word eqName, const word solverName)
        :
            init_(false),
            eqName_(eqName),
            performance_(solverName, eqName)
        {}


    //- Destructor
    virtual ~linearSolverContext()
    {
    }


    // Member Functions

        //- Return value of initialized
        bool initialized() const
        {
            return init_;
        }

        //- Return value of initialized
        bool& initialized()
        {
            return init_;
        }
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
