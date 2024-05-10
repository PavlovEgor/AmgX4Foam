/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
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

\*---------------------------------------------------------------------------*/

#ifdef have_amgx

#include "AmgXLinearSolverContext.H"
#include "deviceCsrMatrix.H"
#include "deviceMsrMatrix.H"
#include "deviceField.H"

// * * * * * * * * * * * * * explicit instantiation * * * * * * * * * * * * //

template class Foam::AmgXLinearSolverContext<Foam::deviceCsrMatrix>;
template class Foam::AmgXLinearSolverContext<Foam::deviceMsrMatrix>;

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    //defineTypeNameAndDebug(AmgXLinearSolverContext, 0);
    defineTemplateTypeNameAndDebug
    (
        AmgXLinearSolverContext<deviceCsrMatrix>,
        0
    );

    defineTemplateTypeNameAndDebug
    (
        AmgXLinearSolverContext<deviceMsrMatrix>,
        0
    );

    template<class matrix> const word AmgXLinearSolverContext<matrix>::packageName = "AmgX";
}

#endif

// ************************************************************************* //
