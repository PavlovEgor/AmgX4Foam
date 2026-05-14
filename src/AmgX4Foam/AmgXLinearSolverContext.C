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

\*---------------------------------------------------------------------------*/

#include "AmgXLinearSolverContext.H"
#include "csrMatrix.H"
#include <sstream>

// * * * * * * * * * * * * * explicit instantiation * * * * * * * * * * * * //

template class Foam::AmgXLinearSolverContext<Foam::csrMatrix>;

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    // defineTypeNameAndDebug(AmgXLinearSolverContext, 0);
    defineTemplateTypeNameAndDebug
    (
        AmgXLinearSolverContext<csrMatrix>,
        0
    );

    // const word AmgXLinearSolverContext::packageName = "AmgX";
    template<class matrix> const word AmgXLinearSolverContext<matrix>::packageName = "AmgX";
}

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

namespace Foam
{

template<class matrix>
word AmgXLinearSolverContext<matrix>::get(const dictionary* dict, const word k)
{
    OStringStream os;
    const bool oldThrowingError = FatalError.throwing(true);
    const bool oldThrowingIOerr = FatalIOError.throwing(true);

    try
    {
        os << dict->get<word>(k);
    }
    catch (const Foam::IOerror& err)
    {
        os << dict->get<scalar>(k);
    }
    catch (const Foam::error& err)
    {
        os << dict->get<scalar>(k);
    }

    FatalError.throwing(oldThrowingError);
    FatalIOError.throwing(oldThrowingIOerr);

    return os.str();
}

template<class matrix>
string AmgXLinearSolverContext<matrix>::writeConfigurationString(const dictionary& configDict)
{
    OStringStream configStream;
    word scope = "";
    word subDictKey = "";
    const dictionary* dict = &configDict;
    bool haveSubDict;

    do
    {
        haveSubDict = false;
        wordList keyList = dict->toc();
        forAll(keyList, i)
        {
            word key = keyList[i];
            if(dict->isDict(key))
            {
                subDictKey = key;
                haveSubDict = true;
            }
            else if(key != "scope" && key != "solver")
            {
                if(scope != "") configStream << scope << ":" << key << "=" << get(dict, key) << ",";
                else configStream << key << "=" << get(dict, key) << ",";
            }
        }

        if(haveSubDict == true)
        {
            dict = dict->findDict(subDictKey);
            word newScope = dict->get<word>("scope");
            if(dict->found("solver"))
            {
                word newSolver = dict->get<word>("solver");
                if(scope != "") configStream << scope << ":" << subDictKey << "(" << newScope << ")=" << newSolver << ",";
                else configStream << subDictKey << "(" << newScope << ")=" << newSolver << ",";
            }
            scope = newScope;
        }
    } while (haveSubDict == true);

    DebugInfo << nl << "AmgX configuration string: " << configStream.str() << nl << nl;
    return configStream.str();
}

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class matrix>
AmgXLinearSolverContext<matrix>::AmgXLinearSolverContext
(
    const word eqName,
    const word solverName,
    const fileName& optionsFile,
    const dictionary solverDict
)
:
    linearSolverContext(eqName, solverName),
    loaded_(false),
    updated_(false),
    updateMatrixCoefficients_(solverDict.getOrDefault<bool>("updateMatrixCoefficients", true)),
    Amat_(solverDict.get<word>("mode"))
{
    int err = 0;
    word mode = solverDict.get<word>("mode");
    word dataLocation = solverDict.get<word>("dataLocation");

    if (isFile(optionsFile))
    {
        if(!Pstream::parRun())
        {
            amgx_.initialize(mode, dataLocation, optionsFile);
        }
        else
        {
            const label nReq = Pstream::nRequests();
            amgx_.initialize(Pstream::myWorldID(), mode, dataLocation, optionsFile);
            Pstream::waitRequests(nReq);
        }
    }
    else
    {
        err = 1;
        loaded_ = 0;
        Info<< "Error: AmgX-" << eqName_ << " cannot be initialized without a valid config file" << nl;
        Info<< optionsFile << " cannot be found" << nl;
    }

    if (!err)
    {
        Info<< "Initializing AmgX-" << eqName_ << " context" << nl;
        loaded_ = 1;
    }
    else
    {
        Info<< "Could not initialize AmgX-" << eqName_ << nl;
    }
}

template<class matrix>
AmgXLinearSolverContext<matrix>::AmgXLinearSolverContext
(
    const word eqName,
    const word solverName,
    const dictionary solverDict
)
:
    linearSolverContext(eqName, solverName),
    loaded_(false),
    updated_(false),
    updateMatrixCoefficients_(solverDict.getOrDefault<bool>("updateMatrixCoefficients", true)),
    Amat_(solverDict.get<word>("mode"))
{
    word mode = solverDict.get<word>("mode");
    word dataLocation = solverDict.get<word>("dataLocation");

    string configStr;
    if (solverDict.found("AmgXconfig"))
    {
        configStr = writeConfigurationString(solverDict.subDict("AmgXconfig"));
    }
    else
    {
        fileName configFile = solverDict.get<fileName>("AmgXconfigPath");
        configFile.expand();
        configStr = string(configFile);
    }

    if(!Pstream::parRun())
    {
        amgx_.initialize(mode, dataLocation, configStr);
    }
    else
    {
        const label nReq = Pstream::nRequests();
        amgx_.initialize(Pstream::myWorldID(), mode, dataLocation, configStr);
        Pstream::waitRequests(nReq);
    }

    Info<< "Initializing AmgX-" << eqName_ << " context" << nl;
    loaded_ = 1;
}

// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

template<class matrix>
AmgXLinearSolverContext<matrix>::~AmgXLinearSolverContext()
{
    if (loaded_ > 0)
    {
        Info<< "Finalizing AmgX-" << eqName_ << nl;
        amgx_.finalize();
        Amat_.finalize();
        loaded_ = 0;
    }
    else if (!loaded_)
    {
        Info<< "AmgX-" << eqName_ << " already finalized" << nl;
    }
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class matrix>
void AmgXLinearSolverContext<matrix>::updateConfig(const dictionary& solverDict)
{
    string configStr = writeConfigurationString(solverDict.subDict("AmgXconfig"));
    amgx_.updateConfig(configStr);
}

template<class matrix>
bool AmgXLinearSolverContext<matrix>::loaded() const
{
    return loaded_;
}

template<class matrix>
bool AmgXLinearSolverContext<matrix>::updated() const
{
    return updated_;
}

template<class matrix>
bool& AmgXLinearSolverContext<matrix>::updated()
{
    return updated_;
}

template<class matrix>
bool AmgXLinearSolverContext<matrix>::doUpdateMatrixCoefficients() const
{
    return updateMatrixCoefficients_;
}

} // End namespace Foam

// ************************************************************************* //
