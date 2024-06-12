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

Class
    Foam::AmgXLinearSolverContext

Description
    A struct containing a AmgX solver for each equation to be solved

SourceFiles
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

    // Private member function

    //- Generic get value from dict
    word get(const dictionary* dict, const word k)
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

    //- Concatenation of AmgX setting in a configuration string
    string writeConfigurationString(const dictionary& configDict)
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
                    haveSubDict =  true;
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
                word newSolver = dict->get<word>("solver");
                if(scope != "") configStream << scope << ":" << subDictKey << "(" << newScope << ")=" << newSolver << ",";
                else configStream << subDictKey << "(" << newScope << ")=" << newSolver << ",";
                scope = newScope;
            }

        } while (haveSubDict == true);

        DebugInfo << nl << "AmgX configuration string: " << configStream.str() << nl << nl;

        return configStream.str();
    }

public:

    //- Runtime information
        TypeName("AmgXLinearSolverContext");

    //- Public Members

        AmgXWrapper amgx_;
        matrix Amat_;

        static const word packageName;


    //- Constructors

        //- Constructor with external .json file
        AmgXLinearSolverContext
        (
            const word eqName,
            const word solverName,
            const fileName& optionsFile,
            const dictionary solverDict
        )
        :
            linearSolverContext(eqName, solverName),
            loaded_(false),
			Amat_(solverDict.get<word>("mode"))
        {
            int err = 0;

            word mode = "dDDI";
            //word mode = solverDict.get<word>("mode");
            word dataLocation = solverDict.get<word>("dataLocation");

            if (isFile(optionsFile))
            {
                // AMGx initialize
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
                // AMGx initialize no arguments
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

        //- Constructor with configuration in fvSoltuion dictionary
        AmgXLinearSolverContext(const word eqName, const word solverName, const dictionary solverDict)
        :
            linearSolverContext(eqName, solverName),
            loaded_(false),
			Amat_(solverDict.get<word>("mode"))
        {
            word mode = "dDDI";
            //word mode = solverDict.get<word>("mode");
            word dataLocation =solverDict.get<word>("dataLocation");

            string configStr = writeConfigurationString(solverDict.subDict("AmgXconfig"));

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

    //- Destructor
        virtual ~AmgXLinearSolverContext()
        {
            if (loaded_ > 0)
            {
                Info<< "Finalizing AmgX-" << eqName_ << nl;
                // amgx_.finalize();
                // Amat.finalise();
                loaded_ = 0;
            }
            else if (!loaded_)
            {
                Info<< "AmgX-" << eqName_ << " already finalized" << nl;
            }
        }


    // Member Functions

        bool loaded() const
        {
            return loaded_;
        }
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

#endif

// ************************************************************************* //
