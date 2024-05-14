/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2016-2022 OpenCFD Ltd.
    Copyright (C) 2022-2023 Cineca
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

#include "AmgXWrapper.H"

#include "PstreamGlobals.H"

// initialize AmgXWrapper::count to 0
int Foam::AmgXWrapper::count = 0;

// initialize AmgXWrapper::rsrc to nullptr;
AMGX_resources_handle Foam::AmgXWrapper::rsrc = nullptr;


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

/* \implements AmgXWrapper::AmgXWrapper */
/*Foam::AmgXWrapper::AmgXWrapper
(
    const MPI_Comm &comm,
    const std::string &modeStr,
    const std::string &cfgFile
)
{
    initialize(comm, modeStr, cfgFile);
}*/

// * * * * * * * * * * * * * * * Destructor * * * * * * * * * * * * * * * * * //

/* \implements AmgXWrapper::~AmgXWrapper */
Foam::AmgXWrapper::~AmgXWrapper()
{
    if (isInitialised)
        finalize();
}

// * * * * * * * * * * * * * * * Utilities * * * * * * * * * * * * * * * * * * //

void checkAmgXerror(AMGX_RC code, Foam::word function)
{
    char buff[256];
    AMGX_get_error_string(code, buff, 256);
    if(code != AMGX_RC_OK){
        AMGX_get_error_string(code, buff, 256);
        Foam::Info << function << "returned: " << buff << Foam::nl;
    }
}

// * * * * * * * * * * * * * * Member functions  * * * * * * * * * * * * * * * //

/* \implements AmgXWrapper::initialize*/
void Foam::AmgXWrapper::initialize(
    const word &modeStr,
    const word &dataLocation,
    const string &configStr
)
{
    //- increase the number of AmgXWrapper instances
    count += 1;

    //- get the mode of AmgX solver
    setMode(modeStr);

    initAmgX(configStr);

    dataOrigin_ = dataLocation;

    isInitialised = true;
}

/* \implements AmgXWrapper::initialize*/
void Foam::AmgXWrapper::initialize(
    const label &commId,
    const word &modeStr,
    const word &dataLocation,
    const string &configStr
)
{
    //- increase the number of AmgXWrapper instances
    count += 1;

    //- get the mode of AmgX solver
    setMode(modeStr);

    //- initialize communicators and corresponding information
    initComms(commId);

    initAmgX(configStr);

    dataOrigin_ = dataLocation;

    isInitialised = true;
}

/* \implements AmgXWrapper::setMode */
void Foam::AmgXWrapper::setMode(const word &modeStr)
{
    if (modeStr == "dDDI")
        mode = AMGX_mode_dDDI;
    else if (modeStr == "dDFI")
        mode = AMGX_mode_dDFI;
    else if (modeStr == "dFFI")
        mode = AMGX_mode_dFFI;
    else // NOTA: non ho usato la funzione SETERRQ perchè non ho capito dove è implementata e non ho MPI in questo caso
        Info << modeStr.c_str() << " is not an available mode! Available modes are: dDDI, dDFI, dFFI." <<  nl;
}


/* \implements AmgXWrapper::initComms */
void Foam::AmgXWrapper::initComms(const int &commId)
{
    //- duplicate the communicator
    gpuWorld_ = commId;

    //- get size and rank for communicator
    gpuWorldSize_ = Pstream::nProcs(gpuWorld_);
    myGpuWorldRank_ = Pstream::myProcNo(gpuWorld_);

    cudaGetDeviceCount(&nDevs_);
    cudaGetDevice(&devID_);
}


/* \implements AmgXWrapper::initAmgX */
void Foam::AmgXWrapper::initAmgX(const string &configStr)
{
    //- only the first instance (AmgX solver) is in charge of initializing AmgX
    if (count == 1)
    {
        //- initialize AmgX
        AMGX_SAFE_CALL(AMGX_initialize());

        //- only the master process can output something on the screen
        AMGX_SAFE_CALL(AMGX_register_print_callback(
                    [](const char *msg, int length)->void
                    {Info << msg << nl;}));

        //- let AmgX to handle errors returned
        AMGX_SAFE_CALL(AMGX_install_signal_handler());
    }

    //- create an AmgX configure object
    if(configStr.contains("system"))
    {
        AMGX_SAFE_CALL(AMGX_config_create_from_file(&cfg, configStr.c_str()));
    }
    else
    {
        AMGX_SAFE_CALL(AMGX_config_create(&cfg, configStr.c_str()));
    }

    //- let AmgX handle returned error codes internally
    AMGX_SAFE_CALL(AMGX_config_add_parameters(&cfg, "exception_handling=1"));

    //- create an AmgX resource object, only the first instance is in charge
    if (count == 1)
    {
        if (!Pstream::parRun())
        {
            AMGX_resources_create_simple(&rsrc, cfg);
        }
        else
        {
            AMGX_resources_create(
                &rsrc, cfg, &(PstreamGlobals::MPICommunicators_[gpuWorld_]), 1, &devID_);
        }
    }

    //- create AmgX vector object for unknowns and RHS
    AMGX_vector_create(&AmgXP, rsrc, mode);
    AMGX_vector_create(&AmgXRHS, rsrc, mode);

    //- create AmgX matrix object for unknowns and RHS
    checkAmgXerror(AMGX_matrix_create(&AmgXA, rsrc, mode), "Matrix creation");

    //- create an AmgX solver object
    AMGX_solver_create(&solver, rsrc, mode, cfg);

    //- obtain the default number of rings based on current configuration
    AMGX_config_get_default_number_of_rings(cfg, &ring);
}

/* \implements AmgXWrapper::finalize */
void Foam::AmgXWrapper::finalize()
{
    //- skip if this instance has not been initialised
    if (!isInitialised)
    {
        fprintf(stderr,
                "This AmgXWrapper has not been initialised. "
                "Please initialise it before finalization.\n");
    }

    //- destroy solver instance
    AMGX_solver_destroy(solver);

    //- destroy matrix instance
    AMGX_matrix_destroy(AmgXA);

    //- destroy RHS and unknown vectors
    AMGX_vector_destroy(AmgXP);
    AMGX_vector_destroy(AmgXRHS);

    //- only the last instance need to destroy resource and finalizing AmgX
    if (count == 1)
    {
        AMGX_resources_destroy(rsrc);
        AMGX_SAFE_CALL(AMGX_config_destroy(cfg));

        AMGX_SAFE_CALL(AMGX_finalize());
    }
    else
    {
        AMGX_config_destroy(cfg);
    }

    //- decrease the number of instances
    count -= 1;

    //- change status
    isInitialised = false;
}

/* \implements AmgXWrapper::setOperator */
void Foam::AmgXWrapper::setOperator
(
    const label nGlobalRows,
    const csrAdressing* matrix
)
{
    const label nLocalRows = matrix->ownerStart().size() - 1;
    const label nLocalNz = matrix->colIndices().size();
    const label nBlocks = matrix->nBlocks();
    
    //- Check the matrix size is not larger than tolerated by AmgX
    if(nLocalRows > std::numeric_limits<int>::max())
    {
        fprintf(stderr,
                "AmgX does not support a global number of rows greater than "
                "what can be stored in 32 bits (nGlobalRows = %d).\n",
                nLocalRows);
    }

    if (nLocalNz > std::numeric_limits<int>::max())
    {
        fprintf(stderr,
                "AmgX does not support non-zeros per (consolidated) rank greater than"
                "what can be stored in 32 bits (nLocalNz = %d).\n",
                nLocalNz);
    }

    const int * ownStart; // = matrix->ownerStart().cdata();
    const int * colInd; // = matrix->colIndices().cdata();
    const void * matValues; // = matrix->values().cdata();

    if(dataOrigin_ == "host")
    {
        /*AMGX_pin_memory((void*) matrix->ownerStart().cdata(), (nLocalRows+1)*sizeof(int));
        AMGX_pin_memory((void*) matrix->colIndices().cdata(), (nLocalNz+1)*sizeof(int));
        AMGX_pin_memory((void*) matrix->values().cdata(), (nLocalRows+1)*sizeof(double));*/
        cudaMalloc((void**) &ownStart, sizeof(int)*(nLocalRows+1));
        cudaMalloc((void**) &colInd, sizeof(int)*nLocalNz);
        cudaMalloc((void**) &matValues, sizeof(double)*nLocalNz);
        cudaMemcpy((void*) ownStart, (const void*) matrix->ownerStart().cdata(), sizeof(int)*(nLocalRows+1), cudaMemcpyHostToDevice);
        cudaMemcpy((void*) colInd, (const void*) matrix->colIndices().cdata(), sizeof(int)*nLocalNz, cudaMemcpyHostToDevice);
        cudaMemcpy((void*) matValues, (const void*) matrix->values().cdata(), sizeof(double)*nLocalNz, cudaMemcpyHostToDevice);
    }
    else
    {
        ownStart = matrix->ownerStart().cdata();
        colInd = matrix->colIndices().cdata();
        matValues = matrix->values().cdata();
    }

    //- upload matrix A to AmgX
    if (!Pstream::parRun())
    {
        AMGX_matrix_upload_all(
            AmgXA, nLocalRows, nLocalNz, nBlocks, nBlocks,
            ownStart, colInd, matValues, nullptr);
    }
    else
    {
        AMGX_distribution_handle dist;
        AMGX_distribution_create(&dist, cfg);

        //- Must persist until after we call upload
        labelList offsets(gpuWorldSize_ + 1, 0);

        //- Determine the number of rows per GPU
        labelList nRowsPerGPU(gpuWorldSize_, 0);
        nRowsPerGPU.data()[myGpuWorldRank_] = nLocalRows;
        Pstream::allGatherList(nRowsPerGPU, UPstream::msgType(), gpuWorld_);

        //- Calculate the global offsets
        for(int i = 0; i < gpuWorldSize_; ++i)
        {
            offsets.data()[i+1] = offsets.data()[i] + nRowsPerGPU.data()[i];
        }

        AMGX_distribution_set_partition_data(dist, AMGX_DIST_PARTITION_OFFSETS, offsets.data());

        //- Set the column indices size, 32- / 64-bit
        AMGX_distribution_set_32bit_colindices(dist, true);

        AMGX_matrix_upload_distributed(
            AmgXA, nGlobalRows, nLocalRows, nLocalNz, nBlocks, nBlocks,
            ownStart, colInd, matValues, nullptr, dist);

        AMGX_distribution_destroy(dist);
    }

    //- bind the matrix A to the solver
    AMGX_solver_setup(solver, AmgXA);

    //- connect (bind) vectors to the matrix
    AMGX_vector_bind(AmgXP, AmgXA);
    AMGX_vector_bind(AmgXRHS, AmgXA);
}


/* \implements AmgXWrapper::updateOperator */
void Foam::AmgXWrapper::updateOperator
(
    const csrAdressing* matrix
)
{
    const label nLocalRows = matrix->ownerStart().size() - 1;
    const label nLocalNz = matrix->values().size();
    const void * matValues = matrix->values().cdata();

    //- Replace the coefficients for the CSR matrix A within AmgX
    AMGX_matrix_replace_coefficients(AmgXA, nLocalRows, nLocalNz, matValues, nullptr);

    //- Re-setup the solver (a reduced overhead setup that accounts for consistent matrix structure)
    AMGX_solver_resetup(solver, AmgXA);
}


/* \implements AmgXWrapper::solve */
void Foam::AmgXWrapper::solve
(
    scalar* pscalar,
    const scalar* bscalar,
    const csrAdressing* matrix
)
{
    const label nLocalRows = matrix->ownerStart().size() - 1;
    const int nBlocks = matrix->nBlocks();
    
    //- Upload vectors to AmgX
    AMGX_vector_upload(AmgXP, nLocalRows, nBlocks, pscalar);
    AMGX_vector_upload(AmgXRHS, nLocalRows, nBlocks, bscalar);

    //- Solve
    AMGX_solver_solve(solver, AmgXRHS, AmgXP);

    //- Get the status of the solver
    AMGX_SOLVE_STATUS status;
    AMGX_solver_get_status(solver, &status);

    //- Check whether the solver successfully solved the problem
    if (status != AMGX_SOLVE_SUCCESS)
    {
        fprintf(stderr, "AmgX solver failed to solve the system! "
                        "The error code is %d.\n",
                status);
    }

    // Download data from device
    AMGX_vector_download(AmgXP, pscalar);
}


/* \implements AmgXWrapper::getIters */
void Foam::AmgXWrapper::getIters(label &iter)
{
    AMGX_solver_get_iterations_number(solver, &iter);
}


/* \implements AmgXWrapper::getResidual */
void Foam::AmgXWrapper::getResidual(const label &iter, scalarField &res)
{
    AMGX_solver_get_iteration_residual(solver, iter, 0, res.data());
}


// * * * * * * * * * * * * * Explicit instantiations  * * * * * * * * * * * //


// ************************************************************************* //
