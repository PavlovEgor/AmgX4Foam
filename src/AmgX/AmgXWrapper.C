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

// #include <iostream>
// #include <fstream>

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

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << " (error code " << err << "): " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
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

    if(gpuProc_) initAmgX(configStr);

    dataOrigin_ = dataLocation;

    isInitialised = true;
}


void Foam::AmgXWrapper::initialiseMatrixComms(csrAdressing* matrix)
{
    labelList gpuWorldProcs(gpuWorldSize_);

    MPI_Allgather(&myGlobalWorldRank_, 1, MPI_INT, gpuWorldProcs.data(), 1, MPI_INT, gpuWorld_);

    MPI_Barrier(globalWorld_);

    label gpuWorldIdx = UPstream::allocateCommunicator(globalWorldIdx_, gpuWorldProcs);
    matrix->initializeComms(gpuWorldIdx, gpuProc_);

    MPI_Barrier(globalWorld_);
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
    globalWorldIdx_ = commId;
    globalWorld_ = PstreamGlobals::MPICommunicators_[commId];

    //- get size and rank for communicator
    globalWorldSize_ = Pstream::nProcs(commId);
    myGlobalWorldRank_ = Pstream::myProcNo(commId);

    // Get the communicator for processors on the same node (local world)
    MPI_Comm_split_type(globalWorld_, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &localWorld_);
    MPI_Comm_set_name(localWorld_, "localWorld");

    // get size and rank for local communicator
    MPI_Comm_size(localWorld_, &localWorldSize_);
    MPI_Comm_rank(localWorld_, &myLocalWorldRank_);

    cudaGetDeviceCount(&nDevs_);
    
    if (localWorldSize_ == nDevs_)
    {
        devID_ = myLocalWorldRank_;
        gpuProc_ = true;
    }
    else if (localWorldSize_ > nDevs_)
    {
        int nBasic = localWorldSize_ / nDevs_,
            nRemain = localWorldSize_ % nDevs_;

        if (myLocalWorldRank_ < (nBasic+1)*nRemain)
        {
            devID_ = myLocalWorldRank_ / (nBasic + 1);
            if (myLocalWorldRank_ % (nBasic + 1) == 0)  gpuProc_ = 0;
        }
        else
        {
            devID_ = (myLocalWorldRank_ - (nBasic+1)*nRemain) / nBasic + nRemain;
            if ((myLocalWorldRank_ - (nBasic+1)*nRemain) % nBasic == 0) gpuProc_ = true;
        }

    }
    else
    {
        Info << "CUDA devices per node are more than the MPI processes launched on the node. Only " 
        << localWorldSize_ << " CUDA devices will be used." << nl;
        
        devID_ = myLocalWorldRank_;
        gpuProc_ = true;
    }

    cudaSetDevice(devID_);

    MPI_Barrier(globalWorld_);

    // split the global world into a world involved in AmgX and a null world
    MPI_Comm_split(globalWorld_, (int) gpuProc_, 0, &globalGpuWorld_);

    // get size and rank for the communicator corresponding to gpuWorld
    if (gpuProc_)
    {
        MPI_Comm_set_name(globalGpuWorld_, "globalGpuWorld");
        MPI_Comm_size(globalGpuWorld_, &globalGpuWorldSize_);
        MPI_Comm_rank(globalGpuWorld_, &myGlobalGpuWorldRank_);
    }

    // split local world into worlds corresponding to each CUDA device
    MPI_Comm_split(localWorld_, devID_, 0, &gpuWorld_);
    MPI_Comm_set_name(gpuWorld_, "gpuWorld");

    // get size and rank for the communicator corresponding to myWorld
    MPI_Comm_size(gpuWorld_, &gpuWorldSize_);
    MPI_Comm_rank(gpuWorld_, &myGpuWorldRank_);

    MPI_Barrier(globalWorld_);
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
            AMGX_resources_create(&rsrc, cfg, &globalGpuWorld_, 1, &devID_);
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

    if(gpuProc_)
    {
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
        
        if (Pstream::parRun()) MPI_Comm_free(&globalGpuWorld_);

        if(pCons_)
        {
            cudaFree(pCons_);
            cudaFree(rhsCons_);
        }
    }

    if (Pstream::parRun()) 
    {
        MPI_Comm_free(&gpuWorld_);
        MPI_Comm_free(&localWorld_);    
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
    if(gpuProc_)
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

            if(matrix->isConsolidated())
            {
                label nConsRows = matrix->nConsRows();
                checkCudaError(cudaMalloc((void**) &pCons_, sizeof(scalar)*nConsRows), "pCons_ cudaMalloc");
                checkCudaError(cudaMalloc((void**) &rhsCons_, sizeof(scalar)*nConsRows), "rhsCons_ cudamalloc");

                cudaIpcGetMemHandle(&pConsHandle_, pCons_);
                cudaIpcGetMemHandle(&rhsConsHandle_, rhsCons_);
            }
        }
        else
        {
            ownStart = matrix->ownerStart().cdata();
            colInd = matrix->colIndices().cdata();
            matValues = matrix->values().cdata();
        }

        //- upload matrix A to AmgX
        if (globalGpuWorldSize_ == 1 || !Pstream::parRun())
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
            labelList offsets(globalGpuWorldSize_ + 1, Zero);

            //- Determine the number of rows per GPU
            labelList nRowsPerGPU(globalGpuWorldSize_, Zero);
            /*nRowsPerGPU.data()[globalGpuWorldSize_] = nLocalRows;
            Pstream::allGatherList(nRowsPerGPU, UPstream::msgType(), gpuGlobalWorld_);*/
            MPI_Allgather(&nLocalRows, 1, MPI_INT, nRowsPerGPU.data(), 1, MPI_INT, globalGpuWorld_);
 
            //- Calculate the global offsets
            for(int i = 0; i < globalGpuWorldSize_; ++i)
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

    if(matrix->isConsolidated())
    {
        MPI_Bcast(&pConsHandle_, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 0, gpuWorld_);
        MPI_Bcast(&rhsConsHandle_, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 0, gpuWorld_);
        
        if(!gpuProc_)
        {
            cudaIpcOpenMemHandle((void**) &pCons_, pConsHandle_, cudaIpcMemLazyEnablePeerAccess );
            cudaIpcOpenMemHandle((void**) &rhsCons_, rhsConsHandle_, cudaIpcMemLazyEnablePeerAccess );
        }
    }
}


/* \implements AmgXWrapper::updateOperator */
void Foam::AmgXWrapper::updateOperator
(
    const csrAdressing* matrix
)
{
    if(gpuProc_)
    {
        const label nLocalRows = matrix->ownerStart().size() - 1;
        const label nLocalNz = matrix->values().size();
        const void * matValues = matrix->values().cdata();

        //- Replace the coefficients for the CSR matrix A within AmgX
        AMGX_matrix_replace_coefficients(AmgXA, nLocalRows, nLocalNz, matValues, nullptr);

        //- Re-setup the solver (a reduced overhead setup that accounts for consistent matrix structure)
        AMGX_solver_resetup(solver, AmgXA);
    }
}


/* \implements AmgXWrapper::solve */
void Foam::AmgXWrapper::solve
(
    const int nLocalRows,
    scalar* pscalar,
    const scalar* bscalar,
    const csrAdressing* matrix
)
{    
    scalar * p;
    const scalar * b;
    label consDispl;
    label nRows, nBlocks;

    // nLocalRows = matrix->ownerStart().size() - 1;
    nBlocks = matrix->nBlocks();

    if(matrix->isConsolidated())
    {       
        consDispl = matrix->rowsConsDisp().cdata()[myGpuWorldRank_];
        cudaMemcpy((void*) &pCons_[consDispl], pscalar, nLocalRows*sizeof(scalar), cudaMemcpyHostToDevice); // cudaMemcpyDefault);
        checkCudaError(cudaMemcpy((void**) &(rhsCons_[consDispl]), (void*) bscalar, (size_t) (nLocalRows*sizeof(scalar)), cudaMemcpyDefault ), // cudaMemcpyDefault);
                       "b cudaMemcpy");
        p = pCons_;
        b = rhsCons_;
        nRows = matrix->nConsRows();

        cudaDeviceSynchronize();
        MPI_Barrier(gpuWorld_);

        /*double* rhs;
        rhs = new double[nRows];
        cudaMemcpy((void*)rhs, (const void*)b, sizeof(double)*nRows, cudaMemcpyDeviceToHost );
        std::string fileName = "bscalar" + std::to_string(Pstream::myProcNo());
        std::ofstream outFile5(fileName);
        outFile5 << "bscalar:" << nl;
        for(int i=0; i< nRows; ++i) outFile5 << rhs[i] << nl;
        outFile5.close();*/
    }
    else
    {
        p = pscalar;
        b = bscalar;
        nRows = nLocalRows;
    }
    
    if (gpuProc_)
    {    
        //- Upload vectors to AmgX
        AMGX_vector_upload(AmgXP, nRows, nBlocks, p);
        AMGX_vector_upload(AmgXRHS, nRows, nBlocks, b);

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
        AMGX_vector_download(AmgXP, p);

        if(matrix->isConsolidated()) cudaDeviceSynchronize();
    }
    if(Pstream::parRun()) MPI_Barrier(gpuWorld_); //necessary

    if (matrix->isConsolidated())
    {
        checkCudaError(cudaMemcpy((void*) pscalar, &pCons_[consDispl], nLocalRows*sizeof(scalar), cudaMemcpyDeviceToHost),
                       "pscalar back cudaMemcpy");

        cudaDeviceSynchronize();
        MPI_Barrier(gpuWorld_);
    }
}


/* \implements AmgXWrapper::getIters */
void Foam::AmgXWrapper::getIters(label &iter)
{
    if (gpuProc_) AMGX_solver_get_iterations_number(solver, &iter);
}


/* \implements AmgXWrapper::getResidual */
void Foam::AmgXWrapper::getResidual(const label &iter, scalarField &res)
{
    if (gpuProc_) AMGX_solver_get_iteration_residual(solver, iter, 0, res.data());
}


// * * * * * * * * * * * * * Explicit instantiations  * * * * * * * * * * * //


// ************************************************************************* //
