# AmgX4Foam

AmgX4Foam is a module for OpenFOAM that uses NVIDIA's external AmgX library to solve systems of linear equations on the GPU.

## Installing AmgX

At the very beginning, you need to install the AmgX library itself. It is available with the source code on

```
git clone https://github.com/NVIDIA/AMGX.git
```

You can find instructions on how to build the library in the ReadMe of this repository. In your bashrc, you need to specify the paths to the AmgX includes and library, so that you can use them when building the module in Make openfoam. In my case, it is:

```
#AMGX
export AMGX_DIR=$HOME/AMGX
export AMGX_INC=$AMGX_DIR/include
export AMGX_LIB=$AMGX_DIR/build
export LD_LIBRARY_PATH=$AMGX_DIR/build:$LD_LIBRARY_PATH
```

## Installing AmgX4Foam

As I understand it, it is recommended to build it in a special folder `$WM_PROJECT_DIR/modules`.  Clone the AmgX4Foam module from the repository:

```
git clone https://github.com/PavlovEgor/AmgX4Foam.git
```

(origin: https://gitlab.hpc.cineca.it/exafoam/foamExternalSolvers). There is an `Allwmake` that needs to be run with the `-cu` flag, and the flag values can be viewed inside the script. It is possible that it will not build the first time, so you will need to check in `/wmake` and `/src/Make` that all the dependencies and libraries are specified.

## Using in a case

After building, the module will be compiled in `$FOAM_USER_LIBBIN`, and it can be connected to the calculation in the standard way in `controlDict`:

```libs (AmgX4Foam);```

From now on, all changes are made only in fvSolution. Let's consider an example for pressure:

```
p
{
    solver            AmgX;
    mode              dDDI;
    dataLocation      device;

    AmgXconfig
    {
        config_version      2;
        
        solver
        {
            scope               "main";
            solver              FGMRES;
            max_iters           100;
            tolerance           1e-6;
            norm                L1_SCALED;
            convergence         RELATIVE_INI;
            monitor_residual    1;        
            // print_solve_stats   1;
            // obtain_timings      1;
            gmres_n_restart     20;
            store_res_history   1;
            
            preconditioner
            {
                scope               "amg";
                solver              AMG;
                max_iters           1;
                cycle               V;
                presweeps           1;
                postsweeps          1;
                interpolator        D2;
                monitor_residual    1;        
                store_res_history   1;        
                print_solve_stats   0;
                // print_grid_stats    1;
                print_vis_data      0;
            }
        }
    }
}
```

If the library was successfully compiled, then OpenFOAM should see the new AmgX solver. The choice of `mode` doesn't seem to affect anything, as `dDDI` is already embedded in the code. `dataLocation` can definitely take the value `host`, but it doesn't seem to be used yet.

`AmgXconfig` provides a description of the iterative solver. AmgX itself has many examples of such solvers in the form of json (see `$AMGX_DIR/build/configs`). Here's an example of the corresponding json:

```
{
    "config_version": 2, 
    "solver": {
        "preconditioner": {
            "print_grid_stats": 1, 
            "print_vis_data": 0, 
            "solver": "AMG", 
            "print_solve_stats": 0, 
            "interpolator": "D2",
            "presweeps": 1, 
            "max_iters": 1, 
            "monitor_residual": 0, 
            "store_res_history": 0, 
            "scope": "amg", 
            "cycle": "V", 
            "postsweeps": 1
        }, 
        "solver": "FGMRES", 
        "print_solve_stats": 0, 
        "obtain_timings": 0, 
        "max_iters": 100, 
        "monitor_residual": 1, 
        "gmres_n_restart": 20, 
        "convergence": "RELATIVE_INI", 
        "scope": "main", 
        "tolerance" : 1e-06, 
        "norm": "L1_SCALED"
    }
}
```
I've modified the code a bit, and now you can specify the path to the json configuration file directly. However, it's worth noting that the original code includes a constructor for initializing the context directly from json, but I haven't been able to figure out how to work with it. The syntax for using json is as follows:
```
p
{
    solver            AmgX;
    mode              dDDI;
    dataLocation      device;

    AmgXconfigPath    "$WM_PROJECT_USER_DIR/run/openFoamTests/tubeTests/case/system/FGMRES_CLASSICAL_AGGRESSIVE_PMIS.json";
}
```

### Important notes

- Be sure to use `"store_res_history": 1` at the `solver` level. OpenFOAM uses both the initial and final residuals (the initial one for convergence criteria, and the final one is just printed).

- For the OpenFOAM calculation to match the equivalent one in AmgX, the same norms must be used; otherwise, the convergence criteria may not work correctly. To do this, set `"norm": "L1_SCALED"`.

- If you are using configurations from AmgX, make sure that `print_solve_stats`, `obtain_timings`, and other keys responsible for logging any data are disabled; otherwise, the log will quickly become cluttered.

## Project Dependencies

The following components are required for this project to work correctly:

| Component | Version |
| :--- | :--- |
| OpenFOAM | v2512 |
| CUDA | 13.1 |
| OpenMPI | 4.1.8 |

> **Note:** Make sure your environment variables (e.g., `$CUDA_DIR`, `$OPENMPI_DIR`) point to the correct installations of these versions.