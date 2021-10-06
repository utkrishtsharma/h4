#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include "likwid-stuff.h"
#ifdef LIKWID_PERFMON
#include <likwid-marker.h>
#else
#define LIKWID_MARKER_INIT
#define LIKWID_MARKER_THREADINIT
#define LIKWID_MARKER_SWITCH
#define LIKWID_MARKER_REGISTER(regionTag)
#define LIKWID_MARKER_START(regionTag)
#define LIKWID_MARKER_STOP(regionTag)
#define LIKWID_MARKER_CLOSE
#define LIKWID_MARKER_GET(regionTag, nevents, events, time, count)
#endif


const char* dgemm_desc = "Blocked dgemm, OpenMP-enabled";


/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are n-by-n matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm_blocked(int n, int block_size, double* A, double* B, double* C) 
{

    LIKWID_MARKER_INIT;
#pragma omp parallel
{
    LIKWID_MARKER_REGISTER("foo");
}


#pragma omp parallel
{
    LIKWID_MARKER_START("foo");
    #pragma omp for
    for(int i = 0; i < n; i += block_size)
        {
                for(int j = 0; j < n; j += block_size)
                {
                        __builtin_prefetch((const void*)( &C[i,j] ),0,0);


                        for(int k = 0; k < n; k += block_size)
                        {

                         __builtin_prefetch((const void*)( &A[i,k] ),0,0);
                          __builtin_prefetch((const void*)( &B[k,j]),0,0);

                                for(int x = i; x < i + block_size; x++)
                                {

                                        for(int y = j; y < j + block_size; y++)
                                        {


                                                for(int z = k; z < k + block_size; z++)
                                                {
                                                        C[x + y * n] += A[x + z * n] * B[z + y * n];

                                                }
                                        }

                                }
                        }

                }
        }
        LIKWID_MARKER_STOP("foo");
}
    LIKWID_MARKER_CLOSE;
//      return 0;






   // insert your code here: implementation of blocked matrix multiply with copy optimization and OpenMP parallelism enabled

   // be sure to include LIKWID_MARKER_START(MY_MARKER_REGION_NAME) inside the block of parallel code,
   // but before your matrix multiply code, and then include LIKWID_MARKER_STOP(MY_MARKER_REGION_NAME)
   // after the matrix multiply code but before the end of the parallel code block.

	
  // std::cout << Insert your blocked matrix multiply with copy optimization, openmp-parallel edition here  << std::endl;
}
