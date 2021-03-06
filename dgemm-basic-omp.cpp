#include <iostream>^Z:
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

const char* dgemm_desc = "Basic implementation, OpenMP-enabled, three-loop dgemm.";

/*
 * This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are n-by-n matrices stored in column-major format.
 * On exit, A and B maintain their input values.
 */
void square_dgemm(int n, double* A, double* B, double* C)
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
    for (int i=0;i<n;i++){
       for(int j=0;j<n;j++){
           for(int k=0;k<n;k++){
                   C[i + j*n] += A[i + k*n] * B[k + j*n];  }
       }}
       LIKWID_MARKER_STOP("foo");
}
    LIKWID_MARKER_CLOSE;
}
