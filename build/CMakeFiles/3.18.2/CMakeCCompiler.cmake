set(CMAKE_C_COMPILER "/opt/cray/pe/craype/2.6.2/bin/cc")
set(CMAKE_C_COMPILER_ARG1 "")
set(CMAKE_C_COMPILER_ID "Intel")
set(CMAKE_C_COMPILER_VERSION "19.0.0.20190206")
set(CMAKE_C_COMPILER_VERSION_INTERNAL "")
set(CMAKE_C_COMPILER_WRAPPER "CrayPrgEnv")
set(CMAKE_C_STANDARD_COMPUTED_DEFAULT "11")
set(CMAKE_C_COMPILE_FEATURES "c_std_90;c_function_prototypes;c_std_99;c_restrict;c_variadic_macros;c_std_11;c_static_assert")
set(CMAKE_C90_COMPILE_FEATURES "c_std_90;c_function_prototypes")
set(CMAKE_C99_COMPILE_FEATURES "c_std_99;c_restrict;c_variadic_macros")
set(CMAKE_C11_COMPILE_FEATURES "c_std_11;c_static_assert")

set(CMAKE_C_PLATFORM_ID "Linux")
set(CMAKE_C_SIMULATE_ID "GNU")
set(CMAKE_C_COMPILER_FRONTEND_VARIANT "")
set(CMAKE_C_SIMULATE_VERSION "7.5.0")




set(CMAKE_AR "/usr/bin/ar")
set(CMAKE_C_COMPILER_AR "")
set(CMAKE_RANLIB "/usr/bin/ranlib")
set(CMAKE_C_COMPILER_RANLIB "")
set(CMAKE_LINKER "/global/common/cori/software/altd/2.0/bin/ld")
set(CMAKE_MT "")
set(CMAKE_COMPILER_IS_GNUCC )
set(CMAKE_C_COMPILER_LOADED 1)
set(CMAKE_C_COMPILER_WORKS TRUE)
set(CMAKE_C_ABI_COMPILED TRUE)
set(CMAKE_COMPILER_IS_MINGW )
set(CMAKE_COMPILER_IS_CYGWIN )
if(CMAKE_COMPILER_IS_CYGWIN)
  set(CYGWIN 1)
  set(UNIX 1)
endif()

set(CMAKE_C_COMPILER_ENV_VAR "CC")

if(CMAKE_COMPILER_IS_MINGW)
  set(MINGW 1)
endif()
set(CMAKE_C_COMPILER_ID_RUN 1)
set(CMAKE_C_SOURCE_FILE_EXTENSIONS c;m)
set(CMAKE_C_IGNORE_EXTENSIONS h;H;o;O;obj;OBJ;def;DEF;rc;RC)
set(CMAKE_C_LINKER_PREFERENCE 10)

# Save compiler ABI information.
set(CMAKE_C_SIZEOF_DATA_PTR "8")
set(CMAKE_C_COMPILER_ABI "ELF")
set(CMAKE_C_LIBRARY_ARCHITECTURE "")

if(CMAKE_C_SIZEOF_DATA_PTR)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_C_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_C_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_C_COMPILER_ABI}")
endif()

if(CMAKE_C_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "")
endif()

set(CMAKE_C_CL_SHOWINCLUDES_PREFIX "")
if(CMAKE_C_CL_SHOWINCLUDES_PREFIX)
  set(CMAKE_CL_SHOWINCLUDES_PREFIX "${CMAKE_C_CL_SHOWINCLUDES_PREFIX}")
endif()





set(CMAKE_C_IMPLICIT_INCLUDE_DIRECTORIES "/opt/cray/pe/libsci/19.06.1/INTEL/16.0/x86_64/include;/opt/cray/pe/mpt/7.7.10/gni/mpich-intel/16.0/include;/opt/cray/rca/2.2.20-7.0.1.1_4.70__g8e3fb5b.ari/include;/opt/cray/alps/6.6.58-7.0.1.1_6.26__g437d88db.ari/include;/opt/cray/xpmem/2.2.20-7.0.1.1_4.26__g0475745.ari/include;/opt/cray/gni-headers/5.0.12.0-7.0.1.1_6.44__g3b1768f.ari/include;/opt/cray/pe/pmi/5.0.14/include;/opt/cray/ugni/6.0.14.0-7.0.1.1_7.59__ge78e5b0.ari/include;/opt/cray/udreg/2.3.2-7.0.1.1_3.57__g8175d3d.ari/include;/opt/cray/wlm_detect/1.3.3-7.0.1.1_4.23__g7109084.ari/include;/opt/cray/krca/2.2.6-7.0.1.1_5.51__gb641b12.ari/include;/opt/cray-hss-devel/9.0.0/include;/opt/intel/compilers_and_libraries_2019.3.199/linux/pstl/include;/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/include;/global/common/sw/install/cray-cnl7-haswell/likwid-5.2.0-ec6ahrr/include;/opt/intel/compilers_and_libraries_2019.3.199/linux/compiler/include/intel64;/opt/intel/compilers_and_libraries_2019.3.199/linux/compiler/include/icc;/opt/intel/compilers_and_libraries_2019.3.199/linux/compiler/include;/usr/local/include;/usr/lib64/gcc/x86_64-suse-linux/7/include;/usr/lib64/gcc/x86_64-suse-linux/7/include-fixed;/usr/include")
set(CMAKE_C_IMPLICIT_LINK_LIBRARIES "hugetlbfs;AtpSigHandler;AtpSigHCommData;rca;darshan;z;mpich_intel;sci_intel_mpi;sci_intel;imf;m;ifcoremt;ifport;pthread;imf;svml;irng;m;ipgo;decimal;cilkrts;stdc++;gcc;gcc_s;irc;svml;c;gcc;gcc_s;irc_s;dl;c")
set(CMAKE_C_IMPLICIT_LINK_DIRECTORIES "/opt/cray/pe/libsci/19.06.1/INTEL/16.0/x86_64/lib;/opt/cray/dmapp/default/lib64;/opt/cray/pe/mpt/7.7.10/gni/mpich-intel/16.0/lib;/opt/cray/rca/2.2.20-7.0.1.1_4.70__g8e3fb5b.ari/lib64;/usr/common/software/darshan/3.2.1/lib;/opt/cray/pe/atp/2.1.3/libApp;/opt/intel/compilers_and_libraries_2019.3.199/linux/compiler/lib/intel64;/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64;/opt/intel/compilers_and_libraries_2019.3.199/linux/compiler/lib/intel64_lin;/usr/lib64/gcc/x86_64-suse-linux/7;/usr/lib64;/lib64;/usr/x86_64-suse-linux/lib;/lib;/usr/lib")
set(CMAKE_C_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")
