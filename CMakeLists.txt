cmake_minimum_required(VERSION 3.16)
project(LMMC LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Force MSVC to treat all source files (including UTF-8 LAMMP headers without BOM)
# as UTF-8. Avoids C2026/C4819 errors when including lammp/lmmp.h on locales whose
# default code page is not UTF-8 (e.g. Chinese Windows / GBK).
if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:C>:/utf-8>)
endif()

# GCC/Clang: be explicit that source and execution charset are UTF-8.
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        $<$<COMPILE_LANGUAGE:C>:-finput-charset=UTF-8>
        $<$<COMPILE_LANGUAGE:C>:-fexec-charset=UTF-8>
    )
endif()

option(LMMC_BUILD_TESTS "Build LMMC tests" ON)
option(LMMC_BUILD_EXAMPLES "Build LMMC examples" ON)
option(LMMC_BUILD_SHARED "Build LMMC as a shared library" ON)

set(LAMMP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/LAMMP" CACHE PATH "Path to LAMMP root")
set(LAMMP_INCLUDE_DIR "${LAMMP_ROOT}/include" CACHE PATH "Path to LAMMP include directory")
set(LAMMP_BUILD_DIR_HINT "${CMAKE_CURRENT_SOURCE_DIR}/build/lammp" CACHE PATH "Preferred out-of-source LAMMP build directory")
set(LAMMP_STATIC_LIBRARY "" CACHE FILEPATH "Path to prebuilt static LAMMP library")

if(NOT EXISTS "${LAMMP_INCLUDE_DIR}/lammp/lmmp.h")
    message(FATAL_ERROR "LAMMP include headers not found in ${LAMMP_INCLUDE_DIR}")
endif()

set(_LAMMP_SEARCH_PATHS
    "${LAMMP_BUILD_DIR_HINT}/dist/lammp/lib/Release"
    "${LAMMP_BUILD_DIR_HINT}/dist/lammp/lib/Debug"
    "${LAMMP_BUILD_DIR_HINT}/dist/lammp/lib/RelWithDebInfo"
    "${LAMMP_BUILD_DIR_HINT}/dist/lammp/lib/MinSizeRel"
    "${LAMMP_ROOT}/dist/lammp/lib/Release"
    "${LAMMP_ROOT}/dist/lammp/lib/Debug"
    "${LAMMP_ROOT}/dist/lammp/lib/RelWithDebInfo"
    "${LAMMP_ROOT}/dist/lammp/lib/MinSizeRel"
)

find_library(_LAMMP_STATIC_LIBRARY_HINT
    NAMES LammpCore libLammpCore
    PATHS
        "${LAMMP_BUILD_DIR_HINT}/dist/lammp/lib/Release"
        "${LAMMP_BUILD_DIR_HINT}/dist/lammp/lib/Debug"
        "${LAMMP_BUILD_DIR_HINT}/dist/lammp/lib/RelWithDebInfo"
        "${LAMMP_BUILD_DIR_HINT}/dist/lammp/lib/MinSizeRel"
    NO_DEFAULT_PATH
)

if(_LAMMP_STATIC_LIBRARY_HINT)
    string(REPLACE "\\" "/" _lammp_static_norm "${LAMMP_STATIC_LIBRARY}")
    string(REPLACE "\\" "/" _lammp_root_norm "${LAMMP_ROOT}")
    string(FIND "${_lammp_static_norm}" "${_lammp_root_norm}/dist/lammp/lib/" _lammp_legacy_pos)
    if(NOT LAMMP_STATIC_LIBRARY OR _lammp_legacy_pos EQUAL 0)
        set(LAMMP_STATIC_LIBRARY "${_LAMMP_STATIC_LIBRARY_HINT}" CACHE FILEPATH "Path to prebuilt static LAMMP library" FORCE)
    endif()
endif()

if(NOT LAMMP_STATIC_LIBRARY)
    # find_library will not search when a cache entry exists with an empty string.
    unset(LAMMP_STATIC_LIBRARY CACHE)
    find_library(LAMMP_STATIC_LIBRARY
        NAMES LammpCore libLammpCore
        PATHS ${_LAMMP_SEARCH_PATHS}
        NO_DEFAULT_PATH
    )
endif()

set(LMMC_SOURCES
    src/blas_backend.c
    src/complex.c
    src/diagnostic.c
    src/dense_arithmetic.c
    src/dense_elementwise.c
    src/dense_products.c
    src/dense_solvers.c
    src/dense_storage.c
    src/eigen_general.c
    src/eigen_internal.c
    src/eigen_symmetric.c
    src/singular_value_decomposition.c
    src/fft.c
    src/init.c
    src/interp.c
    src/itersolve.c
    src/lsr_boolean_sets.c
    src/lsr_complex_sets.c
    src/lsr_constants_units.c
    src/lsr_linalg_elementwise.c
    src/lsr_linalg_structural.c
    src/lsr_numeric_sets.c
    src/lsr_scalar_functions.c
    src/lsr_statistics_random.c
    src/lsr_text_sets.c
    src/itersolve_minres_lsqr.c
    src/linear_algebra.c
    src/nonlinear.c
    src/numeric.c
    src/ode_common.c
    src/ode_explicit.c
    src/ode_implicit.c
    src/ode_rosenbrock.c
    src/ode_singly_diagonally_implicit.c
    src/optimize.c
    src/precond.c
    src/quadrature.c
    src/random.c
    src/sparse_arithmetic.c
    src/sparse_block.c
    src/sparse_builder.c
    src/sparse_conversion.c
    src/sparse_products.c
    src/sparse_storage.c
    src/sparse_symmetric.c
    src/sparse_direct.c
    src/probability_continuous.c
    src/probability_discrete.c
    src/statistics_combinatorics.c
    src/statistics_descriptive.c
    src/statistics_internal.c
    src/status.c
    src/tensor.c
)

if(LMMC_BUILD_SHARED)
    add_library(lmmc SHARED ${LMMC_SOURCES})
    set_target_properties(lmmc PROPERTIES
        OUTPUT_NAME lmmc
        EXPORT_NAME lmmc
    )
    if(WIN32)
        set_target_properties(lmmc PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
else()
    add_library(lmmc STATIC ${LMMC_SOURCES})
endif()
target_include_directories(lmmc
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${LAMMP_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:include>
)

# LAMMP selects MSVC import declarations through this platform define.
if(MSVC)
    target_compile_definitions(lmmc PRIVATE LAMMP_WINDOWS)
endif()

# Optional BLAS/LAPACK acceleration
option(LMMC_USE_BLAS "Enable external BLAS/LAPACK acceleration" OFF)
if(LMMC_USE_BLAS)
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    target_compile_definitions(lmmc PRIVATE LMMC_USE_BLAS=1)
    target_link_libraries(lmmc PRIVATE ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif()

if(TARGET LammpCore)
    # LammpCore is already a CMake target (e.g. built via add_subdirectory
    # from the parent project).  Link directly against it.
    target_link_libraries(lmmc PUBLIC LammpCore)
    target_compile_definitions(lmmc PRIVATE LMMC_USE_LAMMP_ALLOC=1)
    if(WIN32)
        set_target_properties(LammpCore PROPERTIES PREFIX "")
    endif()
    message(STATUS "LMMC: linking against LammpCore CMake target")
elseif(LAMMP_STATIC_LIBRARY)
    if(NOT EXISTS "${LAMMP_STATIC_LIBRARY}")
        message(FATAL_ERROR "LAMMP_STATIC_LIBRARY does not exist: ${LAMMP_STATIC_LIBRARY}")
    endif()
    add_library(lammp_static UNKNOWN IMPORTED GLOBAL)
    set_target_properties(lammp_static PROPERTIES
        IMPORTED_LOCATION "${LAMMP_STATIC_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${LAMMP_INCLUDE_DIR}"
    )
    target_link_libraries(lmmc PUBLIC lammp_static)
    target_compile_definitions(lmmc PRIVATE LMMC_USE_LAMMP_ALLOC=1)
    message(STATUS "LMMC: linking static LAMMP library: ${LAMMP_STATIC_LIBRARY}")
elseif(EXISTS "${LAMMP_ROOT}/CMakeLists.txt")
    add_subdirectory("${LAMMP_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/lammp" EXCLUDE_FROM_ALL)
    if(WIN32)
        set_target_properties(LammpCore PROPERTIES PREFIX "")
    endif()
    target_link_libraries(lmmc PUBLIC LammpCore)
    target_compile_definitions(lmmc PRIVATE LMMC_USE_LAMMP_ALLOC=1)
    message(STATUS "LMMC: building the LammpCore dependency from ${LAMMP_ROOT}")
else()
    message(FATAL_ERROR
        "LAMMP core was not found. "
        "Build LAMMP first (e.g. cmake -S LAMMP -B build/lammp; cmake --build build/lammp --config Release), "
        "or set -DLAMMP_STATIC_LIBRARY=<path> explicitly."
    )
endif()

option(LMMC_ENABLE_FAST_MATH "Enable aggressive floating point optimizations (-ffast-math, /fp:fast)" OFF)
option(LMMC_ENABLE_NATIVE_ARCH "Enable native architecture instructions (-march=native, /arch:AVX2)" OFF)

set(_gcc_opts "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic>")
if(LMMC_ENABLE_FAST_MATH)
    list(APPEND _gcc_opts "$<$<CONFIG:Release>:$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-ffast-math>$<$<CXX_COMPILER_ID:MSVC>:/fp:fast>>")
endif()
if(LMMC_ENABLE_NATIVE_ARCH)
    list(APPEND _gcc_opts "$<$<CONFIG:Release>:$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-march=native>$<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2>>")
endif()
list(APPEND _gcc_opts "$<$<CONFIG:Release>:$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O3>$<$<CXX_COMPILER_ID:MSVC>:/O2>>")
target_compile_options(lmmc PRIVATE ${_gcc_opts})

function(lmmc_prepare_runtime target_name)
    if(WIN32 AND LMMC_BUILD_SHARED)
        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:lmmc>"
                "$<TARGET_FILE_DIR:${target_name}>"
            VERBATIM
        )
        if(TARGET LammpCore)
            add_custom_command(TARGET ${target_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "$<TARGET_FILE:LammpCore>"
                    "$<TARGET_FILE_DIR:${target_name}>"
                VERBATIM
            )
        endif()
    endif()
endfunction()

if(LMMC_BUILD_EXAMPLES)
    add_subdirectory(example)
endif()

if(LMMC_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
