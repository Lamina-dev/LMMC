cmake_minimum_required(VERSION 3.26)
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
option(LMMC_USE_BLAS "Enable external BLAS/LAPACK acceleration" OFF)
option(LMMC_USE_LAMMP_ASM "Enable the LAMMP assembly backend" OFF)
option(LMMC_WARNINGS_AS_ERRORS "Treat LMMC compiler warnings as errors" OFF)
option(LMMC_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(LMMC_ENABLE_TSAN "Enable ThreadSanitizer" OFF)
option(LMMC_ENABLE_COVERAGE "Enable compiler coverage instrumentation" OFF)
option(LMMC_ENABLE_FAST_MATH "Enable aggressive floating point optimizations" OFF)
option(LMMC_ENABLE_NATIVE_ARCH "Enable native architecture instructions" OFF)

if(LMMC_ENABLE_SANITIZERS AND LMMC_ENABLE_TSAN)
    message(FATAL_ERROR "LMMC_ENABLE_SANITIZERS and LMMC_ENABLE_TSAN are mutually exclusive")
endif()
if(LMMC_ENABLE_COVERAGE AND (LMMC_ENABLE_SANITIZERS OR LMMC_ENABLE_TSAN))
    message(FATAL_ERROR "LMMC_ENABLE_COVERAGE cannot be combined with sanitizer instrumentation")
endif()

set(LAMMP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/LAMMP" CACHE PATH "Path to LAMMP root")
set(LAMMP_INCLUDE_DIR "${LAMMP_ROOT}/include" CACHE PATH "Path to LAMMP include directory")
set(LAMMP_BUILD_DIR_HINT "${CMAKE_CURRENT_BINARY_DIR}/lammp" CACHE PATH "Preferred out-of-source LAMMP build directory")
set(LAMMP_STATIC_LIBRARY "" CACHE FILEPATH "Path to prebuilt static LAMMP library")

set(_lmmc_lammp_root "${LAMMP_ROOT}")
cmake_path(ABSOLUTE_PATH _lmmc_lammp_root
    BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" NORMALIZE)
set(_lmmc_lammp_include "${LAMMP_INCLUDE_DIR}")
cmake_path(ABSOLUTE_PATH _lmmc_lammp_include
    BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" NORMALIZE)
set(_lmmc_lammp_build_hint "${LAMMP_BUILD_DIR_HINT}")
cmake_path(ABSOLUTE_PATH _lmmc_lammp_build_hint
    BASE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}" NORMALIZE)

if(NOT EXISTS "${_lmmc_lammp_include}/lammp/lmmp.h")
    message(FATAL_ERROR "LAMMP include headers not found in ${_lmmc_lammp_include}")
endif()

set(_lmmc_lammp_search_paths
    "${_lmmc_lammp_build_hint}/dist/lammp/lib/Release"
    "${_lmmc_lammp_build_hint}/dist/lammp/lib/Debug"
    "${_lmmc_lammp_build_hint}/dist/lammp/lib/RelWithDebInfo"
    "${_lmmc_lammp_build_hint}/dist/lammp/lib/MinSizeRel"
    "${_lmmc_lammp_root}/dist/lammp/lib/Release"
    "${_lmmc_lammp_root}/dist/lammp/lib/Debug"
    "${_lmmc_lammp_root}/dist/lammp/lib/RelWithDebInfo"
    "${_lmmc_lammp_root}/dist/lammp/lib/MinSizeRel"
)
unset(_lmmc_lammp_discovered_library CACHE)
find_library(_lmmc_lammp_discovered_library
    NAMES LammpCore libLammpCore
    PATHS ${_lmmc_lammp_search_paths}
    NO_DEFAULT_PATH
)
if(LAMMP_STATIC_LIBRARY)
    set(_lmmc_lammp_static_library "${LAMMP_STATIC_LIBRARY}")
    cmake_path(ABSOLUTE_PATH _lmmc_lammp_static_library
        BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" NORMALIZE)
else()
    set(_lmmc_lammp_static_library "${_lmmc_lammp_discovered_library}")
endif()

set(LMMC_SOURCES
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

function(lmmc_apply_instrumentation target_name)
    if(LMMC_ENABLE_SANITIZERS OR LMMC_ENABLE_TSAN)
        if(WIN32 OR NOT CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
            message(FATAL_ERROR "The configured LMMC sanitizer requires GCC or Clang on a non-Windows platform")
        endif()
        if(LMMC_ENABLE_SANITIZERS)
            set(_lmmc_instrumentation_flags
                -fsanitize=address,undefined -fno-omit-frame-pointer)
        else()
            set(_lmmc_instrumentation_flags
                -fsanitize=thread -fno-omit-frame-pointer)
        endif()
        target_compile_options(${target_name} PRIVATE
            ${_lmmc_instrumentation_flags})
        target_link_options(${target_name} PRIVATE
            ${_lmmc_instrumentation_flags})
    elseif(LMMC_ENABLE_COVERAGE)
        if(WIN32 OR NOT CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
            message(FATAL_ERROR "LMMC coverage requires GCC or Clang on a non-Windows platform")
        endif()
        target_compile_options(${target_name} PRIVATE --coverage -O0 -g)
        target_link_options(${target_name} PRIVATE --coverage)
    endif()
endfunction()

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
        $<INSTALL_INTERFACE:include>
)
target_include_directories(lmmc SYSTEM
    PUBLIC
        $<BUILD_INTERFACE:${_lmmc_lammp_include}>
)
if(UNIX)
    target_link_libraries(lmmc PUBLIC m)
endif()

# LAMMP selects MSVC import declarations through this platform define.
if(MSVC)
    target_compile_definitions(lmmc PRIVATE LAMMP_WINDOWS)
endif()

# Optional BLAS/LAPACK acceleration
if(LMMC_USE_BLAS)
    target_sources(lmmc PRIVATE src/blas_backend.c)
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    target_compile_definitions(lmmc PRIVATE LMMC_USE_BLAS=1)
    target_link_libraries(lmmc PRIVATE ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif()

if(TARGET LammpCore)
    target_link_libraries(lmmc PUBLIC LammpCore)
    target_compile_definitions(lmmc PRIVATE LMMC_USE_LAMMP_ALLOC=1)
    if(WIN32)
        set_target_properties(LammpCore PROPERTIES PREFIX "")
    endif()
    message(STATUS "LMMC: linking against LammpCore CMake target")
elseif(_lmmc_lammp_static_library)
    if(NOT EXISTS "${_lmmc_lammp_static_library}")
        message(FATAL_ERROR "LAMMP_STATIC_LIBRARY does not exist: ${_lmmc_lammp_static_library}")
    endif()
    add_library(lammp_static UNKNOWN IMPORTED GLOBAL)
    set_target_properties(lammp_static PROPERTIES
        IMPORTED_LOCATION "${_lmmc_lammp_static_library}"
        INTERFACE_INCLUDE_DIRECTORIES "${_lmmc_lammp_include}"
    )
    target_link_libraries(lmmc PUBLIC lammp_static)
    target_compile_definitions(lmmc PRIVATE LMMC_USE_LAMMP_ALLOC=1)
    message(STATUS "LMMC: linking static LAMMP library: ${_lmmc_lammp_static_library}")
elseif(EXISTS "${_lmmc_lammp_root}/CMakeLists.txt")
    if(NOT DEFINED LAMMP_BUILD_TESTS)
        set(LAMMP_BUILD_TESTS OFF)
    endif()
    if(NOT DEFINED LAMMP_BUILD_EXAMPLES)
        set(LAMMP_BUILD_EXAMPLES OFF)
    endif()
    if(NOT DEFINED LAMMP_BUILD_BENCHMARKS)
        set(LAMMP_BUILD_BENCHMARKS OFF)
    endif()
    if(NOT DEFINED USE_ASM)
        set(USE_ASM "${LMMC_USE_LAMMP_ASM}")
    endif()
    add_subdirectory("${_lmmc_lammp_root}"
        "${CMAKE_CURRENT_BINARY_DIR}/lammp" EXCLUDE_FROM_ALL)
    set_target_properties(LammpCore PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            "${CMAKE_CURRENT_BINARY_DIR}/lammp/dist/lammp/bin"
        LIBRARY_OUTPUT_DIRECTORY
            "${CMAKE_CURRENT_BINARY_DIR}/lammp/dist/lammp/lib"
        ARCHIVE_OUTPUT_DIRECTORY
            "${CMAKE_CURRENT_BINARY_DIR}/lammp/dist/lammp/lib"
    )
    foreach(_lmmc_config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
        set_target_properties(LammpCore PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY_${_lmmc_config}
                "${CMAKE_CURRENT_BINARY_DIR}/lammp/dist/lammp/bin"
            LIBRARY_OUTPUT_DIRECTORY_${_lmmc_config}
                "${CMAKE_CURRENT_BINARY_DIR}/lammp/dist/lammp/lib"
            ARCHIVE_OUTPUT_DIRECTORY_${_lmmc_config}
                "${CMAKE_CURRENT_BINARY_DIR}/lammp/dist/lammp/lib"
        )
    endforeach()
    if(WIN32)
        set_target_properties(LammpCore PROPERTIES PREFIX "")
    endif()
    target_link_libraries(lmmc PUBLIC LammpCore)
    target_compile_definitions(lmmc PRIVATE LMMC_USE_LAMMP_ALLOC=1)
    message(STATUS "LMMC: building the LammpCore dependency from ${_lmmc_lammp_root}")
else()
    message(FATAL_ERROR
        "LAMMP core was not found. Set LAMMP_STATIC_LIBRARY or provide LAMMP_ROOT.")
endif()

lmmc_apply_instrumentation(lmmc)
if(TARGET LammpCore)
    lmmc_apply_instrumentation(LammpCore)
endif()

set(_lmmc_warning_options
    $<$<COMPILE_LANG_AND_ID:C,GNU,Clang,AppleClang>:-Wall>
    $<$<COMPILE_LANG_AND_ID:C,GNU,Clang,AppleClang>:-Wextra>
    $<$<COMPILE_LANG_AND_ID:C,GNU,Clang,AppleClang>:-Wpedantic>
    $<$<COMPILE_LANG_AND_ID:C,MSVC>:/W4>
)
if(LMMC_WARNINGS_AS_ERRORS)
    list(APPEND _lmmc_warning_options
        $<$<COMPILE_LANG_AND_ID:C,GNU,Clang,AppleClang>:-Werror>
        $<$<COMPILE_LANG_AND_ID:C,MSVC>:/WX>
    )
endif()
if(LMMC_ENABLE_FAST_MATH)
    list(APPEND _lmmc_warning_options
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANG_AND_ID:C,GNU,Clang,AppleClang>>:-ffast-math>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANG_AND_ID:C,MSVC>>:/fp:fast>
    )
endif()
if(LMMC_ENABLE_NATIVE_ARCH)
    list(APPEND _lmmc_warning_options
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANG_AND_ID:C,GNU,Clang,AppleClang>>:-march=native>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANG_AND_ID:C,MSVC>>:/arch:AVX2>
    )
endif()
target_compile_options(lmmc PRIVATE ${_lmmc_warning_options})

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
