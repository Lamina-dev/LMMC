cmake_minimum_required(VERSION 3.16)
project(LMMC LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(LMMC_BUILD_TESTS "Build LMMC tests" ON)
option(LMMC_BUILD_EXAMPLES "Build LMMC examples" ON)
option(LMMC_BUILD_SHARED "Build shared LMMC library (for Python ctypes/FFI)" ON)
option(LMMC_REQUIRE_LAMMP_STATIC "Require prebuilt static LAMMP library" ON)

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
    src/dense.c
    src/itersolve.c
    src/linear_algebra.c
    src/nonlinear.c
    src/numeric.c
    src/ode.c
    src/precond.c
    src/sparse.c
    src/stats.c
    src/status.c
    src/tensor.c
)

add_library(lmmc STATIC ${LMMC_SOURCES})
set(LMMC_LIB_TARGETS lmmc)

if(LMMC_BUILD_SHARED)
    add_library(lmmc_shared SHARED ${LMMC_SOURCES})
    set_target_properties(lmmc_shared PROPERTIES OUTPUT_NAME lmmc_shared)
    if(WIN32)
        set_target_properties(lmmc_shared PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
    list(APPEND LMMC_LIB_TARGETS lmmc_shared)
endif()

foreach(_lmmc_target IN LISTS LMMC_LIB_TARGETS)
    target_include_directories(${_lmmc_target}
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${LAMMP_INCLUDE_DIR}
    )
endforeach()

if(LAMMP_STATIC_LIBRARY)
    if(NOT EXISTS "${LAMMP_STATIC_LIBRARY}")
        message(FATAL_ERROR "LAMMP_STATIC_LIBRARY does not exist: ${LAMMP_STATIC_LIBRARY}")
    endif()
    add_library(lammp_static UNKNOWN IMPORTED GLOBAL)
    set_target_properties(lammp_static PROPERTIES
        IMPORTED_LOCATION "${LAMMP_STATIC_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${LAMMP_INCLUDE_DIR}"
    )
    foreach(_lmmc_target IN LISTS LMMC_LIB_TARGETS)
        target_link_libraries(${_lmmc_target} PUBLIC lammp_static)
        target_compile_definitions(${_lmmc_target} PRIVATE LMMC_USE_LAMMP_ALLOC=1)
    endforeach()
    message(STATUS "LMMC: linking static LAMMP library: ${LAMMP_STATIC_LIBRARY}")
elseif(LMMC_REQUIRE_LAMMP_STATIC)
    message(FATAL_ERROR
        "LMMC_REQUIRE_LAMMP_STATIC=ON but LAMMP core library was not found. "
        "Build LAMMP first (e.g. cmake -S LAMMP -B build/lammp; cmake --build build/lammp --config Release), "
        "or set -DLAMMP_STATIC_LIBRARY=<path> explicitly."
    )
else()
    message(WARNING "LMMC: LAMMP core library not found, fallback to C runtime allocator bridge is enabled only because LMMC_REQUIRE_LAMMP_STATIC=OFF")
endif()

if(MSVC)
    foreach(_lmmc_target IN LISTS LMMC_LIB_TARGETS)
        target_compile_options(${_lmmc_target} PRIVATE /W4)
    endforeach()
else()
    foreach(_lmmc_target IN LISTS LMMC_LIB_TARGETS)
        target_compile_options(${_lmmc_target} PRIVATE -Wall -Wextra -Wpedantic)
    endforeach()
endif()

function(lmmc_copy_lammp_runtime target_name)
    if(WIN32 AND LAMMP_STATIC_LIBRARY)
        get_filename_component(_lammp_lib_name "${LAMMP_STATIC_LIBRARY}" NAME_WE)
        get_filename_component(_lammp_lib_dir "${LAMMP_STATIC_LIBRARY}" DIRECTORY)
        get_filename_component(_lammp_lib_root "${_lammp_lib_dir}" DIRECTORY)
        get_filename_component(_lammp_dist_root "${_lammp_lib_root}" DIRECTORY)
        get_filename_component(_lammp_cfg_dir "${_lammp_lib_dir}" NAME)
        set(_lammp_runtime_base "${_lammp_lib_name}")
        if(_lammp_runtime_base MATCHES "^lib(.+)$")
            set(_lammp_runtime_base "${CMAKE_MATCH_1}")
        endif()

        set(_lammp_runtime_dll "${_lammp_dist_root}/bin/${_lammp_cfg_dir}/${_lammp_runtime_base}.dll")
        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_lammp_runtime_dll}"
                "$<TARGET_FILE_DIR:${target_name}>"
            COMMENT "LMMC: copy ${_lammp_runtime_base}.dll for ${target_name}"
            VERBATIM
        )
    endif()
endfunction()

if(LMMC_BUILD_EXAMPLES)
    add_subdirectory(example)
endif()

if(LMMC_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
