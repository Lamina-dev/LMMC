cmake_minimum_required(VERSION 3.26)
project(LMMC VERSION 1.0.0 LANGUAGES C)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Force MSVC to treat all source files (including UTF-8 LMMP headers without BOM)
# as UTF-8. Avoids C2026/C4819 errors when including lmmp/lmmp.h on locales whose
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

option(LMMC_BUILD_TESTS "Build LMMC tests" ${PROJECT_IS_TOP_LEVEL})
option(LMMC_BUILD_EXAMPLES "Build LMMC examples" ${PROJECT_IS_TOP_LEVEL})
option(LMMC_BUILD_SHARED "Build LMMC as a shared library" ON)
set(LMMC_LMMP_ASM "AUTO" CACHE STRING "LMMP assembly mode")
set_property(CACHE LMMC_LMMP_ASM PROPERTY STRINGS GENERIC X64 ARM64 AUTO)
option(LMMC_WARNINGS_AS_ERRORS "Treat LMMC compiler warnings as errors" OFF)
option(LMMC_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(LMMC_ENABLE_TSAN "Enable ThreadSanitizer" OFF)
option(LMMC_ENABLE_COVERAGE "Enable compiler coverage instrumentation" OFF)
option(LMMC_ENABLE_CLANG_TIDY "Run clang-tidy on LMMC C sources" OFF)
option(LMMC_DEBUG_LEAKS "Track live persistent bridge allocations" OFF)

if(LMMC_ENABLE_SANITIZERS AND LMMC_ENABLE_TSAN)
    message(FATAL_ERROR "LMMC_ENABLE_SANITIZERS and LMMC_ENABLE_TSAN are mutually exclusive")
endif()
if(LMMC_ENABLE_COVERAGE AND (LMMC_ENABLE_SANITIZERS OR LMMC_ENABLE_TSAN))
    message(FATAL_ERROR "LMMC_ENABLE_COVERAGE cannot be combined with sanitizer instrumentation")
endif()

set(LMMP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/LMMP" CACHE PATH "Path to LMMP root")
set(LMMP_INCLUDE_DIR "${LMMP_ROOT}/include" CACHE PATH "Path to LMMP include directory")
set(LMMP_LIBRARY "" CACHE FILEPATH "Path to a prebuilt LMMP library")

set(_lmmc_lmmp_root "${LMMP_ROOT}")
cmake_path(ABSOLUTE_PATH _lmmc_lmmp_root
    BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" NORMALIZE)
set(_lmmc_lmmp_include "${LMMP_INCLUDE_DIR}")
cmake_path(ABSOLUTE_PATH _lmmc_lmmp_include
    BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" NORMALIZE)

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
    src/memory_bridge.c
    src/interp.c
    src/itersolve.c
    src/stdlib_boolean_sets.c
    src/stdlib_complex_sets.c
    src/stdlib_constants_units.c
    src/stdlib_linalg_elementwise.c
    src/stdlib_linalg_structural.c
    src/stdlib_numeric_sets.c
    src/stdlib_scalar_functions.c
    src/stdlib_statistics_random.c
    src/stdlib_text_sets.c
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
        # Threaded tests share coverage counters even for thread-local APIs.
        target_compile_options(${target_name} PRIVATE
            --coverage -fprofile-update=atomic -O0 -g)
        target_link_options(${target_name} PRIVATE --coverage)
    endif()
endfunction()

if(LMMC_BUILD_SHARED)
    add_library(lmmc SHARED ${LMMC_SOURCES})
    set_target_properties(lmmc PROPERTIES
        OUTPUT_NAME lmmc
        EXPORT_NAME lmmc
        VERSION "${PROJECT_VERSION}"
        SOVERSION "${PROJECT_VERSION_MAJOR}"
    )
else()
    add_library(lmmc STATIC ${LMMC_SOURCES})
endif()
target_include_directories(lmmc
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_include_directories(lmmc SYSTEM
    PUBLIC
        $<BUILD_INTERFACE:${_lmmc_lmmp_include}>
)
# Export only the explicitly maintained shared-library ABI. Test-only and
# conditional debug hooks are appended below without making header formatting
# part of the export contract.
set(_lmmc_public_symbol_file
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/lmmc_public_symbols.txt")
file(STRINGS "${_lmmc_public_symbol_file}" _lmmc_export_symbols
    REGEX "^lmmc_[A-Za-z0-9_]+$")
if(NOT _lmmc_export_symbols)
    message(FATAL_ERROR
        "LMMC public symbol list is empty: ${_lmmc_public_symbol_file}")
endif()
if(LMMC_BUILD_TESTS)
    list(APPEND _lmmc_export_symbols lmmc_memory_fail_after_for_test)
endif()
if(LMMC_DEBUG_LEAKS)
    list(APPEND _lmmc_export_symbols lmmc_debug_leaks_get_count)
endif()
list(REMOVE_DUPLICATES _lmmc_export_symbols)
list(SORT _lmmc_export_symbols)

if(WIN32 AND LMMC_BUILD_SHARED)
    set(_lmmc_export_file "${CMAKE_CURRENT_BINARY_DIR}/lmmc.exports.def")
    file(WRITE "${_lmmc_export_file}" "EXPORTS\n")
    foreach(_lmmc_symbol IN LISTS _lmmc_export_symbols)
        file(APPEND "${_lmmc_export_file}" "    ${_lmmc_symbol}\n")
    endforeach()
    target_sources(lmmc PRIVATE "${_lmmc_export_file}")
elseif(APPLE AND LMMC_BUILD_SHARED)
    set(_lmmc_export_file "${CMAKE_CURRENT_BINARY_DIR}/lmmc.exports")
    file(WRITE "${_lmmc_export_file}" "")
    foreach(_lmmc_symbol IN LISTS _lmmc_export_symbols)
        file(APPEND "${_lmmc_export_file}" "_${_lmmc_symbol}\n")
    endforeach()
    target_link_options(lmmc PRIVATE
        "LINKER:-exported_symbols_list,${_lmmc_export_file}")
elseif(UNIX AND LMMC_BUILD_SHARED)
    set(_lmmc_export_file "${CMAKE_CURRENT_BINARY_DIR}/lmmc.exports.map")
    file(WRITE "${_lmmc_export_file}" "LMMC_1.0 {\n  global:\n")
    foreach(_lmmc_symbol IN LISTS _lmmc_export_symbols)
        file(APPEND "${_lmmc_export_file}" "    ${_lmmc_symbol};\n")
    endforeach()
    file(APPEND "${_lmmc_export_file}" "  local: *;\n};\n")
    target_link_options(lmmc PRIVATE
        "LINKER:--version-script,${_lmmc_export_file}")
endif()
target_include_directories(lmmc PRIVATE "${_lmmc_lmmp_include}/lmmp")
if(UNIX)
    target_link_libraries(lmmc PUBLIC m)
endif()

# LMMP headers select MinGW import declarations through this platform define.
if(WIN32)
    target_compile_definitions(lmmc PRIVATE LMMP_WINDOWS)
endif()

if(LMMC_ENABLE_CLANG_TIDY)
    find_program(LMMC_CLANG_TIDY_EXECUTABLE NAMES clang-tidy REQUIRED)
    set_property(TARGET lmmc PROPERTY C_CLANG_TIDY
        "${LMMC_CLANG_TIDY_EXECUTABLE};--config-file=${CMAKE_CURRENT_SOURCE_DIR}/.clang-tidy")
endif()

if(LMMC_DEBUG_LEAKS)
    target_compile_definitions(lmmc PUBLIC LMMC_DEBUG_LEAKS=1)
endif()


set(_lmmc_bundled_lmmp OFF)
if(TARGET lmmp)
    target_link_libraries(lmmc PUBLIC lmmp)
    message(STATUS "LMMC: linking against the lmmp CMake target")
elseif(LMMP_LIBRARY)
    set(_lmmc_lmmp_library "${LMMP_LIBRARY}")
    cmake_path(ABSOLUTE_PATH _lmmc_lmmp_library
        BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" NORMALIZE)
    if(NOT EXISTS "${_lmmc_lmmp_library}")
        message(FATAL_ERROR "LMMP_LIBRARY does not exist: ${_lmmc_lmmp_library}")
    endif()
    add_library(lmmp_imported UNKNOWN IMPORTED GLOBAL)
    set_target_properties(lmmp_imported PROPERTIES
        IMPORTED_LOCATION "${_lmmc_lmmp_library}"
        INTERFACE_INCLUDE_DIRECTORIES "${_lmmc_lmmp_include}"
    )
    target_link_libraries(lmmc PUBLIC lmmp_imported)
    message(STATUS "LMMC: linking prebuilt LMMP library: ${_lmmc_lmmp_library}")
elseif(EXISTS "${_lmmc_lmmp_root}/CMakeLists.txt")
    if(NOT DEFINED LMMP_BUILD_TESTS)
        set(LMMP_BUILD_TESTS OFF)
    endif()
    if(NOT DEFINED LMMP_BUILD_EXAMPLES)
        set(LMMP_BUILD_EXAMPLES OFF)
    endif()
    if(NOT DEFINED LMMP_BUILD_BENCHMARKS)
        set(LMMP_BUILD_BENCHMARKS OFF)
    endif()
    if(NOT DEFINED CACHE{LMMP_ASM})
        set(LMMP_ASM "${LMMC_LMMP_ASM}" CACHE STRING
            "Assembly mode: GENERIC | X64 | ARM64 | AUTO")
    endif()
    add_subdirectory("${_lmmc_lmmp_root}"
        "${CMAKE_CURRENT_BINARY_DIR}/lmmp" EXCLUDE_FROM_ALL)
    if(NOT TARGET liblmmp)
        message(FATAL_ERROR "LMMP did not define its liblmmp target")
    endif()
    target_include_directories(liblmmp PUBLIC
        $<BUILD_INTERFACE:${_lmmc_lmmp_include}>
        $<BUILD_INTERFACE:${_lmmc_lmmp_include}/lmmp>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/lmmp>
    )
    if(WIN32)
        target_compile_definitions(liblmmp INTERFACE LMMP_WINDOWS)
    endif()
    set(_lmmc_bundled_lmmp ON)
    set_target_properties(liblmmp PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            "${CMAKE_CURRENT_BINARY_DIR}/lmmp/dist/lmmp/bin"
        LIBRARY_OUTPUT_DIRECTORY
            "${CMAKE_CURRENT_BINARY_DIR}/lmmp/dist/lmmp/lib"
        ARCHIVE_OUTPUT_DIRECTORY
            "${CMAKE_CURRENT_BINARY_DIR}/lmmp/dist/lmmp/lib"
    )
    foreach(_lmmc_config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
        set_target_properties(liblmmp PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY_${_lmmc_config}
                "${CMAKE_CURRENT_BINARY_DIR}/lmmp/dist/lmmp/bin"
            LIBRARY_OUTPUT_DIRECTORY_${_lmmc_config}
                "${CMAKE_CURRENT_BINARY_DIR}/lmmp/dist/lmmp/lib"
            ARCHIVE_OUTPUT_DIRECTORY_${_lmmc_config}
                "${CMAKE_CURRENT_BINARY_DIR}/lmmp/dist/lmmp/lib"
        )
    endforeach()
    target_link_libraries(lmmc PUBLIC lmmp)
    message(STATUS "LMMC: building LMMP from ${_lmmc_lmmp_root}")
else()
    message(FATAL_ERROR
        "LMMP core was not found. Set LMMP_LIBRARY or provide LMMP_ROOT.")
endif()

# Embedded builds leave installation and export ownership to their parent.
# A relocatable standalone package includes the LMMP target built above;
# externally supplied targets and prebuilt libraries remain build-only modes.
if(PROJECT_IS_TOP_LEVEL AND _lmmc_bundled_lmmp)
    install(TARGETS lmmc liblmmp
        EXPORT LMMCTargets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )

    install(DIRECTORY include/lmmc/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/lmmc
        FILES_MATCHING PATTERN "*.h")
    file(GLOB _lmmc_lmmp_public_headers CONFIGURE_DEPENDS
        "${_lmmc_lmmp_include}/lmmp/*.h")
    install(FILES ${_lmmc_lmmp_public_headers}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/lmmp)

    install(EXPORT LMMCTargets
        NAMESPACE LMMC::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LMMC
    )
    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/LMMCConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/LMMCConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LMMC
    )
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/LMMCConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY ExactVersion
    )
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/LMMCConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/LMMCConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LMMC
    )
elseif(PROJECT_IS_TOP_LEVEL)
    message(STATUS
        "LMMC: standalone installation requires bundled LMMP; external LMMP is build-only")
endif()

lmmc_apply_instrumentation(lmmc)
if(TARGET liblmmp)
    lmmc_apply_instrumentation(liblmmp)
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
target_compile_options(lmmc PRIVATE ${_lmmc_warning_options})

function(lmmc_prepare_runtime target_name)
    if(WIN32)
        if(LMMC_BUILD_SHARED)
            add_custom_command(TARGET ${target_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "$<TARGET_FILE:lmmc>"
                    "$<TARGET_FILE_DIR:${target_name}>"
                VERBATIM
            )
        endif()
        if(TARGET liblmmp)
            add_custom_command(TARGET ${target_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "$<TARGET_FILE:liblmmp>"
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
    target_compile_definitions(lmmc PRIVATE LMMC_BUILD_TESTS=1)
endif()

if(LMMC_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
