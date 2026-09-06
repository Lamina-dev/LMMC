cmake_minimum_required(VERSION 3.26)
project(lmmc_package_consumer LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

find_package(LMMC CONFIG REQUIRED)

add_executable(lmmc_package_consumer main.c)
target_link_libraries(lmmc_package_consumer PRIVATE LMMC::lmmc)

get_target_property(_lmmc_include_dirs LMMC::lmmc INTERFACE_INCLUDE_DIRECTORIES)
set(_lmmc_install_include "")
foreach(_include_dir IN LISTS _lmmc_include_dirs)
    if(EXISTS "${_include_dir}/lmmc/lmmc.h")
        set(_lmmc_install_include "${_include_dir}")
        break()
    endif()
endforeach()
if(NOT _lmmc_install_include)
    message(FATAL_ERROR "LMMC::lmmc does not expose an installed include directory")
endif()
if(EXISTS "${_lmmc_install_include}/lmmp/impl")
    message(FATAL_ERROR "Private LMMP implementation headers were installed")
endif()

file(GLOB _lmmc_headers CONFIGURE_DEPENDS
    "${_lmmc_install_include}/lmmc/*.h")
file(GLOB _lmmp_headers CONFIGURE_DEPENDS
    "${_lmmc_install_include}/lmmp/*.h")
foreach(_installed_header IN LISTS _lmmc_headers _lmmp_headers)
    file(RELATIVE_PATH _header "${_lmmc_install_include}" "${_installed_header}")
    string(MAKE_C_IDENTIFIER "${_header}" _target)
    set(_source "${CMAKE_CURRENT_BINARY_DIR}/standalone_${_target}.c")
    file(WRITE "${_source}" "#include <${_header}>\n")
    add_library("standalone_${_target}" OBJECT "${_source}")
    target_link_libraries("standalone_${_target}" PRIVATE LMMC::lmmc)
endforeach()
