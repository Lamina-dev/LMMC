cmake_minimum_required(VERSION 3.26)
project(lmmc_package_consumer LANGUAGES C CXX)

set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(LMMC CONFIG REQUIRED)
include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/PackageConsumerChecks.cmake")
package_check_target(LMMC::lmmc c_std_11)
package_check_target(LMMC::liblmmp "")

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
foreach(_private_dir internal visitors lmmp/impl lmmc/internal)
    if(EXISTS "${_lmmc_install_include}/${_private_dir}")
        message(FATAL_ERROR "Private header directory was installed: ${_private_dir}")
    endif()
endforeach()

package_check_headers(LMMC::lmmc "${_lmmc_install_include}" "lmmc/*.h" C CXX)
package_check_headers(LMMC::liblmmp "${_lmmc_install_include}" "lmmp/*.h" C CXX)
package_check_report(lmmc_package_consumer
    LMMC::lmmc LMMC::liblmmp)
