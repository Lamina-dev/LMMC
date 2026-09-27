include_guard(GLOBAL)

function(lmmc_quality_require_tools)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c
            "import importlib.metadata; print(importlib.metadata.version('lizard') + ';' + importlib.metadata.version('pygments'))"
        RESULT_VARIABLE _status
        OUTPUT_VARIABLE _version
        ERROR_VARIABLE _error
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT _status EQUAL 0 OR NOT _version STREQUAL "1.24.0;2.21.0")
        message(FATAL_ERROR
            "Quality gates require lizard==1.24.0 and pygments==2.21.0 in ${Python3_EXECUTABLE}; "
            "found '${_version}'. Install cmake/quality-requirements.txt with "
            "pip --require-hashes. ${_error}")
    endif()
    set(Python3_EXECUTABLE "${Python3_EXECUTABLE}" PARENT_SCOPE)
endfunction()

function(_lmmc_quality_collect_targets directory output)
    get_property(_targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
    get_property(_subdirectories DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
    foreach(_directory IN LISTS _subdirectories)
        _lmmc_quality_collect_targets("${_directory}" _child_targets)
        list(APPEND _targets ${_child_targets})
    endforeach()
    set(${output} "${_targets}" PARENT_SCOPE)
endfunction()

function(_lmmc_quality_json_string value output)
    string(REPLACE "\\" "\\\\" _escaped "${value}")
    string(REPLACE "\"" "\\\"" _escaped "${_escaped}")
    string(REPLACE "\n" "\\n" _escaped "${_escaped}")
    string(REPLACE "\r" "\\r" _escaped "${_escaped}")
    string(REPLACE "\t" "\\t" _escaped "${_escaped}")
    set(${output} "\"${_escaped}\"" PARENT_SCOPE)
endfunction()

function(_lmmc_quality_register_project prefix root)
    set(_directory "${CMAKE_BINARY_DIR}/quality-consumers")
    file(MAKE_DIRECTORY "${_directory}")
    get_cmake_property(_variables VARIABLES)
    list(REMOVE_DUPLICATES _variables)
    set(_settings "")
    set(_separator "")
    foreach(_variable IN LISTS _variables)
        if(_variable MATCHES "^(CMAKE_|LMCAS_|LMMC_|LMMP_)")
            _lmmc_quality_json_string("${_variable}" _key)
            _lmmc_quality_json_string("${${_variable}}" _value)
            string(APPEND _settings "${_separator}${_key}:${_value}")
            set(_separator ",\n")
        endif()
    endforeach()
    _lmmc_quality_json_string("${root}" _source)
    _lmmc_quality_json_string("${CMAKE_BINARY_DIR}" _build)
    if(root STREQUAL CMAKE_SOURCE_DIR)
        set(_reuse true)
    else()
        set(_reuse false)
    endif()
    set(_spec "${_directory}/${prefix}-configuration.json")
    file(WRITE "${_spec}"
        "{\"project\":\"${prefix}\",\"source\":${_source},\"build\":${_build},"
        "\"reuse\":${_reuse},\"settings\":{${_settings}}}\n")
    set_property(GLOBAL PROPERTY "_lmmc_quality_spec_${prefix}" "${_spec}")
    set_property(GLOBAL PROPERTY "_lmmc_quality_reuse_${prefix}" "${_reuse}")
    get_property(_scheduled GLOBAL PROPERTY _lmmc_quality_consumers_scheduled)
    if(NOT _scheduled)
        set_property(GLOBAL PROPERTY _lmmc_quality_consumers_scheduled TRUE)
        set_property(GLOBAL PROPERTY _lmmc_quality_python "${Python3_EXECUTABLE}")
        cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}"
            CALL _lmmc_quality_prepare_consumers)
    endif()
endfunction()


function(_lmmc_quality_prepare_consumers)
    get_property(_projects GLOBAL PROPERTY _lmmc_quality_consumer_projects)
    if(NOT _projects)
        return()
    endif()
    list(REMOVE_DUPLICATES _projects)
    list(SORT _projects)
    get_property(_python GLOBAL PROPERTY _lmmc_quality_python)
    set(_parallel 4)
    set(_command "${_python}" -B
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/prepare_quality_consumers.py"
        --work-dir "${CMAKE_BINARY_DIR}/quality-consumers/$<CONFIG>"
        "--config=$<CONFIG>" --parallel "${_parallel}")
    set(_dependencies "")
    foreach(_project IN LISTS _projects)
        get_property(_spec GLOBAL PROPERTY "_lmmc_quality_spec_${_project}")
        if(NOT _spec)
            message(FATAL_ERROR "Missing package producer configuration for ${_project}")
        endif()
        list(APPEND _command --project-config "${_spec}")
        get_property(_reuse GLOBAL PROPERTY "_lmmc_quality_reuse_${_project}")
        if(_reuse)
            list(APPEND _dependencies "${_project}")
        endif()
    endforeach()
    add_custom_target(quality_prepare_consumers ALL
        COMMAND ${_command}
        VERBATIM
        COMMENT "Installing and checking quality package consumers")
    if(_dependencies)
        add_dependencies(quality_prepare_consumers ${_dependencies})
    endif()
    get_property(_gates GLOBAL PROPERTY _lmmc_quality_consumer_gates)
    foreach(_gate IN LISTS _gates)
        add_dependencies("${_gate}" quality_prepare_consumers)
    endforeach()
    get_property(_tests GLOBAL PROPERTY _lmmc_quality_consumer_tests)
    if(_tests)
        add_test(NAME quality_prepare_consumers COMMAND ${_command})
        set_tests_properties(quality_prepare_consumers PROPERTIES
            FIXTURES_SETUP quality_consumers
            RESOURCE_LOCK quality_consumers
            PROCESSORS "${_parallel}")
    endif()
endfunction()


# 在子目录注册完成后收集实际目标归属；未注册的源码由归属检查报错。
function(lmmc_add_quality_gates prefix root)
    _lmmc_quality_register_project("${prefix}" "${root}")
    _lmmc_quality_collect_targets("${root}" _targets)
    set(_entries "")
    foreach(_target IN LISTS _targets)
        get_target_property(_type "${_target}" TYPE)
        if(_type STREQUAL "UTILITY" OR _type STREQUAL "INTERFACE_LIBRARY")
            continue()
        endif()
        get_target_property(_source_dir "${_target}" SOURCE_DIR)
        file(RELATIVE_PATH _target_directory "${root}" "${_source_dir}")
        if(NOT _target_directory STREQUAL "" AND
           NOT _target_directory STREQUAL "LMMC" AND
           NOT _target_directory MATCHES "^(LMMC/)?(include|src|tests|test|examples|example|benchmarks)(/|$)")
            continue()
        endif()
        if(_target_directory MATCHES "(^|/)(build|dist|\\.git)(/|$)")
            continue()
        endif()
        get_target_property(_sources "${_target}" SOURCES)
        if(NOT _sources)
            continue()
        endif()
        foreach(_source IN LISTS _sources)
            # 对象聚合复用已编译的翻译单元。
            if(_source MATCHES "^\\$<TARGET_OBJECTS:[^>]+>$")
                continue()
            endif()
            if(_source MATCHES "\\$<")
                message(FATAL_ERROR
                    "Quality ownership cannot resolve source generator expression "
                    "'${_source}' on ${_target}; use explicit source membership "
                    "or a configure-time condition.")
            endif()
            cmake_path(ABSOLUTE_PATH _source BASE_DIRECTORY "${_source_dir}" NORMALIZE)
            file(RELATIVE_PATH _relative "${root}" "${_source}")
            if(NOT _relative MATCHES "^(LMMC/)?(include|src|tests|test|examples|example|benchmarks)/")
                continue()
            endif()
            if(_relative MATCHES "(^|/)(build|dist|\\.git)(/|$)")
                continue()
            endif()
            string(TOLOWER "${_source}" _lower_source)
            if(NOT _lower_source MATCHES "\\.(c|cc|cpp|cxx)$")
                continue()
            endif()
            get_source_file_property(_header_only "${_source}"
                TARGET_DIRECTORY "${_target}" HEADER_FILE_ONLY)
            if(_header_only)
                continue()
            endif()
            # 按扫描范围筛选文件，保留重复归属以检查每个编译副本。
            _lmmc_quality_json_string("${_source}" _path_json)
            _lmmc_quality_json_string("${_target}" _owner_json)
            list(APPEND _entries "    {\"path\":${_path_json},\"owner\":${_owner_json}}")
        endforeach()
    endforeach()
    list(JOIN _entries ",\n" _source_json)
    set(_manifest "${CMAKE_CURRENT_BINARY_DIR}/${prefix}_source_manifest.json")
    file(WRITE "${_manifest}" "{\n  \"sources\": [\n${_source_json}\n  ]\n}\n")

    set(_checker "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/check_quality.py")
    foreach(_mode IN ITEMS complexity structure ownership integers naming)
        if(_mode STREQUAL "complexity")
            set(_target_name "${prefix}_quality")
        elseif(_mode STREQUAL "integers")
            set(_target_name "${prefix}_integer_boundaries")
        elseif(_mode STREQUAL "naming")
            set(_target_name "${prefix}_naming")
        else()
            set(_target_name "${prefix}_source_${_mode}")
        endif()
        set(_command "${Python3_EXECUTABLE}" "${_checker}"
            --root "${root}" --mode "${_mode}")
        if(_mode STREQUAL "ownership")
            list(APPEND _command --manifest "${_manifest}")
            string(TOUPPER "${prefix}" _option_prefix)
            set("${_option_prefix}_QUALITY_CONSUMER_MANIFESTS" "" CACHE STRING
                "Source manifests from independently configured installed-package consumers")
            set(_consumer_manifests "${${_option_prefix}_QUALITY_CONSUMER_MANIFESTS}")
            set(_automatic_consumers FALSE)
            if(NOT _consumer_manifests)
                set(_automatic_consumers TRUE)
                set(_projects lmmc)
                if(prefix STREQUAL "lmcas")
                    list(APPEND _projects lmcas)
                endif()
                foreach(_project IN LISTS _projects)
                    list(APPEND _consumer_manifests
                        "${CMAKE_BINARY_DIR}/quality-consumers/$<CONFIG>/${_project}/source_ownership.json")
                endforeach()
                set_property(GLOBAL APPEND PROPERTY _lmmc_quality_consumer_projects ${_projects})
                set_property(GLOBAL APPEND PROPERTY _lmmc_quality_consumer_gates "${_target_name}")
            endif()
            foreach(_consumer_manifest IN LISTS _consumer_manifests)
                list(APPEND _command --manifest "${_consumer_manifest}")
            endforeach()
        endif()
        add_custom_target("${_target_name}"
            COMMAND ${_command}
                --report "${CMAKE_CURRENT_BINARY_DIR}/${_target_name}.json"
            VERBATIM
            COMMENT "Checking ${prefix} ${_mode}"
        )
        if(BUILD_TESTING)
            add_test(NAME "${_target_name}" COMMAND ${_command})
            if(_mode STREQUAL "ownership" AND _automatic_consumers)
                set_tests_properties("${_target_name}" PROPERTIES
                    FIXTURES_REQUIRED quality_consumers
                    RESOURCE_LOCK quality_consumers)
                set_property(GLOBAL APPEND PROPERTY _lmmc_quality_consumer_tests "${_target_name}")
            endif()
        endif()
    endforeach()
    if(BUILD_TESTING)
        add_test(NAME "${prefix}_naming_negative_fixtures"
            COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/naming_negative_fixtures.py"
                --root "${root}")
    endif()
endfunction()
