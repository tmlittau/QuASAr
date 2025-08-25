cmake_minimum_required(VERSION 3.20)
project(quasar_convert LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(QUASAR_USE_STIM "Enable Stim support" OFF)
option(QUASAR_USE_MQT "Enable MQT DD support" OFF)

include(FetchContent)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

if(QUASAR_USE_STIM)
  # Stim requires C++20 for std::span and other features
  set(CMAKE_CXX_STANDARD 20)
endif()

if(QUASAR_USE_STIM)
    FetchContent_Declare(
        stim
        GIT_REPOSITORY https://github.com/quantumlib/Stim.git
        GIT_TAG v1.15.0
    )

FetchContent_Populate(stim)
add_subdirectory(${stim_SOURCE_DIR} ${stim_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

if(QUASAR_USE_MQT)
FetchContent_Declare(
    mqtcore
    GIT_REPOSITORY https://github.com/munich-quantum-toolkit/core.git
    GIT_TAG v1.11.1
    GIT_SHALLOW OFF
    GIT_SUBMODULES "extern/dd_package"
)
FetchContent_Populate(mqtcore)
add_subdirectory(${mqtcore_SOURCE_DIR}/extern/dd_package ${mqtcore_BINARY_DIR}/dd_package EXCLUDE_FROM_ALL)
endif()

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(quasar_convert MODULE conversion_engine.cpp binding.cpp)

if(QUASAR_USE_STIM)
target_link_libraries(quasar_convert PRIVATE libstim)
target_compile_definitions(quasar_convert PRIVATE QUASAR_USE_STIM)
endif()

if(QUASAR_USE_MQT)
target_link_libraries(quasar_convert PRIVATE MQT::DDPackage)
target_include_directories(quasar_convert PRIVATE ${mqtcore_SOURCE_DIR}/extern/dd_package/include)
target_compile_definitions(quasar_convert PRIVATE QUASAR_USE_MQT)
endif()

enable_testing()
add_test(NAME conversion_engine_tests
    COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_engine.py)
set_tests_properties(conversion_engine_tests PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}")

add_test(NAME stim_mqt_tests
    COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_stim_mqt.py)
set_tests_properties(stim_mqt_tests PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}")
