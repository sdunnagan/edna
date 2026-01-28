cmake_minimum_required(VERSION 3.18)
project(edna LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fno-omit-frame-pointer")
set(CMAKE_C_FLAGS_DEBUG   "-g -O0 -fno-omit-frame-pointer")

option(EDNA_USE_CUDA "Link CUDA toolkit libs (needed if deps built with CUDA and you want link-time enforcement)" ON)
set(EDNA_PREFIX "$ENV{EDNA_TOP_DIR}/deps/install" CACHE PATH "Edna deps install prefix")
set(EDNA_INCLUDE_DIR "${EDNA_PREFIX}/include")
set(EDNA_LIB_DIR     "${EDNA_PREFIX}/lib")

# Make CMake able to find package configs installed under EDNA_PREFIX
# whisper installs: ${prefix}/lib/cmake/whisper/whisper-config.cmake
list(PREPEND CMAKE_PREFIX_PATH "${EDNA_PREFIX}")

#------------------------------------------------------------------------------
# Dependencies
#------------------------------------------------------------------------------

find_package(ALSA REQUIRED)
find_package(Threads REQUIRED)

if(EDNA_USE_CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

# Prefer installed CMake packages when present
find_package(whisper CONFIG REQUIRED)
find_package(ggml    CONFIG REQUIRED)

# llama.cpp does not reliably ship a CMake package in all layouts; link by path.
set(LLAMA_LIBRARY_SO "${EDNA_LIB_DIR}/libllama.so")
set(LLAMA_LIBRARY_A  "${EDNA_LIB_DIR}/libllama.a")

if(EXISTS "${LLAMA_LIBRARY_SO}")
  set(LLAMA_LIBRARY "${LLAMA_LIBRARY_SO}")
elseif(EXISTS "${LLAMA_LIBRARY_A}")
  set(LLAMA_LIBRARY "${LLAMA_LIBRARY_A}")
else()
  message(FATAL_ERROR "Missing llama library at: ${LLAMA_LIBRARY_SO} or ${LLAMA_LIBRARY_A}")
endif()

# libfvad (static)
set(FVAD_LIBRARY "${EDNA_LIB_DIR}/libfvad.a")
if(NOT EXISTS "${FVAD_LIBRARY}")
  message(FATAL_ERROR "Missing fvad at: ${FVAD_LIBRARY}")
endif()

#------------------------------------------------------------------------------
# Target
#------------------------------------------------------------------------------

add_executable(edna
  src/main.cpp
  src/asr_whisper.cpp
  src/llm_llama.cpp
  src/tts_coqui.cpp
  src/state_machine.cpp
)

# Extra debug niceties regardless of build type (harmless in Release)
target_compile_options(edna PRIVATE
  -fno-omit-frame-pointer
)

target_include_directories(edna PRIVATE
  ${ALSA_INCLUDE_DIRS}
  "${EDNA_INCLUDE_DIR}"
  "${CMAKE_SOURCE_DIR}/src"
)

# RPATH: make runtime self-contained (no LD_LIBRARY_PATH)
# Use ":" not ";" because this string goes into ELF.
set(_edna_rpath "${EDNA_LIB_DIR}")

set_target_properties(edna PROPERTIES
  BUILD_RPATH   "${_edna_rpath}"
  INSTALL_RPATH "${_edna_rpath}"
)

# Link order: your objects, then libraries.
# Link whisper as a real dependency so its transitive ggml deps are pulled in consistently.
target_link_libraries(edna PRIVATE
  ${ALSA_LIBRARIES}
  Threads::Threads
  dl
  m
  "${FVAD_LIBRARY}"
  whisper
  "${LLAMA_LIBRARY}"
)

# If you're worried about random system libs being pulled in, keep as-needed.
target_link_options(edna PRIVATE "-Wl,--as-needed")

# CUDA toolkit libs: optional.
# Deps (whisper/ggml/llama) typically DT_NEEDED what they need already, but this
# forces the toolkit presence at link time.
if(EDNA_USE_CUDA)
  target_link_libraries(edna PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cublasLt
  )
endif()

#------------------------------------------------------------------------------
# Diagnostics
#------------------------------------------------------------------------------

message(STATUS "prefix:  ${EDNA_PREFIX}")
message(STATUS "include: ${EDNA_INCLUDE_DIR}")
message(STATUS "libdir:  ${EDNA_LIB_DIR}")
message(STATUS "rpath:   ${_edna_rpath}")
message(STATUS "fvad:    ${FVAD_LIBRARY}")
message(STATUS "llama:   ${LLAMA_LIBRARY}")
message(STATUS "whisper: (CMake package target 'whisper')")
message(STATUS "ggml:    (CMake package target(s) from ggml-config.cmake)")
