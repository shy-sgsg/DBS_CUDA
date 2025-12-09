cmake_minimum_required(VERSION 3.18)
project(dbs_core LANGUAGES CXX CUDA)

# ===== 编译选项 =====
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(LIBOMP_ROOT "/opt/homebrew/opt/libomp")  # OpenMP for Apple M1/M2

# ===== CUDA 配置 =====
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_60,code=sm_60")  # adjust for your GPU

if(APPLE)
  # Apple Silicon 默认 arm64；如需通用二进制可设 "arm64;x86_64"
  set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "" FORCE)
endif()

if(MSVC)
  add_compile_options(/W4 /permissive-)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ★ 允许开关 OpenMP（默认开启）
option(ENABLE_OPENMP "Enable OpenMP parallelization" ON)

# ★ 新增：通过开关控制是否无视构建类型强制定义 DEBUG
#   ON  -> 无论什么配置都定义 DEBUG（发布环境排障用）
#   OFF -> 正常：Debug 定义 DEBUG；Release/RelWithDebInfo/MinSizeRel 定义 NDEBUG
option(ENABLE_DEBUG_LOG "Force define DEBUG macro for verbose logging" OFF)

# ===== 依赖库 =====
# TinyXML2
find_package(TinyXML2 REQUIRED)
message(STATUS "TinyXML2 found: ${TinyXML2_DIR}")

# ===== 依赖库：OpenMP（可选） =====
if(ENABLE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
  else()
    message(WARNING "OpenMP not found; continuing without -fopenmp")
  endif()
endif()

# ===== FFTW3f (single-precision) =====
set(FFTW3F_TARGET "")
set(FFTW3F_THREADS_TARGET "")

if(NOT FFTW3F_TARGET)
  find_library(FFTW3F_LIB NAMES fftw3f PATHS /opt/homebrew/lib /usr/local/lib /usr/lib)
  if(FFTW3F_LIB)
    add_library(FFTW3::fftw3f INTERFACE IMPORTED)
    target_link_libraries(FFTW3::fftw3f INTERFACE "${FFTW3F_LIB}")
    set(FFTW3F_TARGET FFTW3::fftw3f)
    message(STATUS "Using FFTW by manual search: ${FFTW3F_LIB}")
  endif()
endif()

if(NOT FFTW3F_THREADS_TARGET)
  find_library(FFTW3F_THREADS_LIB NAMES fftw3f_threads PATHS /usr/lib64 /usr/lib /usr/local/lib /opt/homebrew/lib)
  if(FFTW3F_THREADS_LIB)
    add_library(FFTW3::fftw3f_threads INTERFACE IMPORTED)
    target_link_libraries(FFTW3::fftw3f_threads INTERFACE "${FFTW3F_THREADS_LIB}")
    set(FFTW3F_THREADS_TARGET FFTW3::fftw3f_threads)
    message(STATUS "Using FFTW threads by manual search: ${FFTW3F_THREADS_LIB}")
  endif()
endif()

# ===== FFTW3f (single-precision) =====
set(FFTW3_TARGET "")
set(FFTW3_THREADS_TARGET "")

if(NOT FFTW3_TARGET)
  find_library(FFTW3_LIB NAMES fftw3 PATHS /opt/homebrew/lib /usr/local/lib /usr/lib)
  if(FFTW3_LIB)
    add_library(FFTW3::fftw3 INTERFACE IMPORTED)
    target_link_libraries(FFTW3::fftw3 INTERFACE "${FFTW3_LIB}")
    set(FFTW3_TARGET FFTW3::fftw3)
    message(STATUS "Using FFTW by manual search: ${FFTW3_LIB}")
  endif()
endif()

if(NOT FFTW3_THREADS_TARGET)
  find_library(FFTW3_THREADS_LIB NAMES fftw3_threads PATHS /usr/lib64 /usr/lib /usr/local/lib /opt/homebrew/lib)
  if(FFTW3_THREADS_LIB)
    add_library(FFTW3::fftw3_threads INTERFACE IMPORTED)
    target_link_libraries(FFTW3::fftw3_threads INTERFACE "${FFTW3_THREADS_LIB}")
    set(FFTW3_THREADS_TARGET FFTW3::fftw3_threads)
    message(STATUS "Using FFTW threads by manual search: ${FFTW3_THREADS_LIB}")
  endif()
endif()

# ===== 线程库（fftw3_threads 需要）=====
find_package(Threads REQUIRED)

# ===== CUDA 和 cuFFT =====
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA Toolkit found: ${CUDAToolkit_VERSION}")
message(STATUS "cuFFT library: ${CUDA_cufft_LIBRARY}")

# ===== 源码文件 =====
# 建议显式列出，也可用 GLOB
set(SRC
  src/main.cpp
  src/DbsStitcher.cpp
  src/DbsIO.cpp
  src/buildMosaic.cpp
  src/azFftShiftAndRecenterMat.cpp
  src/mapPosToBeam.cpp
  src/loadXML.cpp
  src/processAllBeams.cpp
  src/lodepng.cpp  # PNG写出
  src/RangeCompressCuFFT.cu  # cuFFT 距离脉压
  # ... 其余源文件
)

add_executable(dbs_core ${SRC})

# Test executable for comparing CPU(FFTW) and GPU(cuFFT)
add_executable(test_range_compress
  src/test_range_compress.cpp
  src/RangeCompressCuFFT.cu
  src/DbsIO.cpp
)

target_include_directories(test_range_compress
  PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(test_range_compress
  PRIVATE
    ${FFTW3F_TARGET}
    ${FFTW3_TARGET}
    tinyxml2::tinyxml2
    Threads::Threads
    CUDA::cufft
)

target_include_directories(dbs_core
  PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# OpenMP
if(ENABLE_OPENMP AND APPLE)
  target_compile_options(dbs_core PRIVATE -Xpreprocessor -fopenmp)
  target_link_libraries(dbs_core PRIVATE "${LIBOMP_ROOT}/lib/libomp.dylib")
  target_include_directories(dbs_core PRIVATE "${LIBOMP_ROOT}/include")
endif()

# ===== 链接库 =====
target_link_libraries(dbs_core
  PRIVATE
    ${FFTW3F_TARGET}
    ${FFTW3_TARGET}
    tinyxml2::tinyxml2
    Threads::Threads
    CUDA::cufft
)

# 如果找到了 fftw3_threads，则一并链接（用于 colfft 的多线程）
if(FFTW3_THREADS_TARGET)
  target_link_libraries(dbs_core PRIVATE ${FFTW3_THREADS_TARGET})
  target_link_libraries(dbs_core PRIVATE ${FFTW3F_THREADS_TARGET})
endif()

# ===== 宏定义：DEBUG / NDEBUG =====
if(ENABLE_DEBUG_LOG)
  target_compile_definitions(dbs_core PRIVATE DEBUG)
else()
  target_compile_definitions(dbs_core PRIVATE
    $<$<CONFIG:Debug>:DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
    $<$<CONFIG:RelWithDebInfo>:NDEBUG>
    $<$<CONFIG:MinSizeRel>:NDEBUG>
  )
endif()

# Debug 友好
if(NOT MSVC)
  target_compile_options(dbs_core PRIVATE
    $<$<CONFIG:Debug>:-O0 -g3 -fno-omit-frame-pointer>
  )
endif()

# Windows: /utf-8
if(MSVC)
  target_compile_options(dbs_core PRIVATE /utf-8)
endif()

# ===== 可选：安装规则 =====
# install(TARGETS dbs_core RUNTIME DESTINATION bin)
