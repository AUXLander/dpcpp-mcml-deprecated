cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

set(PROJECT_NAME dpcpp-mcml)

project(${PROJECT_NAME})

# Set DPC++ Compiler
set(CMAKE_C_COMPILER icx)
set(CMAKE_CXX_COMPILER icpx)

# Set default props for projects
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable boost
set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
set(Boost_NO_SYSTEM_PATHS TRUE) 

if (Boost_NO_SYSTEM_PATHS)
  set(BOOST_ROOT "C:\\Boost")
  set(BOOST_INCLUDE_DIRS "${BOOST_ROOT}/include")
  set(BOOST_LIBRARY_DIRS "${BOOST_ROOT}/lib")
endif (Boost_NO_SYSTEM_PATHS)

include_directories( ${BOOST_INCLUDE_DIRS}/boost-1_78 )

# enable DPC++ snippet
# find_package(IntelDPCPP REQUIRED)

# simple open gl view
add_executable(${PROJECT_NAME} 
               src/main.cpp 
               src/mcml.hpp
               src/io.cpp
               src/nr.cpp
               src/nr.cpp
               src/nr.hpp

               src/data/tracker.h

               src/io/io.hpp
               src/io/io.cpp

               src/math/math.cpp
               src/math/math.h
               src/math/random.cpp
               src/math/random.h
               
               src/dpcpp/utils.hpp
               )

add_executable( dpcpp-test 
                src/dpcpp/main.cpp
                src/dpcpp/utils.hpp )
