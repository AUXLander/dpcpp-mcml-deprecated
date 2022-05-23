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

#find_package(Boost 1.78.0 COMPONENTS system) 

include_directories( ${Boost_INCLUDE_DIRS} )

#target_link_libraries( ${PROJECT_NAME} ${Boost_SYSTEM_LIBRARY} )

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
               )

add_executable( dpcpp-test src/dpcpp/main.cpp )
