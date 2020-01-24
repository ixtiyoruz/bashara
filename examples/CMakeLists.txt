cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
foreach(path ${CMAKE_PREFIX_PATH})
  message("Path: " ${path})
endforeach(path)

if(WIN32 AND NOT MINGW)
  if(NOT DEFINED CMAKE_DEBUG_POSTFIX)
    set(CMAKE_DEBUG_POSTFIX "d")
  endif()
endif()

IF(NOT DEFINED CMAKE_BUILD_TYPE)
  # No effect for multi-configuration generators (e.g. for Visual Studio)
  SET(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose: RelWithDebInfo Release Debug MinSizeRel None")
ENDIF()

PROJECT(libfreenect2_examples)

SET(MY_DIR ${libfreenect2_examples_SOURCE_DIR})
SET(DEPENDS_DIR "${MY_DIR}/../depends" CACHE STRING "Dependency directory")
find_package(OpenCV REQUIRED)

set(Caffe_DIR "/home/ixtiyor/Downloads/caffe/build/include") 
#set(BOOST_ROOT "/home/ixtiyor/boost")
#set(BOOSTROOT "/home/ixtiyor/boost")
#set(Boost_DIR "/home/ixtiyor/boost")

#https://nanxiao.me/en/be-wary-of-boost-not-boost-in-cmakelists-txt

FIND_PACKAGE( Boost 1.58.0 REQUIRED COMPONENTS  program_options regex thread system  )
MESSAGE([Main] "Boost_INCLUDE_DIR"=${Boost_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
include_directories(${MYSQLCPPCONN_INCLUDE_DIRS})

find_package(Caffe)
add_definitions(${Caffe_DEFINITIONS})
MESSAGE( [Main] " Caffe_INCLUDE_DIRS = ${Caffe_INCLUDE_DIRS}")
MESSAGE( [Main] " Caffe_DEFINITIONS = ${Caffe_DEFINITIONS}")
MESSAGE( [Main] " Caffe_LIBRARIES = ${Caffe_LIBRARIES}") # caffe
MESSAGE( [Main] " Caffe_CPU_ONLY = ${Caffe_CPU_ONLY}")
MESSAGE( [Main] " Caffe_HAVE_CUDA = ${Caffe_HAVE_CUDA}")
MESSAGE( [Main] " Caffe_HAVE_CUDNN = ${Caffe_HAVE_CUDNN}")
include_directories(${Caffe_INCLUDE_DIRS})

OPTION(ENABLE_OPENGL "Enable OpenGL support" ON)
# default libraries
set(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_FLAGS -pthread)

# The example build system is standalone and will work out-of-tree with these files copied
SET(freenect2_ROOT_DIR ${MY_DIR}/..)
SET(flextGL_SOURCES ${freenect2_ROOT_DIR}/src/flextGL.cpp)
SET(flextGL_INCLUDE_DIRS ${freenect2_ROOT_DIR}/src) # for flextGL.h

FIND_PACKAGE(PkgConfig)    # try find PKGConfig as it will be used if found
LIST(APPEND CMAKE_MODULE_PATH ${freenect2_ROOT_DIR}/cmake_modules) # FindGLFW3.cmake

IF(TARGET freenect2)
  MESSAGE(STATUS "Using in-tree freenect2 target")
  SET(freenect2_LIBRARIES freenect2)
  SET(freenect2_DLLS ${LIBFREENECT2_DLLS})
ELSE()
  FIND_PACKAGE(freenect2 REQUIRED)
  # Out-of-tree build will have to have DLLs manually copied.
ENDIF()

INCLUDE_DIRECTORIES(
  ${freenect2_INCLUDE_DIR}
)



SET(Protonect_src
  Protonect.cpp
)

SET(Protonect_LIBRARIES
  ${freenect2_LIBRARIES}
)

SET(Protonect_DLLS
  ${freenect2_DLLS}
)

IF(ENABLE_OPENGL)
  FIND_PACKAGE(GLFW3)
  FIND_PACKAGE(OpenGL)
  IF(GLFW3_FOUND AND OPENGL_FOUND)
    INCLUDE_DIRECTORIES(
      ${GLFW3_INCLUDE_DIRS}
      ${flextGL_INCLUDE_DIRS}
    )

    LIST(APPEND Protonect_DLLS ${GLFW3_DLL})
    LIST(APPEND Protonect_src
      viewer.cpp
      ${flextGL_SOURCES}
    )
    LIST(APPEND Protonect_LIBRARIES
      ${GLFW3_LIBRARIES}
      ${OPENGL_gl_LIBRARY}
    )
    ADD_DEFINITIONS(-DEXAMPLES_WITH_OPENGL_SUPPORT=1)
  ENDIF()
ENDIF(ENABLE_OPENGL)

# now it is time for streamer library
add_subdirectory(streamer)
add_subdirectory(json_spirit)


ADD_EXECUTABLE(Protonect
  ${Protonect_src}
)
TARGET_LINK_LIBRARIES(Protonect 
  ${Protonect_LIBRARIES}
  ${OpenCV_LIBS}
  ${Caffe_LIBRARIES}
  ${Boost_LIBRARIES}
  ecto_streamer_serv
)

IF(WIN32)
  INSTALL(TARGETS Protonect DESTINATION bin)
  LIST(REMOVE_DUPLICATES Protonect_DLLS)
  FOREACH(FILEI ${Protonect_DLLS})
    ADD_CUSTOM_COMMAND(TARGET Protonect POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${FILEI} $<TARGET_FILE_DIR:Protonect>
    )
  ENDFOREACH(FILEI)
  INSTALL(FILES ${Protonect_DLLS} DESTINATION bin)
ENDIF()



