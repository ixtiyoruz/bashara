# Install script for directory: /home/ixtiyor/Downloads/libfreenect2

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/ixtiyor/freenect2")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "RelWithDebInfo")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libfreenect2.so.0.2.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libfreenect2.so.0.2"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libfreenect2.so"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHECK
           FILE "${file}"
           RPATH "/usr/lib/x86_64-linux-gnu")
    endif()
  endforeach()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES
    "/home/ixtiyor/Downloads/libfreenect2/build/lib/libfreenect2.so.0.2.0"
    "/home/ixtiyor/Downloads/libfreenect2/build/lib/libfreenect2.so.0.2"
    "/home/ixtiyor/Downloads/libfreenect2/build/lib/libfreenect2.so"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libfreenect2.so.0.2.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libfreenect2.so.0.2"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libfreenect2.so"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHANGE
           FILE "${file}"
           OLD_RPATH ":::::::::::::::::::::::::"
           NEW_RPATH "/usr/lib/x86_64-linux-gnu")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/home/ixtiyor/Downloads/libfreenect2/include/libfreenect2" REGEX "/[^/]*\\.in$" EXCLUDE)
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/home/ixtiyor/Downloads/libfreenect2/build/libfreenect2")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/freenect2" TYPE FILE FILES "/home/ixtiyor/Downloads/libfreenect2/build/freenect2Config.cmake")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/freenect2" TYPE FILE FILES "/home/ixtiyor/Downloads/libfreenect2/build/freenect2ConfigVersion.cmake")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/ixtiyor/Downloads/libfreenect2/build/freenect2.pc")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/OpenNI2/Drivers/libfreenect2-openni2.so.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/OpenNI2/Drivers/libfreenect2-openni2.so"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHECK
           FILE "${file}"
           RPATH "/home/ixtiyor/freenect2/lib")
    endif()
  endforeach()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/OpenNI2/Drivers" TYPE SHARED_LIBRARY FILES
    "/home/ixtiyor/Downloads/libfreenect2/build/lib/libfreenect2-openni2.so.0"
    "/home/ixtiyor/Downloads/libfreenect2/build/lib/libfreenect2-openni2.so"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/OpenNI2/Drivers/libfreenect2-openni2.so.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/OpenNI2/Drivers/libfreenect2-openni2.so"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHANGE
           FILE "${file}"
           OLD_RPATH "/home/ixtiyor/Downloads/libfreenect2/build/lib:"
           NEW_RPATH "/home/ixtiyor/freenect2/lib")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/ixtiyor/Downloads/libfreenect2/build/doc/cmake_install.cmake")
  include("/home/ixtiyor/Downloads/libfreenect2/build/examples/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/ixtiyor/Downloads/libfreenect2/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
