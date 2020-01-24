set(BOOST_ROOT "/home/ixtiyor/boost")
set(BOOSTROOT "/home/ixtiyor/boost")
set(Boost_DIR "/home/ixtiyor/boost")

include_directories(${MYSQLCPPCONN_INCLUDE_DIRS})

add_library(ecto_streamer_serv STATIC
  connection.cpp
  mime_types.cpp
  reply.cpp
  request_handler.cpp
  request_parser.cpp
  server.cpp
  mjpeg_server.cpp
  request_objects.cpp
  dbconnector.cpp
    )

target_link_libraries(ecto_streamer_serv
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES}
  mysqlclient
  mysqlcppconn
  )