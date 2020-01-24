#include <iostream>
#include <string>
#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/lexical_cast.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <pthread.h>
#include <signal.h>
#include <boost/foreach.hpp>
#include <boost/algorithm/string.hpp>

#include "mjpeg_server.hpp"
#include "mysql_connection.h"
#include "server.hpp"
#include "mime_types.hpp"

#define BOUNDARYSTRING "--BOUNDARYSTRING\r\n"

namespace http
{
  namespace server
  {
     det_struct global_detection_data;
     DatabaseFaces db_tools;

    int registering = 0;
    std::string url = "tcp://127.0.0.1:3306";
    std::string username = "root";
    std::string password = "1";
    // sql::Driver driver = sql::mysql::get_driver_instance();
    std::string database_name = "faces";
    streamer::streamer():watchers_(0)
    {
      if(!db_tools.checkConnection()){
        db_tools.connect(url, username, password, database_name);
      }

      // if(!cap.isOpened())
      // cap.open("../test.mp4");

      //test image.
      cv::Mat image;
      
      // cap.read(image);
      // cv::imshow("img", image);
      // cv::waitKey(3);
      // if(image.empty())
      
      image =cv::Mat(cv::Size(640, 480), CV_8UC3, cv::Scalar(std::rand() % 255, std::rand() % 255, std::rand() % 255));
      global_detection_data.draw_frame = image;
      post_image(95);
    }

    void
    streamer::handle_initial_header_write(connection_ptr conn, const boost::system::error_code& e)
    {
      if (!e)
      {
        try
        {
          reply& rep = conn->reply_;

          //send the boundary synchronously
          rep.headers.clear();
          rep.content = BOUNDARYSTRING;
          boost::asio::write(conn->socket_, rep.content_to_buffers());

          //scope for lock.
          {
            boost::unique_lock<boost::mutex> lock(mtx_);
            ++watchers_;
            while (!cond_.timed_wait(lock, boost::posix_time::milliseconds(1000)))
            {
              std::cerr << "Time out occured." << std::endl;
            }
            rep.content.clear(); //fill with image here.
            rep.content.append(jpg_buffer_.begin(), jpg_buffer_.end());
            --watchers_;
          }

          rep.headers.push_back(header("Content-type", "image/jpg"));
          rep.headers.push_back(header("Content-Length", boost::lexical_cast<std::string>(rep.content.size())));

          std::vector<boost::asio::const_buffer> buffers = rep.headers_to_buffers();
          BOOST_FOREACH(const boost::asio::const_buffer& buff, rep.content_to_buffers())
              {
                buffers.push_back(buff);
              }
          buffers.push_back(boost::asio::buffer(misc_strings::crlf));

          //async it, on finish it will come back to this callback.
          conn->async_write(
              buffers,
              boost::bind(&streamer::handle_initial_header_write, shared_from_this(), conn,
                          boost::asio::placeholders::error));
        } catch (std::runtime_error& e)
        {
          std::cerr << e.what() << std::endl;
        }
      }
    }

    void
    streamer::handle_stream(connection_ptr conn, const request& req, const std::string& path, const std::string& query,
                            reply& rep)
    {
      std::cout << path << "\t handle_stream \t" << query<< BOUNDARYSTRING<<  std::endl;
      // Fill out the reply to be sent to the client.
      rep.status = reply::ok;
      rep.headers.clear();
      rep.content.clear();
      rep.headers.push_back(header("Connection", "close"));
      rep.headers.push_back(header("Max-Age", "0"));
      rep.headers.push_back(header("Expires", "0"));
      rep.headers.push_back(header("Cache-Control", "no-cache"));
      rep.headers.push_back(header("Pragma", "no-cache"));
      rep.headers.push_back(header("Content-Type", "multipart/x-mixed-replace; boundary=" BOUNDARYSTRING));
      conn->async_write(
          rep.to_buffers(),
          boost::bind(&streamer::handle_initial_header_write, shared_from_this(), conn,
                      boost::asio::placeholders::error));
    }

    int
    streamer::post_image(int quality, bool wait)
    {
      while (!watchers_ && wait && !boost::this_thread::interruption_requested())
      {
        boost::this_thread::sleep(boost::posix_time::milliseconds(100));
      }
      int watchers = watchers_;
      {
        boost::lock_guard<boost::mutex> lock(mtx_);
        jpg_buffer_.clear();

        std::vector<int> params(2);
        params[0] = CV_IMWRITE_JPEG_QUALITY;
        params[1] = std::min(std::max(double(quality),0.0),100.0);
        cv::imencode(".jpg", global_detection_data.draw_frame, jpg_buffer_, params);
      }

      cond_.notify_all();
      return watchers;
    }
    void handle_list_all(server_ptr serv, connection_ptr conn, const request& req, const std::string& path,
                    const std::string&query, reply&rep)
    {
      //this redirect the browser so it thinks the stream url is unique
      rep.status = reply::ok;
      rep.headers.clear();
      rep.content = boost::join(list_all_streams(serv, "/_all", path), "\n") + "\n";
      rep.headers.push_back(header("Content-Length", boost::lexical_cast<std::string>(rep.content.size())));
      rep.headers.push_back(header("Content-Type", mime_types::extension_to_type("txt")));
      conn->async_write(rep.to_buffers());
    }

    void
    handle_redirect(connection_ptr conn, const request& req, const std::string& path, const std::string&query,
                    reply&rep)
    {
      std::cout << path << "\t handle_redirect \t" << query << std::endl;
      //this redirect the browser so it thinks the stream url is unique
      rep.status = reply::moved_temporarily;
      rep.headers.clear();
      rep.content.clear();
      rep.headers.push_back(header("Location", "/_stream" + path + "/" + boost::lexical_cast<std::string>(std::rand())));
      // rep.headers.push_back(header("Location", path));
      conn->async_write(rep.to_buffers());
    }

    std::vector<std::string>
    list_all_streams(server_ptr serv, const std::string& prefix, const std::string&path)
    {
      std::string rpath = path;
      boost::replace_first(rpath, prefix, "");
      std::vector<std::string> images;
      try
      {
        images = serv->get_request_handlers("GET", "/_stream" + rpath + "(.*)");
        BOOST_FOREACH(std::string& s,images)
            {
              boost::replace_first(s, "/_stream/", "/");
              boost::replace_last(s, "/(.*)", "");
            }

      } catch (std::exception& e)
      {
        std::cerr << e.what() << std::endl;
      }
      return images;
    }

    
    void
    handle_register(server_ptr serv, connection_ptr conn, const request& req, const std::string& path,
                    const std::string&query, reply&rep)
    {
      std::cout <<"registering " << registering <<std::endl;
      //this redirect the browser so it thinks the stream url is unique
      rep.status = reply::no_content;
      rep.headers.clear();
      // rep.content = "registered";
      // rep.headers.push_back(header("Content-Length", boost::lexical_cast<std::string>(rep.content.size())));
      // rep.headers.push_back(header("Content-Type", mime_types::extension_to_type("txt")));
      
      conn->async_write(rep.to_buffers());
    }
    
    void register_streamer(server_ptr serv, streamer_ptr stmr, const std::string& path)
    {
      std::cout << path << "\t register_streamer \t"<< std::endl ;
      
      serv->register_request_handler(path, "GET", handle_redirect);

      serv->register_request_handler("/_stream" + path + "/(.*)", "GET",
                                     boost::bind(&streamer::handle_stream, stmr, _1, _2, _3, _4, _5));
    }

    void init_register(server_ptr serv)
    { 
      serv->register_request_handler("/_register(.*)",  "GET", boost::bind(handle_register, serv, _1, _2, _3, _4, _5));
      serv->register_request_handler("/_register(.*)",  "POST", boost::bind(handle_register, serv, _1, _2, _3, _4, _5));
    }

    server_ptr get_server(const std::string& address, const std::string& port, const std::string& doc_root,
                          std::size_t thread_pool_size){
      std::cout << address << "\t init_streaming_server \t" << port<< std::endl;
      server_ptr s(new server(address, port, doc_root, thread_pool_size));
      return s;
    }
    
    server_ptr
    init_streaming_server(const std::string& address, const std::string& port, const std::string& doc_root,
                          std::size_t thread_pool_size)
    {
      server_ptr s(new server(address, port, doc_root, thread_pool_size));
      s->register_request_handler("/_all(.*)", "GET", boost::bind(handle_list_all, s, _1, _2, _3, _4, _5));
      s->register_request_handler("/_all(.*)", "POST", boost::bind(handle_list_all, s, _1, _2, _3, _4, _5));
      return s;
    }
  }
}