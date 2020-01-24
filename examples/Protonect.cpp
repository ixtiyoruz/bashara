/*
 * This file is part of the OpenKinect Project. http://www.openkinect.org
 *
 * Copyright (c) 2011 individual OpenKinect contributors. See the CONTRIB file
 * for details.
 *
 * This code is licensed to you under the terms of the Apache License, version
 * 2.0, or, at your option, the terms of the GNU General Public License,
 * version 2.0. See the APACHE20 and GPL2 files for the text of the licenses,
 * or the following URLs:
 * http://www.apache.org/licenses/LICENSE-2.0
 * http://www.gnu.org/licenses/gpl-2.0.txt
 *
 * If you redistribute this file in source form, modified or unmodified, you
 * may:
 *   1) Leave this header intact and distribute it under the same terms,
 *      accompanying it with the APACHE20 and GPL20 files, or
 *   2) Delete the Apache 2.0 clause and accompany it with the GPL2 file, or
 *   3) Delete the GPL v2 clause and accompany it with the APACHE20 file
 * In all cases you must keep the copyright notice intact and include a copy
 * of the CONTRIB file.
 *
 * Binary distributions must follow the binary distribution requirements of
 * either License.
 */

/** @file Protonect.cpp Main application file. */

#include <iostream>
#include <cstdlib>
#include <string> 
#include <signal.h>
#include <caffe/caffe.hpp>
#include <thread>
#include <atomic> 
#include <bits/stdc++.h> 
#include <iostream> 
#include <sys/stat.h> 
#include <sys/types.h> 
#include "streamer/mjpeg_server.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/face.hpp>
#include <opencv2/objdetect.hpp>
#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/lexical_cast.hpp>
/// [headers]
#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/registration.h>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/logger.h>

#include "streamer/server.hpp"
#include "streamer/mime_types.hpp"

using namespace cv;
using namespace cv::dnn;
using namespace http;
using namespace server;

bool protonect_shutdown = false; ///< Whether the running application should shut down.

void sigint_handler(int s)
{
  protonect_shutdown = true;
}

bool protonect_paused = false;
libfreenect2::Freenect2Device *devtopause;

//Doing non-trivial things in signal handler is bad. If you want to pause,
//do it in another thread.
//Though libusb operations are generally thread safe, I cannot guarantee
//everything above is thread safe when calling start()/stop() while
//waitForNewFrame().
void sigusr1_handler(int s)
{
  if (devtopause == 0)
    return;
/// [pause]
  if (protonect_paused)
    devtopause->start();
  else
    devtopause->stop();
  protonect_paused = !protonect_paused;
/// [pause]
}

//The following demostrates how to create a custom logger
/// [logger]
#include <fstream>
#include <cstdlib>
class MyFileLogger: public libfreenect2::Logger
{
private:
  std::ofstream logfile_;
public:
  MyFileLogger(const char *filename)
  {
    if (filename)
      logfile_.open(filename);
    level_ = Debug;
  }
  bool good()
  {
    return logfile_.is_open() && logfile_.good();
  }
  virtual void log(Level level, const std::string &message)
  {
    logfile_ << "[" << libfreenect2::Logger::level2str(level) << "] " << message << std::endl;
  }
};
/// [logger]


// string converter
namespace patch
{
    template < typename T > std::string to_string( const T& n )
    {
        std::ostringstream stm ;
        stm << n ;
        return stm.str() ;
    }
}
//  void handle_register(http::server::server_ptr serv, http::server::connection_ptr conn, const http::server::request& req, const std::string& path, const std::string&query, http::server::reply& rep)
// {
//   std::cout << "starting to register" << std::endl;
//   //this redirect the browser so it thinks the stream url is unique
//   rep.status = http::server::reply::no_content;
//   rep.headers.clear();
//   // rep.content = "registered";
//   // rep.headers.push_back(header("Content-Length", boost::lexical_cast<std::string>(rep.content.size())));
//   // rep.headers.push_back(header("Content-Type", mime_types::extension_to_type("txt")));
//   conn->async_write(rep.to_buffers());
// }

void draw_everithing(cv::Mat img_todraw, det_struct detection_data){

  //TO do !!!! currently obj_ids not exist , put it later
    for(int i = 0;i < detection_data.rect_faces.size(); i++){

      cv::rectangle(img_todraw, detection_data.rect_faces[i], cv::Scalar(255, 0, 0));
      cv::Point textOrg(detection_data.rect_faces[i].x, detection_data.rect_faces[i].y-5);
      // std::string text = "id: " + patch::to_string(detections[i].obj_id);
      // int thickness =1;
      // cv::putText(img_todraw, text, textOrg, cv::FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, cv::Scalar::all(255), thickness, 3);
    }
}


void regiter_new_faces(det_struct detection_data){

  // cv::namedWindow("Registering", CV_WINDOW_AUTOSIZE);
  int i =0;
  while(1){
      while(true){
        std::cout << "Agar ko`rsatilgan rasm ni ro`yhatdan o`tkazishni xohlasangiz y ni bosing" << std::endl; 
        cv::imshow("registered",detection_data.faces[i]);
        int k = cv::waitKey(0);
        std:: cout << k << std::endl;
        if(k == 27){
          i++;
        }
        if(k == 121 || k == 89){
          i++;
          break;    
        }
      }
      
      if(i >= detection_data.features.size()) break;
      std::string folder_name("./registered_faces/");
      char * folder_name_char = new char [folder_name.length()+1];
      std::strcpy (folder_name_char, folder_name.c_str());
      // Creating a directory 
      if (mkdir(folder_name_char, 0777) == -1) 
          std::cerr << "Error :  " << strerror(errno) << std::endl; 
    
      else
          std::cout << "Directory registered_faces created";
      std::string str_name;
      std::cout << "iltimos ismni kiriting\n ismning uzunligi 10 dan oshmasin va keraksiz simvollardan iborat bo`lmasin.\n masalan user_1 yoki user1 bo`lishi mumkin" << std::endl; 
      std::getline(std::cin, str_name);
      // Declare a file
      cv::FileStorage file(folder_name +str_name , cv::FileStorage::WRITE);
      // Write to file!
      file << str_name << detection_data.features[0];

      std::cout << "yangi " << str_name<< " nomli ism ro`yhatdan o`tkazildi";  
  }
}

template<typename T>
class send_one_replaceable_object_t {
    const bool sync;
    std::atomic<T *> a_ptr;
public:

    void send(T const& _obj) {
        T *new_ptr = new T;
        *new_ptr = _obj;
        if (sync) {
            std::cout << "send one rep obj" << sync << std::endl;
            while (a_ptr.load()) std::this_thread::sleep_for(std::chrono::milliseconds(3));
        }
        std::unique_ptr<T> old_ptr(a_ptr.exchange(new_ptr));
    }

    T receive() {
        std::unique_ptr<T> ptr;
        do {
            while(!a_ptr.load()) std::this_thread::sleep_for(std::chrono::milliseconds(3));
            ptr.reset(a_ptr.exchange(NULL));
        } while (!ptr);
        T obj = *ptr;
        return obj;
    }

    bool is_object_present() {
        return (a_ptr.load() != NULL);
    }

    send_one_replaceable_object_t(bool _sync) : sync(_sync), a_ptr(NULL)
    {}
};


/// [main]
/**
 * Main application entry point.
 *
 * Accepted argumemnts:
 * - cpu Perform depth processing with the CPU.
 * - gl  Perform depth processing with OpenGL.
 * - cl  Perform depth processing with OpenCL.
 * - <number> Serial number of the device to open.
 * - -noviewer Disable viewer window.
 */
int main(int argc, char *argv[])
/// [main]
{
  std::string program_path(argv[0]);
  std::cerr << "Version: " << LIBFREENECT2_VERSION << std::endl;
  std::cerr << "Environment variables: LOGFILE=<protonect.log>" << std::endl;
  std::cerr << "Usage: " << program_path << " [-gpu=<id>] [gl | cl | clkde | cuda | cudakde | cpu] [<device serial>]" << std::endl;
  std::cerr << "        [-noviewer] [-norgb | -nodepth] [-help] [-version]" << std::endl;
  std::cerr << "        [-frames <number of frames to process>]" << std::endl;
  std::cerr << "To pause and unpause: pkill -USR1 Protonect" << std::endl;
  size_t executable_name_idx = program_path.rfind("Protonect");
   
  std::string binpath = "/";

  if(executable_name_idx != std::string::npos)
  {
    binpath = program_path.substr(0, executable_name_idx);
  }

  libfreenect2::setGlobalLogger(NULL);
// #if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
//   // avoid flooing the very slow Windows console with debug messages
//   libfreenect2::setGlobalLogger(libfreenect2::createConsoleLogger(libfreenect2::Logger::Info));
// #else
//   // create a console logger with debug level (default is console logger with info level)
// /// [logging]
//   libfreenect2::setGlobalLogger(libfreenect2::createConsoleLogger(libfreenect2::Logger::Debug));
// /// [logging]
// #endif
// /// [file logging]
//   MyFileLogger *filelogger = new MyFileLogger(getenv("LOGFILE"));
//   if (filelogger->good())
//     libfreenect2::setGlobalLogger(filelogger);
//   else
//     delete filelogger;
/// [file logging]

/// [context]
  libfreenect2::Freenect2 freenect2;
  libfreenect2::Freenect2Device *dev = 0;
  libfreenect2::PacketPipeline *pipeline = 0;
/// [context]

  std::string serial = "";
  bool detection_sync = false;  
  bool cam_enabled = false;
  bool enable_rgb = true;
  bool enable_depth = true;
  int deviceId = -1;
  size_t framemax = -1;


  for(int argI = 1; argI < argc; ++argI)
  {
    const std::string arg(argv[argI]);

    if(arg == "-help" || arg == "--help" || arg == "-h" || arg == "-v" || arg == "--version" || arg == "-version")
    {
      // Just let the initial lines display at the beginning of main
      return 0;
    }
    else if(arg.find("-gpu=") == 0)
    {
      if (pipeline)
      {
        std::cerr << "-gpu must be specified before pipeline argument" << std::endl;
        return -1;
      }
      deviceId = atoi(argv[argI] + 5);
    }
    else if(arg == "cpu")
    {
      if(!pipeline)
/// [pipeline]
        pipeline = new libfreenect2::CpuPacketPipeline();
/// [pipeline]
    }
    else if(arg == "gl")
    {
#ifdef LIBFREENECT2_WITH_OPENGL_SUPPORT
      if(!pipeline)
        pipeline = new libfreenect2::OpenGLPacketPipeline();
#else
      std::cout << "OpenGL pipeline is not supported!" << std::endl;
#endif
    }
    else if(arg == "cl")
    {
#ifdef LIBFREENECT2_WITH_OPENCL_SUPPORT
      if(!pipeline)
        pipeline = new libfreenect2::OpenCLPacketPipeline(deviceId);
#else
      std::cout << "OpenCL pipeline is not supported!" << std::endl;
#endif
    }
    else if(arg == "clkde")
    {
#ifdef LIBFREENECT2_WITH_OPENCL_SUPPORT
      if(!pipeline)
        pipeline = new libfreenect2::OpenCLKdePacketPipeline(deviceId);
#else
      std::cout << "OpenCL pipeline is not supported!" << std::endl;
#endif
    }
    else if(arg == "cuda")
    {
#ifdef LIBFREENECT2_WITH_CUDA_SUPPORT
      if(!pipeline)
        pipeline = new libfreenect2::CudaPacketPipeline(deviceId);
#else
      std::cout << "CUDA pipeline is not supported!" << std::endl;
#endif
    }
    else if(arg == "cudakde")
    {
#ifdef LIBFREENECT2_WITH_CUDA_SUPPORT
      if(!pipeline)
        pipeline = new libfreenect2::CudaKdePacketPipeline(deviceId);
#else
      std::cout << "CUDA pipeline is not supported!" << std::endl;
#endif
    }
    else if(arg.find_first_not_of("0123456789") == std::string::npos) //check if parameter could be a serial number
    {
      serial = arg;
    }
    else if(arg == "-cam" || arg == "--cam")
    {
      std::string cam_str = "camera";
      std::string kinect_str = "kinect";
      ++argI;
      std::string str(argv[argI]);
      std::cout << str.compare(kinect_str) << "\t" << str.compare(cam_str)  << "\t" << str << std::endl;
      if(str.compare(kinect_str) == 0){
        cam_enabled = false;
      }else if(str.compare(cam_str) == 0){
        cam_enabled = true;
      }
    }
    else if(arg == "-norgb" || arg == "--norgb")
    {
      enable_rgb = false;
    }
    else if(arg == "-nodepth" || arg == "--nodepth")
    {
      enable_depth = false;
    }
    else if(arg == "-frames")
    {
      ++argI;
      framemax = strtol(argv[argI], NULL, 0);
      if (framemax == 0) {
        std::cerr << "invalid frame count '" << argv[argI] << "'" << std::endl;
        return -1;
      }
    }
    else
    {
      std::cout << "Unknown argument: " << arg << std::endl;
    }
  }
if(!cam_enabled){
  if (!enable_rgb && !enable_depth)
  {
    std::cerr << "Disabling both streams is not allowed!" << std::endl;
    return -1;
  }

/// [discovery]
  if(freenect2.enumerateDevices() == 0)
  {
    std::cout << "no device connected!" << std::endl;
    return -1;
  }

  if (serial == "")
  {
    serial = freenect2.getDefaultDeviceSerialNumber();
  }
/// [discovery]

  if(pipeline)
  {
/// [open]
    dev = freenect2.openDevice(serial, pipeline);
/// [open]
  }
  else
  {
    dev = freenect2.openDevice(serial);
  }

  if(dev == 0)
  {
    std::cout << "failure opening device!" << std::endl;
    return -1;
  }

  devtopause = dev;

  signal(SIGINT,sigint_handler);
#ifdef SIGUSR1
  signal(SIGUSR1, sigusr1_handler);
#endif
  protonect_shutdown = false;
}
/// [listeners]
  int types = 0;
  if(!cam_enabled){
  if (enable_rgb)
    types |= libfreenect2::Frame::Color;
  if (enable_depth)
    types |= libfreenect2::Frame::Ir | libfreenect2::Frame::Depth;
    }
  libfreenect2::SyncMultiFrameListener listener(types);
  libfreenect2::FrameMap frames;
  if(!cam_enabled){
    dev->setColorFrameListener(&listener);
    dev->setIrAndDepthFrameListener(&listener);

/// [start]
  if (enable_rgb && enable_depth)
  {
    if (!dev->start())
      return -1;
  }
  else
  {
    if (!dev->startStreams(enable_rgb, enable_depth))
      return -1;
  }


  std::cout << "device serial: " << dev->getSerialNumber() << std::endl;
  std::cout << "device firmware: " << dev->getFirmwareVersion() << std::endl;
  }
/// [listeners]
  
/// [start]

/// [registration setup]
  libfreenect2::Registration* registration;
  if(!cam_enabled){
   registration = new libfreenect2::Registration(dev->getIrCameraParams(), dev->getColorCameraParams());
  }
  libfreenect2::Frame undistorted(512, 424, 4), registered(512, 424, 4);
/// [registration setup]

  size_t framecount = 0;

  cv::CascadeClassifier face_cascade;
  face_cascade.load( "haarcascade_frontalface_alt.xml" );
  if(face_cascade.empty())
  {
    std::cout<<"Error Loading haar cascade XML file"<<std::endl;
    return 0;
  }
     // Detect faces
     
    
  std::string modelTxt1  = "pretrained/resnet50_scratch_caffe/resnet50_scratch.prototxt";
  std::string modelBin1  = "pretrained/resnet50_scratch_caffe/resnet50_scratch.caffemodel";
   // Load Caffe model using Caffe.
  caffe::Caffe::set_mode(caffe::Caffe::CPU);
  caffe::Net<float> caffeNet(modelTxt1, caffe::TEST);
  caffeNet.CopyTrainedLayersFrom(modelBin1);
  
  // Run Caffe model using Caffe.
  caffe::Blob<float>* caffeInput = caffeNet.input_blobs()[0];
  // Wrap Caffe's input blob to cv::Mat.
  cv::Mat caffeInputMat(caffeInput->shape(), CV_32F, (char*)caffeInput->cpu_data());
    
  int k = 9;
  

  // here we will defince server
  using namespace http::server;
  std::size_t num_threads = 8;
  std::string doc_root = "./";
  //this initializes the redirect behavor, and the /_all handlers
  server_ptr s(init_streaming_server("127.0.0.1", "8082", doc_root, num_threads));
  init_register(s);

  streamer_ptr stmr(new streamer);//a stream per image, you can register any number of these.
  register_streamer(s, stmr, "/stream_0");
  

  s->start();
  int quality = 100; 
  bool wait = false; //don't wait for there to be more than one webpage looking at us.
  cv::VideoCapture cap_cam;
/// [loop start]
  while(true)
  {
    // //std::cout <<"frame max" << framemax << std::endl;
    // if (!listener.waitForNewFrame(frames, 10*1000)) // 10 sconds
    // {
    //   std::cout << "timeout!" << std::endl;
    //   return -1;
    // }

    std::atomic<int> fps_cap_counter(0), fps_det_counter(0);
    std::atomic<int> current_fps_cap(0), current_fps_det(0);
    std::atomic<bool> exit_flag(false);
    std::chrono::steady_clock::time_point steady_start, steady_end;
    std::cout <<  "main  "<< exit_flag << std::endl;
    
    const bool sync = detection_sync; // sync data exchange
    // bool registering = false;
    send_one_replaceable_object_t<det_struct> cap2prepare_and_detect_face(sync), cap2draw(sync),
        prepare2getfeature(sync), getfeature2draw(sync), draw2show(sync),drawr2show(sync), draw2write(sync), draw2net(sync), registertoshow(sync);
    std::thread t_cap, t_prepare_and_detect_face, t_get_feature, t_post, t_draw, t_write, t_network, t_register;
    cv::Size  frame_size(0, 0);
    // capture new video-frame
    if (t_cap.joinable()) t_cap.join();

    t_cap = std::thread([&](){
    
      uint64_t frame_id = 0;
      det_struct detection_data;
      do {
        if (exit_flag) {
            std::cout << " exit_flag: detection_data.cap_frame.size = " << detection_data.cap_frame.size() << std::endl;
             if(!cam_enabled){
            listener.release(frames);
            dev->stop();
            dev->close();
             }else{
               cap_cam.release();
             }
            detection_data.exit_flag = true;
            detection_data.cap_frame = cv::Mat(frame_size, CV_8UC3);
            cv::destroyAllWindows();
        }else{
              
              cv::Mat ir_draw;
              if(cam_enabled){
                if(!cap_cam.isOpened())
                  cap_cam.open(0);
                  cap_cam >> ir_draw;
              }else{
                if (!listener.waitForNewFrame(frames, 10*1000)) // 10 sconds
              {
                std::cout << "timeout!" << std::endl;
                return -1;
              }

              libfreenect2::Frame *rgb = frames[libfreenect2::Frame::Color];
              libfreenect2::Frame *ir = frames[libfreenect2::Frame::Ir];
              libfreenect2::Frame *depth = frames[libfreenect2::Frame::Depth];


              registration->apply(rgb, depth, &undistorted, &registered);
              
              // cv::Mat undistorted_mat = cv::Mat(
              // registered.height, registered.width,
              // CV_8UC4, undistorted.data);
              
              // cv::Mat undistorted_mat = cv::Mat(
              // registered.height, registered.width,
              // CV_8UC4, undistorted.data);
              
              // cv::Mat registered_mat = cv::Mat(
              // registered.height, registered.width,
              // CV_8UC4, registered.data);

              cv::Mat ir_mat = cv::Mat(
              ir->height, ir->width,
              CV_32FC1, ir->data) / 65535.;
              
              
              double  minVal,  maxVal;
              cv::minMaxLoc(ir_mat,  &minVal,  &maxVal);  //find  minimum  and  maximum  intensities
              ir_mat.convertTo(ir_draw,CV_8U,255.0/(maxVal  -  minVal),  -minVal);
              
              if(frame_size.height == 0){
                frame_size = ir_draw.size();
              }
              listener.release(frames);
              }

              detection_data.cap_frame = ir_draw;

              
        }
        //  if (!detection_sync) {
        cap2draw.send(detection_data);       // skip detection
        //  }
         cap2prepare_and_detect_face.send(detection_data);
        framecount++;
        fps_cap_counter++;
      } while (!detection_data.exit_flag);
      std::cout << " t_cap exit \n";

    });

     // pre-processing video frame (resize, convertion)
      t_prepare_and_detect_face = std::thread([&]()
      {
          det_struct detection_data;
          do {
              detection_data = cap2prepare_and_detect_face.receive();
              
              std::vector<cv::Rect> rect_faces;
              std::vector<cv::Mat> faces;
              face_cascade.detectMultiScale( detection_data.cap_frame, rect_faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(50, 50) );

              for(int i = 0; i < rect_faces.size();i++){
                  det_struct detected;
                  cv::Rect rect(rect_faces[i].x, rect_faces[i].y, rect_faces[i].width, rect_faces[i].height);
                  cv::Mat croppedImage = detection_data.cap_frame(rect);
                  //getting the features 
                  cv::Mat inputMat;
                  croppedImage.convertTo(inputMat, CV_32F);  // Cast to floats
                  cv::resize(inputMat, inputMat, cv::Size(224, 224));  // Resize.
                  faces.push_back(inputMat);
              }
              detection_data.faces = faces;
              detection_data.rect_faces = rect_faces;


              
              prepare2getfeature.send(detection_data);    // detection
              

          } while (!detection_data.exit_flag);
          std::cout << " t_prepare_and_detect_face exit \n";
      });

      // detection by Yolo
      if (t_get_feature.joinable()) t_get_feature.join();
      t_get_feature = std::thread([&]()
      {
          det_struct detection_data;
          do {
              detection_data = prepare2getfeature.receive();
              std::vector<Mat> features;
              for(int i = 0; i < detection_data.faces.size();i++){
                  cv::Mat inputMat;
                  inputMat = cv::dnn::blobFromImage(detection_data.faces[i], 1.0, cv::Size(224, 224), cv::Scalar(91.4953, 103.8827, 131.0912));
                  // Copy image.
                  inputMat.copyTo(caffeInputMat);
                  caffe::Blob<float>* caffeOut = caffeNet.Forward()[0];
                  
                  // Print results.
                  cv::Mat caffeOutMat(caffeOut->shape(), CV_32F, (char*)caffeOut->cpu_data());
                  //std::cout << "features :" << caffeOutMat << std::endl;
                  features.push_back(caffeOutMat);
              }
              
              detection_data.features = features;
              
              getfeature2draw.send(detection_data);
          } while (!detection_data.exit_flag);
          std::cout << " t_get_feature exit \n";
       });  

    // draw rectangles (and track objects)
    t_draw = std::thread([&]()
    {
        det_struct detection_data;
        do {

              // get new Detection result if present
              if (getfeature2draw.is_object_present()) {
                  cv::Mat old_cap_frame = detection_data.cap_frame;   // use old captured frame
                  detection_data = getfeature2draw.receive();
                  if (!old_cap_frame.empty()) detection_data.cap_frame = old_cap_frame;
              }
              else{
                std::vector <Rect> old_rect_faces = detection_data.rect_faces;
                det_struct tmp;
                tmp = cap2draw.receive();
                detection_data.cap_frame = tmp.cap_frame;
                detection_data.draw_frame = tmp.draw_frame;
                detection_data.rect_faces = old_rect_faces;
              }

              cv::Mat cap_frame = detection_data.cap_frame;
              cv::Mat draw_frame = detection_data.cap_frame.clone();
              draw_everithing(draw_frame, detection_data);
              detection_data.draw_frame = draw_frame;
              draw2show.send(detection_data);
           } while (!detection_data.exit_flag);
           std::cout << " t_draw exit \n";
      });

    //   // draw rectangles (and track objects)
    // if (t_register.joinable()) t_register.join();
    // t_register = std::thread([&]()
    // {
    //     det_struct detection_data;
    //     do {
    //         detection_data = registertoshow.receive();
    //         registering = true;
    //         std::string folder_name("./registered_faces/");
    //         char * folder_name_char = new char [folder_name.length()+1];
    //         std::strcpy (folder_name_char, folder_name.c_str());
    //         // Creating a directory 
    //         if (mkdir(folder_name_char, 0777) == -1) 
    //             std::cerr << "Error :  " << strerror(errno) << std::endl; 
          
    //         else
    //             std::cout << "Directory registered_faces created";
    //         std::string str_name;
    //         std::cout << "iltimos ismni kiriting\n ismning uzunligi 10 dan oshmasin va keraksiz simvollardan iborat bo`lmasin.\n masalan user_1 yoki user1 bo`lishi mumkin" << std::endl; 
    //         std::getline(std::cin, str_name);
    //         // Declare a file
    //         cv::FileStorage file(folder_name +str_name , cv::FileStorage::WRITE);
    //         std::cout <<"starting to write to a file " << std::endl;
            
    //         // Write to file!
    //         file << str_name << detection_data.features[0];

    //         file.release();
    //         std::cout << "yangi " << str_name<< " nomli ism ro`yhatdan o`tkazildi";  

    //         drawr2show.send(detection_data);
    //         registering= false;
    //        } while (!detection_data.exit_flag);
    //        std::cout << " t_draw exit \n";
    //   });

    det_struct detection_data;
    do{
      steady_end = std::chrono::steady_clock::now();
      float time_sec = std::chrono::duration<double>(steady_end - steady_start).count();
      if (time_sec >= 1) {
          current_fps_cap = fps_cap_counter.load() / time_sec;
          steady_start = steady_end;
          fps_det_counter = 0;
          fps_cap_counter = 0;
      }
      detection_data = draw2show.receive();
      cv::Mat ir_draw = detection_data.draw_frame;
      // if(drawr2show.is_object_present()){
      //   det_struct detection_data_r = drawr2show.receive();
      //   cv::Mat reg_image = detection_data.faces[0];
      //   cv::imshow("registering image", reg_image);
        
      // }
      // if(!registering)
      std::cout << "equalizing gacha keldik" << std::endl;
      global_detection_data = detection_data;
      int n_viewers = stmr->post_image(quality, wait);
      
      cv::imshow("registered",ir_draw);
      int k = cv::waitKey(3);
      if(k == 27){

          detection_data.exit_flag = true;
          exit_flag = true;
      }
      if(k == 114){
        // std::cout << k << std::endl;
        // regiter_new_faces(detection_data);
        //   registertoshow.send(detection_data);
        // else
        //   std::cout << detection_data.features.size() << std::endl;
      }
      // std::cout << detection_data.exit_flag << std::endl; 
      } while(!detection_data.exit_flag);

      if(!exit_flag){
        // std::cout << detection_data.exit_flag << std::endl; 
        // wait for all threads
        if (t_cap.joinable()) t_cap.join();
        if (t_prepare_and_detect_face.joinable()) t_prepare_and_detect_face.join();
        if (t_get_feature.joinable()) t_get_feature.join();
        // if (t_post.joinable()) t_post.join();
        if (t_draw.joinable()) t_draw.join();
        // if (t_write.joinable()) t_write.join();
        // if (t_network.joinable()) t_network.join();
      }
      dev->stop();
      dev->close();
      cv::destroyAllWindows();
      break;
// #ifdef EXAMPLES_WITH_OPENGL_SUPPORT
    
//     if (enable_rgb)
//     {
//       viewer.addFrame("RGB", rgb);
//     if (enable_depth)
//     {
//       viewer.addFrame("ir", ir);
//       viewer.addFrame("depth", depth);
//     }
//     if (enable_rgb && enable_depth)
//     {
//       viewer.addFrame("registered", &registered);
//     }
    

//     protonect_shutdown = protonect_shutdown || viewer.render();
// #endif

/// [loop end]
    // listener.release(frames);
    /** libfreenect2::this_thread::sleep_for(libfreenect2::chrono::milliseconds(100)); */
  }
/// [loop end]

  // TODO: restarting ir stream doesn't work!
  // TODO: bad things will happen, if frame listeners are freed before dev->stop() :(
/// [stop]
  
/// [stop]

  delete registration;

  return 0;
}
