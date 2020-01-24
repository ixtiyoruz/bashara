#ifndef DBCONNECTOR
#define DBCONNECTOR

#include <functional>
#include <iostream>
#include <mysql_connection.h>

#include <cppconn/driver.h>
#include <cppconn/exception.h>
#include <cppconn/resultset.h>
#include <cppconn/statement.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class DatabaseFaces{
    private:
    sql::Driver *driver;
    sql::Connection *con;
    sql::Statement *stmt;
    sql::ResultSet *res;
    bool is_connected=false;

    public:
    bool connect(std::string url, std::string username, std::string password, std::string database_name);
    cv::Mat get_feature(std::string id);
    cv::Mat get_face_image(std::string id);
    bool add_face(int id, std::string name, cv::Mat photo, int quantity);
    bool del_face(int id);
    bool checkConnection();
    bool set_connected(bool is_connected);
    ~DatabaseFaces();  
};
#endif