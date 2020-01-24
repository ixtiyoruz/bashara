#include "dbconnector.h"
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
// #include "mysql_driver.h" 

/*
mysql> describe yuzlar
    -> ;
+----------+----------------------+------+-----+---------+-------+
| Field    | Type                 | Null | Key | Default | Extra |
+----------+----------------------+------+-----+---------+-------+
| ID       | smallint(5) unsigned | YES  |     | NULL    |       |
| Name     | varchar(40)          | YES  |     | NULL    |       |
| Photo    | blob                 | YES  |     | NULL    |       |
| Quantity | int(10) unsigned     | YES  |     | NULL    |       |
+----------+----------------------+------+-----+---------+-------+
4 rows in set (0.00 sec)
*/
// class DatabaseFaces{
//     private:
//     sql::Driver *driver;
//     sql::Connection *con;
//     sql::Statement *stmt;
//     sql::ResultSet *res;

//     public:
//     void connect(std::string url, std::string username, std::string password);
//     void get_feature(std::string id);
//     void get_face_image(str::string id);
// }
bool DatabaseFaces::connect(std::string url, std::string username, std::string password, std::string database_name){
    try {
    /* Create a connection */
    driver = get_driver_instance();
    con = driver->connect(url, username, password);
    
    /* Connect to the MySQL test database */
    con->setSchema(database_name);
    stmt = con->createStatement();
    } catch (sql::SQLException &e) {
    std::cout << "# ERR: SQLException in " << __FILE__;
    std::cout << "(" << __FUNCTION__ << ") on line " << __LINE__ << std::endl;
    std::cout << "# ERR: " << e.what();
    std::cout << " (MySQL error code: " << e.getErrorCode();
    std::cout << ", SQLState: " << e.getSQLState() << " )" << std::endl;
    return false;
}
return true;
}

bool DatabaseFaces::add_face(int id, std::string name, cv::Mat photo, int quantity){
    try {
        res = stmt->executeQuery("select ID, Name, Photo, Quantity from yuzlar");
        while (res->next()) {
            std::cout << "\t... MySQL replies: ";
            /* Access column data by alias or column name */
            std::cout << res->getString("_message") << std::endl;
            std::cout << "\t... MySQL says it again: ";
            /* Access column data by numeric offset, 1 is the first column */
            std::cout << res->getString(1) << std::endl;
        }
    } catch (sql::SQLException &e) {
        std::cout << "# ERR: SQLException in " << __FILE__;
        std::cout << "(" << __FUNCTION__ << ") on line " << __LINE__ << std::endl;
        std::cout << "# ERR: " << e.what();
        std::cout << " (MySQL error code: " << e.getErrorCode();
        std::cout << ", SQLState: " << e.getSQLState() << " )" << std::endl;
        return false;
    }
}
DatabaseFaces::~DatabaseFaces(){
    delete res;
    delete stmt;
    delete con;
}
bool DatabaseFaces::checkConnection(){
    return is_connected;
}
bool DatabaseFaces::set_connected(bool is_connected){
    this->is_connected = is_connected;
}

// try {
//     sql::Driver *driver;
//     sql::Connection *con;
//     sql::Statement *stmt;
//     sql::ResultSet *res;

//     /* Connect to the MySQL test database */
//     con->setSchema("faces");

//     stmt = con->createStatement();
//     res = stmt->executeQuery("select ID, Name, Photo, Quantity from yuzlar");
//     while (res->next()) {
//         cout << "\t... MySQL replies: ";
//         /* Access column data by alias or column name */
//         cout << res->getString("_message") << endl;
//         cout << "\t... MySQL says it again: ";
//         /* Access column data by numeric offset, 1 is the first column */
//         cout << res->getString(1) << endl;
//     }
//     delete res;
//     delete stmt;
//     delete con;

// } catch (sql::SQLException &e) {
//     cout << "# ERR: SQLException in " << __FILE__;
//     cout << "(" << __FUNCTION__ << ") on line " << __LINE__ << endl;
//     cout << "# ERR: " << e.what();
//     cout << " (MySQL error code: " << e.getErrorCode();
//     cout << ", SQLState: " << e.getSQLState() << " )" << endl;
// }
