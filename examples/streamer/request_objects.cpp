
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

struct det_struct {
    cv::Mat cap_frame;
    cv::Mat draw_frame;
    bool new_detection;
    uint64_t frame_id;
    bool exit_flag;
    std::vector<cv::Mat> faces;
    std::vector<cv::Mat> features;
    std::vector <unsigned int> obj_ids;
    std::vector <cv::Rect> rect_faces;       // (x,y) - top-left corner, (w, h) - width & height of bounded box

    det_struct(): exit_flag(false), new_detection(false) {}    
};



