#include <chrono>
#include <string>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>

using namespace std::chrono_literals;

class CameraNode : public rclcpp::Node
{
public:
  CameraNode()
  : Node("camera_node"), frames_(0), ticks_(0)
  {
    device_ = declare_parameter<std::string>("device", "/dev/video0");
    save_dir_ = declare_parameter<std::string>("save_dir", "");
    pub_ = create_publisher<sensor_msgs::msg::Image>("image_raw", 10);
    timer_ = create_wall_timer(33ms, std::bind(&CameraNode::grab, this));  // 约30fps
  }

private:
  void grab()
  {
    if (!cap_.isOpened()) {
      if (++ticks_ % 30 != 1) { return; }  
      cap_.open(device_, cv::CAP_V4L2);
      if (!cap_.isOpened()) {
        cap_.open(device_, cv::CAP_ANY); 
      }
      if (!cap_.isOpened()) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
          "等待视频源可用: %s", device_.c_str());
        return;
      }
      RCLCPP_INFO(get_logger(), "已打开视频源: %s", device_.c_str());
    }
    cv::Mat frame;
    if (!cap_.read(frame) || frame.empty()) {
      if (++fails_ > 60) {         
        cap_.release();
        fails_ = 0;
        RCLCPP_WARN(get_logger(), "视频流中断，准备重开: %s", device_.c_str());
      }
      return;
    }
    fails_ = 0;
    ++frames_;
    auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
    msg->header.stamp = now();
    msg->header.frame_id = "camera";
    pub_->publish(*msg);
    if (!save_dir_.empty() && frames_ % 30 == 1) {  
      cv::imwrite(save_dir_ + "/camera_frame.png", frame);
    }
  }

  cv::VideoCapture cap_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::string device_;
  std::string save_dir_;
  int frames_;
  int ticks_;
  int fails_ = 0;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CameraNode>());
  rclcpp::shutdown();
  return 0;
}
