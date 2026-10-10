#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

class DetectorNode : public rclcpp::Node
{
public:
  DetectorNode()
  : Node("detector_node"), recv_(0)
  {
    save_dir_ = declare_parameter<std::string>("save_dir", "");
    sub_ = create_subscription<sensor_msgs::msg::Image>(
      "image_raw", 10,
      std::bind(&DetectorNode::onImage, this, std::placeholders::_1));
    RCLCPP_INFO(get_logger(), "识别节点已启动，等待 image_raw ...");
  }

private:

  void detectBlue(const cv::Mat & bgr, cv::Mat & mask)
  {
    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, cv::Scalar(85, 60, 175), cv::Scalar(115, 255, 255), mask);
    clean(mask);
  }


  void detectRed(const cv::Mat & bgr, cv::Mat & mask)
  {
    std::vector<cv::Mat> ch;
    cv::split(bgr, ch);                    // ch[0]=B ch[1]=G ch[2]=R
    cv::Mat others;
    cv::max(ch[0], ch[1], others);
    cv::Mat dom;
    cv::subtract(ch[2], others, dom, cv::Mat(), CV_16S);
    cv::Mat dom8;
    dom.convertTo(dom8, CV_8U);  
    cv::Mat m1;
    cv::threshold(dom8, m1, 20, 255, cv::THRESH_BINARY);
    cv::Mat vmax, m2;
    cv::max(ch[2], others, vmax);
    cv::threshold(vmax, m2, 235, 255, cv::THRESH_BINARY);
    cv::bitwise_and(m1, m2, mask);
    clean(mask);
  }

  void clean(cv::Mat & mask)
  {
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN,
      cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
      cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9)));
  }

  void annotate(cv::Mat & vis, const cv::Mat & mask, const std::string & label, int text_row)
  {
    std::vector<std::vector<cv::Point>> cs;
    cv::findContours(mask, cs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    int best = -1;
    double best_area = 500.0;              
    for (size_t i = 0; i < cs.size(); ++i) {
      double a = cv::contourArea(cs[i]);
      if (a > best_area) { best_area = a; best = static_cast<int>(i); }
    }
    if (best < 0) { return; }
    cv::Moments m = cv::moments(cs[best]);
    int cx = static_cast<int>(m.m10 / m.m00);
    int cy = static_cast<int>(m.m01 / m.m00);
    cv::drawContours(vis, cs, best, cv::Scalar(0, 255, 0), 2);
    cv::circle(vis, cv::Point(cx, cy), 4, cv::Scalar(0, 255, 0), -1);
    cv::putText(vis, label + " Center:(" + std::to_string(cx) + "," + std::to_string(cy) +
      ") Area:" + std::to_string(static_cast<int>(best_area)),
      cv::Point(8, 24 * text_row), cv::FONT_HERSHEY_SIMPLEX, 0.6,
      cv::Scalar(0, 0, 255), 2);
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
      "%s 色块 -> 中心(%d,%d) 面积=%d", label.c_str(), cx, cy, static_cast<int>(best_area));
  }

  void onImage(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    cv::Mat frame = cv_bridge::toCvShare(msg, "bgr8")->image.clone();
    ++recv_;

    cv::Mat blue_mask, red_mask;
    detectBlue(frame, blue_mask);
    detectRed(frame, red_mask);

    cv::Mat vis = frame.clone();
    annotate(vis, blue_mask, "BLUE", 1);
    annotate(vis, red_mask, "RED", 2);

    cv::Mat thresh;
    cv::hconcat(blue_mask, red_mask, thresh);   

    cv::imshow("Webcam", vis);
    cv::imshow("Threshold", thresh);
    cv::waitKey(1);

    if (!save_dir_.empty() && recv_ % 30 == 1) {
      cv::imwrite(save_dir_ + "/original.png", frame);
      cv::imwrite(save_dir_ + "/webcam.png", vis);
      cv::imwrite(save_dir_ + "/threshold.png", thresh);
    }
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
  std::string save_dir_;
  int recv_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DetectorNode>());
  rclcpp::shutdown();
  return 0;
}
