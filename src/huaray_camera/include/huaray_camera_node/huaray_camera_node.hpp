#pragma once

#include <opencv2/imgproc.hpp>

#include "IMVApi.h"
#include "IMVDefines.h"
#include "camera_info_manager/camera_info_manager.hpp"
#include "image_transport/image_transport.hpp"

namespace HuarayCamera
{
class HuarayCameraNode : public rclcpp::Node
{
 public:
  explicit HuarayCameraNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
  ~HuarayCameraNode() override;

 private:
  enum class HuarayStateEnum : uint8_t
  {
    STOPPED,
    RUNNING
  };

  struct Parameters
  {
    double exposure_time;
    double gain;
    bool autocap;
    double frame_rate;
    std::string frame_id;
    std::string camera_name;
    int interface_type;
  };

  struct Protect
  {
    std::mutex mux;
    std::condition_variable is_quit;
    std::thread protect_thread;
  };

  bool Read(cv::Mat& image, rclcpp::Time& stamp);
  void CaptureInit();
  void CaptureStop();
  void ProtectRunning();

  void SetFloatValue(const std::string& name, double value);
  void SetEnumValue(const std::string& name, unsigned int value);
  void SetEnumSymbol(const std::string& name, const std::string& value);

  Parameters params_;
  std::unique_ptr<camera_info_manager::CameraInfoManager> camera_info_manager_;

  sensor_msgs::msg::Image image_msg_;
  sensor_msgs::msg::CameraInfo camera_info_msg_;

  IMV_HANDLE handle_{nullptr};

  std::atomic<bool> is_opened_{false};
  std::atomic<bool> is_grabbing_{false};

  std::atomic<HuarayStateEnum> state_{HuarayStateEnum::STOPPED};
  std::atomic<bool> running_{true};

  std::thread capture_thread_;
  Protect guard_;

  image_transport::CameraPublisher camera_pub_;
};
}  // namespace HuarayCamera