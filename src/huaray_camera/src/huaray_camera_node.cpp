#include "huaray_camera_node/huaray_camera_node.hpp"

#include <chrono>

using namespace std::chrono_literals;

namespace HuarayCamera
{
HuarayCameraNode::HuarayCameraNode(const rclcpp::NodeOptions& options)
    : rclcpp::Node("huaray_camera_node", options)
{
  // 声明参数（默认值可根据你的实际情况调整）
  params_.exposure_time = this->declare_parameter<double>("exposure_time", 1000.0);  // us
  params_.gain = this->declare_parameter<double>("gain", 16.0);
  params_.autocap = this->declare_parameter<bool>("autocap", true);
  params_.frame_rate = this->declare_parameter<double>("frame_rate", 249.0);
  params_.frame_id =
      this->declare_parameter<std::string>("frame_id", "camera_optical_frame");
  params_.camera_name =
      this->declare_parameter<std::string>("camera_name", "narrow_stereo");
  // 新增：接口类型参数 (0=All, 1=GigE, 2=USB3)
  params_.interface_type =
      static_cast<int>(this->declare_parameter<int>("interface_type", 0));

  RCLCPP_INFO(this->get_logger(), "params has been initialized.");

  // 创建 publisher，话题名保持与原代码一致
  camera_pub_ = image_transport::create_camera_publisher(this, "image_raw",
                                                         rmw_qos_profile_sensor_data);
  RCLCPP_INFO(this->get_logger(), "Camera publisher created.");
  // 初始化相机
  CaptureInit();
  RCLCPP_INFO(this->get_logger(), "Camera initialized.");

  // 创建守护线程，负责自动重启
  guard_.protect_thread = std::thread(&HuarayCameraNode::ProtectRunning, this);

  camera_info_manager_ =
      std::make_unique<camera_info_manager::CameraInfoManager>(this, params_.camera_name);
  auto camera_info_url = this->declare_parameter(
      "camera_info_url", "package://huaray_camera/config/camera_info.yaml");
  if (camera_info_manager_->validateURL(camera_info_url))
  {
    camera_info_manager_->loadCameraInfo(camera_info_url);
    camera_info_msg_ = camera_info_manager_->getCameraInfo();
  }
  else
  {
    RCLCPP_WARN(this->get_logger(), "Invalid camera info URL: %s",
                camera_info_url.c_str());
  }

  RCLCPP_INFO(this->get_logger(), "Guard thread created.");
  // 创建取流线程
  capture_thread_ = std::thread(
      [this]()
      {
        RCLCPP_INFO(this->get_logger(), "Huaray SDK capture thread started.");

        while (running_.load())
        {
          // 如果状态是停止，则暂停等待
          if (state_.load() == HuarayStateEnum::STOPPED)
          {
            std::this_thread::sleep_for(10ms);
            continue;
          }

          cv::Mat image;
          rclcpp::Time stamp;

          bool ok = Read(image, stamp);
          if (!ok || image.empty())
          {
            // read_frame 内部已经负责在严重错误时切换状态并通知守护线程
            continue;
          }

          // 将 cv::Mat 转成 sensor_msgs::msg::Image
          image_msg_.header.stamp = stamp;
          image_msg_.header.frame_id = params_.frame_id;
          image_msg_.encoding = "rgb8";
          image_msg_.is_bigendian = false;
          image_msg_.height = image.rows;
          image_msg_.width = image.cols;
          image_msg_.step = static_cast<uint32_t>(image.cols * image.channels());
          image_msg_.data.assign(image.datastart, image.dataend);

          camera_pub_.publish(image_msg_, camera_info_msg_);
        }

        RCLCPP_INFO(this->get_logger(), "Huaray SDK capture thread exit.");
      });
}

HuarayCameraNode::~HuarayCameraNode()
{
  RCLCPP_INFO(this->get_logger(), "Destroying HuarayCameraNode...");

  running_.store(false);

  // 通知守护线程退出
  guard_.is_quit.notify_all();

  // 先停采集线程
  if (capture_thread_.joinable())
  {
    capture_thread_.join();
  }

  // 关闭相机
  CaptureStop();

  // 再停守护线程
  if (guard_.protect_thread.joinable())
  {
    guard_.protect_thread.join();
  }

  RCLCPP_INFO(this->get_logger(), "HuarayCameraNode destroyed.");
}

bool HuarayCameraNode::Read(cv::Mat& img, rclcpp::Time& timestamp)
{
  if (state_.load() == HuarayStateEnum::STOPPED || handle_ == nullptr ||
      !is_grabbing_.load())
  {
    return false;
  }

  IMV_Frame frame{};
  int ret{};
  unsigned int n_msec = 100;

  auto start = std::chrono::steady_clock::now();
  ret = IMV_GetFrame(handle_, &frame, n_msec);

  if (ret != IMV_OK)
  {
    // 超时不算严重错误，只是没有新帧
    if (ret == IMV_TIMEOUT)
    {
      return false;
    }
    RCLCPP_ERROR(this->get_logger(), "IMV_GetFrame failed: 0x%X, switching to Stopped.",
                 ret);
    state_.store(HuarayStateEnum::STOPPED);
    guard_.is_quit.notify_all();
    return false;
  }

  auto now = std::chrono::steady_clock::now();
  auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now - start);
  if (duration_ns < std::chrono::nanoseconds(2'000'000))
  {
    // RCLCPP_WARN(this->get_logger(), "Read frame too fast, dropped.");
    IMV_ReleaseFrame(handle_, &frame);
    return false;
  }

  // 时间戳：这里用 ROS2 的 now()，语义上与原来的"采集时间"保持一致
  timestamp = this->now();

  // 将原始 buffer 封装为 cv::Mat（8bit 单通道 Bayer）
  cv::Mat raw_img(cv::Size(static_cast<int>(frame.frameInfo.width),
                           static_cast<int>(frame.frameInfo.height)),
                  CV_8U, frame.pData);

  const auto& frame_info = frame.frameInfo;
  auto pixel_type = frame_info.pixelFormat;

  static const std::unordered_map<IMV_EPixelType, int> TYPE_MAP = {
      {gvspPixelBayGR8, cv::COLOR_BayerGR2BGR},
      {gvspPixelBayRG8, cv::COLOR_BayerRG2BGR},
      {gvspPixelBayGB8, cv::COLOR_BayerGB2BGR},
      {gvspPixelBayBG8, cv::COLOR_BayerBG2BGR}};

  auto it = TYPE_MAP.find(pixel_type);
  if (it == TYPE_MAP.end())
  {
    RCLCPP_WARN(this->get_logger(), "Unsupported pixel format: 0x%X", pixel_type);
    IMV_ReleaseFrame(handle_, &frame);
    state_.store(HuarayStateEnum::STOPPED);
    guard_.is_quit.notify_all();
    return false;
  }

  cv::Mat dst_image;
  cv::cvtColor(raw_img, dst_image, it->second);
  img = dst_image;

  ret = IMV_ReleaseFrame(handle_, &frame);
  if (ret != IMV_OK)
  {
    RCLCPP_ERROR(this->get_logger(),
                 "IMV_ReleaseFrame failed: 0x%X, switching to Stopped.", ret);
    state_.store(HuarayStateEnum::STOPPED);
    guard_.is_quit.notify_all();
    return false;
  }

  return true;
}

void HuarayCameraNode::CaptureInit()
{
  if (!running_.load())
  {
    return;
  }

  int ret{};
  IMV_DeviceList device_list{};

  // 根据参数选择接口类型
  unsigned int interface_type = interfaceTypeAll;  // 默认枚举所有类型
  switch (params_.interface_type)
  {
    case 1:
      interface_type = interfaceTypeGige;
      RCLCPP_INFO(this->get_logger(), "Enumerating GigE devices...");
      break;
    case 2:
      interface_type = interfaceTypeUsb3;
      RCLCPP_INFO(this->get_logger(), "Enumerating USB3 devices...");
      break;
    default:
      interface_type = interfaceTypeAll;
      RCLCPP_INFO(this->get_logger(), "Enumerating all interface types (GigE + USB3)...");
      break;
  }

  ret = IMV_EnumDevices(&device_list, interface_type);
  if (ret != IMV_OK)
  {
    RCLCPP_ERROR(this->get_logger(), "IMV_EnumDevices failed: 0x%X", ret);
    return;
  }

  RCLCPP_INFO(this->get_logger(), "Found %u device(s).", device_list.nDevNum);

  if (device_list.nDevNum == 0)
  {
    RCLCPP_ERROR(this->get_logger(), "Not found camera! Please check:");
    RCLCPP_ERROR(this->get_logger(), "  1. Camera is connected and powered on");
    RCLCPP_ERROR(this->get_logger(), "  2. Camera driver is installed properly");
    RCLCPP_ERROR(this->get_logger(),
                 "  3. For GigE: Check network settings and IP config");
    RCLCPP_ERROR(this->get_logger(),
                 "  4. For USB3: Check USB connection and permissions");
    return;
  }

  // 打印设备信息便于调试
  for (unsigned int i = 0; i < device_list.nDevNum; i++)
  {
    IMV_DeviceInfo* dev_info = &device_list.pDevInfo[i];
    RCLCPP_INFO(this->get_logger(), "Device[%u]: Key=%s, Type=%d", i, dev_info->cameraKey,
                dev_info->nCameraType);
  }

  // 使用设备键创建句柄（更可靠的方式）
  char* camera_key = device_list.pDevInfo[1].cameraKey;
  ret = IMV_CreateHandle(&handle_, modeByCameraKey, static_cast<void*>(camera_key));
  if (ret != IMV_OK)
  {
    RCLCPP_ERROR(this->get_logger(), "IMV_CreateHandle failed: 0x%X", ret);
    handle_ = nullptr;
    return;
  }
  RCLCPP_INFO(this->get_logger(), "Handle created successfully.");

  ret = IMV_Open(handle_);
  if (ret != IMV_OK)
  {
    RCLCPP_ERROR(this->get_logger(), "IMV_Open failed: 0x%X", ret);
    RCLCPP_ERROR(this->get_logger(), "Possible causes:");
    RCLCPP_ERROR(this->get_logger(),
                 "  1. Camera is already opened by another application");
    RCLCPP_ERROR(this->get_logger(), "  2. Network/USB connection issue");
    RCLCPP_ERROR(this->get_logger(), "  3. Insufficient permissions");
    // 销毁句柄
    IMV_DestroyHandle(handle_);
    handle_ = nullptr;
    return;
  }
  is_opened_.store(true);
  RCLCPP_INFO(this->get_logger(), "Camera opened successfully.");

  // 设置缓冲区数量
  unsigned int n_image_node_num = 3;
  ret = IMV_SetBufferCount(handle_, n_image_node_num);
  if (IMV_OK != ret)
  {
    RCLCPP_WARN(this->get_logger(), "IMV_SetBufferCount failed: 0x%X (non-fatal)", ret);
    // 这不是致命错误，继续
  }

  if (!params_.autocap)
  {
    ret = IMV_SetEnumFeatureSymbol(handle_, "AcquisitionMode", "Continuous");
    if (IMV_OK != ret)
    {
      RCLCPP_WARN(this->get_logger(), "Set Acquisition Mode to Continuous fail! 0x%X",
                  ret);
    }

    //    将触发模式设置为开启 (On)
    //    参数 "TriggerMode" 的值: 0 表示 Off, 1 表示 On
    ret = IMV_SetEnumFeatureValue(handle_, "TriggerMode", 1);
    if (IMV_OK != ret)
    {
      RCLCPP_WARN(this->get_logger(), "Set Trigger Mode to On fail! 0x%X", ret);
    }

    //    设置触发源为外部硬件触发 (Line0)
    //    可用的值通常有 "Line0", "Line1", "Line2", "Software", "FrequencyConverter" 等
    //    请根据您的物理接线选择正确的一项
    ret = IMV_SetEnumFeatureSymbol(handle_, "TriggerSource", "Line0");
    if (IMV_OK != ret)
    {
      RCLCPP_WARN(this->get_logger(), "Set Trigger Source to Line0 fail! 0x%X", ret);
    }

    //    (可选) 设置触发激活方式
    //    例如设置为上升沿触发 "RisingEdge"
    //    其他可选值如 "FallingEdge", "LevelHigh", "LevelLow"
    ret = IMV_SetEnumFeatureSymbol(handle_, "TriggerActivation", "RisingEdge");
    if (IMV_OK != ret)
    {
      RCLCPP_WARN(this->get_logger(), "Set Trigger Activation to RisingEdge fail! 0x%X",
                  ret);
    }
  }
  else
  {
    // 将触发模式设置为关闭 (Off) - 自由运行模式
    ret = IMV_SetEnumFeatureValue(handle_, "TriggerMode", 0);
    if (IMV_OK != ret)
    {
      RCLCPP_WARN(this->get_logger(), "Set Trigger Mode to Off fail! 0x%X", ret);
    }
  }

  // 曝光、增益、白平衡等（这些设置失败不应该阻止相机启动）
  SetEnumSymbol("BalanceWhiteAuto", "Continuous");
  SetEnumSymbol("ExposureAuto", "Off");
  SetEnumSymbol("GainAuto", "Off");
  SetFloatValue("ExposureTime", params_.exposure_time);
  SetFloatValue("Gain", params_.gain);

  // 设置 ADC 位深为 8 Bits（某些相机可能不支持此属性）
  // ret = IMV_SetEnumFeatureValue(handle_, "ADCBitDepth", 2);
  // if (IMV_OK != ret)
  // {
  //   RCLCPP_ERROR(this->get_logger(), "Set ADC Bit Depth to 8 Bits fail! 0x%X", ret);
  // }

  // 帧率
  ret = IMV_SetDoubleFeatureValue(handle_, "AcquisitionFrameRate", params_.frame_rate);
  if (ret != IMV_OK)
  {
    RCLCPP_ERROR(this->get_logger(),
                 "IMV_SetDoubleFeatureValue(set framerate) failed: 0x%X", ret);
    return;
  }
  ret = IMV_StartGrabbing(handle_);
  if (ret != IMV_OK)
  {
    RCLCPP_ERROR(this->get_logger(), "IMV_StartGrabbing failed: 0x%X", ret);
    // 关闭相机并销毁句柄
    IMV_Close(handle_);
    is_opened_.store(false);
    IMV_DestroyHandle(handle_);
    handle_ = nullptr;
    return;
  }
  is_grabbing_.store(true);

  state_.store(HuarayStateEnum::RUNNING);
  RCLCPP_INFO(this->get_logger(), "Huaray camera initialized and started successfully.");
  return;
}

void HuarayCameraNode::CaptureStop()
{
  state_.store(HuarayStateEnum::STOPPED);

  if (handle_ == nullptr)
  {
    RCLCPP_DEBUG(this->get_logger(), "CaptureStop: handle is null, nothing to do.");
    return;
  }

  int ret{};

  // 只有在正在采集时才停止采集
  if (is_grabbing_.load())
  {
    ret = IMV_StopGrabbing(handle_);
    if (ret != IMV_OK)
    {
      RCLCPP_WARN(this->get_logger(), "IMV_StopGrabbing failed: 0x%X", ret);
    }
    else
    {
      RCLCPP_DEBUG(this->get_logger(), "IMV_StopGrabbing succeeded.");
    }
    is_grabbing_.store(false);
  }

  // 只有在相机已打开时才关闭
  if (is_opened_.load())
  {
    ret = IMV_Close(handle_);
    if (ret != IMV_OK)
    {
      RCLCPP_WARN(this->get_logger(), "IMV_Close failed: 0x%X", ret);
    }
    else
    {
      RCLCPP_DEBUG(this->get_logger(), "IMV_Close succeeded.");
    }
    is_opened_.store(false);
  }

  // 销毁句柄
  ret = IMV_DestroyHandle(handle_);
  if (ret != IMV_OK)
  {
    RCLCPP_WARN(this->get_logger(), "IMV_DestroyHandle failed: 0x%X", ret);
  }
  else
  {
    RCLCPP_DEBUG(this->get_logger(), "IMV_DestroyHandle succeeded.");
  }

  handle_ = nullptr;
  RCLCPP_INFO(this->get_logger(), "Huaray camera stopped and handle destroyed.");
}

void HuarayCameraNode::ProtectRunning()
{
  RCLCPP_INFO(this->get_logger(), "Protect thread started.");

  // 初始化时等待一小段时间，让 CaptureInit 有机会完成
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  std::unique_lock<std::mutex> lock(this->guard_.mux);
  while (running_.load())
  {
    // 等待条件变量，带超时以便定期检查
    this->guard_.is_quit.wait_for(
        lock, std::chrono::milliseconds(500),
        [this]
        {
          return (this->state_.load() == HuarayStateEnum::STOPPED) ||
                 (!this->running_.load());
        });

    if (!this->running_.load())
    {
      break;
    }

    // 只有当状态确实是 STOPPED 时才尝试重启
    if (this->state_.load() == HuarayStateEnum::STOPPED)
    {
      RCLCPP_INFO(this->get_logger(), "Camera stopped, attempting to restart...");
      this->CaptureStop();

      // 延时防止频繁重启，给相机一些恢复时间
      std::this_thread::sleep_for(std::chrono::seconds(2));

      if (!this->running_.load())
      {
        break;
      }

      this->CaptureInit();

      // 如果初始化仍然失败，等待更长时间再重试
      if (this->state_.load() == HuarayStateEnum::STOPPED)
      {
        RCLCPP_WARN(this->get_logger(),
                    "Camera restart failed, will retry in 5 seconds...");
        std::this_thread::sleep_for(std::chrono::seconds(5));
      }
    }
  }
  RCLCPP_INFO(this->get_logger(), "Protect thread exit.");
}

void HuarayCameraNode::SetFloatValue(const std::string& name, double value)
{
  if (handle_ == nullptr || !is_opened_.load())
  {
    return;
  }

  int ret = IMV_SetDoubleFeatureValue(handle_, name.c_str(), value);
  if (ret != IMV_OK)
  {
    RCLCPP_WARN(this->get_logger(), "IMV_SetDoubleFeatureValue(\"%s\", %f) failed: 0x%X",
                name.c_str(), value, ret);
  }
}

void HuarayCameraNode::SetEnumValue(const std::string& name, unsigned int value)
{
  if (handle_ == nullptr || !is_opened_.load())
  {
    return;
  }

  int ret = IMV_SetEnumFeatureValue(handle_, name.c_str(), value);
  if (ret != IMV_OK)
  {
    RCLCPP_WARN(this->get_logger(), "IMV_SetEnumFeatureValue(\"%s\", %u) failed: 0x%X",
                name.c_str(), value, ret);
  }
}

void HuarayCameraNode::SetEnumSymbol(const std::string& name, const std::string& value)
{
  if (handle_ == nullptr || !is_opened_.load())
  {
    return;
  }

  int ret = IMV_SetEnumFeatureSymbol(handle_, name.c_str(), value.c_str());
  if (ret != IMV_OK)
  {
    RCLCPP_WARN(this->get_logger(),
                "IMV_SetEnumFeatureSymbol(\"%s\", \"%s\") failed: 0x%X", name.c_str(),
                value.c_str(), ret);
  }
}
}  // namespace HuarayCamera

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(HuarayCamera::HuarayCameraNode)
