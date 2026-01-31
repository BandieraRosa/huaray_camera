# ros2_huaray_camera

A ROS2 packge for IRayple USB3.0 industrial camera

## Usage

```
ros2 launch huaray_camera huaray_camera.launch.py
```

dafault choose the **second** device. 
if at **NUC**, change line 262: 
const char* camera_key = device_list.pDevInfo[1].cameraKey; 
to:
const char* camera_key = device_list.pDevInfo[0].cameraKey; 

