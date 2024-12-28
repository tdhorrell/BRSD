// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__type_support.cpp.em
// with input from multisense_msgs:msg/RawLidarCal.idl
// generated code does not contain a copyright notice
#include "multisense_msgs/msg/detail/raw_lidar_cal__rosidl_typesupport_fastrtps_cpp.hpp"
#include "multisense_msgs/msg/detail/raw_lidar_cal__struct.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_fastrtps_cpp/identifier.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_fastrtps_cpp/wstring_conversion.hpp"
#include "fastcdr/Cdr.h"


// forward declaration of message dependencies and their conversion functions

namespace multisense_msgs
{

namespace msg
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_multisense_msgs
cdr_serialize(
  const multisense_msgs::msg::RawLidarCal & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: laser_to_spindle
  {
    cdr << ros_message.laser_to_spindle;
  }
  // Member: camera_to_spindle_fixed
  {
    cdr << ros_message.camera_to_spindle_fixed;
  }
  return true;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_multisense_msgs
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  multisense_msgs::msg::RawLidarCal & ros_message)
{
  // Member: laser_to_spindle
  {
    cdr >> ros_message.laser_to_spindle;
  }

  // Member: camera_to_spindle_fixed
  {
    cdr >> ros_message.camera_to_spindle_fixed;
  }

  return true;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_multisense_msgs
get_serialized_size(
  const multisense_msgs::msg::RawLidarCal & ros_message,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Member: laser_to_spindle
  {
    size_t array_size = 16;
    size_t item_size = sizeof(ros_message.laser_to_spindle[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: camera_to_spindle_fixed
  {
    size_t array_size = 16;
    size_t item_size = sizeof(ros_message.camera_to_spindle_fixed[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_multisense_msgs
max_serialized_size_RawLidarCal(
  bool & full_bounded,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;
  (void)full_bounded;


  // Member: laser_to_spindle
  {
    size_t array_size = 16;

    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: camera_to_spindle_fixed
  {
    size_t array_size = 16;

    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  return current_alignment - initial_alignment;
}

static bool _RawLidarCal__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  auto typed_message =
    static_cast<const multisense_msgs::msg::RawLidarCal *>(
    untyped_ros_message);
  return cdr_serialize(*typed_message, cdr);
}

static bool _RawLidarCal__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  auto typed_message =
    static_cast<multisense_msgs::msg::RawLidarCal *>(
    untyped_ros_message);
  return cdr_deserialize(cdr, *typed_message);
}

static uint32_t _RawLidarCal__get_serialized_size(
  const void * untyped_ros_message)
{
  auto typed_message =
    static_cast<const multisense_msgs::msg::RawLidarCal *>(
    untyped_ros_message);
  return static_cast<uint32_t>(get_serialized_size(*typed_message, 0));
}

static size_t _RawLidarCal__max_serialized_size(bool & full_bounded)
{
  return max_serialized_size_RawLidarCal(full_bounded, 0);
}

static message_type_support_callbacks_t _RawLidarCal__callbacks = {
  "multisense_msgs::msg",
  "RawLidarCal",
  _RawLidarCal__cdr_serialize,
  _RawLidarCal__cdr_deserialize,
  _RawLidarCal__get_serialized_size,
  _RawLidarCal__max_serialized_size
};

static rosidl_message_type_support_t _RawLidarCal__handle = {
  rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
  &_RawLidarCal__callbacks,
  get_message_typesupport_handle_function,
};

}  // namespace typesupport_fastrtps_cpp

}  // namespace msg

}  // namespace multisense_msgs

namespace rosidl_typesupport_fastrtps_cpp
{

template<>
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_EXPORT_multisense_msgs
const rosidl_message_type_support_t *
get_message_type_support_handle<multisense_msgs::msg::RawLidarCal>()
{
  return &multisense_msgs::msg::typesupport_fastrtps_cpp::_RawLidarCal__handle;
}

}  // namespace rosidl_typesupport_fastrtps_cpp

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, multisense_msgs, msg, RawLidarCal)() {
  return &multisense_msgs::msg::typesupport_fastrtps_cpp::_RawLidarCal__handle;
}

#ifdef __cplusplus
}
#endif
