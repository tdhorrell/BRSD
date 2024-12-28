// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from multisense_msgs:msg/StampedPps.idl
// generated code does not contain a copyright notice
#include "multisense_msgs/msg/detail/stamped_pps__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "multisense_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "multisense_msgs/msg/detail/stamped_pps__struct.h"
#include "multisense_msgs/msg/detail/stamped_pps__functions.h"
#include "fastcdr/Cdr.h"

#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"
# ifdef __clang__
#  pragma clang diagnostic ignored "-Wdeprecated-register"
#  pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
# endif
#endif
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif

// includes and forward declarations of message dependencies and their conversion functions

#if defined(__cplusplus)
extern "C"
{
#endif

#include "builtin_interfaces/msg/detail/time__functions.h"  // data, host_time

// forward declare type support functions
ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_multisense_msgs
size_t get_serialized_size_builtin_interfaces__msg__Time(
  const void * untyped_ros_message,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_multisense_msgs
size_t max_serialized_size_builtin_interfaces__msg__Time(
  bool & full_bounded,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_multisense_msgs
const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, builtin_interfaces, msg, Time)();


using _StampedPps__ros_msg_type = multisense_msgs__msg__StampedPps;

static bool _StampedPps__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _StampedPps__ros_msg_type * ros_message = static_cast<const _StampedPps__ros_msg_type *>(untyped_ros_message);
  // Field name: data
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, builtin_interfaces, msg, Time
      )()->data);
    if (!callbacks->cdr_serialize(
        &ros_message->data, cdr))
    {
      return false;
    }
  }

  // Field name: host_time
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, builtin_interfaces, msg, Time
      )()->data);
    if (!callbacks->cdr_serialize(
        &ros_message->host_time, cdr))
    {
      return false;
    }
  }

  return true;
}

static bool _StampedPps__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _StampedPps__ros_msg_type * ros_message = static_cast<_StampedPps__ros_msg_type *>(untyped_ros_message);
  // Field name: data
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, builtin_interfaces, msg, Time
      )()->data);
    if (!callbacks->cdr_deserialize(
        cdr, &ros_message->data))
    {
      return false;
    }
  }

  // Field name: host_time
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, builtin_interfaces, msg, Time
      )()->data);
    if (!callbacks->cdr_deserialize(
        cdr, &ros_message->host_time))
    {
      return false;
    }
  }

  return true;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_multisense_msgs
size_t get_serialized_size_multisense_msgs__msg__StampedPps(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _StampedPps__ros_msg_type * ros_message = static_cast<const _StampedPps__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name data

  current_alignment += get_serialized_size_builtin_interfaces__msg__Time(
    &(ros_message->data), current_alignment);
  // field.name host_time

  current_alignment += get_serialized_size_builtin_interfaces__msg__Time(
    &(ros_message->host_time), current_alignment);

  return current_alignment - initial_alignment;
}

static uint32_t _StampedPps__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_multisense_msgs__msg__StampedPps(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_multisense_msgs
size_t max_serialized_size_multisense_msgs__msg__StampedPps(
  bool & full_bounded,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;
  (void)full_bounded;

  // member: data
  {
    size_t array_size = 1;


    for (size_t index = 0; index < array_size; ++index) {
      current_alignment +=
        max_serialized_size_builtin_interfaces__msg__Time(
        full_bounded, current_alignment);
    }
  }
  // member: host_time
  {
    size_t array_size = 1;


    for (size_t index = 0; index < array_size; ++index) {
      current_alignment +=
        max_serialized_size_builtin_interfaces__msg__Time(
        full_bounded, current_alignment);
    }
  }

  return current_alignment - initial_alignment;
}

static size_t _StampedPps__max_serialized_size(bool & full_bounded)
{
  return max_serialized_size_multisense_msgs__msg__StampedPps(
    full_bounded, 0);
}


static message_type_support_callbacks_t __callbacks_StampedPps = {
  "multisense_msgs::msg",
  "StampedPps",
  _StampedPps__cdr_serialize,
  _StampedPps__cdr_deserialize,
  _StampedPps__get_serialized_size,
  _StampedPps__max_serialized_size
};

static rosidl_message_type_support_t _StampedPps__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_StampedPps,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, multisense_msgs, msg, StampedPps)() {
  return &_StampedPps__type_support;
}

#if defined(__cplusplus)
}
#endif
