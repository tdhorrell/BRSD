// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from multisense_msgs:msg/RawLidarData.idl
// generated code does not contain a copyright notice
#include "multisense_msgs/msg/detail/raw_lidar_data__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "multisense_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "multisense_msgs/msg/detail/raw_lidar_data__struct.h"
#include "multisense_msgs/msg/detail/raw_lidar_data__functions.h"
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

#include "builtin_interfaces/msg/detail/time__functions.h"  // time_end, time_start
#include "rosidl_runtime_c/primitives_sequence.h"  // distance, intensity
#include "rosidl_runtime_c/primitives_sequence_functions.h"  // distance, intensity

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


using _RawLidarData__ros_msg_type = multisense_msgs__msg__RawLidarData;

static bool _RawLidarData__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _RawLidarData__ros_msg_type * ros_message = static_cast<const _RawLidarData__ros_msg_type *>(untyped_ros_message);
  // Field name: scan_count
  {
    cdr << ros_message->scan_count;
  }

  // Field name: time_start
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, builtin_interfaces, msg, Time
      )()->data);
    if (!callbacks->cdr_serialize(
        &ros_message->time_start, cdr))
    {
      return false;
    }
  }

  // Field name: time_end
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, builtin_interfaces, msg, Time
      )()->data);
    if (!callbacks->cdr_serialize(
        &ros_message->time_end, cdr))
    {
      return false;
    }
  }

  // Field name: angle_start
  {
    cdr << ros_message->angle_start;
  }

  // Field name: angle_end
  {
    cdr << ros_message->angle_end;
  }

  // Field name: distance
  {
    size_t size = ros_message->distance.size;
    auto array_ptr = ros_message->distance.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: intensity
  {
    size_t size = ros_message->intensity.size;
    auto array_ptr = ros_message->intensity.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  return true;
}

static bool _RawLidarData__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _RawLidarData__ros_msg_type * ros_message = static_cast<_RawLidarData__ros_msg_type *>(untyped_ros_message);
  // Field name: scan_count
  {
    cdr >> ros_message->scan_count;
  }

  // Field name: time_start
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, builtin_interfaces, msg, Time
      )()->data);
    if (!callbacks->cdr_deserialize(
        cdr, &ros_message->time_start))
    {
      return false;
    }
  }

  // Field name: time_end
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, builtin_interfaces, msg, Time
      )()->data);
    if (!callbacks->cdr_deserialize(
        cdr, &ros_message->time_end))
    {
      return false;
    }
  }

  // Field name: angle_start
  {
    cdr >> ros_message->angle_start;
  }

  // Field name: angle_end
  {
    cdr >> ros_message->angle_end;
  }

  // Field name: distance
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->distance.data) {
      rosidl_runtime_c__uint32__Sequence__fini(&ros_message->distance);
    }
    if (!rosidl_runtime_c__uint32__Sequence__init(&ros_message->distance, size)) {
      return "failed to create array for field 'distance'";
    }
    auto array_ptr = ros_message->distance.data;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: intensity
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->intensity.data) {
      rosidl_runtime_c__uint32__Sequence__fini(&ros_message->intensity);
    }
    if (!rosidl_runtime_c__uint32__Sequence__init(&ros_message->intensity, size)) {
      return "failed to create array for field 'intensity'";
    }
    auto array_ptr = ros_message->intensity.data;
    cdr.deserializeArray(array_ptr, size);
  }

  return true;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_multisense_msgs
size_t get_serialized_size_multisense_msgs__msg__RawLidarData(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _RawLidarData__ros_msg_type * ros_message = static_cast<const _RawLidarData__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name scan_count
  {
    size_t item_size = sizeof(ros_message->scan_count);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name time_start

  current_alignment += get_serialized_size_builtin_interfaces__msg__Time(
    &(ros_message->time_start), current_alignment);
  // field.name time_end

  current_alignment += get_serialized_size_builtin_interfaces__msg__Time(
    &(ros_message->time_end), current_alignment);
  // field.name angle_start
  {
    size_t item_size = sizeof(ros_message->angle_start);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name angle_end
  {
    size_t item_size = sizeof(ros_message->angle_end);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name distance
  {
    size_t array_size = ros_message->distance.size;
    auto array_ptr = ros_message->distance.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name intensity
  {
    size_t array_size = ros_message->intensity.size;
    auto array_ptr = ros_message->intensity.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

static uint32_t _RawLidarData__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_multisense_msgs__msg__RawLidarData(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_multisense_msgs
size_t max_serialized_size_multisense_msgs__msg__RawLidarData(
  bool & full_bounded,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;
  (void)full_bounded;

  // member: scan_count
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: time_start
  {
    size_t array_size = 1;


    for (size_t index = 0; index < array_size; ++index) {
      current_alignment +=
        max_serialized_size_builtin_interfaces__msg__Time(
        full_bounded, current_alignment);
    }
  }
  // member: time_end
  {
    size_t array_size = 1;


    for (size_t index = 0; index < array_size; ++index) {
      current_alignment +=
        max_serialized_size_builtin_interfaces__msg__Time(
        full_bounded, current_alignment);
    }
  }
  // member: angle_start
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: angle_end
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: distance
  {
    size_t array_size = 0;
    full_bounded = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: intensity
  {
    size_t array_size = 0;
    full_bounded = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  return current_alignment - initial_alignment;
}

static size_t _RawLidarData__max_serialized_size(bool & full_bounded)
{
  return max_serialized_size_multisense_msgs__msg__RawLidarData(
    full_bounded, 0);
}


static message_type_support_callbacks_t __callbacks_RawLidarData = {
  "multisense_msgs::msg",
  "RawLidarData",
  _RawLidarData__cdr_serialize,
  _RawLidarData__cdr_deserialize,
  _RawLidarData__get_serialized_size,
  _RawLidarData__max_serialized_size
};

static rosidl_message_type_support_t _RawLidarData__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_RawLidarData,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, multisense_msgs, msg, RawLidarData)() {
  return &_RawLidarData__type_support;
}

#if defined(__cplusplus)
}
#endif
