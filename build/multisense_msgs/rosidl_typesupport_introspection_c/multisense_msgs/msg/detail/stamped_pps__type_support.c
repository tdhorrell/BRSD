// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from multisense_msgs:msg/StampedPps.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "multisense_msgs/msg/detail/stamped_pps__rosidl_typesupport_introspection_c.h"
#include "multisense_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "multisense_msgs/msg/detail/stamped_pps__functions.h"
#include "multisense_msgs/msg/detail/stamped_pps__struct.h"


// Include directives for member types
// Member `data`
// Member `host_time`
#include "builtin_interfaces/msg/time.h"
// Member `data`
// Member `host_time`
#include "builtin_interfaces/msg/detail/time__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void StampedPps__rosidl_typesupport_introspection_c__StampedPps_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  multisense_msgs__msg__StampedPps__init(message_memory);
}

void StampedPps__rosidl_typesupport_introspection_c__StampedPps_fini_function(void * message_memory)
{
  multisense_msgs__msg__StampedPps__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember StampedPps__rosidl_typesupport_introspection_c__StampedPps_message_member_array[2] = {
  {
    "data",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(multisense_msgs__msg__StampedPps, data),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "host_time",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(multisense_msgs__msg__StampedPps, host_time),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers StampedPps__rosidl_typesupport_introspection_c__StampedPps_message_members = {
  "multisense_msgs__msg",  // message namespace
  "StampedPps",  // message name
  2,  // number of fields
  sizeof(multisense_msgs__msg__StampedPps),
  StampedPps__rosidl_typesupport_introspection_c__StampedPps_message_member_array,  // message members
  StampedPps__rosidl_typesupport_introspection_c__StampedPps_init_function,  // function to initialize message memory (memory has to be allocated)
  StampedPps__rosidl_typesupport_introspection_c__StampedPps_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t StampedPps__rosidl_typesupport_introspection_c__StampedPps_message_type_support_handle = {
  0,
  &StampedPps__rosidl_typesupport_introspection_c__StampedPps_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_multisense_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, multisense_msgs, msg, StampedPps)() {
  StampedPps__rosidl_typesupport_introspection_c__StampedPps_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, builtin_interfaces, msg, Time)();
  StampedPps__rosidl_typesupport_introspection_c__StampedPps_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, builtin_interfaces, msg, Time)();
  if (!StampedPps__rosidl_typesupport_introspection_c__StampedPps_message_type_support_handle.typesupport_identifier) {
    StampedPps__rosidl_typesupport_introspection_c__StampedPps_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &StampedPps__rosidl_typesupport_introspection_c__StampedPps_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
