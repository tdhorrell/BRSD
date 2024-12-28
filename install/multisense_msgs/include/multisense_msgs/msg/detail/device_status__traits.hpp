// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from multisense_msgs:msg/DeviceStatus.idl
// generated code does not contain a copyright notice

#ifndef MULTISENSE_MSGS__MSG__DETAIL__DEVICE_STATUS__TRAITS_HPP_
#define MULTISENSE_MSGS__MSG__DETAIL__DEVICE_STATUS__TRAITS_HPP_

#include "multisense_msgs/msg/detail/device_status__struct.hpp"
#include <rosidl_runtime_cpp/traits.hpp>
#include <stdint.h>
#include <type_traits>

// Include directives for member types
// Member 'time_stamp'
// Member 'uptime'
#include "builtin_interfaces/msg/detail/time__traits.hpp"

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<multisense_msgs::msg::DeviceStatus>()
{
  return "multisense_msgs::msg::DeviceStatus";
}

template<>
inline const char * name<multisense_msgs::msg::DeviceStatus>()
{
  return "multisense_msgs/msg/DeviceStatus";
}

template<>
struct has_fixed_size<multisense_msgs::msg::DeviceStatus>
  : std::integral_constant<bool, has_fixed_size<builtin_interfaces::msg::Time>::value> {};

template<>
struct has_bounded_size<multisense_msgs::msg::DeviceStatus>
  : std::integral_constant<bool, has_bounded_size<builtin_interfaces::msg::Time>::value> {};

template<>
struct is_message<multisense_msgs::msg::DeviceStatus>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // MULTISENSE_MSGS__MSG__DETAIL__DEVICE_STATUS__TRAITS_HPP_
