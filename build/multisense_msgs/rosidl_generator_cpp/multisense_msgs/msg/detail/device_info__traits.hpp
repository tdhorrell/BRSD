// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from multisense_msgs:msg/DeviceInfo.idl
// generated code does not contain a copyright notice

#ifndef MULTISENSE_MSGS__MSG__DETAIL__DEVICE_INFO__TRAITS_HPP_
#define MULTISENSE_MSGS__MSG__DETAIL__DEVICE_INFO__TRAITS_HPP_

#include "multisense_msgs/msg/detail/device_info__struct.hpp"
#include <rosidl_runtime_cpp/traits.hpp>
#include <stdint.h>
#include <type_traits>

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<multisense_msgs::msg::DeviceInfo>()
{
  return "multisense_msgs::msg::DeviceInfo";
}

template<>
inline const char * name<multisense_msgs::msg::DeviceInfo>()
{
  return "multisense_msgs/msg/DeviceInfo";
}

template<>
struct has_fixed_size<multisense_msgs::msg::DeviceInfo>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<multisense_msgs::msg::DeviceInfo>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<multisense_msgs::msg::DeviceInfo>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // MULTISENSE_MSGS__MSG__DETAIL__DEVICE_INFO__TRAITS_HPP_
