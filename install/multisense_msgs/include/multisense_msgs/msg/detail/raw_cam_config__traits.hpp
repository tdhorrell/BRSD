// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from multisense_msgs:msg/RawCamConfig.idl
// generated code does not contain a copyright notice

#ifndef MULTISENSE_MSGS__MSG__DETAIL__RAW_CAM_CONFIG__TRAITS_HPP_
#define MULTISENSE_MSGS__MSG__DETAIL__RAW_CAM_CONFIG__TRAITS_HPP_

#include "multisense_msgs/msg/detail/raw_cam_config__struct.hpp"
#include <rosidl_runtime_cpp/traits.hpp>
#include <stdint.h>
#include <type_traits>

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<multisense_msgs::msg::RawCamConfig>()
{
  return "multisense_msgs::msg::RawCamConfig";
}

template<>
inline const char * name<multisense_msgs::msg::RawCamConfig>()
{
  return "multisense_msgs/msg/RawCamConfig";
}

template<>
struct has_fixed_size<multisense_msgs::msg::RawCamConfig>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<multisense_msgs::msg::RawCamConfig>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<multisense_msgs::msg::RawCamConfig>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // MULTISENSE_MSGS__MSG__DETAIL__RAW_CAM_CONFIG__TRAITS_HPP_
