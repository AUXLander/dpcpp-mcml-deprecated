#pragma once

#include <CL/sycl.hpp>

#include <vector>
#include <iostream>

void get_devices_information()
{
	std::vector<sycl::platform> platforms = sycl::platform::get_platforms();

	for (auto& platform : platforms)
	{
		std::cout << "Platform: " << platform.get_info<sycl::info::platform::name>() << std::endl;

		std::vector<sycl::device> devices = platform.get_devices();
		for (auto& device : devices)
		{
			std::cout << "Device: " << device.get_info<sycl::info::device::name>() << std::endl;
		}

		std::cout << std::endl;
	}
}