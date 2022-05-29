#include <CL/sycl.hpp>
#include <vector>
#include <iostream>
#include <numeric>
#include "utils.hpp"



// ����������� �������� �������, ����� ��� ����������
// ��� �������� nVidia ���� ��������� ���������� https://github.com/intel/llvm

//using namespace sycl;

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

template <typename T = float, const size_t dimention = 1>
using read_accessor_t = sycl::accessor<T, dimention, sycl::access::mode::read, sycl::access::target::global_buffer>;

template <typename T = float, const size_t dimention = 1>
using write_accessor_t = sycl::accessor<float, 1, sycl::access::mode::write, sycl::access::target::global_buffer>;

// �������
struct Add
{
	using read_accessor = read_accessor_t<float, 1>;
	using write_accessor = write_accessor_t<float, 1>;

	read_accessor m_in_a;
	read_accessor m_in_b;

	write_accessor m_out_c;

	void operator()(sycl::id<1> index) const
	{
		m_out_c[index] = m_in_a[index] + m_in_b[index];
	}
};

// 1) ������ �� ��������� � ������ �������� ����� �� ������?
// 2) ����������� � local mem � work group: ���� �� ����� � ������� �������?
// 2.1) ������ ������ ���������� GROUP_SIZE ���������. ���������� � ����� ������?
// 3) GROUP_SIZE > 8 ������ ������� ���������� �� ���������? ������ �� ��������...
// 4) sycl::program ????? �� ��������
// 5) ������ ����� ���� 2 ���������������� parallel_for �� ������ ������ ���-�� �� �����������?

// ����� ������� ������� � debug � release �������? (�������������)

constexpr size_t N = 1024;

constexpr size_t GROUP_SIZE = 8; // ������ ������� ���������� �� ���������?? 
constexpr size_t GROUP_COUNT = (N / GROUP_SIZE);

int main(int argc, char* argv[])
{
	get_devices_information();

	std::vector<float> a(N, 1.0f), b(N, 2.0f), c(N, 0.0f);
	try
	{
		{
			sycl::buffer<float, 1> buf_a(a.__storage(), a.size());
			sycl::buffer<float, 1> buf_b(b.__storage(), b.size());
			sycl::buffer<float, 1> buf_c(c.__storage(), c.size());

			sycl::property_list props{ sycl::property::queue::enable_profiling() };

			sycl::queue gpu_queue(sycl::gpu_selector{}, [](sycl::exception_list exceptions) {
				for (auto& exception : exceptions)
				{
					try
					{
						std::rethrow_exception(exception);
					}
					catch (sycl::exception& exception)
					{
						std::cerr << "Asynch error: " << exception.what() << std::endl;
					}
				}
				}, props); // props �������������� ��������, ������� ����� ��� ��������������

				// 4) sycl::program ????? �� ��������
				// 
				//sycl::program ????? �� ��������... 
				//sycl::program add_program(gpu_queue.get_context());
				//add_program.build_with_kernel_type<Add>();
				//auto add_kernel = add_program.get_kernel<Add>();
				//cgh.parallel_for(add_kernel, sycl::range<1>(N), Add { in_a, in_b, out_c }; // ��� � submit

				// common group handler - cgh
			sycl::event event = gpu_queue.submit([&](sycl::handler& cgh) {
				sycl::stream output(1024, 80, cgh);

				auto in_a = buf_a.get_access<sycl::access::mode::read>(cgh);
				auto in_b = buf_b.get_access<sycl::access::mode::read>(cgh);
				auto out_c = buf_c.get_access<sycl::access::mode::write>(cgh);

				auto work_items = sycl::nd_range<2>(sycl::range<2>(12, 12), sycl::range<2>(4, 4));

				// 5) ������ ����� ���� 2 ���������������� parallel_for �� ������ ������ ���-�� �� �����������?
				//cgh.parallel_for(work_items, [=](sycl::nd_item<2> item) {
				//	// ����������� ����� ������������?
				//	//if (item.get_global_id(0) == 4 && item.get_global_id(1) == 5)
				//	//{
				//		output << "global range: {" << item.get_global_range(0) << ", "
				//									<< item.get_global_range(1) << "}"	<< sycl::endl;

				//		output << "global id: {"	<< item.get_global_id(0) << ", "
				//									<< item.get_global_id(1) << "}"		<< sycl::endl;

				//		output << "group range: {"	<< item.get_group_range(0) << ", "
				//									<< item.get_group_range(1) << "}"	<< sycl::endl;

				//		output << "group id: {"		<< item.get_group(0) << ", "
				//									<< item.get_group(1) << "}"			<< sycl::endl; 

				//		output << "local range: {"	<< item.get_local_range(0) << ", "
				//									<< item.get_local_range(1) << "}"	<< sycl::endl;

				//		output << "local id: {"		<< item.get_local_id(0) << ", "
				//									<< item.get_local_id(1) << "}"		<< sycl::endl; 
				//	//}
				//});

				// ������ ���� ������ ���� �� ���� parallel_for ? (�� ����� �������������������� ����� ����?)

				//cgh.parallel_for<class Add>(sycl::range<1>(N), 
				//	[=](sycl::id<1> index) // ���������� ���������� � work item - id
				//	//[=](sycl::item<1> item) // ������ ���������� � work item
				//		[[intel::reqd_sub_group_size(16)]] // ��� ����� ������ ����� 16 ���������������� ������� ������������ � ���� ��� ������
				//	{
				//		const size_t index = item.get_id();
				//		out_c[index] = in_a[index] + in_b[index]; // Kernel code 

				//		if (index == 0)
				//		{
				//			output << "Device index is zero!" << sycl::endl;
				//		}
				//	});

				//cgh.parallel_for(sycl::range<1>(N), Add{ in_a , in_b, out_c });



				// sycl::accessor<T, dimensions, access_mode, access_target, placeholder>
				// T				- ����������� ��� ������
				// dimensions
				// access_mode
				// access_target	- ������, depricated ��������, ����� device, constant_buffer, local - ��� ����� �������� ������ (�������������, �����������, ��������� ������)
				// placeholder		- ��-��������� false, ��� ��������, ��� �� ����� ���� ������ ������ ������ submit (������������� ����������� � common group �� handler)
				//					  ����� ����� ������� ��� submit, �� ����� ������������� �������� � cgh ����� required (?), ����� ��������� ���������� 

				// 2) ����������� � local mem � work group: ���� �� ����� � ������� �������?
				// 2.1) ������ ������ ���������� GROUP_SIZE ���������. ���������� � ����� ������?
				// ������ ������ ���������� GROUP_SIZE ���������. ������, ��� ��� ���������� �������, ������� ����� ������ �������?
				// � ������ ������ GROUP_SIZE �� ������� ���� ����������, ��� ����� ���� ������������ ����������
				sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> lmem_a(GROUP_SIZE, cgh);

				cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(N), sycl::range<1>(GROUP_SIZE)),
					[=](sycl::nd_item<1> nd_item) // ���������� ���������� ���������� � work group
					{
						const size_t global_index = nd_item.get_global_id(0);
						const size_t local_index = nd_item.get_local_id(0);

						sycl::multi_ptr<float[GROUP_SIZE], sycl::access::address_space::local_space> ptr = sycl::ext::oneapi::group_local_memory<float[GROUP_SIZE]>(nd_item.get_group());

						float* lmem_b = *ptr; // ������������� ���������� ��� ������ float[GROUP_SIZE], ��� ���������� ���������

						// ������ ������ ������ ���������� ������
						lmem_a[local_index] = in_a[global_index];
						lmem_b[local_index] = in_b[global_index];

						// ��������� ������ �� ���������� � ��������� ������, ��� ���� barrier ������������ data race, �.�. ��� ������ ��������, ���� �� ������ �� ����� �����.
						// � ������ ������ ��� �� �����������
						nd_item.barrier(sycl::access::fence_space::local_space);

						out_c[global_index] = lmem_a[local_index] + lmem_b[local_index]; // Kernel code 

						// out_c[index] = sqrtf(out_c[index]);

						if (global_index == 0)
						{
							output << "Device index is zero!" << sycl::endl;
						}
					});


				// ������������� ���������� � ����� ���� �������� � ������:
				// Synch error: Attempt to set multiple actions for the command group. Command group must consist of a single kernel or explicit memory operation. -59 (CL_INVALID_OPERATION)
				//cgh.host_task(
				//	[=]() {
				//		output << "Host task!" << sycl::endl;

				//		// ����������� �� CPU	
				//	});

				//cgh.single_task(
				//	[=]() {
				//		output << "Single task!" << sycl::endl;

				//		// ��� �� ���������� �� ����� ������
				//		// ��� ���������� ������ �������, ��� ������������ ������ ����-����
				//	});

				// ������ ��������, �������������� ���������, ������������ ���������, ����������� �������� ������ ������� ����

				});

			gpu_queue.wait();

			uint64_t start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
			uint64_t end = event.get_profiling_info<sycl::info::event_profiling::command_end>();

			std::cout << "Kernel execution time: " << (end - start) << " ns" << std::endl;
		}
	}
	catch (sycl::exception exception)
	{
		std::cerr << "Synch error: " << exception.what() << std::endl;
	}
	catch (...)
	{
		std::cerr << "General error" << std::endl;
	}

	for (const auto& it : c)
	{
		assert(it == 3.0f);
	}

	std::vector<float> r(GROUP_COUNT, 0.0f);
	try
	{
		{
			sycl::property_list props{ sycl::property::queue::enable_profiling() };

			sycl::queue gpu_queue(sycl::cpu_selector{}, props); // ����������� �� cpu_selector 

			sycl::buffer<float, 1> buf_b(b.__storage(), b.size());
			sycl::buffer<float, 1> buf_r(r.__storage(), r.size());

			sycl::event event = gpu_queue.submit(
				[&](sycl::handler& cgh) {

					sycl::stream output(1024, 80, cgh);

					auto in_b = buf_b.get_access<sycl::access::mode::read, sycl::access::target::constant_buffer>(cgh);
					auto out_r = buf_r.get_access<sycl::access::mode::write>(cgh);

					// 3) GROUP_SIZE > 8 ������ ������� ���������� �� ���������? ������ �� ��������...
					cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(N), sycl::range<1>(GROUP_SIZE)),
						[=](sycl::nd_item<1> item)
						{
							const size_t global_index = item.get_global_id(0);
							const size_t local_index = item.get_local_id(0);

							const float x = in_b[global_index] * in_b[global_index];
							const float dot = sycl::reduce_over_group(item.get_group(), x, 0.0f, sycl::ext::oneapi::plus<float>()); // ����� ���� ����� �� ���� work group

							// ��������� �� ��� ������...
							output << global_index << '\n';

							if (local_index == 0)
							{
								const size_t group_index = item.get_group(0);

								out_r[group_index] = dot;
							}
						});
				});

			event.wait_and_throw();

			// ����� ��� ������ �� scope ����������� �������� �� host � device
			// ���� �� ����������� ��� ���� ���������� accessor ��� r, �� ��� ������� �� ����������, �.�. �� ����� ������� ������ � ������.
			auto host_r = buf_r.get_access<sycl::access::mode::read>();

			std::cout << host_r[0] << '\n';

			const auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
			const auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();

			std::cout << "Kernel execution time: " << (end - start) << " ns" << std::endl;
		}
	}
	catch (sycl::exception exception)
	{
		std::cerr << "Synch error: " << exception.what() << std::endl;
	}
	catch (...)
	{
		std::cerr << "General error" << std::endl;
	}

	// ???????????
	assert(std::accumulate(r.begin(), r.end(), 0.0f) == 4.0f * N);

	return 0;
}