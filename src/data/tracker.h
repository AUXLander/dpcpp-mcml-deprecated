#pragma once
#include <vector>
#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <functional>
#include <shared_mutex>
#include <ios>
#include <cassert>
#include <map>
#include <set>
#include <deque>
#include <unordered_map>
#include <set>

#include <boost/functional/hash.hpp>


enum : size_t
{
	DPI_X = 10U,
	DPI_Y = 10U,
	DPI_Z = 10U,
};

enum : size_t
{
	DOTS_PER_X     = 1U,
	DOTS_PER_Y     = DOTS_PER_X * DPI_X,
	DOTS_PER_Z	   = DOTS_PER_Y * DPI_Y,
	DOTS_PER_LAYER = DOTS_PER_Z * DPI_Z
};

struct point
{
	size_t x_idx;
	size_t y_idx;
	size_t z_idx;
	size_t l_idx;

private:
	size_t index;

public:

	point(size_t x_idx, size_t y_idx, size_t z_idx, size_t l_idx) :
		x_idx(x_idx), y_idx(y_idx), z_idx(z_idx), l_idx(l_idx - 1U)
	{
		index = this->x_idx * DOTS_PER_X +
				this->z_idx * DOTS_PER_Y +
				this->y_idx * DOTS_PER_Z +
			    this->l_idx * DOTS_PER_LAYER;
	}

	inline size_t to_index() const noexcept
	{
		return index;
	}

	bool operator==(const point& other) const
	{
		return x_idx == other.x_idx && 
			   y_idx == other.y_idx && 
			   z_idx == other.z_idx && 
			   l_idx == other.l_idx;
	}

	struct hash
	{
		size_t operator()(const point& p) const
		{
			size_t hash = 0;

			boost::hash_combine(hash, p.x_idx);
			boost::hash_combine(hash, p.y_idx);
			boost::hash_combine(hash, p.z_idx);
			boost::hash_combine(hash, p.l_idx);

			return hash;
		}
	};
};


static double z_max = 0;
static double z_min = 0;


struct tracker
{
	struct local_thread_storage
	{
		std::set<size_t> zs;

		using dot_counter_t = size_t;

		constexpr static double min_x = -1.0;
		constexpr static double min_y = -1.0;
		constexpr static double min_z = -0.0;

		constexpr static double max_x = +1.0;
		constexpr static double max_y = +1.0;
		constexpr static double max_z = +0.2;

		constexpr static double step_x = (max_x - min_x) / static_cast<double>(DPI_X);
		constexpr static double step_y = (max_y - min_y) / static_cast<double>(DPI_Y);
		constexpr static double step_z = (max_z - min_z) / static_cast<double>(DPI_Z);

		friend class tracker;

		enum class combine_mode_e
		{
			COMBINE_ADD,
		};

	private:

		std::unordered_map<point, size_t, point::hash> __heathash;

		size_t __total_count;

	public:
		local_thread_storage() :
			__total_count {0}
		{;}

		local_thread_storage(const local_thread_storage& other) = delete;

		local_thread_storage(local_thread_storage&& other) = default;

		void track(const double& x, const double& y, const double& z, size_t layer)
		{
			const point key {
				static_cast<dot_counter_t>((x - min_x) / step_x),
				static_cast<dot_counter_t>((y - min_y) / step_y),
				static_cast<dot_counter_t>((z - min_z) / step_z),
				layer
			};

			auto [it, inserted] = __heathash.try_emplace(key, 0U);

			it->second += 1U;

			// auto& item = __heatmap[key.to_index()];

			// ++item;

			// __heathash.try_emplace(key, &item);

			++__total_count;


			z_max = std::fmax(z_max, z);
			z_min = std::fmin(z_min, z);

			zs.insert(key.z_idx);

		}

		void combine(const local_thread_storage& rhs, combine_mode_e mode)
		{
			for (auto item : rhs.zs)
			{
				zs.insert(item);
			}

			switch (mode)
			{
				case combine_mode_e::COMBINE_ADD:
				{
					for (auto [key, value] : rhs.__heathash)
					{
						auto [it, inserted] = __heathash.try_emplace(key, 0U);

						it->second += value;
					}

					__total_count += rhs.__total_count;
				};
				break;
			}
		}

		void write(std::fstream& stream) const 
		{
			const size_t layers_count = 5U;

			std::deque<dot_counter_t> __heatmap(DPI_X * DPI_Y * DPI_Z * layers_count, 0U);

			for (const auto [key, value] : __heathash)
			{
				const auto index = key.to_index();

				if (index <= __heatmap.size())
				{
					__heatmap[index] = value;
				}
			}

			stream.write((const char*)&__total_count, sizeof(__total_count));

			for (const auto count : __heatmap)
			{
				stream.write((const char*)&count, sizeof(count));
			}
		}
	};

private:

	local_thread_storage __main_thread_storage;

	constexpr static auto openmode = std::ios_base::binary | std::ios_base::out;

	struct fstream_deleter
	{
		void operator()(std::fstream* s)
		{
			assert(s);
			s->close();
			delete s;
		}
	};

	std::unique_ptr<std::fstream, fstream_deleter> __file;

	mutable std::shared_mutex __mutex;

	tracker() {;}
	
public:

	static tracker& instance()
	{
		static tracker INSTANCE;
		return INSTANCE;
	}

	void track(local_thread_storage&& __storage)
	{
		std::unique_lock lock(__mutex);

		__main_thread_storage.combine(std::move(__storage), local_thread_storage::combine_mode_e::COMBINE_ADD);
	}

	void set_file(const char* filename)
	{
		__file.reset(new std::fstream{ filename, openmode });
	}

	template<typename Tfunc>
	void set_headers(Tfunc fgen)
	{
		std::shared_lock lock(__mutex);

		assert(__file);

		if (__file)
		{
			fgen(*__file);

			__file->flush();
		}
	}

	void write() const
	{
		std::shared_lock lock(__mutex);

		assert(__file);

		if (__file)
		{
			double min_x = -1.0;
			double min_y = -1.0;
			double min_z = -0.0;

			double max_x = +1.0;
			double max_y = +1.0;
			double max_z = +0.2;

			size_t dpi_x = DPI_X;
			size_t dpi_y = DPI_Y;
			size_t dpi_z = DPI_Z;

			auto& stream = *__file;

			__file->flush();

			__main_thread_storage.write(stream);

			__file->flush();

			// num of layers + photons
			__file->seekg(sizeof(short) + sizeof(long));
			__file->seekp(sizeof(short) + sizeof(long));

			__file->write((const char*)&dpi_x, sizeof(dpi_x));
			__file->write((const char*)&dpi_y, sizeof(dpi_y));
			__file->write((const char*)&dpi_z, sizeof(dpi_z));

			__file->write((const char*)&min_x, sizeof(min_x));
			__file->write((const char*)&min_y, sizeof(min_y));
			__file->write((const char*)&min_z, sizeof(min_z));

			__file->write((const char*)&max_x, sizeof(max_x));
			__file->write((const char*)&max_y, sizeof(max_y));
			__file->write((const char*)&max_z, sizeof(max_z));

			__file->flush();
		}
	}
};