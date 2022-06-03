#pragma once

#include <cassert>

//#include "../mcmlnr.hpp"
#include <memory>
#include <numeric>
#include <algorithm>
#include <unordered_map>

#include <array>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/functional/hash.hpp>

#include <CL/sycl.hpp>

#include <iostream>
#include <iomanip>

#define STRLEN 256		/* String length. */

struct LayerStruct;

#define CPP_COMPILE


/***********************************************************
 *	Report error message to stderr, then exit the program
 *	with signal 1.
 ****/
void nrerror(const char error_text[]);

struct base_data_accessor
{
	enum size_e : size_t
	{
		x = 0U,
		y,

		TOTAL_SIZE_LENGTH
	};

	using dimension_sizes_t = std::array<size_t, TOTAL_SIZE_LENGTH>;

	const dimension_sizes_t size { 0U };

	template<typename ...Args>
	base_data_accessor(Args&& ...args) :
		size({ std::forward<Args>(args)... })
	{;}
};


template<typename T>
struct sycl_device_accessor : public base_data_accessor
{
	using value_t = typename T;

	using pointer_t = value_t*;
	using reference_t = value_t&;
	using const_reference_t = const value_t&;

	template<typename ...Args>
	sycl_device_accessor<value_t>(sycl::handler& cgh, sycl::buffer<value_t, 1U>& buffer, Args&& ...args) :
		base_data_accessor(std::forward<Args>(args)...),
		__read_accessor(buffer.get_access<sycl::access::mode::read>(cgh)),
		__write_accessor(buffer.get_access<sycl::access::mode::write>(cgh)),
		__universal_accessor(buffer.get_access<sycl::access::mode::read_write>(cgh))
	{;}

	sycl_device_accessor<value_t>(const sycl_device_accessor<value_t>&) = delete;
	sycl_device_accessor<value_t>(sycl_device_accessor<value_t>&&) = default;
	~sycl_device_accessor<value_t>() = default;

	reference_t get_value(size_t i, size_t j)
	{
		assert(i < size[size_e::y]);
		assert(j < size[size_e::x]);

		return __write_accessor[size[size_e::x] * i + j];
	}

	const_reference_t get_value(size_t i, size_t j) const
	{
		assert(i < size[size_e::y]);
		assert(j < size[size_e::x]);

		return __read_accessor[size[size_e::x] * i + j];
	}

// private:

public: // debug

	sycl::accessor<value_t, 1U, sycl::access::mode::read,       sycl::access::target::global_buffer> __read_accessor;
	sycl::accessor<value_t, 1U, sycl::access::mode::write,	    sycl::access::target::global_buffer> __write_accessor;
	sycl::accessor<value_t, 1U, sycl::access::mode::read_write, sycl::access::target::global_buffer> __universal_accessor;
};

template<typename T>
struct raw_data_accessor : public base_data_accessor
{
	using value_t = typename T;

	using pointer_t = value_t*;
	using reference_t = value_t&;
	using const_reference_t = const value_t&;


	template<typename ...Args>
	raw_data_accessor<value_t>(Args&& ...args) :
		base_data_accessor(std::forward<Args>(args)...),
		__storage(__allocation(size))
	{;}

	raw_data_accessor<value_t>(const raw_data_accessor<value_t>&) = delete;
	raw_data_accessor<value_t>(raw_data_accessor<value_t>&&) = default;
	~raw_data_accessor<value_t>()
	{
		delete[] __storage;
	}

	reference_t get_value(size_t i, size_t j)
	{
		assert(i < size[size_e::y]);
		assert(j < size[size_e::x]);

		return __storage[size[size_e::x] * i + j];
	}

	const_reference_t get_value(size_t i, size_t j) const
	{
		const_cast<raw_data_accessor<value_t>*>(this)->get_value(i, j);
	}

	pointer_t p_data()
	{
		// return __storage.get();
		return __storage;
	}

private:

	//std::unique_ptr<value_t[]> __storage;

	value_t* __storage { nullptr };

	pointer_t __allocation(const dimension_sizes_t& size)
	{
		const auto sz = std::accumulate(size.begin(), size.end(), 1.0, std::multiplies<size_t>{});

		return new T[sz]{ static_cast<T>(0) };
	}
};

template<typename T>
struct hash_data_accessor : public base_data_accessor
{
	using value_t = typename T;

	using pointer_t = value_t*;
	using reference_t = value_t&;
	using const_reference_t = const value_t&;

	using key_t = dimension_sizes_t;

	struct hash_t
	{
		size_t operator()(const key_t& k) const
		{
			size_t seed = 0;

			for (const auto& value : k)
			{
				boost::hash_combine(seed, value);
			}

			return seed;
		}
	};

	template<typename ...Args>
	hash_data_accessor<value_t>(Args&& ...args) :
		base_data_accessor(std::forward<Args>(args)...)
	{;}

	hash_data_accessor<value_t>(const hash_data_accessor<value_t>&) = delete;
	hash_data_accessor<value_t>(hash_data_accessor<value_t>&&) = default;
	~hash_data_accessor<value_t>() = default;

	reference_t get_value(size_t i, size_t j)
	{
		assert(i < size[size_e::y]);
		assert(j < size[size_e::x]);

		auto [it, inserted] = __storage.try_emplace(key_t({ i, j }), static_cast<value_t>(0));

		return it->second;
	}

	const_reference_t get_value(size_t i, size_t j) const
	{
		const_cast<hash_data_accessor<value_t>*>(this)->get_value(i, j);
	}

private:

	std::unordered_map<key_t, value_t, hash_t> __storage;
};


template<class T, template<typename...> class data_accessor>
struct matrix
{
	using value_t = T;
	using size_e = base_data_accessor::size_e;
	using dimension_sizes_t = base_data_accessor::dimension_sizes_t;

	using data_accessor_t = data_accessor<value_t>;

	const dimension_sizes_t& size;
	data_accessor_t          data;

	matrix(const matrix<value_t, data_accessor>& other) = delete;

	matrix(matrix<value_t, data_accessor>&& other) = default;

	matrix(size_t size_x, size_t size_y) :
		data(size_x, size_y),
		size(data.size)
	{;}

	size_t length() const
	{
		return std::accumulate(size.begin(), size.end(), 1.0, std::multiplies<size_t>{});
	}

	const dimension_sizes_t& get_size() const
	{
		return size;
	}

	value_t& on(size_t x, size_t y)
	{
		return data.get_value(y, x);
	}

	const value_t& on(size_t x, size_t y) const
	{
		return data.get_value(y, x);
	}

	void print(std::ostream& fd, bool skip_zeros = true)
	{
		auto stored_flags = fd.flags();

		fd << std::fixed << std::setprecision(3);

		for (size_t i = 0; i < size[size_e::y]; ++i)
		{
			for (size_t j = 0; j < size[size_e::x]; ++j)
			{
				if (skip_zeros && std::abs(at(i, j)) < 1e-7)
				{
					fd << std::setw(8) << "        " << "  ";
				}
				else
				{
					fd << std::setw(8) << at(i, j) << "  ";
				}
			}

			fd << '\n';
		}

		fd.setf(stored_flags);
	}

private:

	value_t& at(size_t i, size_t j)
	{
		return data.get_value(i, j);
	}

	const value_t& at(size_t i, size_t j) const
	{
		return data.get_value(i, j);
	}
};


/****
 *	Input parameters for each independent run.
 *
 *	z and r are for the cylindrical coordinate system. [cm]
 *	a is for the angle alpha between the photon exiting
 *	direction and the surface normal. [radian]
 *
 *	The grid line separations in z, r, and alpha
 *	directions are dz, dr, and da respectively.  The numbers
 *	of grid lines in z, r, and alpha directions are
 *	nz, nr, and na respectively.
 *
 *	The member layerspecs will point to an array of
 *	structures which store parameters of each layer.
 *	This array has (number_layers + 2) elements. One
 *	element is for a layer.
 *	The layers 0 and (num_layers + 1) are for top ambient
 *	medium and the bottom ambient medium respectively.
 ****/
struct InputStruct
{
	char out_fname[STRLEN];	/* output file name. */
	char out_fformat;		/* output file format. */
							  /* 'A' for ASCII, */
							  /* 'B' for binary. */
	long num_photons; 		/* to be traced. */
	double Wth; 				/* play roulette if photon */
							  /* weight < Wth.*/

	double dz;				/* z grid separation.[cm] */
	double dr;				/* r grid separation.[cm] */
	double da;				/* alpha grid separation. */
							  /* [radian] */
	short nz;					/* array range 0..nz-1. */
	short nr;					/* array range 0..nr-1. */
	short na;					/* array range 0..na-1. */

	short	num_layers;			/* number of layers. */
	LayerStruct* layerspecs { nullptr };	/* layer parameters. */


	InputStruct() = default;

	InputStruct(const InputStruct&) = default;

	void free()
	{
		if (layerspecs)
		{
			::free(layerspecs);
		}
	}
};

template<class T, size_t dims>
struct buffer
{
	std::unique_ptr<sycl::buffer<T, dims>> data;

	template<typename ...Args>
	buffer<T, dims>(Args && ...args) :
		data{ new sycl::buffer<T, dims>(std::forward<Args>(args)...) }
	{;}

	void reset(sycl::buffer<T, dims>* pointer = nullptr)
	{
		data.reset(pointer);
	}

	operator sycl::buffer<T, dims>&()
	{
		return *data;
	}
};

struct ResultBlock
{
	using value_t = double;

	matrix<value_t, raw_data_accessor> matrix;
	buffer<value_t, 2U>	 buf_m; //sycl::buffer<value_t, 2U>		   buf_m;

	std::vector<value_t> r;
	buffer<value_t, 1U>	 buf_r; //sycl::buffer<value_t, 1U> buf_r;

	std::vector<value_t> a;
	buffer<value_t, 1U>  buf_a; // sycl::buffer<value_t, 1U> buf_a;

	value_t              value;
	buffer<value_t, 1U>  buf_v;// sycl::buffer<value_t, 1U> buf_v;

	ResultBlock(size_t rsz, size_t asz) :
		matrix(rsz, asz), buf_m(matrix.data.p_data(), sycl::range<2>(asz, rsz)),
		r(rsz),			  buf_r(r.data(), r.size()),
		a(asz),			  buf_a(a.data(), a.size()),
		value(0.0),		  buf_v(&value, 1)
	{;}

	ResultBlock(size_t rsz, size_t asz, size_t r_size, size_t a_size) :
		matrix(rsz, asz), buf_m(matrix.data.p_data(), sycl::range<2>(asz, rsz)),
		r(r_size),		  buf_r(r.data(), r.size()),
		a(a_size),		  buf_a(a.data(), a.size()),
		value(0.0),		  buf_v(&value, 1)
	{;}

	void reset()
	{
		buf_m.reset();
		buf_r.reset();
		buf_a.reset();
		buf_v.reset();
	}
};


/****
 *	Structures for scoring physical quantities.
 *	z and r represent z and r coordinates of the
 *	cylindrical coordinate system. [cm]
 *	a is the angle alpha between the photon exiting
 *	direction and the normal to the surfaces. [radian]
 *	See comments of the InputStruct.
 *	See manual for the physcial quantities.
 ****/


 //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\

struct OutStruct
{
	using value_t = double;

	using vector_t = std::vector<value_t>;
	using matrix_t = matrix<value_t, raw_data_accessor>;

public:

	value_t Rsp;

	ResultBlock Rd_rblock;

	matrix_t& Rd_ra;
	vector_t& Rd_r;
	vector_t& Rd_a;
	value_t&  Rd;

	ResultBlock A_rblock;

	matrix_t& A_rz;
	vector_t& A_z;
	vector_t& A_l;
	value_t&  A;

	ResultBlock Tt_rblock;

	matrix_t& Tt_ra;
	vector_t& Tt_r;
	vector_t& Tt_a;
	value_t&  Tt;

	OutStruct(const InputStruct& cfg) :
		Rsp(0.0),
		/* Allocate the arrays and the matrices. */
		Rd_rblock(cfg.nr, cfg.na), Rd_ra(Rd_rblock.matrix), Rd_r(Rd_rblock.r), Rd_a(Rd_rblock.a), Rd(Rd_rblock.value),
		A_rblock(cfg.nr, cfg.nz, cfg.nz, cfg.num_layers + 2), A_rz(A_rblock.matrix), A_z(A_rblock.r), A_l(A_rblock.a), A(A_rblock.value),
		Tt_rblock(cfg.nr, cfg.na), Tt_ra(Tt_rblock.matrix), Tt_r(Tt_rblock.r), Tt_a(Tt_rblock.a), Tt(Tt_rblock.value)
	{
		/***********************************************************
		 *	Allocate the arrays in OutStruct for one run, and
		 *	array elements are automatically initialized to zeros.
		 ****/

		/* remember to use nl+2 because of 2 for ambient. */

		if (cfg.nz <= 0 || cfg.nr <= 0 || cfg.na <= 0 || cfg.num_layers <= 0)
		{
			//throw std::logic_error("Wrong grid parameters.\n");

			//nrerror("Wrong grid parameters.\n");
		}
	}

	void reset()
	{
		Tt_rblock.reset();
		A_rblock.reset();
		Rd_rblock.reset();
	}
};

//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\

template<typename T>
struct access_block
{
	template<size_t dimensions>
	using buffer_t = typename sycl::buffer<T, dimensions>;

	constexpr static auto mode = sycl::access::mode::read_write;

	template<size_t dimensions>
	using rw_accessor = typename sycl::accessor<T, dimensions, mode>;

	rw_accessor<2U> matrix;
	rw_accessor<1U> r;
	rw_accessor<1U> a;
	rw_accessor<1U> v;

	access_block<T>(sycl::handler& cgh, buffer_t<2U>& buf_m, buffer_t<1U>& buf_r, buffer_t<1U>& buf_a, buffer_t<1U>& buf_v) :
		matrix(buf_m.template get_access<mode>(cgh)),
		r(buf_r.template get_access<mode>(cgh)),
		a(buf_a.template get_access<mode>(cgh)),
		v(buf_v.template get_access<mode>(cgh))
	{;}

	access_block<T>(const access_block<T>&) = default;
};

template<typename T>
struct access_output
{
	T Rsp;

	access_block<T> A;
	access_block<T> Rd;
	access_block<T> Tt;

	access_output<T>(sycl::handler& cgh, OutStruct& out) :
		Rsp(out.Rsp),

		A (cgh, out.A_rblock.buf_m,  out.A_rblock.buf_r,  out.A_rblock.buf_a,  out.A_rblock.buf_v),
		Rd(cgh, out.Rd_rblock.buf_m, out.Rd_rblock.buf_r, out.Rd_rblock.buf_a, out.Rd_rblock.buf_v),
		Tt(cgh, out.Tt_rblock.buf_m, out.Tt_rblock.buf_r, out.Tt_rblock.buf_a, out.Tt_rblock.buf_v)

	{;}
};