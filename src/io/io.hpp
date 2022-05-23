#pragma once

//#include <CL/sycl.hpp>

#include <cassert>

#include <memory>
#include <numeric>
#include <algorithm>
#include <functional>


#define STRLEN 256		/* String length. */

struct LayerStruct;

#define CPP_COMPILE


/***********************************************************
 *	Report error message to stderr, then exit the program
 *	with signal 1.
 ****/
void nrerror(const char error_text[]);


template<class T, template<typename...> class data_accessor>
struct matrix
{
	using data_accessor_t = data_accessor<T>;

	enum size_e : size_t
	{
		x = 0U,
		y,
		z
	};

	T* allocation(const std::vector<size_t>& size)
	{
		const auto unaligned_size = std::accumulate(size.begin(), size.end(), 1.0, std::multiplies<size_t>{});

		return new T[unaligned_size]{ static_cast<T>(0) };
	}

private:

	std::vector<size_t> __size;
	data_accessor_t     __data;

	T& at(size_t i, size_t j)
	{
		assert(i < __size[size_e::y]);
		assert(j < __size[size_e::x]);

		return __data[__size[size_e::x] * i + j];
	}

	const T& at(size_t i, size_t j) const
	{
		assert(i < __size[size_e::y]);
		assert(j < __size[size_e::x]);

		return __data[__size[size_e::x] * i + j];
	}

public:

	matrix(const matrix<T, data_accessor>& other) = delete;

	matrix(matrix<T, data_accessor>&& other) = delete;

	matrix(size_t size_x, size_t size_y, data_accessor_t&& accessor) :
		__size({ size_x, size_y }),
		__data(std::move(accessor))
	{
	
		std::cout << &__data[0] << std::endl;
	}

	size_t size(size_t index) const
	{
		if (index >= __size.size())
		{
			return 0;
		}

		return __size[index];
	}

	std::vector<size_t> size() const
	{
		return __size;
	}

	T& on(size_t x, size_t y)
	{
		assert(x < __size[size_e::x]);
		assert(y < __size[size_e::y]);

		return __data[__size[size_e::x] * y + x];
	}

	const T& on(size_t x, size_t y) const
	{
		assert(x < __size[size_e::x]);
		assert(y < __size[size_e::y]);

		return __data[__size[size_e::x] * y + x];
	}

	void print(std::ostream& fd, bool skip_zeros = true)
	{
		auto stored_flags = fd.flags();

		fd << std::fixed << std::setprecision(3);

		for (size_t i = 0; i < size_y; ++i)
		{
			for (size_t j = 0; j < size_x; ++j)
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
};


template<class T>
struct raw_index_accessor
{
	T* indexed_values;

	raw_index_accessor<T>(T* ptr) :
		indexed_values(reinterpret_cast<T*>(ptr))
	{;}

	raw_index_accessor<T>(raw_index_accessor<T>&&) = default;
	raw_index_accessor<T>(const raw_index_accessor<T>&) = default;

	T& operator[](size_t index)
	{
		return indexed_values[index];
	}

	const T& operator[](size_t index) const
	{
		return indexed_values[index];
	}
};
//
//template<class T, sycl::access::mode MODE>
//struct shared_index_accessor
//{
//	sycl::accessor<T, 1U, MODE, sycl::access::target::global_buffer> indexed_values;
//
//	shared_index_accessor<T, MODE>(sycl::buffer<T, 1U>& buffer, sycl::handler& cgh) :
//		indexed_values(buffer.get_access<MODE>(cgh))
//	{;}
//
//	shared_index_accessor<T, MODE>(shared_index_accessor<T, MODE>&&) = default;
//	shared_index_accessor<T, MODE>(const shared_index_accessor<T, MODE>&) = default;
//
//	T& operator[](size_t index)
//	{
//		return indexed_values[index];
//	}
//
//	const T& operator[](size_t index) const
//	{
//		return indexed_values[index];
//	}
//};



template<class T>
struct ResultBlockT
{
	std::unique_ptr<T> data;

	matrix<T, raw_index_accessor> matrix;

	std::vector<T> r;
	std::vector<T> a;
	std::vector<T> value; // single size

	ResultBlockT(size_t rsz, size_t asz) :
		data (new T[rsz * asz]{0.0}),
		matrix(rsz, asz, raw_index_accessor<T>{ data.get() }),
		r(rsz), a(asz), value(1U, 0.0)
	{;}

	ResultBlockT(size_t rsz, size_t asz, size_t r_size, size_t a_size) :
		data(new T[rsz * asz]{ 0.0 }),
		matrix(rsz, asz, raw_index_accessor<T>{ data.get() }),
		r(r_size), a(a_size), value(1U, 0.0)
	{;}

	//Sum2DRd, Sum2DTt
	void Sum2D()
	{
		size_t nr = matrix.size(0); // size x
		size_t na = matrix.size(1); // size y

		size_t ir, ia;

		double sum;

		for (ir = 0; ir < nr; ir++)
		{
			sum = 0.0;

			for (ia = 0; ia < na; ia++)
			{
				sum += matrix.on(ir, ia);
			}

			this->r[ir] = sum;
		}

		for (ia = 0; ia < na; ia++)
		{
			sum = 0.0;

			for (ir = 0; ir < nr; ir++)
			{
				sum += matrix.on(ir, ia);
			}

			this->a[ia] = sum;
		}

		this->value[0] = std::accumulate(r.begin(), r.end(), 0.0, std::plus<double>{});
	}
};


using ResultBlock = ResultBlockT<double>;


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
	char	 out_fname[STRLEN];	/* output file name. */
	char	 out_fformat;		/* output file format. */
							    /* 'A' for ASCII, */
							    /* 'B' for binary. */
	long	 num_photons; 		/* to be traced. */
	double Wth; 				/* play roulette if photon */
							    /* weight < Wth.*/

	double dz;				    /* z grid separation.[cm] */
	double dr;				    /* r grid separation.[cm] */
	double da;				    /* alpha grid separation. */
							    /* [radian] */
	short nz;					/* array range 0..nz-1. */
	short nr;					/* array range 0..nr-1. */
	short na;					/* array range 0..na-1. */

	short	num_layers;			/* number of layers. */
	LayerStruct* layerspecs;	/* layer parameters. */

	//std::unique_ptr<ResultBlock> A_rblock  { nullptr };
	//std::unique_ptr<ResultBlock> Rd_rblock { nullptr };
	//std::unique_ptr<ResultBlock> Tt_rblock { nullptr };

	//InputStruct()
	//{
		// Rd_rblock(std::make_unique<matrix_t>(cfg.nr, cfg.na), cfg.nr, cfg.na), 
		// A_rz(std::make_unique<matrix_t>(cfg.nr, cfg.nz)), A_z(cfg.nz), A_l(cfg.num_layers + 2), A(0.0), 
		// Tt_rblock(std::make_unique<matrix_t>(cfg.nr, cfg.na), cfg.nr, cfg.na), 
	//}

	void initialize()
	{
		//A_rblock.reset(new ResultBlock(nr, nz, nz, num_layers + 2));
		//Rd_rblock.reset(new ResultBlock(nr, na));
		//Tt_rblock.reset(new ResultBlock(nr, na));
	}

	void free()
	{
		if (layerspecs)
		{
			::free(layerspecs);
		}
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

//template <typename T, const size_t dimention>
//using read_accessor = sycl::accessor<T, dimention, sycl::access::mode::read, sycl::access::target::global_buffer>;
//
//template <typename T, const size_t dimention>
//using write_accessor = sycl::accessor<T, dimention, sycl::access::mode::write, sycl::access::target::global_buffer>;

 //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\

template<class T>
struct OutStructT
{
	using vector_t = std::vector<T>;
	using matrix_t = matrix<T, raw_index_accessor>;

	T Rsp;

	std::unique_ptr<ResultBlock>  A_rblock;
	std::unique_ptr<ResultBlock> Rd_rblock;
	std::unique_ptr<ResultBlock> Tt_rblock;

	vector_t& A; // single value
	matrix_t& A_rz;
	vector_t& A_z;
	vector_t& A_l;

	vector_t& Rd; // single value
	matrix_t& Rd_ra;
	vector_t& Rd_r;
	vector_t& Rd_a;

	vector_t& Tt; // single value
	matrix_t& Tt_ra;
	vector_t& Tt_r;
	vector_t& Tt_a;

	OutStructT(InputStruct& cfg) :
		A_rblock(new ResultBlock(cfg.nr, cfg.nz, cfg.nz, cfg.num_layers + 2)),
		Rd_rblock(new ResultBlock(cfg.nr, cfg.na)),
		Tt_rblock(new ResultBlock(cfg.nr, cfg.na)),

		Rsp (0.0),
		A     (A_rblock->value),
		A_rz  (A_rblock->matrix),
		A_z   (A_rblock->r),
		A_l   (A_rblock->a),
		///
		Rd    (Rd_rblock->value),
		Rd_ra (Rd_rblock->matrix),
		Rd_r  (Rd_rblock->r),
		Rd_a  (Rd_rblock->a),
		///
		Tt    (Tt_rblock->value),
		Tt_ra (Tt_rblock->matrix),
		Tt_r  (Tt_rblock->r),
		Tt_a  (Tt_rblock->a)
		///
	{
		/***********************************************************
			*	Allocate the arrays in OutStruct for one run, and      *
			*	array elements are automatically initialized to zeros. *
			***********************************************************/

		/* remember to use nl+2 because of 2 for ambient. */

		if (cfg.nz <= 0 || cfg.nr <= 0 || cfg.na <= 0 || cfg.num_layers <= 0)
		{
			throw std::logic_error("Wrong grid parameters.\n");

			//nrerror("Wrong grid parameters.\n");
		}
	}

	//OutStructT(InputStruct& cfg) :
	//	Rsp(0.0), 		
	//	
	//	/* Allocate the arrays and the matrices. */
	//	// Rd_rblock(std::make_unique<matrix_t>(cfg.nr, cfg.na), cfg.nr, cfg.na), 
	//	Rd_ra(cfg.Rd_rblock->matrix), Rd_r(cfg.Rd_rblock->r), Rd_a(cfg.Rd_rblock->a), Rd(cfg.Rd_rblock->value),

	//	// A_rz(std::make_unique<matrix_t>(cfg.nr, cfg.nz)), A_z(cfg.nz), A_l(cfg.num_layers + 2), A(0.0), 
	//	A_rz(cfg.A_rblock->matrix), A_z(cfg.A_rblock->r), A_l(cfg.A_rblock->a), A(cfg.A_rblock->value),

	//	// Tt_rblock(std::make_unique<matrix_t>(cfg.nr, cfg.na), cfg.nr, cfg.na), 
	//	Tt_ra(cfg.Tt_rblock->matrix), Tt_r(cfg.Tt_rblock->r), Tt_a(cfg.Tt_rblock->a), Tt(cfg.Tt_rblock->value)
	//{
	//	/***********************************************************
	//	 *	Allocate the arrays in OutStruct for one run, and      *
	//	 *	array elements are automatically initialized to zeros. *
	//	 ***********************************************************/

	//	/* remember to use nl+2 because of 2 for ambient. */

	//	if (cfg.nz <= 0 || cfg.nr <= 0 || cfg.na <= 0 || cfg.num_layers <= 0)
	//	{
	//		throw std::logic_error("Wrong grid parameters.\n");

	//		//nrerror("Wrong grid parameters.\n");
	//	}
	//}
};

using OutStruct = typename OutStructT<double>;

//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\