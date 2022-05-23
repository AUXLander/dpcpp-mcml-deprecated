#pragma once

#include <cassert>

//#include "../mcmlnr.hpp"
#include <memory>
#include <numeric>
#include <algorithm>

#include <boost/numeric/ublas/matrix.hpp>

#define STRLEN 256		/* String length. */

struct LayerStruct;

#define CPP_COMPILE


/***********************************************************
 *	Report error message to stderr, then exit the program
 *	with signal 1.
 ****/
void nrerror(const char error_text[]);


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

	double dz;				/* z grid separation.[cm] */
	double dr;				/* r grid separation.[cm] */
	double da;				/* alpha grid separation. */
							  /* [radian] */
	short nz;					/* array range 0..nz-1. */
	short nr;					/* array range 0..nr-1. */
	short na;					/* array range 0..na-1. */

	short	num_layers;			/* number of layers. */
	LayerStruct* layerspecs;	/* layer parameters. */

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


 //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\

struct OutStruct
{
	using value_type = double;

	using matrix = boost::numeric::ublas::matrix<value_type>;
	using vector = std::vector<value_type>;

	using unique_matrix_ptr = std::unique_ptr<matrix>;
	using unique_vector_ptr = std::unique_ptr<vector>;
	
private:
	struct matrix_adaptor
	{
		const unique_matrix_ptr mtx;

		struct matrix_column_adaptor
		{
			const unique_matrix_ptr& mtx;

			const size_t i;

			matrix_column_adaptor(const unique_matrix_ptr& mtx, const size_t i)
				: mtx(mtx), i(i)
			{;}

			value_type& operator[](const size_t& j)
			{
				return mtx->at_element(i, j);
			}

			const value_type& operator[](const size_t& j) const
			{
				return mtx->at_element(i, j);
			}
		};

		matrix_adaptor(unique_matrix_ptr&& mtx_)
			: mtx(std::move(mtx_))
		{
			for (auto i = 0; i < mtx->size1(); ++i)
			{
				for (auto j = 0; j < mtx->size2(); ++j)
				{
					mtx->at_element(i, j) = 0.0;
				}
			}
		}

		matrix_column_adaptor operator[](const size_t& i)
		{
			return matrix_column_adaptor(mtx, i);
		}

		matrix_column_adaptor operator[](const size_t& i) const
		{
			return matrix_column_adaptor(mtx, i);
		}
	};

public:

	struct ResultBlock
	{
		matrix_adaptor matrix;
		vector r;
		vector a;
		value_type value;
		
		ResultBlock(unique_matrix_ptr&& matrix, size_t rsz, size_t asz)
			: matrix(std::move(matrix)), r(rsz), a(asz), value(0.0)
		{;}

		//Sum2DRd, Sum2DTt
		void Sum2D()
		{
			size_t nr = matrix.mtx->size1();
			size_t na = matrix.mtx->size2();
			size_t ir, ia;

			double sum;

			for (ir = 0; ir < nr; ir++)
			{
				sum = 0.0;

				for (ia = 0; ia < na; ia++)
				{
					sum += this->matrix[ir][ia];
				}

				this->r[ir] = sum;
			}

			for (ia = 0; ia < na; ia++)
			{
				sum = 0.0;

				for (ir = 0; ir < nr; ir++)
				{
					sum += this->matrix[ir][ia];
				}

				this->a[ia] = sum;
			}

			this->value = std::accumulate(r.begin(), r.end(), 0.0, std::plus<double>{});
		}
	};

	value_type Rsp;

	ResultBlock Rd_rblock;

	matrix_adaptor& Rd_ra;
	vector& Rd_r;
	vector& Rd_a;
	value_type& Rd;

	matrix_adaptor A_rz;
	vector A_z;
	vector A_l;
	value_type A;

	ResultBlock Tt_rblock;

	matrix_adaptor& Tt_ra;
	vector& Tt_r;
	vector& Tt_a;
	value_type& Tt;

	OutStruct(InputStruct cfg)
		: Rsp(0.0), A(0.0),
		  /* Allocate the arrays and the matrices. */
		  Rd_rblock(std::make_unique<matrix>(cfg.nr, cfg.na), cfg.nr, cfg.na), Rd_ra(Rd_rblock.matrix), Rd_r(Rd_rblock.r), Rd_a(Rd_rblock.a), Rd(Rd_rblock.value),
		  A_rz(std::make_unique<matrix>(cfg.nr, cfg.nz)), A_z(cfg.nz), A_l(cfg.num_layers + 2),
		  Tt_rblock(std::make_unique<matrix>(cfg.nr, cfg.na), cfg.nr, cfg.na), Tt_ra(Tt_rblock.matrix), Tt_r(Tt_rblock.r), Tt_a(Tt_rblock.a), Tt(Tt_rblock.value)
	{
		/***********************************************************
		 *	Allocate the arrays in OutStruct for one run, and
		 *	array elements are automatically initialized to zeros.
		 ****/

		/* remember to use nl+2 because of 2 for ambient. */

		if (cfg.nz <= 0 || cfg.nr <= 0 || cfg.na <= 0 || cfg.num_layers <= 0)
		{
			throw std::logic_error("Wrong grid parameters.\n");

			//nrerror("Wrong grid parameters.\n");
		}
	}

	~OutStruct()
	{;}
};

//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\