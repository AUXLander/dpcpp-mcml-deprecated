/***********************************************************
 *  Copyright Univ. of Texas M.D. Anderson Cancer Center
 *  1992.
 *
 *	Some routines modified from Numerical Recipes in C,
 *	including error report, array or matrix declaration
 *	and releasing.
 ****/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cmath>

#include "mcml.hpp"

double Rspecular(LayerStruct* Layerspecs_Ptr)
{
	double r1, r2;
	/* direct reflections from the 1st and 2nd layers. */
	double temp;

	temp = (Layerspecs_Ptr[0].n - Layerspecs_Ptr[1].n) / (Layerspecs_Ptr[0].n + Layerspecs_Ptr[1].n);
	r1 = temp * temp;

	if ((Layerspecs_Ptr[1].mua == 0.0) && (Layerspecs_Ptr[1].mus == 0.0))
	{
		/* glass layer. */
		temp = (Layerspecs_Ptr[1].n - Layerspecs_Ptr[2].n) / (Layerspecs_Ptr[1].n + Layerspecs_Ptr[2].n);

		r2 = temp * temp;

		r1 = r1 + (1 - r1) * (1 - r1) * r2 / (1 - r1 * r2);
	}

	return (r1);
}


PhotonStruct::PhotonStruct(InputStruct& cfg) :
	In_Ptr{ cfg }, input(cfg), track()
{;}

/***********************************************************
 *	Initialize a photon packet.
 ****/
void PhotonStruct::init(const double Rspecular, LayerStruct* Layerspecs_Ptr)
{
	w = 1.0 - Rspecular;
	dead = 0;
	layer = 1; // LAYER CHANGE
	step_size = 0;
	sleft = 0;

	x = 0.0; // COORD CHANGE
	y = 0.0;
	z = 0.0;
	
	ux = 0.0;
	uy = 0.0;
	uz = 1.0;

	/* glass layer. */
	if ((Layerspecs_Ptr[1].mua == 0.0) && (Layerspecs_Ptr[1].mus == 0.0))
	{
		layer = 2; // LAYER CHANGE
		z = Layerspecs_Ptr[2].z0;
	}

	track.track(x, y, z, layer);
}

void PhotonStruct::spin(const double anisotropy)
{
	const double ux = this->ux;
	const double uy = this->uy;
	const double uz = this->uz;

	const double cost = SpinTheta(anisotropy); /* cosine and sine of the polar deflection angle theta. */
	const double sint = std::sqrt(1.0 - cost * cost); /* sqrt() is faster than sin(). */

	const double psi = 2.0 * PI * RandomNum(); /* spin psi 0-2pi. */

	const double cosp = std::cos(psi); /* cosine and sine of the azimuthal angle psi. */
	const double sinp = setsign<double, uint64_t>(std::sqrt(1.0 - cosp * cosp), psi < PI);

	if (fabs(uz) > COSZERO) /* normal incident. */
	{
		this->ux = sint * cosp;
		this->uy = sint * sinp;
		this->uz = cost * sgn(uz); /* SIGN() is faster than division. */
	}
	else /* regular incident. */
	{
		const double temp = std::sqrt(1.0 - uz * uz);

		this->ux = sint * (ux * uz * cosp - uy * sinp) / temp + ux * cost;
		this->uy = sint * (uy * uz * cosp + ux * sinp) / temp + uy * cost;
		this->uz = -sint * cosp * temp + uz * cost;
	}
}

void PhotonStruct::hop()
{
	// COORD CHANGE

	x += step_size * ux;
	y += step_size * uy;
	z += step_size * uz;

	track.track(x, y, z, layer);
}

void PhotonStruct::step_size_in_glass()
{
	const auto& olayer = get_current_layer();

	/* Stepsize to the boundary. */
	if (uz > 0.0)
	{
		step_size = (olayer.z1 - z) / uz;
	}
	else if (uz < 0.0)
	{
		step_size = (olayer.z0 - z) / uz;
	}
	else
	{
		step_size = 0.0;
	}
}

bool PhotonStruct::hit_boundary()
{
	const auto& olayer = get_current_layer();

	double dl_b;

	if (uz > 0.0)
	{
		dl_b = (olayer.z1 - z) / uz;
	}
	else if (uz < 0.0)
	{
		dl_b = (olayer.z0 - z) / uz;
	}

	const bool hit = (uz != 0.0 && step_size > dl_b);

	if (hit)
	{
		sleft = (step_size - dl_b) * (olayer.mua + olayer.mus);
		step_size = dl_b;
	}

	return hit;
}

void PhotonStruct::roulette()
{
	if (w != 0.0 && RandomNum() < CHANCE)
	{
		w /= CHANCE;
	}
	else
	{
		dead = true;
	}
}

void PhotonStruct::drop(OutStruct& output)
{
	double dwa;		/* absorbed weight.*/
	size_t iz, ir;	/* index to z & r. */
	double mua, mus;

	/* compute array indices. */
	iz = static_cast<size_t>(z / input.dz);
	iz = std::min<size_t>(iz, input.nz - 1);

	ir = static_cast<size_t>(std::sqrt(x * x + y * y) / input.dr);
	ir = std::min<size_t>(ir, input.nr - 1);
	

	const auto& olayer = get_current_layer();

	/* update photon weight. */
	mua = olayer.mua;
	mus = olayer.mus;

	dwa = w * mua / (mua + mus);
	
	w -= dwa;

	/* assign dwa to the absorption array element. */
	output.A_rz[ir][iz] += dwa;
}


void PhotonStruct::record_t(double reflectance, OutStruct& output)
{
	size_t ir, ia;	/* index to r & angle. */

	ir = static_cast<size_t>(std::sqrt(x * x + y * y) / input.dr);
	ir = std::min<size_t>(ir, input.nr - 1);

	ia = static_cast<size_t>(std::acos(uz) / input.da);
	ia = std::min<size_t>(ia, input.na - 1);

	output.Tt_ra[ir][ia] += w * (1.0 - reflectance);

	w *= reflectance;
}

void PhotonStruct::record_r(double reflectance, OutStruct& output)
{
	size_t ir, ia;

	ir = static_cast<size_t>(std::sqrt(x * x + y * y) / input.dr);
	ir = std::min<size_t>(ir, input.nr - 1);

	ia = static_cast<size_t>(std::acos(-uz) / input.da);
	ia = std::min<size_t>(ia, input.na - 1);

	output.Rd_ra[ir][ia] += w * (1.0 - reflectance);

	w *= reflectance;
}

void PhotonStruct::cross_up_or_not(OutStruct& Out_Ptr)
{
	// this->uz; /* z directional cosine. */
	double uz1 = 0.0;					/* cosines of transmission alpha. always positive */
	double r = 0.0;				/* reflectance */
	double ni = In_Ptr.layerspecs[layer].n;
	double nt = In_Ptr.layerspecs[layer - 1].n;

	/* Get r. */
	if (-uz <= In_Ptr.layerspecs[layer].cos_crit0)
	{
		r = 1.0; /* total internal reflection. */
	}
	else
	{
		r = RFresnel(ni, nt, -uz, &uz1);
	}

	if (RandomNum() > r) /* transmitted to layer-1. */
	{
		if (layer == 1)
		{
			uz = -uz1;
			record_r(0.0, Out_Ptr);
			dead = true;
		}
		else
		{
			layer--; // LAYER CHANGE

			ux *= ni / nt;
			uy *= ni / nt;
			uz = -uz1;

			track.track(x, y, z, layer);
		}
	}
	else /* reflected. */
	{
		this->uz = -uz;
	}
}

void PhotonStruct::cross_down_or_not(OutStruct& output)
{
	//this->uz; /* z directional cosine. */
	double uz1 = 0.0;	/* cosines of transmission alpha. */
	double r = 0.0;	/* reflectance */
	double ni = In_Ptr.layerspecs[layer].n;
	double nt = In_Ptr.layerspecs[layer + 1].n;

	/* Get r. */
	if (uz <= In_Ptr.layerspecs[layer].cos_crit1)
	{
		r = 1.0; /* total internal reflection. */
	}
	else
	{
		r = RFresnel(ni, nt, uz, &uz1);
	}

	if (RandomNum() > r) /* transmitted to layer+1. */
	{
		if (layer == In_Ptr.num_layers)
		{
			uz = uz1;
			record_t(0.0, output);
			dead = true;
		}
		else
		{
			layer++; // LAYER CHANGE

			ux *= ni / nt;
			uy *= ni / nt;
			uz = uz1;

			track.track(x, y, z, layer);
		}
	}
	else /* reflected. */
	{
		uz = -uz;
	}
}

void PhotonStruct::cross_or_not(OutStruct& Out_Ptr)
{
	if (this->uz < 0.0)
	{
		cross_up_or_not(Out_Ptr);
	}
	else
	{
		cross_down_or_not(Out_Ptr);
	}
}

void PhotonStruct::hop_in_glass(OutStruct& Out_Ptr)
{
	if (uz == 0.0)
	{
		/* horizontal photon in glass is killed. */
		dead = true;
	}
	else
	{
		step_size_in_glass();
		hop(); // Move the photon s away in the current layer of medium. 
		cross_or_not(Out_Ptr);
	}
}

void PhotonStruct::hop_drop_spin(OutStruct& output)
{
	const auto& olayer = get_current_layer();

	if (olayer.is_glass())
	{
		hop_in_glass(output);
	}
	else
	{
		const double mua = olayer.mua;
		const double mus = olayer.mus;

		if (sleft == 0.0)
		{
			double rnd;

			do
			{
				rnd = RandomNum();
			}
			while (rnd <= 0.0); 

			step_size = - std::log(rnd) / (mua + mus);
		}
		else
		{
			step_size = sleft / (mua + mus);
			sleft = 0.0;
		}

		const auto hit = hit_boundary();

		hop();

		if (hit)
		{
			cross_or_not(output);
		}
		else
		{
			drop(output);
			spin(olayer.anisotropy);
		}
	}

	if (w < In_Ptr.Wth && !dead)
	{
		roulette();
	}
}