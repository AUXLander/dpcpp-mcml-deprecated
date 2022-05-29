#pragma once

#include "data/tracker.h"
#include "io/io.hpp"

#include "math/math.h"

struct LayerStruct
{
	double z0, z1;	/* z coordinates of a layer. [cm] */
	double n;			/* refractive index of a layer. */
	double mua;	    /* absorption coefficient. [1/cm] */
	double mus;	    /* scattering coefficient. [1/cm] */
	double anisotropy;		    /* anisotropy. */

	double cos_crit0, cos_crit1;

	inline bool is_glass() const
	{
		return mua == 0.0 && mus == 0.0;
	}
};

double Rspecular(LayerStruct* Layerspecs_Ptr);


struct PhotonStruct
{
	double x{0}, y{0}, z{0};
	double ux{0}, uy{0}, uz{0};
	double w{0};

	bool dead {false};

	size_t layer{0};

	double step_size{0};
	double sleft{0};

	const InputStruct& input;

	OutStruct& output;

	tracker::local_thread_storage track;

	PhotonStruct(const InputStruct&, OutStruct&);

	~PhotonStruct()
	{
		auto& g = tracker::instance();

		g.track(std::move(track));
	}

	void init(const double Rspecular, LayerStruct* Layerspecs_Ptr);

	void spin(const double anisotropy);

	void hop();

	void step_size_in_glass();
	bool hit_boundary();
	void roulette();
	void record_r(double Refl, OutStruct& Out_Ptr);
	void record_t(double Refl, OutStruct& Out_Ptr);
	void drop(OutStruct& Out_Ptr);

	void cross_up_or_not(OutStruct& Out_Ptr);
	void cross_down_or_not(OutStruct& Out_Ptr);

	void cross_or_not(OutStruct& Out_Ptr);

	void hop_in_glass(OutStruct& Out_Ptr);
	void hop_drop_spin(OutStruct& Out_Ptr);

	LayerStruct& get_current_layer()
	{
		assert(input.layerspecs);

		return input.layerspecs[layer];
	}
};