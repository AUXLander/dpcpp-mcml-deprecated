#pragma once
#include <cmath>
#include "random.h"

// #define PARTIALREFLECTION 0
  /* 1=split photon, 0=statistical reflection. */

constexpr double COSZERO = 1.0 - 1.0E-12;
/* cosine of about 1e-6 rad. */

constexpr double COS90D = 1.0E-6;
/* cosine of about 1.57 - 1e-6 rad. */



constexpr double PI = 3.1415926;
constexpr double WEIGHT = 1E-4;		/* Critical weight for roulette. */
constexpr double CHANCE = 0.1; /* Chance of roulette survival. */


template <typename T>
inline T sgn(T val)
{
	return (T(0) < val) - (val < T(0));
}


template<typename T, typename Y>
inline T setsign(T value, bool sign)
{
	// true is positive

	constexpr auto signbit_offs = sizeof(T) * 8U - 1U;
	constexpr auto signbit_mask = ~(static_cast<Y>(1U) << signbit_offs);

	Y& reinterpret = *reinterpret_cast<Y*>(&value);

	reinterpret = (reinterpret & signbit_mask) | (static_cast<Y>(!sign) << signbit_offs);

	return value;
}


/***********************************************************
 *	Choose (sample) a new theta angle for photon propagation
 *	according to the anisotropy.
 *
 *	If anisotropy g is 0, then
 *		cos(theta) = 2*rand-1.
 *	otherwise
 *		sample according to the Henyey-Greenstein function.
 *
 *	Returns the cosine of the polar deflection angle theta.
 ****/

double SpinTheta(const double anisotropy);

/***********************************************************
 *	Compute the Fresnel reflectance.
 *
 *	Make sure that the cosine of the incident angle a1
 *	is positive, and the case when the angle is greater
 *	than the critical angle is ruled out.
 *
 * 	Avoid trigonometric function operations as much as
 *	possible, because they are computation-intensive.
 ****/
double RFresnel(double n1,	/* incident refractive index.*/
	double n2,	/* transmit refractive index.*/
	double ca1,	/* cosine of the incident. angle. 0<a1<90 degrees*/
	double* ca2_Ptr)  /* pointer to the cosine of the transmission angle. a2 > 0. */
	;