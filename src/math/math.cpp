#include "math.h"

double SpinTheta(const double anisotropy)
{
	double cost;

	if (anisotropy == 0.0)
	{
		cost = 2 * RandomNum() - 1;
	}
	else
	{
		double temp = (1 - anisotropy * anisotropy) / (1 - anisotropy + 2 * anisotropy * RandomNum());

		cost = (1 + anisotropy * anisotropy - temp * temp) / (2 * anisotropy);

		if (cost < -1)
		{
			cost = -1;
		}
		else if (cost > 1)
		{
			cost = 1;
		}
	}

	return cost;
}

double RFresnel(double n1, double n2, double ca1, double* ca2_Ptr)
{
	double r;

	if (n1 == n2) /** matched boundary. **/
	{
		*ca2_Ptr = ca1;
		r = 0.0;
	}
	else if (ca1 > COSZERO) /** normal incident. **/
	{
		*ca2_Ptr = ca1;
		r = (n2 - n1) / (n2 + n1);
		r *= r;
	}
	else if (ca1 < COS90D) /** very slant. **/
	{
		*ca2_Ptr = 0.0;
		r = 1.0;
	}
	else /** general. **/
	{
		double sa1, sa2; /* sine of the incident and transmission angles. */
		double ca2;

		sa1 = std::sqrt(1 - ca1 * ca1);
		sa2 = n1 * sa1 / n2;
		if (sa2 >= 1.0)
		{
			/* double check for total internal reflection. */
			*ca2_Ptr = 0.0;
			r = 1.0;
		}
		else
		{
			double cap, cam; /* cosines of the sum ap or */
							 /* difference am of the two */
							 /* angles. ap = a1+a2 */
							 /* am = a1 - a2. */
			double sap, sam; /* sines. */

			*ca2_Ptr = ca2 = std::sqrt(1.0 - sa2 * sa2);

			cap = ca1 * ca2 - sa1 * sa2; /* c+ = cc - ss. */
			cam = ca1 * ca2 + sa1 * sa2; /* c- = cc + ss. */
			sap = sa1 * ca2 + ca1 * sa2; /* s+ = sc + cs. */
			sam = sa1 * ca2 - ca1 * sa2; /* s- = sc - cs. */

			r = 0.5 * sam * sam * (cam * cam + cap * cap) / (sap * sap * cam * cam);
			/* rearranged for speed. */
		}
	}

	return r;
}