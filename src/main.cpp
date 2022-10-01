 /****
  *	THINKCPROFILER is defined to generate profiler calls in
  *	Think C. If 1, remember to turn on "Generate profiler
  *	calls" in the options menu.
  ****/
#define THINKCPROFILER 0

  /* GNU cc does not support difftime() and CLOCKS_PER_SEC.*/
#define GNUCC 0

#if THINKCPROFILER
#include <profile.h>
#include <console.h>
#endif

#include "mcml.hpp"
#include "dpcpp/utils.hpp"

#include <oneapi/dpl/random>

#include <CL/sycl/detail/cl.h>
#include <CL/sycl/detail/pi.h>

double RFresnel(double n1, double n2, double ca1, double& ca2_Ptr)
{
	double r;

	if (n1 == n2)
	{
		ca2_Ptr = ca1;
		r = 0.0;
	}
	else if (ca1 > COSZERO)
	{
		ca2_Ptr = ca1;
		r = (n2 - n1) / (n2 + n1);
		r *= r;
	}
	else if (ca1 < COS90D)
	{
		ca2_Ptr = 0.0;
		r = 1.0;
	}
	else
	{
		double sa1, sa2;
		double ca2;

		sa1 = std::sqrt(1.0 - ca1 * ca1);
		sa2 = n1 * sa1 / n2;

		if (sa2 >= 1.0)
		{
			ca2_Ptr = 0.0;
			r = 1.0;
		}
		else
		{
			double cap, cam;
			double sap, sam;

			ca2_Ptr = ca2 = std::sqrt(1.0 - sa2 * sa2);

			cap = ca1 * ca2 - sa1 * sa2; /* c+ = cc - ss. */
			cam = ca1 * ca2 + sa1 * sa2; /* c- = cc + ss. */
			sap = sa1 * ca2 + ca1 * sa2; /* s+ = sc + cs. */
			sam = sa1 * ca2 - ca1 * sa2; /* s- = sc - cs. */

			r = 0.5 * sam * sam * (cam * cam + cap * cap) / (sap * sap * cam * cam);
		}
	}

	return r;
}

template<sycl::access::mode MODE, sycl::access::target TARGET>
struct PhotonStruct
{
	using constant_accessor_t = sycl::accessor<LayerStruct, 1U, sycl::access::mode::read, sycl::access::target::constant_buffer>;

	using output_accessor_t = access_output<double, TARGET>;

	double x{ 0 }, y{ 0 }, z{ 0 };
	double ux{ 0 }, uy{ 0 }, uz{ 0 };
	double w{ 0 };

	bool dead{ false };

	size_t layer{ 0 };

	double step_size{ 0 };
	double sleft{ 0 };

	oneapi::dpl::minstd_rand engine;
	oneapi::dpl::uniform_real_distribution<double> distr;

	const InputStruct& input;

	const constant_accessor_t& layerspecs;

	const output_accessor_t& output;

	PhotonStruct<MODE, TARGET>(const InputStruct& input, const constant_accessor_t& l, const output_accessor_t& output) :
		engine{ 100, 0 }, distr{ 0.0, 1.0 }, input{ input }, layerspecs{ l }, output{ output }
	{;}

	~PhotonStruct<MODE, TARGET>() = default;

	void init(const double Rspecular)
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

		if ((layerspecs[1].mua == 0.0) && (layerspecs[1].mus == 0.0))
		{
			layer = 2; // LAYER CHANGE
			z = layerspecs[2].z0;
		}

		// track.track(x, y, z, layer);
	}

	void spin(const double anisotropy)
	{
		const double ux = this->ux;
		const double uy = this->uy;
		const double uz = this->uz;

		const double cost = SpinTheta(anisotropy);
		const double sint = std::sqrt(1.0 - cost * cost);

		const double psi = 2.0 * PI * get_random();

		const double cosp = std::cos(psi);
		const double sinp = setsign<double, uint64_t>(std::sqrt(1.0 - cosp * cosp), psi < PI);

		if (fabs(uz) > COSZERO)
		{
			this->ux = sint * cosp;
			this->uy = sint * sinp;
			this->uz = cost * sgn(uz);
		}
		else
		{
			const double temp = std::sqrt(1.0 - uz * uz);

			this->ux = sint * (ux * uz * cosp - uy * sinp) / temp + ux * cost;
			this->uy = sint * (uy * uz * cosp + ux * sinp) / temp + uy * cost;
			this->uz = -sint * cosp * temp + uz * cost;
		}
	}


	void hop()
	{
		// COORD CHANGE

		x += step_size * ux;
		y += step_size * uy;
		z += step_size * uz;

		// track.track(x, y, z, layer);
	}

	void step_size_in_glass()
	{
		const auto& olayer = get_current_layer();

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

	bool hit_boundary()
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

	void roulette()
	{
		if (w != 0.0 && get_random() < CHANCE)
		{
			w /= CHANCE;
		}
		else
		{
			dead = true;
		}
	}

	void record_r(double reflectance)
	{
		size_t ir, ia;

		ir = static_cast<size_t>(std::sqrt(x * x + y * y) / input.dr);
		ir = std::min<size_t>(ir, input.nr - 1);

		ia = static_cast<size_t>(std::acos(-uz) / input.da);
		ia = std::min<size_t>(ia, input.na - 1);

		output.Rd.matrix[ir][ia] += w * (1.0 - reflectance);

		w *= reflectance;
	}

	void record_t(double reflectance)
	{
		size_t ir, ia;

		ir = static_cast<size_t>(std::sqrt(x * x + y * y) / input.dr);
		ir = std::min<size_t>(ir, input.nr - 1);

		ia = static_cast<size_t>(std::acos(uz) / input.da);
		ia = std::min<size_t>(ia, input.na - 1);

		output.Tt.matrix[ia][ir] += w * (1.0 - reflectance);

		w *= reflectance;
	}

	void drop()
	{
		double dwa;
		size_t iz, ir;
		double mua, mus;

		iz = static_cast<size_t>(z / input.dz);
		iz = std::min<size_t>(iz, input.nz - 1);

		ir = static_cast<size_t>(std::sqrt(x * x + y * y) / input.dr);
		ir = std::min<size_t>(ir, input.nr - 1);

		const auto& olayer = get_current_layer();

		mua = olayer.mua;
		mus = olayer.mus;

		dwa = w * mua / (mua + mus);

		w -= dwa;

		output.A.matrix[iz][ir] += dwa;
	}

	void cross_up_or_not()
	{
		double uz1 = 0.0;
		double r = 0.0;
		double ni = layerspecs[layer].n;
		double nt = layerspecs[layer - 1].n;

		if (-uz <= layerspecs[layer].cos_crit0)
		{
			r = 1.0;
		}
		else
		{
			r = RFresnel(ni, nt, -uz, uz1);
		}

		if (get_random() > r)
		{
			if (layer == 1)
			{
				uz = -uz1;
				record_r(0.0);
				dead = true;
			}
			else
			{
				layer--; // LAYER CHANGE

				ux *= ni / nt;
				uy *= ni / nt;
				uz = -uz1;

				// track.track(x, y, z, layer);
			}
		}
		else
		{
			uz = -uz;
		}
	}

	void cross_down_or_not()
	{
		double uz1 = 0.0;
		double r = 0.0;
		double ni = layerspecs[layer].n;
		double nt = layerspecs[layer + 1].n;

		if (uz <= layerspecs[layer].cos_crit1)
		{
			r = 1.0;
		}
		else
		{
			r = RFresnel(ni, nt, uz, uz1);
		}

		if (get_random() > r)
		{
			if (layer == input.num_layers)
			{
				uz = uz1;
				record_t(0.0);
				dead = true;
			}
			else
			{
				layer++; // LAYER CHANGE

				ux *= ni / nt;
				uy *= ni / nt;
				uz = uz1;

				// track.track(x, y, z, layer);
			}
		}
		else
		{
			uz = -uz;
		}
	}

	void cross_or_not()
	{
		if (uz < 0.0)
		{
			cross_up_or_not();
		}
		else
		{
			cross_down_or_not();
		}
	}

	double SpinTheta(const double anisotropy)
	{
		double cost;

		if (anisotropy == 0.0)
		{
			cost = 2.0 * get_random() - 1.0;
		}
		else
		{
			const double temp = (1.0 - anisotropy * anisotropy) / (1.0 - anisotropy + 2.0 * anisotropy * get_random());

			cost = (1 + anisotropy * anisotropy - temp * temp) / (2.0 * anisotropy);

			if (cost < -1.0)
			{
				cost = -1.0;
			}
			else if (cost > 1.0)
			{
				cost = 1;
			}
		}

		return cost;
	}

	void hop_in_glass()
	{
		if (uz == 0.0)
		{
			dead = true;
		}
		else
		{
			step_size_in_glass();
			hop();
			cross_or_not();
		}
	}

	void hop_drop_spin()
	{
		const auto& olayer = get_current_layer();

		if (olayer.is_glass())
		{
			hop_in_glass();
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
					rnd = get_random();
				}
				while (rnd <= 0.0);

				step_size = -std::log(rnd) / (mua + mus);
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
				cross_or_not();
			}
			else
			{
				drop();
				spin(olayer.anisotropy);
			}
		}

		if (w < input.Wth && !dead)
		{
			roulette();
		}
	}

	inline const LayerStruct& get_current_layer() const
	{
		return layerspecs[layer];
	}

	inline double get_random()
	{
		return distr(engine);
	}
};

double Rspecular(LayerStruct* Layerspecs_Ptr)
{
	double r1;
	double r2;
	double temp;

	temp = (Layerspecs_Ptr[0].n - Layerspecs_Ptr[1].n) / (Layerspecs_Ptr[0].n + Layerspecs_Ptr[1].n);
	r1 = temp * temp;

	if ((Layerspecs_Ptr[1].mua == 0.0) && (Layerspecs_Ptr[1].mus == 0.0))
	{
		temp = (Layerspecs_Ptr[1].n - Layerspecs_Ptr[2].n) / (Layerspecs_Ptr[1].n + Layerspecs_Ptr[2].n);

		r2 = temp * temp;

		r1 = r1 + (1 - r1) * (1 - r1) * r2 / (1 - r1 * r2);
	}

	return r1;
}

/*	Declare before they are used in main(). */
FILE* GetFile(char*);
short ReadNumRuns(FILE*);
void ReadParm(FILE*, InputStruct*);
void CheckParm(FILE*, InputStruct*);
void InitOutputData(InputStruct, OutStruct*);
void FreeData(InputStruct, OutStruct*);
//double Rspecular(LayerStruct*);

//void hop_drop_spin(InputStruct*, PhotonStruct<sycl::access::mode MODE, sycl::access::target TARGET>*, OutStruct&);
void SumScaleResult(InputStruct, OutStruct&);
void WriteResult(InputStruct, const OutStruct&, char*);


/***********************************************************
 *	If F = 0, reset the clock and return 0.
 *
 *	If F = 1, pass the user time to Msg and print Msg on
 *	screen, return the real time since F=0.
 *
 *	If F = 2, same as F=1 except no printing.
 *
 *	Note that clock() and time() return user time and real
 *	time respectively.
 *	User time is whatever the system allocates to the
 *	running of the program;
 *	real time is wall-clock time.  In a time-shared system,
 *	they need not be the same.
 *
 *	clock() only hold 16 bit integer, which is about 32768
 *	clock ticks.
 ****/
time_t PunchTime(char F, char* Msg)
{
#if GNUCC
	return(0);
#else
	static clock_t ut0;	/* user time reference. */
	static time_t  rt0;	/* real time reference. */
	double secs;
	char s[STRLEN];

	if (F == 0) {
		ut0 = clock();
		rt0 = time(NULL);
		return(0);
	}
	else if (F == 1) {
		secs = (clock() - ut0) / (double)CLOCKS_PER_SEC;
		if (secs < 0) secs = 0;	/* clock() can overflow. */

		secs = 3;// hardcoded to prevent SHA256 errors

		sprintf(s, "User time: %8.0lf sec = %8.2lf hr.  %s\n",
			secs, secs / 3600.0, Msg);
		puts(s);
		strcpy(Msg, s);
		return(difftime(time(NULL), rt0));
	}
	else if (F == 2) return(difftime(time(NULL), rt0));
	else return(0);
#endif
}

/***********************************************************
 *	Print the current time and the estimated finishing time.
 *
 *	P1 is the number of computed photon packets.
 *	Pt is the total number of photon packets.
 ****/
void PredictDoneTime(long P1, long Pt)
{
	time_t now, done_time;
	struct tm* date;
	char s[80];

	now = time(NULL);
	date = localtime(&now);
	strftime(s, 80, "%H:%M %x", date);
	printf("Now %s, ", s);

	char buff[1] = "";

	done_time = now + (time_t)(PunchTime(2, buff) / (double)P1 * (Pt - P1));

	date = localtime(&done_time);
	strftime(s, 80, "%H:%M %x", date);
	printf("End %s\n", s);
}

/***********************************************************
 *	Report time and write results.
 ****/
void ReportResult(InputStruct In_Parm, OutStruct& Out_Parm)
{
	char time_report[STRLEN];

	strcpy(time_report, " Simulation time of this run.");
	PunchTime(1, time_report);

	SumScaleResult(In_Parm, Out_Parm);
	WriteResult(In_Parm, Out_Parm, time_report);
}

/***********************************************************
 *	Get the file name of the input data file from the
 *	argument to the command line.
 ****/
void GetFnameFromArgv(int argc, const char* argv[], char* input_filename)
{
	if (argc >= 2) {			/* filename in command line */
		strcpy(input_filename, argv[1]);
	}
	else
		input_filename[0] = '\0';
}

/***********************************************************
 *	Execute Monte Carlo simulation for one independent run.
 ****/
void DoOneRun(short NumRuns, InputStruct& input)
{
	long num_photons = input.num_photons;
	long photon_rep = 10;
	long photon_idx = num_photons; // photon index

	char buff[1] = "";

	PunchTime(0, buff);

	// singleton 
	auto& g = tracker::instance();
	g.set_file("BINARY_DATA.BIN");

	double reserved = 0.0;

	g.set_headers([&](std::fstream& stream) {

		// num of layers
		stream.write((const char*)&input.num_layers, sizeof(input.num_layers));

		// num of photons
		stream.write((const char*)&input.num_photons, sizeof(input.num_photons));

		// dpi's 
		stream.write((const char*)&reserved, sizeof(size_t));
		stream.write((const char*)&reserved, sizeof(size_t));
		stream.write((const char*)&reserved, sizeof(size_t));

		// min's 
		stream.write((const char*)&reserved, sizeof(reserved));
		stream.write((const char*)&reserved, sizeof(reserved));
		stream.write((const char*)&reserved, sizeof(reserved));

		// max's 
		stream.write((const char*)&reserved, sizeof(reserved));
		stream.write((const char*)&reserved, sizeof(reserved));
		stream.write((const char*)&reserved, sizeof(reserved));
	});

	OutStruct global_output(input);

	{
		global_output.Rsp = Rspecular(input.layerspecs);

		sycl::buffer<LayerStruct, 1> l_buf(input.layerspecs, (size_t)(input.num_layers + 2));

		///////////////

		sycl::property_list props{ sycl::property::queue::enable_profiling() };

		sycl::queue queue(sycl::gpu_selector{}, 
			[](sycl::exception_list exceptions)
			{
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
			}, 
			props
		);

		constexpr size_t work_item_size  = 1U;
		constexpr size_t work_group_size = 2048; // 512U;

		try
		{
			constexpr auto read = sycl::access::mode::read;
			constexpr auto read_write = sycl::access::mode::read_write;

			constexpr auto target_global = sycl::access::target::global_buffer;
			constexpr auto target_local = sycl::access::target::local;
			constexpr auto target_constant = sycl::access::target::constant_buffer;


			const auto kernel = [&](sycl::handler& cgh) 
			{
				sycl::stream cout(1024, 80, cgh);

				access_output<double> output(cgh, global_output);

				const auto layerspecs = l_buf.get_access<read, target_constant>(cgh);

				const auto photons_per_group = num_photons / (work_item_size * work_group_size);

				// access_output<double, target_local> loutput(cgh, global_output, false);

				auto global_range = sycl::range<1>(work_group_size);
				auto local_range  = sycl::range<1>(work_item_size);

				cgh.parallel_for(sycl::nd_range<1>(global_range, local_range),
					[=](sycl::nd_item<1> item)
					{
						PhotonStruct<read_write, target_global> photon(input, layerspecs, output);

						for (size_t local_index = 0; local_index < photons_per_group; ++local_index)
						{
							photon.init(output.Rsp);

							do
							{
								photon.hop_drop_spin();
							}
							while (!photon.dead);
						}

						// все потоки дожидаются
						// item.barrier(sycl::access::fence_space::local_space);
					}
				);
			};

			{
				sycl::event event = queue.submit(kernel);

				uint64_t start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
				uint64_t end = event.get_profiling_info<sycl::info::event_profiling::command_end>();

				std::cout << "Kernel execution time: " << (double)(end - start) * 1e-6 << " ms" << std::endl;
			}

			queue.wait_and_throw();
		}
		catch (sycl::exception& exception)
		{
			std::cerr << "Kernel error: " << exception.what() << std::endl;
		}



		// g.write();

		global_output.reset();
	}

	ReportResult(input, global_output);

	input.free();
}


int main(const int argc, const char* argv[])
{
	const int   c_argc = 2;
	const char* c_argv[] = {
		argv[0],
		"F:\\UserData\\Projects\\LightTransport\\build\\wcy_lo.mci"
	};

	get_devices_information();

	char input_filename[STRLEN];

	FILE* input_file_ptr;

	short num_runs;	/* number of independent runs. */

	InputStruct in_parm;

	ShowVersion("Version 1.2, 1993");
	GetFnameFromArgv(c_argc, c_argv, input_filename);
	input_file_ptr = GetFile(input_filename);
	CheckParm(input_file_ptr, &in_parm);
	num_runs = ReadNumRuns(input_file_ptr);

	while (num_runs--)
	{
		ReadParm(input_file_ptr, &in_parm);
		DoOneRun(num_runs, in_parm);
	}

	fclose(input_file_ptr);

	return 0;
}
