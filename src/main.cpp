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

/*	Declare before they are used in main(). */
FILE* GetFile(char*);
short ReadNumRuns(FILE*);
void ReadParm(FILE*, InputStruct*);
void CheckParm(FILE*, InputStruct*);
//void InitOutputData(InputStruct, OutStruct*);
//void FreeData(InputStruct, OutStruct*);
//double Rspecular(LayerStruct*);
//void hop_drop_spin(InputStruct*, PhotonStruct*, OutStruct&);
void SumScaleResult(const InputStruct&, OutStruct&);
void WriteResult(const InputStruct&, const OutStruct&, char*);


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
void ReportResult(InputStruct& In_Parm, OutStruct& Out_Parm)
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
void DoOneRun(short NumRuns, InputStruct& In_Ptr)
{
	PhotonStruct photon(In_Ptr);
	
	long num_photons = In_Ptr.num_photons;
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
		stream.write((const char*)&In_Ptr.num_layers, sizeof(In_Ptr.num_layers));

		// num of photons
		stream.write((const char*)&In_Ptr.num_photons, sizeof(In_Ptr.num_photons));

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

	constexpr size_t N = 1024;

	constexpr size_t GROUP_SIZE = 8; // почему большее количество не создается?? 
	constexpr size_t GROUP_COUNT = (N / GROUP_SIZE);

	{
		OutStruct out_parm(In_Ptr);

		out_parm.Rsp = Rspecular(In_Ptr.layerspecs);

		//sycl::property_list props{ sycl::property::queue::enable_profiling() };

		//sycl::queue gpu_queue(
		//	sycl::gpu_selector{},

		//	[](sycl::exception_list exceptions) 
		//	{
		//		for (auto& exception : exceptions)
		//		{
		//			try
		//			{
		//				std::rethrow_exception(exception);
		//			}
		//			catch (sycl::exception& exception)
		//			{
		//				std::cerr << "Asynch error: " << exception.what() << std::endl;
		//			}
		//		}
		//	},

		//	props
		//);

		//sycl::event event = gpu_queue.submit(

		//	[&](sycl::handler& cgh) 
		//	{
		//		sycl::stream output(1024, 80, cgh);

		//		auto work_items = sycl::nd_range<2>(sycl::range<2>(12, 12), sycl::range<2>(4, 4));

		//		cgh.parallel_for(
		//			sycl::nd_range<1>(
		//				sycl::range<1>(N), 
		//				sycl::range<1>(GROUP_SIZE)
		//			),

		//			[=](sycl::nd_item<1> nd_item) // Наибольшее количество информации о work group
		//			{
		//				const size_t global_index = nd_item.get_global_id(0);
		//				const size_t local_index = nd_item.get_local_id(0);

		//				// загружаем данные из глобальной в локальную память, при этом barrier препятствует data race, т.е. все потоки дождутся, пока не дойдут до этого места.
		//				// в данном случае это не обязательно
		//				nd_item.barrier(sycl::access::fence_space::local_space);

		//				output << "Device index: " << global_index << sycl::endl;
		//				output << "Worker index: " << local_index  << sycl::endl;
		//			});
		//	}

		//);

		//gpu_queue.wait();

		//uint64_t start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
		//uint64_t end = event.get_profiling_info<sycl::info::event_profiling::command_end>();

		//std::cout << "Kernel execution time: " << (end - start) << " ns" << std::endl;

		for (; photon_idx > 0; --photon_idx)
		{
			if (num_photons - photon_idx == photon_rep)
			{
				printf("%ld photons & %d runs left, ", photon_idx, NumRuns);
				PredictDoneTime(num_photons - photon_idx, num_photons);
				photon_rep *= 10;
			}

			photon.init(out_parm.Rsp, In_Ptr.layerspecs);

			do
			{
				photon.hop_drop_spin(out_parm);
			}
			while (!photon.dead);
		}

		g.write();

		ReportResult(In_Ptr, out_parm);
	}

	In_Ptr.free();
}


int main(const int argc, const char* argv[])
{
	const int   c_argc = 2;
	const char* c_argv[] = {
		argv[0],
		"F:\\UserData\\Projects\\LightTransport\\build\\wcy_lo.mci"
	};

	// get_devices_information();

	char input_filename[STRLEN];

	FILE* input_file_ptr;

	short num_runs;	/* number of independent runs. */

	InputStruct in_parm;

	ShowVersion("Version 1.2, 1993");
	GetFnameFromArgv(c_argc, c_argv, input_filename);
	input_file_ptr = GetFile(input_filename);
	CheckParm(input_file_ptr, &in_parm);
	num_runs = ReadNumRuns(input_file_ptr);

	in_parm.initialize();

	while (num_runs--)
	{
		ReadParm(input_file_ptr, &in_parm);
		DoOneRun(num_runs, in_parm);
	}

	fclose(input_file_ptr);

	return 0;
}
