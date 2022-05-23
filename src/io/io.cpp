#include "io.hpp"

void nrerror(const char error_text[])
{
	fprintf(stderr, "%s\n", error_text);
	fprintf(stderr, "...now exiting to system...\n");
	exit(1);
}