#pragma once

/* testing program using fixed rnd seed. */
#define STANDARDTEST 1

float random_c(int* idum);

/***********************************************************
 *	Generate a random number between 0 and 1.  Take a
 *	number as seed the first time entering the function.
 *	The seed is limited to 1<<15.
 *	We found that when idum is too large, ran3 may return
 *	numbers beyond 0 and 1.
 ****/
double RandomNum(void);
