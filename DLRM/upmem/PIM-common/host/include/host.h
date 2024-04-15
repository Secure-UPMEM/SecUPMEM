#ifndef __HOST_H
#define __HOST_H

#include <sys/time.h>

/* Everything in this file can be used on the host */

#define TIME_DIFFERENCE(_start, _end) \
    ((_end.tv_sec + _end.tv_nsec / 1.0e9) - \
    (_start.tv_sec + _start.tv_nsec / 1.0e9))

#define TIME_DIFFERENCE_NSEC(_start, _end) \
    ((_end.tv_nsec < _start.tv_nsec)) ? \
    ((_end.tv_sec - 1 - (_start.tv_sec)) * 1e9 + _end.tv_nsec + 1e9 - _start.tv_nsec) : \
    ((_end.tv_sec - (_start.tv_sec)) * 1e9 + _end.tv_nsec - _start.tv_nsec)

#define TIME_DIFFERENCE_GETTIMEOFDAY(_start, _end) \
    ((_end.tv_sec + _end.tv_usec / 1.0e6) - \
    (_start.tv_sec + _start.tv_usec / 1.0e6))

#endif	/* __HOST_H */

