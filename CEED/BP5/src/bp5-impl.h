#if !defined(__LIBBP5_IMPL_H__)
#define __LIBBP5_IMPL_H__

#include <bp5.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

struct bp5_t {
  unsigned verbose;
};

// Dynamic memory allocation function.
#define bp5_calloc(T, n) ((T *)calloc(n, sizeof(T)))

// Dynamic memory deallocation function.
void bp5_free_(void **p);
#define bp5_free(p) bp5_free_((void **)p)

#ifdef __cplusplus
}
#endif

#endif // __LIBBP5_IMPL_H__
