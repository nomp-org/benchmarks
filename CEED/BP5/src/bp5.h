#if !defined(__LIBBP5_H__)
#define __LIBBP5_H__

#ifdef __cplusplus
extern "C" {
#endif

struct bp5_t;
struct bp5_t *bp5_init(int *argc, char ***argv);

void bp5_debug(int verbose, const char *fmt, ...);

void bp5_error(const char *fmt, ...);

void bp5_assert_(int cond, const char *fmt, const char *file,
                 const unsigned line);
#define bp5_assert(COND, MSG) bp5_assert_(COND, MSG, __FILE__, __LINE__)

void bp5_finalize(struct bp5_t **bp5);

#ifdef __cplusplus
}
#endif

#endif // __LIBBP5_H__
