#if !defined(__LIBBP5_H__)
#define __LIBBP5_H__

#define BP5_VISIBILITY(mode) __attribute__((visibility(#mode)))

#if defined(__cplusplus)
#define BP5_EXTERN extern "C" BP5_VISIBILITY(default)
#else
#define BP5_EXTERN extern BP5_VISIBILITY(default)
#endif

#if defined(__cplusplus)
#define BP5_INTERN extern "C" BP5_VISIBILITY(hidden)
#else
#define BP5_INTERN extern BP5_VISIBILITY(hidden)
#endif

struct bp5_t;
BP5_EXTERN struct bp5_t *bp5_init(int *argc, char ***argv);

BP5_EXTERN void bp5_debug(int verbose, const char *fmt, ...);

BP5_EXTERN void bp5_error(const char *fmt, ...);

BP5_INTERN void bp5_assert_(int cond, const char *fmt, const char *file,
                            const unsigned line);
#define bp5_assert(COND, MSG) bp5_assert_(COND, MSG, __FILE__, __LINE__)

BP5_EXTERN double bp5_run(const struct bp5_t *bp5);

BP5_EXTERN void bp5_finalize(struct bp5_t **bp5);

#endif // __LIBBP5_H__
