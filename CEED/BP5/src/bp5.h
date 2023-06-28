#if !defined(__LIBBP5_H__)
#define __LIBBP5_H__

struct bp5_t;
struct bp5_t *bp5_init(int argc, char **argv);
void bp5_finalize(struct bp5_t **bp5);

#endif // __LIBBP5_H__
