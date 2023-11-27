import loopy as lp

LOOPY_LANGUAGE_VERSION = (2018, 2)


def grid_loop(knl, context):
    (i,) = knl.default_entrypoint.all_inames()
    knl = lp.split_iname(knl, i, 512)
    knl = lp.tag_inames(knl, {f"{i}_outer": "g.0", f"{i}_inner": "l.0"})
    return knl


def gs(knl, context):
    knl = lp.split_iname(knl, "i", 512)
    knl = lp.tag_inames(knl, {"i_outer": "g.0", "i_inner": "l.0", "j*": "ord"})
    return knl


def ax(knl, context):
    knl = lp.tag_inames(
        knl, {"e": "g.0", "i*": "l.0", "j*": "l.1", "k*": "l.2", "l*": "ord"}
    )
    return knl
