import loopy as lp

LOOPY_LANGUAGE_VERSION = (2018, 2)


def grid_loop(knl, context):
    (iname,) = knl.default_entrypoint.all_inames()
    i_inner, i_outer = f"{iname}_inner", f"{iname}_outer"
    knl = lp.split_iname(
        knl, iname, 512, inner_iname=i_inner, outer_iname=i_outer
    )
    knl = lp.tag_inames(knl, {i_outer: "g.0", i_inner: "l.0"})
    return knl


def gs(knl, context):
    i, j, k = sorted(knl.default_entrypoint.all_inames())
    i_inner, i_outer = f"{i}_inner", f"{i}_outer"
    knl = lp.split_iname(knl, i, 512, inner_iname=i_inner, outer_iname=i_outer)
    knl = lp.tag_inames(knl, {i_outer: "g.0", i_inner: "l.0"})
    return knl


def ax(knl, context):
    inames = sorted(knl.default_entrypoint.all_inames())
    return knl
