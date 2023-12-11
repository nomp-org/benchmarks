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
    knl = t_unit.default_entrypoint
    knl = knl.copy(
        domains=[knl.combine_domains(tuple(range(len(knl.domains))))]
    )
    t_unit = t_unit.with_kernel(knl)
    t_unit = lp.add_prefetch(
        t_unit,
        "D",
        sweep_inames=["i", "i__", "j", "j__", "k", "k__", "l", "l_"],
        fetch_outer_inames=frozenset(["e"]),
    )

    knl = lp.tag_inames(
        knl,
        {
            "e": "g.0",
            "i*": "l.0",
            "j*": "l.1",
            "k*": "l.2",
            "l*": "ord",
            "D_dim_0": "l.0",
        },
    )
    t_unit = lp.split_iname(
        t_unit, "D_dim_1", 1, inner_tag="l.1", outer_tag="l.2"
    )

    return knl
