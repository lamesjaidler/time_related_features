def _update_agg_name(agg, output_name, default_name):
    if output_name is None:
        agg.name = default_name
    else:
        agg.name = output_name
    return agg
