
def istep_from_value_on_list(list_, value):
    for ipoint, (point1, point2) in enumerate(zip(list_[:-1],
                                                  list_[1:])):

        point1_s, point2_s = sorted((point1, point2))
        if (point1_s <= value) and (value <= point2_s):
            alpha = (value-point1_s)/(point2_s-point1_s)
            if alpha < 0 or alpha > 1:
                raise ValueError
            return ipoint + alpha
    values = [p for p in list_]
    min_values = min(values)
    max_values = max(values)
    raise ValueError('Specified value not found in list_: {} not in [{}, {}]'.format(value, min_values, max_values))


def interpolate_from_istep(objects, istep):
    n_objects = len(objects)
    if (istep < 0) or (istep > n_objects - 1):
        raise(ValueError('Extrapolating is not supported'))
    istep1 = int(istep)
    if istep1 == istep:
        # No interpolation needed
        return objects[int(istep)]
    else:
        alpha = istep - istep1
        point1 = objects[istep1]
        point2 = objects[istep1+1]
        return (1-alpha)*point1+(alpha)*point2