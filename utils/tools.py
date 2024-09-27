"""tools to combine dictionary

"""
# import collections
from collections.abc import Mapping


def dict_update(d, u):
    """Improved update for nested dictionaries.

    Arguments:
        d: The dictionary to be updated.
        u: The update dictionary.

    Returns:
        The updated dictionary.
    """
    for k, v in u.items():
        # if isinstance(v, collections.Mapping):
        if isinstance(v, Mapping):
            d[k] = dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d
