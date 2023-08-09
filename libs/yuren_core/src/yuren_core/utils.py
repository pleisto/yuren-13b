import numpy as np


def last_index_of_list(lst: list, target: object):
    """
    Find the last index of target in lst.
    Same as rindex in Ruby.
    """
    return max(np.where(np.array(lst) == target)[0])
