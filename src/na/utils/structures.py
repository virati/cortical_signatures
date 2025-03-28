from collections import defaultdict


def nestdict():
    """
    Create a nested dictionary
    """
    return defaultdict(nestdict)
