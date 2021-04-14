import collections

class ClosedEnvSignal(Exception):
    pass

class State:
    pass

def ensure_iterable(obj):
    """
    Ensures that the returned object is an interable. Returns unmodified obj
     f it is an iterable, or [obj] if it is not.
    """
    if not isinstance(obj, collections.abc.Iterable):
        obj = [obj]
    return obj

def unwrap_if_scalar(obj):
    """
    Unwraps obj if it is a sequence with a single item. Returns obj[0] if
    len(obj) == 1 and obj otherwise.
    """
    if len(obj) == 1:
        return obj[0]
    else:
        return obj