import time

def time_magic(func):
    """
    Decorator to measure the execution time of a function.
    Returns a tuple: (function output, elapsed time in seconds)
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        return result, elapsed
    return wrapper
