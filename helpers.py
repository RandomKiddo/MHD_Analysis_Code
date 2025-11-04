"""
Helper functions to prevent re-declaration of useful functions and operations.
"""

import time

from functools import wraps 

# * Adapted from pg. 31 of High Performance Python by Gorelick & Ozsvald, 2nd ed. 
# Function decorator to time a function.
def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t0 = time.time()
        returns = fn(*args, **kwargs)
        tf = time.time()
        print(f'Fcn *{fn.__name__}* completed in {tf-t0}s.')
        return returns
    return measure_time

