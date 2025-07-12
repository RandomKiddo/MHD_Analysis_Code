import warnings
import os
import importlib.util as iu
import inspect
import argparse
import time


from typing import *
from types import *
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


def safe_load_module_from_path(path: str) -> ModuleType:
    if not os.path.isfile(path):
        raise FileNotFoundError(f'No such file: {path}.')

    module_name = os.path.splitext(os.path.basename(path))[0]
    spec = iu.spec_from_file_location(module_name, path)
    module = iu.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def validate_function_signature(func: Callable[[float], float], name: str) -> None:
    sig = inspect.signature(func)
    params = sig.parameters.values()

    # Must have at least one required positional or positional-or-keyword argument (e.g., y)
    has_required_arg = any(
        p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default == p.empty
        for p in params
    )

    if not has_required_arg:
        raise TypeError(f"Function '{name}' must have at least one required positional argument (like 'y')")


@timefn
def newton_raphson(f: Callable[[float], float], df: Callable[[float], float], y0: float, tol: float = 1e-5, max_iter: int = 100) -> Tuple[float, float, int]:
    y = y0
    for _ in range(max_iter):
        fy = f(y)
        dfy = df(y)

        if dfy == 0:
            warnings.warn('') # todo warning message
            return y
        
        y_new = y - fy/dfy
        err = abs(y_new - y)
        if err < tol:
            return y_new, err, _+1 
        y = y_new
    
    raise RuntimeError('') # todo error message


def full_inversion(f: Callable[[float], float], df: Callable[[float], float], y0: float, tol: float = 1e-5, max_iter: int = 100) -> float:
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generalized inversion solver with user-provided functions.')

    parser.add_argument('function_file', type=str, help="Path to a .py file with 'f(y)' and 'df(y)' defined.")
    parser.add_argument('-y0', type=float, default=1.0, help='Initial guess for inversion solver. Defaults to 1.0.')
    parser.add_argument('-tol', type=float, default=1e-5, help='Convergence tolerance. Defaults to 1e-5.')
    parser.add_argument('-max_iter', type=int, default=100, help='Maximum number of iterations. Defaults to 100.')

    args = parser.parse_args()

    module = safe_load_module_from_path(args.function_file)

    if not hasattr(module, 'f') or not callable(module.f):
        raise AttributeError("The module must define a function named 'f(y)'")
    if not hasattr(module, 'df') or not callable(module.df):
        raise AttributeError("The module must define a function named 'df(y)'")
    
    validate_function_signature(module.f, 'f')
    validate_function_signature(module.df, 'df')

    y_root, err, iters = newton_raphson(module.f, module.df, args.y0, args.tol, args.max_iter)
    print(f'Root found: y = {y_root}. Precision error: {err}. Found in {iters} iterations.')

