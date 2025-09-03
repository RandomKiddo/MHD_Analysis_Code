"""
This code is for inversion steps on astrophysical functions (but can also be used generally).
Currently, the user must provide a single-variable inversion function 'f' and its derivative function 'df'. 
The module path and filename are then passed as a command line argument and the inversion is done.

The Newton-Raphson inversion code is original. 

! The full inversion is a more robust way of doing inversion. The code is original but the idea comes from
! Matt Coleman, Research Scientist at Princeton University, who utilized the concept in the Athena++ astrophysical
! MHD codebase: https://github.com/PrincetonUniversity/athena.
The steps are as follows:
1. Check relative error
    a. If error > 0.01, use secant method for one step.
    b. Else use Newton-Raphson for one step.
2. If step goes outside bracketing values, revert to Brent-Dekker (assuming bracketing values are given and exist).
3. Update bracketing values.
Loop untils error < tolerance or n iterations > n max iterations.

The inversion formulae used are from Wikipedia.
Newton-Raphson: https://en.wikipedia.org/wiki/Newton%27s_method.
Brent-Dekker: https://en.wikipedia.org/wiki/Brent%27s_method.
Secant: https://en.wikipedia.org/wiki/Secant_method.
Bisection: https://en.wikipedia.org/wiki/Bisection_method.
"""

# todo update comments

import warnings
import os
import importlib.util as iu
import inspect
import argparse
import time
import numpy as np
import sys 


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
    """
    Loads a python module from a path to be used for inversion. <br>
    :param path: String path to module with python file name. <br>
    :return: The module as ModuleType.
    """

    # Check if file is found.
    if not os.path.isfile(path):
        raise FileNotFoundError(f'No such file: {path}.')

    # Retrieve module name and the module itself from the spec.
    module_name = os.path.splitext(os.path.basename(path))[0]
    spec = iu.spec_from_file_location(module_name, path)
    module = iu.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def validate_function_signature(func: Callable[[float], float], name: str) -> None:
    """
    Validates the function signature by checking its parameters. <br>
    :param func: The callable function, taking in a float and returning a float. <br>
    :param name: String name of the function.
    """

    # Inspect the signature of the function and fetch its parameters. 
    sig = inspect.signature(func)
    params = sig.parameters.values()

    # Check for the required positional arguments.
    required_positional = [
        p for p in params
        if p.default == p.empty and p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
    ]

    # Ensure that there is only one required positional arg. 
    if not len(required_positional) != 1:
        raise TypeError(f"Function '{name}' must have at least one required positional argument (like 'y')")


@timefn
def newton_raphson(f: Callable[[float], float], df: Callable[[float], float], y0: float, tol: float = 1e-5, max_iter: int = 100) -> Tuple[float, float, int]:
    """
    Newton-Raphson inversion. See Newton-Raphson: https://en.wikipedia.org/wiki/Newton%27s_method. <br>
    :param f: The inversion callable function, taking a float and returning a float. <br>
    :param df: The inversion callable derivative function, taking a float and returning a float. <br>
    :param y0: The float initial guess value to use. Pick a sensible one for your task. <br>
    :param tol: The precision or tolerance for the relative error to use for the inversion, defaults to 1e-5. <br>
    :param max_iter: The max number of iterations to use in the inversion, defaults to 100. <br>
    :return: A tuple of the inversion y-value, the relative error, and the number of iterations used.
    """

    # Initial guess.
    y = y0

    # Enter loop. We loop while we haven't reached max iterations.
    for _ in range(max_iter):
        # Calculate f(y) and f'(y) [called df(y)].
        fy = f(y)
        dfy = df(y)

        # Check if the derivative is 0. If it is, then we must return the latest value, as NR divides by the derivative to step.
        # This is logical since a curve of zero slope will not yield a 0, or infinite roots if y=0 (trivial).
        if dfy == 0:
            warnings.warn('Derivative of f(y) is zero. Return latest y.')
            return y
        
        # Calculate the new y based on NR-inversion and check if its error is less than the required tolerance. If so, we return
        # the new y, the error, and the iteration number. If not, we set this as the new y and continue inversion.
        y_new = y - fy/dfy
        err = abs(y_new - y)
        if err < tol:
            return y_new, err, _+1 
        y = y_new
    
    # The root could not be found in the iteration time. This is likely due to precision, not enough iterations, or bad initial y0 guess (among others).
    raise RuntimeError(f'Could not converge Newton-Raphson step in {max_iter} iterations. Latest prec: {abs(y_new-y)}') 


@timefn
def brent_dekker(f: Callable[[float], float], a: float, b: float, tol: float = 1e-5, max_iter: int = 100) -> float:
    """
    Brent-Dekker inversion. See Brent's Method: https://en.wikipedia.org/wiki/Brent%27s_method. <br>
    :param f: The inversion callable function, taking a float and returning a float. <br>
    :param a: The left bracketing value. Pick a sensible one for your task. <br>
    :param b: The right bracketing value. Pick a sensible one for your task. <br>
    :param tol: The precision or tolerance for the relative error to use for the inversion, defaults to 1e-5. <br>
    :param max_iter: The max number of iterations to use in the inversion, defaults to 100. <br>
    :return: A tuple of the inversion y-value, the relative error, and the number of iterations used.
    """

    # Assign the left and right brackets.
    left_bracket = f(a)
    right_bracket = f(b)

    # Check if the values are bracketed. If f(a)*f(b) >= 0, then it is not bracketed, as a f(y) must change signs for a root by IVT (and thus f(a)*f(b) must be negative).
    # If the brackets are flipped, they are swapped, and a warning is given.
    if left_bracket * right_bracket >= 0:
        raise ValueError('The root is not bracketed, f(a)*f(b) >= 0.')
    elif np.abs(left_bracket) < np.abs(right_bracket):
        left_bracket, right_bracket = right_bracket, left_bracket
        warnings.warn('|f(a)| < |f(b)|, swapping bracketing values.')
    
    # Define c and s initial values (parameters for inversion).
    c = a
    s = b

    def s_func(a0: float, b0: float, c0: float) -> float:
        """
        Returns the value of s using inverse quadratic interpolation. See https://en.wikipedia.org/wiki/Inverse_quadratic_interpolation. <br>
        :param a0: The current value for a (left bracket).
        :param b0: The current value for b (right bracket).
        :param c0: The current value for c.
        """

        # Inverse quadratic interpolation
        one = (a0 * f(b0) * f(c0)) / ((f(a0) - f(b0)) * (f(a0) - f(c0)))
        two = (b0 * f(a0) * f(c0)) / ((f(b0) - f(a0)) * (f(b0) - f(c0)))
        three = (c0 * f(a0) * f(b0)) / ((f(c0) - f(a0)) * (f(c0) - f(b0)))

        return one + two + three

    iter_no = 0
    mflag = True  # Boolean flag that tracks whether last step was a bisection step.
    d = 0
    delta = 2 * sys.float_info.epsilon * abs(b) + sys.float_info.epsilon  # Machine precision values based on right bracket, a.k.a. internal tolerance.

    # Begin inversion.
    while ((f(b) != 0 or f(s) != 0) or np.abs(b-a) > tol) and iter_no < max_iter:
        
        if f(a) != f(c) and f(b) != f(c):
            # Use inverse quadratic interpolation.
            s = s_func(a, b, c)
        else:
            # Use secant method. See https://en.wikipedia.org/wiki/Secant_method.
            s = b - f(b) * ((b - a) / (f(b) - f(a)))
                
        if ( not ((3*a + b)/4 < s < b) or (mflag and np.abs(s-b) >= np.abs(b-c)/2) or (not mflag and np.abs(s-b) >= np.abs(c-d)/2) 
            or (mflag and np.abs(b-c) < np.abs(delta)) or (not mflag and np.abs(c-d) < np.abs(delta))):
            # Use bisection method. See https://en.wikipedia.org/wiki/Bisection_method.
            s = (a+b)/2
            mflag = True  # Bisection method was used, so set this to True.
        else:
            mflag = False  # Bisection method was *not* used, so set this to False.
        
        # Update bracketing values based on s.
        d = c
        c = b

        if f(a) * f(s) < 0:
            b = s
        else:
            a = s
        
        # See if bracketing values need to be swapped.
        if np.abs(f(a)) < np.abs(f(b)):
            a, b = b, a
        
        # Update internal tolerance based on b.
        delta = 2 * sys.float_info.epsilon * abs(b) + sys.float_info.epsilon

        # Increment iteration number.
        iter_no += 1
    
    # The root could not be found in the iteration time. This is likely due to precision, not enough iterations, or bad initial bracketing values (among others).
    if (iter_no >= max_iter):
        raise RuntimeError(f'Could not converge Brent-Dekker step in {max_iter} iterations. Latest prec: {np.abs(b-a)}. Last s: {s}.') 
        
    # Return the s value (location of root), the relative error, and the iteration number.
    return s, np.abs(b-a), iter_no


# todo test
@timefn
def secant(f: Callable[[float], float], y0: float, y1: float, tol: float = 1e-5, max_iter: int = 100, stopping_method: int = 1) -> float:
    """
    Secant method inversion. See https://en.wikipedia.org/wiki/Secant_method. <br>
    :param f: The inversion callable function, taking a float and returning a float. <br>
    :param y0: The float first initial guess value to use. Pick a sensible one for your task. <br>
    :param y1: The float second initial guess value to use. Pick a sensible one for your task. <br>
    :param tol: The precision or tolerance for the relative error to use for the inversion, defaults to 1e-5. <br>
    :param max_iter: The max number of iterations to use in the inversion, defaults to 100. <br>
    :param stopping_condition: Int value representing which stopping condition to use (1, 2, or 3), defaults to 1. See below for more info. <br>
    :return: A tuple of the inversion y-value, the relative error, and the number of iterations used. <br>
    <br>
    On the error stopping conditions, we have three different possible conditions: <br>
        1. Calculate error using abs(y0-y1) < tol. <br>
        2. Calculate error using abs(y0/y1 - 1) < tol. <br>
        3. Calculate error using abs(f(y1)) < tol. <br>
        If any other integer is given, the default behavior is then 1. <br>
    """

    # Begin inversion. We require no set-up.
    for _ in range(max_iter):
        # Calculate the root of the linear function through (y0, f(y0)) and (y1, f(y1)).
        # Set the new boundary values.
        y2 = y1 - f(y1) * (y1 - y0) / float(f(y1) - f(y0))
        y0, y1 = y1, y2

        # Calculate the error. We have three conditions:
        # 1. Calculate error using abs(y0-y1) < tol.
        # 2. Calculate error using abs(y0/y1 - 1) < tol.
        # 3. Calculate error using abs(f(y1)) < tol.
        # If any other integer is given, the default behavior is then 1. 
        err = None

        if stopping_method == 2:
            err = abs(y0/y1 - 1)
        elif stopping_method == 3:
            err = abs(f(y1))
        else:
            err = abs(y0-y1)
        
        # Check if the error is less than the tolerance. If so, return the root found.
        if err < tol:
            return y2, err, _+1

    # The root could not be found in the iteration time. This is likely due to precision, not enough iterations, bad initial values, or secant method is unable to invert this with the given values (among others).
    raise RuntimeError(f'Could not converge Secant step in {max_iter} iterations. Latest prec: {err}. Last y2: {y2}.') 


# todo test
@timefn 
def bisection(f: Callable[[float], float], a: float, b:float, tol: float = 1e-5, max_iter: int = 100) -> float:
    """
    Bisection method inversion. See https://en.wikipedia.org/wiki/Bisection_method. <br>
    :param a: The float left endpoint value to use. Pick a sensible one for your task. <br>
    :param b: The float right endpoint value to use. Pick a sensible one for your task. <br>
    :param tol: The precision or tolerance for the relative error to use for the inversion, defaults to 1e-5. <br>
    :param max_iter: The max number of iterations to use in the inversion, defaults to 100. <br>
    """

    # Check if the left endpoint is indeed left of the right endpoint. Swap them otherwise and output a warning.
    if b < a:
        a, b = b, a
        warnings.warn('b < a, swapping endpoint values.')
    
    # Check bisection method condition for sign change. Fail otherwise.
    if not ((f(a) < 0 and f(b) > 0) or (f(a) > 0 and f(b) < 0)):
        raise RuntimeError('Bisection condition was not met. Either f(a) < 0 and f(b) > 0 or f(a) > 0 and f(b) < 0 must hold.')
    
    # Begin inversion.
    iter_no = 1
    while iter_no <= max_iter:
        # Find the midpoint of the end point.
        c = (a+b) / 2

        # Check if we have found a root or are within tolerance for one. If so, return the root, relative error, and iteration number.
        if f(c) == 0 or (b-a)/2 < tol:
            return c, (b-a)/2, iter_no
        
        # Increment iteration number.
        iter_no += 1

        # Update interval values based on the sign of f(a) and f(c). If they have the same sign, update a, else update b.
        if np.sign(f(c)) == np.sign(f(a)):
            a = c
        else:
            b = c
    
    # The root could not be found in the iteration time. This is likely due to precision, not enough iterations, bad initial values, or secant method is unable to invert this with the given values (among others).
    raise RuntimeError(f'Could not converge Bisection step in {max_iter} iterations. Latest prec: {(b-a)/2}. Last c: {c}.') 


def full_inversion(f: Callable[[float], float], df: Callable[[float], float], y0: float, brackets: tuple, tol: float = 1e-5, max_iter: int = 100) -> float:
    """
    The steps are as follows:
    1. Check relative error
        a. If error > 0.01, use secant method for one step.
        b. Else use Newton-Raphson for one step.
    2. If step goes outside bracketing values, revert to Brent-Dekker (assuming bracketing values are given and exist).
    3. Update bracketing values.
    Loop untils error < tolerance or n iterations > n max iterations.
    """

    err = 1
    while err > tol:
        pass 
    
    pass  # todo complete


if __name__ == '__main__':
    # Argument parsing for command-line usage.
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

    # y_root, err, iters = newton_raphson(module.f, module.df, args.y0, args.tol, args.max_iter)
    # print(f'Root found: y = {y_root}. Precision error: {err}. Found in {iters} iterations.')

