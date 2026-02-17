"""
This code is for inversion steps on astrophysical functions (but can also be used generally).
Currently, the user must provide a single-variable inversion function 'f' and its derivative function 'df'. 
The module path and filename are then passed as a command line argument and the inversion is done.
This module can also be used normally, by important functions in another python file and passing in values.

The code is original but based off of multiple pseudo-algorithms and mathematical formulae. 

! The hybrid inversion is a more robust way of doing inversion. The code is original but the idea comes from
! Matt Coleman, Research Scientist at Princeton University, who utilized the concept in the Athena++ astrophysical
! MHD codebase: https://github.com/PrincetonUniversity/athena.
The steps are as follows:
1. Check relative error.
    a. If error > 0.01, use secant method for one step.
    b. Else use Newton-Raphson for one step.
2. If step goes outside bracketing values, revert to Brent-Dekker (assuming bracketing values are given and exist).
3. Update bracketing values.
Loop untils error < tolerance or n iterations > n max iterations.

The inversion formulae used are (typically) from Wikipedia.
Newton-Raphson: https://en.wikipedia.org/wiki/Newton%27s_method.
Brent-Dekker: https://en.wikipedia.org/wiki/Brent%27s_method.
Secant: https://en.wikipedia.org/wiki/Secant_method.
Bisection: https://en.wikipedia.org/wiki/Bisection_method.

On general root-finding algorithms:
https://en.wikipedia.org/wiki/Root-finding_algorithm.

* Copyright © 2025 RandomKiddo
"""


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
from helpers import timefn


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
    if len(required_positional) != 1:
        raise TypeError(f"Function '{name}' must have at least one required positional argument (like 'y')")


@timefn
def newton_raphson(f: Callable[[float], float], df: Callable[[float], float], y0: float, tol: float = 1e-5, max_iter: int = 100, verbose: bool = True) -> Tuple[float, float, int]:
    """
    Newton-Raphson inversion. See Newton-Raphson: https://en.wikipedia.org/wiki/Newton%27s_method. <br>
    :param f: The inversion callable function, taking a float and returning a float. <br>
    :param df: The inversion callable derivative function, taking a float and returning a float. <br>
    :param y0: The float initial guess value to use. Pick a sensible one for your task. <br>
    :param tol: The precision or tolerance for the relative error to use for the inversion, defaults to 1e-5. <br>
    :param max_iter: The max number of iterations to use in the inversion, defaults to 100. <br>
    :param verbose: If convergence warnings should be returned to the console, defaults to True. <br>
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
    if verbose: 
        warnings.warn(f'Could not converge Newton-Raphson step in {max_iter} iterations. Latest prec: {abs(y_new-y)}') 
    return y_new, err, max_iter


@timefn
def brent_dekker(f: Callable[[float], float], a: float, b: float, tol: float = 1e-5, max_iter: int = 100, verbose: bool = True) -> Tuple[float, float, int]:
    """
    Brent-Dekker inversion. See Brent's Method: https://en.wikipedia.org/wiki/Brent%27s_method. <br>
    :param f: The inversion callable function, taking a float and returning a float. <br>
    :param a: The left bracketing value. Pick a sensible one for your task. <br>
    :param b: The right bracketing value. Pick a sensible one for your task. <br>
    :param tol: The precision or tolerance for the relative error to use for the inversion, defaults to 1e-5. <br>
    :param max_iter: The max number of iterations to use in the inversion, defaults to 100. <br>
    :param verbose: If convergence warnings should be returned to the console, defaults to True. <br>
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
    if (iter_no >= max_iter) and verbose:
        warnings.warn(f'Could not converge Brent-Dekker step in {max_iter} iterations. Latest prec: {np.abs(b-a)}. Last s: {s}.') 
        
    # Return the s value (location of root), the relative error, and the iteration number.
    return s, np.abs(b-a), iter_no


# todo test
@timefn
def secant(f: Callable[[float], float], y0: float, y1: float, tol: float = 1e-5, max_iter: int = 100, stopping_method: int = 1, verbose: bool = True) -> Tuple[float, float, int]:
    """
    Secant method inversion. See https://en.wikipedia.org/wiki/Secant_method. <br>
    :param f: The inversion callable function, taking a float and returning a float. <br>
    :param y0: The float first initial guess value to use. Pick a sensible one for your task. <br>
    :param y1: The float second initial guess value to use. Pick a sensible one for your task. <br>
    :param tol: The precision or tolerance for the relative error to use for the inversion, defaults to 1e-5. <br>
    :param max_iter: The max number of iterations to use in the inversion, defaults to 100. <br>
    :param stopping_condition: Int value representing which stopping condition to use (1, 2, or 3), defaults to 1. See below for more info. <br>
    :param verbose: If convergence warnings should be returned to the console, defaults to True. <br>
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
    if verbose:
        warnings.warn(f'Could not converge Secant step in {max_iter} iterations. Latest prec: {err}. Last y2: {y2}.') 
    return y2, err, max_iter


# todo test
@timefn 
def bisection(f: Callable[[float], float], a: float, b:float, tol: float = 1e-5, max_iter: int = 100, verbose: bool = True) -> Tuple[float, float, int]:
    """
    Bisection method inversion. See https://en.wikipedia.org/wiki/Bisection_method. <br>
    :param a: The float left endpoint value to use. Pick a sensible one for your task. <br>
    :param b: The float right endpoint value to use. Pick a sensible one for your task. <br>
    :param tol: The precision or tolerance for the relative error to use for the inversion, defaults to 1e-5. <br>
    :param max_iter: The max number of iterations to use in the inversion, defaults to 100. <br>
    :param verbose: If convergence warnings should be returned to the console, defaults to True. <br>
    :return: A tuple of the inversion y-value, the relative error, and the number of iterations used.
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
    if verbose:
        warnings.warn(f'Could not converge Bisection step in {max_iter} iterations. Latest prec: {(b-a)/2}. Last c: {c}.') 
    return c, (b-a)/2, max_iter


def hybrid_inversion(f: Callable[[float], float], df: Callable[[float], float], y0: float, a: float, b: float, tol: float = 1e-5, max_iter: int = 100,
                     verbose: bool = True) -> Tuple[float, float, int]:
    """
    Hybrid method inversion. See below for details. <br>
    :param f: The inversion callable function, taking a float and returning a float. <br>
    :param df: The inversion callable derivative function, taking a float and returning a float. <br>
    :param y0: The float first initial guess value to use. Pick a sensible one for your task. <br>
    :param a: The left side bracketing value. Pick a sensible one for your task. <br>
    :param b: The right side bracketing value. Pick a sensible one for your task. <br>
    :param tol: The precision or tolerance for the relative error to use for the inversion, defaults to 1e-5. <br>
    :param max_iter: The max number of iterations to use in the inversion, defaults to 100. <br>
    :param verbose: If convergence warnings should be returned to the console, defaults to True. <br>
    :return: A tuple of the inversion y-value, the relative error, and the number of iterations used. <br>
    <br>
    The steps of this inversion are as follows:
    1. Check relative error.
        a. If error > 0.01, use secant method for one step.
        b. Else use Newton-Raphson for one step.
    2. If step goes outside bracketing values, revert to Brent-Dekker (assuming bracketing values are given and exist).
    3. Update bracketing values.
    Loop untils error < tolerance or n iterations > n max iterations. 

    ! The error method used for the secant step is stopping_method=1, or abs(y0-y1) < tol. 
    """

    # Initialize values.
    y_prev = y_curr = y0
    iter_no = 0
    err = f(y_curr) / y_curr if y_curr != 0 else f(y_curr)  # Temporary error value

    # Check if the initial guess is within the brackets. If not, start a Brent-Dekker step.
    if not (a <= y_curr <= b):
        warnings.warn('Initial y0 is not within bracketing values. Reverting to Brent-Dekker')
        y_curr, err, _ = brent_dekker(f, a, b, tol, max_iter=1, verbose=False)
    
    # Begin inversion.
    while abs(err) > tol and iter_no < max_iter:
        # First method will likely be secant, so we must ensure that y_curr != y_prev, else we'll get a division by zero error.
        y_prev = y0 + tol*2  # s.t. f(y_curr) - f(y_prev) != 0

        # Check the initial error for inversion method to use.
        if abs(err) > 0.01:
            try:
                # Large error, we use secant method for one step.
                y_next, err, _ = secant(f, y_prev, y_curr, tol, max_iter=1, stopping_method=1, verbose=False)
            except:
                # If an error is returned, secant method had some failure, and we revert to Brent-Dekker for this step instead.
                warnings.warn('Secant failed, reverting to Brent-Dekker.')
                y_next, err, _ = brent_dekker(f, a, b, tol, max_iter=1, verbose=False)
        else:
            try:
                # Smaller error, we use Newton-Raphson for one step.
                y_next, err, _ = newton_raphson(f, df, y_curr, tol, max_iter=1, verbose=False)
            except:
                # If an error is returned, Newton-Raphson had some failure, and we revert to Brent-Dekker for this step instead.
                warnings.warn('Newton-Raphson failed, reverting to Brent-Dekker')
                y_next, err, _ = brent_dekker(f, a, b, tol, max_iter=1, verbose=False)
        
        # If our step goes outside the bracketing values, we revert to a Brent-Dekker step to update them.
        if not (a <= y_next <= b):
            y_next, err, _ = brent_dekker(f, a, b, tol, max_iter=1, verbose=False)
        
        # Update y values.
        y_prev = y_curr
        y_curr = y_next

        # Check if the root is still bracketed and update bracketing values accordingly.
        if f(a)*f(y_curr) < 0:
            b = y_curr
        else:
            a = y_curr
        
        # ! The error is pre-calculated in each method, so we don't need to update the error.

        # Increment iterations.
        iter_no += 1
    
    # The root could not be found in the iteration time. Multiple factors could have caused this.
    if iter_no >= max_iter and verbose:
        warnings.warn(f'Could not converge Bisection step in {max_iter} iterations. Latest prec: {err}. Last y: {y_curr}.') 
    return y_curr, err, iter_no+1


if __name__ == '__main__':
    # Argument parsing for command-line usage.
    parser = argparse.ArgumentParser(description='Generalized inversion solver with user-provided functions.')

    parser.add_argument('function_file', type=str, help="Path to a .py file with 'f(y)' and 'df(y)' defined.")
    parser.add_argument('-y0', type=float, default=1.0, help='Initial guess for inversion solver. Defaults to 1.0.')
    parser.add_argument('-tol', type=float, default=1e-5, help='Convergence tolerance. Defaults to 1e-5.')
    parser.add_argument('-max_iter', type=int, default=100, help='Maximum number of iterations. Defaults to 100.')
    parser.add_argument('-verbose', action='store_true', help='If convergence warnings should be returned to the console.')
    parser.add_argument('-y1', type=float, default=0.0, help='Initial second guess for inversion solver. Defaults to 0.0.')
    parser.add_argument('-a', type=float, default=0.0, help='The left bracketing value. Defaults to 0.0.')
    parser.add_argument('-b', type=float, default=1.0, help='The right bracketing value. Defaults to 1.0.')
    parser.add_argument('-stopcon', type=int, default=1, help='The int stopping condition number for secant inversion. Defaults to 1.')

    args = parser.parse_args()

    module = safe_load_module_from_path(args.function_file)

    print('Which inversion method would you like to use (enter the integer)?')
    method = None
    while method is None:
        inp = input('[1]: Newton Raphson | [2]: Brent-Dekker | [3]: Secant | [4]: Bisection | [5]: Hybrid\n')
        try:
            method = int(inp)
        except:
            method = None
            print('Invalid input, please try again.')
        else:
            if not (1 <= method <= 5):
                method = None
                print('Invalid input, please try again.')
    
    if not hasattr(module, 'f') or not callable(module.f):
        raise AttributeError("The module must define a function named 'f(y)'")
    if (method == 1 or method == 5) and (not hasattr(module, 'df') or not callable(module.df)):
        raise AttributeError("The module must define a function named 'df(y)'")
    elif method == 1 or method == 1 and (hasattr(module, 'df') and callable(module.df)):
        validate_function_signature(module.df, 'df')
    
    validate_function_signature(module.f, 'f')
    
    res = None
    if method == 1:
        res = newton_raphson(module.f, module.df, args.y0, args.tol, args.max_iter, args.verbose)
    elif method == 2:
        res = brent_dekker(module.f, args.a, args.b, args.tol, args.max_iter, args.verbose)
    elif method == 3:
        res = secant(module.f, args.y0, args.y1, args.tol, args.max_iter, args.stopcon, args.verbose)
    elif method == 4:
        res = bisection(module.f, args.a, args.b, args.tol, args.max_iter, args.verbose)
    else:
        res = hybrid_inversion(module.f, module.df, args.y0, args.a, args.b, args.tol, args.max_iter, args.verbose)

    print(f'Last value of root: {res[0]}, with error: {res[1]}, in {res[2]} iterations.')

