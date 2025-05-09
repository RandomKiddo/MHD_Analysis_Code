import h5py
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpl
import sys
sys.path.insert(0, '/Volumes/RESEARCHUSB/Research/DEBUG/vis/python')
import os
import athena_read
import plot_lines
import math
import argparse
import warnings

def plot(prim_file: str, uov_file: str, output_path: str, deg_th: int=45,
    n_th: int = 128, th_max: float = 180, iso: bool = False, cT: float = 5e9,
    deg_phi: int = 180, n_phi: int = 1, phi_max: float = 360):
    """
    Plots the 1D profiles using the Athena++ simulation dataframe files.
    :param prim_file: The string path to the prim athdf file to use.
    :param uov_file: The string path to the uov athdf file to use.
    :param output_path: The string output path to output the plot. 
    :param deg_th: The int theta degree value to plot for. Defaults to 45.
    :param n_th: The number of cells in the theta direction. Defaults to 128.
    :param th_max: The maximum value in the theta direction in degrees. Defaults to 180.
    :param iso: If the simulation is isothermal, so P=rho*cT^2. Defaults to False.
    :param cT: Isothermal sound speed, used when needed, in cm/s. Defaults to 5e9.
    :param deg_phi: The int phi degree value to plot for. Defaults to 180.
    :param n_phi: The number of cells in the phi direction. Defaults to 1.
    :param phi_max: The maximum value in the phi direction in degrees. Defaults to 360. 
    """

    # Read data frames
    df = athena_read.athdf(prim_file)
    df_uov = athena_read.athdf(uov_file)

    # Fix font family and get the subplots
    font_family = 'Georgia'
    plt.rcParams['font.family'] = font_family
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 7))

    # The theta and phi values to plot for
    theta = int((n_th/th_max)*deg_th)
    phi = int((n_phi/phi_max)*deg_phi)

    # r coordinate list
    x = df['x1v'][:]

    # Rho plotting
    y = df['rho'][phi][theta] # 45deg 15
    axes[0][0].plot(x, y)
    axes[0][0].set_xscale('log')
    axes[0][0].set_yscale('log')
    axes[0][0].set_xlabel(r'$r\ ({\rm cm})$')
    axes[0][0].set_ylabel(r'$\rho\ ({\rm g}\ {\rm cm}^{-3})$')
    axes[0][0].tick_params(axis='both', top=True, right=True, which='both', direction='in')

    # Temperature plotting (not fixed for isothermal)
    y = df_uov['dt1'][phi][theta]*8.6173e-11
    axes[0][1].plot(x, y)
    axes[0][1].set_xscale('log')
    axes[0][1].set_xlabel(r'$r\ ({\rm cm})$')
    axes[0][1].set_ylabel(r'${\rm Temperature\ (MeV)}$')
    axes[0][1].tick_params(axis='both', top=True, right=True, which='both', direction='in')

    # Pressure plotting (gas and magnetic)
    if not iso:
        y1 = df['press'][phi][theta]
    else:
        y1 = df['rho'][phi][theta]*(cT**2)
    axes[0][2].plot(x, y1, label=r'${\rm Gas}$')
    y2 = (np.power(df['Bcc1'][phi][theta], 2) + np.power(df['Bcc2'][phi][theta], 2) + np.power(df['Bcc3'][phi][theta], 2))/2
    axes[0][2].plot(x, y2, linestyle='dotted', label=r'${\rm Magnetic}$', color='#1f77b4')
    axes[0][2].legend()
    axes[0][2].set_xscale('log')
    axes[0][2].set_yscale('log')
    axes[0][2].set_xlabel(r'$r\ ({\rm cm})$')
    axes[0][2].set_ylabel(r'${\rm Pressure\ }({\rm g}\ {\rm cm}^{-1}\ {\rm s}^{-2})$')
    axes[0][2].tick_params(axis='both', top=True, right=True, which='both', direction='in')

    # q dot outflow plotting
    y = df_uov['dt2'][phi][theta]/1e21
    axes[1][0].plot(x, y)
    axes[1][0].set_xscale('log')
    axes[1][0].set_xlabel(r'$r\ ({\rm cm})$')
    axes[1][0].set_ylabel(r'$\dot{q}$/$10^{21}\ ({\rm ergs}\ {\rm g}^{-1}\ {\rm s}^{-1})$')
    axes[1][0].tick_params(axis='both', top=True, right=True, which='both', direction='in')

    # Velocity plotting (r, phi, magnetic, and cs)
    y1 = df['vel1'][phi][theta]
    axes[1][1].plot(x, y1, label=r'$v_{r}$')
    y2 = df['vel3'][phi][theta]
    axes[1][1].plot(x, y2, linestyle='--', label=r'$v_{\phi}$', color='#1f77b4')
    y3 = df['Bcc1'][phi][theta]/(np.sqrt(4 * math.pi) * np.sqrt(df['rho'][phi][theta]))
    axes[1][1].plot(x, y3, linestyle='-.', label=r'$\frac{B_r}{\sqrt{4\pi\rho}}$', color='#1f77b4')
    y4 = df_uov['dt3'][phi][theta]
    axes[1][1].plot(x, y4, linestyle='dotted', label=r'$c_{s}$', color='#1f77b4')
    axes[1][1].legend()
    axes[1][1].set_xscale('log')
    try:
        axes[1][1].set_yscale('log')
    except ValueError:
        axes[1][1].set_yscale('linear')
        warnings.warn('Log scale could not be set, defaulting to linear.')
    axes[1][1].set_xlabel(r'$r\ ({\rm cm})$')
    axes[1][1].set_ylabel(r'${\rm Velocity\ }({\rm cm}\ {\rm s}^{-1})$')
    axes[1][1].tick_params(axis='both', top=True, right=True, which='both', direction='in')

    # M dot plotting
    y = 4 * math.pi * np.power(df['x1v'], 2) * df['vel1'][phi][theta] * df['rho'][phi][theta] / 1.988e33
    axes[1][2].plot(x, y)
    axes[1][2].set_xscale('log')
    try:
        axes[1][2].set_yscale('log')
    except ValueError:
        axes[1][2].set_yscale('linear')
        warnings.warn('Log scale could not be set, defaulting to linear.')
    axes[1][2].set_xlabel(r'$r\ ({\rm cm})$')
    axes[1][2].set_ylabel(r'$\dot{M}\ (M_{\rm sun}\ {\rm s}^{-1})$')
    axes[1][2].tick_params(axis='both', top=True, right=True, which='both', direction='in')

    # Tight layout
    plt.tight_layout()

    # Output path and save figure
    path = os.path.abspath(output_path)
    mpl.rcParams["savefig.directory"] = os.path.dirname(output_path)
    plt.savefig(path, dpi=300, format='png')

if __name__ == '__main__':
    # Argument parsing for command-line usage
    parser = argparse.ArgumentParser(prog='1D Profile Plotter Athena++', 
        description='Plots 1D profiles from Athena++ magnetar wind simulations.')
    parser.add_argument('-prim', type=str, action='store', 
        help='The filename of the prim athdf file, possibly including path.')
    parser.add_argument('-uov', type=str, action='store',
        help='The filename of the uov athdf file, possibly including path.')
    parser.add_argument('-s', '-save', type=str, action='store',
        help='The location and file name to store the PNG file outputted, possibly including path.')
    parser.add_argument('-dth', '-degth', type=int, action='store', default=45,
        help='The int degree value to plot the profiles for. Defaults to 45 deg.')
    parser.add_argument('-nth', type=int, action='store', default=128,
        help='The number of theta-direction cells in the simulation. Defaults to 128.')
    parser.add_argument('-thmax', type=int, action='store', default=180,
        help='The max theta value of the simulation, in degrees. Defaults to 180 deg.')
    parser.add_argument('-iso', action='store_true',
        help='If the simulation is isothermal so P=rho*cT^2.')
    parser.add_argument('-cT', type=float, action='store', default=5e9,
        help='The isothermal sound speed, if required, in cm/s. Defaults to 5e9 cm/s.')

    args = parser.parse_args()

    plot(prim_file=args.prim, uov_file=args.uov, output_path=args.s, deg_th=args.dth, 
        n_th=args.nth, th_max=args.thmax, iso=args.iso, cT=args.cT)

    