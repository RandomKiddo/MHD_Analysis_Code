import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import sys
sys.path.insert(0, '/Volumes/RESEARCHUSB/Research/DEBUG/vis/python')
import athena_read
import argparse
import cmasher as cmr
import math
import os
import scipy.ndimage
import warnings

from scipy.interpolate import RectBivariateSpline
from typing import *

_sentinel = object()
_default_title = r'P_{\star}=200\ {\rm ms},\ B_0=3\times 10^{15}\ {\rm G},\ L_{\bar{\nu_{\rm e}}}=8\times 10^{51}\ {\rm ergs}\ {\rm s}^{-1}'

def plot(prim_file: str, uov_file: str, output_path: str, obound: float = None, density: float = 15,
        vr_range: tuple = None, vphi_range: tuple = None, tol: float = 1e-15,
        title: object = _sentinel,
        iso: bool = False, cT: float = 5e9, calculate_iso_quantities: bool = False, star_constants: dict = {},
        smooth_surfaces: bool = False) -> None:
    """
    Plots the 1D profiles using the Athena++ simulation dataframe files. <br>
    :param prim_file: The string path to the prim athdf file to use. <br>
    :param uov_file: The string path to the uov athdf file to use. <br>
    :param output_path: The string output path to output the plot. <br>
    :param obound: Float value representing outer boundary for plotting in cm. Defaults to None. <br>
    :param density: The density of field lines to plot in the streamplot. Defaults to 15. <br>
    :param vr_range: R-velocity bar range to use. Defaults to None (meaning matplotlib decides). <br>
    :param vphi_range: Phi-velocity bar range to use. Defaults to None (meaning matplotlib decides). <br>
    :param tol: The fast magnetosonic surface tolerance to use when making the contour line. Defaults to 1e-15. <br>
    :param title: The title to use for the plot. Defaults to a LaTeX-set rotating magnetar of set luminosity title. <br>
    :param iso: If the simulation is isothermal. Defaults to False. <br>
    :param cT: The isothermal sound speed, only used if iso is True. Defaults to 5e9 cm/s. <br>
    :param calculate_iso_quantities: If isothermal xi quantities should be calculated and outputted. Defaults to False. <br>
    :param star_constants: Star constants dict to use in calculating iso quantities, with keys M*, R*, B*, Omega*, and rho*. Defaults to {}. <br>
    :param smooth_surfaces: If Gaussian smoothing should be applied to plot the surfaces. Defaults to False.
    """

    # Read the Athena data frames
    df = athena_read.athdf(prim_file)
    df_uov = athena_read.athdf(uov_file)

    # Set font
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.family'] = 'STIXGeneral'

    # Get the polar projection for matplotlib
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    #print(f"Time: {df['Time']}")

    # Get the minimum and maximum r values
    r_min = np.min(df['x1v'][:])
    r_max = np.max(df['x1v'][:])

    # Set r_max outer boundary
    if obound is not None:
        r_max = obound
        if r_max < r_min:
            r_min = 0
    else:
        obound = r_max

    # Create legends for the surface contour lines
    # We make this first so it gets covered up later
    if not iso:
        sonic_surface_name = 'Adiabatic Sonic Surface'
    else:
        sonic_surface_name = 'Mach Surface'
    legend_elements = [plt.Line2D([0], [0], color=c, label=f'{l}') for c, l in zip(
        ['red', '#1f77b4', 'orange'], 
        [sonic_surface_name, 'Alfvén Surface', 'Fast Magnetosonic Surface']
    )]
    ax.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, -0.1))

    # Fetch all the values (magnetic fields, theta/r values, etc.)
    Br = df['Bcc1'][0]
    Btheta = df['Bcc2'][0]
    Bphi = df['Bcc3'][0]

    thetaf = df['x2f']
    theta = df['x2v']
    rf = df['x1f']
    r = df['x1v']
    mask = (r >= r_min) & (r <= r_max)
    
    r_true = df['x1v']
    r_true = r_true[mask]
    theta_true = df['x2v']

    # Prepare r-velocity data
    r = r_true
    theta = np.linspace(thetaf[0], thetaf[-1], theta_true.shape[0])
    r, theta = np.meshgrid(r, theta)
    vr = df['vel1'][0]/1e9
    vr = vr[:, mask]

    # R-Velocity colormesh profiles
    if vr_range is None:
        res1 = ax.pcolormesh(theta, r, vr, cmap=cm.magma, shading='gouraud', vmin=0)
    else:
        res1 = ax.pcolormesh(theta, r, vr, cmap=cm.magma, shading='gouraud', vmin=vr_range[0], vmax=vr_range[1])

    # Prepeare phi-velocity data
    r = r_true
    theta = np.linspace(-thetaf[-1], -thetaf[0], theta_true.shape[0])
    r, theta = np.meshgrid(r, theta)
    vphi = df['vel3'][0]/1e9
    vphi = vphi[:, mask]
    vphi_max = np.max(vphi)
    vphi_min = -vphi_max

    # Phi-Velocity colormesh profiles
    if vphi_range is None:
        res2 = ax.pcolormesh(theta, r, vphi, cmap=cmr.wildfire_r, shading='gouraud', vmin=vphi_min, vmax=vphi_max)
    else:
        res2 = ax.pcolormesh(theta, r, vphi, cmap=cmr.wildfire_r, shading='gouraud', vmin=vphi_range[0], vmax=vphi_range[1])

    # Colorbars for the colormesh profiles
    cbar = plt.colorbar(res1, ax=ax, label=r'$v_r\ \left[10^9\ {\rm cm}\ {\rm s}^{-1}\right]$', location='left', fraction=0.05, extend='max')
    if not np.all(vphi == 0):
        cbar = plt.colorbar(res2, ax=ax, label=r'$v_{\phi}\ \left[10^9\ {\rm cm}\ {\rm s}^{-1}\right]$', location='right', fraction=0.05, extend='both')
    else:
        cbar = plt.colorbar(res2, ax=ax, label=r'$v_{\phi}\ \left[10^9\ {\rm cm}\ {\rm s}^{-1}\right]$', location='right', fraction=0.05)

    # Interpolate magnetic field data to make streamplots of magnetic field lines
    thetaf = df['x2f']  # face centers theta
    theta = df['x2v']  # cell centers theta
    rf = df['x1f']  # face centers radius
    r = df['x1v']  # cell centers radius

    mask = (r >= r_min) & (r <= r_max)
    r = r[mask]  # Apply mask

    n_theta_seeds = 32
    n_r_seeds = 32
    theta_seed_vals = np.linspace(thetaf[1], thetaf[-2], n_theta_seeds)

    r0 = r[0]
    stream_seeds_upper = np.array([theta_seed_vals, [r0] * n_r_seeds]).T
    theta_seed_vals_mirrored = np.linspace(-thetaf[-2], -thetaf[1], n_theta_seeds)
    stream_seeds_lower = np.array([theta_seed_vals_mirrored, [r0] * n_r_seeds]).T
    
    interp_nth = len(theta)*2
    theta_interp = np.linspace(theta[0], theta[-1], interp_nth)
    theta_interp_mirrored = np.linspace(-theta[-1],-theta[0], interp_nth)

    dr = rf[1] - rf[0]
    interp_n_r = int(np.ceil((obound - r0)/dr))
    r_interp = np.linspace(r[0], obound, int(interp_n_r))
    
    # SciPy interp2d was initially used. Due to deprecation, RectBivariateSpline is used to interpolate instead.
    # https://docs.scipy.org/doc/scipy/tutorial/interpolate/interp_transition_guide.html#interp-transition-guide
    rbs = RectBivariateSpline(r, theta, Br[:, mask].T)
    B1 = rbs(r_interp, theta_interp)
    rbs = RectBivariateSpline(r, theta, Btheta[:, mask].T)
    B2 = rbs(r_interp, theta_interp)

    # Mirror data from RectBivariateSpline
    B1_flipped = np.zeros_like(B1)
    for i in range(len(r_interp)):
        B1_flipped[i, :] = B1[i, ::-1]  
    B2_flipped = np.zeros_like(B2)
    for i in range(len(r_interp)):
        B2_flipped[i, :] = -B2[i, ::-1] 

    # Streamplots for the magnetic field lines
    s1 = ax.streamplot(theta_interp, r_interp, B2 / r_interp[:, None], B1, density=density, linewidth=0.5, color='white',
                    start_points=stream_seeds_upper, arrowsize=0.5, broken_streamlines=False)
    s2 = ax.streamplot(theta_interp_mirrored, r_interp, B2_flipped / r_interp[:, None], B1_flipped, density=density, linewidth=0.5, color='white',
                    start_points=stream_seeds_lower, arrowsize=0.5, broken_streamlines=False)

    # Calculate different wind speeds
    if not iso:
        cs = df_uov['dt3'][0]
    else:
        cs = cT
    v_poloidal = np.sqrt(((df['vel1'][0]**2)+(df['vel2'][0]**2)))
    v_alfven = np.sqrt(((Br**2)+(Btheta**2))/(4*np.pi*df['rho'][0]))
    mag_B = np.sqrt((Br**2)+(Btheta**2)+(Bphi**2))
    cos_theta = Br/mag_B
    v_fast_magnetosonic_sq = 0.5*(
        (v_alfven**2)+(cs**2)+np.sqrt(
            ((v_alfven**2)+(cs**2))**2 - 4*(v_alfven**2)*(cs**2)*(cos_theta**2)
        )
    )
    v_fast_magnetosonic = np.sqrt(v_fast_magnetosonic_sq)

    # Full plot range for contour plotting
    #theta_full = np.concatenate([-theta_true[::-1], theta_true])
    theta_full = np.linspace(0, 2*np.pi, 2*len(theta_true))
    r_full = r_true
    theta_grid, r_grid = np.meshgrid(theta_full, r_full, indexing='ij')

    # Adiabatic Sonic Surface / General Sonic Surface
    if not iso:
        frac = cs[:, mask]/v_poloidal[:, mask]
        if smooth_surfaces:
            frac = scipy.ndimage.gaussian_filter(frac, sigma=1.0, order=0)
        frac_full = np.concatenate([frac[::-1, :], frac], axis=0)
        ax.contour(theta_grid, r_grid, frac_full, [1], colors='red', linewidths=1)
    else:
        frac = df['vel1'][0][:, mask]/cT
        if smooth_surfaces:
            frac = scipy.ndimage.gaussian_filter(frac, sigma=1.0, order=0)
        frac_full = np.concatenate([frac[::-1, :], frac], axis=0)
        ax.contour(theta_grid, r_grid, frac_full, [1], colors='red', linewidths=1)

    # Alfvén Surface
    frac = v_alfven[:, mask]/v_poloidal[:, mask]
    if smooth_surfaces:
        frac = scipy.ndimage.gaussian_filter(frac, sigma=1.0, order=0)
    frac_full = np.concatenate([frac[::-1, :], frac], axis=0)
    ax.contour(theta_grid, r_grid, frac_full, [1], colors='#1f77b4', linewidths=1)

    # Fast Magnetosonic Surface
    frac = v_fast_magnetosonic[:, mask]/v_poloidal[:, mask]
    if smooth_surfaces:
        frac = scipy.ndimage.gaussian_filter(frac, sigma=1.0, order=0) 
    frac_full = np.concatenate([frac[::-1, :], frac], axis=0)
    ax.contour(theta_grid, r_grid, frac_full, [1], colors='orange', linewidths=1)

    # Offset axes
    ax.set_theta_offset(0.5*np.pi)

    # Set the actual outer boundary limits
    ax.set_ylim([0, np.max(r)])
    ax.set_yscale('linear')

    # Remove grid
    ax.grid(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Calculate iso quantites
    if iso and calculate_iso_quantities:
        if len([_ for _ in ('M*', 'R*', 'B*', 'Omega*', 'rho*') if _ not in star_constants]) > 0:
            warnings.warn('Isothermal and calculating iso quantities enabled, but star constants missing key quantities.')
        else:
            M, R, B, Omega, rho = star_constants['M*'], star_constants['R*'], star_constants['B*'], star_constants['Omega*'], star_constants['rho*']
            G = 6.6743e-8

            xi_B = np.sqrt(((B**2)*R)/(8*np.pi*G*M*rho))
            xi_Omega = np.sqrt(((Omega**2)*(R**3))/(2*G*M))
            xi_T = np.sqrt(((cT**2)*R)/(2*G*M))

            if title and title is not _sentinel:
                warnings.warn('Title given, but isothermal quantities are being calculated. The iso quantities will replace the title.')
            
            xi_B = round(xi_B, 3)
            xi_T = round(xi_T, 3)

            exponent = int(math.floor(math.log10(abs(xi_Omega))))
            significand = xi_Omega / (10**exponent)
            significand_str = f'{significand:.2f}'

            title = fr'\xi_B={xi_B:.2f},\ \xi_\Omega={significand_str}\times 10^{{{exponent}}},\ \xi_T={xi_T:.2f}'

    # Check for default title
    if title is _sentinel and not iso and not calculate_iso_quantities:
        title = _default_title

    # Set title
    ax.set_title(f'${title}$')  # For use with LaTeX

    # todo add scale bar
    
    plt.gca().set_aspect('equal')
    plt.tight_layout()

    path = os.path.abspath(output_path)
    mpl.rcParams["savefig.directory"] = os.path.dirname(output_path)
    plt.savefig(path, dpi=300, format='png')

    plt.close()


if __name__ == '__main__':
    # Argument parsing for command-line usage
    parser = argparse.ArgumentParser(prog='2D Profile Plotter Athena++',
                                    description='Plots 2D profiles from Athena++ magnetar wind simulations.')
    parser.add_argument('-prim', type=str, action='store',
                        help='The filename of the prim athdf file, possibly including path.')
    parser.add_argument('-uov', type=str, action='store',
                        help='The filename of the uov athdf file, possibly including path.')
    parser.add_argument('-s', '-save', type=str, action='store',
                        help='The location and file name to store the PNG file outputted, possibly including path.')
    parser.add_argument('-ob', '-obound', type=float, action='store', default=None,
                        help='The outer boundary max value float for plotting in cm. Defaults to None.')
    parser.add_argument('-d', '-dens', type=float, action='store', default=15,
                        help='The density of the streamplot lines to plot. Defaults to 15.')
    parser.add_argument('-vrmin', type=float, action='store', default=None,
                        help='The r-velocity minimum to use in the colorbar. Defaults to None.')
    parser.add_argument('-vrmax', type=float, action='store', default=None,
                        help='The r-velocity maximum to use in the colorbar. Defaults to None.')
    parser.add_argument('-vphimin', type=float, action='store', default=None,
                        help='The phi-velocity minimum to use in the colorbar. Defaults to None.')
    parser.add_argument('-vphimax', type=float, action='store', default=None,
                        help='The phi-velocity maximum to use in the colorbar. Defaults to None.')
    parser.add_argument('-tol', action='store', type=float, default=1e-15,
                        help='The fast magnetosonic tolerance to use when plotting. Defaults to 1e-15.')
    parser.add_argument('-t', '-title', type=str, action='store', default=r'P_{\star}=200\ {\rm ms},\ B_0=3\times 10^{15}\ {\rm G},\ L_{\bar{\nu_{\rm e}}}=8\times 10^{51}\ {\rm ergs}\ {\rm s}^{-1}',
                        help='The title to use for the plot. Defaults to a LaTeX-set rotating magnetar of set luminosity title.')
    parser.add_argument('-iso', action='store_true',
                        help='If the simulation is isothermal so P=rho*cT^2.')
    parser.add_argument('-cT', type=float, action='store', default=5e9,
                        help='The isothermal sound speed, if required, in cm/s. Defaults to 5e9 cm/s.')
    parser.add_argument('-ciq', action='store_true',
                        help='If isothermal xi quantities should be calculated and outputted. Defaults to False.')
    parser.add_argument('-Bst', type=float, action='store', default=3e15,
                        help='B* value for isothermal zeta quantity. Defaults to 3e15 G.')
    parser.add_argument('-Mst', type=float, action='store', default=1.4*1.9885e33,
                        help='M* value for isothermal zeta quantity. Defaults to 1.4 Msun in g.')
    parser.add_argument('-rhost', type=float, action='store', default=4e11,
                        help='rho* value for isothermal zeta quantity. Defaults to 4e11 g/cm^3.')
    parser.add_argument('-Omegast', type=float, action='store', default=300/1000,
                        help='Omega* value for isothermal zeta quantity. Defaults to 0.3 s (300 ms).')
    parser.add_argument('-Rst', type=float, action='store', default=1.2e6,
                        help='R* value for isothermal zeta quantity. Defaults to 1.2e6 cm.')
    parser.add_argument('-sm', '-smooth', action='store_true',
                        help='If Gaussian smoothing should be applied to surface plots. Defaults to False.')
    parser.add_argument('-slen', '-scalelen', type=float, action='store', default=50,
                        help='The scale bar length to use for the plot in km. Defaults to 50 km.')

    args = parser.parse_args()

    vr_range = []
    vphi_range = []
    if args.vrmin is None or args.vrmax is None:
        vr_range = None
    else:
        vr_range.append(args.vrmin)
        vr_range.append(args.vrmax)
        vr_range = tuple(vr_range)
    if args.vphimin is None or args.vphimax is None:
        vphi_range = None
    else:
        vphi_range.append(args.vphimin)
        vphi_range.append(args.vphimax)
        vphi_range = tuple(vphi_range)
    
    constants = {
        'B*': args.Bst, 'M*': args.Mst, 'rho*': args.rhost,
        'Omega*': args.Omegast, 'R*': args.Rst
    }

    plot(prim_file=args.prim, uov_file=args.uov, output_path=args.s, obound=args.ob, density=args.d,
         vr_range=vr_range, vphi_range=vphi_range, tol=args.tol, title=args.t, iso=args.iso, cT=args.cT,
         calculate_iso_quantities=args.ciq, star_constants=constants, smooth_surfaces=args.sm,
         scale_bar_length=args.slen)

