"""
This code is for analysis on athdf files from Athena++: https://github.com/PrincetonUniversity/athena
This code is original, but the outputs were made to resemble those in Prasanna et. al. (general EOS) and Raives et. al. (isothermal EOS)

Prasanna et. al.: https://ui.adsabs.harvard.edu/abs/2022MNRAS.517.3008P/abstract
                  https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.3141P/abstract
                  https://ui.adsabs.harvard.edu/abs/2024ApJ...973...91P/abstract

Ravies et. al.: https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.4418R/abstract
"""

import sys
sys.path.insert(0, '/Volumes/RESEARCHUSB/Research/DEBUG/vis/python')

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import athena_read
import argparse
import cmasher as cmr
import math
import scipy.ndimage
import warnings
import yaml
import cProfile
import io
import pstats

from scipy.interpolate import RectBivariateSpline
from typing import *
from sim_configs import *


_default_title = r'P_{\star}=200\ {\rm ms},\ B_0=3\times 10^{15}\ {\rm G},\ L_{\bar{\nu}_{\rm e}}=8\times 10^{51}\ {\rm ergs}\ {\rm s}^{-1}'


def plot(config: SimulationConfig, stellar_properties: StellarPropConfig, eos_config: EOSConfig) -> None:
    """
    Plots the 2D profiles using the Athena++ simulation dataframe files. <br>
    :param config: Simulation configuration dataclass from YAML. <br>
    :param stellar_properties: Stellar properties constants dataclass from YAML. <br>
    :param eos_config: EOS configuration dataclass from YAML.
    """

    # Read the Athena data frames
    df = athena_read.athdf(config.prim_file)
    df_uov = athena_read.athdf(config.uov_file)

    # Set font
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.family'] = 'STIXGeneral'

    # Get the polar projection for matplotlib
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    # Get the minimum and maximum r values
    r_min = np.min(df['x1v'][:])
    r_max = np.max(df['x1v'][:])

    # Set r_max outer boundary
    if config.outer_boundary is not None:
        r_max = float(config.outer_boundary)
        if r_max < r_min:
            r_min = 0
    else:
        config.outer_boundary = r_max

    # Set colors (for ease of changing and use later)
    # * Paul Tol's vibrant color scheme: https://packages.tesselle.org/khroma/articles/tol.html
    sonic_surface_color = '#cc3311'
    alfven_surface_color = '#0077bb'
    magnetosonic_surface_color = '#ee7733'

    # Create legends for the surface contour lines
    # We make this first so it gets covered up later
    if not isinstance(eos_config, IsothermalEOSConfig):
        sonic_surface_name = 'Adiabatic Sonic Surface'
    else:
        sonic_surface_name = 'Mach Surface'
    legend_elements = [plt.Line2D([0], [0], color=c, label=f'{l}') for c, l in zip(
        [sonic_surface_color, alfven_surface_color, magnetosonic_surface_color], 
        [sonic_surface_name, 'Alfvén Surface', 'Fast Magnetosonic Surface'],
    )]
    ax.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, -0.1))

    # Prep phi slice
    deg_phi, n_phi, phi_max = float(config.deg_phi), float(config.n_phi), float(config.phi_max)
    phi = int((n_phi/phi_max)*deg_phi)

    # Fetch all the values (magnetic fields, theta/r values, etc.).
    # ! These are in Lorentz-Heaviside units.
    Br = df['Bcc1'][phi]
    Btheta = df['Bcc2'][phi]
    Bphi = df['Bcc3'][phi]

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
    vr = df['vel1'][phi]/1e9
    vr = vr[:, mask]

    # r-Velocity colormesh profiles, or isothermal poloidal velocity over cT profiles
    iso_colormesh_condition = isinstance(eos_config, IsothermalEOSConfig) and eos_config.use_iso_colormesh
    if config.vr_range is None:
        if not iso_colormesh_condition:
            res1 = ax.pcolormesh(theta, r, vr, cmap=cm.magma, shading='gouraud', vmin=0)
        else:
            vp_over_cT  = (np.sqrt(((df['vel1'][phi]**2)+(df['vel2'][phi]**2)))/float(eos_config.cT))[:, mask]
            res1 = ax.pcolormesh(theta, r, vp_over_cT, cmap=cm.magma, shading='gouraud', vmin=0)
    else:
        if not iso_colormesh_condition:
            res1 = ax.pcolormesh(theta, r, vr, cmap=cm.magma, shading='gouraud', vmin=config.vr_range[0], vmax=config.vr_range[1])
        else:
            vp_over_cT  = (np.sqrt(((df['vel1'][phi]**2)+(df['vel2'][phi]**2)))/float(eos_config.cT))[:, mask]
            res1 = ax.pcolormesh(theta, r, vp_over_cT, cmap=cm.magma, shading='gouraud', vmin=config.vr_range[0], vmax=config.vr_range[1])

    # Prepare phi-velocity data
    r = r_true
    theta = np.linspace(-thetaf[-1], -thetaf[0], theta_true.shape[0])
    r, theta = np.meshgrid(r, theta)
    vphi = df['vel3'][phi]/1e9
    vphi = vphi[:, mask]
    vphi_max = np.max(vphi)
    vphi_min = -vphi_max

    # Phi-Velocity colormesh profiles, or isothermal beta plasma profiles
    # * Iceburn used as colormap for CVD-friendliness as a diverging colorscheme: https://cmasher.readthedocs.io/user/diverging/iceburn.html#iceburn
    if config.vphi_range is None:
        if not iso_colormesh_condition:
            res2 = ax.pcolormesh(theta, r, vphi, cmap=cmr.iceburn, shading='gouraud', vmin=vphi_min, vmax=vphi_max)
        else:
            mag_B_sq = (Br**2)+(Btheta**2)+(Bphi**2)
            beta = 2*(float(eos_config.cT)**2)*df['rho'][phi]/mag_B_sq
            beta = np.log10(beta[:, mask])
            beta_max = np.max(beta)
            res2 = ax.pcolormesh(theta, r, beta, cmap=cmr.iceburn, shading='gouraud', vmin=-beta_max, vmax=beta_max)
    else:
        if not iso_colormesh_condition:
            res2 = ax.pcolormesh(theta, r, vphi, cmap=cmr.iceburn, shading='gouraud', vmin=config.vphi_range[0], vmax=config.vphi_range[1])
        else:
            mag_B_sq = (Br**2)+(Btheta**2)+(Bphi**2)
            beta = 2*(float(eos_config.cT)**2)*df['rho'][phi]/mag_B_sq
            beta = np.log10(beta[:, mask])
            res2 = ax.pcolormesh(theta, r, beta, cmap=cmr.iceburn, shading='gouraud', vmin=config.vphi_range[0], vmax=config.vphi_range[1])

    # Colorbars for the colormesh profiles
    if not iso_colormesh_condition:
        cbar = plt.colorbar(res1, ax=ax, label=r'$v_r\ \left[10^9\ {\rm cm}\ {\rm s}^{-1}\right]$', location='left', fraction=0.05, extend='max')
        if not np.all(vphi == 0):
            cbar = plt.colorbar(res2, ax=ax, label=r'$v_{\phi}\ \left[10^9\ {\rm cm}\ {\rm s}^{-1}\right]$', location='right', fraction=0.05, extend='both')
        else:
            cbar = plt.colorbar(res2, ax=ax, label=r'$v_{\phi}\ \left[10^9\ {\rm cm}\ {\rm s}^{-1}\right]$', location='right', fraction=0.05)
    else:
        cbar = plt.colorbar(res1, ax=ax, label=r'$v_{\rm p}/c_T$', location='left', fraction=0.05, extend='max')
        if not np.all(vphi == 0):
            cbar = plt.colorbar(res2, ax=ax, label=r'$\log_{10}{(\beta)}$', location='right', fraction=0.05, extend='both')
        else:
            cbar = plt.colorbar(res2, ax=ax, label=r'$\log_{10}{(\beta)}$', location='right', fraction=0.05)

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
    interp_n_r = int(np.ceil((r_max - r0)/dr))
    r_interp = np.linspace(r[0], r_max, int(interp_n_r))
    
    # SciPy interp2d was initially used. Due to deprecation, RectBivariateSpline is used to interpolate instead.
    # * https://docs.scipy.org/doc/scipy/tutorial/interpolate/interp_transition_guide.html#interp-transition-guide
    rbs = RectBivariateSpline(r, theta, Br[:, mask].T)
    B1 = rbs(r_interp, theta_interp)
    rbs = RectBivariateSpline(r, theta, Btheta[:, mask].T)
    B2 = rbs(r_interp, theta_interp)

    # Mirror data from RectBivariateSpline
    B1_flipped = B1[:, ::-1]  
    B2_flipped = -B2[:, ::-1] 

    # Streamplots for the magnetic field lines
    s1 = ax.streamplot(theta_interp, r_interp, B2 / r_interp[:, None], B1, density=config.density, linewidth=0.5, color='white',
                    start_points=stream_seeds_upper, arrowsize=0.5, broken_streamlines=False)
    s2 = ax.streamplot(theta_interp_mirrored, r_interp, B2_flipped / r_interp[:, None], B1_flipped, density=config.density, linewidth=0.5, color='white',
                    start_points=stream_seeds_lower, arrowsize=0.5, broken_streamlines=False)

    # Calculate different wind speeds
    if not isinstance(eos_config, IsothermalEOSConfig):
        cs = df_uov['dt3'][phi]
    else:
        cs = float(eos_config.cT)
    v_poloidal = ((df['vel1'][phi]**2)+(df['vel2'][phi]**2))
    v_alfven = ((Br**2)+(Btheta**2))/(df['rho'][phi])
    mag_B = np.sqrt((Br**2)+(Btheta**2)+(Bphi**2))
    cos_theta = Br/mag_B
    v_fast_magnetosonic = 0.5*(
            (v_alfven)+(cs**2)+np.sqrt(
                ((v_alfven)+(cs**2))**2 - 4*(v_alfven)*(cs**2)*(cos_theta**2)
            )
    )

    # Full plot range for contour plotting
    theta_full = np.linspace(0, 2*np.pi, 2*len(theta_true))
    r_full = r_true
    theta_grid, r_grid = np.meshgrid(theta_full, r_full, indexing='ij')

    # Adiabatic Sonic Surface / General Sonic Surface
    if not isinstance(eos_config, IsothermalEOSConfig):
        frac = cs[:, mask]/v_poloidal[:, mask]
        if bool(config.smooth_surfaces):
            frac = scipy.ndimage.gaussian_filter(frac, sigma=1.0, order=0)
        frac_full = np.concatenate([frac[::-1, :], frac], axis=0)
        ax.contour(theta_grid, r_grid, frac_full, [1], colors=sonic_surface_color, linewidths=1)
    else:
        frac = df['vel1'][phi][:, mask]/float(eos_config.cT)
        if bool(config.smooth_surfaces):
            frac = scipy.ndimage.gaussian_filter(frac, sigma=1.0, order=0)
        frac_full = np.concatenate([frac[::-1, :], frac], axis=0)
        ax.contour(theta_grid, r_grid, frac_full, [1], colors=sonic_surface_color, linewidths=1)

    # Alfvén Surface
    frac = v_alfven[:, mask]/v_poloidal[:, mask]
    if bool(config.smooth_surfaces):
        frac = scipy.ndimage.gaussian_filter(frac, sigma=1.0, order=0)
    frac_full = np.concatenate([frac[::-1, :], frac], axis=0)
    ax.contour(theta_grid, r_grid, frac_full, [1], colors=alfven_surface_color, linewidths=1)

    # Fast Magnetosonic Surface
    frac = v_fast_magnetosonic[:, mask]/v_poloidal[:, mask]
    if bool(config.smooth_surfaces):
        frac = scipy.ndimage.gaussian_filter(frac, sigma=1.0, order=0) 
    frac_full = np.concatenate([frac[::-1, :], frac], axis=0)
    ax.contour(theta_grid, r_grid, frac_full, [1], colors=magnetosonic_surface_color, linewidths=1)

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
    if isinstance(eos_config, IsothermalEOSConfig) and eos_config.calculate_iso_quantities:
        M, R, B, Omega, rho = stellar_properties.M, stellar_properties.R, stellar_properties.B, stellar_properties.Omega, stellar_properties.rho
        M, R, B, Omega, rho = float(M), float(R), float(B), float(Omega), float(rho)
        G = 6.6743e-8  # G in CGS

        xi_B = np.sqrt(((B**2)*R)/(8*np.pi*G*M*rho))
        xi_Omega = np.sqrt(((Omega**2)*(R**3))/(2*G*M))
        xi_T = np.sqrt(((float(eos_config.cT)**2)*R)/(2*G*M))

        if config.title:
            warnings.warn('Title given, but isothermal quantities are being calculated. The iso quantities will replace the title.')
        
        xi_B = round(xi_B, 3)
        xi_T = round(xi_T, 3)

        if xi_Omega != 0.0:
            exponent = int(math.floor(math.log10(abs(xi_Omega))))
            significand = xi_Omega / (10**exponent)
            significand_str = f'{significand:.2f}'

            config.title = fr'\xi_B={xi_B:.2f},\ \xi_\Omega={significand_str}\times 10^{{{exponent}}},\ \xi_T={xi_T:.2f}'
        else:
            config.title = fr'\xi_B={xi_B:.2f},\ \xi_\Omega={xi_Omega:.2f},\ \xi_T={xi_T:.2f}'

    # Check for default title
    if not config.title:
        config.title = _default_title

    # Set title
    ax.set_title(f'${config.title}$')  # For use with LaTeX

    # todo add scale bar 
    
    plt.gca().set_aspect('equal')
    plt.tight_layout()

    path = os.path.abspath(config.output_path)
    mpl.rcParams["savefig.directory"] = os.path.dirname(config.output_path)
    plt.savefig(path, dpi=300, format='png')

    plt.close()


def load_config(yaml_path: str) -> Tuple[SimulationConfig, StellarPropConfig, EOSConfig]:
    """
    Loads dataclass configs from YAML. <br>
    :param yaml_path: Path to YAML file to read. <br>
    :return: A tuple containing the SimulationConfig, StellarPropConfig, and EOSConfig instances.
    """

    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    sim = SimulationConfig(**cfg['simulation'])
    prop = StellarPropConfig(**cfg['prop'])
    eos_type = cfg['eos']['type']

    if eos_type == 'isothermal':
        eos = IsothermalEOSConfig(**cfg['eos'])
    elif eos_type == 'general':
        eos = GeneralEOSConfig(**cfg['eos'])
    else:
        warnings.warn("EOS type not recognized. Defaulting to 'general'")
        eos = GeneralEOSConfig(**cfg['eos'])
    
    return sim, prop, eos 


def main(config_path: str) -> None:
    """
    Main function used for plotting and/or cProfile profiling. <br>
    :param config_path: The location of the config path. 
    """

    sim, prop, eos = load_config(config_path)
    plot(sim, prop, eos)


if __name__ == '__main__':
    # Argument parsing for command-line usage
    parser = argparse.ArgumentParser(prog='2D Profile Plotter Athena++',
                                    description='Plots 2D profiles from Athena++ magnetar wind simulations.')
    parser.add_argument('-config', type=str,
                        help='Path to YML config file.')
    parser.add_argument('-cprofile', action='store_true',
                        help='Enable cProfile performance profiling.')
    
    args = parser.parse_args()

    if not args.cprofile:
        main(args.config)
    else:
        pr = cProfile.Profile()
        pr.enable()
        main(args.config)
        pr.disable()

        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats()
        print(s.getvalue())
