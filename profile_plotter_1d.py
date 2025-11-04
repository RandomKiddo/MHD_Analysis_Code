"""
This code is for analysis on athdf files from Athena++: https://github.com/PrincetonUniversity/athena
This code is original, but the outputs were made to resemble those in Prasanna et. al.

Prasanna et. al.: https://ui.adsabs.harvard.edu/abs/2022MNRAS.517.3008P/abstract
                  https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.3141P/abstract
                  https://ui.adsabs.harvard.edu/abs/2024ApJ...973...91P/abstract
"""

import sys
sys.path.insert(0, '/Volumes/RESEARCHUSB/Research/DEBUG/vis/python')

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpl
import os
import athena_read
import math
import argparse
import warnings
import yaml

from tqdm import tqdm
from sim_configs import *
from dataclasses import replace


def plot(config: SimulationConfig, eos_config: EOSConfig, additional_config: AdditionalSimulation1DConfig) -> None:
    """
    Plots the 1D profiles using the Athena++ simulation dataframe files. <br>
    :param config: Simulation configuration dataclass from YAML. <br>
    :param eos_config: EOS configuration dataclass from YAML. <br>
    :param additional_config: Additional configuration dataclass from YAML for 1d profile plotting.
    """

    if bool(additional_config.recursive):  # Recursive behavior
        assert config.prim_file == config.uov_file

        # Count number of dataframes
        file_count = 0
        for fn in os.listdir(config.prim_file):
            if fn.endswith('.athdf') and not fn.startswith('.'):
                file_count += 1
        
        # Recurisve functionality
        for _ in tqdm(range(file_count//2)):
            pgen_name = config.pgen_name
            prim = os.path.join(config.prim_file, f'{pgen_name}.prim.{_:05d}.athdf')
            uov = os.path.join(config.uov_file, f'{pgen_name}.uov.{_:05d}.athdf')
            output = os.path.join(config.output_path, f'{_:05d}.png')

            new_config = replace(config, prim_file=prim, uov_file=uov, output_path=output)
            new_additional_config = replace(additional_config, recursive=False)
            plot(new_config, eos_config, new_additional_config)
        return  # Cut function
            

    # Read data frames
    df = athena_read.athdf(config.prim_file)
    df_uov = athena_read.athdf(config.uov_file)

    # Fix font family and get the subplots
    font_family = 'Georgia'
    plt.rcParams['font.family'] = font_family
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 7))

    # The theta and phi values to plot for
    theta = int((float(additional_config.n_theta)/float(additional_config.theta_max))*float(additional_config.deg_theta))
    phi = int((float(config.n_phi)/float(config.phi_max))*float(config.deg_phi))

    # r coordinate list
    x = df['x1v'][:]

    # Rho plotting
    y = df['rho'][phi][theta]  # 45deg 15
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
    if not isinstance(eos_config, IsothermalEOSConfig):
        y1 = df['press'][phi][theta]
    else:
        y1 = df['rho'][phi][theta]*(float(eos_config.cT)**2)
    axes[0][2].plot(x, y1, label=r'${\rm Gas}$')
    y2 = (np.power(df['Bcc1'][phi][theta], 2) + np.power(df['Bcc2'][phi][theta], 2) + np.power(df['Bcc3'][phi][theta], 2))/2
    if np.any(y2):
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
    if np.any(y2):
        axes[1][1].plot(x, y2, linestyle='--', label=r'$v_{\phi}$', color='#1f77b4')
    y3 = df['Bcc1'][phi][theta]/(np.sqrt(df['rho'][phi][theta]))
    if np.any(y3):
        axes[1][1].plot(x, y3, linestyle='-.', label=r'$\frac{B_r}{\sqrt{4\pi\rho}}$', color='#1f77b4')
    if not isinstance(eos_config, IsothermalEOSConfig):
        y4 = df_uov['dt3'][phi][theta]
        axes[1][1].plot(x, y4, linestyle='dotted', label=r'$c_{s}$', color='#1f77b4')
    else:
        axes[1][1].axhline(y=float(eos_config.cT), linestyle='dotted', label=r'$c_{T}$', color='#1f77b4')
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

    # Title
    time = df['Time']
    plt.suptitle(fr'Profile Plot, $t\approx {round(time,5)},\ \theta\approx {int(additional_config.deg_theta)}^\circ,\ \phi\approx {int(config.deg_phi)}^\circ$')

    # Tight layout
    plt.tight_layout()

    # Output path and save figure
    path = os.path.abspath(config.output_path)
    mpl.rcParams["savefig.directory"] = os.path.dirname(config.output_path)
    plt.savefig(path, dpi=300, format='png')

    # Close the plot
    plt.close()


def load_config(yaml_path: str) -> Tuple[SimulationConfig, EOSConfig, AdditionalSimulation1DConfig]:
    """
    Loads dataclass configs from YAML. <br>
    :param yaml_path: Path to YAML file to read. <br>
    :return: A tuple containing the SimulationConfig, EOSConfig, and AdditionalSimulation1DConfig instances.
    """

    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    sim = SimulationConfig(**cfg['simulation'])
    eos_type = cfg['eos']['type']
    add = AdditionalSimulation1DConfig(**cfg['1d'])

    if eos_type == 'isothermal':
        eos = IsothermalEOSConfig(**cfg['eos'])
    elif eos_type == 'general':
        eos = GeneralEOSConfig(**cfg['eos'])
    else:
        warnings.warn("EOS type not recognized. Defaulting to 'general'")
        eos = GeneralEOSConfig(**cfg['eos'])
    
    return sim, eos, add


if __name__ == '__main__':
    # Argument parsing for command-line usage
    parser = argparse.ArgumentParser(prog='1D Profile Plotter Athena++', 
                                     description='Plots 1D profiles from Athena++ magnetar wind simulations.')
    parser.add_argument('-config', type=str,
                        help='Path to YML config file.')

    args = parser.parse_args()

    sim, eos, add = load_config(args.config)
    plot(sim, eos, add) 

    