import sys
sys.path.insert(0, '/Volumes/RESEARCHUSB/Research/DEBUG/vis/python')

import imageio
import os
import profile_plotter_2d as prf
import argparse
import time
import athena_read
import numpy as np
import yaml
import warnings

from typing import *
from tqdm import tqdm
from sim_configs import *
from dataclasses import replace


def animate(config: SimulationConfig, animation_config: AnimationConfig) -> None:
    """
    Animates a directory of 2D profile images of all the same size. <br>
    :param config: Simulation configuration dataclass from YAML. <br>
    :param animation_config: Animation configuration dataclass from YAML.
    """

    t0 = time.time()

    images = []
    file_list = [fn for fn in os.listdir(config.output_path) if not fn.startswith('.') and fn.endswith(animation_config.extension)]
    for fn in tqdm(file_list):
        fp = os.path.join(config.output_path, fn)
        im = imageio.imread(fp)
        images.append(im)
    
    imageio.mimsave(os.path.join(config.output_path, animation_config.animation_name), images)

    print(f'Fcn *animate* completed in {time.time()-t0}s')


def receive(config: SimulationConfig) -> Tuple[List, List]:
    """
    Receives the current primitive and uov data file paths. <br>
    :param config: Simulation configuration dataclass from YAML. <br>
    :return: A tuple of lists, the first being the sorted prim files and the other being the sorted uov files.
    """

    prim = []
    uov = []

    for root, _, files in os.walk(config.prim_file):
        for fn in files:
            if '.athdf' not in str(fn) or fn.startswith('.'):
                continue
            if 'prim' in str(fn):
                prim.append(os.path.join(root, fn))
            else:
                uov.append(os.path.join(root, fn))
    prim = sorted(prim)
    uov = sorted(uov)

    return prim, uov


def create(config: SimulationConfig, stellar_properties: StellarPropConfig, eos_config: EOSConfig, animation_config: AnimationConfig) -> None:
    """
    Creates a directory of 2D profile images. <br>
    :param config: Simulation configuration dataclass from YAML. <br>
    :param stellar_properties: Stellar properties constants dataclass from YAML. <br>
    :param eos_config: EOS configuration dataclass from YAML. <br>
    :param animation_config: Animation configuration dataclass from YAML.
    """
    
    t0 = time.time()

    prim, uov = receive(config)

    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)
    
    for _, (pfp, ufp) in enumerate(tqdm(zip(prim, uov), total=len(prim))):
        new_config = replace(config, prim_file=pfp, uov_file=ufp, output_path=os.path.join(config.output_path, f'{_:05d}.{animation_config.extension}'))
        prf.plot(new_config, stellar_properties, eos_config)
    
    print(f'Fcn *create* completed in {time.time()-t0}s')


def find_min_max(config: SimulationConfig) -> None:
    """
    Finds the maximum and minimum v_r and v_phi values over the given boundaries. <br>
    :param config: The SimulationConfig instance, dictating simulation configuration. 
    """

    t0 = time.time()

    prim, uov = receive(config)

    values = {
        'vr': {
            'max': 0, 'min': 0
        },
        'vphi': {
            'max': 0, 'min': 0
        }
    }

    avgs_r = []
    avgs_phi = []
    for _ in tqdm(range(len(prim))):
        pfp = prim[_]
        df = athena_read.athdf(pfp)
        vr = df['vel1'][0]/1e9
        vphi = df['vel3'][0]/1e9

        r = df['x1v']
        theta = df['x2v']

        mask = r < float(config.outer_boundary)
        vr = vr[:, mask]
        vphi = vphi[:, mask]

        avgs_r.append(np.average(vr))
        avgs_phi.append(np.average(vphi))

        if values['vr']['max'] < np.max(vr):
            values['vr']['max'] = np.max(vr)
        if values['vr']['min'] > np.min(vr):
            values['vr']['min'] = np.min(vr)
        if values['vphi']['max'] < np.max(vphi):
            values['vphi']['max'] = np.max(vphi)
        if values['vphi']['min'] > np.min(vphi):
            values['vphi']['min'] = np.min(vphi)
    
    print(values)
    print(sum(avgs_r)/len(avgs_r))
    print(sum(avgs_phi)/len(avgs_phi))

    print(f'Fcn *find_max_min* completed in {time.time()-t0}s')


def load_config(yaml_path: str) -> Tuple[SimulationConfig, StellarPropConfig, EOSConfig, AnimationConfig]:
    """
    Loads dataclass configs from YAML. <br>
    :param yaml_path: Path to YAML file to read. <br>
    :return: A tuple containing the SimulationConfig, StellarPropConfig, EOSConfig, and AnimationConfig instances.
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
    
    animation = AnimationConfig(**cfg['animation'])
    
    return sim, prop, eos, animation


if __name__ == '__main__':
    # Argument parsing for command-line usage
    parser = argparse.ArgumentParser(prog='2D Profile Plot Animator Athena++',
                                    description='Animates 2D profiles from Athena++ magnetar wind simulations.')
    parser.add_argument('-config', type=str,
                        help='Path to YML config file.')

    args = parser.parse_args()

    sim, prop, eos, animation = load_config(args.config)

    if bool(animation.find):
        find_min_max(sim)
    else:
        if bool(animation.create):
            create(sim, prop, eos, animation)
        animate(sim, animation)

