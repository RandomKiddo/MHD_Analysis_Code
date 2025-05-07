import imageio
import os
import profile_plotting_2d as prf
import argparse
import time

import athena_read
import numpy as np

from typing import *

def animate(path: str, output_path: str, ext: str = 'png') -> None:
    """
    Animates a directory of 2D profile images of all the same size.
    :param path: The path to the directory of images.
    :param output_path: The output path and filename of the animation, including extension (recommeded: '.gif')
    :param ext: The extension of the image files. Defaults to 'png'. 
    """
    t0 = time.time()

    images = []
    count = 0
    for fn in os.listdir(path):
        if fn.startswith('.'):
            continue
        if fn.endswith(ext):
            fp = os.path.join(path, fn)
            im = imageio.imread(fp)
            images.append(im)
            print(count)
            count += 1
    
    imageio.mimsave(output_path, images)

    print(f'Fcn *animate* completed in {time.time()-t0}s')

def receive(path: str) -> Tuple[List, List]:
    """
    Receives the current primitive and uov data file paths.
    :param path: The path to the directory containing the prim and uov files.
    """
    prim = []
    uov = []

    for root, _, files in os.walk(path):
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

def create(path: str, output_dir: str, ext: str = 'png', vr_range: tuple = None, vphi_range: tuple = None, density: int = 15, 
           obound: float = None) -> None:
    """
    Creates a directory of 2D profile images.
    :param path: Path to the directory containing the prim and uov files.
    :param output_dir: The output directory for the images.
    :param ext: The extension of the images to use. Defaults to 'png'.
    :param vr_range: R-velocity bar range to use. Defaults to None (meaning matplotlib decides).
    :param vphi_range: Phi-velocity bar range to use. Defaults to None (meaning matplotlib decides).
    :param density: The density of vector field lines to use in the streamplot.
    :param obound: The outer boundary to use when plotting
    """
    t0 = time.time()

    prim, uov = receive(path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    count = 0
    max_vr, min_vr, max_vphi, min_vphi = 0, 0, 0, 0
    for pfp, ufp in zip(prim, uov):
        prf.plot(prim_file=pfp, uov_file=ufp, output_path=os.path.join(output_dir, f'{count:05d}.{ext}'), vr_range=vr_range, 
                vphi_range=vphi_range, density=density, obound=obound)
        print(count)
        count += 1
    
    print(max_vr, min_vr, max_vphi, min_vphi)
    
    print(f'Fcn *create* completed in {time.time()-t0}s')

def find_max_min(path: str, ext: str = 'png') -> None:
    t0 = time.time()

    prim, uov = receive(path)

    values = {
        'vr': {
            'max': 0, 'min': 0
        },
        'vphi': {
            'max': 0, 'min': 0
        }
    }

    count = 0
    for pfp in prim:
        df = athena_read.athdf(pfp)
        vr = df['vel1'][0]/10e9
        vphi = df['vel3'][0]/10e9
        if values['vr']['max'] < np.max(vr):
            values['vr']['max'] = np.max(vr)
        if values['vr']['min'] > np.min(vr):
            values['vr']['min'] = np.min(vr)
        if values['vphi']['max'] < np.max(vphi):
            values['vphi']['max'] = np.max(vphi)
        if values['vphi']['min'] > np.min(vphi):
            values['vphi']['min'] = np.min(vphi)
        print(count)
        count += 1
    
    print(values)

    print(f'Fcn *find_max_min* completed in {time.time()-t0}s')

def find_max_min_inner(path: str, ext: str = 'png') -> None:
    t0 = time.time()

    prim, uov = receive(path)

    values = {
        'vr': {
            'max': 0, 'min': 0
        },
        'vphi': {
            'max': 0, 'min': 0
        }
    }

    count = 0
    avgs_r = []
    avgs_phi = []
    for pfp in prim:
        df = athena_read.athdf(pfp)
        vr = df['vel1'][0]/10e9
        vphi = df['vel3'][0]/10e9

        r = df['x1v']
        theta = df['x2v']

        mask = r < 20000000.0
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
        print(count)
        count += 1
    
    print(values)
    print(sum(avgs_r)/len(avgs_r))
    print(sum(avgs_phi)/len(avgs_phi))

    print(f'Fcn *find_max_min_inner* completed in {time.time()-t0}s')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='2D Profile Plot Animator Athena++',
                                    description='Animates 2D profiles from Athena++ magnetar wind simulations.')
    parser.add_argument('-c', '--create', action='store_true', default=False,
                        help='Create the 2d profile plots to animate (using pre-defined behavior).')
    parser.add_argument('-p', '--path', type=str, action='store', help='Path to athdf files, possibly including root path.')
    parser.add_argument('-o', '--opath', type=str, action='store', 
                        help='Path to store images from athdf files, possibly including root path.')
    parser.add_argument('-oi', '--oimage', type=str, action='store', 
                        help='Output file for animation, possibly including root path, including extension.')
    parser.add_argument('-ext', '--extension', type=str, action='store', default='png',
                        help='Image file extension to use and search for.')
    parser.add_argument('-vrmin', type=float, action='store', default=None,
                        help='The r-velocity minimum to use in the colorbar.')
    parser.add_argument('-vrmax', type=float, action='store', default=None,
                        help='The r-velocity maximum to use in the colorbar.')
    parser.add_argument('-vphimin', type=float, action='store', default=None,
                        help='The phi-velocity minimum to use in the colorbar.')
    parser.add_argument('-vphimax', type=float, action='store', default=None,
                        help='The phi-velocity maximum to use in the colorbar.')
    parser.add_argument('-d', '--density', type=int, action='store', default=15,
                        help='The density of the vector lines in the streamplot to use.')
    parser.add_argument('-f', '--find', action='store_true', default=False,
                        help='Find the minimum and maximum v_r and v_phi values across the entire dataset.')
    parser.add_argument('-ob', action='store', type=float, default=None, 
                        help='The outer boundary to use for the slice plots.')

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

    if args.find:
        find_max_min_inner(path=args.path, ext=args.extension)
    else:
        if args.create:
            create(path=args.path, output_dir=args.opath, ext=args.extension, vr_range=vr_range, vphi_range=vphi_range, density=args.density, obound=args.ob)
        animate(path=args.opath, output_path=args.oimage, ext=args.extension)

