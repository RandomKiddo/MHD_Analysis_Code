import imageio
import os
import profile_plotting_2d as prf
import argparse
import time
import sys
sys.path.insert(0, '/Volumes/RESEARCHUSB/Research/DEBUG/vis/python')
import athena_read
import numpy as np

from typing import *
from tqdm import tqdm

def animate(path: str, output_path: str, ext: str = 'png') -> None:
    """
    Animates a directory of 2D profile images of all the same size. <br>
    :param path: The path to the directory of images. <br> 
    :param output_path: The output path and filename of the animation, including extension (recommeded: '.mp4') <br>
    :param ext: The extension of the image files. Defaults to 'png'. 
    """
    t0 = time.time()

    images = []
    file_list = [fn for fn in os.listdir(path) if not fn.startswith('.') and fn.endswith(ext)]
    for fn in tqdm(file_list):
        fp = os.path.join(path, fn)
        im = imageio.imread(fp)
        images.append(im)
    
    imageio.mimsave(output_path, images)

    print(f'Fcn *animate* completed in {time.time()-t0}s')

def receive(path: str) -> Tuple[List, List]:
    """
    Receives the current primitive and uov data file paths. <br>
    :param path: The path to the directory containing the prim and uov files. <br>
    :return: A tuple of lists, the first being the sorted prim files and the other being the sorted uov files.
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

def create(path: str, output_dir: str, ext: str = 'png', vr_range: tuple = None, vphi_range: tuple = None, density: float = 15, 
           obound: float = None, tol: float = 1e15, 
           title: str = r'$P_{\star}=200\ {\rm ms},\ B_0=3\times 10^{15}\ {\rm G},\ L_{\bar{\nu_{\rm e}}}=8\times 10^{51}\ {\rm ergs}\ {\rm s}^{-1}$') -> None:
    """
    Creates a directory of 2D profile images. <br>
    :param path: Path to the directory containing the prim and uov files. <br>
    :param output_dir: The output directory for the images. <br>
    :param ext: The extension of the images to use. Defaults to 'png'. <br>
    :param vr_range: R-velocity bar range to use. Defaults to None (meaning matplotlib decides). <br>
    :param vphi_range: Phi-velocity bar range to use. Defaults to None (meaning matplotlib decides). <br>
    :param density: The density of vector field lines to use in the streamplot. <br>
    :param obound: The outer boundary to use when plotting. <br>
    :param tol: The fast magnetosonic surface tolerance to use when making the contour line. Defaults to 1e-15. <br>
    :param title: The title to use for the plot. Defaults to a LaTeX-set rotating magnetar of set luminosity title.
    """
    t0 = time.time()

    prim, uov = receive(path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for _, (pfp, ufp) in enumerate(tqdm(zip(prim, uov), total=len(prim))):
        prf.plot(prim_file=pfp, uov_file=ufp, output_path=os.path.join(output_dir, f'{_:05d}.{ext}'), vr_range=vr_range, 
                vphi_range=vphi_range, density=density, obound=obound, tol=tol, title=title)
    
    print(f'Fcn *create* completed in {time.time()-t0}s')

def find_max_min(path: str, ext: str = 'png') -> None:
    """
    Finds the maximum and minimum v_r and v_phi values over the entire dataset. <br>
    :param path: Path to the directory containing the prim and uov files. <br>
    :param ext: The extension of the images to use. Defaults to 'png'.
    """
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

    for _ in tqdm(range(len(prim))):
        pfp = prim[_]
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
    
    print(values)

    print(f'Fcn *find_max_min* completed in {time.time()-t0}s')

def find_max_min_inner(path: str, ext: str = 'png') -> None:
    """
    Finds the maximum and minimum v_r and v_phi values over the inner boundary, set to 2e7 cm. <br>
    :param path: Path to the directory containing the prim and uov files. <br>
    :param ext: The extension of the images to use. Defaults to 'png'.
    """
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

    avgs_r = []
    avgs_phi = []
    for _ in tqdm(range(len(prim))):
        pfp = prim[_]
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
    
    print(values)
    print(sum(avgs_r)/len(avgs_r))
    print(sum(avgs_phi)/len(avgs_phi))

    print(f'Fcn *find_max_min_inner* completed in {time.time()-t0}s')


if __name__ == '__main__':
    # Argument parsing for command-line usage
    parser = argparse.ArgumentParser(prog='2D Profile Plot Animator Athena++',
                                    description='Animates 2D profiles from Athena++ magnetar wind simulations.')
    parser.add_argument('-c', '-create', action='store_true', default=False,
                        help='Create the 2d profile plots to animate (using pre-defined behavior). Defaults to False.')
    parser.add_argument('-p', '-path', type=str, action='store', help='Path to athdf files, possibly including root path.')
    parser.add_argument('-o', '-opath', type=str, action='store', 
                        help='Path to store images from athdf files, possibly including root path.')
    parser.add_argument('-oi', '-oimage', type=str, action='store', 
                        help='Output file for animation, possibly including root path, including extension.')
    parser.add_argument('-ext', '-extension', type=str, action='store', default='png',
                        help='Image file extension to use and search for excluding leading dot as a str. Defaults to png.')
    parser.add_argument('-vrmin', type=float, action='store', default=None,
                        help='The r-velocity minimum to use in the colorbar. Defaults to None.')
    parser.add_argument('-vrmax', type=float, action='store', default=None,
                        help='The r-velocity maximum to use in the colorbar. Defaults to None.')
    parser.add_argument('-vphimin', type=float, action='store', default=None,
                        help='The phi-velocity minimum to use in the colorbar. Defaults to None.')
    parser.add_argument('-vphimax', type=float, action='store', default=None,
                        help='The phi-velocity maximum to use in the colorbar. Defaults to None.')
    parser.add_argument('-d', '-dens', type=float, action='store', default=15,
                        help='The density of the vector lines in the streamplot to use. Defaults to 15.')
    parser.add_argument('-f', '-find', action='store_true', default=False,
                        help='Find the minimum and maximum v_r and v_phi values across the entire dataset. Defaults to False. Overrides other flags.')
    parser.add_argument('-ob', action='store', type=float, default=None, 
                        help='The outer boundary to use for the slice plots. Defaults to None.')
    parser.add_argument('-tol', action='store', type=float, default=1e-15,
                        help='The fast magnetosonic tolerance to use when plotting. Defaults to 1e-15.')
    parser.add_argument('-fi', '-findinner', action='store_true', default=False,
                        help='Find the minimum and maximum v_r and v_phi values across the inner boundary area. Defaults to False. Overrides other flags.')
    parser.add_argument('-t', '-title', type=str, action='store', default=r'$P_{\star}=200\ {\rm ms},\ B_0=3\times 10^{15}\ {\rm G},\ L_{\bar{\nu_{\rm e}}}=8\times 10^{51}\ {\rm ergs}\ {\rm s}^{-1}$',
                        help='The title to use for the animation plot. Defaults to a LaTeX-set rotating magnetar of set luminosity title.')

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

    if args.f:
        find_max_min(path=args.p, ext=args.ext)
    elif args.fi:
        find_max_min_inner(path=args.p, ext=args.ext)
    else:
        if args.c:
            create(path=args.p, output_dir=args.o, ext=args.ext, vr_range=vr_range, vphi_range=vphi_range, density=args.d, 
                   obound=args.ob, tol=args.tol, title=args.t)
        animate(path=args.o, output_path=args.oi, ext=args.ext)

