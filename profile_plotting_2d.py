import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import athena_read
import argparse
import cmasher as cmr
import math
import os
import scipy.ndimage

from scipy.interpolate import RectBivariateSpline

def plot(prim_file: str, uov_file: str, output_path: str, obound: float = None, density: int = 15,
        vr_range: tuple = None, vphi_range: tuple = None, tol: float = 1e-15) -> None:
    """
    Plots the 1D profiles using the Athena++ simulation dataframe files.
    :param prim_file: The string path to the prim athdf file to use.
    :param uov_file: The string path to the uov athdf file to use.
    :param output_path: The string output path to output the plot.
    :param obound: Float value representing outer boundary for plotting in cm. Defaults to None.
    :param density: The density of field lines to plot in the streamplot. Defaults to 15.
    :param vr_range: R-velocity bar range to use. Defaults to None (meaning matplotlib decides).
    :param vphi_range: Phi-velocity bar range to use. Defaults to None (meaning matplotlib decides).
    :param tol: The fast magnetosonic surface tolerance to use when making the contour line. 
    """

    # Read the Athena data frames
    df = athena_read.athdf(prim_file)
    df_uov = athena_read.athdf(uov_file)

    # Set font
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.family'] = 'STIXGeneral'

    # Get the polar projection for matplotlib
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    print(f"Time: {df['Time']}")

    # Get the minimum and maximum r values
    r_min = np.min(df['x1v'][:])
    r_max = np.max(df['x1v'][:])

    # Create legends for the surface contour lines
    # We make this first so it gets covered up later
    legend_elements = [plt.Line2D([0], [0], color=c, label=f'{l}') for c, l in zip(
        ['red', '#1f77b4', 'orange'], 
        ['Adiabatic Sonic Surface', 'Alfvén Surface', 'Fast Magnetosonic Surface']
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
    
    r_true = df['x1v']
    theta_true = df['x2v']

    # Prepare r-velocity data
    r = np.linspace(r_min, r_max, r_true.shape[0])
    theta = np.linspace(thetaf[0], thetaf[-1], theta_true.shape[0])
    r, theta = np.meshgrid(r, theta)
    vr = df['vel1'][0]/10e9

    # R-Velocity colormesh profiles
    if vr_range is None:
        res1 = ax.pcolormesh(theta, r, vr, cmap=cm.magma, shading='gouraud', vmin=0)
    else:
        res1 = ax.pcolormesh(theta, r, vr, cmap=cm.magma, shading='gouraud', vmin=vr_range[0], vmax=vr_range[1])

    # Prepeare phi-velocity data
    r = np.linspace(r_min, r_max, r_true.shape[0])
    theta = np.linspace(-thetaf[-1], -thetaf[0], theta_true.shape[0])
    r, theta = np.meshgrid(r, theta)
    vphi = df['vel3'][0]/10e9
    vphi_max = np.max(vphi)
    vphi_min = -vphi_max

    # Phi-Velocity colormesh profiles
    if vphi_range is None:
        res2 = ax.pcolormesh(theta, r, vphi, cmap=cmr.wildfire_r, shading='gouraud', vmin=vphi_min, vmax=vphi_max)
    else:
        res2 = ax.pcolormesh(theta, r, vphi, cmap=cmr.wildfire_r, shading='gouraud', vmin=vphi_range[0], vmax=vphi_range[1])

    # Colorbars for the colormesh profiles
    cbar = plt.colorbar(res1, ax=ax, label=r'$v_r\ \left[10^9\ {\rm cm}\ {\rm s}^{-1}\right]$', location='left', fraction=0.05)
    cbar = plt.colorbar(res2, ax=ax, label=r'$v_{\phi}\ \left[10^9\ {\rm cm}\ {\rm s}^{-1}\right]$', location='right', fraction=0.05)

    # Calculate different wind speeds
    cs = df_uov['dt3'][0]/10e9
    v_poloidal = np.sqrt(np.power(df['vel1'][0]/10e9, 2)+np.power(df['vel2'][0]/10e9, 2))
    mag_B = np.sqrt(np.power(Br, 2)+np.power(Btheta, 2)+np.power(Bphi, 2))
    cos_theta = np.divide(Br, mag_B)
    v_alfven = np.sqrt(
        np.divide(np.power(Br, 2)+np.power(Btheta, 2), 4*math.pi*df['rho'][0])
    )/10e9
    outside = np.power(v_poloidal, 2)+np.power(cs, 2)
    inside = np.sqrt(
        np.power(outside, 2)-(4*np.multiply(np.power(v_poloidal, 2), np.power(cs, 2))) # assume radial propagation
    )
    v_fast_magnetosonic_sq = 0.5*(outside + inside)
    v_fast_magnetosonic = np.sqrt(v_fast_magnetosonic_sq)

    # To plot the surfaces, we need to split the linspace into left and right portions due to data setup
    rl = np.linspace(r_min, r_max, r_true.shape[0])
    thetal = np.linspace(thetaf[0], thetaf[-1], theta_true.shape[0])
    rl, thetal = np.meshgrid(rl, thetal)

    rr = np.linspace(r_min, r_max, r_true.shape[0])
    thetar = np.linspace(-thetaf[-1], -thetaf[0], theta_true.shape[0])
    rr, thetar = np.meshgrid(rr, thetar)

    # Adiabatic Sonic Surface
    frac = np.divide(cs, v_poloidal)
    frac = scipy.ndimage.gaussian_filter(frac, sigma=1.0, order=0) 
    ax.contour(thetal, rl, frac, [1], colors='red')
    ax.contour(thetar, rr, frac[::-1], [1], colors='red')

    # Alfvén Surface
    frac = np.divide(v_alfven, v_poloidal)
    frac = scipy.ndimage.gaussian_filter(frac, sigma=1.0, order=0)
    ax.contour(thetal, rl, frac, [1], colors='#1f77b4')
    ax.contour(thetar, rr, frac[::-1], [1], colors='#1f77b4')

    # Fast Magnetosonic Surface
    frac = np.divide(v_fast_magnetosonic, v_poloidal)
    frac = scipy.ndimage.gaussian_filter(frac, sigma=1.0, order=0) 
    ax.contour(thetal, rl, frac, [1+tol], colors='orange') # tol=1e-15
    ax.contour(thetar, rr, frac[::-1], [1+tol], colors='orange')

    # Interpolate magnetic field data to make streamplots of magnetic field lines
    thetaf = df['x2f']
    theta = df['x2v']
    rf = df['x1f']
    r = df['x1v']
    st_nth=32
    st_nr=32
    st_pts = np.linspace(thetaf[1], thetaf[-2], st_nth)

    r0 = r[0]
    index = r.shape[0]-2
    st_pts = np.array([st_pts, [r0] * st_nr]).T
    st_pts2 = np.linspace(-thetaf[-2],-thetaf[1], st_nth)
    st_pts2 = np.array([st_pts2, [r0] * st_nr]).T
    interp_nth = len(theta)*2
    thetab = np.linspace(theta[0], theta[-1], interp_nth)
    theta2b = np.linspace(-theta[-1],-theta[0], interp_nth)
    interp_nr = int(np.ceil(rf[index+2] / (rf[1] - rf[0])))

    rb = np.linspace(r[0], rf[index+2], int(interp_nr/10.0))
    
    # SciPy interp2d was initially used. Due to deprecation, RectBivariateSpline is used to interpolate instead.
    rbs = RectBivariateSpline(r, theta, Br.T)
    B1 = rbs(rb, thetab)
    rbs = RectBivariateSpline(r, theta, Btheta.T)
    B2 = rbs(rb, thetab)

    # Mirror data from RectBivariateSpline
    B1_flipped = np.zeros_like(B1)
    for i in range(len(rb)):
        B1_flipped[i, :] = B1[i, ::-1]  
    B2_flipped = np.zeros_like(B2)
    for i in range(len(rb)):
        B2_flipped[i, :] = -B2[i, ::-1] 

    # Streamplots for the magnetic field lines
    s1 = ax.streamplot(thetab, rb, B2 / rb[:, None], B1, density=density, linewidth=0.5, color='white',
                    start_points=st_pts, arrowsize=0.5, broken_streamlines=False)
    s2 = ax.streamplot(theta2b, rb, B2_flipped / rb[:, None], B1_flipped, density=density, linewidth=0.5, color='white',
                    start_points=st_pts2, arrowsize=0.5, broken_streamlines=False)

    # Set outer boundary if exits
    if obound is not None:
        r_max = obound
        if r_max < r_min:
            r_min = 0
    ax.set_ylim([r_min, r_max])

    # Offset axes
    ax.set_theta_offset(0.5*np.pi)

    # Remove grid
    ax.grid(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Set title
    ax.set_title(r'$P_{\star}=200\ {\rm ms},\ B_0=3\times 10^{15}\ {\rm G},\ L_{\bar{\nu_{\rm e}}}=8\times 10^{51}\ {\rm ergs}\ {\rm s}^{-1}$')

    plt.gca().set_aspect('equal')
    plt.tight_layout()

    path = os.path.abspath(output_path)
    mpl.rcParams["savefig.directory"] = os.path.dirname(output_path)
    plt.savefig(path, dpi=300, format='png')

    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='2D Profile Plotter Athena++',
                                    description='Plots 2D profiles from Athena++ magnetar wind simulations.')
    parser.add_argument('-prim', type=str, action='store',
                        help='The filename of the prim athdf file, possibly including path.')
    parser.add_argument('-uov', type=str, action='store',
                        help='The filename of the uov athdf file, possibly including path.')
    parser.add_argument('-s', '-save', type=str, action='store',
                        help='The location and file name to store the PNG file outputted, possibly including path.')
    parser.add_argument('-ob', '-obound', type=float, action='store', default=None,
                        help='The outer boundary max value float for plotting in cm.')
    parser.add_argument('-d', '-dens', type=int, action='store', default=15,
                        help='The density of the streamplot lines to plot.')

    args = parser.parse_args()

    plot(prim_file=args.prim, uov_file=args.uov, output_path=args.s, obound=args.ob, density=args.d)

