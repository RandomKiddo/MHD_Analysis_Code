import numpy as np


from dataclasses import dataclass
from typing import *


@dataclass
class SimulationConfig:
    """
    Simulation Config. <br>
    Data class for simulation, configuring the 2d profile output.
    """

    prim_file: str
    """Location of prim file."""

    uov_file: str
    """Location of uov file."""

    output_path: str
    """Plot output location."""

    outer_boundary: Optional[float] = None
    """Plot outer boundary."""

    density: float = 1
    """Field line density."""

    vr_range: Optional[Tuple[float, float]] = None
    """vr colorbar range."""

    vphi_range: Optional[Tuple[float, float]] = None
    """vphi colorbar range."""

    tol: float = 1e-15
    """Fast magnetosonic surface plotting tolerance."""

    title: Optional[str] = None
    """Plot title."""

    smooth_surfaces: bool = False
    """Use gaussian smoothing for surfaces."""

    deg_phi: float = 180
    """Phi degree value to use."""

    n_phi: int = 1
    """Number of phi cells."""

    phi_max: float = 360
    """Max phi degree value."""

    verbose: int = 0
    """
    If details should be printed to the console. Will ruin clean tqdm outputs from loops. <br>
    Set 0 for no outputs, 1 for full outputs.
    """

    pgen_name: str = 'parker'
    """
    The pgen name that effects the output athdf files.
    """


@dataclass
class StellarPropConfig:
    """
    Stellar Properties Config. <br>
    Stellar properties config, like stellar mass, radius, etc.
    """

    M: float = 1.4*1.9885e33
    """Stellar mass in g."""

    R: float = 1.2e6
    """Stellar radius in cm."""

    B: float = 3e15
    """Stellar magnetic field in G."""

    Omega: float = 2*np.pi/0.3
    """Stellar rotation Omega in s^-1."""

    rho: float = 4e11
    """Stellar density in g/cm^3."""


@dataclass
class IsothermalEOSConfig:
    """
    Isothermal EOS Config. <br>
    Isothermal EOS data class, containing iso constants.
    """

    type: str
    """EOS type, i.e., 'general', 'isothermal', etc. (should be 'isothermal')."""

    cT: float = 3e9
    """Isothermal sound speed in cm/s."""

    calculate_iso_quantities: bool = False
    """If iso xi constants should be calculated, will override title."""

    use_iso_colormesh: bool = False
    """If iso colormesh should be used instead of the typical vr and vphi colormesh."""


@dataclass
class GeneralEOSConfig:
    """General EOS Config."""

    type: str
    """EOS type, should be 'general'."""


EOSConfig = Union[IsothermalEOSConfig, GeneralEOSConfig]
"""EOS config type"""


@dataclass
class AnimationConfig:
    """
    Animation Config. <br>
    With AnimationConfig, use SimulationConfig's prim_file, uov_file, and output_path as directories. <br>
    Use prim_file=uov_file. <br>
    Include animation output extension in output_path, i.e., mp4. 
    """
    
    animation_name: str
    """Animation name plus extension (i.e., mp4)"""

    create: bool = False
    """Create 2d profile plots to animate"""

    extension: str = 'png'
    """Output plots file extension"""

    find: bool = False
    """Find the maximum and minimum velocity values"""


@dataclass
class AdditionalSimulation1DConfig:
    """
    Additional Simulation 1D Config. <br>
    Data class for 1d profile plotting, configuring the 1d profile output. <br>
    This adds additional information required for 1d profile plotting.
    """

    deg_theta: float = 45
    """Theta degree value to use."""

    n_theta: int = 128
    """Number of theta cells."""

    theta_max: float = 180
    """Max theta degree value."""

    recursive: bool = False
    """If the plotter should run recursively, outputting profile plots for every athdf file in a directory."""

