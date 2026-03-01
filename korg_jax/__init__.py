"""korg_jax — JAX port of Korg.jl for stellar spectral synthesis."""

# Enable float64 by default; set KORGMAX_FLOAT32=1 to use float32 (faster on GPU, less precise).
import os, jax
if os.environ.get("KORGMAX_FLOAT32", "0") == "0":
    jax.config.update("jax_enable_x64", True)

from . import constants
from . import atomic_data
from . import species
from . import abundances
from . import wavelengths
from . import linelist
from . import atmosphere
from . import read_statmech
from . import cubic_splines
from . import voigt
from . import statmech
from . import continuum_absorption
from . import line_absorption
from . import hydrogen_lines
from . import utils
from . import radiative_transfer
from . import molecular_cross_sections
from . import interpolation
from . import synthesize
from . import synth

__version__ = "0.1.0"
