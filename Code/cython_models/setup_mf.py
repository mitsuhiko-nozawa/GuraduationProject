from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy as np

sourcefiles = ['MF_cy.pyx']
setup(
    cmdclass = {'build_ext':build_ext},
    ext_modules = [Extension('MF_cy', sourcefiles)],
    include_dirs = [np.get_include()]
)

# python setup_mf.py built_ext --inplace