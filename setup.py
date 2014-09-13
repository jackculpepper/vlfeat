#!/usr/bin/env python

from distutils.core import setup, Extension

_vlfeat = Extension('_vlfeat',
                    include_dirs = ['/usr/local/include/python2.7',
                                    '/usr/local/lib/python2.7/site-packages/numpy/core/include',
                                    './vl',
                                    '.'],
                    libraries = ['boost_python'],
                    extra_compile_args = ['-fopenmp', '-DVL_DISABLE_AVX'],
                    extra_link_args = ['-fopenmp'],
                    sources = ['vl/aib.c',
                               'vl/generic.c',
                               'vl/hikmeans.c',
                               'vl/ikmeans.c',
                               'vl/imopv.c',
                               'vl/mathop.c',
                               'vl/mathop_sse2.c',
                               'vl/mathop_avx.c',
                               'vl/pgm.c',
                               'vl/rodrigues.c',
                               'vl/stringop.c',
                               'vl/getopt_long.c',
                               'vl/host.c',
                               'vl/imopv_sse2.c',
                               'vl/mser.c',
                               'vl/random.c',
                               'vl/sift.c',
                               'vl/dsift.c',
                               'vl/quickshift.c',
                               'vlfeat/kmeans/vl_hikmeans.cpp',
                               'vlfeat/kmeans/vl_ikmeans.cpp',
                               'vlfeat/kmeans/vl_hikmeanspush.cpp',
                               'vlfeat/kmeans/vl_ikmeanspush.cpp',
                               'vlfeat/mser/vl_erfill.cpp',
                               'vlfeat/mser/vl_mser.cpp',
                               'vlfeat/sift/vl_sift.cpp',
                               'vlfeat/sift/vl_dsift.cpp',
                               'vlfeat/sift/vl_siftdescriptor.cpp',
                               'vlfeat/imop/vl_imsmooth.cpp',
                               'vlfeat/misc/vl_binsum.cpp',
                               'vlfeat/quickshift/vl_quickshift.cpp',
                               'vlfeat/py_vlfeat.cpp',
                              ])




setup(name='vlfeat',
    packages=['vlfeat', 'vlfeat.plotop', 'vlfeat.kmeans', 'vlfeat.kmeans',
              'vlfeat.quickshift', 'vlfeat.mser', 'vlfeat.misc'],
    ext_modules = [_vlfeat],
)

