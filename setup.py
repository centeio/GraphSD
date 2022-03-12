#! /usr/bin/env python

import os

from setuptools import find_packages, setup

exec(open('graphsd/_version.py').read())
ver_file = os.path.join('graphsd', '_version.py')
with open(ver_file) as f:
    exec(f.read())

DISTNAME = 'graph-sd'
DESCRIPTION = "Mining graphs with Subgroup Discovery",
LONG_DESCRIPTION = 'A package to look for unusual social interaction patterns with subgroup discovery.'
# with codecs.open('README.rst', encoding='utf-8-sig') as f:
#     LONG_DESCRIPTION = f.read()
MAINTAINER = 'C. Centeio Jorge'
MAINTAINER_EMAIL = 'c.jorge@tudelft.nl'
URL = 'https://github.com/centeio/GraphSD'
LICENSE = 'new BSD'
DOWNLOAD_URL = 'https://github.com/centeio/GraphSD'
VERSION = __version__
INSTALL_REQUIRES = ['pandas',
                    'numpy',
                    'orangecontrib3',
                    'scipy']
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7']
EXTRAS_REQUIRE = {
    'tests': [
        'pytest',
        'pytest-cov'],
    'docs': [
        'sphinx',
        'sphinx-gallery',
        'sphinx_rtd_theme',
        'numpydoc',
        'matplotlib'
    ]
}

setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      long_description=LONG_DESCRIPTION,
      zip_safe=False,  # the package can run out of an .egg file
      classifiers=CLASSIFIERS,
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE)
