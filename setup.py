import os
from setuptools import setup


NAME = "cmeutils"
here = os.path.abspath(os.path.dirname(__file__))
about = {}
with open(os.path.join(here, NAME, "__version__.py")) as f:
    exec(f.read(), about)

setup(name=NAME,
      version=about["__version__"],
      description='Helpful functions used in the CME lab',
      url='https://github.com/cmelab/cmeutils',
      author='CME Lab',
      author_email='ericjankowski@boisestate.edu',
      license='GPLv3',
      packages=find_packages(),
      package_dir={'cmeutils': 'cmeutils'},
      zip_safe=False,
      )
