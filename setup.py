from setuptools import setup, find_packages

setup(
        name='cme_lab_utils',
        version='0.0',
        description='Helpful functions used in the CME lab',
        url='https://gitlab.com/bsu/cme-lab/cme_lab_utils',
        author='CME Lab',
        author_email='ericjankowski@boisestate.edu',
        license='GPLv3',
        packages=find_packages(),
        package_dir={'cme_lab_utils': 'cme_lab_utils'},
        zip_safe=False,
        )
