from setuptools import setup, find_packages

setup(
    name='ylftir',
    author='Eron Ristich',
    author_email='eristich@asu.edu',
    description='Python package to perform deconvolution of spectra. Designed primarily for FTIR deconvolution of silk films.',
    packages=find_packages(where='src'),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib'
    ],
    package_dir={'': 'src'},
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
