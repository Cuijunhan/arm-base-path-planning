from setuptools import setup, find_packages

setup(
    name='s_curve_planner',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A package for generating joint trajectories using S-curve interpolation and visualizing them.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'matplotlib',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)