from setuptools import setup, find_packages
setup(
    name='safe_rl',
    packages=find_packages(),
    install_requires=[
        'gym',
        'joblib',
        'matplotlib',
        'mpi4py',
        'mujoco_py',
        'numpy',
        'seaborn',
        'tensorflow',
    ],
)