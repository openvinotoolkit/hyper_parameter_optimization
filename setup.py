from setuptools import setup, find_packages

setup(
    name='hpopt',
    version='0.3.0',
    url='https://github.com/openvinotoolkit/hyper_parameter_optimization',
    packages=find_packages(),
    description='Hyper-parameters Optimization',
    long_description='A Python library of automatic hyper-parameters optimization',
    install_requires=[
        "bayesian-optimization >= 1.2.0",
        "scipy>=1.8",
        "torch"
    ],
)

