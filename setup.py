from setuptools import setup, find_packages

setup(
    name="sublens",
    version="1.0.0",
    packages=find_packages(include=["sublens", "sublens.*"]),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.22.0",
        "scipy>=1.8.0",
        "astropy>=5.0.0",
        "matplotlib>=3.5.0"
    ],
    author="Antigravity Gravitational Lensing Research Team",
    description="Physics-Informed Astrometric Inversion and Subhalo Discovery Toolkit for Strong Gravitational Lensing",
    python_requires=">=3.9",
)
