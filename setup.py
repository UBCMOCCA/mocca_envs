try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages


setup(
    name="mocca_envs",
    version="0.1",
    install_requires=[
        "pybullet",
        "gym>=0.10.10",
        "numpy",
        "torch>=1.1.0",
        # And any other dependencies
    ],
    packages=find_packages(include="mocca_envs*"),
    include_package_data=True,
    zip_safe=False,
)
