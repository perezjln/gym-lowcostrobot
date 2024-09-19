from setuptools import find_packages, setup

setup(
    name="gym_lowcostrobot",
    version="0.0.1",
    description="Low cost robot gymnasium environments",
    author="Julien Perez",
    author_email="julien.perez@epita.fr",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["gymnasium>=0.29", "mujoco>=3.0", "PyOpenGL==3.1.1a1"],
)
