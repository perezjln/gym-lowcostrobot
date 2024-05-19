from setuptools import find_packages, setup

setup(
    name="gym_lowcostrobot",
    version="0.0.1",
    description="Low cost robot gymnasium environments",
    author="Julien Perez",
    author_email="julien.perez@epita.fr",
    packages=find_packages(),
    install_requires=["gymnasium", "mujoco"],
)
