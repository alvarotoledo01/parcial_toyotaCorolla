from setuptools import find_packages, setup

setup(
    name="mi_proyecto",
    packages=find_packages(exclude=["mi_proyecto_tests"]),
    install_requires=[
        "dagster",
        "dagster-cloud"
    ],
    extras_require={"dev": ["dagster-webserver", "pytest"]},
)
