#at the moment not needed, only for creating a package.


from setuptools import find_packages, setup

setup(
    name="heart_attack_risk",
    version="0.0",
    description="Predicting Heart Attack Risk using a CNN",
    long_description="",
    author="Valentin Leeb",
    author_email="valentin.leeb@tum.de",
    license="Apache Software License",
    home_page="",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.2",
        "scipy",
        "matplotlib",
        "pytorch",
        # "pytest",
        # "pytest-cov",
        # "pytest-xdist",
        # "nbmake",
        # "pytest-github-actions-annotate-failures",
        # "mpire",
    ],
)
