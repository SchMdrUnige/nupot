import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nupot",
    version="0.1.1",
    author="Sacha Medaer",
    author_email="sacha@medaer.me",
    python_requires=">=3.9.0",
    description="Neutrino Physics Symbolic Calculator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/???",
    download_url='https://github.com/???',
    license='???',
    packages=setuptools.find_packages(exclude=("tests",)),
    include_package_data=True,	# controls whether non-code files are copied when package is installed
    install_requires=["scipy", "numpy", "matplotlib", "pillow", "pyfftw",
                      "typing_extensions", "sympy"],
    classifiers=[
        "Natural Language :: English",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 or later "
        "(GPLv3+)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
)
