import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ann-inverter",
    version="0.0.1",
    author="bence bogdandy, zsolt toth",
    author_email="{bogdandy.bence,zsolt.toth}@uni-eszterhazy.hu",
    description="ANN Inverter",
    long_description=long_description,
    url="https://github.com/Benczus/inversion_server",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)