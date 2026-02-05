# Scientific-Computing-Project-WiSe2025_26-CIFAR-10-cGAN
The repository for all code belonging to the Scientific Computing Project at TU Bergakademie Freiberg in winter semester 2025/26, implementing a cGAN on the CIFAR-10 dataset.

## Installation and Setup
This project was developed and tested in Python version 3.14.
It is recommended to also use this version as other Python versions may have breaking changes regarding our interfaces.
1. You can install this python version [here](https://www.python.org/downloads/release/python-3143/). Other versions may work as well but are not recommended.
2. Download (or retrieve otherwise) the latest version of the repository's source code and unpack it.
3. It is recommended to run this project in a virtual environment. For this, open a terminal and change to the unpacked repository directory on your device. Then run
```
python -m venv .venv
```
to create the environment. If you wish to not use a virtual environment, skip this and step 4.
4. Every time you want to work with this environment now, you have to activate it. This works via 
```
.venv\Scripts\activate.bat
```
on Windows or via 
```
source .venv/bin/activate
```
on other OSes.
5. To ensure `pip` is installed in your environment and of the latest version, run
```
python -m ensurepip --upgrade
python -m pip install --upgrade pip
```
6. Install all project dependencies using
```
pip install -r requirements.txt
```
7. **TODO:** Describe how to run project.

### Installation troubleshooting
If you run into any errors during installation, try the following:
1. Run the `pip` command using its alternate version `python -m pip`, e.g. instead of `pip install *` run `python -m pip install *`. This issue is sometimes caused by not having the linking for `pip.exe` installed correctly.
2. Run
```
python -m pip install --upgrade setuptools
```
