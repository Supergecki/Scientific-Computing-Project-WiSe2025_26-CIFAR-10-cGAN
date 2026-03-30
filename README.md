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
## Installation and Setup


7. **Running the Project**
The project uses YAML configuration files to manage hyperparameters and model architectures. To train a model, use the `train.py` script and point to the desired configuration.

* **To train the Baseline (DCGAN) model:**
    ```bash
    python training/train.py --config config/baseline_config.yaml
    ```
* **To train the Improved (BigGAN) model:**
    ```bash
    python training/improved_config.yaml
    ```
* **To visualize training results:**
    Once a training log is generated in the `results/` folder, run:
    ```bash
    python evaluation/visualize.py --path ./results/biggan_run_log.txt
    ```

### Installation troubleshooting
[... (Keep your existing troubleshooting points) ...]

## Project Structure
* `config/`: YAML files containing all hyperparameters and model settings.
* `models/`: Implementations of Baseline, SN-ResNet, and BigGAN architectures.
* `training/`: Main training loop and loss function implementations.
* `evaluation/`: Scripts for calculating FID scores and plotting training curves.
* `results/`: Storage for generated sample grids and training logs.

## Hardware Requirements
Training the BigGAN architecture is computationally intensive. We recommend:
* **GPU:** NVIDIA GPU with at least 8GB VRAM (support for CUDA 12.x).
* **Estimated Training Time:** ~4-4 hours for 100 epochs on an NIVIDA T4 (comparable to RTX 2070 super (although less memory than the T4)) class GPU.

### Installation troubleshooting
If you run into any errors during installation, try the following:
1. Run the `pip` command using its alternate version `python -m pip`, e.g. instead of `pip install *` run `python -m pip install *`. This issue is sometimes caused by not having the linking for `pip.exe` installed correctly.
2. Run
```
python -m pip install --upgrade setuptools
```
3. Check that your Python executable can be found in your system's PATH variable.

## Development notes
1. To update requirements, run
```
./util/requirements_update.bat
```
in the base directory.
