# CoCoNet

__CoCoNet__ is a tool for construction and conversion of neural networks 
across different standards.
See the [LICENSE](https://github.com/NeVerTools/CoCoNet/blob/main/LICENSE.txt) 
for usage terms. \
__CoCoNet__ is written in Python, and relies on the 
[pyNeVer](https://www.github.com/nevertools/pynever) API.

---
## Execution requirements

__CoCoNet__ can be executed on any system running Python >= 3.9.5 \
The instructions below have been tested on Windows, 
Ubuntu Linux and Mac OS x86 and ARM-based Mac OS.

## Linux, Mac OS x86 & Windows
The packages required in order to run __CoCoNet__ are the [pyNeVer](https://www.github.com/nevertools/pynever) API
and the [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) framework, which can be installed via PIP

```bash
pip install pynever PyQt6
```

After the installation, you can run __CoCoNet__ from the root directory

```bash
python CoCoNet/coconet.py
```

## ARM-based Mac OS

Since the Python packages needed are incompatible with "Python for ARM Platform" you can install 
[miniforge](https://github.com/conda-forge/miniforge) for arm64 (Apple Silicon) and create a Python virtual environment.

Create a new environment using Python 3.9.5 and activate it

```bash
$ conda create -n myenv python=3.9.5
$ conda activate myenv
$ conda install -c apple tensorflow-deps
```

You can now run PIP for installing the libraries and run __CoCoNet__

```bash
$ pip install tensorflow-macos tensorflow-metal
$ pip install pynever PyQt6
$ python CoCoNet/coconet.py
```

Note that each time you want to run __CoCoNet__ you'll need to activate the Conda environment.

## Usage

__CoCoNet__ is a GUI for constructing and converting Neural Networks, but it 
also provides a simple CLI usage for checking whether a NN is compliant
with [VNN-LIB](www.vnnlib.org) and quick-converting networks in the ONNX format.
Typing

```bash
python CoCoNet/coconet.py -h
```

shows the possible command-line instructions available.
