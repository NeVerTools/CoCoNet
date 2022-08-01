# CoCoNet

CoCoNet is a tool for construction and conversion of neural networks 
across different standards.
See the [LICENSE](https://github.com/NeVerTools/NeVer2/blob/main/LICENSE.txt) 
for usage terms. \
CoCoNet is written in Python, and relies on the 
[pyNeVer](https://www.github.com/nevertools/pynever) API.

---

# DISCLAIMER: This is an early alpha version, many bugs are yet to be fixed.

---
## Execution requirements

CoCoNet can be executed on any system running Python >= 3.8. \
The instructions below have been tested on Windows, 
Ubuntu Linux and Mac OS x86 and ARM-based Mac OS.

## Linux, Mac OS x86 & Windows
There is a number of Python packages required in order to
run CoCoNet. All the following packages can be installed
via PIP

```bash
pip install numpy PyQt5 pysmt pynever
```

After the installation, you can run CoCoNet from the root directory

```bash
python CoCoNet/coconet.py
```

## ARM-based Mac OS

Since the Python packages needed are incompatible with "Python for ARM
Platform" you can install [Anaconda](https://www.anaconda.com/) using
Rosetta and create a x86 Python virtual environment.

Create a new environment using Python 3.9.5 and activate it

```bash
$ conda create -n myenv python=3.9.5
$ conda activate myenv
```

You can now run PIP for installing the libraries and run CoCoNet

```bash
$ pip install numpy PyQt5 onnx torch torchvision pysmt pynever
$ python CoCoNet/coconet.py
```

Note that each time you want to run CoCoNet you'll need to activate 
the Conda environment.

## Usage

CoCoNet is a GUI for constructing and converting Neural Networks, but it 
also provides a simple CLI usage for checking whether a NN is compliant
with VNN-LIB and quick-converting networks in the ONNX format.
Typing

```bash
python CoCoNet/coconet.py -h
```

shows the possible command-line instructions available.
