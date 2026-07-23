# Installation

phy requires Python 3.10 or newer. A fresh virtual environment is recommended so
that Qt and scientific Python dependencies from another project do not interfere
with the GUI.

## Install the stable release

### Linux and macOS

Create and activate a virtual environment, then install phy from PyPI:

```bash
python3 -m venv phy-env
source phy-env/bin/activate
python -m pip install --upgrade pip
python -m pip install phy
```

If `python3` is not available but `python` is Python 3.10 or newer, use `python`
in the first command.

### Windows PowerShell

```powershell
py -3.10 -m venv phy-env
.\phy-env\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install phy
```

You may choose another installed Python version from 3.10 onward. For example,
use `py -3.12` for Python 3.12. If PowerShell prevents activation, either adjust
its execution policy for your user or run the environment's Python directly:

```powershell
.\phy-env\Scripts\python.exe -m pip install phy
.\phy-env\Scripts\phy.exe --version
```

### Check the installation

With the environment active:

```bash
phy --version
phy --help
```

The first command prints the installed phy version. The second should list
commands including `template-gui` and `template-describe`.

To upgrade the stable installation later:

```bash
python -m pip install --upgrade phy
```

## Install the latest phy and phylib source

Use this installation when testing an unreleased fix or developing phy. It
installs editable checkouts of both
[phylib](https://github.com/cortex-lab/phylib) and
[phy](https://github.com/cortex-lab/phy), with each repository initially on its
current default branch. An editable install uses the Python code in the checkout,
so code edits take effect without building a new package.

Choose a parent directory where the two repositories and the virtual environment
can live side by side.

### Linux and macOS

```bash
mkdir phy-source
cd phy-source

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

git clone https://github.com/cortex-lab/phylib.git
git clone https://github.com/cortex-lab/phy.git

python -m pip install --editable ./phylib
python -m pip install --editable ./phy
```

To run the phy test suite or build its documentation, install phy's development
dependencies instead in the final command:

```bash
python -m pip install --editable "./phy[dev]"
```

### Windows PowerShell

```powershell
New-Item -ItemType Directory phy-source
Set-Location phy-source

py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

git clone https://github.com/cortex-lab/phylib.git
git clone https://github.com/cortex-lab/phy.git

python -m pip install --editable .\phylib
python -m pip install --editable .\phy
```

For the phy test and documentation dependencies, replace the final command with:

```powershell
python -m pip install --editable ".\phy[dev]"
```

### Verify which source is running

Run the following commands from either operating system:

```bash
phy --version
python -m pip show phy phylib
git -C phy log -1 --oneline
git -C phylib log -1 --oneline
```

For both packages, `pip show` should include an **Editable project location**
pointing to the checkout you just cloned. The Git commands identify the exact
commits being used. Development builds of phy may also show a development
version or Git suffix in `phy --version`.

### Update both checkouts

Save or commit your own source changes before updating. From the `phy-source`
directory, with the virtual environment active:

```bash
git -C phylib pull --ff-only
git -C phy pull --ff-only
python -m pip install --upgrade --editable ./phylib
python -m pip install --upgrade --editable ./phy
```

In Windows PowerShell, the same commands work; `./phylib` and `./phy` may also be
written as `.\phylib` and `.\phy`.

Re-running the editable installation is worthwhile after an update because
dependencies and package metadata can change even though ordinary Python source
edits are already visible.

## Start phy

The Template GUI opens a spike-sorting output described by a `params.py` file:

```bash
phy template-describe /path/to/output/params.py
phy template-gui /path/to/output/params.py
```

`template-describe` is a useful non-GUI check before the first launch. See
[Preparing a dataset](dataset.md) if phy cannot load the output, or
[Quickstart](quickstart.md) for a guided first session.

## Legacy Kwik support

The Template GUI is the recommended workflow for current template-based sorters.
The legacy Kwik GUI needs additional packages:

```bash
python -m pip install klusta klustakwik2
```

It can then be opened with `phy kwik-gui path/to/file.kwik`.
