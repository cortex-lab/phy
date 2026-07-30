# Installation

phy requires Python 3.10 or newer. A fresh virtual environment is recommended so
that Qt and scientific Python dependencies from another project do not interfere
with the GUI.

## Install the stable release

### Recommended: uv

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) if it is
not already available, then install phy as an isolated command-line tool:

```bash
uv tool install --python 3.12 phy
```

This command is the same on Linux, macOS, and Windows PowerShell. `uv` installs
Python 3.12 if necessary, creates an environment dedicated to phy, and makes the
`phy` command available without manually activating that environment.

To upgrade the stable installation later:

```bash
uv tool upgrade phy
```

### Alternative: venv and pip

If you prefer Python's standard environment tools, create and activate a virtual
environment before installing phy.

On Linux and macOS:

```bash
python3 -m venv phy-env
source phy-env/bin/activate
python -m pip install --upgrade pip
python -m pip install phy
```

On Windows PowerShell:

```powershell
py -3.12 -m venv phy-env
.\phy-env\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install phy
```

If PowerShell prevents activation, either adjust its execution policy for your
user or run `.\phy-env\Scripts\python.exe -m pip install phy` followed by
`.\phy-env\Scripts\phy.exe --version`.

To upgrade this pip installation later, activate its environment and run
`python -m pip install --upgrade phy`.

### Check the installation

After either installation method:

```bash
phy --version
phy --help
```

The first command prints the installed phy version. The second should list
commands including `template-gui` and `template-describe`.

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

git clone https://github.com/cortex-lab/phylib.git
git clone https://github.com/cortex-lab/phy.git

cd phy
uv sync --dev
uv pip install --editable ../phylib
```

`uv sync` creates `phy/.venv`, installs phy editably, and installs the locked
development dependencies. The final command replaces the released phylib
package with the sibling source checkout.

### Windows PowerShell

```powershell
New-Item -ItemType Directory phy-source
Set-Location phy-source

git clone https://github.com/cortex-lab/phylib.git
git clone https://github.com/cortex-lab/phy.git

Set-Location phy
uv sync --dev
uv pip install --editable ..\phylib
```

These commands create the same editable development environment as the Linux and
macOS instructions.

### Verify which source is running

Run the following commands from either operating system:

```bash
uv run phy --version
uv pip show phy phylib
git log -1 --oneline
git -C ../phylib log -1 --oneline
```

For both packages, `uv pip show` should include an **Editable project location**
pointing to the checkout you just cloned. The Git commands identify the exact
commits being used. Development builds of phy may also show a development
version or Git suffix in `phy --version`.

### Update both checkouts

Save or commit your own source changes before updating. From the `phy` checkout:

```bash
git -C ../phylib pull --ff-only
git pull --ff-only
uv sync --dev
uv pip install --editable ../phylib
```

In Windows PowerShell, `../phylib` may also be written as `..\phylib`.
Re-running the sync picks up lockfile and package metadata changes, while the
last command ensures the sibling phylib checkout remains installed.

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
The legacy Kwik GUI needs additional packages. Install them in a separate
uv-managed environment because `klustakwik2` needs NumPy, setuptools, and Cython
available while it builds:

```bash
uv venv --python 3.12 phy-kwik-env
uv pip install --python phy-kwik-env phy klusta cython
uv pip install --python phy-kwik-env --no-build-isolation klustakwik2
```

Activate the environment using the command printed by `uv venv`. It can then be
opened with `phy kwik-gui path/to/file.kwik`.
