# MLPC2020

For the paths in the notebook to work, make sure that the jupyter notebook lies in the same directory as the 'train' folder from the provided zip file

## Setup

This project requires an installation of *Python 3* as well as the following
libraries:

* Jupyter
* NumPy
* scikit-learn
* matplotlib
* seaborn
* pandas

Those may be installed manually or via the given [requirements.txt](./requirements.txt) file.

## Virtual environment

> Note: Those instructions only apply to Linux systems

A virtual environment essentially represents a project-based installation of
Python including dependencies. This allows using different package or
distribution versions without conflicts between projects or platforms.
To create a virtual environment, `venv` has to be installed:

```bash
python -m pip install venv
```

Then it may be initialized by via

```bash
python -m venv .envs
```

where `.envs` is the location in which the virtual environment is set up
(actually this name is also already registered in the [gitignore](./gitignore) file as
to not distribute environment files to other users).

After this is done, the environment may be loaded in the current shell by
sourcing the init script:

```bash
source .envs/bin/activate
```

Then, every `python` or `pip` call actually references the installation in the
environment.

The requirements may then be loaded via

```bash
pip install -r requirements.txt
```

And (when modified) may be persisted into a `requirements.txt` file via

```bash
pip freeze > requirements.txt
```

To leave the virtual environment in the current shell, simply use

```bash
deactivate
```
