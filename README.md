# Lstm-Autoencoder



## Overview

## ⚙️ Installation

A convenient `make` command is provided to install the project.
It will create a virtual environment with the correct python version and install all packages with `poetry`.
In addition, all development tools are installed with `brew` on macOS if they are not already installed.

```bash
make install
```

## 🚧 Usage

Hydra is used to manage the configuration of the project which is stored in `src/lstm_autoencoder/conf`.
The project can be run from the main entry point `src/main.py` with the following command.

```bash
lstm_autoencoder
```
For more information on how to use hydra, please refer to the [documentation](https://hydra.cc/) or run the following command:

```bash
lstm_autoencoder --help
```

## 🧪 Testing

The source code is tested with [pytest](https://docs.pytest.org/en/stable/).
Every node and function should be tested with a unit test.
The tests are located in the `tests` folder where every pipeline step has its own folder.
For more information on how to write tests with pytest the [documentation](https://docs.pytest.org/en/stable/) or the example tests in the `tests` folder can be used as a reference.
You can run your tests as follows:

```bash
make test
```

## Commit Conventions

```bash
commit -m "<type>(<scope>): <description>

[body]

[footer(s)]
"
```

Example:

```bash
commit -m "feat(model): adding new model for training a forecasting model"
```

```bash
commit -m "build: update pandas to version 2.0"
```

Type of commit:
- **fix**: Bugfixes
- **feat**: New features
- **refactor**: Code change that neither fixes a bug nor adds a feature
- **docs**: Documentation-only changes
- **test**: Addition or correction of tests
- **build**: Changes of build components or external dependencies, like pip, docker ...
- **perf**: Code changes that improve the performance or general execution time
- **ci**: Changes to CI-configuration files and scripts
- **style**: Code style changes

Optional commit:
- **scope**: Context of the change
- **body**: Concise description of the change.
- **footer**: Consequences, which arise from the change

## 🕵️‍♀️ FAQ

### Question 1

Answer 1

### Question 2

Answer 2
