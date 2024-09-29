#! /usr/bin/env bash

# Generate .zshrc file if not exists
touch ${ZDOTDIR-~}/.zshrc

# Install Homebrew <https://brew.sh/index_de>
if ! command -v brew &>/dev/null; then
    echo "Install Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"

    # Update Homebrew
    brew update
fi


# Install Direnv
if ! command -v direnv &>/dev/null; then
    echo "Install Direnv..."
    brew install direnv

    # Add hook to .zshrc
    printf "\n# Run direnv hook\n" >>${ZDOTDIR-~}/.zshrc
    echo "eval \"\$(direnv hook zsh )\"" >>${ZDOTDIR-~}/.zshrc
    source ${ZDOTDIR-~}/.zshrc
fi


# Install Pyenv
if ! command -v pyenv &>/dev/null; then
    echo "Install Pyenv..."
    brew install pyenv

    # pyenv post installation setup
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ${ZDOTDIR-~}/.zshrc
    echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ${ZDOTDIR-~}/.zshrc
    echo 'eval "$(pyenv init -)"' >> ${ZDOTDIR-~}/.zshrc

    # Add build dependencies to ensure proper python compilations
    # https://github.com/pyenv/pyenv/wiki#suggested-build-environment
    brew install openssl readline sqlite3 xz zlib tcl-tk

    source ${ZDOTDIR-~}/.zshrc
fi

# Install Pipx
if ! command -v pipx &>/dev/null; then
    echo "Install Pipx..."
    brew install pipx
    pipx ensurepath
fi



# Install Poetry
if ! command -v poetry &>/dev/null; then
    echo "Install Poetry..."
    pipx install poetry==1.8.2
fi
