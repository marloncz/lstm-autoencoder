#! /usr/bin/env bash

# Update pyenv if python version is not yet available
if ! pyenv install -list | grep -q 3.11.9; then
    brew update && brew upgrade pyenv
fi
