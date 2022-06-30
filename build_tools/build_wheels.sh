#!/bin/bash

set -e
set -x

if [[ "$RUNNER_OS" == "macOS" ]]; then
    export CC=gcc-11
    export CXX=g++-11
fi

