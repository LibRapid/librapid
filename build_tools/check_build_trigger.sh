#!/bin/bash

set -e
set -x

COMMIT_MSG=$(git log --no-merges -1 --oneline)

# if [[ "$GITHUB_EVENT_NAME" == schedule ||
#       "$COMMIT_MSG" =~ \[ci\ build\] ||
#       "$COMMIT_MSG" =~ \[ci\ build\ gh\] ]]; then
#     echo "::set-output name=build::true"
# fi

# I don't know how to do this, so I'm just gonna cheat
echo "::set-output name=build::true"
