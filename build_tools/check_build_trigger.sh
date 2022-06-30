#!/bin/bash

set -e
set -x

COMMIT_MSG=$(git log --no-merges -1 --oneline)

# The commit marker "[ci build]" or "[ci build gh]" will trigger the build when required
if [[ "$GITHUB_EVENT_NAME" == schedule ||
      "$COMMIT_MSG" =~ \[ci\ build\] ||
      "$COMMIT_MSG" =~ \[ci\ build\ gh\] ]]; then
    echo "::set-output name=build::true"
fi