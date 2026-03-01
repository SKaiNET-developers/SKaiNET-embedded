#!/bin/sh
set -e
git config --global --add safe.directory /docs
if [ ! -d .git ]; then
  git init -q
  git config user.email "build@docs"
  git config user.name "build"
  git add -A
  git commit -q -m "init" --allow-empty
fi
exec /antora/node_modules/.bin/antora "$@"
