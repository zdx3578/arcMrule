#!/usr/bin/env bash
# Simple convenience launcher for popper
set -e
POPPER=${POPPER:-popper}   # assumes 'popper' is on $PATH
DIR=$(cd "$(dirname "$0")"; pwd)
cd "$DIR/popper"
$POPPER examples.pl