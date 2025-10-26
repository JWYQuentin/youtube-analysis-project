#!/usr/bin/env bash
set -euo pipefail

make init
make train
make app
