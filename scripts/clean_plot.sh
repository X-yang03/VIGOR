#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
rm -f -- "$root/figures/random_update.pdf" "$root/figures/localized_update.pdf"
