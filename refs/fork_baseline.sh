#!/usr/bin/env bash
set -euo pipefail

baseline_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

clone_revision() {
    local name="$1" url="$2" revision="$3"
    local target="${baseline_dir}/${name}"

    if [[ -e "$target" ]]; then
        if [[ -d "$target/.git" ]] &&
            [[ "$(git -C "$target" rev-parse HEAD)" == "$revision" ]]; then
            printf 'Already present: %s @ %s\n' "$name" "$revision"
            return
        fi
        printf 'Refusing to overwrite existing directory: %s\n' "$target" >&2
        exit 1
    fi

    git init --quiet "$target"
    git -C "$target" remote add origin "$url"
    git -C "$target" fetch --depth 1 origin "$revision"
    git -C "$target" checkout --detach FETCH_HEAD
    printf 'Ready: %s @ %s\n' "$name" "$revision"
}

# Match the Git HEADs of our evaluation.
clone_revision hnswlib https://github.com/nmslib/hnswlib.git \
    c1b9b79af3d10c6ee7b5d0afa1ce851ae975254c
clone_revision PRO-HNSW https://github.com/huijunjin/PRO-HNSW.git \
    adbfae262f8ee1f4974b94420ed7baef76d471e0
clone_revision Wolverine https://github.com/LDW2020/Wolverine.git \
    221314b8d38bff3d89e9a34d5075dae55b06ea3b
