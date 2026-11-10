# Baseline sources

Run from the artifact root:

```bash
bash refs/fork_baseline.sh
```

The script downloads shallow checkouts into this directory and selects the exact
Git HEADs recorded during our evaluation:

| Directory | Repository | Commit |
| --- | --- | --- |
| `hnswlib/` | [nmslib/hnswlib](https://github.com/nmslib/hnswlib) | `c1b9b79af3d10c6ee7b5d0afa1ce851ae975254c` |
| `PRO-HNSW/` | [huijunjin/PRO-HNSW](https://github.com/huijunjin/PRO-HNSW) | `adbfae262f8ee1f4974b94420ed7baef76d471e0` |
| `Wolverine/` | [LDW2020/Wolverine](https://github.com/LDW2020/Wolverine) | `221314b8d38bff3d89e9a34d5075dae55b06ea3b` |

Requires Git and network access to GitHub. Checkouts use detached HEAD; local
uncommitted edits are not copied. Re-running skips repositories at
the pinned commit and leaves their working trees untouched; other existing
directories are not overwritten.

PRO-HNSW's implementation is in `src/hnswalg.h`; its original layout is
preserved. The root CMakeLists.txt combines this header with the pinned hnswlib
headers in the build directory. Wolverine's benchmark adaptations are also
applied there, leaving all three source repositories unchanged.
