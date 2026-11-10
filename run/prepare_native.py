#!/usr/bin/env python3
"""Prepare pinned vendor headers in the build directory.

Apply Wolverine memory/prefetch fixes and its total thread budget; refs/ is unchanged.
"""
import argparse
from pathlib import Path
import re

def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Expected one Wolverine boundary: {old!r}')
    return text.replace(old, new, 1)

def function_range(text, signature):
    start = text.index(signature)
    opening = text.index('{', start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return start, end

def prepare_wolverine(text):
    # loadIndex calls clear() before flags are allocated; upstream leaves the
    # pointer uninitialized and frees a new[] allocation with free(). Fix only
    # the generated benchmark copy, without changing repair edge selection.
    text = replace_once(text, '    bool *deleteFlags;', '    bool *deleteFlags = nullptr;')
    text = replace_once(text, '        free(deleteFlags);',
                        '        delete[] deleteFlags;\n        deleteFlags = nullptr;')
    empty_copy = 'memcpy(result.data(), ll, size * sizeof(tableint));'
    if text.count(empty_copy) != 2:
        raise ValueError('Expected both Wolverine adjacency copy helpers')
    text = text.replace(empty_copy, 'if (size) ' + empty_copy)
    # Cached upper lists have no padding beyond maxM. Upstream lookahead
    # prefetch reads the next ID even at the end of a full list (ASan OOB).
    # Keep prefetching valid neighbors without reading nonexistent IDs.
    guarded = 0
    def bounded_prefetch(match):
        nonlocal guarded
        call = match.group(0)
        condition = None
        for token, bound in [('*(datal + j + 1)', 'j + 1 < size'),
                             ('*(data + j + 1)', 'j < size'),
                             ('*(datal + 1)', 'size > 1'),
                             ('*(data + 1)', 'size > 0'), ('*datal', 'size > 0')]:
            if token in call:
                condition = bound
                break
        if condition:
            guarded += 1
            return f'if ({condition}) ' + call
        return call
    text = re.sub(r'_mm_prefetch\([^;]*\);', bounded_prefetch, text)
    if guarded != 22:
        raise ValueError(f'Unexpected Wolverine prefetch layout: {guarded} neighbor reads')

    # Share one executor between deletion and insertion without retaining an
    # OpenMP team. Include the caller in numThreads (seven helpers for t8).
    signature = '    inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {'
    start, end = function_range(text, signature)
    executor = '''    inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
        if (!numThreads) numThreads = std::max<size_t>(1, std::thread::hardware_concurrency());
        if (start >= end) return;
        std::atomic<size_t> current(start);
        std::exception_ptr lastException;
        std::mutex lastExceptMutex;
        auto worker = [&](size_t threadId) {
            while (true) {
                const size_t id = current.fetch_add(1, std::memory_order_relaxed);
                if (id >= end) break;
                try { fn(id, threadId); }
                catch (...) {
                    std::lock_guard<std::mutex> lock(lastExceptMutex);
                    lastException = std::current_exception();
                    current.store(end, std::memory_order_relaxed);
                    break;
                }
            }
        };
        std::vector<std::thread> threads;
        threads.reserve(numThreads - 1);
        try {
            for (size_t threadId = 0; threadId + 1 < numThreads; ++threadId)
                threads.emplace_back(worker, threadId);
        } catch (...) {
            current.store(end, std::memory_order_relaxed);
            for (auto& thread : threads) thread.join();
            throw;
        }
        worker(numThreads - 1);
        for (auto& thread : threads) thread.join();
        if (lastException) std::rethrow_exception(lastException);
    }'''
    text = text[:start] + executor + text[end:]

    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wolverine", action="store_true")
    parser.add_argument("--overlay", type=Path, help="PRO-HNSW hnswalg.h")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    names = ("hnswlib.h", "hnswalg.h", "space_l2.h", "space_ip.h",
             "visited_list_pool.h", "bruteforce.h", "stop_condition.h")
    for name in names:
        source = args.overlay if name == "hnswalg.h" and args.overlay else args.source / name
        text = source.read_text()
        if name == "hnswalg.h" and args.wolverine:
            text = prepare_wolverine(text)
        target = args.output / name
        if not target.exists() or target.read_text() != text:
            target.write_text(text)


if __name__ == "__main__":
    main()
