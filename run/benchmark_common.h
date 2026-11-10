#pragma once
// Shared main-experiment infrastructure; baseline update algorithms live in their .cpp files.
#define BOOST_BIND_GLOBAL_PLACEHOLDERS
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <cstring>
#include <fstream>
#include <optional>
#include <atomic>
#include <exception>
#include <omp.h>
#include <sched.h>
#include <chrono>
#include <memory>
#include <iomanip>
#include <map>
#include <sstream>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
#include <cstddef>
#include <ctime>

// JSON configuration

namespace paper {
namespace fs = std::filesystem;
using Json = boost::property_tree::ptree;

struct Config {
    std::string engine, dataset, space, workload, name;
    fs::path base, query, output, index_dir, gt_dir, initial_index;
    size_t initial = 0, batch = 0, rounds = 100, update_offset = 0, capacity = 0;
    size_t threads = 8, gt_threads = 8, build_threads = 1, M = 16, ef_c = 200, ef_s = 100;
    size_t queries = 1000, k = 10, seed = 100, interval = 10, theta_in = 32;
    std::vector<int> cpus;
    bool local() const {
        return workload == "localized_update";
    }
    bool cosine() const {
        return space == "cos";
    }
    size_t pool_size() const {
        return initial + batch * rounds;
    }
};

inline fs::path data_path(const std::string &text) {
    if (text.empty())
        return {};
    fs::path p(text);
    return fs::absolute(p.is_absolute() ? p : fs::path(PAPER_ROOT) / p).lexically_normal();
}

inline size_t integer(const Json &json, const char *key, size_t fallback) {
    const auto text = json.get<std::string>(key, std::to_string(fallback));
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument(std::string(key) + " must be a nonnegative integer");
    return std::stoull(text);
}

inline Config read_config(const fs::path &path, const std::string &engine) {
    Json j;
    boost::property_tree::read_json(path.string(), j);
    std::set<std::string> keys = {
        "name",   "engine",        "dataset",        "space",         "workload",   "base",
        "query",  "index_dir",     "gt_dir",         "initial_index", "initial",    "batch",
        "rounds", "update_offset", "index_capacity", "threads",       "gt_threads", "build_threads",
        "M",      "ef_c",          "ef_s",           "queries",       "k",          "seed",
        "cpus"};
    if (engine == "prohnsw")
        keys.insert("maintenance_interval");
    if (engine == "wolverine")
        keys.insert("theta_in");
    std::set<std::string> seen;
    for (const auto &field : j) {
        if (!keys.count(field.first) || !seen.insert(field.first).second)
            throw std::invalid_argument("Unknown or duplicate config field: " + field.first);
    }
    Config c;
    c.engine = j.get<std::string>("engine");
    if (c.engine != engine)
        throw std::invalid_argument("Config engine does not match benchmark_" + engine);
    c.dataset = j.get<std::string>("dataset");
    c.space = j.get<std::string>("space");
    c.workload = j.get<std::string>("workload");
    c.name = j.get<std::string>("name", "");
    c.base = data_path(j.get<std::string>("base"));
    c.query = data_path(j.get<std::string>("query"));
    c.index_dir =
        data_path(j.get<std::string>("index_dir", (c.base.parent_path() / "index").string()));
    c.gt_dir =
        data_path(j.get<std::string>("gt_dir", (c.base.parent_path() / "ground_truth").string()));
    c.initial_index = data_path(j.get<std::string>("initial_index", ""));
    c.initial = integer(j, "initial", 0);
    c.batch = integer(j, "batch", 0);
    c.rounds = integer(j, "rounds", 100);
    c.update_offset = integer(j, "update_offset", c.initial);
    c.capacity = integer(j, "index_capacity", c.initial);
    c.threads = integer(j, "threads", 8);
    c.gt_threads = integer(j, "gt_threads", c.threads);
    c.build_threads = integer(j, "build_threads", 1);
    c.M = integer(j, "M", 16);
    c.ef_c = integer(j, "ef_c", 200);
    c.ef_s = integer(j, "ef_s", 100);
    c.queries = integer(j, "queries", 1000);
    c.k = integer(j, "k", 10);
    c.seed = integer(j, "seed", 100);
    c.interval = integer(j, "maintenance_interval", 10);
    c.theta_in = integer(j, "theta_in", 32);
    if (auto list = j.get_child_optional("cpus"))
        for (const auto &entry : *list) {
            if (!entry.first.empty())
                throw std::invalid_argument("cpus must be an array");
            const int cpu = entry.second.get_value<int>();
            if (cpu < 0)
                throw std::invalid_argument("cpus must contain nonnegative CPU IDs");
            c.cpus.push_back(cpu);
        }
    if (c.space != "l2" && c.space != "cos")
        throw std::invalid_argument("space must be l2 or cos");
    if (c.workload != "random_update" && !c.local())
        throw std::invalid_argument("Unsupported workload");
    if (!c.initial || !c.batch || c.batch > c.initial || !c.queries || !c.k || c.k > c.initial ||
        c.M < 2 || c.M > 10000 || !c.ef_c || !c.ef_s || !c.interval || !c.theta_in || !c.threads ||
        !c.gt_threads || !c.build_threads || c.capacity < c.initial || c.update_offset < c.initial)
        throw std::invalid_argument("Invalid size, thread budget or graph parameter");
    if (c.rounds > (std::numeric_limits<size_t>::max() - c.initial) / c.batch ||
        c.pool_size() > static_cast<size_t>(std::numeric_limits<int32_t>::max()))
        throw std::invalid_argument("Logical labels exceed the ivecs GT format");
    if (c.threads > 1024 || c.gt_threads > 1024 || c.build_threads > 1024)
        throw std::invalid_argument("Thread count is too large");
    if (engine == "vigor" && c.build_threads != 1)
        throw std::invalid_argument("VIGOR initial construction uses one caller");
    if (!c.cpus.empty() && std::set<int>(c.cpus.begin(), c.cpus.end()).size() != c.cpus.size())
        throw std::invalid_argument("Duplicate CPU IDs");
    if (!c.cpus.empty() && c.cpus.size() < std::max({c.threads, c.gt_threads, c.build_threads}))
        throw std::invalid_argument("cpus must cover every requested thread budget");
    return c;
}

inline fs::path metrics_path(const Config &c, std::string name) {
    if (name.empty()) {
        const auto now = std::chrono::system_clock::now();
        const auto seconds = std::chrono::system_clock::to_time_t(now);
        std::tm local{};
        localtime_r(&seconds, &local);
        const auto micros =
            std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count() %
            1000000;
        std::ostringstream generated;
        generated << c.engine << '_' << c.dataset << '_' << c.initial << '_' << c.batch << '_'
                  << c.rounds << '_' << c.workload << '_' << std::put_time(&local, "%Y%m%d_%H%M%S")
                  << '_' << std::setw(6) << std::setfill('0') << micros;
        name = generated.str();
    }
    if (name == "." || name == ".." || name.find_first_of("/\\") != std::string::npos)
        throw std::invalid_argument("--name must be a single directory name");
    return fs::path(PAPER_ROOT) / "exp_results" / name / "metric.csv";
}

inline std::ofstream create_metrics(const fs::path &path) {
    fs::create_directories(path.parent_path());
    const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0664);
    if (fd < 0)
        throw std::runtime_error("Cannot create metric.csv (it may already exist): " +
                                 path.string());
    close(fd);
    std::ofstream out(path);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out << std::fixed << std::setprecision(6);
    return out;
}
} // namespace paper

// fvecs/fbin input and normalization

struct DenseVectors {
    size_t count{0};
    size_t dim{0};
    std::vector<float> values;

    const float *get(size_t id) const {
        return values.data() + id * dim;
    }
    float *getMutable(size_t id) {
        return values.data() + id * dim;
    }

    void reserve(size_t vectors) {
        values.reserve(vectors * dim);
    }

    size_t append(const float *data, size_t vectors) {
        const size_t begin = count;
        values.insert(values.end(), data, data + vectors * dim);
        count += vectors;
        return begin;
    }
};

struct VectorBatch {
    size_t source_begin{0};
    size_t count{0};
    size_t dim{0};
    std::vector<float> values;

    const float *get(size_t id) const {
        return values.data() + id * dim;
    }
};

// Optional physical layout of a logical input stream:
// [0, initial_count) followed by [update_offset, EOF). Logical IDs stay dense.
struct VectorInputLayout {
    size_t initial_count;
    size_t update_offset;
};

class VectorReader {
  public:
    VectorReader(const std::string &filename, size_t max_vectors = 0,
                 std::optional<VectorInputLayout> layout = std::nullopt)
        : filename_(filename), input_(filename, std::ios::binary), layout_(layout) {
        if (!input_.is_open())
            throw std::runtime_error("Failed to open file: " + filename);
        std::string extension = std::filesystem::path(filename).extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        fbin_ = extension == ".fbin";
        size_t declared_count = 0;
        if (fbin_) {
            uint32_t header[2] = {};
            input_.read(reinterpret_cast<char *>(header), sizeof(header));
            if (!input_ || header[1] == 0)
                throw std::runtime_error("Invalid fbin header: " + filename);
            declared_count = header[0];
            dim_ = header[1];
            header_bytes_ = sizeof(header);
        } else {
            int32_t record_dim = 0;
            input_.read(reinterpret_cast<char *>(&record_dim), sizeof(record_dim));
            if (!input_ || record_dim <= 0)
                throw std::runtime_error("Invalid fvecs header: " + filename);
            dim_ = static_cast<size_t>(record_dim);
        }
        if (dim_ > (std::numeric_limits<size_t>::max() - sizeof(int32_t)) / sizeof(float))
            throw std::runtime_error("Vector dimension overflow: " + filename);
        record_bytes_ = dim_ * sizeof(float) + (fbin_ ? 0 : sizeof(int32_t));
        input_.seekg(0, std::ios::end);
        const std::streamoff file_size = input_.tellg();
        if (file_size < static_cast<std::streamoff>(header_bytes_) ||
            (static_cast<uint64_t>(file_size) - header_bytes_) % record_bytes_ != 0)
            throw std::runtime_error("Malformed vector file: " + filename);
        const size_t total = (static_cast<uint64_t>(file_size) - header_bytes_) / record_bytes_;
        if (fbin_ && total != declared_count)
            throw std::runtime_error("fbin count does not match file size: " + filename);
        size_t available = total;
        if (layout_) {
            if (layout_->initial_count > total || layout_->update_offset < layout_->initial_count ||
                layout_->update_offset > total)
                throw std::invalid_argument("Invalid initial/update vector ranges: " + filename);
            available = layout_->initial_count + (total - layout_->update_offset);
        }
        count_ = max_vectors > 0 ? std::min(max_vectors, available) : available;
        seekRow(0);
    }

    size_t dimension() const {
        return dim_;
    }
    size_t count() const {
        return count_;
    }

    std::optional<VectorBatch> readBatch(size_t max_count) {
        if (max_count == 0)
            throw std::invalid_argument("vector batch size must be positive");
        if (next_ >= count_)
            return std::nullopt;
        size_t batch_count = std::min(max_count, count_ - next_);
        if (layout_ && layout_->update_offset != layout_->initial_count) {
            if (next_ < layout_->initial_count)
                batch_count = std::min(batch_count, layout_->initial_count - next_);
            else if (next_ == layout_->initial_count)
                seekRow(layout_->update_offset);
        }
        VectorBatch batch;
        batch.source_begin = next_;
        batch.count = batch_count;
        batch.dim = dim_;
        batch.values.resize(batch_count * dim_);
        if (fbin_) {
            // fbin has one header, then contiguous floats: read directly into
            // the queue payload, without a raw buffer or per-vector copies.
            readBytes(reinterpret_cast<char *>(batch.values.data()),
                      batch.values.size() * sizeof(float));
        } else {
            raw_buffer_.resize(batch_count * record_bytes_);
            readBytes(raw_buffer_.data(), raw_buffer_.size());
            for (size_t i = 0; i < batch_count; ++i) {
                int32_t record_dim = 0;
                const char *record = raw_buffer_.data() + i * record_bytes_;
                std::memcpy(&record_dim, record, sizeof(record_dim));
                if (record_dim <= 0 || static_cast<size_t>(record_dim) != dim_)
                    throw std::runtime_error("Dimension mismatch in fvecs file: " + filename_);
                std::memcpy(batch.values.data() + i * dim_, record + sizeof(record_dim),
                            dim_ * sizeof(float));
            }
        }
        next_ += batch_count;
        return batch;
    }

  private:
    void seekRow(size_t row) {
        // Only validated in-file rows reach here, so this fits the file offset.
        input_.clear();
        input_.seekg(static_cast<std::streamoff>(header_bytes_ + row * record_bytes_),
                     std::ios::beg);
        if (!input_)
            throw std::runtime_error("Failed to seek vector file: " + filename_);
    }

    void readBytes(char *destination, size_t bytes) {
        if (bytes > static_cast<size_t>(std::numeric_limits<std::streamsize>::max()))
            throw std::runtime_error("Vector batch exceeds stream size limit");
        input_.read(destination, static_cast<std::streamsize>(bytes));
        if (!input_)
            throw std::runtime_error("Truncated vector file: " + filename_);
    }

    std::string filename_;
    std::ifstream input_;
    std::optional<VectorInputLayout> layout_;
    bool fbin_{false};
    size_t header_bytes_{0};
    size_t record_bytes_{0};
    size_t dim_{0};
    size_t count_{0};
    size_t next_{0};
    std::vector<char> raw_buffer_;
};

inline void normalize_batch(VectorBatch &batch) {
    for (size_t i = 0; i < batch.count; ++i) {
        float *vector = batch.values.data() + i * batch.dim;
        double norm2 = 0.0;
        for (size_t d = 0; d < batch.dim; ++d)
            norm2 += static_cast<double>(vector[d]) * vector[d];
        if (norm2 <= 0.0)
            continue;
        const float inverse = 1.0f / static_cast<float>(std::sqrt(norm2));
        for (size_t d = 0; d < batch.dim; ++d)
            vector[d] *= inverse;
    }
}

inline DenseVectors load_vectors_flat(const std::string &filename, size_t &dim,
                                      size_t max_vectors = 0,
                                      std::optional<VectorInputLayout> layout = std::nullopt) {
    VectorReader reader(filename, max_vectors, layout);
    dim = reader.dimension();
    DenseVectors result;
    result.dim = dim;
    result.reserve(reader.count());
    while (auto batch = reader.readBatch(4096))
        result.append(batch->values.data(), batch->count);
    return result;
}

inline void normalize_vectors(DenseVectors &data) {
    for (size_t i = 0; i < data.count; ++i) {
        float *vector = data.getMutable(i);
        double norm2 = 0.0;
        for (size_t d = 0; d < data.dim; ++d)
            norm2 += static_cast<double>(vector[d]) * vector[d];
        if (norm2 <= 0.0)
            continue;
        const float inverse = 1.0f / static_cast<float>(std::sqrt(norm2));
        for (size_t d = 0; d < data.dim; ++d)
            vector[d] *= inverse;
    }
}

// CPU affinity and caller thread budgets

namespace paper {
inline void set_affinity(const std::vector<int> &cpus) {
    if (cpus.empty())
        return;
#if defined(__linux__)
    cpu_set_t allowed, selected;
    if (sched_getaffinity(0, sizeof(allowed), &allowed))
        throw std::runtime_error("Cannot read CPU affinity");
    CPU_ZERO(&selected);
    for (int cpu : cpus) {
        if (cpu >= CPU_SETSIZE || !CPU_ISSET(cpu, &allowed))
            throw std::runtime_error("CPU outside inherited affinity");
        CPU_SET(cpu, &selected);
    }
    if (sched_setaffinity(0, sizeof(selected), &selected))
        throw std::runtime_error("Cannot set CPU affinity");
#else
    throw std::runtime_error("cpus requires Linux");
#endif
}

// Includes the caller; there is no nested prune team or background update worker.
template <class F> void parallel_for(size_t n, size_t threads, F fn) {
    if (threads == 1) {
        for (size_t i = 0; i < n; ++i)
            fn(i);
        return;
    }
    std::exception_ptr error;
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1)
    for (size_t i = 0; i < n; ++i) {
        try {
            fn(i);
        } catch (...) {
#pragma omp critical(paper_update_error)
            {
                if (!error)
                    error = std::current_exception();
            }
        }
    }
    if (error)
        std::rethrow_exception(error);
}

inline void prepare_team(size_t threads) {
    omp_set_dynamic(0);
    omp_set_num_threads(static_cast<int>(threads));
    int actual = 0;
#pragma omp parallel num_threads(threads)
    {
#pragma omp single
        actual = omp_get_num_threads();
    }
    if (actual != static_cast<int>(threads))
        throw std::runtime_error("OpenMP could not create the requested team");
}
} // namespace paper

// Common measurements

namespace paper {
struct Timer {
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    double seconds() const {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }
};
struct Graph {
    size_t total = 0, valid = 0;
    double degree = 0;
};
struct Stages {
    double remove_s = 0, repair_s = 0, resolve_s = 0;
};
} // namespace paper

// Parameter-matched initial caches

namespace paper {
using Fields = std::map<std::string, std::string>;
inline Fields read_fields(const fs::path &path) {
    std::ifstream f(path);
    Fields result;
    for (std::string line; std::getline(f, line);) {
        const auto split = line.find('=');
        if (split != std::string::npos)
            result[line.substr(0, split)] = line.substr(split + 1);
    }
    return result;
}
inline void write_fields(const fs::path &path, const Fields &fields) {
    const auto temp = path.string() + ".tmp." + std::to_string(getpid());
    std::ofstream out(temp);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    for (const auto &item : fields)
        out << item.first << '=' << item.second << '\n';
    out.close();
    fs::rename(temp, path);
}
// Different native baselines may request the same absent initial cache.
// Serialize construction, then let all readers reuse the completed graph.
class CacheLock {
    int fd_ = -1;

  public:
    explicit CacheLock(const fs::path &path) {
        fs::create_directories(path.parent_path());
        fd_ = open((path.string() + ".lock").c_str(), O_CREAT | O_RDWR, 0664);
        if (fd_ < 0)
            throw std::runtime_error("Cannot open initial-cache lock");
        if (flock(fd_, LOCK_EX)) {
            close(fd_);
            throw std::runtime_error("Cannot lock initial cache");
        }
    }
    ~CacheLock() {
        if (fd_ >= 0)
            close(fd_);
    }
    CacheLock(const CacheLock &) = delete;
    CacheLock &operator=(const CacheLock &) = delete;
};
inline bool same_fields(const Fields &saved, const Fields &wanted) {
    for (const auto &item : wanted) {
        auto found = saved.find(item.first);
        if (found == saved.end())
            return false;
        if (item.first == "base" || item.first == "query") {
            if (data_path(found->second.substr(0, found->second.find('|'))) !=
                data_path(item.second))
                return false;
        } else if (found->second != item.second)
            return false;
    }
    return true;
}
inline Fields index_fields(const Config &c, size_t dim, bool native) {
    return {{"format", native ? "paper-native-v1" : "vigor-initial-v5"},
            {"base", c.base.string()},
            {"space", c.space},
            {"dim", std::to_string(dim)},
            {"initial", std::to_string(c.initial)},
            {"capacity", std::to_string(c.capacity)},
            {"M", std::to_string(c.M)},
            {"ef_c", std::to_string(c.ef_c)},
            {"seed", std::to_string(c.seed)},
            {"construction_threads", std::to_string(c.build_threads)},
            {"construction_simd", "SSE"},
            {"construction_fma", "0"},
            {"construction_fast_math", "0"}};
}
inline bool index_matches(const fs::path &path, const Config &c, size_t dim, bool native,
                          bool explicit_path = false) {
    if (!fs::exists(path) || (native && !fs::exists(path.string() + ".rng")))
        return false;
    auto saved = read_fields(path.string() + ".meta"), wanted = index_fields(c, dim, native);
    auto format = saved["format"];
    if (native) {
        if (format != "paper-native-v1" && format != "native-initial-v2" &&
            format != "native-initial-parallel-v1")
            return false;
    } else if (format != "vigor-initial-v5")
        return false;
    wanted.erase("format");
    if (explicit_path && !saved.count("construction_simd")) {
        // Migrated v5 caches may not record the old compiler. No source/hash check.
        wanted.erase("construction_simd");
        wanted.erase("construction_fma");
        wanted.erase("construction_fast_math");
    }
    if (saved.count("construction_simd")) {
        auto &isa = saved["construction_simd"];
        std::transform(isa.begin(), isa.end(), isa.begin(),
                       [](unsigned char v) { return std::toupper(v); });
    }
    return same_fields(saved, wanted);
}
inline fs::path index_path(const Config &c, size_t dim, bool native) {
    std::ostringstream name;
    name << (native ? "hnsw" : "vigor") << '_' << c.dataset << '_' << c.space << "_d" << dim << "_n"
         << c.initial << "_cap" << c.capacity << "_M" << c.M << "_ef" << c.ef_c << "_seed" << c.seed
         << "_t" << c.build_threads << "_sse.bin";
    return c.index_dir / name.str();
}
inline fs::path find_index(const Config &c, size_t dim, bool native) {
    if (!c.initial_index.empty() && fs::exists(c.initial_index)) {
        if (!index_matches(c.initial_index, c, dim, native, true))
            throw std::runtime_error("Initial cache parameters do not match config: " +
                                     c.initial_index.string());
        return c.initial_index;
    }
    const auto preferred = index_path(c, dim, native);
    if (index_matches(preferred, c, dim, native))
        return preferred;
    if (!fs::exists(c.index_dir))
        return {};
    std::vector<fs::path> candidates;
    for (const auto &entry : fs::directory_iterator(c.index_dir)) {
        if (entry.path().extension() != ".meta")
            continue;
        auto binary = entry.path();
        binary.replace_extension("");
        if (index_matches(binary, c, dim, native))
            candidates.push_back(binary);
    }
    std::sort(candidates.begin(), candidates.end());
    return candidates.empty() ? fs::path{} : candidates.front();
}
} // namespace paper

// Replacement traces and shared ground truth

namespace paper {
// Labels are dense, never reused, and identify positions in the logical input.
// Physical vector rows may use a separate offset (Deep100M: 100M).
struct Plan {
    std::vector<size_t> removed, added;
    size_t anchor = std::numeric_limits<size_t>::max();
};
inline float distance(const float *a, const float *b, size_t dim, bool cosine) {
    float result = 0;
    for (size_t d = 0; d < dim; ++d) {
        if (cosine)
            result += a[d] * b[d];
        else {
            const float x = a[d] - b[d];
            result += x * x;
        }
    }
    return cosine ? 1 - result : result;
}
inline Plan plan_update(const Config &c, size_t round, std::vector<size_t> &labels,
                        const DenseVectors &pool, std::mt19937 &rng) {
    Plan p;
    if (c.local()) {
        std::uniform_int_distribution<size_t> pick(0, labels.size() - 1);
        p.anchor = labels[pick(rng)];
        std::vector<std::pair<float, size_t>> distances;
        distances.reserve(labels.size());
        const auto *anchor = pool.get(p.anchor);
        for (auto label : labels)
            distances.emplace_back(distance(anchor, pool.get(label), pool.dim, c.cosine()), label);
        std::partial_sort(distances.begin(), distances.begin() + c.batch, distances.end());
        for (size_t i = 0; i < c.batch; ++i)
            p.removed.push_back(distances[i].second);
    } else {
        std::shuffle(labels.begin(), labels.end(),
                     std::mt19937(static_cast<unsigned>(c.seed + round)));
        p.removed.assign(labels.begin(), labels.begin() + c.batch);
    }
    const size_t first = c.initial + (round - 1) * c.batch;
    for (size_t i = 0; i < c.batch; ++i)
        p.added.push_back(first + i);
    return p;
}
inline void finish_plan(const Plan &p, std::vector<size_t> &labels,
                        const std::vector<uint8_t> &active) {
    labels.erase(
        std::remove_if(labels.begin(), labels.end(), [&](size_t label) { return !active[label]; }),
        labels.end());
    labels.insert(labels.end(), p.added.begin(), p.added.end());
}
using Truth = std::vector<std::vector<size_t>>;
inline Truth compute_truth(const Config &c, const DenseVectors &pool, const DenseVectors &queries,
                           const std::vector<uint8_t> &active) {
    Truth truth(c.queries);
#pragma omp parallel for num_threads(c.gt_threads) schedule(static)
    for (size_t q = 0; q < c.queries; ++q) {
        std::priority_queue<std::pair<float, size_t>> best;
        for (size_t id = 0; id < pool.count; ++id)
            if (active[id]) {
                const auto item = std::make_pair(
                    distance(queries.get(q), pool.get(id), pool.dim, c.cosine()), id);
                if (best.size() < c.k)
                    best.push(item);
                else if (item < best.top()) {
                    best.pop();
                    best.push(item);
                }
            }
        auto &row = truth[q];
        row.resize(best.size());
        for (size_t i = row.size(); i > 0; --i) {
            row[i - 1] = best.top().second;
            best.pop();
        }
    }
    return truth;
}

class GroundTruth {
    const Config &c_;
    fs::path directory_;
    Fields fields_;
    fs::path round_file(const fs::path &variant, size_t round) const {
        std::ostringstream name;
        name << "round_" << std::setw(6) << std::setfill('0') << round << ".ivecs";
        return variant / name.str();
    }
    bool compatible(const fs::path &directory) const {
        const auto saved = read_fields(directory / "manifest.txt");
        auto expected = fields_;
        if (!saved.count("update_offset") && c_.update_offset == c_.initial)
            expected.erase("update_offset");
        // Total rounds/pool do not affect the active set in a shared prefix.
        // Old replacement manifests may also contain these fixed flags.
        return same_fields(saved, expected) &&
               (!saved.count("delete") || saved.at("delete") == "1") &&
               (!saved.count("append") || saved.at("append") == "1") &&
               (!saved.count("reuse_label") || saved.at("reuse_label") == "0");
    }

  public:
    GroundTruth(const Config &c, size_t dim) : c_(c) {
        fields_ = {{"base", c.base.string()},
                   {"query", c.query.string()},
                   {"space", c.space},
                   {"dim", std::to_string(dim)},
                   {"initial", std::to_string(c.initial)},
                   {"batch", std::to_string(c.batch)},
                   {"seed", std::to_string(c.seed)},
                   {"update_offset", std::to_string(c.update_offset)},
                   {"workload", c.local() ? "localized" : "dynamic"}};
        std::ostringstream name;
        name << c.workload << '_' << c.dataset << '_' << c.space << "_d" << dim << "_n" << c.initial
             << "_b" << c.batch << "_offset" << c.update_offset << "_seed" << c.seed;
        directory_ = c.gt_dir / name.str();
        if (fs::exists(directory_ / "manifest.txt") && !compatible(directory_))
            throw std::runtime_error("GT cache parameters do not match config");
    }
    const fs::path &directory() const {
        return directory_;
    }
    Truth get(size_t round, const DenseVectors &pool, const DenseVectors &queries,
              const std::vector<uint8_t> &active) {
        if (round == 0 && c_.local()) {
            Config random = c_;
            random.workload = "random_update";
            return GroundTruth(random, pool.dim).get(0, pool, queries, active);
        }
        fs::path cached;
        std::pair<size_t, size_t> smallest = {SIZE_MAX, SIZE_MAX};
        if (fs::exists(c_.gt_dir))
            for (const auto &trace : fs::directory_iterator(c_.gt_dir)) {
                if (!trace.is_directory() || !compatible(trace.path()))
                    continue;
                for (const auto &entry : fs::directory_iterator(trace.path())) {
                    if (!entry.is_directory())
                        continue;
                    size_t q = 0, k = 0;
                    char extra = 0;
                    if (std::sscanf(entry.path().filename().c_str(), "q%zu_k%zu%c", &q, &k,
                                    &extra) != 2)
                        continue;
                    const auto file = round_file(entry.path(), round);
                    if (q >= c_.queries && k >= c_.k && fs::exists(file) &&
                        std::make_pair(q, k) < smallest) {
                        cached = file;
                        smallest = {q, k};
                    }
                }
            }
        if (!cached.empty()) {
            if (round == 0 || round % 10 == 0 || round == c_.rounds)
                std::cout << "Using GT round " << round << " from " << cached << '\n';
            std::ifstream in(cached, std::ios::binary);
            Truth result(c_.queries);
            for (auto &row : result) {
                int32_t count = 0;
                in.read(reinterpret_cast<char *>(&count), 4);
                if (count < static_cast<int32_t>(c_.k))
                    throw std::runtime_error("Invalid GT row: " + cached.string());
                row.resize(c_.k);
                for (auto &label : row) {
                    int32_t value = -1;
                    in.read(reinterpret_cast<char *>(&value), 4);
                    if (value < 0)
                        throw std::runtime_error("Invalid GT label");
                    label = static_cast<size_t>(value);
                }
                in.seekg((count - static_cast<int32_t>(c_.k)) * 4, std::ios::cur);
                if (!in)
                    throw std::runtime_error("Truncated GT: " + cached.string());
            }
            return result;
        }
        std::cout << "Computing GT round " << round << " with " << c_.gt_threads << " threads\n";
        auto result = compute_truth(c_, pool, queries, active);
        const auto variant =
            directory_ / ("q" + std::to_string(c_.queries) + "_k" + std::to_string(c_.k));
        fs::create_directories(variant);
        const auto target = round_file(variant, round),
                   temporary = fs::path(target.string() + ".tmp." + std::to_string(getpid()));
        std::ofstream out(temporary, std::ios::binary);
        out.exceptions(std::ios::badbit | std::ios::failbit);
        for (const auto &row : result) {
            const int32_t count = row.size();
            out.write(reinterpret_cast<const char *>(&count), 4);
            for (auto label : row) {
                const int32_t value = label;
                out.write(reinterpret_cast<const char *>(&value), 4);
            }
        }
        out.close();
        fs::rename(temporary, target);
        if (!fs::exists(directory_ / "manifest.txt"))
            write_fields(directory_ / "manifest.txt", fields_);
        return result;
    }
};
} // namespace paper

// Canonical native construction, implemented in a separate translation unit

namespace paper {
// One canonical constructor for all native baselines. The implementation lives
// in its own namespace/TU so vendor inline symbols cannot override one another.
class NativeInitialIndex {
    struct Impl;
    std::unique_ptr<Impl> impl_;

  public:
    NativeInitialIndex(bool cosine, size_t dim, size_t capacity, size_t M, size_t ef_c,
                       size_t seed);
    ~NativeInitialIndex();
    void add(const float *vector, size_t label);
    void save(const std::string &path);
};
} // namespace paper

#ifdef PAPER_NATIVE_BENCHMARK
#include "hnswlib/hnswlib.h"

// Common native-index loading, construction and graph measurements

namespace paper {
template <bool PhysicalDelete = false, class Index = hnswlib::HierarchicalNSW<float>>
class NativeEngineBase {
    std::unique_ptr<hnswlib::SpaceInterface<float>> space_;
    std::unique_ptr<NativeInitialIndex> initial_;
    size_t dim_;

  public:
    static constexpr bool native = true;
    std::unique_ptr<Index> index;
    NativeEngineBase(const Config &c, size_t dim) : dim_(dim) {
        if (c.cosine())
            space_ = std::make_unique<hnswlib::InnerProductSpace>(dim);
        else
            space_ = std::make_unique<hnswlib::L2Space>(dim);
    }
    void create(const Config &c) {
        initial_ =
            std::make_unique<NativeInitialIndex>(c.cosine(), dim_, c.capacity, c.M, c.ef_c, c.seed);
    }
    void load(const fs::path &path, const Config &c) {
        index.reset();
        index = std::make_unique<Index>(space_.get(), path.string(), false, c.capacity, true);
        std::ifstream state(path.string() + ".rng");
        if (!(state >> std::ws >> index->level_generator_ >> std::ws >>
              index->update_probability_generator_))
            throw std::runtime_error("Missing/invalid cached RNG states");
        if (index->getCurrentElementCount() != c.initial || index->getDeletedCount())
            throw std::runtime_error("Unexpected cached live corpus");
    }
    void add_initial(const float *vector, size_t label) {
        initial_->add(vector, label);
    }
    void save_initial(const fs::path &path) {
        initial_->save(path.string());
    }
    void finish_initial(const fs::path &path, const Config &c) {
        initial_.reset();
        load(path, c);
    }
    void configure(const Config &c) {
        index->setEf(c.ef_s);
    }
    Graph graph() const {
        Graph g;
        g.total = index->getCurrentElementCount();
        size_t edges = 0;
        for (size_t id = 0; id < g.total; ++id) {
            if (index->isMarkedDeleted(id))
                continue;
            if constexpr (PhysicalDelete)
                if (index->deleteFlags[id])
                    continue;
            ++g.valid;
            auto *list = index->get_linklist0(id);
            auto *neighbors = reinterpret_cast<hnswlib::tableint *>(list + 1);
            for (size_t j = 0; j < index->getListCount(list); ++j) {
                if (index->isMarkedDeleted(neighbors[j]))
                    continue;
                if constexpr (PhysicalDelete)
                    if (index->deleteFlags[neighbors[j]])
                        continue;
                ++edges;
            }
        }
        if (g.valid)
            g.degree = double(edges) / g.valid;
        return g;
    }
};
} // namespace paper
#endif

// Shared evaluation loop and CSV output

namespace paper {
inline constexpr const char *csv_header =
    "batch,engine,anchor,recall,update_total_s,avg_deg,total_cnt,valid_cnt,valid_ratio,expand,"
    "checked,stale_hit,stale_ratio,dist_comp,threads,remove_s,repair_s,resolve_s";

template <class Engine> int execute(int argc, char **argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::cout << "benchmark_" << Engine::name() << " CONFIG.json [--name NAME] [--check]\n";
        return 0;
    }
    if (argc < 2)
        throw std::invalid_argument("Usage: benchmark_" + std::string(Engine::name()) +
                                    " CONFIG.json [--name NAME] [--check]");
    bool check = false, named = false;
    std::string name;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--check") {
            check = true;
            continue;
        }
        if (arg == "--name" || arg.rfind("--name=", 0) == 0) {
            if (named)
                throw std::invalid_argument("--name specified more than once");
            named = true;
            if (arg == "--name") {
                if (i + 1 >= argc || std::string(argv[i + 1]).rfind("--", 0) == 0)
                    throw std::invalid_argument("--name requires a directory name");
                name = argv[++i];
            } else {
                name = arg.substr(7);
            }
            if (name.empty())
                throw std::invalid_argument("--name must not be empty");
            continue;
        }
        throw std::invalid_argument("Unsupported option: " + arg);
    }
    Config c = read_config(argv[1], Engine::name());
    c.output = metrics_path(c, named ? name : c.name);
    VectorReader reader(c.base.string(), c.pool_size(),
                        VectorInputLayout{c.initial, c.update_offset});
    if (reader.count() < c.pool_size())
        throw std::runtime_error("Insufficient base/update vectors");
    const size_t dim = reader.dimension();
    VectorReader query_info(c.query.string(), c.queries);
    if (query_info.count() != c.queries || query_info.dimension() != dim)
        throw std::runtime_error("Query dimensions/count mismatch");
    auto cached = find_index(c, dim, Engine::native);
    GroundTruth gt(c, dim);
    std::cout << c.engine << " / " << c.dataset << " / " << c.workload << " | threads=" << c.threads
              << " | initial=" << c.initial << " | batch=" << c.batch << " | rounds=" << c.rounds
              << " | update_offset=" << c.update_offset << " | M=" << c.M << " ef_c=" << c.ef_c
              << " ef_s=" << c.ef_s << '\n'
              << "Initial cache: " << (cached.empty() ? "miss " : "hit ")
              << (cached.empty() ? index_path(c, dim, Engine::native) : cached) << '\n'
              << "GT cache: " << gt.directory() << '\n'
              << "Metrics: " << c.output.string() << std::endl;
    if constexpr (!Engine::native)
        std::cout << "Repair: target-start search, ef_r=50, c=3; serial one-shot prune\n";
    if (c.engine == "wolverine")
        std::cout << "Wolverine++: delete model 4, theta_in=" << c.theta_in << "; caller + "
                  << c.threads - 1 << " helpers\n";
    if (c.engine == "prohnsw")
        std::cout << "PRO-HNSW: ROE each batch, RDN/RUE interval=" << c.interval << '\n';
    if (check)
        return 0;
    auto csv = create_metrics(c.output);
    csv << csv_header << '\n';
    csv.flush();
    set_affinity(c.cpus);
    omp_set_dynamic(0);
    std::unique_ptr<CacheLock> cache_lock;
    if (cached.empty()) {
        cache_lock = std::make_unique<CacheLock>(index_path(c, dim, Engine::native));
        cached = find_index(c, dim, Engine::native);
    }
    DenseVectors pool;
    pool.dim = dim;
    pool.reserve(c.pool_size());
    Engine engine(c, dim);
    if (cached.empty())
        engine.create(c);
    else
        engine.load(cached, c);
    for (size_t loaded = 0; loaded < c.initial;) {
        auto chunk = reader.readBatch(std::min<size_t>(65536, c.initial - loaded));
        if (!chunk)
            throw std::runtime_error("Truncated initial vectors");
        if (c.cosine())
            normalize_batch(*chunk);
        pool.append(chunk->values.data(), chunk->count);
        if (cached.empty()) {
            size_t first = loaded;
            if (first == 0) {
                engine.add_initial(pool.get(0), 0);
                ++first;
            }
            parallel_for(pool.count - first, c.build_threads, [&](size_t i) {
                const size_t label = first + i;
                engine.add_initial(pool.get(label), label);
            });
        }
        loaded = pool.count;
        if (loaded == c.initial || loaded % (65536 * 16) == 0)
            std::cout << "Initial vectors " << loaded << '/' << c.initial << std::endl;
    }
    fs::path initial = cached;
    if (cached.empty()) {
        initial = index_path(c, dim, Engine::native);
        if (fs::exists(initial))
            throw std::runtime_error("Cache filename exists with different/incomplete metadata: " +
                                     initial.string());
        fs::create_directories(c.index_dir);
        const fs::path temp = initial.string() + ".tmp";
        engine.save_initial(temp);
        fs::rename(temp, initial);
        if constexpr (Engine::native)
            fs::rename(temp.string() + ".rng", initial.string() + ".rng");
        write_fields(initial.string() + ".meta", index_fields(c, dim, Engine::native));
        engine.finish_initial(initial, c);
        std::cout << "Saved initial cache " << initial << std::endl;
    }
    cache_lock.reset();
    engine.configure(c);
    size_t query_dim = 0;
    auto queries = load_vectors_flat(c.query.string(), query_dim, c.queries);
    if (c.cosine())
        normalize_vectors(queries);
    std::vector<uint8_t> active(c.pool_size(), 0);
    std::fill_n(active.begin(), c.initial, 1);
    std::vector<size_t> labels(c.initial);
    std::iota(labels.begin(), labels.end(), 0);
    auto evaluate = [&](size_t batch, size_t anchor, double elapsed, const Stages &stages) {
        const auto truth = gt.get(batch, pool, queries, active);
        const auto graph = engine.graph();
        size_t hits = 0;
        for (size_t q = 0; q < c.queries; ++q) {
            auto result = engine.index->searchKnn(queries.get(q), c.k);
            while (!result.empty()) {
                const auto label = result.top().second;
                result.pop();
                hits += std::find(truth[q].begin(), truth[q].end(), label) != truth[q].end();
            }
        }
        const double recall = double(hits) / (c.queries * c.k);
        csv << batch << ',' << Engine::name() << ',';
        if (anchor != std::numeric_limits<size_t>::max())
            csv << anchor;
        csv << ',' << recall << ',' << elapsed << ',' << graph.degree << ',' << graph.total << ','
            << graph.valid << ',' << double(graph.valid) / graph.total;
        csv << ",,,,,"; // Query work counters are not collected.
        csv << ',' << c.threads
            << ',' << stages.remove_s << ',' << stages.repair_s << ',' << stages.resolve_s << '\n';
        csv.flush();
        if (batch == 0 || batch % 10 == 0 || batch == c.rounds)
            std::cout << "Round " << batch << '/' << c.rounds << " | recall=" << recall
                      << " | update_s=" << elapsed << std::endl;
    };
    evaluate(0, std::numeric_limits<size_t>::max(), 0, {});
    // The historical localized workload starts updating the pristine graph.
    // Discard any round-0 stale-edge compaction after recording its initial row.
    if constexpr (!Engine::native)
        if (c.local()) {
            engine.load(initial, c);
            engine.configure(c);
        }
    std::mt19937 rng(static_cast<unsigned>(c.seed));
    for (size_t round = 1; round <= c.rounds; ++round) {
        const auto plan = plan_update(c, round, labels, pool, rng);
        size_t remaining = c.batch;
        while (remaining) {
            auto chunk = reader.readBatch(std::min<size_t>(remaining, 65536));
            if (!chunk)
                throw std::runtime_error("Truncated update vectors");
            if (c.cosine())
                normalize_batch(*chunk);
            pool.append(chunk->values.data(), chunk->count);
            remaining -= chunk->count;
        }
        for (auto label : plan.removed)
            active[label] = 0;
        for (auto label : plan.added)
            active[label] = 1;
        // Wolverine creates its own std::thread team; do not warm an OMP team.
        if (c.engine == "wolverine")
            omp_set_num_threads(static_cast<int>(c.threads));
        else
            prepare_team(c.threads);
        Timer update;
        const auto stages = engine.update(c, plan, pool, round);
        const double seconds = update.seconds();
        finish_plan(plan, labels, active);
        evaluate(round, plan.anchor, seconds, stages);
    }
    return 0;
}
template <class Engine> int main(int argc, char **argv) {
    try {
        return execute<Engine>(argc, argv);
    } catch (const std::exception &e) {
        std::cerr << "benchmark_" << Engine::name() << ": " << e.what() << '\n';
        return 1;
    }
}
} // namespace paper
