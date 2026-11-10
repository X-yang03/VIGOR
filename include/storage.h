#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace vigor {

using tableint = unsigned int;
using labeltype = size_t;

inline constexpr size_t kCacheLineBytes = 64;
static_assert(std::numeric_limits<tableint>::digits == 32, "Edges require a 32-bit tableint");
inline constexpr size_t kGenerationBits = 4;
inline constexpr uint8_t kGenerationMask = (1u << kGenerationBits) - 1;
inline constexpr size_t kNodeIdBits = 32 - kGenerationBits;
inline constexpr tableint kNodeIdMask = (tableint{1} << kNodeIdBits) - 1;
inline constexpr tableint kInvalidTableId = std::numeric_limits<tableint>::max();
// Reserve the highest physical ID so no live packed edge equals the sentinel.
inline constexpr size_t kMaxNodeCapacity = kNodeIdMask;
inline constexpr size_t kMaxElementCapacity = kMaxNodeCapacity * 5 / 6;
inline constexpr uint32_t kIndexFormatVersion = 5; // explicit cached level; 28-bit ID / 4-bit generation

// Maximum snapshot capacity; 255 is reserved for the empty pruned-tail marker.
inline constexpr size_t ALIGNED_M = 254;

struct alignas(8) NodeMetaData {
    std::atomic<tableint> vector_id{kInvalidTableId};
    std::atomic<uint8_t> generation{0};
    std::atomic<uint8_t> size{0};
    std::atomic<uint8_t> is_pruned{0};
    std::atomic<uint8_t> is_deleted{0};
};

struct alignas(8) ColdNodeMetaData {
    std::atomic<tableint> down_node_id{kInvalidTableId};
    // An 8-bit token wraps after 256 mutations; callers must account for this
    // limitation when relying on optimistic snapshot/commit validation.
    std::atomic<uint8_t> mutation_version{0};
    std::atomic<uint8_t> level{0};
    std::atomic_flag lock = ATOMIC_FLAG_INIT;
};

// A non-owning view into the continuous adjacency pool, not per-node metadata.
struct AlignedNode {
    tableint* neighbors;
};

struct ConstAlignedNode {
    const tableint* neighbors;
};

static_assert(sizeof(NodeMetaData) == 8, "Hot metadata must stay compact");
static_assert(sizeof(ColdNodeMetaData) == 8, "Cold metadata must stay compact");
static_assert(std::is_trivially_destructible<NodeMetaData>::value,
              "Aligned pools require trivial metadata destruction");
static_assert(std::is_trivially_destructible<ColdNodeMetaData>::value,
              "Aligned pools require trivial metadata destruction");

inline constexpr tableint packEdge(uint8_t generation, tableint physical_id) noexcept {
    return (static_cast<tableint>(generation & kGenerationMask) << kNodeIdBits) |
           (physical_id & kNodeIdMask);
}

inline constexpr tableint getPhysicalId(tableint edge) noexcept {
    return edge & kNodeIdMask;
}

inline constexpr uint8_t getEdgeGeneration(tableint edge) noexcept {
    return static_cast<uint8_t>(edge >> kNodeIdBits);
}

// Thread-safe reusable allocator shared by node and vector pools.
class IDAllocator {
private:
    std::deque<tableint> free_list_;
    std::atomic<tableint> watermark_{0};
    mutable std::mutex lock_;

public:
    IDAllocator() = default;

    // Disable copy for safety
    IDAllocator(const IDAllocator&) = delete;
    IDAllocator& operator=(const IDAllocator&) = delete;

    tableint alloc() {
        std::lock_guard<std::mutex> guard(lock_);
        if (!free_list_.empty()) {
            tableint id = free_list_.front();
            free_list_.pop_front();
            return id;
        }
        return watermark_.fetch_add(1, std::memory_order_relaxed);
    }

    void free(tableint id) {
        std::lock_guard<std::mutex> guard(lock_);
        free_list_.push_back(id);
    }

    tableint getWatermark() const {
        return watermark_.load(std::memory_order_relaxed);
    }

    int getActualSize() const {
        std::lock_guard<std::mutex> guard(lock_);
        return static_cast<int>(watermark_.load(std::memory_order_relaxed) -
                                free_list_.size());
    }

    size_t getFreeListSize() const {
        std::lock_guard<std::mutex> guard(lock_);
        return free_list_.size();
    }

    std::vector<tableint> getFreeListSnapshot() const {
        std::lock_guard<std::mutex> guard(lock_);
        return std::vector<tableint>(free_list_.begin(), free_list_.end());
    }

    void restore(tableint watermark, const std::vector<tableint>& free_list) {
        std::lock_guard<std::mutex> guard(lock_);
        watermark_.store(watermark, std::memory_order_relaxed);
        free_list_.clear();
        for (tableint id : free_list) {
            if (id >= watermark) {
                throw std::runtime_error("Invalid free-list id while loading index");
            }
            free_list_.push_back(id);
        }
    }

};

namespace detail {

// Advise only whole pages owned by a large pool. The kernel may decline;
// allocation, alignment and the serialized layout remain independent of THP.
inline void adviseHugePages(void* memory, size_t bytes) noexcept {
#if defined(__linux__) && defined(MADV_HUGEPAGE) && !defined(VIGOR_DISABLE_HUGEPAGES)
    if (bytes < 2 * 1024 * 1024) return;
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) return;
    const uintptr_t page = static_cast<uintptr_t>(page_size);
    const uintptr_t begin = reinterpret_cast<uintptr_t>(memory);
    const uintptr_t end = begin + bytes;
    const uintptr_t first = begin + (page - begin % page) % page;
    const uintptr_t last = end - end % page;
    if (last > first)
        (void)madvise(reinterpret_cast<void*>(first), last - first, MADV_HUGEPAGE);
#else
    (void)memory;
    (void)bytes;
#endif
}

inline size_t alignedAllocationSize(size_t bytes) {
    if (bytes == 0 || bytes > std::numeric_limits<size_t>::max() - (kCacheLineBytes - 1)) {
        throw std::runtime_error("Invalid aligned allocation size");
    }
    return (bytes + kCacheLineBytes - 1) & ~(kCacheLineBytes - 1);
}

template<typename T>
T* allocateAligned(size_t count, const char* description) {
    if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
        throw std::runtime_error(std::string("Allocation size overflow for ") + description);
    }
    void* memory = std::aligned_alloc(kCacheLineBytes,
                                      alignedAllocationSize(count * sizeof(T)));
    if (!memory) {
        throw std::runtime_error(std::string("Failed to allocate ") + description);
    }
    return static_cast<T*>(memory);
}

}  // namespace detail

template<typename dist_t>
class AlignedStorage {
public:
    // Query-hot topology and metadata live in separate aligned pools.
    tableint* node_pool_ = nullptr;
    NodeMetaData* metadata_pool_ = nullptr;
    ColdNodeMetaData* cold_metadata_pool_ = nullptr;

    char* vector_data_pool_ = nullptr;

    std::unordered_map<labeltype, tableint> label_lookup_;
    mutable std::mutex label_mutex_;

    std::vector<labeltype> id_to_label_;
    std::vector<tableint> vector_to_upper_node_;

    IDAllocator node_id_allocator_;
    IDAllocator vector_id_allocator_;

    size_t max_nodes_;
    size_t neighbor_slots_;
    size_t dim_size_;
    size_t vector_pool_size_;
    size_t prefetch_cache_lines_;

    AlignedStorage(size_t max_nodes, size_t dim_bytes, size_t M = 16)
        : max_nodes_(max_nodes), neighbor_slots_(2 * M),
          dim_size_(dim_bytes) {
        if (max_nodes_ == 0 || max_nodes_ > kMaxNodeCapacity) {
            throw std::runtime_error("Requested max_nodes exceeds the packed node-ID limit");
        }
        if (dim_size_ == 0 || max_nodes_ > std::numeric_limits<size_t>::max() / dim_size_ ||
            max_nodes_ > std::numeric_limits<size_t>::max() / nodeStride()) {
            throw std::runtime_error("Invalid vector-pool size");
        }

        vector_pool_size_ = max_nodes_ * dim_size_;
        try {
            node_pool_ = detail::allocateAligned<tableint>(max_nodes_ * neighbor_slots_, "node pool");
            metadata_pool_ = detail::allocateAligned<NodeMetaData>(max_nodes_, "hot metadata pool");
            cold_metadata_pool_ =
                detail::allocateAligned<ColdNodeMetaData>(max_nodes_, "cold metadata pool");
            vector_data_pool_ =
                detail::allocateAligned<char>(vector_pool_size_, "vector pool");

            detail::adviseHugePages(node_pool_, max_nodes_ * nodeStride());
            detail::adviseHugePages(metadata_pool_, max_nodes_ * sizeof(NodeMetaData));
            detail::adviseHugePages(cold_metadata_pool_, max_nodes_ * sizeof(ColdNodeMetaData));
            for (size_t i = 0; i < max_nodes_; ++i) {
                new (&metadata_pool_[i]) NodeMetaData();
                new (&cold_metadata_pool_[i]) ColdNodeMetaData();
            }

            detail::adviseHugePages(vector_data_pool_, vector_pool_size_);
            std::memset(vector_data_pool_, 0, vector_pool_size_);
            id_to_label_.resize(max_nodes_);
            vector_to_upper_node_.resize(max_nodes_, kInvalidTableId);
            // Avoid repeated bucket rehashing during a large initial build.
            label_lookup_.reserve(max_nodes_);
        } catch (...) {
            releasePools();
            throw;
        }

        prefetch_cache_lines_ =
            std::clamp((dim_size_ + kCacheLineBytes - 1) / kCacheLineBytes,
                       size_t{1}, size_t{4});
    }

    ~AlignedStorage() {
        releasePools();
    }

    // Disable copy
    AlignedStorage(const AlignedStorage&) = delete;
    AlignedStorage& operator=(const AlignedStorage&) = delete;

    // --- Core API ---

    // Allocate physical node ID
    tableint allocNode() {
        tableint id = node_id_allocator_.alloc();
        if (id >= max_nodes_) {
            node_id_allocator_.free(id);
            throw std::runtime_error("Node pool exhausted (Static Mode)");
        }
        lockNode(id);
        std::memset(getNode(id).neighbors, 0, nodeStride());
        resetNodeState(id);
        unlockNode(id);
        return id;
    }

    void freeNode(tableint id) {
        lockNode(id);
        resetNodeState(id);
        unlockNode(id);
        node_id_allocator_.free(id);
    }

    int getActualNodeCount() const {
        return node_id_allocator_.getActualSize();
    }

    int getActualVectorCount() const {
        return vector_id_allocator_.getActualSize();
    }

    // Allocate and write vector data
    tableint allocVector(const void* data_point, labeltype label) {
        tableint vec_id = vector_id_allocator_.alloc();
        if (vec_id >= max_nodes_) {
            vector_id_allocator_.free(vec_id);
            throw std::runtime_error("Vector pool exhausted (Static Mode)");
        }

        std::memcpy(getVector(vec_id), data_point, dim_size_);
        id_to_label_[vec_id] = label;
        try {
            std::lock_guard<std::mutex> lock(label_mutex_);
            if (!label_lookup_.emplace(label, vec_id).second)
                throw std::runtime_error("Label already exists");
        } catch (...) {
            vector_id_allocator_.free(vec_id);
            throw;
        }
        return vec_id;
    }

    // The replacement caller owns this deleted record. It has completed repair
    // and deliberately overwrites without waiting for older readers.
    void updateVector(tableint vec_id, const void* data_point, labeltype label) {
        std::memcpy(getVector(vec_id), data_point, dim_size_);
        id_to_label_[vec_id] = label;
        std::lock_guard<std::mutex> lock(label_mutex_);
        if (!label_lookup_.emplace(label, vec_id).second)
            throw std::runtime_error("Label already exists");
    }

    void freeVector(tableint vec_id) {
        std::lock_guard<std::mutex> lock(label_mutex_);
        labeltype label = id_to_label_[vec_id];
        const auto it = label_lookup_.find(label);
        if (it != label_lookup_.end() && it->second == vec_id) label_lookup_.erase(it);
        vector_to_upper_node_[vec_id] = kInvalidTableId;
        vector_id_allocator_.free(vec_id);
    }

    // Accessors (Hot Path - Inline)
    inline size_t nodeStride() const {
        return neighbor_slots_ * sizeof(tableint);
    }

    inline AlignedNode getNode(tableint id) {
        return {node_pool_ + static_cast<size_t>(id) * neighbor_slots_};
    }

    inline ConstAlignedNode getNode(tableint id) const {
        return {node_pool_ + static_cast<size_t>(id) * neighbor_slots_};
    }

    inline NodeMetaData& getNodeMetaData(tableint id) {
        return metadata_pool_[id];
    }

    inline const NodeMetaData& getNodeMetaData(tableint id) const {
        return metadata_pool_[id];
    }

    inline uint8_t getGeneration(tableint id) const {
        return metadata_pool_[id].generation;
    }

    inline void setGeneration(tableint id, uint8_t generation) {
        metadata_pool_[id].generation = generation & kGenerationMask;
    }

    inline uint8_t bumpGeneration(tableint id) {
        // Writers hold the node lock. Publish only the wrapped value: fetch_add
        // followed by masking would briefly expose generation 16 to readers.
        auto& generation = metadata_pool_[id].generation;
        const uint8_t next = (generation.load() + 1) & kGenerationMask;
        generation.store(next);
        return next;
    }

    inline tableint getVectorId(tableint id) const {
        return metadata_pool_[id].vector_id;
    }

    inline void setVectorId(tableint id, tableint vec_id) {
        metadata_pool_[id].vector_id = vec_id;
    }

    inline uint8_t getSize(tableint id) const {
        return metadata_pool_[id].size.load(std::memory_order_relaxed);
    }

    inline void setSize(tableint id, uint8_t size) {
        metadata_pool_[id].size.store(size, std::memory_order_relaxed);
    }

    inline uint8_t getPruned(tableint id) const {
        return metadata_pool_[id].is_pruned.load(std::memory_order_relaxed);
    }

    inline void setPruned(tableint id, uint8_t pruned) {
        metadata_pool_[id].is_pruned.store(pruned, std::memory_order_relaxed);
    }

    inline bool isDeleted(tableint id) const {
        return metadata_pool_[id].is_deleted.load(std::memory_order_acquire) != 0;
    }

    inline void setDeleted(tableint id, bool deleted) {
        metadata_pool_[id].is_deleted.store(deleted ? 1 : 0, std::memory_order_release);
    }

    inline uint8_t getMutationVersion(tableint id) const {
        return cold_metadata_pool_[id].mutation_version.load(std::memory_order_relaxed);
    }

    inline void setMutationVersion(tableint id, uint8_t version) {
        cold_metadata_pool_[id].mutation_version.store(version, std::memory_order_relaxed);
    }

    inline void bumpMutationVersion(tableint id) {
        auto& version = cold_metadata_pool_[id].mutation_version;
        version.store(version.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
    }

    inline std::atomic_flag& getNodeLock(tableint id) {
        return cold_metadata_pool_[id].lock;
    }

    inline tableint getDownNodeId(tableint id) const {
        return cold_metadata_pool_[id].down_node_id;
    }

    inline bool isBaseLayerNode(tableint id) const noexcept {
        return getNodeLevel(id) == 0;
    }

    inline uint8_t getNodeLevel(tableint id) const noexcept {
        return cold_metadata_pool_[id].level.load(std::memory_order_relaxed);
    }

    inline void initNodeMeta(tableint id, tableint vec_id, tableint down_id) {
        unsigned level = 0;
        if (down_id != kInvalidTableId) {
            if (down_id == id || down_id >= node_id_allocator_.getWatermark())
                throw std::runtime_error("Invalid node down-chain");
            level = unsigned(getNodeLevel(down_id)) + 1;
            if (level > std::numeric_limits<uint8_t>::max())
                throw std::runtime_error("Node level exceeds 255");
        }
        lockNode(id);
        resetNodeState(id, false);
        metadata_pool_[id].vector_id = vec_id;
        cold_metadata_pool_[id].down_node_id = down_id;
        cold_metadata_pool_[id].level.store(static_cast<uint8_t>(level), std::memory_order_relaxed);
        setDeleted(id, false); // publish only fully initialized metadata
        unlockNode(id);
    }

    inline char* getVector(tableint vec_id) {
        return vector_data_pool_ + static_cast<size_t>(vec_id) * dim_size_;
    }

    inline const char* getVector(tableint vec_id) const {
        return vector_data_pool_ + static_cast<size_t>(vec_id) * dim_size_;
    }

    inline labeltype getLabel(tableint vec_id) const {
        return id_to_label_[vec_id];
    }

    inline const char* getVectorByLabel(labeltype label) const {
        std::lock_guard<std::mutex> lock(label_mutex_);
        auto it = label_lookup_.find(label);
        if (it == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        return getVector(it->second);
    }

    inline tableint getUpperNodeID(tableint vec_id) const {
        return vector_to_upper_node_[vec_id];
    }

private:
    void lockNode(tableint id) noexcept {
        while (cold_metadata_pool_[id].lock.test_and_set(std::memory_order_acquire)) {
#if defined(__i386__) || defined(__x86_64__)
            __builtin_ia32_pause();
#endif
        }
    }

    void unlockNode(tableint id) noexcept {
        cold_metadata_pool_[id].lock.clear(std::memory_order_release);
    }

    void resetNodeState(tableint id, bool reset_down = true) noexcept {
        // size/pruned bound all reads. Reconnection overwrites active entries;
        // unused adjacency bytes do not need clearing when reusing a chain.
        NodeMetaData& hot = metadata_pool_[id];
        hot.vector_id = kInvalidTableId;
        hot.size = 0;
        hot.is_pruned = 0;
        hot.is_deleted.store(1, std::memory_order_relaxed);

        ColdNodeMetaData& cold = cold_metadata_pool_[id];
        if (reset_down) cold.down_node_id = kInvalidTableId;
        cold.level.store(0, std::memory_order_relaxed);
        bumpMutationVersion(id); // never reset an optimistic commit token on reuse
    }

    void releasePools() noexcept {
        std::free(vector_data_pool_);
        std::free(cold_metadata_pool_);
        std::free(metadata_pool_);
        std::free(node_pool_);
        vector_data_pool_ = nullptr;
        cold_metadata_pool_ = nullptr;
        metadata_pool_ = nullptr;
        node_pool_ = nullptr;
    }
};

}  // namespace vigor
