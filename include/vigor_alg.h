#pragma once

#include "hnswlib.h"
#include "storage.h"
#include "visited_list_pool.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace vigor {

namespace detail {
inline constexpr size_t kRepairEf = 50;
inline constexpr size_t kRepairCompensate = 3;
inline constexpr size_t kSearchPrefetchWindow = 4;
inline constexpr size_t kSearchPrefetchLines = 8;
} // namespace detail

template<typename dist_t>
class CompareByFirst {
public:
    constexpr bool operator()(const std::pair<dist_t, tableint>& a,
                              const std::pair<dist_t, tableint>& b) const noexcept {
        return a.first < b.first;
    }
};

class NodeLockGuard {
    std::atomic_flag& flag_;
public:
    explicit NodeLockGuard(std::atomic_flag& flag) : flag_(flag) {
        while (flag_.test_and_set(std::memory_order_acquire)) {
#if defined(__x86_64__) || defined(_M_X64)
            _mm_pause();
#endif
        }
    }

    ~NodeLockGuard() noexcept { flag_.clear(std::memory_order_release); }
    NodeLockGuard(const NodeLockGuard&) = delete;
    NodeLockGuard& operator=(const NodeLockGuard&) = delete;
};

template<typename dist_t>
class VigorIndex : public AlgorithmInterface<dist_t> {
public:
    static constexpr uint8_t kPrunedNoWitness = static_cast<uint8_t>(-1);

    // Layered topology slots and shared vector storage
    std::unique_ptr<AlignedStorage<dist_t>> storage_;

    // Entry point: physical node ID (highest layer)
    std::atomic<tableint> enterpoint_node_id_;
    std::atomic<int> maxlevel_;

    // Configuration parameters
    size_t M_;
    size_t ef_construction_;
    size_t ef_;
    double mult_;

    // Random number generator
    std::default_random_engine level_generator_;
    std::mutex level_generator_mutex_;
    std::unique_ptr<VisitedListPool> visited_list_pool_;
    mutable std::mutex global_entry_lock_; // Only used for updating entry point

    SpaceInterface<dist_t> *space_;
    DISTFUNC<dist_t> fstdistfunc_;

    void *dist_func_param_;

    // Disable before an update batch and enable after all update callers join.
    // Stale primary edges remain available to damage discovery while disabled.
    std::atomic<bool> lazyCompact {true};
    std::queue<tableint> deleted_vector_ids_;
    mutable std::mutex deleted_mutex_;
    VigorIndex(SpaceInterface<dist_t> *s, size_t max_elements, size_t M = 16,
                size_t ef_construction = 200, size_t random_seed = 100)
        : M_(M), ef_construction_(ef_construction), space_(s) {

        if (M_ < 2 || M_ > ALIGNED_M / 2) {
            throw std::invalid_argument("M must be in the range [2, ALIGNED_M / 2]");
        }
        if (max_elements == 0 || max_elements > kMaxElementCapacity) {
            throw std::invalid_argument("max_elements exceeds the packed node-ID capacity");
        }

        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();

        // Reserve 1.2 physical topology/vector slots per resident element.
        size_t max_nodes = max_elements * 6 / 5;
        storage_ = std::make_unique<AlignedStorage<dist_t>>(max_nodes, s->get_data_size(), M_);

        mult_ = 1 / log(1.0 * M_);
        maxlevel_.store(-1);
        enterpoint_node_id_.store(kInvalidTableId);
        ef_ = 10;

        // VisitedListPool needs to be sized for max_nodes, not max_elements
        visited_list_pool_ = std::make_unique<VisitedListPool>(1, max_nodes);

        level_generator_.seed(random_seed);
    }

    void setEf(size_t ef) {
        ef_ = ef;
    }

    // Generate random level
    int getRandomLevel(double reverse_size) {
        std::lock_guard<std::mutex> lock(level_generator_mutex_);
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(level_generator_)) * reverse_size;
        return (int)r;
    }

    inline uint8_t getGeneration(tableint id) const {
        return storage_->getGeneration(id);
    }

    inline uint8_t bumpGeneration(tableint id) {
        return storage_->bumpGeneration(id);
    }

    inline tableint getVectorId(tableint id) const {
        return storage_->getVectorId(id);
    }

    inline tableint vectorForEdge(tableint edge) const {
        const tableint id = getPhysicalId(edge);
        const uint8_t generation = getEdgeGeneration(edge);
        if (getGeneration(id) != generation) return kInvalidTableId;
        const tableint vector = getVectorId(id);
        return getGeneration(id) == generation ? vector : kInvalidTableId;
    }

    std::pair<tableint, int> entrySnapshot() const {
        std::lock_guard<std::mutex> lock(global_entry_lock_);
        return {enterpoint_node_id_.load(), maxlevel_.load()};
    }

    inline void setVectorId(tableint id, tableint vec_id) {
        storage_->setVectorId(id, vec_id);
    }

    inline tableint getDownNodeId(tableint id) const {
        return storage_->getDownNodeId(id);
    }

    inline bool isBaseLayerNode(tableint id) const noexcept {
        return storage_->isBaseLayerNode(id);
    }

    inline uint8_t getNodeLevel(tableint id) const {
        if (id == kInvalidTableId || id >= storage_->node_id_allocator_.getWatermark())
            throw std::runtime_error("Cannot read level for an invalid node");
        return storage_->getNodeLevel(id);
    }

    inline uint8_t getNodeSize(tableint id) const {
        return storage_->getSize(id);
    }

    inline void setNodeSize(tableint id, uint8_t size) {
        storage_->setSize(id, size);
    }

    inline uint8_t getPrunedState(tableint id) const {
        return storage_->getPruned(id);
    }

    inline void setPrunedState(tableint id, uint8_t pruned) {
        storage_->setPruned(id, pruned);
    }

    inline uint8_t getMutationVersion(tableint id) const {
        return storage_->getMutationVersion(id);
    }

    inline void bumpMutationVersion(tableint id) {
        storage_->bumpMutationVersion(id);
    }

    inline std::atomic_flag& getNodeLock(tableint id) {
        return storage_->getNodeLock(id);
    }

    inline std::atomic_flag& getNodeLock(tableint id) const {
        return storage_->getNodeLock(id);
    }

    inline bool isDeleted(tableint id) const {
        return storage_->isDeleted(id);
    }

    inline void setDeleted(tableint id, bool deleted) {
        storage_->setDeleted(id, deleted);
    }

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
    inline void prefetchNeighborList(const tableint* neighbors, size_t size) const {
        if (size == 0) return;
        _mm_prefetch(reinterpret_cast<const char*>(neighbors), _MM_HINT_T0);
        if (size > 16) {
            _mm_prefetch(reinterpret_cast<const char*>(neighbors) + 64, _MM_HINT_T0);
        }
    }

    inline void prefetchNodeMetaData(tableint node_id) const {
        _mm_prefetch(reinterpret_cast<const char*>(&storage_->getNodeMetaData(node_id)), _MM_HINT_T0);
    }

    inline void prefetchSearchVector(tableint edge, const vl_type* visited, vl_type tag) const {
        if (visited[getPhysicalId(edge)] == tag) return;
        const tableint vector = vectorForEdge(edge);
        if (vector == kInvalidTableId) return;
        const char* data = storage_->getVector(vector);
        const size_t lines = std::min(detail::kSearchPrefetchLines,
            (storage_->dim_size_ + kCacheLineBytes - 1) / kCacheLineBytes);
        for (size_t line = 0; line < lines; ++line)
            _mm_prefetch(data + line * kCacheLineBytes, _MM_HINT_T0);
    }

    inline void prefetchVectorByVecId(tableint vec_id) const {
        if (vec_id == kInvalidTableId) return;
        const char* vec_ptr = storage_->getVector(vec_id);
        _mm_prefetch(vec_ptr, _MM_HINT_T0);
        if (storage_->prefetch_cache_lines_ > 1) {
            _mm_prefetch(vec_ptr + kCacheLineBytes, _MM_HINT_T0);
        }
        if (storage_->prefetch_cache_lines_ > 2) {
            _mm_prefetch(vec_ptr + 2 * kCacheLineBytes, _MM_HINT_T0);
        }
        if (storage_->prefetch_cache_lines_ > 3) {
            _mm_prefetch(vec_ptr + 3 * kCacheLineBytes, _MM_HINT_T0);
        }
    }

#endif

    inline size_t physicalNeighborSlots() const {   //可用空间槽位，统一为2M
        return M_ * 2;
    }

    inline size_t forwardNeighborLimit(uint8_t level) const {
        return level ? M_ : (M_ * 2);
    }

    inline size_t forwardNeighborLimit(tableint node_id) const {
        return isBaseLayerNode(node_id) ? (M_ * 2) : M_;
    }

    using Candidate = std::pair<dist_t, tableint>;
    class CandidateQueue
        : public std::priority_queue<Candidate, std::vector<Candidate>, CompareByFirst<dist_t>> {
        using Base = std::priority_queue<Candidate, std::vector<Candidate>, CompareByFirst<dist_t>>;
    public:
        using Base::Base;
        const std::vector<Candidate>& values() const noexcept { return this->c; }
    };

    // A unique minimum can be found by a linear scan without copying and
    // draining the heap. If distances tie (or are unordered), retain the old
    // heap-pop result exactly because it controls the next layer's entry.
    tableint nearestCandidateEdge(const CandidateQueue& candidates) const {
        if (candidates.empty()) return kInvalidTableId;
        const auto& values = candidates.values();
        auto nearest = values.begin();
        bool equivalent_minimum = false;
        for (auto it = values.begin() + 1; it != values.end(); ++it) {
            if (it->first < nearest->first) {
                nearest = it;
                equivalent_minimum = false;
            } else if (!(nearest->first < it->first)) {
                equivalent_minimum = true;
            }
        }
        if (!equivalent_minimum) return nearest->second;

        CandidateQueue copy = candidates;
        tableint edge = kInvalidTableId;
        while (!copy.empty()) {
            edge = copy.top().second;
            copy.pop();
        }
        return edge;
    }

    // Each lane follows the active native SSE/AVX kernel's accumulation and
    // reduction order. No arithmetic regrouping or extra vector pairs.
    inline void repairDistance4(const void* query, const void* const* vectors,
                                dist_t* distances) const {
#if defined(USE_AVX)
        if constexpr (std::is_same<dist_t, float>::value) {
            if (fstdistfunc_ == L2SqrSIMD16ExtAVX) {
                const size_t dim = *static_cast<const size_t*>(dist_func_param_);
                const float* q = static_cast<const float*>(query);
                const float* a = static_cast<const float*>(vectors[0]);
                const float* b = static_cast<const float*>(vectors[1]);
                const float* c = static_cast<const float*>(vectors[2]);
                const float* d = static_cast<const float*>(vectors[3]);
                __m256 sa = _mm256_setzero_ps(), sb = sa, sc = sa, sd = sa;
                for (size_t j = 0; j < (dim & ~size_t(15)); j += 8) {
                    const __m256 x = _mm256_loadu_ps(q + j);
                    const __m256 da = _mm256_sub_ps(x, _mm256_loadu_ps(a + j));
                    const __m256 db = _mm256_sub_ps(x, _mm256_loadu_ps(b + j));
                    const __m256 dc = _mm256_sub_ps(x, _mm256_loadu_ps(c + j));
                    const __m256 dd = _mm256_sub_ps(x, _mm256_loadu_ps(d + j));
                    sa = _mm256_add_ps(sa, _mm256_mul_ps(da, da));
                    sb = _mm256_add_ps(sb, _mm256_mul_ps(db, db));
                    sc = _mm256_add_ps(sc, _mm256_mul_ps(dc, dc));
                    sd = _mm256_add_ps(sd, _mm256_mul_ps(dd, dd));
                }
                alignas(32) float tmp[8];
                auto reduce = [&](const __m256 value) {
                    _mm256_store_ps(tmp, value);
                    return tmp[0] + tmp[1] + tmp[2] + tmp[3] +
                           tmp[4] + tmp[5] + tmp[6] + tmp[7];
                };
                distances[0] = reduce(sa); distances[1] = reduce(sb);
                distances[2] = reduce(sc); distances[3] = reduce(sd);
                return;
            }
        }
#endif
#if defined(USE_SSE)
        if constexpr (std::is_same<dist_t, float>::value) {
            if (fstdistfunc_ == L2SqrSIMD16ExtSSE) {
                const size_t dim = *static_cast<const size_t*>(dist_func_param_);
                const float* q = static_cast<const float*>(query);
                const float* a = static_cast<const float*>(vectors[0]);
                const float* b = static_cast<const float*>(vectors[1]);
                const float* c = static_cast<const float*>(vectors[2]);
                const float* d = static_cast<const float*>(vectors[3]);
                __m128 sa = _mm_setzero_ps(), sb = sa, sc = sa, sd = sa;
                for (size_t j = 0; j < (dim & ~size_t(15)); j += 4) {
                    const __m128 x = _mm_loadu_ps(q + j);
                    const __m128 da = _mm_sub_ps(x, _mm_loadu_ps(a + j));
                    const __m128 db = _mm_sub_ps(x, _mm_loadu_ps(b + j));
                    const __m128 dc = _mm_sub_ps(x, _mm_loadu_ps(c + j));
                    const __m128 dd = _mm_sub_ps(x, _mm_loadu_ps(d + j));
                    sa = _mm_add_ps(sa, _mm_mul_ps(da, da));
                    sb = _mm_add_ps(sb, _mm_mul_ps(db, db));
                    sc = _mm_add_ps(sc, _mm_mul_ps(dc, dc));
                    sd = _mm_add_ps(sd, _mm_mul_ps(dd, dd));
                }
                alignas(16) float tmp[4];
                auto reduce = [&](const __m128 value) {
                    _mm_store_ps(tmp, value);
                    return tmp[0] + tmp[1] + tmp[2] + tmp[3];
                };
                distances[0] = reduce(sa); distances[1] = reduce(sb);
                distances[2] = reduce(sc); distances[3] = reduce(sd);
                return;
            }
            if (fstdistfunc_ == InnerProductDistanceSIMD16ExtSSE ||
                fstdistfunc_ == InnerProductDistanceSIMD4ExtSSE) {
                const size_t dim = *static_cast<const size_t*>(dist_func_param_);
                const size_t simd_end = fstdistfunc_ == InnerProductDistanceSIMD16ExtSSE
                    ? (dim & ~size_t(15)) : (dim & ~size_t(3));
                const float* q = static_cast<const float*>(query);
                const float* a = static_cast<const float*>(vectors[0]);
                const float* b = static_cast<const float*>(vectors[1]);
                const float* c = static_cast<const float*>(vectors[2]);
                const float* d = static_cast<const float*>(vectors[3]);
                __m128 sa = _mm_setzero_ps(), sb = sa, sc = sa, sd = sa;
                // The native SSE IP kernels use one accumulator in this same
                // per-vector order. Only independent vector pairs interleave.
                for (size_t j = 0; j < simd_end; j += 4) {
                    const __m128 x = _mm_loadu_ps(q + j);
                    sa = _mm_add_ps(sa, _mm_mul_ps(x, _mm_loadu_ps(a + j)));
                    sb = _mm_add_ps(sb, _mm_mul_ps(x, _mm_loadu_ps(b + j)));
                    sc = _mm_add_ps(sc, _mm_mul_ps(x, _mm_loadu_ps(c + j)));
                    sd = _mm_add_ps(sd, _mm_mul_ps(x, _mm_loadu_ps(d + j)));
                }
                alignas(16) float tmp[4];
                auto reduce = [&](const __m128 value) {
                    _mm_store_ps(tmp, value);
                    return 1.0f - (tmp[0] + tmp[1] + tmp[2] + tmp[3]);
                };
                distances[0] = reduce(sa); distances[1] = reduce(sb);
                distances[2] = reduce(sc); distances[3] = reduce(sd);
                return;
            }
        }
#endif
        for (size_t i = 0; i < 4; ++i)
            distances[i] = fstdistfunc_(query, vectors[i], dist_func_param_);
    }

    template <bool batch_diversification = false>
    std::vector<tableint> getNeighborsByHeuristic(
        CandidateQueue& top_candidates,
        const size_t M_limit,
        std::vector<tableint>& ordered_results) {
        const size_t target_size = ordered_results.size() + M_limit;

        thread_local std::vector<const void*> selected_vectors;
        selected_vectors.clear();
        selected_vectors.reserve(ordered_results.size() + top_candidates.size());

        // Cache selected vector addresses instead of resolving every edge in
        // the inner diversification loop.
        for (tableint selected_edge : ordered_results) {
            tableint selected_vec_id = vectorForEdge(selected_edge);
            if (selected_vec_id == kInvalidTableId) continue;
            selected_vectors.push_back(storage_->getVector(selected_vec_id));
        }

        // Convert to nearest-first order by negating the distance.
        CandidateQueue queue_closest;
        while (!top_candidates.empty()) {
            queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
            top_candidates.pop();
        }

        std::vector<tableint> eliminated_list;

        // Greedily retain candidates that are not shadowed by a closer peer.
        while (!queue_closest.empty()) {
            if (ordered_results.size() >= target_size)
                break;

            Candidate candidate = queue_closest.top();
            dist_t dist_to_query = -candidate.first;
            queue_closest.pop();

            bool is_visible = true;

            tableint cand_v_id = vectorForEdge(candidate.second);
            if (cand_v_id == kInvalidTableId) continue;
            const void* candidate_vec = storage_->getVector(cand_v_id);

            for (size_t selected = 0; selected < selected_vectors.size();) {
#if defined(USE_SSE)
                if constexpr (batch_diversification && std::is_same<dist_t, float>::value) {
                    // Most rejected candidates stop early. Batch the longer
                    // suffix only for the pure, exact built-in SSE kernel.
                    if (fstdistfunc_ == L2SqrSIMD16ExtSSE && selected >= 4 &&
                        selected + 4 <= selected_vectors.size()) {
                        dist_t distances[4];
                        repairDistance4(candidate_vec, selected_vectors.data() + selected, distances);
                        for (size_t lane = 0; lane < 4; ++lane) {
                            if (distances[lane] < dist_to_query) {
                                is_visible = false;
                                break;
                            }
                        }
                        if (!is_visible) break;
                        selected += 4;
                        continue;
                    }
                }
#endif
                const dist_t dist_between = fstdistfunc_(
                    candidate_vec, selected_vectors[selected++], dist_func_param_);
                if (dist_between < dist_to_query) {
                    is_visible = false;
                    break;
                }
            }

            if (is_visible) {
                ordered_results.push_back(candidate.second);
                selected_vectors.push_back(candidate_vec);
            } else {
                eliminated_list.push_back(candidate.second);
            }
        }

        return eliminated_list;
    }

    void mutuallyConnectNewElement(tableint cur_c,
                                      CandidateQueue& top_candidates,
                                      int level) {
        const size_t neighbor_limit = forwardNeighborLimit(static_cast<uint8_t>(level));
        const size_t physical_slots = physicalNeighborSlots();
        const uint8_t cur_gen = getGeneration(cur_c);
        const tableint cur_edge = packEdge(cur_gen, cur_c);
        const tableint cur_vec_id = getVectorId(cur_c);
        const void* cur_vec = storage_->getVector(cur_vec_id);

        // New nodes retain M neighbors; existing base-layer nodes allow 2*M.
        std::vector<tableint> selected_neighbors;
        selected_neighbors.reserve(M_);
        getNeighborsByHeuristic(top_candidates, M_, selected_neighbors);

        auto node_cur = storage_->getNode(cur_c);
        {
            NodeLockGuard guard(getNodeLock(cur_c));
            setNodeSize(cur_c, static_cast<uint8_t>(selected_neighbors.size()));
            for (size_t i = 0; i < selected_neighbors.size(); ++i) {
                node_cur.neighbors[i] = selected_neighbors[i];
            }
            bumpMutationVersion(cur_c);
        }

        for (tableint selected_neighbor : selected_neighbors) {
            const tableint neighbor_id = getPhysicalId(selected_neighbor);
            if (neighbor_id == cur_c) continue;
            auto node_neighbor = storage_->getNode(neighbor_id);

            for (;;) {
                tableint snapshot_edges[ALIGNED_M];
                size_t snapshot_size = 0;
                tableint neighbor_vec_id = kInvalidTableId;
                uint8_t snapshot_version = 0;
                {
                    NodeLockGuard guard(getNodeLock(neighbor_id));
                    if (getGeneration(neighbor_id) != getEdgeGeneration(selected_neighbor)) break;
                    neighbor_vec_id = getVectorId(neighbor_id);
                    if (neighbor_vec_id == kInvalidTableId) break;

                    const size_t neighbor_size = getNodeSize(neighbor_id);
                    bool already_connected = false;
                    for (size_t k = 0; k < neighbor_size; ++k) {
                        if (getPhysicalId(node_neighbor.neighbors[k]) == cur_c) {
                            already_connected = true;
                            node_neighbor.neighbors[k] = cur_edge;
                            removePrimaryDuplicateWitnessUnlocked(neighbor_id, physical_slots, cur_edge);
                            bumpMutationVersion(neighbor_id);
                            break;
                        }
                    }
                    if (already_connected) break;

                    if (neighbor_size < neighbor_limit) {
                        removePrimaryDuplicateWitnessUnlocked(neighbor_id, physical_slots, cur_edge);
                        node_neighbor.neighbors[neighbor_size] = cur_edge;
                        setNodeSize(neighbor_id, static_cast<uint8_t>(neighbor_size + 1));
                        trimWitnessTailUnlocked(neighbor_id, physical_slots);
                        bumpMutationVersion(neighbor_id);
                        break;
                    }

                    // Only the full-list path needs an optimistic snapshot.
                    snapshot_size = neighbor_size;
                    memcpy(snapshot_edges, node_neighbor.neighbors, snapshot_size * sizeof(tableint));
                    snapshot_version = getMutationVersion(neighbor_id);
                }

                CandidateQueue candidates;
                const void* neighbor_vec = storage_->getVector(neighbor_vec_id);
                const dist_t dist_new = fstdistfunc_(cur_vec, neighbor_vec, dist_func_param_);
                candidates.emplace(dist_new, cur_edge);
                for (size_t j = 0; j < snapshot_size; ++j) {
                    const tableint edge = snapshot_edges[j];
                    const tableint nb_id = getPhysicalId(edge);
                    const uint8_t nb_gen = getEdgeGeneration(edge);
                    const NodeMetaData& nb_meta = storage_->getNodeMetaData(nb_id);
                    if (nb_meta.generation != nb_gen) {
                        snapshot_edges[j] = kInvalidTableId;
                        continue;
                    }
                    const tableint cand_vec_id = getVectorId(nb_id);
                    if (cand_vec_id != kInvalidTableId && getGeneration(nb_id) == nb_gen) {
                        const dist_t dist_old = fstdistfunc_(
                            storage_->getVector(cand_vec_id), neighbor_vec, dist_func_param_);
                        candidates.emplace(dist_old, edge);
                    }
                }

                // Compute without holding the node lock; retry on a changed version.
                std::vector<tableint> pruned_neighbors;
                pruned_neighbors.reserve(physical_slots);
                const auto eliminated_list = getNeighborsByHeuristic(
                    candidates, neighbor_limit, pruned_neighbors);
                tableint committed_edges[ALIGNED_M];
                size_t committed_count = 0;
                {
                    NodeLockGuard guard(getNodeLock(neighbor_id));
                    if (getGeneration(neighbor_id) != getEdgeGeneration(selected_neighbor)) break;
                    if (getMutationVersion(neighbor_id) != snapshot_version) continue;
                    for (tableint edge : pruned_neighbors) {
                        const tableint nb_id = getPhysicalId(edge);
                        if (getGeneration(nb_id) != getEdgeGeneration(edge)) continue;
                        node_neighbor.neighbors[committed_count] = edge;
                        committed_edges[committed_count++] = edge;
                    }
                    setNodeSize(neighbor_id, static_cast<uint8_t>(committed_count));
                    bumpMutationVersion(neighbor_id);
                    notePruneEventUnlocked(neighbor_id, physical_slots);
                }
                compactWitnessTailsForRemovedOutgoingEdges(selected_neighbor,
                                                           snapshot_edges, snapshot_size,
                                                           committed_edges, committed_count,
                                                           physical_slots);
                appendWitnessTail(neighbor_id, physical_slots, eliminated_list, getEdgeGeneration(selected_neighbor));
                break;
            }
        }
    }

    void addPointReuseDeletedSlot(const void* data_point, labeltype label, tableint old_vector,
                                 int level = -1) {
        const tableint top = storage_->getUpperNodeID(old_vector);
        patchLocalHole(old_vector);
        storage_->vector_to_upper_node_[old_vector] = kInvalidTableId;
        freeNodeChain(top);
        storage_->updateVector(old_vector, data_point, label);
        insertPoint(data_point, label, old_vector, level);
    }

    // --- Core implementation: Insertion ---
    void addPoint(const void *data_point, labeltype label, bool replace_deleted = false) override {
        addPoint(data_point, label, replace_deleted, -1);
    }

    // Callers can sample levels in input order before launching parallel updates.
    void addPoint(const void* data_point, labeltype label, bool replace_deleted, int level) {
        bool exists = false;
        {
             std::lock_guard<std::mutex> label_lock(storage_->label_mutex_);
             if (storage_->label_lookup_.find(label) != storage_->label_lookup_.end()) exists = true;
        }

        if (exists) {
            return;
        }

        if (replace_deleted) {
            tableint reuse_vec_id = kInvalidTableId;
            {
                std::lock_guard<std::mutex> lock(deleted_mutex_);
                if (!deleted_vector_ids_.empty()) {
                    reuse_vec_id = deleted_vector_ids_.front();
                    deleted_vector_ids_.pop();
                }
            }
            if (reuse_vec_id != kInvalidTableId) {
                addPointReuseDeletedSlot(data_point, label, reuse_vec_id, level);
                return;
            }
        }

        insertPoint(data_point, label, kInvalidTableId, level);
    }

    // A replacement supplies its overwritten vector slot; ordinary insertion
    // allocates one. Both construct new topology using the supplied or sampled level.
    void insertPoint(const void* data_point, labeltype label,
                     tableint vec_id = kInvalidTableId, int level = -1) {
        if (vec_id == kInvalidTableId) vec_id = storage_->allocVector(data_point, label);
        const int curlevel = level >= 0 ? level : getRandomLevel(mult_);

        auto [curr_ep, curr_max_level] = entrySnapshot();

        // 1. Handle first point
        if (curr_ep == kInvalidTableId) {
            std::lock_guard<std::mutex> lock(global_entry_lock_);
            if (enterpoint_node_id_ == kInvalidTableId) {
                tableint prev_down_id = kInvalidTableId;
                for (int l = 0; l <= curlevel; l++) {
                    tableint node_id = storage_->allocNode();
                    storage_->initNodeMeta(node_id, vec_id, prev_down_id);
                    prev_down_id = node_id;
                }
                enterpoint_node_id_.store(prev_down_id);
                maxlevel_.store(curlevel);
                storage_->vector_to_upper_node_[vec_id] = prev_down_id;
                return;
            }
            curr_ep = enterpoint_node_id_.load();
            curr_max_level = maxlevel_;
        }

        // 2. PHASE 1: Top-Down Search (Find neighbors for each layer)
        // We store the results in a vector of priority queues
        using NeighborsPQ = CandidateQueue;
        std::vector<NeighborsPQ> layer_neighbors(curlevel + 1);

        const tableint entry_vector = getVectorId(curr_ep);
        dist_t curdist = std::numeric_limits<dist_t>::max();
        if (entry_vector != kInvalidTableId)
            curdist = fstdistfunc_(data_point, storage_->getVector(entry_vector), dist_func_param_);

        // A. Fast descent from maxlevel down to curlevel + 1
        for (int l = curr_max_level; l > curlevel && curr_ep != kInvalidTableId; l--) {
            greedySearchAtLevel(data_point, curr_ep, curdist);
            curr_ep = getDownNodeId(curr_ep);
        }

        // B. Search candidates for layers curlevel down to 0
        for (int l = std::min(curlevel, curr_max_level); l >= 0 && curr_ep != kInvalidTableId; l--) {
            layer_neighbors[l] = searchLayer(curr_ep, data_point, ef_construction_, l);
            // Use the closest found neighbor as the entry point for the next layer
            if (!layer_neighbors[l].empty()) {
                curr_ep = getPhysicalId(nearestCandidateEdge(layer_neighbors[l]));
            }

            if (l > 0) {
                curr_ep = getDownNodeId(curr_ep);
            }
        }

        // 3. PHASE 2: Bottom-Up Connection
        tableint prev_down_id = kInvalidTableId;
        for (int l = 0; l <= curlevel; l++) {
            tableint node_id = storage_->allocNode();
            storage_->initNodeMeta(node_id, vec_id, prev_down_id);

            // Connect this node to the graph at layer l
            // This makes the node visible to others at this layer
            if (l <= curr_max_level) {
                mutuallyConnectNewElement(node_id, layer_neighbors[l], l);
            }

            prev_down_id = node_id;
        }

        // 4. Final visibility update
        storage_->vector_to_upper_node_[vec_id] = prev_down_id;

        if (curlevel > maxlevel_) {
            std::lock_guard<std::mutex> lock(global_entry_lock_);
            if (curlevel > maxlevel_) {
                enterpoint_node_id_.store(prev_down_id);
                maxlevel_.store(curlevel);
            }
        }
    }

    // Shared upper-layer descent; filtering applies to results, not routing.
    tableint queryEntry(const void* query_data) const {
        tableint curr_ep = enterpoint_node_id_.load();
        if (curr_ep == kInvalidTableId) {
            return kInvalidTableId;
        }

        tableint ep_v_id = getVectorId(curr_ep);
        if (ep_v_id == kInvalidTableId) {
            return kInvalidTableId;
        }
        dist_t curdist = fstdistfunc_(query_data, storage_->getVector(ep_v_id), dist_func_param_);

        for (int l = maxlevel_; l > 0; l--) {
            greedySearchAtLevel(query_data, curr_ep, curdist);
            if (curr_ep != kInvalidTableId) {
                curr_ep = getDownNodeId(curr_ep);
            }
        }

        return curr_ep;
    }

    struct QueryResult {
        labeltype label;
        const void* vector;
        dist_t distance;
    };
    struct QuerySearchOptions {
        BaseFilterFunctor* filter;
        BaseSearchStopCondition<dist_t>* stop;
        // Only extended queries construct this map. Vector addresses remain in
        // the pool, but updates may overwrite their payloads before callbacks.
        std::unordered_map<tableint, QueryResult> results;
    };

    std::vector<std::pair<dist_t, labeltype>>
    queryResults(CandidateQueue candidates, const QuerySearchOptions& options) const {
        std::vector<std::pair<dist_t, labeltype>> result(candidates.size());
        for (size_t i = result.size(); i > 0; --i) {
            const auto candidate = candidates.top();
            candidates.pop();
            result[i - 1] = {candidate.first, options.results.at(candidate.second).label};
        }
        return result;
    }

    // --- Core implementation: Search ---
    std::priority_queue<std::pair<dist_t, labeltype>>
    searchKnn(const void* query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const override {
        std::priority_queue<std::pair<dist_t, labeltype>> result;
        if (!k) return result;
        const tableint entry = queryEntry(query_data);
        if (entry == kInvalidTableId) return result;
        if (isIdAllowed) {
            QuerySearchOptions options{isIdAllowed, nullptr, {}};
            auto candidates = searchLayer<true>(entry, query_data, std::max(ef_, k),
                                                0, &options);
            while (candidates.size() > k) candidates.pop();
            for (const auto& candidate : queryResults(std::move(candidates), options))
                result.push(candidate);
        } else {
            auto candidates = searchLayer(entry, query_data, std::max(ef_, k), 0);
            while (candidates.size() > k) candidates.pop();
            while (!candidates.empty()) {
                const auto candidate = candidates.top();
                candidates.pop();
                const tableint id = getPhysicalId(candidate.second);
                result.emplace(candidate.first, storage_->getLabel(getVectorId(id)));
            }
        }
        return result;
    }

    std::vector<std::pair<dist_t, labeltype>>
    searchStopConditionClosest(const void* query_data,
                               BaseSearchStopCondition<dist_t>& stop_condition,
                               BaseFilterFunctor* isIdAllowed = nullptr) const {
        const tableint entry = queryEntry(query_data);
        if (entry == kInvalidTableId) return {};
        QuerySearchOptions options{isIdAllowed, &stop_condition, {}};
        auto candidates = searchLayer<true>(entry, query_data, 0, 0, &options);
        auto result = queryResults(std::move(candidates), options);
        stop_condition.filter_results(result);
        return result;
    }

    // Exact serialized v5 size. As with saveIndex, callers must join producers.
    size_t indexFileSize() const {
        std::ostringstream rng_stream;
        rng_stream << level_generator_;
        const size_t header = 2 * sizeof(uint32_t) + 5 * sizeof(size_t) +
            sizeof(double) + sizeof(tableint) + sizeof(int);
        const size_t rng = sizeof(size_t) + rng_stream.str().size();
        const size_t allocators = 2 * (sizeof(tableint) + sizeof(size_t)) +
            (storage_->node_id_allocator_.getFreeListSize() +
             storage_->vector_id_allocator_.getFreeListSize()) * sizeof(tableint);
        const size_t nodes = storage_->node_id_allocator_.getWatermark() *
            (storage_->nodeStride() + 2 * sizeof(tableint) + 6 * sizeof(uint8_t));
        const size_t vectors = storage_->vector_id_allocator_.getWatermark() *
            (storage_->dim_size_ + sizeof(labeltype) + sizeof(tableint));
        return header + rng + allocators + nodes + vectors;
    }

    void saveIndex(const std::string &location) override {
        // External producers must be joined; synchronous updates are committed.

        std::ofstream output(location, std::ios::binary);
        if (!output.is_open()) {
            throw std::runtime_error("Cannot open index file for saving: " + location);
        }

        constexpr uint32_t magic = 0x58494841; // "AHIX"
        constexpr uint32_t version = kIndexFormatVersion;
        const size_t max_nodes = storage_->max_nodes_;
        const size_t dim_size = storage_->dim_size_;
        const tableint node_watermark = storage_->node_id_allocator_.getWatermark();
        const tableint vector_watermark = storage_->vector_id_allocator_.getWatermark();
        const tableint enterpoint = enterpoint_node_id_.load(std::memory_order_relaxed);
        const int maxlevel = maxlevel_.load(std::memory_order_relaxed);
        const auto node_free = storage_->node_id_allocator_.getFreeListSnapshot();
        const auto vector_free = storage_->vector_id_allocator_.getFreeListSnapshot();

        writeBinaryPOD(output, magic);
        writeBinaryPOD(output, version);
        writeBinaryPOD(output, max_nodes);
        writeBinaryPOD(output, dim_size);
        writeBinaryPOD(output, M_);
        writeBinaryPOD(output, ef_construction_);
        writeBinaryPOD(output, ef_);
        writeBinaryPOD(output, mult_);
        writeBinaryPOD(output, enterpoint);
        writeBinaryPOD(output, maxlevel);

        std::ostringstream rng_stream;
        rng_stream << level_generator_;
        const std::string rng_state = rng_stream.str();
        const size_t rng_size = rng_state.size();
        writeBinaryPOD(output, rng_size);
        output.write(rng_state.data(), rng_size);

        writeBinaryPOD(output, node_watermark);
        size_t node_free_count = node_free.size();
        writeBinaryPOD(output, node_free_count);
        if (node_free_count) {
            output.write(reinterpret_cast<const char*>(node_free.data()), node_free_count * sizeof(tableint));
        }

        writeBinaryPOD(output, vector_watermark);
        size_t vector_free_count = vector_free.size();
        writeBinaryPOD(output, vector_free_count);
        if (vector_free_count) {
            output.write(reinterpret_cast<const char*>(vector_free.data()), vector_free_count * sizeof(tableint));
        }

        output.write(reinterpret_cast<const char*>(storage_->node_pool_),
                     static_cast<size_t>(node_watermark) * storage_->nodeStride());
        for (tableint i = 0; i < node_watermark; ++i) {
            tableint vector_id = storage_->getVectorId(i);
            uint8_t generation = storage_->getGeneration(i);
            uint8_t size = storage_->getSize(i);
            uint8_t pruned = storage_->getPruned(i);
            uint8_t is_deleted = storage_->isDeleted(i) ? 1 : 0;
            uint8_t mutation_version = storage_->getMutationVersion(i);
            tableint down_node_id = storage_->getDownNodeId(i);
            const uint8_t level = storage_->getNodeLevel(i);

            writeBinaryPOD(output, vector_id);
            writeBinaryPOD(output, generation);
            writeBinaryPOD(output, size);
            writeBinaryPOD(output, pruned);
            writeBinaryPOD(output, is_deleted);
            writeBinaryPOD(output, mutation_version);
            writeBinaryPOD(output, down_node_id);
            writeBinaryPOD(output, level);
        }

        output.write(storage_->vector_data_pool_, static_cast<size_t>(vector_watermark) * dim_size);
        for (tableint i = 0; i < vector_watermark; ++i) {
            writeBinaryPOD(output, storage_->id_to_label_[i]);
            writeBinaryPOD(output, storage_->vector_to_upper_node_[i]);
        }

        if (!output.good()) {
            throw std::runtime_error("Failed while writing index file: " + location);
        }
    }

    void loadIndex(const std::string &location) {
        std::ifstream input(location, std::ios::binary);
        if (!input.is_open()) {
            throw std::runtime_error("Cannot open index file for loading: " + location);
        }

        constexpr uint32_t expected_magic = 0x58494841; // "AHIX"
        constexpr uint32_t expected_version = kIndexFormatVersion;
        uint32_t magic = 0;
        uint32_t version = 0;
        size_t max_nodes = 0;
        size_t dim_size = 0;
        tableint enterpoint = kInvalidTableId;
        int maxlevel = -1;

        readBinaryPOD(input, magic);
        readBinaryPOD(input, version);
        if (magic != expected_magic || version != expected_version) {
            throw std::runtime_error("Unsupported VigorIndex index format");
        }

        readBinaryPOD(input, max_nodes);
        readBinaryPOD(input, dim_size);
        if (dim_size != space_->get_data_size()) {
            throw std::runtime_error("Index dimension does not match current space");
        }

        readBinaryPOD(input, M_);
        if (M_ < 2 || M_ > ALIGNED_M / 2)
            throw std::runtime_error("Index M exceeds the supported adjacency capacity");
        readBinaryPOD(input, ef_construction_);
        readBinaryPOD(input, ef_);
        readBinaryPOD(input, mult_);
        readBinaryPOD(input, enterpoint);
        readBinaryPOD(input, maxlevel);

        size_t rng_size = 0;
        readBinaryPOD(input, rng_size);
        std::string rng_state(rng_size, '\0');
        if (rng_size) input.read(&rng_state[0], rng_size);
        std::istringstream rng_stream(rng_state);
        rng_stream >> level_generator_;

        tableint node_watermark = 0;
        size_t node_free_count = 0;
        readBinaryPOD(input, node_watermark);
        readBinaryPOD(input, node_free_count);
        std::vector<tableint> node_free(node_free_count);
        if (node_free_count) {
            input.read(reinterpret_cast<char*>(node_free.data()), node_free_count * sizeof(tableint));
        }

        tableint vector_watermark = 0;
        size_t vector_free_count = 0;
        readBinaryPOD(input, vector_watermark);
        readBinaryPOD(input, vector_free_count);
        std::vector<tableint> vector_free(vector_free_count);
        if (vector_free_count) {
            input.read(reinterpret_cast<char*>(vector_free.data()), vector_free_count * sizeof(tableint));
        }

        if (node_watermark > max_nodes || vector_watermark > max_nodes) {
            throw std::runtime_error("Corrupted VigorIndex index: watermark exceeds capacity");
        }


        auto new_storage = std::make_unique<AlignedStorage<dist_t>>(max_nodes, dim_size, M_);
        new_storage->node_id_allocator_.restore(node_watermark, node_free);
        new_storage->vector_id_allocator_.restore(vector_watermark, vector_free);

        input.read(reinterpret_cast<char*>(new_storage->node_pool_),
                   static_cast<size_t>(node_watermark) * new_storage->nodeStride());
        for (tableint i = 0; i < node_watermark; ++i) {
            tableint vector_id = kInvalidTableId;
            uint8_t generation = 0;
            uint8_t size = 0;
            uint8_t pruned = 0;
            uint8_t is_deleted = 0;
            uint8_t mutation_version = 0;
            tableint down_node_id = kInvalidTableId;
            uint8_t level = 0;

            readBinaryPOD(input, vector_id);
            readBinaryPOD(input, generation);
            readBinaryPOD(input, size);
            readBinaryPOD(input, pruned);
            readBinaryPOD(input, is_deleted);
            readBinaryPOD(input, mutation_version);
            readBinaryPOD(input, down_node_id);
            readBinaryPOD(input, level);

            if (generation > kGenerationMask)
                throw std::runtime_error("Corrupted VigorIndex generation");

            new_storage->setVectorId(i, vector_id);
            new_storage->setGeneration(i, generation);
            new_storage->setSize(i, size);
            new_storage->setPruned(i, pruned);
            new_storage->setDeleted(i, is_deleted != 0);
            new_storage->setMutationVersion(i, mutation_version);
            new_storage->getNodeLock(i).clear();
            new_storage->cold_metadata_pool_[i].down_node_id.store(down_node_id, std::memory_order_relaxed);
            new_storage->cold_metadata_pool_[i].level.store(level, std::memory_order_relaxed);
        }

        // Stored levels must decrease by exactly one along every down-link.
        // This validates chains without reconstructing levels or allocating scratch.
        for (tableint i = 0; i < node_watermark; ++i) {
            const auto down = new_storage->getDownNodeId(i);
            const unsigned level = new_storage->getNodeLevel(i);
            if (down == kInvalidTableId ? level != 0 :
                (down >= node_watermark || unsigned(new_storage->getNodeLevel(down)) + 1 != level))
                throw std::runtime_error("Corrupted index node level/down-chain");
        }

        input.read(new_storage->vector_data_pool_, static_cast<size_t>(vector_watermark) * dim_size);

        std::unordered_set<tableint> free_vectors(vector_free.begin(), vector_free.end());
        {
            std::lock_guard<std::mutex> label_lock(new_storage->label_mutex_);
            new_storage->label_lookup_.clear();
            std::queue<tableint> rebuilt_deleted_vectors;
            for (tableint i = 0; i < vector_watermark; ++i) {
                labeltype label;
                tableint upper_node;
                readBinaryPOD(input, label);
                readBinaryPOD(input, upper_node);
                new_storage->id_to_label_[i] = label;
                new_storage->vector_to_upper_node_[i] = upper_node;
                bool deleted_vector = upper_node != kInvalidTableId && new_storage->isDeleted(upper_node);
                if (free_vectors.find(i) == free_vectors.end() && !deleted_vector) {
                    new_storage->label_lookup_[label] = i;
                }
                if (deleted_vector) rebuilt_deleted_vectors.push(i);
            }
            std::lock_guard<std::mutex> deleted_lock(deleted_mutex_);
            std::swap(deleted_vector_ids_, rebuilt_deleted_vectors);
        }

        if (!input) {
            throw std::runtime_error("Failed while reading index file: " + location);
        }

        storage_ = std::move(new_storage);
        visited_list_pool_ = std::make_unique<VisitedListPool>(1, max_nodes);
        enterpoint_node_id_.store(enterpoint, std::memory_order_relaxed);
        maxlevel_.store(maxlevel, std::memory_order_relaxed);

    }

    void markDelete(labeltype label, bool force_delete = false) {
        tableint vector, top;
        {
            std::lock_guard<std::mutex> lock(storage_->label_mutex_);
            const auto found = storage_->label_lookup_.find(label);
            if (found == storage_->label_lookup_.end()) return;
            vector = found->second;
            top = storage_->vector_to_upper_node_[vector];
            storage_->label_lookup_.erase(found);
        }
        if (top == kInvalidTableId) {
            if (force_delete) storage_->freeVector(vector);
            return;
        }
        for (tableint node = top; node != kInvalidTableId; node = getDownNodeId(node))
            setDeleted(node, true);
        if (force_delete) {
            try { patchLocalHole(vector); }
            catch (...) {
                freeNodeChain(top);
                storage_->freeVector(vector);
                throw;
            }
            freeNodeChain(top);
            storage_->freeVector(vector);
            return;
        }
        std::lock_guard<std::mutex> lock(deleted_mutex_);
        deleted_vector_ids_.push(vector);
    }

    std::vector<dist_t> getDataByLabel(labeltype label) const {
        const char* vec_ptr = storage_->getVectorByLabel(label);
        // Return the stored record, padding the last element for custom byte
        // payloads instead of copying past a floor-sized destination buffer.
        size_t dim = storage_->dim_size_ / sizeof(dist_t) +
                     (storage_->dim_size_ % sizeof(dist_t) != 0);
        std::vector<dist_t> result(dim);
        memcpy(result.data(), vec_ptr, storage_->dim_size_);
        return result;
    }

    // Greedy upper-layer descent. Soft-deleted nodes can still
    // provide a route; generation and vector availability determine traversal.
    // Query and insertion share prefetching and packed-edge validation.
    void greedySearchAtLevel(const void* query, tableint& entry,
                             dist_t& entry_distance) const {
        bool changed = true;
        while (changed) {
            changed = false;
            tableint neighbors[ALIGNED_M];
            size_t curr_size;
            {
                NodeLockGuard guard(getNodeLock(entry));
                curr_size = getNodeSize(entry);
                std::memcpy(neighbors, storage_->getNode(entry).neighbors,
                            curr_size * sizeof(tableint));
            }
#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
            prefetchNeighborList(neighbors, curr_size);
            if (curr_size > 0) {
                tableint prefetch_id0 = getPhysicalId(neighbors[0]);
                prefetchNodeMetaData(prefetch_id0);
            }
            if (curr_size > 1) {
                tableint prefetch_id1 = getPhysicalId(neighbors[1]);
                prefetchNodeMetaData(prefetch_id1);
            }
#endif
            for (size_t i = 0; i < curr_size; i++) {
                tableint edge = neighbors[i];
                tableint nb_id = getPhysicalId(edge);

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
                if (i + 2 < curr_size) {
                    tableint lookahead_id = getPhysicalId(neighbors[i + 2]);
                    prefetchNodeMetaData(lookahead_id);
                }

                if (i + 1 < curr_size) {
                    prefetchVectorByVecId(vectorForEdge(neighbors[i + 1]));
                }
#endif

                tableint cand_v_id = vectorForEdge(edge);
                if (cand_v_id == kInvalidTableId) continue;

                dist_t d = fstdistfunc_(query, storage_->getVector(cand_v_id), dist_func_param_);
                if (d < entry_distance) {
                    entry_distance = d;
                    entry = nb_id;
                    changed = true;
                }
            }
        }
    }

    bool acceptQueryResult(tableint edge, tableint vector, dist_t distance,
                           QuerySearchOptions& options) const {
        const tableint id = getPhysicalId(edge);
        {
            NodeLockGuard guard(getNodeLock(id));
            if (getGeneration(id) != getEdgeGeneration(edge) || getVectorId(id) != vector ||
                storage_->getNodeMetaData(id).is_deleted.load(std::memory_order_acquire))
                return false;
        }
        const labeltype label = storage_->getLabel(vector);
        if (options.filter && !(*options.filter)(label)) return false;
        const void* data = storage_->getVector(vector);
        options.results.emplace(edge, QueryResult{label, data, distance});
        if (options.stop) options.stop->add_point_to_result(label, data, distance);
        return true;
    }

    struct CompareSearchCandidate {
        bool operator()(const Candidate& a, const Candidate& b) const noexcept {
            return a.first > b.first;
        }
    };

    // Internal helper: single-layer search
    // Default query/insert/repair compile without filter/stop callbacks or maps.
    // Extended queries reuse the same prefetch and witness traversal.
    template<bool extended = false>
    CandidateQueue
    searchLayer(tableint ep_id,
                const void * __restrict data_point,
                size_t ef,
                int layer,
                QuerySearchOptions* options = nullptr) const {
        auto return_visited = [this](VisitedList* list) {
            visited_list_pool_->releaseVisitedList(list);
        };
        std::unique_ptr<VisitedList, decltype(return_visited)> visited_guard(
            visited_list_pool_->getFreeVisitedList(), return_visited);
        VisitedList* vl = visited_guard.get();
        vl_type * __restrict visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        CandidateQueue top_candidates;
        std::priority_queue<Candidate, std::vector<Candidate>, CompareSearchCandidate> candidate_set;
        const size_t physical_slots = physicalNeighborSlots();

        if (ep_id == kInvalidTableId) return top_candidates;
        tableint ep_vector, ep_edge;
        bool ep_deleted;
        {
            NodeLockGuard guard(getNodeLock(ep_id));
            ep_vector = getVectorId(ep_id);
            ep_edge = packEdge(getGeneration(ep_id), ep_id);
            ep_deleted = isDeleted(ep_id);
        }
        if (ep_vector == kInvalidTableId) return top_candidates;

        // Like native HNSW, a deleted entry is expanded but never a result.
        // Repair retains its old vector/adjacency until insertion initializes it.
        dist_t entry_distance = std::numeric_limits<dist_t>::max();
        if (!ep_deleted) {
            entry_distance = fstdistfunc_(data_point, storage_->getVector(ep_vector), dist_func_param_);
            if constexpr (extended) {
                if (acceptQueryResult(ep_edge, ep_vector, entry_distance, *options))
                    top_candidates.emplace(entry_distance, ep_edge);
            } else {
                top_candidates.emplace(entry_distance, ep_edge);
            }
        }
        visited_array[ep_id] = visited_array_tag;
        candidate_set.emplace(entry_distance, ep_edge);

        dist_t lowerBound = top_candidates.empty() ? std::numeric_limits<dist_t>::max() : top_candidates.top().first;

        while (!candidate_set.empty()) {
            const auto current = candidate_set.top();
            const dist_t current_distance = current.first;
            bool stop_search = current_distance > lowerBound && top_candidates.size() == ef;
            if constexpr (extended) {
                if (options->stop)
                    stop_search = options->stop->should_stop_search(current_distance, lowerBound);
            }
            if (stop_search) break;
            candidate_set.pop();

            tableint curNodeNum = getPhysicalId(current.second);

            tableint neighbors[ALIGNED_M];
            size_t size;
            size_t witness_count = 0;
            size_t witness_start = physical_slots;
            bool live_source;
            {
                NodeLockGuard guard(getNodeLock(curNodeNum));
                if (getGeneration(curNodeNum) != getEdgeGeneration(current.second)) continue;
                live_source = !isDeleted(curNodeNum);
                size = getNodeSize(curNodeNum);
                std::memcpy(neighbors, storage_->getNode(curNodeNum).neighbors,
                            physical_slots * sizeof(tableint));
                // A deleted starting node supplies the repair neighborhood at
                // every layer; other upper-layer nodes use primary edges only.
                if (layer == 0 || (curNodeNum == ep_id && !live_source)) {
                    witness_count = witnessTailCountUnlocked(curNodeNum, physical_slots);
                    witness_start = physical_slots - witness_count;
                }
            }

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
            // Lookahead gives the random vector loads several distance calls
            // of lead time. Invalid/stale edges are excluded before their vector
            // payload is prefetched; an acquired pointer may still be reused.
            constexpr size_t prefetch_window = detail::kSearchPrefetchWindow;
            const size_t edge_count = size + witness_count;
            for (size_t j = 0; j < std::min(edge_count, 2 * prefetch_window); ++j) {
                const tableint id = getPhysicalId(neighbors[j < size ? j : witness_start + j - size]);
                _mm_prefetch(reinterpret_cast<const char*>(visited_array + id), _MM_HINT_T0);
                prefetchNodeMetaData(id);
            }
            for (size_t j = 0; j < std::min(edge_count, prefetch_window); ++j)
                prefetchSearchVector(neighbors[j < size ? j : witness_start + j - size],
                                     visited_array, visited_array_tag);
            if (!candidate_set.empty()) {
                const tableint next = getPhysicalId(candidate_set.top().second);
                prefetchNeighborList(storage_->getNode(next).neighbors, physical_slots);
            }
#endif

            const size_t neighbor_count = size + witness_count;
            for (size_t j = 0; j < neighbor_count; ++j) {
                const bool from_primary = j < size;
                const tableint edge = neighbors[from_primary ? j : witness_start + j - size];
#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
                if (j + prefetch_window < edge_count) {
                    const size_t ahead = j + prefetch_window;
                    prefetchSearchVector(neighbors[ahead < size ? ahead : witness_start + ahead - size],
                                         visited_array, visited_array_tag);
                }
                if (j + 2 * prefetch_window < edge_count) {
                    const size_t ahead = j + 2 * prefetch_window;
                    const tableint id = getPhysicalId(neighbors[ahead < size ? ahead : witness_start + ahead - size]);
                    _mm_prefetch(reinterpret_cast<const char*>(visited_array + id), _MM_HINT_T0);
                    prefetchNodeMetaData(id);
                }
#endif
                if (!from_primary && getPhysicalId(edge) == curNodeNum) {
                    continue;
                }
                tableint nb_id = getPhysicalId(edge);
                if (visited_array[nb_id] == visited_array_tag) continue;

                uint8_t nb_gen = getEdgeGeneration(edge);

                const NodeMetaData& nb_meta = storage_->getNodeMetaData(nb_id);
                if (nb_meta.generation != nb_gen) {
                    if (from_primary && live_source) {
                        compactStaleForwardEdgeIfIdle(curNodeNum, edge);
                    }
                    continue;
                }

                visited_array[nb_id] = visited_array_tag;

                // The generation check above already precedes this ID read.
                tableint cand_vec_id = getVectorId(nb_id);
                if (cand_vec_id == kInvalidTableId || getGeneration(nb_id) != nb_gen) {
                    continue;
                }
                const void* __restrict nb_vector = storage_->getVector(cand_vec_id);

                dist_t distance = fstdistfunc_(data_point, nb_vector, dist_func_param_);

                bool can_enter = top_candidates.size() < ef || lowerBound > distance;
                if constexpr (extended) {
                    if (options->stop)
                        can_enter = options->stop->should_consider_candidate(distance, lowerBound);
                }
                if (!can_enter) continue;
                candidate_set.emplace(distance, edge);
                if constexpr (extended) {
                    if (acceptQueryResult(edge, cand_vec_id, distance, *options)) top_candidates.emplace(distance, edge);
                    while (options->stop ? options->stop->should_remove_extra() : top_candidates.size() > ef) {
                        if (top_candidates.empty())
                            throw std::logic_error("Stop condition tried to remove an empty result");
                        const auto removed = top_candidates.top();
                        top_candidates.pop();
                        const auto& record = options->results.at(removed.second);
                        if (options->stop)
                            options->stop->remove_point_from_result(record.label, record.vector, record.distance);
                        options->results.erase(removed.second);
                    }
                } else {
                    const NodeMetaData& meta = storage_->getNodeMetaData(getPhysicalId(edge));
                    if (meta.is_deleted.load(std::memory_order_acquire) == 0) {
                        top_candidates.emplace(distance, edge);
                    }
                    if (top_candidates.size() > ef) top_candidates.pop();
                }
                lowerBound = top_candidates.empty() ? std::numeric_limits<dist_t>::max() : top_candidates.top().first;
            }
        }
        return top_candidates;
    }

    // Witness-tail maintenance (definitions in vigor_witness.h).
    inline size_t trimWitnessTailUnlocked(tableint node_id, size_t max_slots) const;
    inline void notePruneEventUnlocked(tableint node_id, size_t max_slots) const;
    inline size_t witnessTailCountUnlocked(tableint node_id, size_t max_slots) const;
    inline bool forwardListContainsIdUnlocked(const AlignedNode& node,
                                              size_t node_size,
                                              tableint id) const;
    void mergeWitnessTailUnlocked(tableint node_id,
                                  size_t max_slots,
                                  size_t existing_count,
                                  const std::vector<tableint>& filtered_ids,
                                  std::vector<tableint>& merged) const;
    void evictRecoverableWitnesses(tableint node_id,
                                   const std::vector<tableint>& first_hop_edges,
                                   std::vector<tableint>& merged, size_t capacity,
                                   std::vector<tableint>& evicted_ids) const;
    void trimWitnessOverflow(std::vector<tableint>& merged,
                             size_t capacity,
                             const std::vector<tableint>& evicted_ids) const;
    bool hasForwardLinkTo(tableint from_id, tableint target_id, uint8_t target_gen) const;
    void appendWitnessTail(tableint node_id,
                           size_t max_slots,
                           const std::vector<tableint>& eliminated_ids,
                           uint8_t target_gen);
    template<bool MatchPhysicalId = false>
    bool removeWitnessTailEntryUnlocked(tableint node_id, size_t max_slots, tableint witness_edge);
    bool removeWitnessTailEntry(tableint node_id, size_t max_slots, tableint witness_edge);
    inline void removePrimaryDuplicateWitnessUnlocked(tableint node_id,
                                                      size_t max_slots,
                                                      tableint edge);
    void compactWitnessTailsForRemovedOutgoingEdges(tableint src_edge,
                                                    const tableint* old_edges,
                                                    size_t old_sz,
                                                    const tableint* new_edges,
                                                    size_t new_sz,
                                                    size_t max_slots);
    bool removeForwardLinkUnlocked(tableint node, tableint target, uint8_t generation) const;
    bool compactStaleForwardEdgeIfIdle(tableint node, tableint edge) const;
    void freeNodeChain(tableint top);
    bool isLiveEdge(tableint edge) const;
    void replaceEntryPoint(tableint retired_entry);
    bool ensureRepairEdge(tableint source_edge, tableint target_edge, bool* stop = nullptr);
    bool applyCandidateRepairBatch(tableint candidate, const std::vector<tableint>& proposals,
                                   size_t slots);
    void patchLocalHole(tableint old_vector);
    void discoverDamage(tableint old_edge, std::vector<tableint>& targets);
    void repairNeighbors(const std::vector<tableint>& targets, const std::vector<tableint>& candidates);

};

} // namespace vigor

#include "vigor_witness.h"
#include "vigor_repair.h"
