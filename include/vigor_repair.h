#pragma once

// Template definitions included by vigor_alg.h after VigorIndex is complete.
namespace vigor {

template <typename dist_t>
inline bool VigorIndex<dist_t>::removeForwardLinkUnlocked(tableint node_id, tableint target_id,
                                                   uint8_t target_gen) const {
    auto node = storage_->getNode(node_id);
    size_t node_size = getNodeSize(node_id);
    const tableint target_edge = packEdge(target_gen, target_id);
    for (size_t i = 0; i < node_size; i++) {
        if (node.neighbors[i] != target_edge) continue;
        node.neighbors[i] = node.neighbors[node_size - 1];
        storage_->setSize(node_id, static_cast<uint8_t>(node_size - 1));
        return true;
    }
    return false;
}

template <typename dist_t>
inline bool VigorIndex<dist_t>::compactStaleForwardEdgeIfIdle(tableint node_id,
                                                       tableint stale_edge) const {
    if (!lazyCompact.load(std::memory_order_acquire)) return false;
    NodeLockGuard guard(storage_->getNodeLock(node_id));
    if (isDeleted(node_id)) return false;

    if (!removeForwardLinkUnlocked(node_id, getPhysicalId(stale_edge), getEdgeGeneration(stale_edge)))
        return false;
    storage_->bumpMutationVersion(node_id);
    return true;
}

template <typename dist_t>
inline void VigorIndex<dist_t>::freeNodeChain(tableint node) {
    while (node != kInvalidTableId) {
        const tableint down = getDownNodeId(node);
        storage_->freeNode(node);
        node = down;
    }
}

template <typename dist_t> inline bool VigorIndex<dist_t>::isLiveEdge(tableint edge) const {
    if (edge == kInvalidTableId) return false;
    tableint id = getPhysicalId(edge);
    return getVectorId(id) != kInvalidTableId && !isDeleted(id) &&
           getGeneration(id) == getEdgeGeneration(edge);
}

template <typename dist_t>
inline void VigorIndex<dist_t>::replaceEntryPoint(tableint retired_entry) {
    tableint preferred_edge = kInvalidTableId;
    for (tableint node = retired_entry; node != kInvalidTableId; node = getDownNodeId(node)) {
        NodeLockGuard guard(getNodeLock(node));
        const auto& adjacency = storage_->getNode(node);
        for (size_t i = 0; i < getNodeSize(node); ++i) {
            const tableint edge = adjacency.neighbors[i];
            if (isLiveEdge(edge) && getNodeLevel(getPhysicalId(edge)) == getNodeLevel(node)) {
                preferred_edge = edge;
                break;
            }
        }
        if (preferred_edge != kInvalidTableId) break;
    }

    // Release the old chain's node lock before publishing the entry pair.
    std::lock_guard<std::mutex> lock(global_entry_lock_);
    const tableint current = enterpoint_node_id_.load();
    // Another producer may already have published a different entry.
    if (current != retired_entry && current != kInvalidTableId && !isDeleted(current) &&
        getVectorId(current) != kInvalidTableId)
        return;
    if (isLiveEdge(preferred_edge)) {
        tableint preferred_id = getPhysicalId(preferred_edge);
        enterpoint_node_id_.store(preferred_id);
        maxlevel_.store(static_cast<int>(getNodeLevel(preferred_id)));
        return;
    }

    tableint best_node = kInvalidTableId;
    int best_level = -1;
    tableint watermark = storage_->node_id_allocator_.getWatermark();
    for (tableint id = 0; id < watermark; id++) {
        if (getVectorId(id) == kInvalidTableId || isDeleted(id)) continue;
        int level = static_cast<int>(getNodeLevel(id));
        if (level > best_level) {
            best_level = level;
            best_node = id;
        }
    }
    enterpoint_node_id_.store(best_node);
    maxlevel_.store(best_level);
}

// Repair targets only consume free capacity; a concurrent fill never triggers pruning.
template <typename dist_t>
inline bool VigorIndex<dist_t>::ensureRepairEdge(tableint src_edge, tableint dst_edge,
                                               bool* stop) {
    if (stop) *stop = false;
    const tableint src_id = getPhysicalId(src_edge);
    const tableint dst_id = getPhysicalId(dst_edge);
    if (src_id == dst_id) return false;

    NodeLockGuard guard(getNodeLock(src_id));
    const bool deleted = isDeleted(src_id);
    const size_t src_size = getNodeSize(src_id);
    const bool full = src_size >= forwardNeighborLimit(src_id);
    // Batch repair stops on a deleted/full target without taking a precheck lock.
    if (stop && (deleted || full)) {
        *stop = true;
        return false;
    }
    if (deleted || getGeneration(src_id) != getEdgeGeneration(src_edge) ||
        isDeleted(dst_id) || getGeneration(dst_id) != getEdgeGeneration(dst_edge))
        return false;

    const size_t physical_slots = physicalNeighborSlots();
    auto src_node = storage_->getNode(src_id);
    for (size_t i = 0; i < src_size; ++i) {
        if (getPhysicalId(src_node.neighbors[i]) != dst_id) continue;
        if (src_node.neighbors[i] != dst_edge) {
            src_node.neighbors[i] = dst_edge;
            bumpMutationVersion(src_id);
        }
        removePrimaryDuplicateWitnessUnlocked(src_id, physical_slots, dst_edge);
        return true;
    }
    if (full) return false;

    removePrimaryDuplicateWitnessUnlocked(src_id, physical_slots, dst_edge);
    src_node.neighbors[src_size] = dst_edge;
    setNodeSize(src_id, static_cast<uint8_t>(src_size + 1));
    trimWitnessTailUnlocked(src_id, physical_slots);
    bumpMutationVersion(src_id);
    return true;
}

// Apply all reverse repair requests for one candidate in one commit.  A
// candidate that still fits is appended directly; only overflow runs the
// heuristic once on the final union of old and proposed neighbors.
template <typename dist_t>
inline bool VigorIndex<dist_t>::applyCandidateRepairBatch(
    tableint candidate_edge, const std::vector<tableint>& raw_proposals, size_t physical_slots) {
    if (raw_proposals.empty()) return true;

    const tableint candidate_id = getPhysicalId(candidate_edge);
    const uint8_t candidate_gen = getEdgeGeneration(candidate_edge);
    const size_t limit = forwardNeighborLimit(candidate_id);
    auto candidate_node = storage_->getNode(candidate_id);

    // Reuse the buffers across candidates and optimistic retries. This
    // function is not recursive; concurrent repair threads own separate buffers.
    thread_local std::vector<tableint> merged_edges;
    thread_local std::vector<tableint> final_edges;
    thread_local CandidateQueue candidate_queue;
    merged_edges.reserve(physical_slots + raw_proposals.size());

    // Both paths publish under the candidate lock and promote kept witnesses.
    auto commit_neighbors = [&](const std::vector<tableint>& edges, bool pruned) {
        for (tableint proposal : raw_proposals) {
            if (std::find(edges.begin(), edges.end(), proposal) != edges.end())
                removePrimaryDuplicateWitnessUnlocked(candidate_id, physical_slots, proposal);
        }
        memcpy(candidate_node.neighbors, edges.data(), edges.size() * sizeof(tableint));
        setNodeSize(candidate_id, static_cast<uint8_t>(edges.size()));
        if (pruned) notePruneEventUnlocked(candidate_id, physical_slots);
        else trimWitnessTailUnlocked(candidate_id, physical_slots);
        bumpMutationVersion(candidate_id);
    };

    while (true) {
        tableint snapshot_edges[ALIGNED_M];
        size_t snapshot_size;
        tableint candidate_vec_id;
        uint8_t snapshot_version;
        {
            NodeLockGuard guard(getNodeLock(candidate_id));
            if (isDeleted(candidate_id) || getGeneration(candidate_id) != candidate_gen) {
                return false;
            }
            snapshot_size = getNodeSize(candidate_id);
            candidate_vec_id = getVectorId(candidate_id);
            if (candidate_vec_id == kInvalidTableId) return false;

            // Merge valid proposals by physical ID before deciding capacity.
            merged_edges.assign(candidate_node.neighbors, candidate_node.neighbors + snapshot_size);
            bool changed = false;

            for (const auto& proposal : raw_proposals) {
                if (proposal == kInvalidTableId) continue;
                const tableint repair_id = getPhysicalId(proposal);
                const uint8_t repair_gen = getEdgeGeneration(proposal);
                if (repair_id == candidate_id || isDeleted(repair_id) ||
                    getGeneration(repair_id) != repair_gen ||
                    getVectorId(repair_id) == kInvalidTableId) {
                    continue;
                }
                auto existing = std::find_if(
                    merged_edges.begin(), merged_edges.end(),
                    [repair_id](tableint edge) { return getPhysicalId(edge) == repair_id; });
                if (existing == merged_edges.end()) {
                    merged_edges.push_back(proposal);
                    changed = true;
                } else if (*existing != proposal) {
                    *existing = proposal;
                    changed = true;
                }
            }
            if (!changed) return true;

            // Stale slots do not require geometric neighbor selection.
            // Only inspect the entire list when it would otherwise overflow.
            if (merged_edges.size() > limit) {
                merged_edges.erase(
                    std::remove_if(merged_edges.begin(), merged_edges.end(),
                                   [&](tableint edge) {
                                       if (edge == kInvalidTableId) return true;
                                       const tableint id = getPhysicalId(edge);
                                       const NodeMetaData& meta = storage_->getNodeMetaData(id);
                                       return id == candidate_id ||
                                              meta.is_deleted.load(std::memory_order_acquire) !=
                                                  0 ||
                                              meta.generation != getEdgeGeneration(edge) ||
                                              meta.vector_id == kInvalidTableId;
                                   }),
                    merged_edges.end());
            }

            if (merged_edges.size() <= limit) {
                commit_neighbors(merged_edges, false);
                return true;
            }
            snapshot_version = getMutationVersion(candidate_id);
            memcpy(snapshot_edges, candidate_node.neighbors, snapshot_size * sizeof(tableint));
        }

        // Only the expensive overflow path releases the lock and retries
        // if foreground insertion changed the candidate in the meantime.
        const void* candidate_vec = storage_->getVector(candidate_vec_id);
        final_edges.clear();
        final_edges.reserve(limit);

        while (!candidate_queue.empty())
            candidate_queue.pop();
        tableint pending_edges[4];
        const void* pending_vectors[4];
        size_t pending_count = 0;
        auto flush_distances = [&] {
            dist_t distances[4];
            if (pending_count == 4) repairDistance4(candidate_vec, pending_vectors, distances);
            else for (size_t j = 0; j < pending_count; ++j)
                distances[j] = fstdistfunc_(candidate_vec, pending_vectors[j], dist_func_param_);
            for (size_t j = 0; j < pending_count; ++j)
                candidate_queue.emplace(distances[j], pending_edges[j]);
            pending_count = 0;
        };
        for (size_t edge_index = 0; edge_index < merged_edges.size(); ++edge_index) {
            const tableint edge = merged_edges[edge_index];
            if (isDeleted(getPhysicalId(edge))) continue;
            // vectorForEdge validates generation on both sides of the vector-ID
            // read; repeating those metadata reads here adds no protection.
            const tableint vector_id = vectorForEdge(edge);
            if (vector_id == kInvalidTableId) continue;
            pending_edges[pending_count] = edge;
            pending_vectors[pending_count++] = storage_->getVector(vector_id);
            if (pending_count == 4) flush_distances();
        }
        if (pending_count) flush_distances();
        if (candidate_queue.empty()) return false;
        const auto eliminated_edges = this->template getNeighborsByHeuristic<true>(
            candidate_queue, limit, final_edges);

        {
            NodeLockGuard guard(getNodeLock(candidate_id));
            if (isDeleted(candidate_id) || getGeneration(candidate_id) != candidate_gen) {
                return false;
            }

            if (getMutationVersion(candidate_id) != snapshot_version) {
                continue;
            }

            commit_neighbors(final_edges, true);
        }

        compactWitnessTailsForRemovedOutgoingEdges(candidate_edge, snapshot_edges, snapshot_size,
                                                   final_edges.data(), final_edges.size(),
                                                   physical_slots);
        appendWitnessTail(candidate_id, physical_slots, eliminated_edges, candidate_gen);
        return true;
    }
}

template <typename dist_t>
inline void VigorIndex<dist_t>::patchLocalHole(tableint old_vector) {
    const tableint top = storage_->getUpperNodeID(old_vector);   // top layer node
    const void* query = storage_->getVector(old_vector);

    // markDelete owns and marks this chain deleted. Invalidate its packed
    // identities now, but retain vectors and adjacency until reuse initialization.
    for (tableint node = top; node != kInvalidTableId; node = getDownNodeId(node)) {
        if (enterpoint_node_id_.load(std::memory_order_relaxed) == node)
            replaceEntryPoint(node);
        NodeLockGuard guard(getNodeLock(node));
        bumpGeneration(node);
        bumpMutationVersion(node);
    }

    std::vector<tableint> targets, candidates;
    candidates.reserve(detail::kRepairEf);
    for (tableint node = top; node != kInvalidTableId; node = getDownNodeId(node)) {
        const tableint old_edge = packEdge(static_cast<uint8_t>(getGeneration(node) - 1), node);
        discoverDamage(old_edge, targets);
        auto found = searchLayer(node, query, detail::kRepairEf, getNodeLevel(node));
        candidates.clear();
        while (!found.empty()) {
            const tableint edge = found.top().second;
            found.pop();
            candidates.push_back(edge);
        }
        repairNeighbors(targets, candidates);
    }
}

//Pruning-aware damage discovery
template <typename dist_t>
inline void VigorIndex<dist_t>::discoverDamage(
    tableint target, std::vector<tableint>& targets) {
    const tableint deleted_id = getPhysicalId(target);
    const uint8_t deleted_generation = getEdgeGeneration(target);

    const size_t physical_slots = physicalNeighborSlots();
    const auto& adjacency = storage_->getNode(deleted_id);
    size_t primary_size, witness_size;
    bool pruned;
    {
        NodeLockGuard guard(getNodeLock(deleted_id));
        primary_size = getNodeSize(deleted_id);
        pruned = getPrunedState(deleted_id) != 0;
        witness_size = witnessTailCountUnlocked(deleted_id, physical_slots);
    }

    targets.clear();
    targets.reserve(primary_size + witness_size);

    // Each old incoming edge is removed once, so discovered sources can be
    // appended directly. Only the final primary-neighbor pass can overlap them.
    // Find nodes that lost outgoing or incoming connectivity through the deleted node.
    auto process_candidate = [&](tableint victim_edge, bool expand_two_hop) {
        const tableint victim_id = getPhysicalId(victim_edge);
        const uint8_t expected_generation = getEdgeGeneration(victim_edge);

        auto victim_node = storage_->getNode(victim_id);
        if (isDeleted(victim_id) || getGeneration(victim_id) != expected_generation) {
            return;
        }
        bool lost_edge = false;
        tableint one_hop_snapshot[ALIGNED_M];
        size_t curr_size = 0;
        {
            NodeLockGuard guard(getNodeLock(victim_id));
            if (isDeleted(victim_id) || getGeneration(victim_id) != expected_generation) return;
            if (removeForwardLinkUnlocked(victim_id, deleted_id, deleted_generation)) {
                bumpMutationVersion(victim_id);
                lost_edge = true;
            }
            if (getPrunedState(victim_id) != 0) {
                removeWitnessTailEntryUnlocked(victim_id, physical_slots, target);
            }
            if (expand_two_hop) {
                curr_size = getNodeSize(victim_id);
                memcpy(one_hop_snapshot, victim_node.neighbors, curr_size * sizeof(tableint));
            }
        }
        // Only primary neighbors of a pruned target need two-hop inspection.
        if (expand_two_hop) {
            for (size_t i = 0; i < curr_size; i++) {
                const tableint two_hop_edge = one_hop_snapshot[i];
                const tableint two_hop_id = getPhysicalId(two_hop_edge);
                const uint8_t two_hop_generation = getEdgeGeneration(two_hop_edge);
                if (two_hop_generation == getGeneration(two_hop_id)) {
                    bool two_hop_lost_edge = false;
                    {
                        NodeLockGuard two_hop_guard(getNodeLock(two_hop_id));
                        if (getGeneration(two_hop_id) != two_hop_generation ||
                            isDeleted(two_hop_id))
                            continue;
                        if (removeForwardLinkUnlocked(two_hop_id, deleted_id, deleted_generation)) {
                            bumpMutationVersion(two_hop_id);
                            two_hop_lost_edge = true;
                        }
                        if (getPrunedState(two_hop_id) != 0 && !two_hop_lost_edge) {
                            removeWitnessTailEntryUnlocked(two_hop_id, physical_slots, target);
                        }
                    }
                    if (two_hop_lost_edge) {
                        targets.push_back(two_hop_edge);
                    }
                }
            }
        }

        if (lost_edge) {
            targets.push_back(victim_edge);
        }
    };

    for (size_t i = physical_slots - witness_size; i < physical_slots; ++i)
        process_candidate(adjacency.neighbors[i], false);
    for (size_t i = 0; i < primary_size; ++i)
        process_candidate(adjacency.neighbors[i], pruned);
    // Also repair nodes that lost an incoming edge from the deleted target.
    // Append these after discovered sources, preserving the repair order.
    for (size_t i = 0; i < primary_size; ++i) {
        const tableint edge = adjacency.neighbors[i];
        if (!isLiveEdge(edge)) continue;
        const tableint id = getPhysicalId(edge);
        auto existing = std::find_if(targets.begin(), targets.end(),
                                     [id](tableint candidate) { return getPhysicalId(candidate) == id; });
        if (existing == targets.end()) targets.push_back(edge);
        else *existing = edge;
    }
}

template <typename dist_t>
inline void VigorIndex<dist_t>::repairNeighbors(
    const std::vector<tableint>& targets, const std::vector<tableint>& pool) {
    constexpr size_t max_compensate_num = detail::kRepairCompensate;
    const size_t physical_slots = physicalNeighborSlots();
    if (pool.empty() || targets.empty()) return;

    struct CandidateRepairBatch {
        tableint candidate_edge;
        std::vector<tableint> proposals;
    };
    // Keep proposal capacities between layers. Only [0, batch_count) belongs
    // to this repair; a reused entry is reset on its first reference.
    thread_local std::vector<CandidateRepairBatch> candidate_batches;
    thread_local std::unordered_map<tableint, size_t> candidate_positions;
    candidate_positions.clear();
    candidate_positions.reserve(pool.size());
    size_t batch_count = 0;

    auto queue_candidate_repair = [&](tableint candidate_edge, tableint repair_edge) {
        const tableint candidate_id = getPhysicalId(candidate_edge);
        const auto [position, inserted] = candidate_positions.emplace(candidate_id, batch_count);
        if (inserted) {
            if (batch_count == candidate_batches.size()) {
                candidate_batches.push_back({candidate_edge, {}});
            }
            auto& batch = candidate_batches[batch_count++];
            batch.candidate_edge = candidate_edge;
            batch.proposals.clear();
        }
        auto& batch = candidate_batches[position->second];
        if (batch.candidate_edge != candidate_edge) {
            batch.candidate_edge = candidate_edge;
            batch.proposals.clear();
        }
        // Each physical repair item is processed once and all its requests
        // are consecutive. A repeated request can only duplicate the tail.
        if (!batch.proposals.empty() && getPhysicalId(batch.proposals.back()) ==
                                         getPhysicalId(repair_edge))
            batch.proposals.back() = repair_edge;
        else
            batch.proposals.push_back(repair_edge);
    };

    auto choose_nearest = [&](tableint target_edge) -> const std::vector<Candidate>& {
        thread_local std::vector<Candidate> ranked;
        ranked.clear();
        tableint target_id = getPhysicalId(target_edge);
        uint8_t target_gen = getEdgeGeneration(target_edge);
        tableint target_vec_id;
        tableint existing_edges[ALIGNED_M];
        size_t existing_count, max_count;
        {
            NodeLockGuard guard(getNodeLock(target_id));
            const NodeMetaData& target_meta = storage_->getNodeMetaData(target_id);
            if (target_meta.is_deleted.load(std::memory_order_acquire) != 0 ||
                target_meta.generation != target_gen)
                return ranked;
            target_vec_id = target_meta.vector_id;
            existing_count = getNodeSize(target_id);
            const size_t limit = forwardNeighborLimit(target_id);
            if (existing_count >= limit) return ranked;
            max_count = std::min(max_compensate_num, limit - existing_count);
            std::memcpy(existing_edges, storage_->getNode(target_id).neighbors,
                        existing_count * sizeof(tableint));
        }
        if (target_vec_id == kInvalidTableId) return ranked;
        const void* target_vec = storage_->getVector(target_vec_id);

        // The largest distance (and, on ties, largest physical ID) is
        // always the worst retained candidate. A max-heap preserves the
        // previous replacement rule without scanning the retained set.
        ranked.reserve(max_count);
        auto keep_nearest = [&](dist_t d, tableint edge) {
            Candidate candidate(d, edge);
            if (ranked.size() < max_count) {
                ranked.push_back(candidate);
                std::push_heap(ranked.begin(), ranked.end());
            } else if (candidate < ranked.front()) {
                std::pop_heap(ranked.begin(), ranked.end());
                ranked.back() = candidate;
                std::push_heap(ranked.begin(), ranked.end());
            }
        };
        tableint pending_edges[4];
        const void* pending_vectors[4];
        size_t pending_count = 0;
        auto flush_distances = [&] {
            dist_t distances[4];
            if (pending_count == 4) repairDistance4(target_vec, pending_vectors, distances);
            else for (size_t j = 0; j < pending_count; ++j)
                distances[j] = fstdistfunc_(target_vec, pending_vectors[j], dist_func_param_);
            for (size_t j = 0; j < pending_count; ++j)
                keep_nearest(distances[j], pending_edges[j]);
            pending_count = 0;
        };
        for (size_t edge_index = 0; edge_index < pool.size(); ++edge_index) {
            const tableint edge = pool[edge_index];
            tableint id = getPhysicalId(edge);
            uint8_t expected_gen = getEdgeGeneration(edge);
            if (id == target_id) continue;
            // One small stack snapshot per repair item: no per-candidate
            // locking/allocation or distance work for an existing edge.
            // Compare generation too; stale edges still need replacement.
            if (std::find(existing_edges, existing_edges + existing_count, edge) !=
                existing_edges + existing_count)
                continue;
            const NodeMetaData& meta = storage_->getNodeMetaData(id);
            if (meta.is_deleted.load(std::memory_order_acquire) != 0 ||
                meta.generation != expected_gen) {
                continue;
            }
            tableint vec_id = meta.vector_id;
            if (vec_id == kInvalidTableId) continue;
            pending_edges[pending_count] = edge;
            pending_vectors[pending_count++] = storage_->getVector(vec_id);
            if (pending_count == 4) flush_distances();
        }
        if (pending_count) flush_distances();
        std::sort_heap(ranked.begin(), ranked.end());
        return ranked;
    };

    candidate_batches.reserve(std::min(pool.size(), targets.size() * max_compensate_num));

    // Repair each affected node once, but defer candidate-side mutations
    // so all requests to the same candidate can share one commit/prune.
    for (tableint edge : targets) {
        const auto& ranked = choose_nearest(edge);
        for (const auto& cand : ranked) {
            const tableint candidate_id = getPhysicalId(cand.second);
            const uint8_t candidate_gen = getEdgeGeneration(cand.second);
            bool stop;
            const bool connected = ensureRepairEdge(edge, cand.second, &stop);
            if (stop) break;
            if (connected &&
                !isDeleted(candidate_id) && getGeneration(candidate_id) == candidate_gen)
                queue_candidate_repair(cand.second, edge);
        }
    }

    for (size_t i = 0; i < batch_count; ++i) {
        const auto& batch = candidate_batches[i];
        applyCandidateRepairBatch(batch.candidate_edge, batch.proposals, physical_slots);
    }
}

} // namespace vigor
