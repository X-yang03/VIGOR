#pragma once

// Template definitions included by vigor_alg.h after VigorIndex is complete.
namespace vigor {

template <typename dist_t>
inline size_t VigorIndex<dist_t>::trimWitnessTailUnlocked(tableint node_id, size_t max_slots)
    const { // Bound the tail by the unoccupied physical slots.
    uint8_t pruned = getPrunedState(node_id);
    if (pruned == 0) return 0;
    const size_t node_size = getNodeSize(node_id);
    const size_t available = max_slots > node_size ? max_slots - node_size : 0;
    if (available == 0) {
        storage_->setPruned(node_id, kPrunedNoWitness);
        return 0;
    }
    if (pruned == kPrunedNoWitness) return 0;
    if (pruned > available) {
        pruned = static_cast<uint8_t>(available);
        storage_->setPruned(node_id, pruned);
    }
    return pruned;
}

template <typename dist_t>
inline void VigorIndex<dist_t>::notePruneEventUnlocked(tableint node_id,
                                                       size_t max_slots) const {
    if (getPrunedState(node_id) == 0) storage_->setPruned(node_id, kPrunedNoWitness);
    trimWitnessTailUnlocked(node_id, max_slots);
}

template <typename dist_t>
inline size_t VigorIndex<dist_t>::witnessTailCountUnlocked(tableint node_id,
                                                           size_t max_slots) const {
    return trimWitnessTailUnlocked(node_id, max_slots);
}

template <typename dist_t>
inline bool VigorIndex<dist_t>::forwardListContainsIdUnlocked(const AlignedNode& node,
                                                              size_t node_size, tableint id) const {
    for (size_t i = 0; i < node_size; i++) {
        if (getPhysicalId(node.neighbors[i]) == id) return true;
    }
    return false;
}

template <typename dist_t>
inline void VigorIndex<dist_t>::mergeWitnessTailUnlocked(tableint node_id, size_t max_slots,
                                                  size_t existing_count,
                                                  const std::vector<tableint>& filtered_ids,
                                                  std::vector<tableint>& merged) const {
    auto node = storage_->getNode(node_id);
    size_t node_size = getNodeSize(node_id);
    merged.clear();
    merged.reserve(existing_count + filtered_ids.size());

    auto merge_current = [&](tableint witness_edge) {
        const tableint wid = getPhysicalId(witness_edge);
        // Recheck both old records and proposals: a source may have been
        // reused since appendWitnessTail checked its reverse link.
        if (wid == node_id || isDeleted(wid) ||
            getGeneration(wid) != getEdgeGeneration(witness_edge) ||
            forwardListContainsIdUnlocked(node, node_size, wid))
            return;
        auto existing = std::find_if(merged.begin(), merged.end(),
                                     [wid](tableint edge) { return getPhysicalId(edge) == wid; });
        if (existing == merged.end()) {
            merged.push_back(witness_edge);
        } else {
            // A newly validated generation supersedes an earlier snapshot
            // of the same slot, without reordering unrelated witnesses.
            *existing = witness_edge;
        }
    };
    size_t old_start = max_slots - existing_count;
    for (size_t i = old_start; i < max_slots; i++) {
        merge_current(node.neighbors[i]);
    }
    for (tableint witness_edge : filtered_ids) {
        merge_current(witness_edge);
    }
}

template <typename dist_t>
inline void VigorIndex<dist_t>::evictRecoverableWitnesses(
    tableint node_id, const std::vector<tableint>& first_hop_edges,
    std::vector<tableint>& merged, size_t capacity, std::vector<tableint>& evicted_ids) const {
    evicted_ids.clear();
    if (merged.size() <= capacity) return;
    evicted_ids.reserve(merged.size() - capacity);

    // Removing a match also handles repeated two-hop references. Only keep
    // actual evictions for the final merge under the target lock.
    auto evict = [&](tableint id) {
        for (auto witness = merged.begin(); witness != merged.end(); ++witness) {
            if (getPhysicalId(*witness) != id) continue;
            evicted_ids.push_back(id);
            merged.erase(witness);
            break;
        }
        return merged.size() <= capacity;
    };

    for (tableint edge : first_hop_edges) {
        tableint nb_id = getPhysicalId(edge);
        uint8_t nb_gen = getEdgeGeneration(edge);
        if (nb_id == node_id || isDeleted(nb_id) || getGeneration(nb_id) != nb_gen) continue;

        // Witnesses recoverable from the normal local neighborhood are lower priority.
        if (evict(nb_id)) return;

        NodeLockGuard guard(getNodeLock(nb_id));
        if (isDeleted(nb_id) || getGeneration(nb_id) != nb_gen) continue;
        auto nb_node = storage_->getNode(nb_id);
        size_t nb_size = getNodeSize(nb_id);
        for (size_t i = 0; i < nb_size; i++) {
            tableint two_hop_edge = nb_node.neighbors[i];
            tableint two_hop_id = getPhysicalId(two_hop_edge);
            uint8_t two_hop_gen = getEdgeGeneration(two_hop_edge);
            if (two_hop_id == node_id || isDeleted(two_hop_id) ||
                getGeneration(two_hop_id) != two_hop_gen) {
                continue;
            }
            if (evict(two_hop_id)) return;
        }
    }
}

template <typename dist_t>
inline void VigorIndex<dist_t>::trimWitnessOverflow(std::vector<tableint>& merged, size_t capacity,
                                             const std::vector<tableint>& evicted_ids) const {
    // Reapply discovery-order evictions to the latest tail, without discarding
    // witnesses added by another caller while the target lock was released.
    for (tableint id : evicted_ids) {
        if (merged.size() <= capacity) return;
        for (auto witness = merged.begin(); witness != merged.end(); ++witness) {
            if (getPhysicalId(*witness) != id) continue;
            merged.erase(witness);
            break;
        }
    }

    if (merged.size() > capacity) {
        merged.erase(merged.begin(), merged.end() - static_cast<ptrdiff_t>(capacity));
    }
}

template <typename dist_t>
inline bool VigorIndex<dist_t>::hasForwardLinkTo(tableint from_id, tableint target_id,
                                          uint8_t target_gen) const {
    auto from_node = storage_->getNode(from_id);
    if (isDeleted(from_id)) return false;

    NodeLockGuard guard(getNodeLock(from_id));
    if (isDeleted(from_id)) return false;

    size_t from_size = getNodeSize(from_id);
    for (size_t i = 0; i < from_size; i++) {
        if (getPhysicalId(from_node.neighbors[i]) == target_id &&
            getEdgeGeneration(from_node.neighbors[i]) == target_gen) {
            return true;
        }
    }
    return false;
}

template <typename dist_t>
inline void VigorIndex<dist_t>::appendWitnessTail(tableint node_id, size_t max_slots,
                                           const std::vector<tableint>& eliminated_ids,
                                           uint8_t target_gen) {
    if (eliminated_ids.empty()) return;

    std::vector<tableint> filtered_ids; //被淘汰且指向node_id的结点（打包edge格式）
    filtered_ids.reserve(eliminated_ids.size());
    for (tableint candidate : eliminated_ids) {
        tableint candidate_id = getPhysicalId(candidate);
        if (candidate_id == node_id) continue;
        if (isDeleted(candidate_id) || getGeneration(candidate_id) != getEdgeGeneration(candidate))
            continue;
        if (hasForwardLinkTo(candidate_id, node_id, target_gen)) {
            filtered_ids.push_back(candidate);
        }
    }
    if (filtered_ids.empty()) return;

    std::vector<tableint> merged;
    std::vector<tableint> first_hop_edges;
    size_t capacity;
    // Caller marks the prune when committing primary adjacency. Publish the
    // merged tail under the same target lock used to check its current capacity.
    auto commit_tail = [&](size_t node_size) {
        const size_t count = merged.size();
        const size_t start = max_slots - count;
        assert(start >= node_size);
        auto node = storage_->getNode(node_id);
        std::copy(merged.begin(), merged.end(), node.neighbors + start);
        setPrunedState(node_id, count ? static_cast<uint8_t>(count) : kPrunedNoWitness);
    };
    {
        auto node = storage_->getNode(node_id);
        NodeLockGuard guard(getNodeLock(node_id));
        if (isDeleted(node_id) || getGeneration(node_id) != target_gen) return;

        size_t node_size = getNodeSize(node_id);
        capacity = max_slots > node_size ? max_slots - node_size : 0;
        if (capacity == 0) {
            setPrunedState(node_id, kPrunedNoWitness);
            return;
        }

        size_t existing_count = witnessTailCountUnlocked(node_id, max_slots);
        mergeWitnessTailUnlocked(node_id, max_slots, existing_count, filtered_ids, merged);

        if (merged.size() > capacity) {
            first_hop_edges.assign(node.neighbors, node.neighbors + node_size);
        } else {
            commit_tail(node_size);
            return;
        }
    }

    std::vector<tableint> evicted_ids;
    evictRecoverableWitnesses(node_id, first_hop_edges, merged, capacity, evicted_ids);

    {
        NodeLockGuard guard(getNodeLock(node_id));
        if (isDeleted(node_id) || getGeneration(node_id) != target_gen) return;

        size_t node_size = getNodeSize(node_id);
        capacity = max_slots > node_size ? max_slots - node_size : 0;
        if (capacity == 0) {
            setPrunedState(node_id, kPrunedNoWitness);
            return;
        }

        size_t existing_count = witnessTailCountUnlocked(node_id, max_slots);
        mergeWitnessTailUnlocked(node_id, max_slots, existing_count, filtered_ids, merged);
        trimWitnessOverflow(merged, capacity, evicted_ids);
        commit_tail(node_size);
    }
}

//删除尾部记录
template <typename dist_t>
template <bool MatchPhysicalId>
inline bool VigorIndex<dist_t>::removeWitnessTailEntryUnlocked(tableint node_id, size_t max_slots,
                                                        tableint witness_edge) {
    auto node = storage_->getNode(node_id);
    size_t count = witnessTailCountUnlocked(node_id, max_slots);
    if (count == 0) return false;

    size_t start = max_slots - count;
    for (size_t i = start; i < max_slots; i++) {
        if constexpr (MatchPhysicalId) {
            if (getPhysicalId(node.neighbors[i]) != getPhysicalId(witness_edge)) continue;
        } else {
            if (node.neighbors[i] != witness_edge) continue;
        }
        node.neighbors[i] = node.neighbors[start];
        setPrunedState(node_id, (count == 1) ? kPrunedNoWitness : static_cast<uint8_t>(count - 1));
        return true;
    }
    return false;
}

template <typename dist_t>
inline bool VigorIndex<dist_t>::removeWitnessTailEntry(tableint node_id, size_t max_slots,
                                                tableint witness_edge) {
    if (isDeleted(node_id) || getPrunedState(node_id) == 0) return false;
    NodeLockGuard guard(getNodeLock(node_id));
    if (isDeleted(node_id) || getPrunedState(node_id) == 0) return false;
    return removeWitnessTailEntryUnlocked(node_id, max_slots, witness_edge);
}

template <typename dist_t>
inline void VigorIndex<dist_t>::removePrimaryDuplicateWitnessUnlocked(tableint node_id,
                                                                      size_t max_slots,
                                                                      tableint edge) {
    // Promotion also removes an older witness for this slot. Ordinary
    // retirement cleanup must retain exact-generation matching.
    removeWitnessTailEntryUnlocked<true>(node_id, max_slots, edge);
}

// If src loses an outgoing edge to dst, dst may still keep src in its witness
// tail. Remove it with swap-pop so the tail remains compact and useful.
template <typename dist_t>
inline void VigorIndex<dist_t>::compactWitnessTailsForRemovedOutgoingEdges(
    tableint src_edge, const tableint* old_edges, size_t old_sz, const tableint* new_edges,
    size_t new_sz, size_t max_slots) {
    tableint src_id = getPhysicalId(src_edge);
    for (size_t i = 0; i < old_sz; i++) {
        if (old_edges[i] == kInvalidTableId) continue;
        tableint dst_id = getPhysicalId(old_edges[i]);
        uint8_t dst_gen = getEdgeGeneration(old_edges[i]);
        if (dst_id == src_id || isDeleted(dst_id) || getGeneration(dst_id) != dst_gen) continue;
        if (getPrunedState(dst_id) == 0) continue;

        if (std::find(new_edges, new_edges + new_sz, old_edges[i]) == new_edges + new_sz) {
            removeWitnessTailEntry(dst_id, max_slots, src_edge);
        }
    }
}

} // namespace vigor
