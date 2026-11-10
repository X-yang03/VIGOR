#define PAPER_NATIVE_BENCHMARK
#include "benchmark_common.h"

namespace paper {
class ProHnswEngine : public NativeEngineBase<> {
    using Base = NativeEngineBase<>;

  public:
    using Base::Base;
    static const char *name() {
        return "prohnsw";
    }
    Stages update(const Config &c, const Plan &p, const DenseVectors &pool, size_t batch) {
        Stages result;
        std::vector<hnswlib::tableint> removed;
        removed.reserve(p.removed.size());
        for (auto label : p.removed)
            removed.push_back(index->label_lookup_.at(label));
        for (auto label : p.removed)
            index->markDelete(label);
        Timer removal;
        index->removeObsoleteEdges(removed);
        result.remove_s = removal.seconds();
        parallel_for(p.added.size(), c.threads, [&](size_t i) {
            const auto label = p.added[i];
            index->addPoint(pool.get(label), label, true);
        });
        if (batch % c.interval == 0) {
            Timer repair;
            index->repairDisconnectedNodes();
            result.repair_s = repair.seconds();
            omp_set_num_threads(static_cast<int>(c.threads));
            Timer resolve;
            index->resolveEdgeAsymmetry();
            result.resolve_s = resolve.seconds();
        }
        return result;
    }
};
} // namespace paper

int main(int argc, char **argv) {
    return paper::main<paper::ProHnswEngine>(argc, argv);
}
