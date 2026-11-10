#define PAPER_NATIVE_BENCHMARK
#include "benchmark_common.h"

namespace paper {
class HnswEngine : public NativeEngineBase<> {
    using Base = NativeEngineBase<>;

  public:
    using Base::Base;
    static const char *name() {
        return "hnsw";
    }
    Stages update(const Config &c, const Plan &p, const DenseVectors &pool, size_t) {
        for (auto label : p.removed)
            index->markDelete(label);
        parallel_for(p.added.size(), c.threads, [&](size_t i) {
            const auto label = p.added[i];
            index->addPoint(pool.get(label), label, true);
        });
        return {};
    }
};
} // namespace paper

int main(int argc, char **argv) {
    return paper::main<paper::HnswEngine>(argc, argv);
}
