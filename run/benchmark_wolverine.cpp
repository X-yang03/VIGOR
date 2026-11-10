#define PAPER_NATIVE_BENCHMARK
#include "benchmark_common.h"

namespace paper {
class WolverineEngine : public NativeEngineBase<true> {
    using Base = NativeEngineBase<true>;

  public:
    using Base::Base;
    static const char *name() {
        return "wolverine";
    }
    Stages update(const Config &c, const Plan &p, const DenseVectors &pool, size_t) {
        Stages result;
        Timer repair;
        index->patchDelete(p.removed, 4, static_cast<int>(c.theta_in), static_cast<int>(c.threads));
        result.repair_s = repair.seconds();
        index->ParallelFor(0, p.added.size(), c.threads, [&](size_t i, size_t) {
            const auto label = p.added[i];
            index->addPoint(pool.get(label), label, true);
        });
        return result;
    }
};
} // namespace paper

int main(int argc, char **argv) {
    return paper::main<paper::WolverineEngine>(argc, argv);
}
