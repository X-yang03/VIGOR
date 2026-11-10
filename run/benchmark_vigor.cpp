#include "benchmark_common.h"
#include "vigor_alg.h"

namespace paper {
class VigorEngine {
    using Index = vigor::VigorIndex<float>;
    std::unique_ptr<vigor::SpaceInterface<float>> space_;

  public:
    static constexpr bool native = false;
    static const char *name() {
        return "vigor";
    }
    std::unique_ptr<Index> index;
    VigorEngine(const Config &c, size_t dim) {
        if (c.cosine())
            space_ = std::make_unique<vigor::InnerProductSpace>(dim);
        else
            space_ = std::make_unique<vigor::L2Space>(dim);
    }
    void create(const Config &c) {
        index = std::make_unique<Index>(space_.get(), c.capacity, c.M, c.ef_c, c.seed);
    }
    void load(const fs::path &path, const Config &c) {
        index.reset();
        index = std::make_unique<Index>(space_.get(), 1, c.M, c.ef_c, c.seed);
        index->loadIndex(path.string());
        if (index->storage_->getActualVectorCount() != c.initial)
            throw std::runtime_error("Unexpected cached live corpus");
    }
    void add_initial(const float *vector, size_t label) {
        index->addPoint(vector, label);
    }
    void save_initial(const fs::path &path) {
        index->saveIndex(path.string());
    }
    void finish_initial(const fs::path &, const Config &) {}
    void configure(const Config &c) {
        index->setEf(c.ef_s);
    }
    Graph graph() const {
        const auto &s = *index->storage_;
        Graph g;
        g.total = s.getActualVectorCount();
        size_t edges = 0;
        for (size_t id = 0; id < s.node_id_allocator_.getWatermark(); ++id) {
            if (s.getVectorId(id) == vigor::kInvalidTableId ||
                s.getDownNodeId(id) != vigor::kInvalidTableId || s.isDeleted(id))
                continue;
            ++g.valid;
            const auto &node = s.getNode(id);
            for (size_t j = 0; j < s.getSize(id); ++j) {
                const auto other = vigor::getPhysicalId(node.neighbors[j]);
                if (s.getGeneration(other) == vigor::getEdgeGeneration(node.neighbors[j]) &&
                    !s.isDeleted(other))
                    ++edges;
            }
        }
        if (g.valid)
            g.degree = double(edges) / g.valid;
        return g;
    }
    Stages update(const Config &c, const Plan &p, const DenseVectors &pool, size_t) {
        index->lazyCompact.store(false, std::memory_order_release);
        for (auto label : p.removed)
            index->markDelete(label);
        std::vector<int> levels(p.added.size());
        for (auto& level : levels)
            level = index->getRandomLevel(index->mult_);
        parallel_for(p.added.size(), c.threads, [&](size_t i) {
            const auto label = p.added[i];
            index->addPoint(pool.get(label), label, true, levels[i]);
        });
        index->lazyCompact.store(true, std::memory_order_release);
        return {};
    }
};
} // namespace paper

int main(int argc, char **argv) {
    return paper::main<paper::VigorEngine>(argc, argv);
}
