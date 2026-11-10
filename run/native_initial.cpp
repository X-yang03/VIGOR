#include "benchmark_common.h"
#include <fstream>
#include <filesystem>
#include <stdexcept>
// Isolate canonical construction from the baselines' HierarchicalNSW methods.
#define hnswlib paper_native_initial
#include "hnswlib/hnswlib.h"
#undef hnswlib

namespace paper {
struct NativeInitialIndex::Impl {
    std::unique_ptr<paper_native_initial::SpaceInterface<float>> space;
    std::unique_ptr<paper_native_initial::HierarchicalNSW<float>> index;
    Impl(bool cosine, size_t dim, size_t capacity, size_t M, size_t ef_c, size_t seed) {
        if (cosine)
            space = std::make_unique<paper_native_initial::InnerProductSpace>(dim);
        else
            space = std::make_unique<paper_native_initial::L2Space>(dim);
        index = std::make_unique<paper_native_initial::HierarchicalNSW<float>>(
            space.get(), capacity, M, ef_c, seed, true);
    }
};
NativeInitialIndex::NativeInitialIndex(bool cosine, size_t dim, size_t capacity, size_t M,
                                       size_t ef_c, size_t seed)
    : impl_(std::make_unique<Impl>(cosine, dim, capacity, M, ef_c, seed)) {}
NativeInitialIndex::~NativeInitialIndex() = default;
void NativeInitialIndex::add(const float *vector, size_t label) {
    impl_->index->addPoint(vector, label);
}
void NativeInitialIndex::save(const std::string &path) {
    impl_->index->saveIndex(path);
    if (std::filesystem::file_size(path) != impl_->index->indexFileSize())
        throw std::runtime_error("Incomplete initial index write");
    std::ofstream out(path + ".rng");
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out << impl_->index->level_generator_ << '\n'
        << impl_->index->update_probability_generator_ << '\n';
}
} // namespace paper
