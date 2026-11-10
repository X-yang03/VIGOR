// Adapted from hnswlib (Apache-2.0); see LICENSE.hnswlib.
#pragma once
#include "hnswlib.h"

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace vigor {

template<typename dist_t>
class BruteforceSearch : public AlgorithmInterface<dist_t> {
 public:
    char* data_{nullptr};
    size_t maxelements_{0};
    size_t cur_element_count{0};
    size_t size_per_element_{0};
    size_t data_size_{0};
    DISTFUNC<dist_t> fstdistfunc_{nullptr};
    void* dist_func_param_{nullptr};
    std::mutex index_lock;
    std::unordered_map<labeltype, size_t> dict_external_to_internal;

    explicit BruteforceSearch(SpaceInterface<dist_t>* space) {
        setSpace(space);
    }

    BruteforceSearch(SpaceInterface<dist_t>* space, const std::string& location)
        : BruteforceSearch(space) {
        loadIndex(location, space);
    }

    BruteforceSearch(SpaceInterface<dist_t>* space, size_t maxElements)
        : BruteforceSearch(space) {
        const size_t bytes = allocationSize(maxElements, size_per_element_);
        if (bytes) {
            data_ = static_cast<char*>(std::calloc(1, bytes));
            if (!data_) throw std::bad_alloc();
        }
        maxelements_ = maxElements;
    }

    ~BruteforceSearch() override { std::free(data_); }
    BruteforceSearch(const BruteforceSearch&) = delete;
    BruteforceSearch& operator=(const BruteforceSearch&) = delete;

    void addPoint(const void* datapoint, labeltype label, bool /*replace_deleted*/ = false) override {
        std::lock_guard<std::mutex> lock(index_lock);
        auto found = dict_external_to_internal.find(label);
        size_t index;
        if (found == dict_external_to_internal.end()) {
            if (cur_element_count == maxelements_)
                throw std::runtime_error("The number of elements exceeds the specified limit");
            index = cur_element_count;
            dict_external_to_internal.emplace(label, index);
            ++cur_element_count;
        } else {
            index = found->second;
        }
        std::memcpy(data_ + size_per_element_ * index, datapoint, data_size_);
        std::memcpy(data_ + size_per_element_ * index + data_size_, &label, sizeof(label));
    }

    void removePoint(labeltype label) {
        std::lock_guard<std::mutex> lock(index_lock);
        const auto found = dict_external_to_internal.find(label);
        if (found == dict_external_to_internal.end()) return;
        const size_t index = found->second;
        dict_external_to_internal.erase(found);
        const size_t last = cur_element_count - 1;
        if (index != last) {
            const labeltype moved_label = labelAt(last);
            dict_external_to_internal.at(moved_label) = index;
            std::memcpy(data_ + size_per_element_ * index,
                        data_ + size_per_element_ * last, size_per_element_);
        }
        std::memset(data_ + size_per_element_ * last, 0, size_per_element_);
        --cur_element_count;
    }

    std::priority_queue<std::pair<dist_t, labeltype>>
    searchKnn(const void* query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const override {
        std::priority_queue<std::pair<dist_t, labeltype>> result;
        k = std::min(k, cur_element_count);
        if (!k) return result;
        for (size_t i = 0; i < cur_element_count; ++i) {
            const labeltype label = labelAt(i);
            if (isIdAllowed && !(*isIdAllowed)(label)) continue;
            const dist_t distance = fstdistfunc_(query_data, data_ + size_per_element_ * i, dist_func_param_);
            if (result.size() < k || distance <= result.top().first) {
                result.emplace(distance, label);
                if (result.size() > k) result.pop();
            }
        }
        return result;
    }

    // Keep native brute-force serialization: three size_t fields, then the
    // full capacity of vector/label records. Graph caches remain VIGOR v5.
    void saveIndex(const std::string& location) override {
        std::lock_guard<std::mutex> lock(index_lock);
        std::ofstream output(location, std::ios::binary);
        if (!output) throw std::runtime_error("Cannot open brute-force index: " + location);
        writeBinaryPOD(output, maxelements_);
        writeBinaryPOD(output, size_per_element_);
        writeBinaryPOD(output, cur_element_count);
        if (maxelements_) output.write(data_, allocationSize(maxelements_, size_per_element_));
        if (!output) throw std::runtime_error("Cannot write brute-force index: " + location);
    }

    void loadIndex(const std::string& location, SpaceInterface<dist_t>* space) {
        std::lock_guard<std::mutex> lock(index_lock);
        std::ifstream input(location, std::ios::binary | std::ios::ate);
        if (!input) throw std::runtime_error("Cannot open brute-force index: " + location);
        const auto file_size = input.tellg();
        input.seekg(0);
        size_t capacity = 0, stride = 0, count = 0;
        readBinaryPOD(input, capacity);
        readBinaryPOD(input, stride);
        readBinaryPOD(input, count);
        if (!space || space->get_data_size() > std::numeric_limits<size_t>::max() - sizeof(labeltype))
            throw std::runtime_error("Invalid brute-force space");
        const size_t expected_stride = space->get_data_size() + sizeof(labeltype);
        if (!input || stride != expected_stride || count > capacity)
            throw std::runtime_error("Invalid brute-force index header");
        const size_t bytes = allocationSize(capacity, stride);
        if (file_size < 0 || static_cast<uintmax_t>(file_size) != bytes + 3 * sizeof(size_t))
            throw std::runtime_error("Invalid brute-force index length");
        std::unique_ptr<char, decltype(&std::free)> data(
            bytes ? static_cast<char*>(std::malloc(bytes)) : nullptr, &std::free);
        if (bytes && !data) throw std::bad_alloc();
        if (bytes) input.read(data.get(), bytes);
        if (!input) throw std::runtime_error("Cannot read brute-force index");
        std::unordered_map<labeltype, size_t> lookup;
        for (size_t i = 0; i < count; ++i) {
            labeltype label;
            std::memcpy(&label, data.get() + stride * i + space->get_data_size(), sizeof(label));
            if (!lookup.emplace(label, i).second)
                throw std::runtime_error("Duplicate brute-force label");
        }
        setSpace(space);
        std::free(data_);
        data_ = data.release();
        maxelements_ = capacity;
        cur_element_count = count;
        dict_external_to_internal.swap(lookup);
    }

 private:
    void setSpace(SpaceInterface<dist_t>* space) {
        if (!space || !space->get_data_size() ||
            space->get_data_size() > std::numeric_limits<size_t>::max() - sizeof(labeltype))
            throw std::invalid_argument("Invalid brute-force space");
        data_size_ = space->get_data_size();
        size_per_element_ = data_size_ + sizeof(labeltype);
        fstdistfunc_ = space->get_dist_func();
        dist_func_param_ = space->get_dist_func_param();
    }

    static size_t allocationSize(size_t capacity, size_t stride) {
        if (!stride || capacity > (std::numeric_limits<size_t>::max() - 3 * sizeof(size_t)) / stride ||
            capacity > static_cast<size_t>(std::numeric_limits<std::streamsize>::max()) / stride)
            throw std::runtime_error("Brute-force index is too large");
        return capacity * stride;
    }

    labeltype labelAt(size_t index) const {
        labeltype label;
        std::memcpy(&label, data_ + size_per_element_ * index + data_size_, sizeof(label));
        return label;
    }
};
}  // namespace vigor
