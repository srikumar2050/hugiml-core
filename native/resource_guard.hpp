/**
 * resource_guard.hpp — lightweight native memory/timeout guard helpers.
 *
 * Copyright 2026 Srikumar Krishnamoorthy
 * Apache-2.0 License
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#include <unistd.h>
#endif

#if defined(__linux__)
#include <fstream>
#endif

namespace hugiml {

class NativeMemoryError : public std::runtime_error {
public:
    explicit NativeMemoryError(const std::string& msg)
        : std::runtime_error(msg) {}
};

inline uint64_t parse_u64_env(const char* name) {
    const char* v = std::getenv(name);
    if (!v || !*v) return 0;
    char* end = nullptr;
    unsigned long long out = std::strtoull(v, &end, 10);
    if (end == v) return 0;
    return static_cast<uint64_t>(out);
}

inline uint64_t current_rss_bytes() {
#if defined(__linux__)
    std::ifstream f("/proc/self/statm");
    uint64_t size_pages = 0, rss_pages = 0;
    if (f >> size_pages >> rss_pages) {
        long page_size = ::sysconf(_SC_PAGESIZE);
        if (page_size > 0) return rss_pages * static_cast<uint64_t>(page_size);
    }
#endif
    return 0;
}

inline uint64_t address_space_limit_bytes() {
    uint64_t env_limit = parse_u64_env("HUGIML_MAX_NATIVE_BYTES");
    if (env_limit > 0) return env_limit;
#if defined(RLIMIT_AS)
    struct rlimit lim;
    if (::getrlimit(RLIMIT_AS, &lim) == 0 && lim.rlim_cur != RLIM_INFINITY) {
        return static_cast<uint64_t>(lim.rlim_cur);
    }
#endif
    return 0;
}

inline bool mul_overflows_size_t(uint64_t a, uint64_t b) {
    return b != 0 && a > std::numeric_limits<size_t>::max() / b;
}

inline size_t checked_mul_size_t(uint64_t a, uint64_t b, const char* context) {
    if (mul_overflows_size_t(a, b)) {
        throw NativeMemoryError(std::string("HUGIML native memory request overflows size_t in ") + context);
    }
    return static_cast<size_t>(a * b);
}

inline void ensure_native_memory_available(uint64_t requested_bytes,
                                           const std::string& context,
                                           double safety_fraction = 0.90) {
    const uint64_t limit = address_space_limit_bytes();
    if (limit == 0 || requested_bytes == 0) return;
    const uint64_t rss = current_rss_bytes();
    const uint64_t budget = static_cast<uint64_t>(static_cast<double>(limit) * safety_fraction);
    if (rss > budget || requested_bytes > budget - rss) {
        throw NativeMemoryError(
            "HUGIML native OOM guard: refusing allocation/request of " +
            std::to_string(requested_bytes >> 20) + " MiB for " + context +
            " because current RSS is " + std::to_string(rss >> 20) +
            " MiB and the configured address-space budget is " +
            std::to_string(limit >> 20) + " MiB. Reduce n/p/topK/B, use adaptive binning/hotpath, "
            "or raise HUGIML_MAX_NATIVE_BYTES / the process memory limit.");
    }
}

// Non-throwing counterpart of ensure_native_memory_available(), for OPTIONAL
// allocations (e.g. caches that trade memory for speed). Returns false rather
// than raising when the request would not fit, so callers can gracefully fall
// back to a slower/lower-memory code path instead of aborting the whole call.
// Uses a tighter default safety_fraction than the hard guard above, since this
// gates a "nice to have" allocation stacked on top of whatever else the
// caller already budgeted (and already passed its own hard guard) for.
inline bool native_memory_budget_allows(uint64_t requested_bytes,
                                        double safety_fraction = 0.50) {
    if (requested_bytes == 0) return true;
    const uint64_t limit = address_space_limit_bytes();
    if (limit == 0) return true;  // no enforced limit on this platform/config
    const uint64_t rss = current_rss_bytes();
    const uint64_t budget = static_cast<uint64_t>(static_cast<double>(limit) * safety_fraction);
    if (rss > budget) return false;
    return requested_bytes <= (budget - rss);
}

inline void check_timeout_deadline(bool has_deadline,
                                   const std::chrono::steady_clock::time_point& deadline,
                                   const std::string& context) {
    if (has_deadline && std::chrono::steady_clock::now() >= deadline) {
        throw std::runtime_error("hugiml_timeout: " + context + " exceeded timeout_s");
    }
}

}  // namespace hugiml
