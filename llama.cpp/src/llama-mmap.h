#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <cstdio>

struct llama_file;
struct llama_mmap;
struct llama_mlock;

using llama_files  = std::vector<std::unique_ptr<llama_file>>;
using llama_mmaps  = std::vector<std::unique_ptr<llama_mmap>>;
using llama_mlocks = std::vector<std::unique_ptr<llama_mlock>>;

// This is intentionally a one-time mmap admission decision.  It does not
// participate in any Decode-time memory or Expert-prefetch policy.
enum class llama_mmap_populate_policy {
    DEFAULT,
    POPULATE,
    SKIP,
    AUTO,
};

enum class llama_mmap_populate_decision {
    DEFAULT,
    POPULATE,
    SKIP,
};

enum class llama_mmap_memory_source {
    UNAVAILABLE,
    CGROUP,
    MEMAVAILABLE,
};

struct llama_mmap_populate_admission_input {
    int32_t expert_count = 0;
    int32_t expert_used_count = 0;
    uint64_t total_model_mapping_bytes = 0;
    bool prefetch_requested = false;
    bool numa = false;
};

struct llama_mmap_populate_admission {
    llama_mmap_populate_policy requested_policy = llama_mmap_populate_policy::DEFAULT;
    llama_mmap_populate_decision decision = llama_mmap_populate_decision::DEFAULT;
    llama_mmap_memory_source memory_source = llama_mmap_memory_source::UNAVAILABLE;
    bool legacy_skip_populate = false;
    bool model_is_moe = false;
    bool sparse_moe = false;
    bool fit_ratio_available = false;
    bool prefetch_requested = false;
    bool numa = false;
    int32_t expert_count = 0;
    int32_t expert_used_count = 0;
    uint64_t total_model_mapping_bytes = 0;
    uint64_t memory_current_bytes = 0;
    uint64_t memory_max_bytes = 0;
    uint64_t memory_headroom_bytes = 0;
    double fit_ratio = 0.0;
    double fit_threshold = 1.0;
    const char * reason = "DEFAULT_POLICY";
};

// The environment and Linux memory state are sampled once by this function.
// The returned result is then shared by every mapping of a split GGUF model.
llama_mmap_populate_admission llama_mmap_populate_admit(
        const llama_mmap_populate_admission_input & input);
void llama_mmap_log_populate_admission(const llama_mmap_populate_admission & admission);
const char * llama_mmap_populate_policy_name(llama_mmap_populate_policy policy);
const char * llama_mmap_populate_decision_name(llama_mmap_populate_decision decision);
const char * llama_mmap_memory_source_name(llama_mmap_memory_source source);

struct llama_file {
    llama_file(const char * fname, const char * mode, bool use_direct_io = false);
    llama_file(FILE * file);
    ~llama_file();

    size_t tell() const;
    size_t size() const;

    int file_id() const; // fileno overload

    void seek(size_t offset, int whence) const;

    void read_raw(void * ptr, size_t len);
    void read_raw_unsafe(void * ptr, size_t len);
    void read_aligned_chunk(void * dest, size_t size);
    uint32_t read_u32();

    void write_raw(const void * ptr, size_t len) const;
    void write_u32(uint32_t val) const;

    size_t read_alignment() const;
    bool has_direct_io() const;
private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct llama_mmap {
    llama_mmap(const llama_mmap &) = delete;
    llama_mmap(
            struct llama_file * file,
            size_t prefetch = (size_t) -1,
            bool numa = false,
            const llama_mmap_populate_admission * populate_admission = nullptr);
    ~llama_mmap();

    size_t size() const;
    void * addr() const;

    void unmap_fragment(size_t first, size_t last);

    static const bool SUPPORTED;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

// When LLAMA_MMAP_DECODE_NORMAL=1, move the file descriptions retained by
// mmap from the startup SEQUENTIAL advice back to NORMAL at the first decode
// step.  This is a process-wide one-shot operation and is a no-op by default.
void llama_mmap_decode_normal_once(uint64_t step);

struct llama_mlock {
    llama_mlock();
    ~llama_mlock();

    void init(void * ptr);
    void grow_to(size_t target_size);

    static const bool SUPPORTED;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

size_t llama_path_max();
