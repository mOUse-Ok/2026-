#include "../src/llama-mmap.h"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <limits>

#if defined(__linux__)
#include <dirent.h>
#include <unistd.h>

namespace {

int fd_count() {
    DIR * dir = opendir("/proc/self/fd");
    if (!dir) {
        std::perror("opendir /proc/self/fd");
        std::exit(EXIT_FAILURE);
    }

    int count = 0;
    while (readdir(dir)) {
        ++count;
    }
    closedir(dir);
    // Exclude . and ..; the directory fd itself is present while counting.
    return count - 2;
}

void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "test-mmap-phase-advice: %s\n", message);
        std::exit(EXIT_FAILURE);
    }
}

} // namespace
#endif

int main() {
#if !defined(__linux__)
    std::printf("test-mmap-phase-advice: skipped (Linux-only)\n");
    return EXIT_SUCCESS;
#else
    llama_mmap_populate_admission_input sparse_input;
    sparse_input.expert_count = 64;
    sparse_input.expert_used_count = 8;
    sparse_input.total_model_mapping_bytes = 4096;
    sparse_input.prefetch_requested = true;

    setenv("LLAMA_MMAP_SKIP_POPULATE", "1", 1);
    setenv("LLAMA_MMAP_POPULATE_POLICY", "populate", 1);
    require(llama_mmap_populate_admit(sparse_input).decision ==
                    llama_mmap_populate_decision::POPULATE,
            "explicit populate did not override the legacy skip switch");

    setenv("LLAMA_MMAP_POPULATE_POLICY", "skip", 1);
    require(llama_mmap_populate_admit(sparse_input).decision ==
                    llama_mmap_populate_decision::SKIP,
            "explicit skip did not force MAP_POPULATE off");

    setenv("LLAMA_MMAP_POPULATE_POLICY", "default", 1);
    require(llama_mmap_populate_admit(sparse_input).decision ==
                    llama_mmap_populate_decision::SKIP,
            "legacy skip no longer applies under the default policy");

    setenv("LLAMA_MMAP_POPULATE_POLICY", "auto", 1);
    sparse_input.total_model_mapping_bytes = std::numeric_limits<uint64_t>::max();
    const llama_mmap_populate_admission auto_skip = llama_mmap_populate_admit(sparse_input);
    require(auto_skip.sparse_moe, "Sparse-MoE metadata was not recognized");
    require(auto_skip.memory_source != llama_mmap_memory_source::UNAVAILABLE,
            "Linux memory headroom was unexpectedly unavailable");
    require(auto_skip.decision == llama_mmap_populate_decision::SKIP,
            "AUTO did not skip a Sparse-MoE model that cannot fit in headroom");

    sparse_input.total_model_mapping_bytes = 1;
    const llama_mmap_populate_admission auto_default = llama_mmap_populate_admit(sparse_input);
    require(auto_default.decision == llama_mmap_populate_decision::DEFAULT,
            "AUTO did not preserve default behavior for a fitting Sparse-MoE model");

    unsetenv("LLAMA_MMAP_POPULATE_POLICY");
    unsetenv("LLAMA_MMAP_SKIP_POPULATE");

    char path[] = "/tmp/llama-mmap-phase-advice-XXXXXX";
    const int raw_fd = mkstemp(path);
    require(raw_fd != -1, "mkstemp failed");
    require(ftruncate(raw_fd, 4096) == 0, "ftruncate failed");
    close(raw_fd);

    setenv("LLAMA_MMAP_DECODE_NORMAL", "1", 1);
    unsetenv("LLAMA_MMAP_SKIP_SEQUENTIAL_FADVISE");

    const int before = fd_count();
    {
        llama_file file(path, "rb");
        require(fd_count() == before + 1, "llama_file fd count mismatch");
        {
            llama_mmap mapping(&file, 0, false);
            require(fd_count() == before + 2,
                    "Decode NORMAL did not retain exactly one duplicated fd");
            llama_mmap_decode_normal_once(7);
            require(fd_count() == before + 2,
                    "Decode NORMAL unexpectedly changed retained fd count");
        }
        require(fd_count() == before + 1,
                "duplicated fd was not released with mmap destruction");
    }
    require(fd_count() == before, "llama_file fd was not released");

    unlink(path);
    std::printf("test-mmap-phase-advice: all tests passed\n");
    return EXIT_SUCCESS;
#endif
}
