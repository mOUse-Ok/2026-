#include "../src/llama-mmap.h"

#include <cstdio>
#include <cstdlib>

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
