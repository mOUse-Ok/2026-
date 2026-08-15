#include "llama-mmap.h"

#include "llama-impl.h"

#include "trace_event.h"

#include "ggml.h"

#include <cstring>
#include <climits>
#include <cstdlib>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <cerrno>
#include <algorithm>
#include <limits>
#include <mutex>
#include <string>

#ifdef __has_include
    #if __has_include(<unistd.h>)
        #include <unistd.h>
        #include <fcntl.h>
        #include <sys/stat.h>
        #if defined(_POSIX_MAPPED_FILES)
            #include <sys/mman.h>
        #endif
        #if defined(_POSIX_MEMLOCK_RANGE)
            #include <sys/resource.h>
        #endif
    #endif
#endif

#if defined(__linux__) && defined(_POSIX_MAPPED_FILES)
namespace {

bool parse_u64_text(const char * text, uint64_t & value) {
    if (!text || !*text) {
        return false;
    }
    errno = 0;
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (end == text || errno == ERANGE) {
        return false;
    }
    while (*end && std::isspace((unsigned char) *end)) {
        ++end;
    }
    if (*end) {
        return false;
    }
    value = (uint64_t) parsed;
    return true;
}

bool read_first_line(const std::string & path, std::string & line) {
    FILE * file = std::fopen(path.c_str(), "r");
    if (!file) {
        return false;
    }
    char buffer[4096];
    const bool read = std::fgets(buffer, sizeof(buffer), file) != nullptr;
    std::fclose(file);
    if (!read) {
        return false;
    }
    line = buffer;
    return true;
}

bool read_u64_file(const std::string & path, uint64_t & value) {
    std::string line;
    return read_first_line(path, line) && parse_u64_text(line.c_str(), value);
}

std::string current_cgroup_v2_dir_for_mmap_admission() {
    FILE * file = std::fopen("/proc/self/cgroup", "r");
    if (!file) {
        return {};
    }

    char line[4096];
    std::string relative;
    while (std::fgets(line, sizeof(line), file)) {
        if (std::strncmp(line, "0::", 3) == 0) {
            relative = line + 3;
            break;
        }
    }
    std::fclose(file);

    while (!relative.empty() && std::isspace((unsigned char) relative.back())) {
        relative.pop_back();
    }
    if (relative.empty() || relative == "/") {
        return relative.empty() ? std::string() : "/sys/fs/cgroup";
    }
    return "/sys/fs/cgroup/" + (relative.front() == '/' ? relative.substr(1) : relative);
}

bool read_memavailable_bytes(uint64_t & value) {
    FILE * file = std::fopen("/proc/meminfo", "r");
    if (!file) {
        return false;
    }

    char line[4096];
    bool found = false;
    while (std::fgets(line, sizeof(line), file)) {
        static constexpr const char * key = "MemAvailable:";
        if (std::strncmp(line, key, std::strlen(key)) != 0) {
            continue;
        }
        unsigned long long parsed_kib = 0;
        found = std::sscanf(line + std::strlen(key), "%llu", &parsed_kib) == 1;
        const uint64_t kib = (uint64_t) parsed_kib;
        if (found && kib <= UINT64_MAX / 1024) {
            value = kib * 1024;
        } else {
            found = false;
        }
        break;
    }
    std::fclose(file);
    return found;
}

llama_mmap_populate_policy parse_populate_policy(bool & valid) {
    valid = true;
    const char * value = std::getenv("LLAMA_MMAP_POPULATE_POLICY");
    if (!value || !*value || std::strcmp(value, "default") == 0) {
        return llama_mmap_populate_policy::DEFAULT;
    }
    if (std::strcmp(value, "populate") == 0) {
        return llama_mmap_populate_policy::POPULATE;
    }
    if (std::strcmp(value, "skip") == 0) {
        return llama_mmap_populate_policy::SKIP;
    }
    if (std::strcmp(value, "auto") == 0) {
        return llama_mmap_populate_policy::AUTO;
    }
    valid = false;
    return llama_mmap_populate_policy::DEFAULT;
}

double parse_fit_threshold() {
    const char * value = std::getenv("LLAMA_MMAP_AUTO_POPULATE_FIT_THRESHOLD");
    if (!value || !*value) {
        return 1.0;
    }
    char * end = nullptr;
    const double parsed = std::strtod(value, &end);
    while (end && *end && std::isspace((unsigned char) *end)) {
        ++end;
    }
    return end != value && end && !*end && std::isfinite(parsed) && parsed > 0.0 ? parsed : 1.0;
}

struct mmap_phase_advice_state {
    std::mutex mutex;
    std::vector<int> fds;
    bool transition_attempted = false;
};

mmap_phase_advice_state & mmap_phase_advice_state_instance() {
    static mmap_phase_advice_state state;
    return state;
}

bool mmap_decode_normal_enabled() {
    const char * value = std::getenv("LLAMA_MMAP_DECODE_NORMAL");
    return value && std::strcmp(value, "1") == 0;
}

bool mmap_phase_advice_register_fd(int fd) {
    auto & state = mmap_phase_advice_state_instance();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.transition_attempted) {
        return false;
    }
    try {
        state.fds.push_back(fd);
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

void mmap_phase_advice_unregister_fd(int fd) {
    auto & state = mmap_phase_advice_state_instance();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto & fds = state.fds;
    fds.erase(std::remove(fds.begin(), fds.end(), fd), fds.end());
}

} // namespace
#endif

const char * llama_mmap_populate_policy_name(llama_mmap_populate_policy policy) {
    switch (policy) {
        case llama_mmap_populate_policy::DEFAULT:  return "default";
        case llama_mmap_populate_policy::POPULATE: return "populate";
        case llama_mmap_populate_policy::SKIP:     return "skip";
        case llama_mmap_populate_policy::AUTO:     return "auto";
    }
    return "default";
}

const char * llama_mmap_populate_decision_name(llama_mmap_populate_decision decision) {
    switch (decision) {
        case llama_mmap_populate_decision::DEFAULT:  return "DEFAULT";
        case llama_mmap_populate_decision::POPULATE: return "POPULATE";
        case llama_mmap_populate_decision::SKIP:     return "SKIP_POPULATE";
    }
    return "DEFAULT";
}

const char * llama_mmap_memory_source_name(llama_mmap_memory_source source) {
    switch (source) {
        case llama_mmap_memory_source::UNAVAILABLE:  return "unavailable";
        case llama_mmap_memory_source::CGROUP:       return "cgroup";
        case llama_mmap_memory_source::MEMAVAILABLE: return "memavailable";
    }
    return "unavailable";
}

llama_mmap_populate_admission llama_mmap_populate_admit(
        const llama_mmap_populate_admission_input & input) {
    llama_mmap_populate_admission admission;
    admission.expert_count = input.expert_count;
    admission.expert_used_count = input.expert_used_count;
    admission.total_model_mapping_bytes = input.total_model_mapping_bytes;
    admission.prefetch_requested = input.prefetch_requested;
    admission.numa = input.numa;
    admission.model_is_moe = input.expert_count > 0;
    admission.sparse_moe = input.expert_count > 0 && input.expert_used_count > 0 &&
            input.expert_used_count < input.expert_count;
    admission.fit_threshold = 1.0;

    bool valid_policy = true;
#if defined(__linux__) && defined(_POSIX_MAPPED_FILES)
    admission.requested_policy = parse_populate_policy(valid_policy);
    admission.fit_threshold = parse_fit_threshold();
    const char * legacy_skip = std::getenv("LLAMA_MMAP_SKIP_POPULATE");
    admission.legacy_skip_populate = legacy_skip && std::strcmp(legacy_skip, "1") == 0;

    const std::string cgroup_dir = current_cgroup_v2_dir_for_mmap_admission();
    uint64_t memory_current = 0;
    uint64_t memory_max = 0;
    std::string memory_max_text;
    if (!cgroup_dir.empty() && read_u64_file(cgroup_dir + "/memory.current", memory_current) &&
            read_first_line(cgroup_dir + "/memory.max", memory_max_text) &&
            memory_max_text.compare(0, 3, "max") != 0 &&
            parse_u64_text(memory_max_text.c_str(), memory_max)) {
        admission.memory_source = llama_mmap_memory_source::CGROUP;
        admission.memory_current_bytes = memory_current;
        admission.memory_max_bytes = memory_max;
        admission.memory_headroom_bytes = memory_max > memory_current ? memory_max - memory_current : 0;
    } else {
        uint64_t memavailable = 0;
        if (read_memavailable_bytes(memavailable)) {
            admission.memory_source = llama_mmap_memory_source::MEMAVAILABLE;
            admission.memory_headroom_bytes = memavailable;
        }
    }
#else
    (void) valid_policy;
#endif

    if (admission.memory_source != llama_mmap_memory_source::UNAVAILABLE &&
            admission.memory_headroom_bytes > 0) {
        admission.fit_ratio_available = true;
        admission.fit_ratio = (double) admission.total_model_mapping_bytes /
                (double) admission.memory_headroom_bytes;
    }

    if (!valid_policy) {
        admission.reason = "INVALID_POLICY_FALLBACK_DEFAULT";
        return admission;
    }

    switch (admission.requested_policy) {
        case llama_mmap_populate_policy::POPULATE:
            admission.decision = llama_mmap_populate_decision::POPULATE;
            admission.reason = "FORCED_POPULATE";
            return admission;
        case llama_mmap_populate_policy::SKIP:
            admission.decision = llama_mmap_populate_decision::SKIP;
            admission.reason = "FORCED_SKIP";
            return admission;
        case llama_mmap_populate_policy::DEFAULT:
            if (admission.legacy_skip_populate) {
                admission.decision = llama_mmap_populate_decision::SKIP;
                admission.reason = "LEGACY_SKIP_POPULATE";
            }
            return admission;
        case llama_mmap_populate_policy::AUTO:
            break;
    }

    if (!admission.sparse_moe) {
        admission.reason = "NOT_SPARSE_MOE";
    } else if (admission.memory_source == llama_mmap_memory_source::UNAVAILABLE) {
        admission.reason = "MEMORY_HEADROOM_UNAVAILABLE";
    } else if (admission.memory_headroom_bytes == 0 ||
            (admission.fit_ratio_available && admission.fit_ratio > admission.fit_threshold)) {
        admission.decision = llama_mmap_populate_decision::SKIP;
        admission.reason = "SPARSE_MOE_MODEL_EXCEEDS_HEADROOM";
    } else {
        admission.reason = "MODEL_FITS_HEADROOM";
    }
    return admission;
}

void llama_mmap_log_populate_admission(const llama_mmap_populate_admission & admission) {
    const std::string fit_ratio = admission.fit_ratio_available ?
            format("%.6f", admission.fit_ratio) : "unavailable";
    const std::string memory_current = admission.memory_source == llama_mmap_memory_source::CGROUP ?
            std::to_string(admission.memory_current_bytes) : "unavailable";
    const std::string memory_max = admission.memory_source == llama_mmap_memory_source::CGROUP ?
            std::to_string(admission.memory_max_bytes) : "unavailable";
    std::fprintf(stderr,
            "[MMAP_POPULATE_ADMISSION] requested_policy=%s legacy_skip_populate=%s "
            "model_is_moe=%s sparse_moe=%s expert_count=%d expert_used_count=%d "
            "model_bytes=%llu memory_source=%s memory_current=%s memory_max=%s "
            "memory_headroom=%llu fit_ratio=%s fit_threshold=%.6f expected_n_predict=unavailable "
            "prefetch_requested=%s numa=%s decision=%s reason=%s\n",
            llama_mmap_populate_policy_name(admission.requested_policy),
            admission.legacy_skip_populate ? "true" : "false",
            admission.model_is_moe ? "true" : "false",
            admission.sparse_moe ? "true" : "false",
            admission.expert_count, admission.expert_used_count,
            (unsigned long long) admission.total_model_mapping_bytes,
            llama_mmap_memory_source_name(admission.memory_source), memory_current.c_str(), memory_max.c_str(),
            (unsigned long long) admission.memory_headroom_bytes, fit_ratio.c_str(), admission.fit_threshold,
            admission.prefetch_requested ? "true" : "false", admission.numa ? "true" : "false",
            llama_mmap_populate_decision_name(admission.decision), admission.reason);
    llm_mem_trace_mmap_populate_admission(
            llama_mmap_populate_policy_name(admission.requested_policy),
            llama_mmap_populate_decision_name(admission.decision), admission.reason,
            admission.model_is_moe ? 1 : 0, admission.sparse_moe ? 1 : 0,
            admission.expert_count, admission.expert_used_count,
            admission.total_model_mapping_bytes,
            llama_mmap_memory_source_name(admission.memory_source),
            admission.memory_current_bytes, admission.memory_max_bytes, admission.memory_headroom_bytes,
            admission.fit_ratio_available ? 1 : 0, admission.fit_ratio, admission.fit_threshold,
            admission.prefetch_requested ? 1 : 0, admission.numa ? 1 : 0,
            admission.legacy_skip_populate ? 1 : 0);
}

#if defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
    #ifndef PATH_MAX
        #define PATH_MAX MAX_PATH
    #endif
    #include <io.h>
#endif

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

#ifdef _WIN32
#    define llama_mmap_ftell _ftelli64
#    define llama_mmap_fseek _fseeki64
#else
#    define llama_mmap_ftell ftello
#    define llama_mmap_fseek fseeko
#endif

// TODO: consider moving to llama-impl.h if needed in more places
#if defined(_WIN32)
static std::string llama_format_win_err(DWORD err) {
    LPSTR buf;
    size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                 NULL, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&buf, 0, NULL);
    if (!size) {
        return "FormatMessageA failed";
    }
    std::string ret(buf, size);
    LocalFree(buf);
    return ret;
}
#endif

// llama_file

struct llama_file::impl {
#if defined(_WIN32)
    HANDLE fp_win32;
    std::string GetErrorMessageWin32(DWORD error_code) const {
        std::string ret;
        LPSTR lpMsgBuf = NULL;
        DWORD bufLen = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                    NULL, error_code, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&lpMsgBuf, 0, NULL);
        if (!bufLen) {
            ret = format("Win32 error code: %lx", error_code);
        } else {
            ret = lpMsgBuf;
            LocalFree(lpMsgBuf);
        }

        return ret;
    }

    impl(const char * fname, const char * mode, [[maybe_unused]] const bool use_direct_io = false) {
        fp = ggml_fopen(fname, mode);
        if (fp == NULL) {
            throw std::runtime_error(format("failed to open %s: %s", fname, strerror(errno)));
        }
        fp_win32 = (HANDLE) _get_osfhandle(_fileno(fp));
        seek(0, SEEK_END);
        size = tell();
        seek(0, SEEK_SET);
    }

    impl(FILE * file) : owns_fp(false) {
        fp = file;
        fp_win32 = (HANDLE) _get_osfhandle(_fileno(fp));
        seek(0, SEEK_END);
        size = tell();
        seek(0, SEEK_SET);
    }

    size_t tell() const {
        LARGE_INTEGER li;
        li.QuadPart = 0;
        BOOL ret = SetFilePointerEx(fp_win32, li, &li, FILE_CURRENT);
        if (!ret) {
            throw std::runtime_error(format("read error: %s", GetErrorMessageWin32(GetLastError()).c_str()));
        }

        return li.QuadPart;
    }

    void seek(size_t offset, int whence) const {
        static_assert(SEEK_SET == FILE_BEGIN, "SEEK_SET != FILE_BEGIN");
        static_assert(SEEK_CUR == FILE_CURRENT, "SEEK_CUR != FILE_CURRENT");
        static_assert(SEEK_END == FILE_END, "SEEK_END != FILE_END");

        LARGE_INTEGER li;
        li.QuadPart = offset;
        BOOL ret = SetFilePointerEx(fp_win32, li, NULL, whence);
        if (!ret) {
            throw std::runtime_error(format("read error: %s", GetErrorMessageWin32(GetLastError()).c_str()));
        }
    }

    void read_raw(void * ptr, size_t len) {
        size_t bytes_read = 0;
        while (bytes_read < len) {
            size_t chunk_size = std::min<size_t>(len - bytes_read, 64*1024*1024);
            DWORD chunk_read = 0;
            BOOL result = ReadFile(fp_win32, reinterpret_cast<char*>(ptr) + bytes_read, chunk_size, &chunk_read, NULL);
            if (!result) {
                throw std::runtime_error(format("read error: %s", GetErrorMessageWin32(GetLastError()).c_str()));
            }
            if (chunk_read < chunk_size || chunk_read == 0) {
                throw std::runtime_error("unexpectedly reached end of file");
            }

            bytes_read += chunk_read;
        }
    }

    uint32_t read_u32() {
        uint32_t val;
        read_raw(&val, sizeof(val));
        return val;
    }

    void write_raw(const void * ptr, size_t len) const {
        size_t bytes_written = 0;
        while (bytes_written < len) {
            size_t chunk_size = std::min<size_t>(len - bytes_written, 64*1024*1024);
            DWORD chunk_written = 0;
            BOOL result = WriteFile(fp_win32, reinterpret_cast<char const*>(ptr) + bytes_written, chunk_size, &chunk_written, NULL);
            if (!result) {
                throw std::runtime_error(format("write error: %s", GetErrorMessageWin32(GetLastError()).c_str()));
            }
            if (chunk_written < chunk_size || chunk_written == 0) {
                throw std::runtime_error("unexpectedly failed to write bytes");
            }

            bytes_written += chunk_written;
        }
    }

    void write_u32(uint32_t val) const {
        write_raw(&val, sizeof(val));
    }

    bool has_direct_io() const {
        return true;
    }

    ~impl() {
        if (fp && owns_fp) {
            std::fclose(fp);
        }
    }
#else
    impl(const char * fname, const char * mode, [[maybe_unused]] const bool use_direct_io = false) : fname(fname) {
#ifdef __linux__
        // Try unbuffered I/O for read only
        if (use_direct_io && std::strcmp(mode, "rb") == 0) {
            if (init_fd()) {
                return;
            }
            LLAMA_LOG_WARN("Failed to open file '%s' with error: %s. Falling back to buffered I/O",
                           fname, strerror(errno));
        }
#endif
        init_fp(mode);
    }

#ifdef __linux__
    bool init_fd() {
        fd = open(fname.c_str(), O_RDONLY | O_DIRECT);

        if (fd != -1) {
            struct stat file_stats{};
            fstat(fd, &file_stats);

            size = file_stats.st_size;
            alignment = file_stats.st_blksize;

            off_t ret = lseek(fd, 0, SEEK_SET);
            if (ret == -1) {
                throw std::runtime_error(format("seek error: %s", strerror(errno)));
            }
            return true;
        }
        return false;
    }
#endif

    void init_fp(const char * mode) {
        fp = ggml_fopen(fname.c_str(), mode);
        if (fp == NULL) {
            throw std::runtime_error(format("failed to open %s: %s", fname.c_str(), strerror(errno)));
        }
        seek(0, SEEK_END);
        size = tell();
        seek(0, SEEK_SET);
    }

    impl(FILE * file) : fname("(file*)"), owns_fp(false) {
        fp = file;
        seek(0, SEEK_END);
        size = tell();
        seek(0, SEEK_SET);
    }

    size_t tell() const {
        if (fd == -1) {
            off_t ret = llama_mmap_ftell(fp);
            if (ret == -1) {
                throw std::runtime_error(format("ftell error: %s", strerror(errno)));
            }

            return (size_t) ret;
        }

        off_t pos = lseek(fd, 0, SEEK_CUR);
        if (pos == -1) {
            throw std::runtime_error(format("lseek error: %s", strerror(errno)));
        }
        return (size_t) pos;
    }

    void seek(size_t offset, int whence) const {
        off_t ret = 0;
        if (fd == -1) {
            ret = llama_mmap_fseek(fp, offset, whence);
        } else {
            ret = lseek(fd, offset, whence);
        }
        if (ret == -1) {
            throw std::runtime_error(format("seek error: %s", strerror(errno)));
        }
    }

    void read_raw_unsafe(void * ptr, size_t len) {
        if (len == 0) {
            return;
        }
        errno = 0;
        if (fd == -1) {
            const size_t curr_off = tell();
            const size_t to_read = std::min(len, size - curr_off);

            std::size_t ret = std::fread(ptr, to_read, 1, fp);
            if (ferror(fp)) {
                throw std::runtime_error(format("read error: %s", strerror(errno)));
            }
            if (to_read > 0 && ret != 1) {
                throw std::runtime_error("unexpectedly reached end of file");
            }
        } else {
            size_t bytes_read = 0;
            while (bytes_read < len) {
                const size_t to_read = len - bytes_read;
                ssize_t ret = ::read(fd, reinterpret_cast<char *>(ptr) + bytes_read, to_read);

                if (ret == -1) {
                    if (errno == EINTR) {
                        continue;  // Interrupted by signal, retry
                    }
                    // Fallback to std::fread in case the DMA controller cannot access the buffer
                    if (errno == EFAULT || errno == EINVAL) {
                        LLAMA_LOG_WARN("%s: Falling back to buffered IO due to %s\n", __func__, strerror(errno));
                        auto curr_off = tell();
                        close(fd);
                        fd = -1;
                        alignment = 1;
                        init_fp("rb");
                        seek(curr_off, SEEK_SET);
                        read_raw_unsafe(ptr, len);
                        return;
                    }
                    throw std::runtime_error(format("read error: %s", strerror(errno)));
                }
                if (ret == 0) {
                    // EOF: allow if this read was only pulling alignment padding past file end
                    off_t pos = lseek(fd, 0, SEEK_CUR);
                    if (pos != -1 && (size_t) pos == size) {
                        std::memset(reinterpret_cast<char *>(ptr) + bytes_read, 0, len - bytes_read);
                        return;
                    }
                    throw std::runtime_error("unexpectedly reached end of file");
                }

                bytes_read += (size_t) ret;
            }
        }
    }

    void read_aligned_chunk(void * dest, size_t size) {
        size_t offset = tell();
        off_t aligned_offset = offset & ~(alignment - 1);
        off_t offset_from_alignment = offset - aligned_offset;
        size_t bytes_to_read = (offset_from_alignment + size + alignment - 1) & ~(alignment - 1);

        void * raw_buffer = nullptr;
        int ret = posix_memalign(&raw_buffer, alignment, bytes_to_read);
        if (ret != 0) {
            throw std::runtime_error(format("posix_memalign failed with error %d", ret));
        }

        struct aligned_buffer_deleter {
            void operator()(void * p) const { free(p); }
        };
        std::unique_ptr<void, aligned_buffer_deleter> buffer(raw_buffer);

        seek(aligned_offset, SEEK_SET);
        read_raw_unsafe(buffer.get(), bytes_to_read);

        uintptr_t actual_data = reinterpret_cast<uintptr_t>(buffer.get()) + offset_from_alignment;
        memcpy(dest, reinterpret_cast<void *>(actual_data), size);
    }

    void read_raw(void * ptr, size_t len) {
        if (has_direct_io()) {
            read_aligned_chunk(ptr, len);
        } else {
            read_raw_unsafe(ptr, len);
        }
    }

    uint32_t read_u32() {
        uint32_t ret;
        read_raw(&ret, sizeof(ret));
        return ret;
    }

    void write_raw(const void * ptr, size_t len) const {
        if (len == 0) {
            return;
        }
        errno = 0;
        size_t ret = std::fwrite(ptr, len, 1, fp);
        if (ret != 1) {
            throw std::runtime_error(format("write error: %s", strerror(errno)));
        }
    }

    void write_u32(uint32_t val) const {
        write_raw(&val, sizeof(val));
    }

    bool has_direct_io() const {
        return fd != -1 && alignment > 1;
    }

    ~impl() {
        if (fd != -1) {
            close(fd);
        } else if (owns_fp) {
            std::fclose(fp);
        }
    }
    int fd = -1;
    std::string fname;
#endif

    size_t read_alignment() const {
        return alignment;
    }

    size_t alignment = 1;

    FILE * fp{};
    size_t size{};
    bool owns_fp = true;
};

llama_file::llama_file(const char * fname, const char * mode, const bool use_direct_io) :
    pimpl(std::make_unique<impl>(fname, mode, use_direct_io)) {}

llama_file::llama_file(FILE * file) : pimpl(std::make_unique<impl>(file)) {}

llama_file::~llama_file() = default;

size_t llama_file::tell() const { return pimpl->tell(); }
size_t llama_file::size() const { return pimpl->size; }

size_t llama_file::read_alignment() const { return pimpl->read_alignment(); }
bool llama_file::has_direct_io() const { return pimpl->has_direct_io(); }

int llama_file::file_id() const {
#ifdef _WIN32
    return _fileno(pimpl->fp);
#else
    if (pimpl->fd != -1) {
        return pimpl->fd;
    }
#if defined(fileno)
    return fileno(pimpl->fp);
#else
    return ::fileno(pimpl->fp);
#endif
#endif
}

void llama_file::seek(size_t offset, int whence) const { pimpl->seek(offset, whence); }
void llama_file::read_raw(void * ptr, size_t len) { pimpl->read_raw(ptr, len); }
#ifdef _WIN32
void llama_file::read_raw_unsafe(void * ptr, size_t len) { pimpl->read_raw(ptr, len); }
#else
void llama_file::read_raw_unsafe(void * ptr, size_t len) { pimpl->read_raw_unsafe(ptr, len); }
#endif

uint32_t llama_file::read_u32() { return pimpl->read_u32(); }

void llama_file::write_raw(const void * ptr, size_t len) const { pimpl->write_raw(ptr, len); }
void llama_file::write_u32(uint32_t val) const { pimpl->write_u32(val); }

// llama_mmap

struct llama_mmap::impl {
#ifdef _POSIX_MAPPED_FILES
    std::vector<std::pair<size_t, size_t>> mapped_fragments;
#ifdef __linux__
    int decode_normal_fd = -1;
#endif

    impl(
            struct llama_file * file,
            size_t prefetch,
            bool numa,
            const llama_mmap_populate_admission * populate_admission) {
        size = file->size();
        int fd = file->file_id();
        int flags = MAP_SHARED;
        if (numa) { prefetch = 0; }
#ifdef __linux__
        bool sequential_advice_applied = false;
        // This opt-out is intentionally benchmark-only.  It lets a runtime
        // compare the historical whole-file sequential hint with the kernel
        // default, without changing normal mmap behavior.
        const char * skip_sequential_fadvise = std::getenv("LLAMA_MMAP_SKIP_SEQUENTIAL_FADVISE");
        if (skip_sequential_fadvise && std::strcmp(skip_sequential_fadvise, "1") == 0) {
            LLAMA_LOG_WARN("llama_mmap: skipping POSIX_FADV_SEQUENTIAL by request\n");
        } else {
            const int result = posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
            if (result != 0) {
                LLAMA_LOG_WARN("warning: posix_fadvise(.., POSIX_FADV_SEQUENTIAL) failed: %s\n",
                        strerror(errno));
            } else {
                sequential_advice_applied = true;
            }
        }
        const bool skip_populate = populate_admission ?
                populate_admission->decision == llama_mmap_populate_decision::SKIP :
                (std::getenv("LLAMA_MMAP_SKIP_POPULATE") &&
                 std::strcmp(std::getenv("LLAMA_MMAP_SKIP_POPULATE"), "1") == 0);
        if (prefetch && !skip_populate) {
            flags |= MAP_POPULATE;
        }
#endif
        const uint64_t mmap_begin_ts_ns = llm_mem_trace_time_ns();
        addr = mmap(NULL, file->size(), PROT_READ, flags, fd, 0);
        if (addr == MAP_FAILED) {
            throw std::runtime_error(format("mmap failed: %s", strerror(errno)));
        }
        llm_mem_trace_model_mmap(
                mmap_begin_ts_ns, llm_mem_trace_time_ns(), (uint64_t) file->size(),
                (flags & MAP_POPULATE) != 0 ? 1 : 0);

        if (prefetch > 0) {
            if (posix_madvise(addr, std::min(file->size(), prefetch), POSIX_MADV_WILLNEED)) {
                LLAMA_LOG_WARN("warning: posix_madvise(.., POSIX_MADV_WILLNEED) failed: %s\n",
                        strerror(errno));
            }
        }
        if (numa) {
            if (posix_madvise(addr, file->size(), POSIX_MADV_RANDOM)) {
                LLAMA_LOG_WARN("warning: posix_madvise(.., POSIX_MADV_RANDOM) failed: %s\n",
                        strerror(errno));
            }
        }

        mapped_fragments.emplace_back(0, file->size());

        // mmap retains a reference to the file description after the loader's
        // original fd is closed.  dup() keeps that *same* open file
        // description reachable for the Decode NORMAL transition; reopening
        // the path would create a distinct file description and be incorrect.
        if (sequential_advice_applied && mmap_decode_normal_enabled()) {
            const int duplicated_fd = dup(fd);
            if (duplicated_fd == -1) {
                LLAMA_LOG_WARN("llama_mmap: cannot retain fd for Decode NORMAL: %s\n", strerror(errno));
            } else if (mmap_phase_advice_register_fd(duplicated_fd)) {
                decode_normal_fd = duplicated_fd;
            } else {
                LLAMA_LOG_WARN("llama_mmap: cannot register retained fd for Decode NORMAL\n");
                close(duplicated_fd);
            }
        }
    }

    static void align_range(size_t * first, size_t * last, size_t page_size) {
        size_t offset_in_page = *first & (page_size - 1);
        size_t offset_to_page = offset_in_page == 0 ? 0 : page_size - offset_in_page;
        *first += offset_to_page;

        *last = *last & ~(page_size - 1);

        if (*last <= *first) {
            *last = *first;
        }
    }

    void unmap_fragment(size_t first, size_t last) {
        int page_size = sysconf(_SC_PAGESIZE);
        align_range(&first, &last, page_size);
        size_t len = last - first;

        if (len == 0) {
            return;
        }

        GGML_ASSERT(first % page_size == 0);
        GGML_ASSERT(last % page_size == 0);
        GGML_ASSERT(last > first);

        void * next_page_start = (uint8_t *) addr + first;

        if (munmap(next_page_start, len)) {
            LLAMA_LOG_WARN("warning: munmap failed: %s\n", strerror(errno));
        }

        std::vector<std::pair<size_t, size_t>> new_mapped_fragments;
        for (const auto & frag : mapped_fragments) {
            if (frag.first < first && frag.second > last) {
                new_mapped_fragments.emplace_back(frag.first, first);
                new_mapped_fragments.emplace_back(last, frag.second);
            } else if (frag.first < first && frag.second > first) {
                new_mapped_fragments.emplace_back(frag.first, first);
            } else if (frag.first < last && frag.second > last) {
                new_mapped_fragments.emplace_back(last, frag.second);
            } else if (frag.first >= first && frag.second <= last) {
            } else {
                new_mapped_fragments.push_back(frag);
            }
        }
        mapped_fragments = std::move(new_mapped_fragments);
    }

    ~impl() {
#ifdef __linux__
        if (decode_normal_fd != -1) {
            mmap_phase_advice_unregister_fd(decode_normal_fd);
            close(decode_normal_fd);
        }
#endif
        for (const auto & frag : mapped_fragments) {
            if (munmap((char *) addr + frag.first, frag.second - frag.first)) {
                LLAMA_LOG_WARN("warning: munmap failed: %s\n", strerror(errno));
            }
        }
    }
#elif defined(_WIN32)
    HANDLE hMapping = nullptr;

    impl(
            struct llama_file * file,
            size_t prefetch,
            bool numa,
            const llama_mmap_populate_admission * populate_admission) {
        GGML_UNUSED(numa);
        GGML_UNUSED(populate_admission);

        size = file->size();

        HANDLE hFile = (HANDLE) _get_osfhandle(file->file_id());

        hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);

        if (hMapping == NULL) {
            DWORD error = GetLastError();
            throw std::runtime_error(format("CreateFileMappingA failed: %s", llama_format_win_err(error).c_str()));
        }

        addr = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
        DWORD error = GetLastError();

        if (addr == NULL) {
            CloseHandle(hMapping);
            throw std::runtime_error(format("MapViewOfFile failed: %s", llama_format_win_err(error).c_str()));
        }

        if (prefetch > 0) {
#if _WIN32_WINNT >= 0x602
            BOOL (WINAPI *pPrefetchVirtualMemory) (HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
            HMODULE hKernel32 = GetModuleHandleW(L"kernel32.dll");

            pPrefetchVirtualMemory = (decltype(pPrefetchVirtualMemory))(void *) GetProcAddress(hKernel32, "PrefetchVirtualMemory");

            if (pPrefetchVirtualMemory) {
                WIN32_MEMORY_RANGE_ENTRY range;
                range.VirtualAddress = addr;
                range.NumberOfBytes = (SIZE_T) std::min(size, prefetch);
                if (!pPrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0)) {
                    LLAMA_LOG_WARN("warning: PrefetchVirtualMemory failed: %s\n",
                            llama_format_win_err(GetLastError()).c_str());
                }
            }
#else
            LLAMA_LOG_DEBUG("skipping PrefetchVirtualMemory because _WIN32_WINNT < 0x602\n");
#endif
        }
    }

    void unmap_fragment(size_t first, size_t last) {
        GGML_UNUSED(first);
        GGML_UNUSED(last);
    }

    ~impl() {
        if (hMapping) {
            if (addr) {
                if (!UnmapViewOfFile(addr)) {
                    LLAMA_LOG_WARN("warning: UnmapViewOfFile failed: %s\n",
                            llama_format_win_err(GetLastError()).c_str());
                }
            }
            if (!CloseHandle(hMapping)) {
                LLAMA_LOG_WARN("warning: CloseHandle failed: %s\n",
                        llama_format_win_err(GetLastError()).c_str());
            }
        }
    }
#else
    impl(
            struct llama_file * file,
            size_t prefetch,
            bool numa,
            const llama_mmap_populate_admission * populate_admission) {
        GGML_UNUSED(file);
        GGML_UNUSED(prefetch);
        GGML_UNUSED(numa);
        GGML_UNUSED(populate_admission);

        throw std::runtime_error("mmap not supported");
    }

    void unmap_fragment(size_t first, size_t last) {
        GGML_UNUSED(first);
        GGML_UNUSED(last);

        throw std::runtime_error("mmap not supported");
    }
#endif

    void * addr;
    size_t size;
};

llama_mmap::llama_mmap(
        struct llama_file * file,
        size_t prefetch,
        bool numa,
        const llama_mmap_populate_admission * populate_admission) :
        pimpl(std::make_unique<impl>(file, prefetch, numa, populate_admission)) {}
llama_mmap::~llama_mmap() = default;

size_t llama_mmap::size() const { return pimpl->size; }
void * llama_mmap::addr() const { return pimpl->addr; }

void llama_mmap::unmap_fragment(size_t first, size_t last) { pimpl->unmap_fragment(first, last); }

void llama_mmap_decode_normal_once(uint64_t step) {
#if defined(__linux__) && defined(_POSIX_MAPPED_FILES)
    if (!mmap_decode_normal_enabled()) {
        return;
    }

    auto & state = mmap_phase_advice_state_instance();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.transition_attempted) {
        return;
    }
    state.transition_attempted = true;

    int result = 0;
    for (const int fd : state.fds) {
        const int current = posix_fadvise(fd, 0, 0, POSIX_FADV_NORMAL);
        if (result == 0 && current != 0) {
            result = current;
        }
    }
    // Keep this one-shot experimental marker independent of the application's
    // configured log level so subprocess runs can verify the transition.
    std::fprintf(stderr,
            "[MMAP_PHASE_ADVICE] transition=SEQUENTIAL_TO_NORMAL phase=Decode step=%llu files=%zu result=%d\n",
            (unsigned long long) step, state.fds.size(), result);
#else
    GGML_UNUSED(step);
#endif
}

#if defined(_POSIX_MEMLOCK_RANGE) || defined(_WIN32)
const bool llama_mmap::SUPPORTED  = true;
#else
const bool llama_mmap::SUPPORTED  = false;
#endif

// llama_mlock

struct llama_mlock::impl {
#ifdef _POSIX_MEMLOCK_RANGE
    static size_t lock_granularity() {
        return (size_t) sysconf(_SC_PAGESIZE);
    }

    bool raw_lock(const void * addr, size_t size) const {
        if (!mlock(addr, size)) {
            return true;
        }

#ifdef __APPLE__
#define MLOCK_SUGGESTION \
        "Try increasing the sysctl values 'vm.user_wire_limit' and 'vm.global_user_wire_limit' and/or " \
        "decreasing 'vm.global_no_user_wire_amount'.  Also try increasing RLIMIT_MEMLOCK (ulimit -l).\n"
#else
#define MLOCK_SUGGESTION \
        "Try increasing RLIMIT_MEMLOCK ('ulimit -l' as root).\n"
#endif

        char* errmsg = std::strerror(errno);
        bool suggest = (errno == ENOMEM);
#if defined(TARGET_OS_VISION) || defined(TARGET_OS_TV) || defined(_AIX) || defined(__HAIKU__)
        // visionOS/tvOS/Haiku don't support RLIMIT_MEMLOCK
        // Skip resource limit checks on these platforms
        suggest = false;
#else
        struct rlimit lock_limit;
        if (suggest && getrlimit(RLIMIT_MEMLOCK, &lock_limit)) {
            suggest = false;
        }
        if (suggest && ((uint64_t)lock_limit.rlim_max > (uint64_t)lock_limit.rlim_cur + size)) {
            suggest = false;
        }
#endif

        LLAMA_LOG_WARN("warning: failed to mlock %zu-byte buffer (after previously locking %zu bytes): %s\n%s",
                size, this->size, errmsg, suggest ? MLOCK_SUGGESTION : "");
        return false;
    }

    static void raw_unlock(void * addr, size_t size) {
        if (munlock(addr, size)) {
            LLAMA_LOG_WARN("warning: failed to munlock buffer: %s\n", std::strerror(errno));
        }
    }
#elif defined(_WIN32)
    static size_t lock_granularity() {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return (size_t) si.dwPageSize;
    }

    bool raw_lock(void * ptr, size_t len) const {
        for (int tries = 1; ; tries++) {
            if (VirtualLock(ptr, len)) {
                return true;
            }
            if (tries == 2) {
                LLAMA_LOG_WARN("warning: failed to VirtualLock %zu-byte buffer (after previously locking %zu bytes): %s\n",
                    len, size, llama_format_win_err(GetLastError()).c_str());
                return false;
            }

            SIZE_T min_ws_size, max_ws_size;
            if (!GetProcessWorkingSetSize(GetCurrentProcess(), &min_ws_size, &max_ws_size)) {
                LLAMA_LOG_WARN("warning: GetProcessWorkingSetSize failed: %s\n",
                        llama_format_win_err(GetLastError()).c_str());
                return false;
            }
            size_t increment = len + 1048576;
            min_ws_size += increment;
            max_ws_size += increment;
            if (!SetProcessWorkingSetSize(GetCurrentProcess(), min_ws_size, max_ws_size)) {
                LLAMA_LOG_WARN("warning: SetProcessWorkingSetSize failed: %s\n",
                        llama_format_win_err(GetLastError()).c_str());
                return false;
            }
        }
    }

    static void raw_unlock(void * ptr, size_t len) {
        if (!VirtualUnlock(ptr, len)) {
            LLAMA_LOG_WARN("warning: failed to VirtualUnlock buffer: %s\n",
                    llama_format_win_err(GetLastError()).c_str());
        }
    }
#else
    static size_t lock_granularity() {
        return (size_t) 65536;
    }

    bool raw_lock(const void * addr, size_t len) const {
        LLAMA_LOG_WARN("warning: mlock not supported on this system\n");
        return false;
    }

    static void raw_unlock(const void * addr, size_t len) {}
#endif

    impl() : addr(NULL), size(0), failed_already(false) {}

    void init(void * ptr) {
        GGML_ASSERT(addr == NULL && size == 0);
        addr = ptr;
    }

    void grow_to(size_t target_size) {
        GGML_ASSERT(addr);
        if (failed_already) {
            return;
        }
        size_t granularity = lock_granularity();
        target_size = (target_size + granularity - 1) & ~(granularity - 1);
        if (target_size > size) {
            if (raw_lock((uint8_t *) addr + size, target_size - size)) {
                size = target_size;
            } else {
                failed_already = true;
            }
        }
    }

    void * addr;
    size_t size;

    bool failed_already;
};

llama_mlock::llama_mlock() : pimpl(std::make_unique<impl>()) {}
llama_mlock::~llama_mlock() = default;

void llama_mlock::init(void * ptr) { pimpl->init(ptr); }
void llama_mlock::grow_to(size_t target_size) { pimpl->grow_to(target_size); }

#if defined(_POSIX_MEMLOCK_RANGE) || defined(_WIN32)
const bool llama_mlock::SUPPORTED = true;
#else
const bool llama_mlock::SUPPORTED = false;
#endif

size_t llama_path_max() {
    return PATH_MAX;
}
