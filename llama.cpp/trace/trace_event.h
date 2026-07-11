#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_tensor;
struct llama_ubatch;

enum llm_mem_trace_phase {
    LLM_MEM_TRACE_PHASE_UNKNOWN = 0,
    LLM_MEM_TRACE_PHASE_PREFILL = 1,
    LLM_MEM_TRACE_PHASE_DECODE  = 2,
};

enum llm_mem_trace_sink {
    LLM_MEM_TRACE_SINK_TENSOR = 0,
    LLM_MEM_TRACE_SINK_KV     = 1,
    LLM_MEM_TRACE_SINK_EXPERT = 2,
    LLM_MEM_TRACE_SINK_MEMORY = 3,
};

#ifdef LLM_MEM_TRACE
void llm_mem_trace_init(const char * dir);
void llm_mem_trace_shutdown(void);
int  llm_mem_trace_enabled(void);
int  llm_mem_trace_sink_enabled(int sink);

uint64_t llm_mem_trace_time_ns(void);
uint64_t llm_mem_trace_next_step(void);

void llm_mem_trace_set_ubatch(const struct llama_ubatch * ubatch, int phase, uint64_t step_id);
void llm_mem_trace_clear_ubatch(void);
const struct llama_ubatch * llm_mem_trace_get_ubatch(void);
int  llm_mem_trace_get_phase(void);
uint64_t llm_mem_trace_get_step(void);
void llm_mem_trace_step_begin(void);
void llm_mem_trace_step_end(void);

void llm_mem_trace_token_begin(int token_idx);
void llm_mem_trace_token_end(int token_idx);

void llm_mem_trace_tensor_begin(const struct ggml_tensor * t);
void llm_mem_trace_tensor_end(const struct ggml_tensor * t);
void llm_mem_trace_tensor_loaded(const struct ggml_tensor * t, const char * stage);
void llm_mem_trace_prefetch_expert_layer(int layer, const int * experts, const float * scores, int n_experts, const char * reason);

void llm_mem_trace_kv_set_rows(const struct ggml_tensor * t);
void llm_mem_trace_kv_reuse(uint32_t n_tokens, uint32_t reused);

void llm_mem_trace_moe_weights(const struct ggml_tensor * t);

void llm_mem_trace_memory_sample(const char * reason);

void llm_mem_trace_write(int sink, const char * line, size_t len);
#else
static inline void llm_mem_trace_init(const char * dir) { (void) dir; }
static inline void llm_mem_trace_shutdown(void) {}
static inline int  llm_mem_trace_enabled(void) { return 0; }
static inline int  llm_mem_trace_sink_enabled(int sink) { (void) sink; return 0; }

static inline uint64_t llm_mem_trace_time_ns(void) { return 0; }
static inline uint64_t llm_mem_trace_next_step(void) { return 0; }

static inline void llm_mem_trace_set_ubatch(const struct llama_ubatch * ubatch, int phase, uint64_t step_id) { (void) ubatch; (void) phase; (void) step_id; }
static inline void llm_mem_trace_clear_ubatch(void) {}
static inline const struct llama_ubatch * llm_mem_trace_get_ubatch(void) { return NULL; }
static inline int  llm_mem_trace_get_phase(void) { return LLM_MEM_TRACE_PHASE_UNKNOWN; }
static inline uint64_t llm_mem_trace_get_step(void) { return 0; }
static inline void llm_mem_trace_step_begin(void) {}
static inline void llm_mem_trace_step_end(void) {}

static inline void llm_mem_trace_token_begin(int token_idx) { (void) token_idx; }
static inline void llm_mem_trace_token_end(int token_idx) { (void) token_idx; }

static inline void llm_mem_trace_tensor_begin(const struct ggml_tensor * t) { (void) t; }
static inline void llm_mem_trace_tensor_end(const struct ggml_tensor * t) { (void) t; }
static inline void llm_mem_trace_tensor_loaded(const struct ggml_tensor * t, const char * stage) { (void) t; (void) stage; }
static inline void llm_mem_trace_prefetch_expert_layer(int layer, const int * experts, const float * scores, int n_experts, const char * reason) {
    (void) layer; (void) experts; (void) scores; (void) n_experts; (void) reason;
}

static inline void llm_mem_trace_kv_set_rows(const struct ggml_tensor * t) { (void) t; }
static inline void llm_mem_trace_kv_reuse(uint32_t n_tokens, uint32_t reused) { (void) n_tokens; (void) reused; }

static inline void llm_mem_trace_moe_weights(const struct ggml_tensor * t) { (void) t; }

static inline void llm_mem_trace_memory_sample(const char * reason) { (void) reason; }

static inline void llm_mem_trace_write(int sink, const char * line, size_t len) { (void) sink; (void) line; (void) len; }
#endif

#ifdef __cplusplus
}

// RAII guard — ensures llm_mem_trace_clear_ubatch() is called on scope exit
// even if an exception escapes process_ubatch(), preventing a dangling pointer
// in TraceContext::ubatch.
class llm_mem_trace_ubatch_guard {
public:
    llm_mem_trace_ubatch_guard() = default;
    ~llm_mem_trace_ubatch_guard() { llm_mem_trace_clear_ubatch(); }
    // non-copyable, non-movable
    llm_mem_trace_ubatch_guard(const llm_mem_trace_ubatch_guard &) = delete;
    llm_mem_trace_ubatch_guard & operator=(const llm_mem_trace_ubatch_guard &) = delete;
};

#endif
