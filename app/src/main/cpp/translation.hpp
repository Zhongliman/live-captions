#pragma once
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

struct TranslationRequestMetadata {
    int64_t session_id = 0;
    int64_t segment_id = 0;
    int32_t revision = 0;
    // Opaque identity for prompt configuration (language, glossary and template version).
    // It is compared in memory only and is never logged.
    std::string cache_scope;
};

struct TranslationProgress {
    int64_t session_id = 0;
    int64_t segment_id = 0;
    int32_t revision = 0;
    int64_t elapsed_ms = 0;
    std::string text;
    // Native generation callbacks are always provisional. EOS is represented by the
    // completed return value from translateWithProgress(), not by a token callback.
    bool provisional = true;
};

using TranslationProgressCallback = std::function<void(const TranslationProgress &)>;

/**
 * Cancellation token for a model load. It is created before the load starts, so a Stop that
 * lands while a multi-GB GGUF is still being read can abort it: the llama.cpp model load
 * progress callback returns false once `cancelled` is set.
 *
 * Ownership: the registry below hands out shared references. A loader keeps its own reference
 * for the whole load, so releasing the id from Java never frees a token the loader still reads.
 * cancel/release are safe in any order and from any thread; release is idempotent.
 */
struct ModelLoadControl {
    std::atomic<bool> cancelled{false};
    void setPhase(std::string value) {
        std::lock_guard<std::mutex> guard(phaseMutex);
        phase = std::move(value);
    }
    std::string phaseSnapshot() const {
        std::lock_guard<std::mutex> guard(phaseMutex);
        return phase + (cancelled.load() ? " (cancellation requested)" : "");
    }
private:
    mutable std::mutex phaseMutex;
    std::string phase = "Waiting for native model load";
};

int64_t create_model_load_control();
void cancel_model_load_control(int64_t id);
void release_model_load_control(int64_t id);
std::shared_ptr<ModelLoadControl> model_load_control(int64_t id);

struct TranslationStats {
    int64_t prefill_ms = 0;
    int64_t decode_ms = 0;
    int64_t first_token_ms = 0;
    int64_t output_tokens = 0;
    int64_t input_tokens = 0;
    int64_t cache_reused_tokens = 0;
    int64_t prefill_decode_us = 0;
    int64_t prefill_sync_us = 0;
    int64_t sampling_us = 0;
    int64_t decode_compute_us = 0;
    int64_t decode_sync_us = 0;
    int64_t first_visible_ms = 0;
    int64_t complete_ms = 0;
    // Net device free-memory drop around model/context initialization; the backend
    // API exposes device-wide free memory rather than per-process allocations.
    int64_t device_bytes_allocated = 0;
    int64_t offloaded_layers = 0;
    int64_t total_layers = 0;
    int64_t fallback_count = 0;
};

class TranslationEngine {
public:
    virtual ~TranslationEngine() = default;
    virtual std::string translate(const std::string & text) = 0;
    virtual std::string translateWithProgress(const std::string & text,
        const TranslationRequestMetadata & request, const TranslationProgressCallback & callback) {
        (void) request;
        (void) callback;
        return translate(text);
    }
    // Used by the native regression/benchmark path to compare prefix-cached results
    // against a full prefill with identical model and sampling parameters.
    virtual std::string translateUncached(const std::string & text) {
        return translateUncachedWithProgress(text, TranslationRequestMetadata{}, {});
    }
    virtual std::string translateUncachedWithProgress(const std::string & text,
        const TranslationRequestMetadata & request, const TranslationProgressCallback & callback) {
        return translateWithProgress(text, request, callback);
    }
    virtual std::string backend() const { return "CPU"; }
    virtual TranslationStats stats() const { return {}; }
    void cancel() { cancelled.store(true); }
    void resetCancellation() { cancelled.store(false); }
protected:
    std::atomic<bool> cancelled{false};
    void checkCancelled() const;
};
std::unique_ptr<TranslationEngine> load_hymt(const std::string & path, int threads,
                                              int batch, int ubatch, bool prefer_opencl,
                                              const std::string & cache_dir,
                                              int64_t load_control = 0);
std::unique_ptr<TranslationEngine> load_opus(const std::string & directory, int threads,
                                              int64_t load_control = 0);
