#include "translation.hpp"
#include "backend-selection.hpp"
#include "llama.h"
#include "ggml-backend.h"
#include "onnxruntime_cxx_api.h"
#include "sentencepiece_processor.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using json = nlohmann::json;
void TranslationEngine::checkCancelled() const {
    if (cancelled.load()) throw std::runtime_error("Translation cancelled");
}

namespace {
std::mutex loadControlRegistryMutex;
std::map<int64_t, std::shared_ptr<ModelLoadControl>> loadControlRegistry;
int64_t nextLoadControlId = 1;
}

int64_t create_model_load_control() {
    std::lock_guard<std::mutex> guard(loadControlRegistryMutex);
    const int64_t id = nextLoadControlId++;
    loadControlRegistry.emplace(id, std::make_shared<ModelLoadControl>());
    return id;
}

void cancel_model_load_control(int64_t id) {
    // Take a reference under the lock: the flag is then set on a token that cannot be freed
    // underneath this call even if the owner releases the id concurrently.
    if (auto control = model_load_control(id)) control->cancelled.store(true);
}

void release_model_load_control(int64_t id) {
    std::lock_guard<std::mutex> guard(loadControlRegistryMutex);
    loadControlRegistry.erase(id);
}

std::shared_ptr<ModelLoadControl> model_load_control(int64_t id) {
    if (id == 0) return nullptr;
    std::lock_guard<std::mutex> guard(loadControlRegistryMutex);
    const auto entry = loadControlRegistry.find(id);
    return entry == loadControlRegistry.end() ? nullptr : entry->second;
}

namespace {
using Model = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using Context = std::unique_ptr<llama_context, decltype(&llama_free)>;
using Sampler = std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>;

std::once_flag loggerInstallOnce;
ggml_log_callback previousLlamaLogger = nullptr;
void * previousLlamaLoggerData = nullptr;
thread_local std::string * modelLoadLog = nullptr;

void captureLlamaLog(ggml_log_level level, const char * text, void *) {
    if (modelLoadLog && text) modelLoadLog->append(text);
    if (previousLlamaLogger) previousLlamaLogger(level, text, previousLlamaLoggerData);
    else if (text) std::fputs(text, stderr);
}

void installLlamaLogCapture() {
    std::call_once(loggerInstallOnce, [] {
        llama_log_get(&previousLlamaLogger, &previousLlamaLoggerData);
        llama_log_set(captureLlamaLog, nullptr);
    });
}

std::pair<int64_t, int64_t> parseOffloadedLayers(const std::string & log) {
    constexpr const char * marker = "offloaded ";
    const auto start = log.rfind(marker);
    if (start == std::string::npos) return {0, 0};
    const auto valueStart = start + std::char_traits<char>::length(marker);
    const auto slash = log.find('/', valueStart);
    if (slash == std::string::npos) return {0, 0};
    const auto end = log.find(" layers to GPU", slash);
    if (end == std::string::npos) return {0, 0};
    try {
        return {std::stoll(log.substr(valueStart, slash - valueStart)),
                std::stoll(log.substr(slash + 1, end - slash - 1))};
    } catch (...) { return {0, 0}; }
}

size_t longestCommonPrefix(const std::vector<llama_token> & left,
                           const std::vector<llama_token> & right) {
    const size_t limit = std::min(left.size(), right.size());
    size_t count = 0;
    while (count < limit && left[count] == right[count]) ++count;
    return count;
}

size_t completeUtf8Prefix(const std::string & value) {
    size_t offset = 0;
    while (offset < value.size()) {
        const auto lead = static_cast<unsigned char>(value[offset]);
        size_t width = 0;
        if (lead <= 0x7f) width = 1;
        else if (lead >= 0xc2 && lead <= 0xdf) width = 2;
        else if (lead >= 0xe0 && lead <= 0xef) width = 3;
        else if (lead >= 0xf0 && lead <= 0xf4) width = 4;
        else return offset;
        if (offset + width > value.size()) return offset;
        for (size_t i = 1; i < width; ++i) {
            const auto continuation = static_cast<unsigned char>(value[offset + i]);
            if ((continuation & 0xc0) != 0x80) return offset;
            if (i == 1 && ((lead == 0xe0 && continuation < 0xa0) ||
                           (lead == 0xed && continuation >= 0xa0) ||
                           (lead == 0xf0 && continuation < 0x90) ||
                           (lead == 0xf4 && continuation >= 0x90))) return offset;
        }
        offset += width;
    }
    return offset;
}

size_t lastCompleteWordBoundary(const std::string & value, size_t completeBytes) {
    size_t boundary = 0;
    for (size_t i = 0; i < completeBytes; ++i) {
        const unsigned char c = static_cast<unsigned char>(value[i]);
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') boundary = i + 1;
        else if (c == '.' || c == '!' || c == '?' || c == ',' || c == ';' || c == ':') boundary = i + 1;
    }
    return boundary;
}

std::string trimProgress(std::string value) {
    while (!value.empty() && (value.back() == ' ' || value.back() == '\t' ||
           value.back() == '\r' || value.back() == '\n')) value.pop_back();
    return value;
}

class HyMt final : public TranslationEngine {
    Model model{nullptr, llama_model_free};
    Context context{nullptr, llama_free};
    // Held for the engine's whole lifetime so the OpenCL-to-CPU reload is cancellable too.
    std::shared_ptr<ModelLoadControl> loadControl;
    std::string path;
    std::string activeBackend = "CPU";
    int threads, batch, ubatch;
    bool opencl = false;
    std::mutex contextOwner;
    TranslationStats lastStats;
    std::vector<llama_token> cachedPromptTokens;
    std::string cachedConfiguration;
    int64_t cachedSession = 0;
    llama_pos cachedPositionMax = -1;
    bool cacheValid = false;
    int64_t deviceBytesAllocated = 0;
    int64_t offloadedLayers = 0;
    int64_t totalLayers = 0;
    int64_t fallbackCount = 0;

    static ggml_backend_dev_t adreno830() {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto device = ggml_backend_dev_get(i);
            auto reg = ggml_backend_dev_backend_reg(device);
            if (!reg || !is_opencl_backend_name(ggml_backend_reg_name(reg))) continue;
            std::string description = ggml_backend_dev_name(device);
            description += " ";
            description += ggml_backend_dev_description(device);
            std::transform(description.begin(), description.end(), description.begin(),
                [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (description.find("adreno") == std::string::npos || description.find("830") == std::string::npos) continue;
            return device;
        }
        return nullptr;
    }

    void throwIfLoadCancelled() const {
        if (cancelled.load() || (loadControl && loadControl->cancelled.load()))
            throw std::runtime_error("Hy-MT2 model load cancelled");
    }

    void loadPhase(const std::string & phase) const {
        if (loadControl) loadControl->setPhase(phase);
    }

    void initialize(ggml_backend_dev_t device) {
        if (batch <= 0 || ubatch <= 0 || ubatch > batch) throw std::runtime_error("Invalid translation batch/ubatch configuration");
        // A Stop that lands before the first weight is read must not start the load at all.
        throwIfLoadCancelled();
        auto params = llama_model_default_params();
        params.progress_callback = [](float progress, void * data) {
            const auto * engine = static_cast<HyMt *>(data);
            const auto percent = static_cast<int>(std::clamp(progress, 0.0f, 1.0f) * 100.0f);
            engine->loadPhase("Reading Hy-MT2 weights: " + std::to_string(percent) + "%");
            return !engine->cancelled.load() &&
                (!engine->loadControl || !engine->loadControl->cancelled.load());
        };
        params.progress_callback_user_data = this;
        params.n_gpu_layers = device ? -1 : 0;
        params.split_mode = LLAMA_SPLIT_MODE_NONE;
        std::array<ggml_backend_dev_t, 2> devices{device, nullptr};
        params.devices = device ? devices.data() : nullptr;
        size_t deviceFreeBefore = 0, deviceTotalBefore = 0;
        if (device) ggml_backend_dev_memory(device, &deviceFreeBefore, &deviceTotalBefore);
        std::string loadLog;
        auto * previousCapture = modelLoadLog;
        modelLoadLog = &loadLog;
        loadPhase(device ? "Loading Hy-MT2 weights for OpenCL" : "Loading Hy-MT2 weights for CPU");
        try { model.reset(llama_model_load_from_file(path.c_str(), params)); }
        catch (...) { modelLoadLog = previousCapture; throw; }
        modelLoadLog = previousCapture;
        // A cancelled progress callback aborts the load and yields a null model.
        throwIfLoadCancelled();
        if (!model) throw std::runtime_error("Hy-MT2 model load failed");
        auto options = llama_context_default_params();
        options.n_ctx = 2048; options.n_batch = batch; options.n_ubatch = ubatch;
        options.n_threads = threads; options.n_threads_batch = threads;
        options.abort_callback = [](void * value) {
            const auto * engine = static_cast<HyMt *>(value);
            return engine->cancelled.load() ||
                (engine->loadControl && engine->loadControl->cancelled.load());
        };
        options.abort_callback_data = this;
        loadPhase(device ? "Initializing OpenCL translation context and kernels" : "Initializing CPU translation context");
        context.reset(llama_init_from_model(model.get(), options));
        throwIfLoadCancelled();
        if (!context) throw std::runtime_error("Hy-MT2 context creation failed");
        opencl = device != nullptr;
        const auto placement = parseOffloadedLayers(loadLog);
        totalLayers = placement.second > 0 ? placement.second : llama_model_n_layer(model.get());
        offloadedLayers = device ? placement.first : 0;
        deviceBytesAllocated = 0;
        if (device) {
            size_t deviceFreeAfter = 0, deviceTotalAfter = 0;
            ggml_backend_dev_memory(device, &deviceFreeAfter, &deviceTotalAfter);
            if (deviceFreeBefore >= deviceFreeAfter)
                deviceBytesAllocated = static_cast<int64_t>(deviceFreeBefore - deviceFreeAfter);
        }
        activeBackend = device
            ? std::string("Adreno OpenCL · ") + ggml_backend_dev_name(device) + " · " +
                std::to_string(offloadedLayers) + "/" + std::to_string(totalLayers) + " layers · " +
                std::to_string(deviceBytesAllocated / (1024 * 1024)) + " MiB free-memory delta"
            : (preferOpenCL ? "CPU · Adreno 830 OpenCL unavailable; fallback" : "CPU");
        cacheValid = false;
        cachedPromptTokens.clear();
        cachedConfiguration.clear();
        cachedPositionMax = -1;
        loadPhase(device ? "OpenCL translation model ready" : "CPU translation model ready");
    }

    std::string run(const std::string & prompt, const TranslationRequestMetadata & request,
                    const TranslationProgressCallback & callback, bool allowPrefixCache) {
        checkCancelled();
        // A cancelled or failed (re)initialization leaves no model behind; never dereference it.
        if (!model || !context) throw std::runtime_error("Hy-MT2 engine is not loaded");
        const auto requestStart = std::chrono::steady_clock::now();
        const auto * vocab = llama_model_get_vocab(model.get());
        const auto * format = llama_model_chat_template(model.get(), nullptr);
        if (!format) throw std::runtime_error("GGUF is missing its chat template");
        llama_chat_message message{"user", prompt.c_str()};
        int length = llama_chat_apply_template(format, &message, 1, true, nullptr, 0);
        if (length <= 0) throw std::runtime_error("Unsupported Hy-MT2 chat template");
        std::vector<char> chat(length + 1);
        int written = llama_chat_apply_template(format, &message, 1, true, chat.data(), chat.size());
        if (written != length) throw std::runtime_error("Chat template length mismatch");
        int count = -llama_tokenize(vocab, chat.data(), length, nullptr, 0, true, true);
        constexpr int max_output = 384;
        if (count <= 0 || count + max_output > 2048) throw std::runtime_error("Translation input exceeds context limit");
        std::vector<llama_token> tokens(count);
        if (llama_tokenize(vocab, chat.data(), length, tokens.data(), count, true, true) != count)
            throw std::runtime_error("Hy-MT2 tokenization failed");

        std::string configuration;
        configuration.reserve(path.size() + request.cache_scope.size() + std::char_traits<char>::length(format) + 24);
        auto appendPart = [&configuration](const std::string & value) {
            configuration.append(std::to_string(value.size()));
            configuration.push_back(':');
            configuration.append(value);
        };
        appendPart(path);
        appendPart(request.cache_scope);
        appendPart(format);
        auto * memory = llama_get_memory(context.get());
        const bool cacheIdentityMatches = allowPrefixCache && cacheValid &&
            cachedSession == request.session_id && cachedConfiguration == configuration &&
            llama_memory_seq_pos_max(memory, 0) == cachedPositionMax;
        size_t reusablePrefix = cacheIdentityMatches ? longestCommonPrefix(cachedPromptTokens, tokens) : 0;
        // llama_sampler_sample() needs logits for the final prompt token. Keep every
        // identical token before it and re-evaluate that one token to rebuild logits.
        const size_t keepTokens = reusablePrefix > 0 ? reusablePrefix - 1 : 0;
        if (!cacheIdentityMatches) {
            llama_memory_clear(memory, true);
        } else if (!llama_memory_seq_rm(memory, 0, static_cast<llama_pos>(keepTokens), -1)) {
            llama_memory_clear(memory, true);
            reusablePrefix = 0;
        }
        cachedPromptTokens.clear();
        cachedConfiguration.clear();
        cacheValid = false;
        cachedPositionMax = -1;
        lastStats.input_tokens = count;
        lastStats.cache_reused_tokens = static_cast<int64_t>(reusablePrefix > 0 ? reusablePrefix - 1 : 0);
        lastStats.device_bytes_allocated = deviceBytesAllocated;
        lastStats.offloaded_layers = offloadedLayers;
        lastStats.total_layers = totalLayers;
        lastStats.fallback_count = fallbackCount;

        auto prefillStart = std::chrono::steady_clock::now();
        for (int offset = static_cast<int>(lastStats.cache_reused_tokens); offset < count; offset += batch) {
            checkCancelled();
            auto input = llama_batch_get_one(tokens.data() + offset, std::min(batch, count - offset));
            const auto decodeCallStart = std::chrono::steady_clock::now();
            if (llama_decode(context.get(), input) != 0) throw std::runtime_error("Hy-MT2 prompt decode failed or cancelled");
            lastStats.prefill_decode_us += std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - decodeCallStart).count();
        }
        lastStats.prefill_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - prefillStart).count();
        auto prefillSyncStart = std::chrono::steady_clock::now();
        llama_synchronize(context.get());
        lastStats.prefill_sync_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - prefillSyncStart).count();
        Sampler sampler(llama_sampler_chain_init(llama_sampler_chain_default_params()), llama_sampler_free);
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_penalties(llama_vocab_n_tokens(vocab), 64, 1.05f, 0, 0));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_k(20));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_p(0.6f, 1));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_temp(0.7f));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_dist(42));
        std::string result;
        auto decodeStart = std::chrono::steady_clock::now();
        auto lastProgressAt = requestStart;
        size_t lastPublishedBytes = 0;
        auto publishProgress = [&]() {
            if (!callback) return;
            const auto validBytes = completeUtf8Prefix(result);
            const auto boundary = lastCompleteWordBoundary(result, validBytes);
            if (boundary <= lastPublishedBytes) return;
            const auto now = std::chrono::steady_clock::now();
            if (now - lastProgressAt < std::chrono::milliseconds(80)) return;
            auto visible = trimProgress(result.substr(0, boundary));
            if (visible.size() <= lastPublishedBytes) return;
            TranslationProgress progress{request.session_id, request.segment_id, request.revision,
                std::chrono::duration_cast<std::chrono::milliseconds>(now - requestStart).count(),
                std::move(visible), true};
            callback(progress);
            lastPublishedBytes = progress.text.size();
            lastProgressAt = now;
            if (lastStats.first_visible_ms == 0) lastStats.first_visible_ms = progress.elapsed_ms;
        };
        for (int i = 0; i < max_output; ++i) {
            checkCancelled();
            if (i > 0) {
                const auto syncStart = std::chrono::steady_clock::now();
                llama_synchronize(context.get());
                lastStats.decode_sync_us += std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - syncStart).count();
            }
            const auto sampleStart = std::chrono::steady_clock::now();
            auto token = llama_sampler_sample(sampler.get(), context.get(), -1);
            lastStats.sampling_us += std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - sampleStart).count();
            if (llama_vocab_is_eog(vocab, token)) {
                lastStats.decode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - decodeStart).count();
                lastStats.complete_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - requestStart).count();
                // If no complete word was shown during decoding, the return value is
                // the first useful visible result at EOS.
                if (callback && lastStats.first_visible_ms == 0)
                    lastStats.first_visible_ms = lastStats.complete_ms;
                if (allowPrefixCache) {
                    cachedPromptTokens = std::move(tokens);
                    cachedConfiguration = std::move(configuration);
                    cachedSession = request.session_id;
                    cachedPositionMax = llama_memory_seq_pos_max(memory, 0);
                    cacheValid = cachedPositionMax >= 0;
                }
                return result;
            }
            char buffer[256];
            int size = llama_token_to_piece(vocab, token, buffer, sizeof(buffer), 0, false);
            if (size < 0) {
                std::vector<char> larger(-size);
                size = llama_token_to_piece(vocab, token, larger.data(), larger.size(), 0, false);
                if (size < 0) throw std::runtime_error("Invalid translation token");
                result.append(larger.data(), size);
            } else result.append(buffer, size);
            if (lastStats.output_tokens == 0) {
                lastStats.first_token_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - requestStart).count();
            }
            ++lastStats.output_tokens;
            publishProgress();
            auto batch = llama_batch_get_one(&token, 1);
            const auto decodeCallStart = std::chrono::steady_clock::now();
            if (llama_decode(context.get(), batch) != 0) throw std::runtime_error("Hy-MT2 decode failed or cancelled");
            lastStats.decode_compute_us += std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - decodeCallStart).count();
        }
        throw std::runtime_error("Hy-MT2 output exceeded token limit; not committing truncated translation");
    }

    void fallbackToCpu() {
        context.reset();
        model.reset();
        initialize(nullptr);
        ++fallbackCount;
        activeBackend = "CPU · Adreno OpenCL inference failed; fallback";
    }
public:
    HyMt(const std::string & modelPath, int cpuThreads, int contextBatch, int contextUbatch,
         bool preferOpenCL, const std::string & cacheDir, int64_t loadControlId)
        : loadControl(model_load_control(loadControlId)), path(modelPath), threads(cpuThreads),
          batch(contextBatch), ubatch(contextUbatch), preferOpenCL(preferOpenCL) {
#if defined(TRANSLATION_OPENCL)
        if (!cacheDir.empty()) setenv("GGML_OPENCL_KERNEL_CACHE_DIR", cacheDir.c_str(), 1);
        // Set these before llama's one-time backend initialization, including CPU A/B runs.
        setenv("OCL_ICD_FILENAMES", "libOpenCL.so", 1);
#else
        (void) cacheDir;
#endif
        static std::once_flag once;
        installLlamaLogCapture();
        throwIfLoadCancelled();
        // Backend registration probes the vendor OpenCL driver even for a CPU model. Vendor
        // calls cannot be forcibly interrupted; the snapshot identifies that stage honestly.
        loadPhase("Initializing llama backends and OpenCL driver");
        std::call_once(once, [] { llama_backend_init(); });
        throwIfLoadCancelled();
        std::string openClFailure;
        if (preferOpenCL) {
            loadPhase("Selecting Adreno OpenCL device");
            auto device = adreno830();
            throwIfLoadCancelled();
            if (device) {
                // initialize() already creates the real backend. A separate init/free probe
                // compiles a disposable kernel context and repeats work before model loading.
                try { initialize(device); return; }
                catch (...) {
                    context.reset(); model.reset();
                    // A cancelled load must not be retried on CPU: Stop asked for no model.
                    throwIfLoadCancelled();
                    ++fallbackCount;
                    openClFailure = "CPU · Adreno OpenCL initialization failed; fallback";
                }
            } else {
                ++fallbackCount;
                openClFailure = "CPU · Adreno 830 OpenCL unavailable; fallback";
            }
        }
        initialize(nullptr);
        if (!openClFailure.empty()) activeBackend = std::move(openClFailure);
    }
    std::string translate(const std::string & prompt) override {
        return translateWithProgress(prompt, TranslationRequestMetadata{}, {});
    }
    std::string translateWithProgress(const std::string & prompt, const TranslationRequestMetadata & request,
                                      const TranslationProgressCallback & callback) override {
        std::lock_guard<std::mutex> owner(contextOwner);
        lastStats = {};
        try {
            auto result = run(prompt, request, callback, true);
            lastStats.fallback_count = fallbackCount;
            return result;
        }
        catch (...) {
            cacheValid = false;
            cachedPromptTokens.clear();
            cachedConfiguration.clear();
            cachedPositionMax = -1;
            if (!opencl || cancelled.load()) {
                if (context) llama_memory_clear(llama_get_memory(context.get()), true);
                throw;
            }
            fallbackToCpu();
            lastStats = {};
            try {
                auto result = run(prompt, request, callback, true);
                lastStats.fallback_count = fallbackCount;
                return result;
            }
            catch (...) {
                cacheValid = false;
                if (context) llama_memory_clear(llama_get_memory(context.get()), true);
                throw;
            }
        }
    }
    std::string translateUncached(const std::string & prompt) override {
        return translateUncachedWithProgress(prompt, TranslationRequestMetadata{}, {});
    }
    std::string translateUncachedWithProgress(const std::string & prompt,
        const TranslationRequestMetadata & request, const TranslationProgressCallback & callback) override {
        std::lock_guard<std::mutex> owner(contextOwner);
        lastStats = {};
        try {
            auto result = run(prompt, request, callback, false);
            lastStats.fallback_count = fallbackCount;
            return result;
        }
        catch (...) {
            cacheValid = false;
            cachedPromptTokens.clear();
            cachedConfiguration.clear();
            cachedPositionMax = -1;
            if (!opencl || cancelled.load()) {
                if (context) llama_memory_clear(llama_get_memory(context.get()), true);
                throw;
            }
            fallbackToCpu();
            lastStats = {};
            try {
                auto result = run(prompt, request, callback, false);
                lastStats.fallback_count = fallbackCount;
                return result;
            } catch (...) {
                cacheValid = false;
                if (context) llama_memory_clear(llama_get_memory(context.get()), true);
                throw;
            }
        }
    }
    std::string backend() const override { return activeBackend; }
    TranslationStats stats() const override { return lastStats; }
private:
    bool preferOpenCL;
};

json read_json(const std::string & path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Missing model metadata: " + path);
    return json::parse(in);
}
Ort::Env & environment() { static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "live-captions"); return env; }
class Opus final : public TranslationEngine {
    Ort::Session encoder{nullptr}, decoder{nullptr};
    Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    sentencepiece::SentencePieceProcessor source, target;
    json vocab;
    std::vector<std::string> pieces, in_names, out_names;
    int layers, heads, head_dim, start_id, eos_id, pad_id, unk_id;
public:
    // ONNX Runtime session creation has no progress hook, so the load is cancellable only
    // between the graph loads. They are small enough that this keeps Stop bounded.
    Opus(const std::string & dir, int threads, int64_t loadControlId) {
        const auto loadControl = model_load_control(loadControlId);
        const auto throwIfLoadCancelled = [&loadControl] {
            if (loadControl && loadControl->cancelled.load())
                throw std::runtime_error("OPUS model load cancelled");
        };
        throwIfLoadCancelled();
        auto config = read_json(dir + "/config.json");
        vocab = read_json(dir + "/vocab.json");
        layers = config.at("decoder_layers"); heads = config.at("decoder_attention_heads");
        head_dim = config.at("d_model").get<int>() / heads;
        start_id = config.at("decoder_start_token_id"); eos_id = config.at("eos_token_id"); pad_id = config.at("pad_token_id"); unk_id = vocab.at("<unk>");
        pieces.resize(config.at("vocab_size").get<size_t>());
        for (const auto & entry : vocab.items()) {
            size_t id = entry.value().get<size_t>();
            if (id >= pieces.size()) throw std::runtime_error("OPUS vocabulary ID out of range");
            pieces[id] = entry.key();
        }
        if (!source.Load(dir + "/source.spm").ok() || !target.Load(dir + "/target.spm").ok())
            throw std::runtime_error("SentencePiece model load failed");
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(threads); options.SetInterOpNumThreads(1);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        throwIfLoadCancelled();
        encoder = Ort::Session(environment(), (dir + "/encoder.onnx").c_str(), options);
        throwIfLoadCancelled();
        decoder = Ort::Session(environment(), (dir + "/decoder.onnx").c_str(), options);
        in_names = {"encoder_attention_mask", "input_ids", "encoder_hidden_states"};
        out_names = {"logits"};
        for (int i = 0; i < layers; ++i) for (const auto * attention : {"decoder", "encoder"}) for (const auto * kind : {"key", "value"}) {
            auto suffix = std::to_string(i) + "." + attention + "." + kind;
            in_names.push_back("past_key_values." + suffix); out_names.push_back("present." + suffix);
        }
        in_names.push_back("use_cache_branch");
        if (decoder.GetInputCount() != in_names.size() || decoder.GetOutputCount() != out_names.size())
            throw std::runtime_error("Unsupported OPUS decoder signature");
        Ort::AllocatorWithDefaultOptions allocator;
        for (size_t i = 0; i < in_names.size(); ++i)
            if (in_names[i] != decoder.GetInputNameAllocated(i, allocator).get()) throw std::runtime_error("Unexpected decoder input name");
        for (size_t i = 0; i < out_names.size(); ++i)
            if (out_names[i] != decoder.GetOutputNameAllocated(i, allocator).get()) throw std::runtime_error("Unexpected decoder output name");
    }
    std::string translate(const std::string & text) override {
        checkCancelled();
        std::vector<std::string> input_pieces;
        if (!source.Encode(text, &input_pieces).ok()) throw std::runtime_error("SentencePiece encoding failed");
        if (input_pieces.size() > 383) throw std::runtime_error("OPUS input too long");
        std::vector<int64_t> ids;
        for (const auto & piece : input_pieces) ids.push_back(vocab.value(piece, unk_id));
        ids.push_back(eos_id);
        std::vector<int64_t> mask(ids.size(), 1);
        const int64_t shape[] = {1, static_cast<int64_t>(ids.size())};
        std::vector<Ort::Value> input;
        input.push_back(Ort::Value::CreateTensor<int64_t>(memory, ids.data(), ids.size(), shape, 2));
        input.push_back(Ort::Value::CreateTensor<int64_t>(memory, mask.data(), mask.size(), shape, 2));
        const char * enc_in[] = {"input_ids", "attention_mask"}; const char * enc_out[] = {"last_hidden_state"};
        Ort::RunOptions run;
        auto hidden = encoder.Run(run, enc_in, input.data(), input.size(), enc_out, 1);
        std::vector<Ort::Value> feed;
        int64_t token = start_id; const int64_t one[] = {1, 1}; const int64_t cache_shape[] = {1, heads, 0, head_dim};
        bool cached = false; const int64_t bool_shape[] = {1};
        feed.push_back(Ort::Value::CreateTensor<int64_t>(memory, mask.data(), mask.size(), shape, 2));
        feed.push_back(Ort::Value::CreateTensor<int64_t>(memory, &token, 1, one, 2));
        feed.push_back(std::move(hidden[0]));
        Ort::AllocatorWithDefaultOptions allocator;
        for (int i = 0; i < layers * 4; ++i) feed.push_back(Ort::Value::CreateTensor<float>(allocator, cache_shape, 4));
        feed.push_back(Ort::Value::CreateTensor<bool>(memory, &cached, 1, bool_shape, 1));
        std::vector<const char *> inputs, outputs;
        for (auto & s : in_names) inputs.push_back(s.c_str());
        for (auto & s : out_names) outputs.push_back(s.c_str());
        std::vector<std::string> translated;
        for (int step = 0; step < 256; ++step) {
            checkCancelled();
            auto result = decoder.Run(run, inputs.data(), feed.data(), feed.size(), outputs.data(), outputs.size());
            auto * logits = result[0].GetTensorMutableData<float>();
            if (result[0].GetTensorTypeAndShapeInfo().GetElementCount() != pieces.size())
                throw std::runtime_error("Unexpected OPUS logits shape");
            logits[pad_id] = -std::numeric_limits<float>::infinity();
            token = std::max_element(logits, logits + pieces.size()) - logits;
            if (token == eos_id) {
                std::string decoded;
                if (!target.Decode(translated, &decoded).ok()) throw std::runtime_error("SentencePiece decoding failed");
                return decoded;
            }
            translated.push_back(pieces.at(token));
            for (int i = 0; i < layers * 4; ++i) {
                // The cached branch returns empty cross-attention tensors: preserve first-step encoder KV.
                if (step > 0 && i % 4 >= 2) continue;
                feed[3 + i] = std::move(result[1 + i]);
            }
            cached = true;
        }
        throw std::runtime_error("OPUS output exceeded token limit");
    }
};
}
std::unique_ptr<TranslationEngine> load_hymt(const std::string & path, int threads,
                                              int batch, int ubatch, bool preferOpenCL,
                                              const std::string & cacheDir,
                                              int64_t loadControl) {
    return std::make_unique<HyMt>(path, threads, batch, ubatch, preferOpenCL, cacheDir, loadControl);
}
std::unique_ptr<TranslationEngine> load_opus(const std::string & path, int threads, int64_t loadControl) {
    return std::make_unique<Opus>(path, threads, loadControl);
}
