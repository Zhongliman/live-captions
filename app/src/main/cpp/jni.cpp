#include "translation.hpp"
#include <jni.h>
#include <exception>
#include <cstdlib>

static std::string utf8(JNIEnv * env, jbyteArray bytes) {
    std::string result(env->GetArrayLength(bytes), '\0');
    env->GetByteArrayRegion(bytes, 0, result.size(), reinterpret_cast<jbyte *>(result.data()));
    return result;
}
static void fail(JNIEnv * env, const std::exception & e) {
    if (!env->ExceptionCheck()) env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), e.what());
}
extern "C" JNIEXPORT jlong JNICALL Java_com_asr_live_i18n_NativeTranslator_load(JNIEnv * env, jobject, jbyteArray path, jint threads, jint batch, jint ubatch, jboolean opus, jboolean preferOpenCL, jbyteArray cacheDir, jlong loadControl) {
    try { return reinterpret_cast<jlong>((opus ? load_opus(utf8(env, path), threads, loadControl) :
        load_hymt(utf8(env, path), threads, batch, ubatch, preferOpenCL,
                  cacheDir ? utf8(env, cacheDir) : "", loadControl)).release()); }
    catch (const std::exception & e) { fail(env, e); return 0; }
}

// The load-cancellation token is created before any model exists and is owned by Java. The
// registry keeps it alive for the duration of a load, so cancel/release cannot race into a
// use-after-free and release is idempotent.
extern "C" JNIEXPORT jlong JNICALL Java_com_asr_live_i18n_NativeModelLoad_createControl(JNIEnv * env, jobject) {
    try { return static_cast<jlong>(create_model_load_control()); }
    catch (const std::exception & e) { fail(env, e); return 0; }
}
extern "C" JNIEXPORT void JNICALL Java_com_asr_live_i18n_NativeModelLoad_cancelControl(JNIEnv *, jobject, jlong id) {
    cancel_model_load_control(static_cast<int64_t>(id));
}
extern "C" JNIEXPORT void JNICALL Java_com_asr_live_i18n_NativeModelLoad_releaseControl(JNIEnv *, jobject, jlong id) {
    release_model_load_control(static_cast<int64_t>(id));
}
extern "C" JNIEXPORT jstring JNICALL Java_com_asr_live_i18n_NativeModelLoad_phaseControl(JNIEnv * env, jobject, jlong id) {
    const auto control = model_load_control(static_cast<int64_t>(id));
    if (!control) return nullptr;
    return env->NewStringUTF(control->phaseSnapshot().c_str());
}
extern "C" JNIEXPORT jbyteArray JNICALL Java_com_asr_live_i18n_NativeTranslator_run(
        JNIEnv * env, jobject, jlong handle, jbyteArray text, jlong sessionId, jlong segmentId,
        jint revision, jbyteArray cacheScope, jobject callback) {
    try {
        TranslationRequestMetadata request{sessionId, segmentId, revision,
            cacheScope ? utf8(env, cacheScope) : std::string{}};
        TranslationProgressCallback progress;
        jclass callbackClass = nullptr;
        jmethodID progressMethod = nullptr;
        if (callback) {
            callbackClass = env->GetObjectClass(callback);
            progressMethod = env->GetMethodID(callbackClass, "onProgress", "(JJIJ[B)V");
            if (!progressMethod) throw std::runtime_error("Native progress callback method is unavailable");
            progress = [env, callback, progressMethod](const TranslationProgress & event) {
                jbyteArray bytes = env->NewByteArray(static_cast<jsize>(event.text.size()));
                if (!bytes) return;
                env->SetByteArrayRegion(bytes, 0, static_cast<jsize>(event.text.size()),
                    reinterpret_cast<const jbyte *>(event.text.data()));
                if (!env->ExceptionCheck()) env->CallVoidMethod(callback, progressMethod,
                    static_cast<jlong>(event.session_id), static_cast<jlong>(event.segment_id),
                    static_cast<jint>(event.revision), static_cast<jlong>(event.elapsed_ms), bytes);
                env->DeleteLocalRef(bytes);
                if (env->ExceptionCheck()) throw std::runtime_error("Native progress callback failed");
            };
        }
        auto result = reinterpret_cast<TranslationEngine *>(handle)->translateWithProgress(
            utf8(env, text), request, progress);
        auto bytes = env->NewByteArray(result.size());
        if (bytes) env->SetByteArrayRegion(bytes, 0, result.size(), reinterpret_cast<const jbyte *>(result.data()));
        if (callbackClass) env->DeleteLocalRef(callbackClass);
        return bytes;
    } catch (const std::exception & e) { fail(env, e); return nullptr; }
}
extern "C" JNIEXPORT void JNICALL Java_com_asr_live_i18n_NativeTranslator_abort(JNIEnv *, jobject, jlong handle) { reinterpret_cast<TranslationEngine *>(handle)->cancel(); }
extern "C" JNIEXPORT void JNICALL Java_com_asr_live_i18n_NativeTranslator_resetCancellation(JNIEnv *, jobject, jlong handle) { reinterpret_cast<TranslationEngine *>(handle)->resetCancellation(); }
extern "C" JNIEXPORT void JNICALL Java_com_asr_live_i18n_NativeTranslator_free(JNIEnv *, jobject, jlong handle) { delete reinterpret_cast<TranslationEngine *>(handle); }
extern "C" JNIEXPORT jstring JNICALL Java_com_asr_live_i18n_NativeTranslator_backend(JNIEnv * env, jobject, jlong handle) {
    return env->NewStringUTF(reinterpret_cast<TranslationEngine *>(handle)->backend().c_str());
}
extern "C" JNIEXPORT jlongArray JNICALL Java_com_asr_live_i18n_NativeTranslator_stats(JNIEnv * env, jobject, jlong handle) {
    const auto stats = reinterpret_cast<TranslationEngine *>(handle)->stats();
    const jlong values[] = {stats.prefill_ms, stats.decode_ms, stats.first_token_ms, stats.output_tokens,
        stats.input_tokens, stats.cache_reused_tokens, stats.prefill_decode_us, stats.prefill_sync_us,
        stats.sampling_us, stats.decode_compute_us, stats.decode_sync_us, stats.first_visible_ms, stats.complete_ms,
        stats.device_bytes_allocated, stats.offloaded_layers, stats.total_layers, stats.fallback_count};
    jlongArray result = env->NewLongArray(17);
    if (result) env->SetLongArrayRegion(result, 0, 17, values);
    return result;
}

extern "C" JNIEXPORT void JNICALL Java_com_asr_live_asr_QnnService_configureDspPath(JNIEnv * env, jobject, jbyteArray path) {
    setenv("ADSP_LIBRARY_PATH", utf8(env, path).c_str(), 1);
}
