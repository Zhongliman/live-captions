package com.asr.live.service

import android.util.Log
import com.asr.live.pipeline.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow

enum class ListeningState { STOPPED, STARTING, LISTENING, STOPPING }

/**
 * Translator readiness, independent of whether model files are installed on disk. Model
 * installation ("Offline ready") only means the files exist; this is whether the in-process
 * runtime can currently translate with them.
 */
enum class TranslatorState { NOT_LOADED, LOADING, READY, FAILED }
data class Performance(
    val asrMs: Long = 0, val asrRtf: Double = 0.0, val translationMs: Long = 0,
    val asrComputeMs: Long = 0, val endpointWaitMs: Long? = null,
    val decodeCalls: Long = 0, val decodeMeanMs: Double = 0.0,
    val decodeP50Ms: Long = 0, val decodeP95Ms: Long = 0, val decodeMaxMs: Long = 0,
    val translationPrefillMs: Long = 0, val translationDecodeMs: Long = 0,
    val translationFirstTokenMs: Long = 0, val translationTokensPerSecond: Double = 0.0,
    val translationInputTokens: Long = 0, val translationOutputTokens: Long = 0,
    val translationCacheTokens: Long = 0, val translationFirstVisibleMs: Long = 0,
    val translationCompleteMs: Long = 0, val translationPrefillDecodeUs: Long = 0,
    val translationPrefillSyncUs: Long = 0, val translationSamplingUs: Long = 0,
    val translationDecodeComputeUs: Long = 0,
    val translationDecodeSyncUs: Long = 0, val translationDeviceBytes: Long = 0,
    val translationOffloadedLayers: Long = 0, val translationTotalLayers: Long = 0,
    val translationFallbackCount: Long = 0,
    val correctionWaitMs: Long = 0,
    val hyWaitMs: Long = 0, val hyComputeMs: Long = 0, val hyDisplayMs: Long = 0,
    val opusBackend: String = "CPU",
    val opusWaitMs: Long = 0, val opusComputeMs: Long = 0,
    val opusPrefillMs: Long = 0, val opusGenerationMs: Long = 0,
    val opusFirstTokenMs: Long = 0, val opusTokensPerSecond: Double = 0.0,
    val resultsComputed: Long = 0, val resultsDisplayed: Long = 0, val resultsRejected: Long = 0,
    val rejectionReasons: Map<String, Int> = emptyMap(), val displayLatencyMs: Long = 0,
    val firstUsefulCaptionMs: Long? = null,
    val audioToProvisionalMs: Long? = null, val stableToProvisionalMs: Long? = null,
    val audioToFinalMs: Long? = null, val finalLatencyMs: Long? = null,
    val audioDepth: Int = 0, val provisionalDepth: Int = 0, val finalDepth: Int = 0,
    val audioSampleRateHz: Int = 0, val audioSamplesCaptured: Long = 0,
    val audioRms: Double = 0.0, val audioPeak: Double = 0.0, val audioClippedSamples: Long = 0,
    val audioTimestampGaps: Long = 0, val audioFirstMonotonicNs: Long? = null,
    val audioLastMonotonicNs: Long? = null,
    val captionBacklogMs: Long = 0,
    val backlogMs: Long = 0, val droppedAudioMs: Long = 0, val skippedTranslations: Int = 0,
    val correctionMs: Long = 0, val correctionRtf: Double = 0.0, val skippedCorrections: Int = 0,
    val appPssKb: Long = 0, val rssKb: Long = 0, val nativeHeapKb: Long = 0,
    val qnnPssKb: Long = 0, val qnnProcessPresent: Boolean = false,
    val thermalMaxC: Double? = null, val readableThermalSensors: Int = 0,
    val javaHeapKb: Long = 0, val availableKb: Long = 0, val modelFilesDiskKb: Long = 0,
    val performanceMode: String = "Balanced",
    val asr: String = "", val translator: String = "ML Kit", val backend: String = "CPU",
    val correctionThreads: Int = 4,
    val translationBackend: String = "CPU", val thermalStatus: String = "Unknown",
    val translationThreads: Int = 4,
    val adpfActive: Boolean = false,
    val profile: String = "Dutch → English", val chunk: String = "560 ms", val threads: Int = 6,
    val translatorState: TranslatorState = TranslatorState.NOT_LOADED,
    val hyModelLoadMs: Long = 0, val hyWarmupMs: Long = 0, val residentTranslatorReused: Boolean = false,
    val translatorPhase: String = "Not loaded", val translatorStartupElapsedMs: Long = 0,
    val startupFinalBuffered: Int = 0, val startupFinalDropped: Int = 0, val startupFinalOldestMs: Long = 0,
    val opusStartupFallbackCount: Int = 0,
    val timeFromListenToAsrReadyMs: Long? = null, val timeFromListenToTranslatorReadyMs: Long? = null,
    val timeFromListenToFirstTranslatedCaptionMs: Long? = null,
    val qualityOverridesPreset: Boolean = false,
)
object CaptionState {
    private val ledger = SegmentLedger()
    private val comparisonLedger = TranslationComparisonLedger()
    private val _lines = MutableStateFlow<List<Caption>>(emptyList())
    val lines = _lines.asStateFlow()
    private val _lifecycle = MutableStateFlow(ListeningState.STOPPED)
    val lifecycle = _lifecycle.asStateFlow()
    private val _running = MutableStateFlow(false)
    val running = _running.asStateFlow()
    private val _error = MutableStateFlow<String?>(null)
    val error = _error.asStateFlow()
    private val _status = MutableStateFlow<String?>(null)
    val status = _status.asStateFlow()
    private val _metrics = MutableStateFlow(Performance())
    val metrics = _metrics.asStateFlow()
    private val _comparisons = MutableStateFlow<List<TranslationComparison>>(emptyList())
    val comparisons = _comparisons.asStateFlow()
    @Volatile private var generation = -1L
    private val displayedResults = linkedSetOf<TranslationResultIdentity>()
    private val computedAtNs = mutableMapOf<TranslationResultIdentity, Long>()
    private val sessionStartedAtNs = mutableMapOf<Long, Long>()
    @Synchronized fun sessionStartNs(id: Long): Long? = sessionStartedAtNs[id]
    @Synchronized fun begin(id: Long, config: SessionConfig, modelName: String) {
        generation = id; ledger.start(id); comparisonLedger.clear(); displayedResults.clear(); computedAtNs.clear()
        sessionStartedAtNs[id] = System.nanoTime()
        sessionStartedAtNs.keys.removeAll { it != id }
        publish(); publishComparisons()
        val translationLabel = when {
            config.opusBenchmarkEnabled -> "OPUS provisional A/B + ${config.quality.label}"
            config.opusProvisionalEnabled -> "OPUS previews + ${config.quality.label} finals"
            else -> config.quality.label
        }
        // A manual quality override makes the preset label alone misleading (e.g. "Balanced ·
        // Hy 1.8B Q8" while the active translator is actually Q4): show the preset name only
        // and let `translator` (already accurate) carry the real quality.
        val presetLabel = if (config.qualityOverridesPreset) "${config.performanceMode.presetName} (custom)"
            else config.performanceMode.label
        _metrics.value = Performance(asr = modelName, translator = translationLabel, profile = config.profile.label, threads = config.threads,
            translationThreads = config.translationThreads,
            performanceMode = presetLabel, correctionThreads = config.correctionThreads,
            backend = if (config.qnn) "QNN initializing · experimental" else "CPU",
            translationBackend = if (config.gpuTranslation && com.asr.live.BuildConfig.OPENCL_ENABLED)
                "Adreno OpenCL requested · initializing" else "CPU",
            chunk = com.asr.live.model.ModelCatalog.byId(config.modelId)?.chunkMs?.let { "$it ms" } ?: "VAD phrases",
            translatorState = TranslatorState.NOT_LOADED, qualityOverridesPreset = config.qualityOverridesPreset)
        _error.value = null; setLifecycle(ListeningState.STARTING)
    }
    @Synchronized fun source(id: Long, text: String, endpoint: Boolean, nowMs: Long): Caption? =
        ledger.source(id, text, endpoint, nowMs)?.also { publish() }
    @Synchronized fun translated(key: SegmentKey, text: String, rank: Int, final: Boolean): Boolean =
        ledger.translate(key, text, rank, final).also { if (it) publish() }
    @Synchronized fun canTranslatePortion(key: SegmentKey, portion: TranslationPortionIdentity, final: Boolean = false): Boolean =
        ledger.canTranslatePortion(key, portion, final)
    @Synchronized fun translatedPortion(key: SegmentKey, portion: TranslationPortionIdentity, text: String,
                                        rank: Int, final: Boolean, requestId: Long): Boolean =
        ledger.translatePortion(key, portion, text, rank, final, requestId).also { if (it) publish() }
    @Synchronized fun comparison(generation: Long, key: SegmentKey, source: String,
                                engine: ComparisonEngine, text: String, elapsedMs: Long) {
        if (generation != this.generation || key.session != generation) return
        comparisonLedger.record(key, source, engine, text, elapsedMs)
        publishComparisons()
    }
    @Synchronized fun vote(key: SegmentKey, choice: TranslationVote): Boolean =
        comparisonLedger.vote(key, choice).also { if (it) publishComparisons() }
    @Synchronized fun revisedTranslated(key: SegmentKey, source: String, translation: String, requestId: Long = 0): Caption? =
        ledger.reviseTranslated(key, source, translation, requestId)?.also { publish() }
    @Synchronized fun skip(key: SegmentKey, reason: String) { ledger.skip(key, reason); publish() }
    fun current(key: SegmentKey) = ledger.current(key)
    fun hasFinalTranslation(key: SegmentKey): Boolean = ledger.caption(key)?.let {
        it.translation.isNotBlank() && it.qualityRank >= 2 &&
            it.stage in setOf(CaptionStage.FINAL, CaptionStage.REVISED)
    } == true
    @Synchronized fun discontinuity(id: Long) { ledger.discontinuity(id); publish() }
    @Synchronized fun cancel(id: Long) {
        if (id != generation) return
        ledger.cancel(); publish(); setLifecycle(ListeningState.STOPPING)
    }
    /** Mark a normal stop while keeping the current segment open for recognizer flush. */
    @Synchronized fun stopping(id: Long) { if (id == generation) setLifecycle(ListeningState.STOPPING) }
    @Synchronized fun stopped(id: Long) {
        if (id != generation) return
        setLifecycle(ListeningState.STOPPED); _status.value = null
    }
    @Synchronized fun listening(id: Long) {
        if (id == generation && _lifecycle.value == ListeningState.STARTING) setLifecycle(ListeningState.LISTENING)
    }
    @Synchronized fun metrics(id: Long, update: (Performance) -> Performance) {
        if (id == generation) _metrics.value = update(_metrics.value)
    }
    @Synchronized fun resultComputed(id: Long, key: SegmentKey, atNs: Long, requestId: Long = 0,
                                     portion: TranslationPortionIdentity? = null) {
        if (id != generation || key.session != id) return
        val capturedPortion = portion ?: ledger.caption(key)?.let { row ->
            TranslationPortionIdentity(id, key.id, row.stableSource.ifBlank { row.source })
        } ?: return
        if (capturedPortion.session != id || capturedPortion.segmentId != key.id) return
        val result = TranslationResultIdentity(capturedPortion, requestId)
        computedAtNs[result] = atNs
        while (computedAtNs.size > 300) {
            val oldest = computedAtNs.keys.first()
            computedAtNs.remove(oldest)
        }
        _metrics.value = _metrics.value.copy(resultsComputed = _metrics.value.resultsComputed + 1)
    }
    @Synchronized fun resultRejected(id: Long, reason: String, computedResult: Boolean = true) {
        if (id != generation) return
        val reasons = _metrics.value.rejectionReasons.toMutableMap()
        reasons[reason] = (reasons[reason] ?: 0) + 1
        while (reasons.size > 8) {
            val oldest = reasons.minByOrNull { it.value }?.key ?: break
            reasons.remove(oldest)
        }
        _metrics.value = _metrics.value.copy(resultsRejected = _metrics.value.resultsRejected + if (computedResult) 1 else 0,
            rejectionReasons = reasons)
    }
    /** Called by the Compose screen after a translated row has entered a rendered frame. */
    @Synchronized fun acknowledgeDisplayed(id: Long, key: SegmentKey, atNs: Long): Boolean {
        if (id != generation || key.session != id) return false
        val row = ledger.caption(key) ?: return false
        if (row.translation.isBlank()) return false
        val portion = row.translationPortion ?: return false
        val result = TranslationResultIdentity(portion, row.translationRequestId)
        if (!displayedResults.add(result)) return false
        while (displayedResults.size > 5_000) displayedResults.remove(displayedResults.first())
        val latency = computedAtNs.remove(result)?.let { ((atNs - it) / 1_000_000L).coerceAtLeast(0) } ?: 0L
        val requestId = result.requestId
        val firstUsefulMs = _metrics.value.firstUsefulCaptionMs ?: sessionStartedAtNs[id]?.let {
            ((atNs - it) / 1_000_000L).coerceAtLeast(0)
        }
        runCatching { Log.d("CaptionPipeline", "session=$id segment=${key.id} revision=${key.revision} request=$requestId atNs=$atNs event=displayed") }
        _metrics.value = _metrics.value.copy(resultsDisplayed = _metrics.value.resultsDisplayed + 1,
            displayLatencyMs = latency, firstUsefulCaptionMs = firstUsefulMs)
        return true
    }
    @Synchronized fun error(id: Long, text: String) { if (id == generation) _error.value = text }
    fun setError(text: String?) { _error.value = text }
    fun setStatus(text: String?) { _status.value = text }
    @Synchronized fun clear() { ledger.clear(); comparisonLedger.clear(); publish(); publishComparisons() }
    private fun publish() { _lines.value = ledger.snapshot() }
    private fun publishComparisons() { _comparisons.value = comparisonLedger.snapshot() }
    private fun setLifecycle(value: ListeningState) { _lifecycle.value = value; _running.value = value != ListeningState.STOPPED }
}
