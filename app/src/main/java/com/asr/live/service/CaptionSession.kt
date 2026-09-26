package com.asr.live.service

import android.content.Context
import android.os.SystemClock
import android.util.Log
import com.asr.live.asr.EngineFactory
import com.asr.live.audio.AudioCapture
import com.asr.live.audio.AudioSignalTelemetry
import com.asr.live.audio.pcm16Sha256
import com.asr.live.i18n.MlKitTranslator
import com.asr.live.i18n.LocalTranslator
import com.asr.live.i18n.ModelLoadScope
import com.asr.live.i18n.NativeModelLoad
import com.asr.live.i18n.NativeTranslator
import com.asr.live.i18n.ResidentTranslatorManager
import com.asr.live.i18n.TranslatorIdentity
import com.asr.live.model.*
import com.asr.live.pipeline.*
import com.asr.live.pipeline.engine.EngineFamily
import com.asr.live.pipeline.engine.EngineRegistry
import java.util.concurrent.CancellationException
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ExecutionException
import java.util.concurrent.FutureTask
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeoutException
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference

internal data class AudioChunk(val samples: FloatArray, val capturedAtNs: Long, val sequence: Long, val firstSample: Long)
internal data class CorrectionRequest(
    val request: TranslationRequest, val samples: FloatArray, val enqueuedAtNs: Long,
    val sourceStartSample: Long = -1, val sourceEndSample: Long = -1,
)
internal data class TranslationRequest(
    val key: SegmentKey,
    val text: String,
    val portionIdentity: TranslationPortionIdentity = TranslationPortionIdentity(key.session, key.id, text),
    val requestId: Long = 0,
    val createdAtNs: Long = 0,
    val queuedAtNs: Long = 0,
    val endpointAt: Long? = null,
    val sourceAudioAtNs: Long? = null,
    val stableSourceAtNs: Long? = null,
    val benchmarkOnly: Boolean = false,
    val isFinal: Boolean = false,
    val correctionSource: String? = null,
)

/** Each native object has a single owning worker. Cancellation never releases an in-use object. */
class CaptionSession(
    private val ctx: Context,
    val generation: Long,
    private val config: SessionConfig,
    private val onFailure: (String) -> Unit,
) {
    private val correctionEnabled = config.correction && config.profile.correctionSupported && ModelCatalog.byId(config.modelId)?.kind == EngineKind.NEMOTRON
    private val lifecycle = SessionLifecycle(correctionEnabled)
    private val failureReported = AtomicBoolean(false)
    private val audioQueue = BoundedMailbox<AudioChunk>(16)
    private val provisional = BoundedMailbox<TranslationRequest>(1)
    private val qualityProvisional = BoundedMailbox<TranslationRequest>(1)
    private val finals = BoundedMailbox<TranslationRequest>(2)
    private val correctionQueue = BoundedMailbox<CorrectionRequest>(1)
    private val correctionDecodeThreads = ConcurrentHashMap.newKeySet<Thread>()
    private val correctionBusy = AtomicBoolean(false)
    @Volatile private var correctionReady = false
    @Volatile private var correctionRtf = 0.0
    private val corrector = Thread(::correctLoop, "endpoint-correction")
    private val sequence = AtomicLong()
    private val capturedSamples = AtomicLong()
    private val audioSignalTelemetry = AudioSignalTelemetry()
    private val requestSequence = AtomicLong()
    private val activeHyRequest = AtomicReference<TranslationRequest?>()
    private val hyScheduleLock = Any()
    private val sessionStartedAtNs = System.nanoTime()
    private val captureFinished = AtomicBoolean(false)
    private val inferenceBusy = AtomicBoolean(false)
    @Volatile private var activeFinalEndpoint: Long? = null
    @Volatile private var segmentAudioStartedAtNs: Long? = null
    private var segmentPcmStartSample: Long? = null
    @Volatile private var lastPartialAtNs: Long? = null
    @Volatile private var opusTranslator: LocalTranslator? = null
    @Volatile private var finalTranslator: LocalTranslator? = null
    // Load tokens exist before their translators do, so Stop can abort a model load that has
    // not yet produced a cancellable handle.
    @Volatile private var hyLoadScope: ModelLoadScope? = null
    @Volatile private var opusLoadScope: ModelLoadScope? = null
    @Volatile private var hyReady = false
    private val primaryEngine = config.primaryEngineId?.let { id -> EngineRegistry.all.firstOrNull { it.engineId == id } }
    private val primaryUsesOpus = primaryEngine?.family == EngineFamily.OPUS
    private val opusBenchmarkEnabled = config.opusBenchmarkEnabled
    // OPUS loads in a fraction of Hy-MT2's time, so while Hy is still loading it can stand in
    // for finals instead of leaving them stuck on the caption line untranslated.
    private val opusFallbackEnabled = config.opusStartupFallbackEnabled
    private val opusWorkerEnabled = opusBenchmarkEnabled || opusFallbackEnabled
    // Recognized speech that arrived before Hy-MT2 finished loading. The steady-state `finals`
    // mailbox is only 2 deep, sized for normal backpressure, not for a multi-second model load;
    // routing loading-time finals through it would silently and permanently drop them (a SKIPPED
    // caption is never revisited). This buffer bridges that window and is drained once Hy is ready.
    private val startupFinals = StartupFinalBuffer<TranslationRequest>()
    private val opusFallbackQueue = BoundedMailbox<TranslationRequest>(4)
    @Volatile private var opusFallbackCount = 0
    private val firstTranslationRecorded = AtomicBoolean(false)
    private val startupWorkers = buildMap {
        put(WORKER_RECOGNITION, true)
        put(WORKER_HY_TRANSLATION, true)
        if (correctionEnabled) put(WORKER_CORRECTION, false)
        if (opusWorkerEnabled) put(WORKER_OPUS, false)
    }
    private val startupBarrier = WorkerStartupBarrier(startupWorkers)
    private val pcm = PcmBuffer()
    private var lastProvisionalAt = 0L
    private var lastProvisionalText = ""
    private val info = ModelCatalog.byId(config.modelId) ?: ModelCatalog.DEFAULT
    private val capture = AudioCapture(ctx,
        onChunk = { samples ->
            if (!lifecycle.isCancelled()) {
                val capturedAtNs = System.nanoTime()
                val firstSample = capturedSamples.getAndAdd(samples.size.toLong())
                val signal = audioSignalTelemetry.accept(samples, capturedAtNs)
                CaptionState.metrics(generation) { it.copy(audioSampleRateHz = signal.sampleRateHz,
                    audioSamplesCaptured = signal.samples, audioRms = signal.rms, audioPeak = signal.peak,
                    audioClippedSamples = signal.clippedSamples, audioTimestampGaps = signal.timestampGaps,
                    audioFirstMonotonicNs = signal.firstMonotonicNs, audioLastMonotonicNs = signal.lastMonotonicNs) }
                val dropped = audioQueue.offer(AudioChunk(samples, capturedAtNs, sequence.incrementAndGet(), firstSample))
                CaptionState.metrics(generation) { it.copy(audioDepth = audioQueue.size(),
                    droppedAudioMs = it.droppedAudioMs + (dropped?.samples?.size ?: 0) / 16) }
            }
        },
        onStarted = { if (!lifecycle.isCancelled()) { trace("microphone-started"); CaptionState.listening(generation) } },
        onFinished = {
            // Always record completion. A capture that ends for any other reason must not leave
            // a later stop() waiting for audio that can never arrive.
            captureFinished.set(true)
            if (lifecycle.isStopping()) audioQueue.closeForDrain()
            trace("microphone-finished", details = "queued=${audioQueue.size()} stopping=${lifecycle.isStopping()}")
        },
        onError = { fail("Microphone: ${it.message}") },
        onFormat = { sampleRate, channels ->
            audioSignalTelemetry.setSampleRate(sampleRate)
            CaptionState.metrics(generation) { it.copy(audioSampleRateHz = sampleRate) }
            trace("microphone-format", details = "sampleRateHz=$sampleRate channels=$channels encoding=PCM16")
        })
    private val asr = Thread(::recognize, "recognition")
    private val opus = Thread(::opusLoop, "opus-ab-translation")
    private val final = Thread(::translateLoop, "hy-translation")
    fun start() {
        trace("session-started")
        if (opusWorkerEnabled) opus.start()
        final.start(); corrector.start(); asr.start()
    }
    /**
     * Gracefully stop input, drain queued PCM, flush ASR and finish caption translations.
     *
     * Everything that only feeds the live caption line — provisional translations, the OPUS A/B
     * branch, an in-flight model load — is abandoned immediately, because no one will read it.
     * Captured PCM and queued finals are preserved and still drain through ASR and Hy.
     */
    fun stop() {
        if (!lifecycle.requestStop()) return
        trace("stop-requested")
        capture.stopCapturing()
        trace("graceful-shutdown-started", details = "graceMs=${ShutdownDeadline.GRACE_MS} " +
            "cancelMs=${ShutdownDeadline.CANCEL_MS} hardMs=${ShutdownDeadline.HARD_MS}")
        discardProvisionalWork("stop requested")
        // Correction is optional quality work. Stop publishes the unrevised final instead of
        // waiting for a second recognizer pass, so no captured speech is lost either way.
        correctionQueue.close().forEach { pending ->
            pending.samples.fill(0f)
            trace("correction-cancelled", pending.request.key, pending.request.requestId, "reason=stop requested")
            enqueueFinal(pending.request)
        }
        // A model still loading after Stop can only cost time and memory.
        runCatching { hyLoadScope?.cancel() }
        runCatching { opusLoadScope?.cancel() }
    }
    /**
     * First shutdown escalation. Drops every remaining translation — including a queued final —
     * but leaves recognition alone, so the ASR flush still publishes trailing speech as source
     * text instead of being cut short by a full cancellation.
     */
    fun abandonTranslations() {
        if (!lifecycle.abandonTranslations()) return
        trace("translation-work-abandoned", details = "reason=graceful window elapsed")
        discardProvisionalWork("graceful window elapsed")
        finals.close().forEach { rejectRequest(it, "final translation dropped at the stop deadline") }
        startupFinals.drainAll().forEach { rejectRequest(it, "final translation dropped at the stop deadline") }
        opusFallbackQueue.close()
        runCatching { finalTranslator?.cancel() }
        runCatching { opusTranslator?.cancel() }
        runCatching { hyLoadScope?.cancel() }; runCatching { opusLoadScope?.cancel() }
    }
    fun cancel() {
        if (!lifecycle.cancel()) return
        trace("cancel-requested")
        interruptWorkers()
    }
    private fun discardProvisionalWork(reason: String) {
        runCatching { opusTranslator?.cancel() }
        synchronized(hyScheduleLock) {
            val active = activeHyRequest.get()?.takeIf { !it.isFinal }
            discardProvisionalTranslations(
                provisionalQueues = listOf(provisional, qualityProvisional),
                activeProvisionalRequestId = active?.requestId,
                cancelRequest = { requestId ->
                    finalTranslator?.cancel(requestId)
                    trace("translation-cancelled", active?.key, requestId, "reason=$reason")
                },
                onDiscarded = { rejectRequest(it, "provisional translation discarded: $reason") },
            )
        }
    }
    private fun interruptWorkers() {
        runCatching { capture.cancel() }
        runCatching { hyLoadScope?.cancel() }; runCatching { opusLoadScope?.cancel() }
        runCatching { opusTranslator?.cancel() }; runCatching { finalTranslator?.cancel() }
        audioQueue.close().forEach { it.samples.fill(0f) }
        provisional.close(); qualityProvisional.close(); finals.close()
        startupFinals.drainAll(); opusFallbackQueue.close()
        correctionQueue.close().forEach { it.samples.fill(0f) }
        listOf(asr, opus, final, corrector).forEach { runCatching { it.interrupt() } }
    }
    /** Non-blocking probe used by the service to decide whether shutdown must escalate. */
    fun isTerminated(): Boolean = join(0)
    /** A short bounded wait. Service callers retry this from an IO coroutine. */
    fun join(timeoutMs: Long = 250): Boolean {
        val startedNs = System.nanoTime()
        val workersStopped = joinThreadsWithin(
            listOf(asr, capture.threadForJoin(), if (opusWorkerEnabled) opus else null, final, corrector), timeoutMs)
        if (!workersStopped) return false
        val remainingMs = TimeUnit.NANOSECONDS.toMillis(
            TimeUnit.MILLISECONDS.toNanos(timeoutMs) - (System.nanoTime() - startedNs)).coerceAtLeast(0)
        return joinThreadsWithin(correctionDecodeThreads.toList(), remainingMs)
    }
    private fun fail(message: String) {
        if (failureReported.compareAndSet(false, true) && lifecycle.cancel()) {
            try { onFailure(message) } finally { interruptWorkers() }
        }
    }
    private fun recognize() {
        var engine: com.asr.live.asr.AsrEngine? = null
        var performanceHint: WorkerPerformanceHint? = null
        var startupReported = false
        // AudioCapture sends one 100 ms (1600 samples at 16 kHz) unit per
        // worker iteration. Nemotron's model chunk profile is not the ADPF
        // workload duration: it can contain several microphone callbacks.
        val continuity = AudioContinuity()
        val decodeStats = com.asr.live.asr.DecodeStatsAccumulator()
        var qnnFailed = false
        var audioMs = 0L
        var computeMs = 0L
        try {
            performanceHint = WorkerPerformanceHint(ctx, AudioCapture.CHUNK_DURATION_MS)
            CaptionState.metrics(generation) { it.copy(adpfActive = performanceHint?.enabled == true) }
            require(info.supports(config.profile.source)) { "${info.shortName} does not support ${config.profile.source}" }
            require(!config.qnn || info.kind != EngineKind.NEMOTRON || info.id == ModelCatalog.NEMOTRON.id) { "QNN currently supports only the separate 560 ms context" }
            ModelStore.verify(ctx, info)
            if (lifecycle.isCancelled()) return
            fun create(): com.asr.live.asr.AsrEngine = if (config.qnn && !qnnFailed && info.kind == EngineKind.NEMOTRON)
                com.asr.live.asr.QnnEngine(ctx, config.profile.source, config.threads, ::partial, ::endpoint,
                    { backend -> if (backend.startsWith("CPU")) qnnFailed = true; CaptionState.metrics(generation) { it.copy(backend = backend) } },
                    { pcm.invalidate(); CaptionState.discontinuity(generation) }, generation,
                    { profile -> if (profile.rpcCalls == 0L || profile.rpcCalls % 64L == 0L) trace("qnn-profile", details =
                        "initMs=${profile.initializeTotalNanos / 1_000_000} libraryMs=${profile.libraryLoadNanos / 1_000_000} " +
                            "dspCopyMs=${profile.dspCopyNanos / 1_000_000} dspSetupMs=${profile.dspSetupNanos / 1_000_000} " +
                        "prepareMs=${profile.recognizerPrepareNanos / 1_000_000} rpcP50Ms=${profile.rpcP50Nanos / 1_000_000} " +
                        "rpcP95Ms=${profile.rpcP95Nanos / 1_000_000} serviceP95Ms=${profile.serviceP95Nanos / 1_000_000} " +
                        "clientQueueP50Ms=${profile.clientQueueP50Nanos / 1_000_000} " +
                        "clientQueueP95Ms=${profile.clientQueueP95Nanos / 1_000_000} " +
                            "binderP95Ms=${profile.binderAndMarshallingP95Nanos / 1_000_000} " +
                            "feedMs=${profile.audioFeedNanos / 1_000_000} decodeCombinedMs=${profile.combinedDecodeNanos / 1_000_000} " +
                            "resultMs=${profile.resultNanos / 1_000_000} endpointMs=${profile.endpointCheckNanos / 1_000_000} " +
                            "graphBreakdown=${profile.graphBreakdown}") },
                    { lifecycle.isStopping() })
            else EngineFactory.create(ctx, info, config.profile.source, "transcribe", ::partial, ::endpoint, config.threads)
            trace("listen-asr-init-started", details = "model=${info.shortName} qnn=${config.qnn}")
            engine = startupBarrier.initialize(WORKER_RECOGNITION) { create() }
            startupReported = true
            val asrReadyMs = (nowNs() - sessionStartedAtNs) / 1_000_000L
            trace("asr-initialized", details = "initMs=$asrReadyMs")
            CaptionState.metrics(generation) { it.copy(timeFromListenToAsrReadyMs = asrReadyMs) }
            if (lifecycle.isCancelled()) return
            // Capture starts as soon as recognition is ready. Translation and correction models
            // keep loading concurrently so a multi-GB GGUF never makes Listen look dead.
            capture.start()
            while (!lifecycle.isCancelled()) {
                // A required worker that fails after capture started still terminates the session.
                // During a stop that failure is expected teardown, and aborting here would throw
                // away the ASR flush the graceful path exists to preserve.
                if (!lifecycle.isStopping()) startupBarrier.requiredFailure(exceptWorker = WORKER_RECOGNITION)?.let {
                    fail("${it.worker} failed to start: ${it.cause.message}")
                    return
                }
                val chunk = audioQueue.poll()
                if (chunk == null) {
                    if (lifecycle.shouldFinishAsr(audioQueue.size() > 0,
                            captureFinished.get() || capture.isFinished(), inferenceBusy.get())) {
                        trace("asr-finish-started")
                        val finishStartedNs = nowNs()
                        checkNotNull(engine).finish()
                        val finishMs = (nowNs() - finishStartedNs) / 1_000_000L
                        computeMs += finishMs
                        val combinedDecodeStats = decodeStats.snapshot(checkNotNull(engine).decodeStats)
                        CaptionState.metrics(generation) { it.copy(asrMs = finishMs,
                            asrComputeMs = computeMs, asrRtf = computeMs.toDouble() / maxOf(1, audioMs),
                            decodeCalls = combinedDecodeStats.calls, decodeMeanMs = combinedDecodeStats.meanNanos / 1_000_000.0,
                            decodeP50Ms = combinedDecodeStats.p50Nanos / 1_000_000L,
                            decodeP95Ms = combinedDecodeStats.p95Nanos / 1_000_000L,
                            decodeMaxMs = combinedDecodeStats.maxNanos / 1_000_000L) }
                        trace("asr-finish-complete", details = "finalizeMs=$finishMs decodeCalls=${combinedDecodeStats.calls}")
                        break
                    }
                    Thread.sleep(10); continue
                }
                if (continuity.gap(chunk.sequence)) {
                    // Audio loss is a discontinuity: do not join speech from either side into one hypothesis.
                    decodeStats.retire(checkNotNull(engine).decodeStats)
                    checkNotNull(engine).release(); engine = null; engine = create()
                    pcm.take(); lastProvisionalText = ""
                    segmentAudioStartedAtNs = null
                    segmentPcmStartSample = null
                    CaptionState.discontinuity(generation)
                    val discarded = chunk.samples.size + audioQueue.drain().sumOf { it.samples.size }
                    CaptionState.metrics(generation) { it.copy(droppedAudioMs = it.droppedAudioMs + discarded / 16) }
                    continuity.reset()
                    continue
                }
                if (segmentAudioStartedAtNs == null) {
                    segmentAudioStartedAtNs = chunk.capturedAtNs
                    segmentPcmStartSample = chunk.firstSample
                }
                pcm.append(chunk.samples)
                val startNs = nowNs()
                inferenceBusy.set(true)
                val activeEngine = checkNotNull(engine)
                val duration = try {
                    activeEngine.accept(chunk.samples)
                    (nowNs() - startNs) / 1_000_000L
                } finally { inferenceBusy.set(false) }
                performanceHint?.report(duration)
                computeMs += duration; audioMs += chunk.samples.size / 16
                val combinedDecodeStats = decodeStats.snapshot(activeEngine.decodeStats)
                CaptionState.metrics(generation) { it.copy(asrMs = duration,
                    asrComputeMs = computeMs, asrRtf = computeMs.toDouble() / maxOf(1, audioMs),
                    decodeCalls = combinedDecodeStats.calls, decodeMeanMs = combinedDecodeStats.meanNanos / 1_000_000.0,
                    decodeP50Ms = combinedDecodeStats.p50Nanos / 1_000_000L,
                    decodeP95Ms = combinedDecodeStats.p95Nanos / 1_000_000L,
                    decodeMaxMs = combinedDecodeStats.maxNanos / 1_000_000L, audioDepth = audioQueue.size(),
                    backlogMs = ((nowNs() - chunk.capturedAtNs) / 1_000_000L).coerceAtLeast(0), captionBacklogMs = captionAge()) }
            }
        } catch (e: InterruptedException) {
            if (!startupReported) { failStartupIfUnreported(WORKER_RECOGNITION, e); startupReported = true }
        } catch (t: Throwable) {
            if (!startupReported) { failStartupIfUnreported(WORKER_RECOGNITION, t); startupReported = true }
            // A teardown that was asked for is not an error worth showing the user.
            if (!lifecycle.isStopping()) fail("Recognition: ${t.message}")
        }
        finally {
            if (!startupReported) failStartupIfUnreported(WORKER_RECOGNITION,
                CancellationException("Recognition stopped before becoming ready"))
            runCatching { performanceHint?.close() }
            runCatching { engine?.release() }
            capture.cancel(); pcm.take()
            lifecycle.markAsrFinished()
            trace("asr-worker-stopped")
        }
    }
    private fun partial(text: String) {
        if (lifecycle.isCancelled() || text.isBlank()) return
        val time = now()
        val atNs = nowNs()
        val row = CaptionState.source(generation, text, false, time) ?: return
        lastPartialAtNs = atNs
        trace("asr-partial", row.key, details = "atNs=$atNs")
        cancelHyIfIncompatible(row)
        // After Stop nothing will read another provisional revision of the live line.
        if (!lifecycle.shouldTranslateProvisional()) return
        if (!ProvisionalTranslationPolicy.shouldTranslate(lastProvisionalText, row.stableSource, time, lastProvisionalAt)) return
        lastProvisionalAt = time; lastProvisionalText = row.stableSource
        val request = TranslationRequest(row.key, row.stableSource,
            requestId = requestSequence.incrementAndGet(), createdAtNs = atNs, queuedAtNs = atNs,
            sourceAudioAtNs = segmentAudioStartedAtNs ?: atNs, stableSourceAtNs = atNs)
        trace("translation-requested", row.key, request.requestId,
            "branch=provisional createdAtNs=${request.createdAtNs} queuedAtNs=${request.queuedAtNs}")
        provisional.offer(request)?.let { rejectRequest(it, "provisional request replaced by newer text") }
        if (opusBenchmarkEnabled) qualityProvisional.offer(request.copy(benchmarkOnly = true))
        depths()
    }
    private fun endpoint(text: String) {
        val audio = pcm.take()
        val sourceStartSample = segmentPcmStartSample
        val sourceEndSample = audio?.let { sourceStartSample?.plus(it.size) }
        segmentPcmStartSample = null
        if (lifecycle.isCancelled()) { audio?.fill(0f); return }
        if (text.isBlank()) { audio?.fill(0f); segmentAudioStartedAtNs = null; lastPartialAtNs = null; return }
        val at = now()
        val atNs = nowNs()
        val row = CaptionState.source(generation, text, true, at) ?: return
        val endpointWaitMs = lastPartialAtNs?.let { ((atNs - it) / 1_000_000L).coerceAtLeast(0) }
        lastPartialAtNs = null
        CaptionState.metrics(generation) { it.copy(endpointWaitMs = endpointWaitMs) }
        val sampleRange = if (sourceStartSample != null && sourceEndSample != null)
            "[$sourceStartSample,$sourceEndSample)" else "unavailable"
        val segmentHash = audio?.let(::pcm16Sha256) ?: "unavailable"
        trace("asr-endpoint", row.key, details = "atNs=$atNs endpointWaitMs=${endpointWaitMs ?: -1} " +
            "sampleRateHz=${AudioCapture.SAMPLE_RATE} sourceSamples=$sampleRange pcm16Sha256=$segmentHash")
        lastProvisionalText = ""
        // Endpoints use Hy once; OPUS is only a live-prefix A/B translator.
        val request = TranslationRequest(row.key, row.source,
            requestId = requestSequence.incrementAndGet(), createdAtNs = atNs, queuedAtNs = atNs,
            endpointAt = at, sourceAudioAtNs = segmentAudioStartedAtNs ?: atNs,
            stableSourceAtNs = atNs, isFinal = true)
        segmentAudioStartedAtNs = null
        val admitted = correctionEnabled && correctionReady && audio != null &&
            CorrectionPolicy.admit(config.profile, audio.size, audioQueue.size(), finals.size(),
                correctionBusy.get() || activeFinalEndpoint != null, correctionRtf) &&
            correctionBusy.compareAndSet(false, true)
        if (admitted) {
            trace("correction-queued", row.key, request.requestId,
                "queuedAtNs=$atNs sampleRateHz=${AudioCapture.SAMPLE_RATE} sourceSamples=$sampleRange pcm16Sha256=$segmentHash")
            val dropped = correctionQueue.offer(CorrectionRequest(request, audio!!, atNs,
                sourceStartSample ?: -1L, sourceEndSample ?: -1L))
            if (dropped != null) {
                dropped.samples.fill(0f)
                correctionBusy.set(false)
                enqueueFinal(request)
            }
        } else {
            audio?.fill(0f)
            enqueueFinal(request)
            if (correctionEnabled) CaptionState.metrics(generation) { it.copy(skippedCorrections = it.skippedCorrections + 1) }
        }
        depths()
    }
    private fun correctLoop() {
        if (!correctionEnabled) return
        var engine: com.asr.live.asr.EndpointCorrector? = null
        var performanceHint: WorkerPerformanceHint? = null
        var startupReported = false
        var engineRetiredByDecode = false
        try {
            performanceHint = WorkerPerformanceHint(ctx, 1500)
            if (lifecycle.isCancelled()) return
            engine = startupBarrier.initialize(WORKER_CORRECTION) {
                ModelStore.verify(ctx, ModelCatalog.PARAKEET)
                com.asr.live.asr.EndpointCorrector(ModelStore.dir(ctx, ModelCatalog.PARAKEET.id), config.correctionThreads).also { it.warmUp() }
            }
            correctionReady = true
            startupReported = true
            while (lifecycle.shouldRunCorrection(correctionQueue.size() > 0, correctionBusy.get())) {
                val job = correctionQueue.poll()
                if (job == null) {
                    Thread.sleep(20); continue
                }
                var samplesOwnedByDecode = false
                try {
                    if (!CaptionState.current(job.request.key)) continue
                    val correctionWaitMs = ((nowNs() - job.enqueuedAtNs) / 1_000_000L).coerceAtLeast(0)
                    CaptionState.metrics(generation) { it.copy(correctionWaitMs = correctionWaitMs) }
                    trace("correction-started", job.request.key, job.request.requestId,
                        "queueWaitMs=$correctionWaitMs")
                    val started = now()
                    val candidate = decodeWithinDeadline(checkNotNull(engine), job,
                        CorrectionPolicy.remainingNanos(job.enqueuedAtNs, nowNs()),
                        onWorkerStarted = { samplesOwnedByDecode = true },
                        onEngineRetired = { engineRetiredByDecode = true })
                    val elapsed = now() - started
                    performanceHint?.report(elapsed)
                    correctionRtf = elapsed.toDouble() / maxOf(1, job.samples.size / 16)
                    CaptionState.metrics(generation) { it.copy(correctionMs = elapsed, correctionRtf = correctionRtf) }
                    trace("correction-complete", job.request.key, job.request.requestId, "computeMs=$elapsed")
                    if (!lifecycle.isCancelled() && finals.size() == 0 && activeFinalEndpoint == null && CorrectionPolicy.accept(job.request.text, candidate,
                            now() - (job.request.endpointAt ?: started), audioQueue.size())) {
                        // Keep the existing caption pair visible until Hy has translated the
                        // second hypothesis. The accepted source and translation are published
                        // together as one new revision.
                        enqueueFinal(job.request.copy(text = candidate, correctionSource = candidate,
                            portionIdentity = TranslationPortionIdentity(generation, job.request.key.id, candidate),
                            requestId = requestSequence.incrementAndGet(), queuedAtNs = nowNs())); depths()
                    } else {
                        CaptionState.metrics(generation) { it.copy(skippedCorrections = it.skippedCorrections + 1) }
                        if (!lifecycle.isCancelled()) enqueueFinal(job.request)
                    }
                } catch (t: Throwable) {
                    if (t is InterruptedException) throw t
                    if (t is TimeoutException) {
                        correctionReady = false
                        CaptionState.metrics(generation) { it.copy(skippedCorrections = it.skippedCorrections + 1) }
                        trace("correction-deadline-exceeded", job.request.key, job.request.requestId,
                            "deadlineMs=${CorrectionPolicy.MAX_LATENCY_MS}; fallback=unrevised")
                        if (!lifecycle.isCancelled()) enqueueFinal(job.request)
                        correctionQueue.drain().forEach { pending ->
                            pending.samples.fill(0f)
                            if (!lifecycle.isCancelled()) enqueueFinal(pending.request)
                        }
                        break
                    }
                    if (!lifecycle.isCancelled()) {
                        CaptionState.error(generation, "Optional correction failed; using Nemotron text: ${t.message}")
                        enqueueFinal(job.request)
                    }
                } finally {
                    if (!samplesOwnedByDecode) job.samples.fill(0f)
                    correctionBusy.set(false)
                }
            }
        } catch (e: InterruptedException) {
            if (!startupReported) { failStartupIfUnreported(WORKER_CORRECTION, e); startupReported = true }
        } catch (t: Throwable) {
            if (!startupReported) { failStartupIfUnreported(WORKER_CORRECTION, t); startupReported = true }
            if (!lifecycle.isStopping()) CaptionState.error(generation, "Optional correction disabled: ${t.message}")
        } finally {
            if (!startupReported) failStartupIfUnreported(WORKER_CORRECTION,
                CancellationException("Correction worker stopped before becoming ready"))
            try {
                runCatching { performanceHint?.close() }
                correctionReady = false
                if (!engineRetiredByDecode) engine?.close()
            } finally {
                lifecycle.markCorrectionFinished()
                trace("correction-worker-stopped")
            }
        }
    }

    /**
     * Enforce the Parakeet endpoint budget while keeping recognizer/audio ownership safe.
     * If native decode outlives its deadline, the caller queues the uncorrected final now;
     * the timed-out worker releases PCM and the recognizer only after decode actually exits.
     */
    private fun decodeWithinDeadline(
        engine: com.asr.live.asr.EndpointCorrector,
        job: CorrectionRequest,
        remainingNanos: Long,
        onWorkerStarted: () -> Unit,
        onEngineRetired: () -> Unit,
    ): String {
        if (remainingNanos <= 0) throw TimeoutException("Parakeet correction deadline elapsed in queue")
        val useGate = NativeResourceUseGate()
        fun closeAfterDecode() { runCatching { engine.close() } }
        val task = FutureTask<String> {
            if (!useGate.beginCall()) {
                job.samples.fill(0f)
                throw CancellationException("Parakeet decode cancelled before start")
            }
            try { engine.decode(job.samples) }
            finally {
                job.samples.fill(0f)
                if (useGate.finishCall()) closeAfterDecode()
            }
        }
        val worker = Thread({
            try { task.run() }
            finally { correctionDecodeThreads.remove(Thread.currentThread()) }
        }, "parakeet-decode-${job.request.requestId}").apply { isDaemon = true }
        correctionDecodeThreads.add(worker)
        try { worker.start() }
        catch (t: Throwable) {
            correctionDecodeThreads.remove(worker)
            job.samples.fill(0f)
            throw t
        }
        onWorkerStarted()
        try {
            return task.get(remainingNanos, TimeUnit.NANOSECONDS)
        } catch (timeout: TimeoutException) {
            onEngineRetired()
            val retirement = useGate.retire()
            task.cancel(retirement.callInUse)
            if (!retirement.callInUse) job.samples.fill(0f)
            if (retirement.closeNow) closeAfterDecode()
            throw timeout
        } catch (interrupted: InterruptedException) {
            onEngineRetired()
            val retirement = useGate.retire()
            task.cancel(retirement.callInUse)
            if (!retirement.callInUse) job.samples.fill(0f)
            if (retirement.closeNow) closeAfterDecode()
            throw interrupted
        } catch (failure: ExecutionException) {
            val cause = failure.cause ?: failure
            if (cause is Exception) throw cause
            throw RuntimeException(cause)
        }
    }

    private fun enqueueFinal(request: TranslationRequest) {
        // The recognized source is already published; only its translation is given up here.
        if (!lifecycle.shouldTranslate()) {
            rejectRequest(request, "translation work abandoned at the stop deadline")
            return
        }
        val queued = request.copy(queuedAtNs = nowNs())
        trace("translation-requested", queued.key, queued.requestId,
            "branch=final createdAtNs=${queued.createdAtNs} queuedAtNs=${queued.queuedAtNs}")
        if (!hyReady) {
            // Hy-MT2 is still loading. The steady-state finals mailbox is far too small to hold
            // this: route it through the startup buffer instead, which is bounded by age as well
            // as count, so nothing is silently and permanently lost to a full 2-deep queue.
            // A displaced fast-path attempt is not a loss: the startup buffer below still holds
            // the authoritative copy for Hy to translate once it is ready.
            if (opusFallbackEnabled) opusFallbackQueue.offer(queued)
            val (evictedByCapacity, evictedByAge) = startupFinals.offerReportingStale(queued)
            (evictedByAge + listOfNotNull(evictedByCapacity)).forEach {
                CaptionState.metrics(generation) { m -> m.copy(skippedTranslations = m.skippedTranslations + 1) }
                rejectRequest(it, "startup backlog expired")
            }
            depths()
            return
        }
        val dropped = synchronized(hyScheduleLock) {
            val lost = finals.offer(queued)
            activeHyRequest.get()?.takeIf { !it.isFinal }?.let { active ->
                // A final only preempts once it is ready to translate. Prefix extensions do
                // not cancel in-flight work, so useful provisional output is not starved.
                finalTranslator?.cancel(active.requestId)
                trace("hy-provisional-preempted", active.key, active.requestId,
                    "forFinalRequest=${queued.requestId}")
            }
            lost
        }
        dropped?.let {
            CaptionState.skip(it.key, "Translation skipped: queue full")
            CaptionState.metrics(generation) { m -> m.copy(skippedTranslations = m.skippedTranslations + 1) }
            rejectRequest(it, "final translation queue full")
        }
    }

    private fun cancelHyIfIncompatible(row: Caption) {
        synchronized(hyScheduleLock) {
            val active = activeHyRequest.get()?.takeIf { !it.isFinal } ?: return
            if (!CaptionState.canTranslatePortion(active.key, active.portionIdentity)) {
                finalTranslator?.cancel(active.requestId)
                trace("hy-provisional-invalidated", active.key, active.requestId,
                    "sourceRevision=${row.key.revision}")
            }
        }
    }
    private fun createTranslator(loadScope: ModelLoadScope?, quality: TranslationQuality = config.quality): LocalTranslator {
        if (quality == TranslationQuality.ML_KIT) return MlKitTranslator(config.profile.source, config.profile.target)
        val bundle = checkNotNull(quality.bundleId)
        val directory = TranslationModels.verify(ctx, bundle)
        val modelIds = listOfNotNull(info.id, bundle, if (opusWorkerEnabled) config.profile.fastBundle else null,
            if (correctionEnabled) ModelCatalog.PARAKEET.id else null,
            if (config.qnn) ModelCatalog.NEMOTRON_QNN.id else null).distinct()
        val estimatedFiles = modelIds.sumOf { id ->
            ModelStore.dir(ctx, id).walkTopDown().filter { it.isFile }.sumOf { it.length() }
        }
        CaptionState.metrics(generation) { it.copy(modelFilesDiskKb = estimatedFiles / 1024) }
        val maxModelBudget = 11L * 1024 * 1024 * 1024
        if (estimatedFiles > maxModelBudget)
            error("Selected resident models exceed the 11 GiB Max Quality budget")
        if (modelBytesFor(bundle) > 3_000_000_000L && !MemoryUsage.canLoad(ctx, estimatedFiles))
            error("Not enough available RAM to keep selected models resident with 2 GiB system headroom")
        val preferOpenCl = config.gpuTranslation && com.asr.live.BuildConfig.OPENCL_ENABLED
        return NativeTranslator(java.io.File(directory, "model.gguf"),
            config.translationThreads, false, config.profile, config.glossary,
            preferOpenCl, ctx.getDir("llama-opencl-cache", Context.MODE_PRIVATE),
            config.translationBatch, config.translationUbatch, loadScope)
    }
    private fun createOpusTranslator(loadScope: ModelLoadScope): LocalTranslator {
        val bundle = checkNotNull(config.profile.fastBundle)
        val directory = TranslationModels.verify(ctx, bundle)
        return NativeTranslator(directory, 2, true, config.profile, "", false,
            ctx.getDir("llama-opencl-cache", Context.MODE_PRIVATE),
            config.translationBatch, config.translationUbatch, loadScope)
    }
    private fun createPrimaryTranslator(loadScope: ModelLoadScope?): LocalTranslator = when (primaryEngine?.family) {
        EngineFamily.ML_KIT -> MlKitTranslator(config.profile.source, config.profile.target)
        EngineFamily.OPUS -> createOpusTranslator(checkNotNull(loadScope))
        EngineFamily.HY_MT2 -> createTranslator(loadScope, primaryEngine?.quality ?: config.quality)
        else -> createTranslator(loadScope)
    }
    private fun modelBytesFor(id: String) = TranslationModels.bundle(ctx, id).size
    private fun hyIdentity() = TranslatorIdentity(
        modelBundle = when (primaryEngine?.family) {
            EngineFamily.OPUS -> checkNotNull(config.profile.fastBundle)
            EngineFamily.HY_MT2 -> checkNotNull(primaryEngine.quality?.bundleId ?: config.quality.bundleId)
            else -> checkNotNull(config.quality.bundleId)
        }, sourceLanguage = config.profile.source,
        targetLanguage = config.profile.target, glossary = config.glossary,
        backendRequestsOpenCl = config.gpuTranslation && com.asr.live.BuildConfig.OPENCL_ENABLED,
        batch = config.translationBatch, ubatch = config.translationUbatch)
    private fun translateLoop() {
        var translator: LocalTranslator? = null
        var performanceHint: WorkerPerformanceHint? = null
        var startupReported = false
        var reusedResident = false
        // Created before the model load so a Stop that lands mid-GGUF has something to cancel.
        // ML Kit downloads its own models and exposes no native load handle, and is cheap enough
        // that residency is not worth the extra bookkeeping.
        val residencyEligible = primaryEngine?.family != EngineFamily.ML_KIT &&
            (primaryEngine?.family != null || config.quality != TranslationQuality.ML_KIT)
        val loadScope = if (residencyEligible) ModelLoadScope(NativeModelLoad()) else null
        hyLoadScope = loadScope
        val loadStartedNs = nowNs()
        CaptionState.metrics(generation) { it.copy(translatorState = TranslatorState.LOADING) }
        try {
            performanceHint = WorkerPerformanceHint(ctx, 750)
            trace("hy-load-started", details = "engine=${config.primaryEngineId ?: config.quality.name} gpu=${config.gpuTranslation}")
            val engine = startupBarrier.initialize(WORKER_HY_TRANSLATION) {
                val built = if (residencyEligible) ResidentTranslatorManager.acquire(
                    RESIDENT_SLOT_HY, hyIdentity(), { createPrimaryTranslator(loadScope) }) { reused -> reusedResident = reused }
                else createPrimaryTranslator(loadScope)
                translator = built
                finalTranslator = built
                val loadMs = (nowNs() - loadStartedNs) / 1_000_000L
                trace("hy-load-completed", details = "backend=${built.backend} reused=$reusedResident loadMs=$loadMs")
                CaptionState.metrics(generation) { it.copy(hyModelLoadMs = loadMs, residentTranslatorReused = reusedResident) }
                // Warm-up only prepares future work; skip it when a stopped session has none, a
                // resident model is already warm, or real translation work is already waiting
                // (translating it is itself useful work, and doing that first gets a caption on
                // screen sooner than a synthetic warm-up sentence would).
                val realWorkWaiting = !startupFinals.isEmpty() || finals.size() > 0
                if (lifecycle.shouldWarmUp() && !reusedResident && !realWorkWaiting) {
                    val warmupStartedNs = nowNs()
                    built.warmUp()
                    CaptionState.metrics(generation) { it.copy(hyWarmupMs = (nowNs() - warmupStartedNs) / 1_000_000L) }
                } else trace("hy-warmup-skipped", details = "reason=${
                    if (!lifecycle.shouldWarmUp()) "stop" else if (reusedResident) "resident" else "work-waiting"}")
                built
            }
            translator = engine
            finalTranslator = engine
            CaptionState.metrics(generation) { it.copy(translationBackend = engine.backend,
                translatorState = TranslatorState.READY) }
            startupReported = true
            hyReady = true
            val translatorReadyMs = (nowNs() - sessionStartedAtNs) / 1_000_000L
            trace("hy-ready", details = "timeFromListenMs=$translatorReadyMs")
            CaptionState.metrics(generation) { it.copy(timeFromListenToTranslatorReadyMs = translatorReadyMs) }
            while (lifecycle.shouldRunHyTranslation(!startupFinals.isEmpty() || finals.size() > 0 ||
                    provisional.size() > 0 || qualityProvisional.size() > 0)) {
                // Pick and publish the active request under the same lock used by final
                // admission. A final can therefore either win the poll or preempt the active
                // provisional; there is no race window in which it misses both.
                val request = synchronized(hyScheduleLock) {
                    // Startup finals buffered while Hy was loading drain first, oldest first, so
                    // recognized speech never waits behind newer live-caption work. Finals carry
                    // committed speech and always drain next. Provisional work only refreshes the
                    // live line, so Stop abandons it instead of prolonging shutdown.
                    (startupFinals.poll() ?: finals.poll() ?: if (!lifecycle.shouldTranslateProvisional()) null
                        else if (opusBenchmarkEnabled) qualityProvisional.poll() else provisional.poll())
                        .also { activeHyRequest.set(it) }
                }
                if (request == null) { Thread.sleep(20); continue }
                depths()
                val requestCurrent = when {
                    request.benchmarkOnly || request.correctionSource != null -> CaptionState.current(request.key)
                    else -> CaptionState.canTranslatePortion(request.key, request.portionIdentity, request.isFinal)
                }
                if (!requestCurrent) {
                    rejectRequest(request, "caption portion incompatible before Hy translation")
                    clearActiveHyRequest(request)
                    continue
                }
                if (request.endpointAt != null && now() - request.endpointAt > 15_000) {
                    if (!request.benchmarkOnly) CaptionState.skip(request.key, "Translation skipped: stale backlog")
                    rejectRequest(request, "Hy request exceeded 15 second queue limit")
                    clearActiveHyRequest(request)
                    continue
                }
                val queueWaitMs = ((nowNs() - request.queuedAtNs) / 1_000_000L).coerceAtLeast(0)
                CaptionState.metrics(generation) { it.copy(hyWaitMs = queueWaitMs) }
                trace("hy-started", request.key, request.requestId, "queueWaitMs=$queueWaitMs")
                val startedNs = nowNs()
                val isFinal = request.isFinal
                if (isFinal) activeFinalEndpoint = request.endpointAt
                try {
                    val result = engine.translate(request.text, request.requestId, request.key.session,
                        request.key.id, request.key.revision) { progress ->
                        if (!request.benchmarkOnly && request.correctionSource == null &&
                            progress.provisional && progress.sessionId == request.key.session &&
                            progress.segmentId == request.key.id && progress.revision == request.key.revision &&
                            !lifecycle.isCancelled()) {
                            val shown = CaptionState.translatedPortion(request.key, request.portionIdentity,
                                progress.text, 0, false, request.requestId)
                            if (shown) trace("hy-progress-displayed", request.key, request.requestId,
                                "elapsedMs=${progress.elapsedMs}")
                        }
                    }
                    if (lifecycle.isCancelled()) break
                    val computedAtNs = nowNs()
                    val elapsed = (computedAtNs - startedNs) / 1_000_000L
                    performanceHint?.report(elapsed)
                    val timings = engine.timings
                    val outputKey = if (request.correctionSource != null) request.key.copy(revision = request.key.revision + 1) else request.key
                    val isCaptionResult = !request.benchmarkOnly
                    if (isCaptionResult) CaptionState.resultComputed(generation, outputKey, computedAtNs,
                        request.requestId, request.portionIdentity)
                    val displayStartedNs = nowNs()
                    val accepted = if (request.benchmarkOnly) {
                        CaptionState.comparison(generation, request.key, request.text,
                            ComparisonEngine.HY_MT2, result, elapsed)
                        false
                    } else if (request.correctionSource != null) {
                        CaptionState.revisedTranslated(request.key, request.correctionSource, result, request.requestId) != null
                    } else CaptionState.translatedPortion(request.key, request.portionIdentity, result,
                        if (isFinal) 2 else 0, isFinal, request.requestId)
                    val displayMs = (nowNs() - displayStartedNs) / 1_000_000L
                    if (isCaptionResult && accepted) recordFirstTranslatedCaption()
                    if (isCaptionResult && !accepted)
                        rejectRequest(request, "caption portion incompatible before Hy result could publish", computedResult = true)
                    trace(if (accepted) "hy-displayed-to-state" else "hy-result-computed", outputKey,
                        request.requestId, "computeMs=$elapsed displayMs=$displayMs final=$isFinal accepted=$accepted " +
                            "inputTokens=${timings.inputTokens} outputTokens=${timings.outputTokens} " +
                            "firstVisibleMs=${timings.firstVisibleMs} completeMs=${timings.completeMs}")
                    CaptionState.metrics(generation) { m -> m.copy(translationMs = elapsed,
                        hyComputeMs = elapsed, hyDisplayMs = displayMs, hyWaitMs = queueWaitMs,
                        translationPrefillMs = timings.prefillMs, translationDecodeMs = timings.decodeMs,
                        translationFirstTokenMs = timings.firstTokenMs,
                        translationInputTokens = timings.inputTokens, translationOutputTokens = timings.outputTokens,
                        translationCacheTokens = timings.cacheReusedTokens,
                        translationFirstVisibleMs = timings.firstVisibleMs, translationCompleteMs = timings.completeMs,
                        translationPrefillDecodeUs = timings.prefillDecodeUs,
                        translationPrefillSyncUs = timings.prefillSyncUs,
                        translationSamplingUs = timings.samplingUs,
                        translationDecodeComputeUs = timings.decodeComputeUs,
                        translationDecodeSyncUs = timings.decodeSyncUs,
                        translationDeviceBytes = timings.deviceBytesAllocated,
                        translationOffloadedLayers = timings.offloadedLayers,
                        translationTotalLayers = timings.totalLayers,
                        translationFallbackCount = timings.fallbackCount,
                        translationTokensPerSecond = timings.tokensPerSecond,
                        translationBackend = engine.backend,
                        audioToProvisionalMs = if (accepted && !isFinal) elapsedSinceNs(request.sourceAudioAtNs) else m.audioToProvisionalMs,
                        stableToProvisionalMs = if (accepted && !isFinal) elapsedSinceNs(request.stableSourceAtNs) else m.stableToProvisionalMs,
                        audioToFinalMs = if (accepted && isFinal) elapsedSinceNs(request.sourceAudioAtNs) else m.audioToFinalMs,
                        finalLatencyMs = if (accepted && isFinal && request.endpointAt != null) now() - request.endpointAt else m.finalLatencyMs) }
                } catch (_: InterruptedException) { break }
                catch (_: CancellationException) {
                    trace("translation-cancelled", request.key, request.requestId,
                        if (lifecycle.isStopping()) "session stopping"
                        else "provisional became obsolete or a final arrived")
                }
                catch (t: Exception) {
                    if (!request.benchmarkOnly && !lifecycle.isStopping()) {
                        CaptionState.skip(request.key, "Translation failed: ${t.message}")
                        rejectRequest(request, "Hy translation failed")
                    }
                }
                finally {
                    if (isFinal) activeFinalEndpoint = null
                    clearActiveHyRequest(request)
                }
                }
        } catch (t: Throwable) {
            val duringLoad = !startupReported
            if (duringLoad) { failStartupIfUnreported(WORKER_HY_TRANSLATION, t); startupReported = true }
            // A stop-cancelled load is the requested outcome, not a failure to report.
            if (lifecycle.isStopping()) trace(if (duringLoad) "hy-load-cancelled" else "hy-worker-aborted",
                details = "reason=${t.message}")
            else { CaptionState.metrics(generation) { it.copy(translatorState = TranslatorState.FAILED) }; fail("Translator: ${t.message}") }
        } finally {
            if (!startupReported) failStartupIfUnreported(WORKER_HY_TRANSLATION,
                CancellationException("Translator stopped before becoming ready"))
            // A resident translator outlives this session on purpose: Stop cancels in-flight
            // requests but never closes the model, so the next Listen skips the GGUF load
            // entirely. Only a non-resident (ML Kit) translator is closed here.
            try {
                runCatching { performanceHint?.close() }
                if (!residencyEligible) translator?.close()
            } finally {
                finalTranslator = null
                hyLoadScope = null
                // The load call has returned, so retiring its token can no longer race it.
                runCatching { loadScope?.release() }
                trace("hy-worker-stopped", details = "residentReused=$reusedResident")
            }
        }
    }
    private fun opusIdentity() = TranslatorIdentity(
        modelBundle = checkNotNull(config.profile.fastBundle), sourceLanguage = config.profile.source,
        targetLanguage = config.profile.target, glossary = "", backendRequestsOpenCl = false,
        batch = config.translationBatch, ubatch = config.translationUbatch)
    /**
     * Serves two purposes depending on preset: the FAST profile's dedicated A/B live-prefix
     * translator (existing behaviour, driven by [provisional]), and — for any profile with an
     * OPUS bundle — an immediate fallback for finals recognized while Hy-MT2 is still loading
     * (driven by [opusFallbackQueue]). Fallback finals are prioritized: correctness beats A/B
     * comparison latency.
     */
    private fun opusLoop() {
        if (!opusWorkerEnabled) return
        var translator: LocalTranslator? = null
        var performanceHint: WorkerPerformanceHint? = null
        var startupReported = false
        var reusedResident = false
        val loadScope = ModelLoadScope(NativeModelLoad())
        opusLoadScope = loadScope
        try {
            performanceHint = WorkerPerformanceHint(ctx, 750)
            val engine = startupBarrier.initialize(WORKER_OPUS) {
                ResidentTranslatorManager.acquire(RESIDENT_SLOT_OPUS, opusIdentity(),
                    { createOpusTranslator(loadScope) }) { reused -> reusedResident = reused }.also {
                    opusTranslator = it
                    if (lifecycle.shouldWarmUp() && !reusedResident) it.warmUp()
                }
            }
            translator = engine
            opusTranslator = engine
            CaptionState.metrics(generation) { it.copy(opusBackend = engine.backend) }
            startupReported = true
            while (lifecycle.shouldRunOpusTranslation()) {
                val fallback = opusFallbackQueue.poll()
                val request = fallback ?: if (opusBenchmarkEnabled) provisional.poll() else null
                if (request == null) { Thread.sleep(20); continue }
                depths()
                if (!CaptionState.current(request.key)) { rejectRequest(request, "caption superseded before OPUS translation"); continue }
                val queueWaitMs = ((nowNs() - request.queuedAtNs) / 1_000_000L).coerceAtLeast(0)
                CaptionState.metrics(generation) { it.copy(opusWaitMs = queueWaitMs) }
                trace(if (fallback != null) "opus-fallback-started" else "opus-started",
                    request.key, request.requestId, "queueWaitMs=$queueWaitMs")
                val startedNs = nowNs()
                try {
                    val result = engine.translate(request.text, request.requestId)
                    if (lifecycle.isCancelled()) break
                    val computedAtNs = nowNs()
                    val elapsed = (computedAtNs - startedNs) / 1_000_000L
                    performanceHint?.report(elapsed)
                    val displayStartedNs: Long
                    val accepted: Boolean
                    if (fallback != null) {
                        // Rank 1: above provisional (0), below a Hy final (2), so Hy can still
                        // upgrade this exact portion later without the UI flickering through an
                        // intermediate stage — translatePortion just republishes as Final again.
                        opusFallbackCount++
                        CaptionState.resultComputed(generation, request.key, computedAtNs, request.requestId,
                            request.portionIdentity)
                        displayStartedNs = nowNs()
                        accepted = CaptionState.translatedPortion(request.key, request.portionIdentity,
                            result, 1, true, request.requestId)
                        CaptionState.metrics(generation) { it.copy(opusStartupFallbackCount = opusFallbackCount) }
                    } else {
                        CaptionState.comparison(generation, request.key, request.text,
                            ComparisonEngine.OPUS, result, elapsed)
                        CaptionState.resultComputed(generation, request.key, computedAtNs, request.requestId,
                            request.portionIdentity)
                        displayStartedNs = nowNs()
                        accepted = CaptionState.translatedPortion(request.key, request.portionIdentity,
                            result, 0, false, request.requestId)
                    }
                    val displayMs = (nowNs() - displayStartedNs) / 1_000_000L
                    if (accepted) recordFirstTranslatedCaption()
                    if (!accepted) rejectRequest(request, "caption revision changed before OPUS result could publish")
                    val timings = engine.timings
                    trace(if (accepted) "opus-displayed-to-state" else "opus-result-computed",
                        request.key, request.requestId, "computeMs=$elapsed displayMs=$displayMs fallback=${fallback != null}")
                    CaptionState.metrics(generation) { it.copy(opusComputeMs = elapsed,
                        opusWaitMs = queueWaitMs, opusPrefillMs = timings.prefillMs,
                        opusGenerationMs = timings.decodeMs, opusFirstTokenMs = timings.firstTokenMs,
                        opusTokensPerSecond = timings.tokensPerSecond) }
                } catch (_: InterruptedException) { break }
                catch (t: Exception) {
                    // A failed fallback attempt is not fatal: the startup buffer still holds the
                    // authoritative copy and Hy will translate it once ready.
                    if (fallback == null && !lifecycle.isStopping()) {
                        CaptionState.error(generation, "OPUS A/B failed: ${t.message}")
                        rejectRequest(request, "OPUS translation failed")
                    } else if (fallback != null) {
                        trace("opus-fallback-failed", request.key, request.requestId, "reason=${t.message}")
                    }
                }
            }
        } catch (t: Throwable) {
            if (!startupReported) { failStartupIfUnreported(WORKER_OPUS, t); startupReported = true }
            if (!lifecycle.isStopping()) CaptionState.error(generation, "OPUS A/B unavailable; Hy-MT2 captions continue: ${t.message}")
        } finally {
            if (!startupReported) failStartupIfUnreported(WORKER_OPUS,
                CancellationException("OPUS worker stopped before becoming ready"))
            // Resident for the same reason Hy is: reloading the OPUS ONNX graphs every session
            // is pure waste when the identity has not changed. Only the manager ever closes it.
            try {
                runCatching { performanceHint?.close() }
            } finally {
                opusTranslator = null
                opusLoadScope = null
                runCatching { loadScope.release() }
                trace("opus-worker-stopped", details = "residentReused=$reusedResident")
            }
        }
    }

    private fun recordFirstTranslatedCaption() {
        if (!firstTranslationRecorded.compareAndSet(false, true)) return
        val elapsedMs = (nowNs() - sessionStartedAtNs) / 1_000_000L
        CaptionState.metrics(generation) { it.copy(timeFromListenToFirstTranslatedCaptionMs = elapsedMs) }
    }
    private fun depths() = CaptionState.metrics(generation) {
        val startup = startupFinals.snapshot()
        it.copy(provisionalDepth = provisional.size() + qualityProvisional.size(), finalDepth = finals.size(),
            captionBacklogMs = captionAge(), startupFinalBuffered = startup.buffered,
            startupFinalDropped = startup.dropped, startupFinalOldestMs = startup.oldestAgeMs)
    }
    private fun failStartupIfUnreported(worker: String, cause: Throwable) {
        if (worker !in startupBarrier.reportedWorkers()) startupBarrier.failed(worker, cause)
    }
    private fun clearActiveHyRequest(request: TranslationRequest) {
        synchronized(hyScheduleLock) { activeHyRequest.compareAndSet(request, null) }
    }
    private fun rejectRequest(request: TranslationRequest, reason: String, computedResult: Boolean = false) {
        CaptionState.resultRejected(generation, reason, computedResult)
        trace("result-rejected", request.key, request.requestId, reason)
    }
    private fun trace(event: String, key: SegmentKey? = null, requestId: Long = 0, details: String = "") {
        val atNs = nowNs()
        Log.d("CaptionPipeline", "session=$generation sessionStartNs=$sessionStartedAtNs atNs=$atNs " +
            "sessionElapsedNs=${(atNs - sessionStartedAtNs).coerceAtLeast(0)} segment=${key?.id ?: 0} " +
            "revision=${key?.revision ?: 0} request=$requestId event=$event $details")
    }
    private fun captionAge(): Long = listOfNotNull(activeFinalEndpoint, finals.peek()?.endpointAt).minOrNull()?.let { (now() - it).coerceAtLeast(0) } ?: 0
    private fun now() = SystemClock.elapsedRealtime()
    private fun nowNs() = System.nanoTime()
    private fun elapsedSinceNs(timestampNs: Long?): Long = timestampNs?.let { ((nowNs() - it) / 1_000_000L).coerceAtLeast(0) } ?: 0L

    companion object {
        private const val WORKER_RECOGNITION = "recognition"
        private const val WORKER_HY_TRANSLATION = "hy-translation"
        private const val WORKER_CORRECTION = "parakeet-correction"
        private const val WORKER_OPUS = "opus-benchmark"
        private const val RESIDENT_SLOT_HY = "hy"
        private const val RESIDENT_SLOT_OPUS = "opus"
    }
}
