package com.asr.live.asr

import android.content.Context
import android.net.Uri
import android.os.SystemClock
import android.util.Log
import com.asr.live.audio.AudioCapture
import com.asr.live.audio.AudioSignalSnapshot
import com.asr.live.audio.AudioSignalTelemetry
import com.asr.live.audio.Pcm16WavReader
import com.asr.live.audio.pcm16Sha256
import com.asr.live.model.ModelCatalog
import com.asr.live.model.ModelStore
import com.asr.live.pipeline.PcmBuffer
import com.asr.live.service.MemoryUsage
import java.security.MessageDigest
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference

enum class AsrWavBenchmarkStage { CPU_VS_QNN, PARAKEET_OFF_VS_ON }

private object AsrWavCandidateOrder {
    private val runCount = AtomicInteger()
    fun candidates() = if (runCount.getAndIncrement() % 2 == 0) listOf(false, true) else listOf(true, false)
}

data class WordErrorCounts(val errors: Int, val substitutions: Int, val deletions: Int, val insertions: Int, val referenceWords: Int) {
    val wer: Double get() = errors.toDouble() / referenceWords.coerceAtLeast(1)
}

data class CorrectionBenchmarkOutput(
    val source: String,
    val corrected: String,
    val usedCorrection: Boolean,
    val queueMs: Long,
    val decodeMs: Long,
    val sourceStartSample: Long,
    val sourceEndSample: Long,
    val pcm16Sha256: String,
    val error: String? = null,
)

data class AsrWavBenchmarkResult(
    val stage: AsrWavBenchmarkStage,
    val backend: String,
    val correctionEnabled: Boolean,
    val inputSha256: String,
    val sampleRateHz: Int,
    val durationMs: Long,
    val hypothesis: String,
    val reference: String,
    val wordErrors: WordErrorCounts?,
    val numberWordErrors: WordErrorCounts?,
    val negationWordErrors: WordErrorCounts?,
    val modelLoadMs: Long,
    val correctionModelLoadMs: Long,
    val elapsedMs: Long,
    val realTimeFactor: Double,
    val firstAsrTextMs: Long?,
    val asrFinalizeMs: Long,
    val acceptP50Ms: Long,
    val acceptP95Ms: Long,
    val decodeCalls: Long,
    val decodeP50Ms: Long,
    val decodeP95Ms: Long,
    val correctionOutputs: List<CorrectionBenchmarkOutput>,
    val correctedHypothesis: String,
    val correctedWordErrors: WordErrorCounts?,
    val correctedNumberWordErrors: WordErrorCounts?,
    val correctedNegationWordErrors: WordErrorCounts?,
    val correctionDropped: Int,
    val correctionErrors: Int,
    val correctionTimedOut: Boolean,
    val audio: AudioSignalSnapshot,
    val maxInputScheduleLagMs: Long,
    val memoryPssKb: Long,
    val qnnPssKb: Long,
    val qnnProcessPresent: Boolean,
    val availableKb: Long,
    val thermalStatus: String,
    val thermalMaxC: Double?,
    val readableThermalSensors: Int,
    val error: String? = null,
) {
    fun reportLine(): String = if (error != null) "$stage · $backend · ERROR: $error" else
        "$stage · $backend · Parakeet ${if (correctionEnabled) "on" else "off"} · SHA-256 $inputSha256\n" +
            "WER ${wordErrors?.let { "%.3f".format(it.wer) } ?: "reference absent"} " +
            "(S/D/I ${wordErrors?.substitutions ?: "—"}/${wordErrors?.deletions ?: "—"}/${wordErrors?.insertions ?: "—"}) · " +
            "number errors ${numberWordErrors?.errors ?: "—"} · negation errors ${negationWordErrors?.errors ?: "—"}\n" +
            "cold Nemotron/QNN load $modelLoadMs ms · Parakeet load $correctionModelLoadMs ms · first ASR text ${firstAsrTextMs ?: "—"} ms · " +
            "ASR finish $asrFinalizeMs ms · stream p50/p95 ${acceptP50Ms}/${acceptP95Ms} ms · " +
            "${decodeCalls} actual decode calls p50/p95 ${decodeP50Ms}/${decodeP95Ms} ms · RTF ${"%.2f".format(realTimeFactor)} · " +
            "schedule lag ${maxInputScheduleLagMs} ms · input ${audio.sampleRateHz} Hz ${audio.channels} ch · " +
            "${audio.samples} samples, callback timestamp gaps ${audio.timestampGaps} " +
            "[${audio.firstMonotonicNs ?: "—"},${audio.lastMonotonicNs ?: "—"}] ns · " +
            "RMS ${"%.4f".format(audio.rms)} peak ${"%.4f".format(audio.peak)} clipped ${audio.clippedSamples} · " +
            "PSS process group ${memoryPssKb / 1024} MiB, :qnn ${if (qnnProcessPresent) "${qnnPssKb / 1024} MiB" else "not running"} · " +
            "thermal $thermalStatus · max readable sensor ${thermalMaxC?.let { "%.1f°C".format(it) } ?: "unavailable"} " +
            "($readableThermalSensors zones) · corrected segments ${correctionOutputs.count { it.usedCorrection }}/${correctionOutputs.size}, " +
            "queue skips $correctionDropped, decode errors $correctionErrors, worker timed out $correctionTimedOut\n" +
            "Parakeet combined WER ${correctedWordErrors?.let { "%.3f".format(it.wer) } ?: "—"} · " +
            "number errors ${correctedNumberWordErrors?.errors ?: "—"} · negation errors ${correctedNegationWordErrors?.errors ?: "—"}\n" +
            "Reference: $reference\nASR: $hypothesis\nParakeet combined: $correctedHypothesis" +
            correctionOutputs.joinToString(separator = "", prefix = "\n") { row ->
                "\nParakeet ${if (row.usedCorrection) "corrected" else "fallback"}: ${row.corrected} · " +
                    "queue/decode ${row.queueMs}/${row.decodeMs} ms · ${row.error ?: "ok"} · " +
                    "source samples [${row.sourceStartSample},${row.sourceEndSample}) · PCM16 ${row.pcm16Sha256}"
            }
}

/** Same-file, 100 ms paced input comparison. Direct injection is isolated from the live mic queue. */
class AsrWavBenchmark(private val context: Context) {
    fun run(uri: Uri, stage: AsrWavBenchmarkStage, reference: String, threads: Int, correctionThreads: Int,
            retainResources: () -> AutoCloseable = { AutoCloseable {} },
            checkCancelled: () -> Unit = {}): List<AsrWavBenchmarkResult> {
        val sha = hashInput(uri)
        val candidates = AsrWavCandidateOrder.candidates()
        val correctionResources = com.asr.live.service.BenchmarkResourceGuard(retainResources)
        val asrResources = com.asr.live.service.BenchmarkResourceGuard(retainResources)
        if (stage == AsrWavBenchmarkStage.CPU_VS_QNN && !com.asr.live.BuildConfig.QNN_ENABLED)
            return listOf(errorResult(stage, sha, "QNN comparison requires a QNN-enabled build"))
        return candidates.map { enabled ->
            checkCancelled()
            if (!correctionResources.isIdle() || !asrResources.isIdle()) return@map errorResult(stage, sha,
                "The previous ASR benchmark worker is still using its native model; wait for cleanup")
            runCatching { runCandidate(uri, sha, stage, reference, threads, correctionThreads, enabled,
                correctionResources, asrResources, checkCancelled) }
                .getOrElse {
                    if (it is java.util.concurrent.CancellationException) throw it
                    errorResult(stage, sha, it.message ?: it.javaClass.simpleName)
                }
        }
    }

    private fun runCandidate(uri: Uri, hash: String, stage: AsrWavBenchmarkStage, reference: String,
                             threads: Int, correctionThreads: Int, enabled: Boolean,
                             correctionResources: com.asr.live.service.BenchmarkResourceGuard,
                             asrResources: com.asr.live.service.BenchmarkResourceGuard,
                             checkCancelled: () -> Unit): AsrWavBenchmarkResult {
        ModelStore.verify(context, ModelCatalog.NEMOTRON)
        val qnn = stage == AsrWavBenchmarkStage.CPU_VS_QNN && enabled
        val correctionEnabled = stage == AsrWavBenchmarkStage.PARAKEET_OFF_VS_ON && enabled
        var backend = if (qnn) "QNN requested · initializing" else "CPU"
        val corrections = CopyOnWriteArrayList<CorrectionBenchmarkOutput>()
        var correctionLane: CorrectionLane? = null
        var engine: AsrEngine? = null
        var engineOwnership: AutoCloseable? = null
        val audioPcm = PcmBuffer()
        var currentSegmentFirstSample: Long? = null
        val completedSegments = CopyOnWriteArrayList<FinalAsrSegment>()
        val acceptMs = mutableListOf<Long>()
        val audioTelemetry = AudioSignalTelemetry()
        val firstTextNs = AtomicReference<Long?>(null)
        val correctionDropped = AtomicInteger()

        try {
            if (correctionEnabled) {
                checkCancelled()
                correctionLane = CorrectionLane(context, correctionThreads, corrections, correctionResources)
                correctionLane.startAndAwaitReady()
            }
            val correctionLoadMs = correctionLane?.modelLoadMs ?: 0L
            val onPartial: (String) -> Unit = { text ->
                if (text.isNotBlank()) firstTextNs.compareAndSet(null, SystemClock.elapsedRealtimeNanos())
            }
            val onFinal: (String) -> Unit = { text ->
                val segment = audioPcm.take()
                val start = currentSegmentFirstSample
                currentSegmentFirstSample = null
                val end = if (segment != null && start != null) start + segment.size else null
                if (text.isNotBlank()) {
                    if (segment != null) completedSegments += FinalAsrSegment(text.trim(), start ?: -1, end ?: -1)
                    else completedSegments += FinalAsrSegment(text.trim(), start ?: -1, start ?: -1)
                    firstTextNs.compareAndSet(null, SystemClock.elapsedRealtimeNanos())
                }
                if (correctionEnabled && text.isNotBlank()) {
                    if (segment == null) correctionDropped.incrementAndGet()
                    else {
                        val accepted = checkNotNull(correctionLane).offer(text.trim(), segment, start ?: -1, end ?: -1)
                        segment.fill(0f)
                        if (!accepted)
                            correctionDropped.incrementAndGet()
                    }
                } else segment?.fill(0f)
            }
            val sessionId = SystemClock.elapsedRealtimeNanos()
            val loadStarted = SystemClock.elapsedRealtimeNanos()
            checkCancelled()
            engineOwnership = asrResources.acquire()
            engine = if (qnn) QnnEngine(context, "nl", threads, onPartial, onFinal,
                { active -> backend = active }, {}, sessionId, { profile ->
                    if (profile.rpcCalls == 0L || profile.rpcCalls % 64L == 0L) Log.i(
                        "LiveCaptionsBenchmark", "qnn_profile session=$sessionId calls=${profile.rpcCalls} " +
                            "init_ns=${profile.initializeTotalNanos} library_ns=${profile.libraryLoadNanos} " +
                            "dsp_copy_ns=${profile.dspCopyNanos} dsp_setup_ns=${profile.dspSetupNanos} " +
                            "recognizer_prepare_ns=${profile.recognizerPrepareNanos} " +
                            "client_queue_p50_ns=${profile.clientQueueP50Nanos} client_queue_p95_ns=${profile.clientQueueP95Nanos} " +
                            "rpc_p50_ns=${profile.rpcP50Nanos} rpc_p95_ns=${profile.rpcP95Nanos} " +
                            "service_p50_ns=${profile.serviceP50Nanos} service_p95_ns=${profile.serviceP95Nanos} " +
                            "binder_p50_ns=${profile.binderAndMarshallingP50Nanos} " +
                            "binder_p95_ns=${profile.binderAndMarshallingP95Nanos} " +
                            "feed_ns=${profile.audioFeedNanos} combined_decode_ns=${profile.combinedDecodeNanos} " +
                            "result_ns=${profile.resultNanos} endpoint_ns=${profile.endpointCheckNanos} " +
                            "graph_breakdown=${profile.graphBreakdown}")
                })
            else EngineFactory.create(context, ModelCatalog.NEMOTRON, "nl", "transcribe", onPartial, onFinal, threads)
            val loadMs = (SystemClock.elapsedRealtimeNanos() - loadStarted) / 1_000_000
            val audioStartNs = SystemClock.elapsedRealtimeNanos()
            var maxLagMs = 0L
            var remainingAudioSamples = 0L
            context.contentResolver.openInputStream(uri)?.use { source ->
                Pcm16WavReader(source).use { wav ->
                    remainingAudioSamples = wav.frameCount
                    val chunk = FloatArray(AudioCapture.CHUNK_SAMPLES)
                    var consumed = 0L
                    var count: Int
                    while (wav.readSamples(chunk).also { count = it } > 0) {
                        checkCancelled()
                        val samples = chunk.copyOf(count)
                        if (currentSegmentFirstSample == null) currentSegmentFirstSample = consumed
                        audioPcm.append(samples)
                        val dueNs = audioStartNs + consumed * 1_000_000_000L / wav.sampleRateHz
                        sleepUntil(dueNs)
                        audioTelemetry.accept(samples, SystemClock.elapsedRealtimeNanos())
                        val acceptStart = SystemClock.elapsedRealtimeNanos()
                        checkNotNull(engine).accept(samples)
                        val acceptEnd = SystemClock.elapsedRealtimeNanos()
                        acceptMs += (acceptEnd - acceptStart) / 1_000_000
                        consumed += count
                        maxLagMs = maxOf(maxLagMs, ((acceptEnd - (audioStartNs + consumed * 1_000_000_000L / wav.sampleRateHz)) / 1_000_000).coerceAtLeast(0))
                        samples.fill(0f)
                    }
                    check(consumed == wav.frameCount) { "WAV sample count changed during read" }
                }
            } ?: error("Could not open selected WAV")
            val finishStart = SystemClock.elapsedRealtimeNanos()
            checkNotNull(engine).finish()
            val finalizeMs = (SystemClock.elapsedRealtimeNanos() - finishStart) / 1_000_000
            val audioEndNs = SystemClock.elapsedRealtimeNanos()
            val decode = checkNotNull(engine).decodeStats
            // Capture process-group PSS and thermal state while both Nemotron and Parakeet are resident.
            val memory = MemoryUsage.sample(context)
            val correctionDrained = correctionLane?.closeAndDrain() ?: true
            val audioStats = audioTelemetry.snapshot()
            val orderedSegments = completedSegments.sortedBy { it.startSample }
            val hypothesis = orderedSegments.joinToString(" ") { it.text }.replace(Regex("\\s+"), " ").trim()
            val correctionByRange = corrections.associateBy { it.sourceStartSample to it.sourceEndSample }
            val correctedHypothesis = orderedSegments.joinToString(" ") { segment ->
                correctionByRange[segment.startSample to segment.endSample]?.corrected ?: segment.text
            }.replace(Regex("\\s+"), " ").trim()
            val normalizedReference = WordErrorScorer.normalize(reference)
            val normalizedHypothesis = WordErrorScorer.normalize(hypothesis)
            val normalizedCorrected = WordErrorScorer.normalize(correctedHypothesis)
            val errors = if (normalizedReference.isEmpty()) null else WordErrorScorer.count(normalizedReference, normalizedHypothesis)
            val referenceNumbers = normalizedReference.filter { token -> token.any(Char::isDigit) }
            val hypothesisNumbers = normalizedHypothesis.filter { token -> token.any(Char::isDigit) }
            val correctedNumbers = normalizedCorrected.filter { token -> token.any(Char::isDigit) }
            val numberErrors = if (normalizedReference.isEmpty()) null else WordErrorScorer.count(referenceNumbers, hypothesisNumbers)
            val negationErrors = if (normalizedReference.isEmpty()) null else
                WordErrorScorer.countNegations(normalizedReference, normalizedHypothesis)
            val correctedErrors = if (normalizedReference.isEmpty()) null else WordErrorScorer.count(normalizedReference, normalizedCorrected)
            val correctedNumberErrors = if (normalizedReference.isEmpty()) null else WordErrorScorer.count(referenceNumbers, correctedNumbers)
            val correctedNegationErrors = if (normalizedReference.isEmpty()) null else
                WordErrorScorer.countNegations(normalizedReference, normalizedCorrected)
            val sortedAccept = acceptMs.sorted()
            val elapsedMs = (audioEndNs - audioStartNs) / 1_000_000
            return AsrWavBenchmarkResult(
                stage = stage, backend = backend, correctionEnabled = correctionEnabled, inputSha256 = hash,
                sampleRateHz = audioStats.sampleRateHz,
                durationMs = remainingAudioSamples * 1000 / AudioCapture.SAMPLE_RATE,
                hypothesis = hypothesis, reference = reference, wordErrors = errors,
                numberWordErrors = numberErrors, negationWordErrors = negationErrors,
                modelLoadMs = loadMs, correctionModelLoadMs = correctionLoadMs, elapsedMs = elapsedMs,
                realTimeFactor = elapsedMs.toDouble() / maxOf(1, remainingAudioSamples * 1000 / AudioCapture.SAMPLE_RATE),
                firstAsrTextMs = firstTextNs.get()?.let { ((it - audioStartNs) / 1_000_000).coerceAtLeast(0) },
                asrFinalizeMs = finalizeMs, acceptP50Ms = percentile(sortedAccept, .50),
                acceptP95Ms = percentile(sortedAccept, .95), decodeCalls = decode.calls,
                decodeP50Ms = decode.p50Nanos / 1_000_000, decodeP95Ms = decode.p95Nanos / 1_000_000,
                correctionOutputs = corrections.toList(), correctedHypothesis = correctedHypothesis,
                correctedWordErrors = correctedErrors, correctedNumberWordErrors = correctedNumberErrors,
                correctedNegationWordErrors = correctedNegationErrors, correctionDropped = correctionDropped.get(),
                correctionErrors = correctionLane?.decodeErrors ?: 0, correctionTimedOut = !correctionDrained,
                audio = audioStats, maxInputScheduleLagMs = maxLagMs, memoryPssKb = memory.appPssKb,
                qnnPssKb = memory.qnnPssKb, qnnProcessPresent = memory.qnnProcessPresent,
                availableKb = memory.availableKb, thermalStatus = memory.thermalStatus,
                thermalMaxC = memory.thermalMaxC, readableThermalSensors = memory.readableThermalSensors)
        } finally {
            audioPcm.take()?.fill(0f)
            correctionLane?.closeAndDrain()
            if (runCatching { engine?.release() }.isSuccess) engineOwnership?.close()
        }
    }

    private fun hashInput(uri: Uri): String {
        val digest = MessageDigest.getInstance("SHA-256")
        context.contentResolver.openInputStream(uri)?.use { input ->
            val bytes = ByteArray(64 * 1024)
            while (true) {
                val count = input.read(bytes)
                if (count < 0) break
                digest.update(bytes, 0, count)
            }
        } ?: error("Could not open selected WAV")
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    private fun sleepUntil(deadlineNs: Long) {
        var remaining = deadlineNs - SystemClock.elapsedRealtimeNanos()
        while (remaining > 0) {
            val millis = remaining / 1_000_000
            if (millis > 0) Thread.sleep(millis) else Thread.yield()
            remaining = deadlineNs - SystemClock.elapsedRealtimeNanos()
        }
    }

    private fun percentile(sorted: List<Long>, p: Double): Long = sorted.getOrNull(
        ((kotlin.math.ceil(sorted.size * p).toInt() - 1).coerceIn(0, (sorted.size - 1).coerceAtLeast(0)))) ?: 0

    private fun errorResult(stage: AsrWavBenchmarkStage, sha: String, error: String) = AsrWavBenchmarkResult(
        stage = stage, backend = "unavailable", correctionEnabled = false, inputSha256 = sha,
        sampleRateHz = 0, durationMs = 0, hypothesis = "", reference = "", wordErrors = null,
        numberWordErrors = null, negationWordErrors = null, modelLoadMs = 0, correctionModelLoadMs = 0,
        elapsedMs = 0, realTimeFactor = 0.0, firstAsrTextMs = null, asrFinalizeMs = 0,
        acceptP50Ms = 0, acceptP95Ms = 0, decodeCalls = 0, decodeP50Ms = 0, decodeP95Ms = 0,
        correctionOutputs = emptyList(), correctedHypothesis = "", correctedWordErrors = null,
        correctedNumberWordErrors = null, correctedNegationWordErrors = null, correctionDropped = 0,
        correctionErrors = 0, correctionTimedOut = false, audio = AudioSignalSnapshot(),
        maxInputScheduleLagMs = 0, memoryPssKb = 0, qnnPssKb = 0, qnnProcessPresent = false,
        availableKb = 0, thermalStatus = "Unavailable", thermalMaxC = null,
        readableThermalSensors = 0, error = error)

    private data class CorrectionJob(
        val source: String, val samples: FloatArray, val enqueuedNs: Long,
        val startSample: Long, val endSample: Long, val pcm16Sha256: String,
    )
    private data class FinalAsrSegment(val text: String, val startSample: Long, val endSample: Long)

    private class CorrectionLane(
        context: Context,
        private val threads: Int,
        private val output: CopyOnWriteArrayList<CorrectionBenchmarkOutput>,
        private val resources: com.asr.live.service.BenchmarkResourceGuard,
    ) {
        private val queue = ArrayBlockingQueue<CorrectionJob>(1)
        private val ready = CountDownLatch(1)
        private val failure = AtomicReference<Throwable?>(null)
        private val failures = AtomicInteger()
        @Volatile var modelLoadMs: Long = 0
            private set
        val decodeErrors: Int get() = failures.get()
        @Volatile private var closing = false
        @Volatile private var drainAttempted = false
        @Volatile private var ownership: AutoCloseable? = null
        private val worker = Thread({
            var corrector: com.asr.live.asr.EndpointCorrector? = null
            try {
                val started = SystemClock.elapsedRealtimeNanos()
                ModelStore.verify(context, ModelCatalog.PARAKEET)
                corrector = com.asr.live.asr.EndpointCorrector(ModelStore.dir(context, ModelCatalog.PARAKEET.id), threads)
                corrector.warmUp()
                modelLoadMs = (SystemClock.elapsedRealtimeNanos() - started) / 1_000_000L
            } catch (t: Throwable) { failure.set(t) }
            finally { ready.countDown() }
            try {
                while (!closing || queue.isNotEmpty()) {
                    val job = queue.poll(100, TimeUnit.MILLISECONDS) ?: continue
                    try {
                        val start = SystemClock.elapsedRealtimeNanos()
                        val result = checkNotNull(corrector).decode(job.samples)
                        val decodeMs = (SystemClock.elapsedRealtimeNanos() - start) / 1_000_000
                        output += CorrectionBenchmarkOutput(job.source, result, true,
                            ((start - job.enqueuedNs) / 1_000_000).coerceAtLeast(0), decodeMs,
                            job.startSample, job.endSample, job.pcm16Sha256)
                    } catch (t: Throwable) {
                        failure.compareAndSet(null, t); failures.incrementAndGet()
                        output += CorrectionBenchmarkOutput(job.source, job.source, false,
                            ((SystemClock.elapsedRealtimeNanos() - job.enqueuedNs) / 1_000_000L).coerceAtLeast(0), 0,
                            job.startSample, job.endSample, job.pcm16Sha256, t.message ?: t.javaClass.simpleName)
                    }
                    finally { job.samples.fill(0f) }
                }
            } catch (_: InterruptedException) {
                Thread.currentThread().interrupt()
            } finally {
                // A timeout only releases the UI. Keep benchmark admission reserved until the
                // worker has actually left native code and freed its recognizer.
                if (runCatching { corrector?.close() }.isSuccess) ownership?.close()
            }
        }, "wav-benchmark-parakeet").apply { isDaemon = true }

        fun startAndAwaitReady() {
            ownership = resources.acquire()
            try { worker.start() }
            catch (t: Throwable) { ownership?.close(); throw t }
            check(ready.await(120, TimeUnit.SECONDS)) { "Parakeet initialization timed out" }
            failure.get()?.let { throw IllegalStateException("Parakeet initialization failed: ${it.message}", it) }
        }

        fun offer(source: String, samples: FloatArray, start: Long, end: Long): Boolean {
            val copied = samples.copyOf()
            val accepted = queue.offer(CorrectionJob(source, copied, SystemClock.elapsedRealtimeNanos(), start, end,
                pcm16Sha256(copied)))
            if (!accepted) copied.fill(0f)
            return accepted
        }

        fun closeAndDrain(): Boolean {
            if (!worker.isAlive) return true
            if (drainAttempted) return false
            drainAttempted = true
            closing = true
            worker.join(30_000)
            val drained = !worker.isAlive
            if (!drained) Log.w("LiveCaptionsBenchmark", "Parakeet is still using its native recognizer; worker retains ownership")
            return drained
        }
    }
}
