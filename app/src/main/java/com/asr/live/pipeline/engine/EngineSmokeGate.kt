package com.asr.live.pipeline.engine

import com.asr.live.i18n.LocalTranslator
import com.asr.live.pipeline.Profile
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

/** Result of Stage A compatibility/smoke testing for one [EngineCandidate]. */
sealed class SmokeOutcome {
    data class Qualified(val actualBackend: String, val loadMs: Long, val translateMs: Long,
                         val backendMismatch: Boolean) : SmokeOutcome() {
        val backend: Backend get() = classifyBackend(actualBackend)
    }

    companion object {
        fun classifyBackend(actualBackend: String): Backend = when {
            actualBackend.contains("SDK-managed", ignoreCase = true) -> Backend.SDK_MANAGED
            actualBackend.contains("OpenCL", ignoreCase = true) &&
                !actualBackend.contains("fallback", ignoreCase = true) -> Backend.OPENCL
            else -> Backend.CPU
        }
    }
    data class Failed(val reason: String) : SmokeOutcome()
    data class Unsupported(val reason: String) : SmokeOutcome()
}

/**
 * A crude, script-aware guard against a translator that loaded but silently echoed or dropped
 * the source instead of translating it. This is a smoke check, not a quality judgment — real
 * quality comes from [ChrFScorer] against the benchmark corpus in Stage B.
 */
fun interface LanguageSmokeCheck { fun looksLikeTargetLanguage(output: String, input: String): Boolean }

object LanguageSmokeChecks {
    private val cjk = Regex("[\\u3400-\\u9FFF\\u3040-\\u30FF\\uAC00-\\uD7AF]")
    val default = LanguageSmokeCheck { output, input ->
        output.isNotBlank() && output.trim() != input.trim()
    }
    /** English/French/Dutch output must not still be in the CJK source script. */
    val notCjk = LanguageSmokeCheck { output, input -> default.looksLikeTargetLanguage(output, input) && !cjk.containsMatchIn(output) }

    fun forProfile(profile: Profile): LanguageSmokeCheck = if (profile.source == "zh") notCjk else default
}

/**
 * Stage A gate: load, translate a nonempty smoke sentence, check it looks like the target
 * language, cancel a second in-flight request, close, and report the actual backend.
 *
 * This exercises the real [LocalTranslator] contract used by [com.asr.live.service.CaptionSession]
 * — the same interface, not a simplified imitation — so a candidate that passes here has
 * demonstrably loaded, tokenized/templated, translated and cleaned up through the real adapter.
 * It does not, by itself, prove on-device latency or quality; those are Stage B/C/D concerns.
 */
object EngineSmokeGate {
    private const val SMOKE_TEXT = "Good morning, could you please send the report before noon?"
    private const val CANCEL_PROBE_TEXT = "This second request should be cancelled before it finishes."
    const val SMOKE_TRANSLATION_TIMEOUT_MS = 60_000L
    const val CANCEL_PROBE_TIMEOUT_MS = 5_000L
    internal const val CLEANUP_WAIT_MS = 1_000L

    internal data class TimedTranslation(val value: String?, val failure: Throwable?, val timedOut: Boolean)
    private data class CancelProbe(val failure: SmokeOutcome.Failed?, val timedOut: Boolean)

    fun run(profile: Profile, requestedOpenCl: Boolean,
            check: LanguageSmokeCheck = LanguageSmokeChecks.forProfile(profile),
            createTranslator: () -> LocalTranslator): SmokeOutcome {
        val loadStartedNs = System.nanoTime()
        val translator = try { createTranslator() }
        catch (t: Throwable) { return SmokeOutcome.Failed("load failed: ${t.message}") }
        val loadMs = (System.nanoTime() - loadStartedNs) / 1_000_000L
        val cleanup = TranslatorCleanup(translator)
        var timedOut = false
        var outcome: SmokeOutcome = try {
            val translateStartedNs = System.nanoTime()
            val smoke = translateBounded(translator, SMOKE_TEXT, SMOKE_REQUEST_ID,
                SMOKE_TRANSLATION_TIMEOUT_MS, cleanup)
            timedOut = smoke.timedOut
            val result = smoke.value
            if (result == null) {
                SmokeOutcome.Failed(if (smoke.timedOut)
                    "smoke translation timed out after ${SMOKE_TRANSLATION_TIMEOUT_MS}ms"
                else "translation failed: ${smoke.failure?.message}")
            } else {
                val translateMs = (System.nanoTime() - translateStartedNs) / 1_000_000L
                if (!check.looksLikeTargetLanguage(result, SMOKE_TEXT)) {
                    SmokeOutcome.Failed("output did not look like the target language: \"${result.take(80)}\"")
                } else {
                    val cancelProbe = checkCancellation(translator)
                    timedOut = timedOut || cancelProbe.timedOut
                    if (cancelProbe.failure != null) cancelProbe.failure
                    else {
                        runCatching { translator.backend }.fold(
                            onSuccess = { backend ->
                                val backendKind = SmokeOutcome.classifyBackend(backend)
                                val mismatch = requestedOpenCl && backendKind !in
                                    setOf(Backend.OPENCL, Backend.SDK_MANAGED)
                                if (mismatch) SmokeOutcome.Failed("OpenCL was requested but the translator ran on $backend")
                                else SmokeOutcome.Qualified(backend, loadMs, translateMs, false)
                            },
                            onFailure = { SmokeOutcome.Failed("backend report failed: ${it.message}") })
                    }
                }
            }
        } catch (t: Throwable) { SmokeOutcome.Failed("unexpected error: ${t.message}") }

        cleanup.start()
        // Once a translation has timed out, cleanup belongs to its own daemon thread. In
        // particular, NativeTranslator.close() waits for the JNI context lock and must never
        // turn the bounded timeout above into an unbounded calibration hang.
        if (!timedOut) {
            val closeOutcome = cleanup.await(CLEANUP_WAIT_MS)
            if (closeOutcome is CleanupOutcome.TimedOut && outcome is SmokeOutcome.Qualified)
                outcome = SmokeOutcome.Failed("cleanup (close) did not finish within ${CLEANUP_WAIT_MS}ms")
            else if (closeOutcome is CleanupOutcome.Failed && outcome is SmokeOutcome.Qualified)
                outcome = SmokeOutcome.Failed("cleanup (close) failed: ${closeOutcome.error.message}")
        }
        return outcome
    }

    /**
     * Starts a request on a worker thread, cancels it almost immediately, and requires the
     * worker to actually finish (not hang) within a bounded window. A translator that ignores
     * cancellation and blocks forever fails Stage A regardless of translation quality.
     */
    private fun checkCancellation(translator: LocalTranslator): CancelProbe {
        val started = CountDownLatch(1)
        val finished = CountDownLatch(1)
        val worker = Thread({
            started.countDown()
            // Either a clean result or an exception is an acceptable response to cancellation;
            // what matters is that the call actually returns instead of hanging.
            try { runCatching { translator.translate(CANCEL_PROBE_TEXT, requestId = CANCEL_PROBE_REQUEST_ID) } }
            finally { finished.countDown() }
        }, "engine-smoke-cancel-probe").apply { isDaemon = true }
        worker.start()
        if (!started.await(2, TimeUnit.SECONDS)) return CancelProbe(
            SmokeOutcome.Failed("cancellation probe never started"), timedOut = true)
        Thread({ runCatching { translator.cancel(CANCEL_PROBE_REQUEST_ID) } }, "engine-smoke-cancel")
            .apply { isDaemon = true }.start()
        val settled = finished.await(CANCEL_PROBE_TIMEOUT_MS, TimeUnit.MILLISECONDS)
        if (!settled) return CancelProbe(
            SmokeOutcome.Failed("translator ignored cancel(requestId) and kept running past its deadline"), timedOut = true)
        return CancelProbe(null, timedOut = false)
    }

    internal fun translateBounded(translator: LocalTranslator, text: String, requestId: Long,
                                  timeoutMs: Long, cleanup: TranslatorCleanup): TimedTranslation {
        val done = CountDownLatch(1)
        val value = AtomicReference<String?>(null)
        val failure = AtomicReference<Throwable?>(null)
        Thread({
            try { value.set(translator.translate(text, requestId)) }
            catch (t: Throwable) { failure.set(t) }
            finally { done.countDown() }
        }, "engine-smoke-translation").apply { isDaemon = true }.start()
        return if (done.await(timeoutMs, TimeUnit.MILLISECONDS))
            TimedTranslation(value.get(), failure.get(), timedOut = false)
        else {
            Thread({ runCatching { translator.cancel(requestId) } }, "engine-smoke-timeout-cancel")
                .apply { isDaemon = true }.start()
            cleanup.start()
            TimedTranslation(null, null, timedOut = true)
        }
    }

    private const val SMOKE_REQUEST_ID = -424_241L
    private const val CANCEL_PROBE_REQUEST_ID = -424_242L
}

internal sealed class CleanupOutcome {
    data object Closed : CleanupOutcome()
    data class Failed(val error: Throwable) : CleanupOutcome()
    data object TimedOut : CleanupOutcome()
}

/** Ensures close is performed once and is never allowed to hold the calibration thread forever. */
internal class TranslatorCleanup(private val translator: LocalTranslator) {
    private val started = AtomicBoolean(false)
    private val done = CountDownLatch(1)
    private val failure = AtomicReference<Throwable?>(null)

    fun start() {
        if (!started.compareAndSet(false, true)) return
        Thread({
            try { translator.close() } catch (t: Throwable) { failure.set(t) }
            finally { done.countDown() }
        }, "engine-translator-cleanup").apply { isDaemon = true }.start()
    }

    fun await(timeoutMs: Long): CleanupOutcome {
        start()
        if (!done.await(timeoutMs, TimeUnit.MILLISECONDS)) return CleanupOutcome.TimedOut
        return failure.get()?.let { CleanupOutcome.Failed(it) } ?: CleanupOutcome.Closed
    }
}
