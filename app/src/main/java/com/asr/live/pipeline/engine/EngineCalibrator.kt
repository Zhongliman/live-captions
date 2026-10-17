package com.asr.live.pipeline.engine

import android.content.Context
import com.asr.live.i18n.LocalTranslator
import com.asr.live.i18n.MlKitTranslator
import com.asr.live.i18n.NativeTranslator
import com.asr.live.i18n.createTranslator
import com.asr.live.model.TranslationModels
import com.asr.live.pipeline.Profile
import com.asr.live.service.MemoryUsage
import com.asr.live.service.BenchmarkResourceGuard
import java.io.File
import java.util.concurrent.CancellationException
import java.util.concurrent.atomic.AtomicLong

/**
 * Builds a real, production [LocalTranslator] for one registry candidate. This calls the exact
 * same constructors [com.asr.live.service.CaptionSession] uses — it is not a simplified stand-in
 * adapter, so a candidate that survives calibration through this factory is proven to work
 * through the real integration path, not a mock of it.
 */
object EngineCandidateFactory {
    fun create(ctx: Context, candidate: EngineCandidate, profile: Profile, preferOpenCl: Boolean): LocalTranslator =
        when (candidate.family) {
            EngineFamily.ML_KIT -> MlKitTranslator(profile.source, profile.target).also { it.prepare() }
            EngineFamily.OPUS -> {
                val bundle = checkNotNull(profile.fastBundle) { "${profile.label} has no OPUS bundle" }
                val directory = TranslationModels.verify(ctx, bundle)
                NativeTranslator(directory, 2, true, profile, "", false,
                    ctx.getDir("llama-opencl-cache", Context.MODE_PRIVATE))
            }
            EngineFamily.HY_MT2 -> {
                val quality = checkNotNull(candidate.quality) { "${candidate.engineId} has no TranslationQuality" }
                val bundle = checkNotNull(quality.bundleId) { "${candidate.engineId} has no GGUF bundle" }
                val directory = TranslationModels.verify(ctx, bundle)
                val weightBytes = TranslationModels.bundle(ctx, bundle).files.single { it.path == "model.gguf" }.size
                val admission = MemoryUsage.assessTranslatorLoad(ctx, weightBytes,
                    largeModel = bundle.startsWith("hymt2-7b-"), openCl = preferOpenCl)
                check(admission.allowed) { "Cannot calibrate ${candidate.displayName}. ${admission.description}" }
                NativeTranslator(File(directory, "model.gguf"), 4, false, profile, "", preferOpenCl,
                    ctx.getDir("llama-opencl-cache", Context.MODE_PRIVATE))
            }
            EngineFamily.TRANSLATE_GEMMA, EngineFamily.GEMMA_LITE_RT, EngineFamily.HY_MT2_ULTRA_LOW_BIT ->
                error("${candidate.engineId} is not integrated in this build: ${candidate.blockerReason}")
        }
}

/**
 * Runs Stage A (compatibility/smoke) and Stage B (source-text quality screening) for every
 * eligible registry candidate on one language pair, applies [EngineSelectionPolicy], and
 * persists the result via [CalibrationStore] so it takes effect immediately in this build.
 *
 * This is real orchestration over the real adapters; what it cannot do without the target
 * device is prove real-world latency (Stage C/D are out of scope for this phase) or that the
 * result generalizes beyond the small corpus in [BenchmarkCorpus].
 */
object EngineCalibrator {
    private const val BENCHMARK_ITEM_TIMEOUT_MS = 120_000L
    private val benchmarkRequestIds = AtomicLong(1)
    /**
     * [factory] defaults to the real [EngineCandidateFactory], which loads real native/SDK
     * translators. Tests inject a fake factory to exercise this orchestration deterministically
     * without a device; production callers should never override it.
     */
    fun calibrate(ctx: Context, profile: Profile, requestedOpenCl: Boolean,
                  includeExtendedTier: Boolean = false, store: CalibrationStore = CalibrationStore(ctx),
                  preparationFailures: Map<String, String> = emptyMap(),
                  retainResources: () -> AutoCloseable = { AutoCloseable {} },
                  checkCancelled: () -> Unit = {},
                  factory: (EngineCandidate, Profile, Boolean) -> LocalTranslator =
                      { candidate, p, openCl -> EngineCandidateFactory.create(ctx, candidate, p, openCl) }): SelectionOutcome {
        val toTest = EngineRegistry.forProfile(profile, includeExtendedTier)
        val skippedExtended = EngineRegistry.forProfile(profile, includeExtendedTier = true) - toTest.toSet()
        val corpus = BenchmarkCorpus.heldOut(profile)
        val resources = BenchmarkResourceGuard(retainResources)
        val isolatedFactory = { candidate: EngineCandidate, p: Profile, openCl: Boolean ->
            checkCancelled()
            resources.createTranslator { factory(candidate, p, openCl) }
        }

        val evaluations = toTest.map { candidate ->
            checkCancelled()
            preparationFailures[candidate.engineId]?.let { reason ->
                CandidateEvaluation(candidate, SmokeOutcome.Failed("model preparation failed: $reason"))
            } ?: evaluateCandidate(candidate, profile, requestedOpenCl, corpus, isolatedFactory, checkCancelled)
        }
        val previousRecord = store.getFresh(profile)?.let { record ->
            EngineRegistry.all.firstOrNull { candidate ->
                candidate.engineId == record.engineId && candidate.status == QualificationTier.CANDIDATE &&
                    candidate.supports(profile) && candidate.quality == record.quality &&
                    record.backend in candidate.supportedBackends
            }?.let { record }
        }
        val previousWinner = previousRecord?.let { record ->
            EngineRegistry.all.first { it.engineId == record.engineId } }

        val outcome = EngineSelectionPolicy.select(profile, evaluations, previousWinner, previousRecord?.backend)
        val reportedIds = outcome.reports.mapTo(mutableSetOf()) { it.candidate.engineId }
        val fullReports = outcome.reports + skippedExtended.filterNot { it.engineId in reportedIds }.map {
            CandidateReport(it, SelectionStatus.NOT_TESTED, "extended tier; run full calibration to include it") }
        val result = outcome.copy(reports = fullReports)

        result.winner?.let { winner ->
            val winnerEvaluation = evaluations.firstOrNull { it.candidate == winner }
            // A retained winner may be from the extended tier and therefore intentionally was
            // not evaluated this run. Keep the original record instead of crashing or rewriting
            // it with this run's unrelated score.
            val newlySelected = result.reports.any {
                it.candidate == winner && it.status == SelectionStatus.SELECTED
            }
            if (newlySelected && winnerEvaluation != null && winnerEvaluation.benchmarkFailure == null &&
                winnerEvaluation.smoke is SmokeOutcome.Qualified) {
                val saved = runCatching {
                    store.save(CalibrationRecord(profile = profile, engineId = winner.engineId,
                        quality = winner.quality, chrfScore = winnerEvaluation.chrfScore,
                        fingerprint = DeviceFingerprint.current(), policyVersion = EngineSelectionPolicy.POLICY_VERSION,
                        corpusVersion = BenchmarkCorpus.VERSION, calibratedAtEpochMs = System.currentTimeMillis(),
                        backend = result.selectedBackend ?: winnerEvaluation.smoke.backend))
                }.getOrDefault(false)
                if (!saved) return result.copy(reason = result.reason +
                    " The winner is active now, but saving the calibration record failed.")
            }
        }
        return result
    }

    private fun evaluateCandidate(candidate: EngineCandidate, profile: Profile, requestedOpenCl: Boolean,
                                   corpus: List<CorpusItem>, factory: (EngineCandidate, Profile, Boolean) -> LocalTranslator,
                                   checkCancelled: () -> Unit): CandidateEvaluation {
        // Do not request an unsupported backend from CPU-only engines. For an OpenCL-capable
        // Hy-MT2 candidate, however, the smoke gate requires actual OpenCL execution; CPU fallback
        // is a failed candidate, never an OpenCL calibration result.
        val candidateOpenCl = requestedOpenCl && Backend.OPENCL in candidate.supportedBackends
        val smoke = EngineSmokeGate.run(profile, candidateOpenCl) { factory(candidate, profile, candidateOpenCl) }
        if (smoke !is SmokeOutcome.Qualified) return CandidateEvaluation(candidate, smoke)

        // Stage A already closed its own instance; Stage B gets a fresh one so a leaked
        // resource in one stage cannot mask or corrupt the other's result.
        val translator = try { factory(candidate, profile, candidateOpenCl) }
        catch (cancelled: CancellationException) { throw cancelled }
        catch (t: Throwable) { return CandidateEvaluation(candidate, SmokeOutcome.Failed("Stage B load failed: ${t.message}")) }
        val cleanup = TranslatorCleanup(translator)
        val expectedBackend = when {
            candidate.family == EngineFamily.ML_KIT -> Backend.SDK_MANAGED
            candidateOpenCl -> Backend.OPENCL
            else -> Backend.CPU
        }
        var cleanupDeferred = false
        var cleanupFailure: String? = null
        var totalScore = 0.0
        var scored = 0
        var negationDropped = 0
        var numbersMissing = 0
        val failedItems = mutableListOf<String>()
        try {
            for ((index, item) in corpus.withIndex()) {
                checkCancelled()
                val attempt = EngineSmokeGate.translateBounded(translator, item.source,
                    benchmarkRequestIds.incrementAndGet(), BENCHMARK_ITEM_TIMEOUT_MS, cleanup)
                if (attempt.timedOut) {
                    failedItems += "${item.category}: timed out after ${BENCHMARK_ITEM_TIMEOUT_MS}ms"
                    corpus.drop(index + 1).forEach { failedItems += "${it.category}: not run after timeout" }
                    cleanupDeferred = true
                    break
                }
                if (attempt.failure != null) {
                    failedItems += "${item.category}: ${attempt.failure.message ?: attempt.failure.javaClass.simpleName}"
                    continue
                }
                val output = attempt.value.orEmpty()
                if (output.isBlank()) {
                    failedItems += "${item.category}: empty output"
                    continue
                }
                val backendResult = runCatching { translator.backend }
                if (backendResult.isFailure) {
                    val backendFailure = backendResult.exceptionOrNull()
                    failedItems += "${item.category}: backend report failed: ${backendFailure?.message}"
                    corpus.drop(index + 1).forEach { failedItems += "${it.category}: not run after backend report failure" }
                    break
                }
                val actualBackend = checkNotNull(backendResult.getOrNull())
                val actualBackendKind = SmokeOutcome.classifyBackend(actualBackend)
                if (actualBackendKind != expectedBackend || actualBackendKind !in candidate.supportedBackends) {
                    failedItems += "${item.category}: expected $expectedBackend but Stage B ran on $actualBackend"
                    corpus.drop(index + 1).forEach { failedItems += "${it.category}: not run after backend mismatch" }
                    break
                }
                totalScore += ChrFScorer.score(output, item.reference)
                scored++
                if (SemanticChecks.negationMarkerDropped(output, item.reference, profile.target)) negationDropped++
                numbersMissing += SemanticChecks.numberDigitsMissing(output, item.reference, profile.target).size
            }
            // Failed outputs contribute zero to the score denominator and any failed item makes
            // the candidate ineligible. They can neither disappear from the average nor leave
            // an apparently qualified partial run.
        } catch (cancelled: CancellationException) { throw cancelled }
        catch (t: Throwable) {
            failedItems += "benchmark exception: ${t.message ?: t.javaClass.simpleName}"
        } finally {
            if (!cleanupDeferred) when (val closed = cleanup.await(EngineSmokeGate.CLEANUP_WAIT_MS)) {
                CleanupOutcome.Closed -> Unit
                is CleanupOutcome.Failed -> cleanupFailure = "cleanup failed: ${closed.error.message}"
                CleanupOutcome.TimedOut -> cleanupFailure = "cleanup did not finish within ${EngineSmokeGate.CLEANUP_WAIT_MS}ms"
            }
        }
        cleanupFailure?.let { failedItems += it }
        val mean = if (corpus.isNotEmpty()) totalScore / corpus.size else null
        return CandidateEvaluation(candidate, smoke, mean, scored, negationDropped, numbersMissing,
            itemsAttempted = corpus.size, benchmarkFailure = failedItems.takeIf { it.isNotEmpty() }
                ?.joinToString("; "))
    }
}
