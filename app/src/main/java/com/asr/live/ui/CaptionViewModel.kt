package com.asr.live.ui

import android.app.Application
import android.content.Context
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.asr.live.model.*
import com.asr.live.pipeline.*
import com.asr.live.service.*
import com.asr.live.i18n.TranslationBenchmarkPlan
import com.asr.live.i18n.TranslationBenchmarkStage
import com.asr.live.asr.AsrWavBenchmark
import com.asr.live.asr.AsrWavBenchmarkResult
import com.asr.live.asr.AsrWavBenchmarkStage
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.util.concurrent.atomic.AtomicInteger
import com.asr.live.overlay.OverlayPreferences
import com.asr.live.overlay.OverlayOptions
import com.asr.live.pipeline.engine.*

data class TranslationBenchmark(
    val model: String, val elapsedMs: Long, val output: String, val error: String? = null,
    val appProcessGroupPssMiB: Long? = null, val availableMiB: Long? = null,
    val backend: String = "CPU", val requestedOpenCl: Boolean = false,
    val p50Ms: Long = 0, val p95Ms: Long = 0, val prefillMs: Long = 0, val decodeMs: Long = 0,
    val loadMs: Long = 0, val batch: Int = 256, val ubatch: Int = 128,
    val threads: Int = 4, val inputTokens: Long = 0, val outputTokens: Long = 0,
    val cacheReusedTokens: Long = 0, val firstVisibleMs: Long = 0, val completeMs: Long = 0,
    val coldRequestMs: Long = 0,
    val prefillDecodeUs: Long = 0, val prefillSyncUs: Long = 0, val decodeComputeUs: Long = 0,
    val samplingUs: Long = 0, val decodeSyncUs: Long = 0,
    val offloadedLayers: Long = 0, val totalLayers: Long = 0, val deviceBytesAllocated: Long = 0,
    val fallbackCount: Long = 0,
    val quality: TranslationQuality? = null,
    val humanReferenceWer: Double? = null,
    val thermalStatus: String = "Unknown",
    val thermalMaxC: Double? = null,
)

data class ManagedModel(val id: String, val label: String, val bytes: Long, val installed: Boolean, val selectable: Boolean = false, val stored: Boolean = installed)
data class EngineCalibrationProgress(val phase: String, val index: Int = 0, val total: Int = 0, val candidate: String? = null)

internal fun shouldRejectConfigUpdate(busy: Boolean, benchmarkBusy: Boolean, running: Boolean,
                                     automaticCalibration: Boolean): Boolean =
    busy || (benchmarkBusy && !automaticCalibration) || running

class CaptionViewModel(app: Application) : AndroidViewModel(app) {
    private val translationBenchmarkRuns = AtomicInteger()
    private val prefs = app.getSharedPreferences("profiles_v2", Context.MODE_PRIVATE)
    private val overlayPrefs = OverlayPreferences(app)
    private val deviceMode = PerformanceMode.defaultFor(if (android.os.Build.VERSION.SDK_INT >= 31) android.os.Build.SOC_MODEL else null, android.os.Build.MANUFACTURER, android.os.Build.MODEL)
    private val initialProfile = Profile.fromId(prefs.getString("profile", null))
    private val calibrationStore = CalibrationStore(app)
    private val initialCalibration = calibrationStore.getFresh(initialProfile)
    private val preferredQuality = if (prefs.getBoolean("quality.userOverride", false))
            TranslationQuality.entries.firstOrNull { it.name == prefs.getString("quality", null) } else null
    private val initialMode = PerformanceMode.entries.firstOrNull { it.name == prefs.getString("performanceMode", null) }
        ?: deviceMode
    val overlay = overlayPrefs.state
    fun updateOverlay(options: OverlayOptions) = overlayPrefs.update(options)
    override fun onCleared() { overlayPrefs.close(); super.onCleared() }
    private val _managed = MutableStateFlow<List<ManagedModel>>(emptyList())
    val managed = _managed.asStateFlow()
    private val _config = MutableStateFlow(SessionConfig(
        profile = initialProfile,
        performanceMode = initialMode,
        modelId = prefs.getString("model", ModelCatalog.DEFAULT.id) ?: ModelCatalog.DEFAULT.id,
        threads = prefs.getInt("threads", 6).coerceIn(1, 8),
        correctionThreads = prefs.getInt("correctionThreads", 4).let { if (it in setOf(2, 4, 6, 8)) it else 4 },
        // Precedence: an explicit manual override, then a fresh automatic-calibration result
        // for this exact runtime/device (stale ones are never reused silently), then the preset.
        quality = preferredQuality
            ?: initialCalibration?.quality
            ?: when (initialMode) {
                PerformanceMode.FAST -> TranslationQuality.HY_Q4
                PerformanceMode.MAX_QUALITY -> TranslationQuality.HY_7B_Q4
                PerformanceMode.BALANCED -> TranslationQuality.HY_Q8
            },
        qnn = if (prefs.getBoolean("qnn.userOverride", false)) prefs.getBoolean("qnn", false) else false,
        gpuTranslation = if (prefs.getBoolean("opencl.userOverride", false)) prefs.getBoolean("gpuTranslation", false)
            else initialCalibration?.backend == Backend.OPENCL,
        translationThreads = 4,
        translationBatch = 256,
        translationUbatch = 128,
        correction = if (prefs.getBoolean("correction.userOverride", false)) prefs.getBoolean("correction", false)
            else initialMode == PerformanceMode.MAX_QUALITY && initialProfile.correctionSupported,
        glossary = prefs.getString("glossary", "") ?: "",
        primaryEngineId = if (preferredQuality == null) initialCalibration?.engineId else null,
    ))
    val config = _config.asStateFlow()
    private val _ready = MutableStateFlow(false)
    val ready = _ready.asStateFlow()
    private val _busy = MutableStateFlow(false)
    val busy = _busy.asStateFlow()
    val lines = CaptionState.lines
    val lifecycle = CaptionState.lifecycle
    val error = CaptionState.error
    val metrics = CaptionState.metrics
    val comparisons = CaptionState.comparisons
    val download = ModelRepository.state
    private val _benchmark = MutableStateFlow<List<TranslationBenchmark>>(emptyList())
    val benchmark = _benchmark.asStateFlow()
    private val _benchmarkBusy = MutableStateFlow(false)
    val benchmarkBusy = _benchmarkBusy.asStateFlow()
    private val _asrBenchmark = MutableStateFlow<List<AsrWavBenchmarkResult>>(emptyList())
    val asrBenchmark = _asrBenchmark.asStateFlow()
    private val _asrBenchmarkError = MutableStateFlow<String?>(null)
    val asrBenchmarkError = _asrBenchmarkError.asStateFlow()
    fun benchmarkAsrWav(uri: android.net.Uri, stage: AsrWavBenchmarkStage, humanReference: String) {
        if (_busy.value || _benchmarkBusy.value || CaptionState.running.value) return
        val current = _config.value
        _benchmarkBusy.value = true
        _asrBenchmarkError.value = null
        _asrBenchmark.value = emptyList()
        viewModelScope.launch {
            try {
                val rows = withContext(Dispatchers.IO) {
                    AsrWavBenchmark(getApplication()).run(uri, stage, humanReference,
                        current.threads, current.correctionThreads)
                }
                _asrBenchmark.value = rows
                rows.forEach { android.util.Log.i("LiveCaptionsBenchmark", it.reportLine()) }
            } catch (t: Throwable) {
                _asrBenchmarkError.value = t.message ?: t.javaClass.simpleName
            } finally { _benchmarkBusy.value = false }
        }
    }
    private val _calibration = MutableStateFlow<com.asr.live.pipeline.engine.SelectionOutcome?>(null)
    val calibration = _calibration.asStateFlow()
    private val _calibrationProgress = MutableStateFlow<EngineCalibrationProgress?>(null)
    val calibrationProgress = _calibrationProgress.asStateFlow()
    /**
     * "Optimize for this phone": Stage A/B calibration (compatibility, smoke, source-text
     * quality) for the current language pair over the curated candidates. Applies the winner
     * immediately in this build via [CalibrationStore] and [update] — no APK release needed.
     * Excludes the Hy-MT2 7B extended tier unless [includeExtendedTier] is set, matching
     * "Full calibration" in the design.
     */
    fun optimizeForThisDevice(includeExtendedTier: Boolean = false) {
        if (_busy.value || _benchmarkBusy.value || CaptionState.running.value) return
        val current = _config.value
        val candidates = EngineRegistry.forProfile(current.profile, includeExtendedTier)
        _benchmarkBusy.value = true
        _calibration.value = null
        viewModelScope.launch {
            try {
                val preparationFailures = linkedMapOf<String, String>()
                candidates.forEachIndexed { index, candidate ->
                    _calibrationProgress.value = EngineCalibrationProgress(
                        "Preparing models", index + 1, candidates.size, candidate.displayName)
                    try {
                        val ready = withContext(Dispatchers.IO) {
                            when (candidate.family) {
                                EngineFamily.ML_KIT -> ModelRepository.prepareTranslation(
                                    getApplication(), candidate.engineId, current.profile.source, current.profile.target)
                                EngineFamily.OPUS -> {
                                    val bundleId = checkNotNull(current.profile.fastBundle)
                                    TranslationModels.present(getApplication(), bundleId) ||
                                        ModelRepository.downloadBundle(getApplication(), bundleId)
                                }
                                EngineFamily.HY_MT2 -> {
                                    val bundleId = checkNotNull(candidate.quality?.bundleId)
                                    TranslationModels.present(getApplication(), bundleId) ||
                                        ModelRepository.downloadBundle(getApplication(), bundleId)
                                }
                                else -> false
                            }
                        }
                        if (!ready) preparationFailures[candidate.engineId] =
                            (ModelRepository.state.value as? ModelRepository.DownloadState.Failed)?.message
                                ?: "model preparation did not complete"
                    } catch (cancelled: CancellationException) { throw cancelled }
                    catch (t: Throwable) { preparationFailures[candidate.engineId] = t.message ?: t.javaClass.simpleName }
                }
                _calibrationProgress.value = EngineCalibrationProgress(
                    "Running smoke and translation quality checks", candidates.size, candidates.size)
                val outcome = withContext(Dispatchers.IO) {
                    EngineCalibrator.calibrate(getApplication(), current.profile, current.gpuTranslation,
                        includeExtendedTier, calibrationStore, preparationFailures)
                }
                _calibration.value = outcome
                outcome.winner?.let { winner ->
                    update(current.copy(primaryEngineId = winner.engineId,
                        quality = winner.quality ?: current.quality,
                        gpuTranslation = outcome.selectedBackend == Backend.OPENCL),
                        isAutomaticCalibration = true)
                }
            } catch (cancelled: CancellationException) { throw cancelled }
            catch (t: Throwable) { CaptionState.setError("Calibration failed: ${t.message ?: t.javaClass.simpleName}") }
            finally {
                _calibrationProgress.value = null
                _benchmarkBusy.value = false
                refreshPresence()
            }
        }
    }
    fun benchmarkTranslation(source: String, humanReference: String, stage: TranslationBenchmarkStage) {
        if (_busy.value || _benchmarkBusy.value || CaptionState.running.value || source.isBlank()) return
        val current = _config.value
        val sources = source.lineSequence().map(String::trim).filter(String::isNotEmpty).take(12).toList()
        if (sources.isEmpty()) return
        val targetReferences = humanReference.lineSequence().map(String::trim).filter(String::isNotEmpty).take(12).toList()
        if (targetReferences.isNotEmpty() && targetReferences.size != sources.size) {
            _benchmark.value = listOf(TranslationBenchmark(stage.name, 0, "",
                "Supply one human English reference line per Dutch source line, or leave all references blank"))
            return
        }
        _benchmarkBusy.value = true
        _benchmark.value = emptyList()
        viewModelScope.launch {
            val results = mutableListOf<TranslationBenchmark>()
            try {
                val planned = TranslationBenchmarkPlan.runs(stage, current.gpuTranslation,
                    com.asr.live.BuildConfig.OPENCL_ENABLED, current.translationThreads,
                    current.translationBatch, current.translationUbatch)
                val runs = if (translationBenchmarkRuns.getAndIncrement() % 2 == 0) planned else planned.reversed()
                for (run in runs) {
                    val quality = run.quality
                    val id = quality.bundleId!!
                    val requestLabel = "${stage.name} · ${quality.label} · ${run.threads}t · ${run.batch}/${run.ubatch}"
                    val result = withContext(Dispatchers.IO) {
                        if (!TranslationModels.present(getApplication(), id)) {
                            TranslationBenchmark(requestLabel, 0, "", "Model not downloaded", quality = quality)
                        } else if (!MemoryUsage.canLoad(getApplication(), TranslationModels.bundle(getApplication(), id).size)) {
                            TranslationBenchmark(requestLabel, 0, "", "Insufficient free RAM with 2 GiB system reserve",
                                requestedOpenCl = run.openCl, batch = run.batch, ubatch = run.ubatch,
                                threads = run.threads, quality = quality)
                        } else runCatching {
                            val dir = TranslationModels.verify(getApplication(), id)
                            val loadStarted = android.os.SystemClock.elapsedRealtime()
                            com.asr.live.i18n.NativeTranslator(java.io.File(dir, "model.gguf"), run.threads,
                                false, current.profile, current.glossary, run.openCl,
                                getApplication<Application>().getDir("llama-opencl-cache", Context.MODE_PRIVATE),
                                run.batch, run.ubatch).use { engine ->
                                val loadMs = android.os.SystemClock.elapsedRealtime() - loadStarted
                                val sessionId = android.os.SystemClock.elapsedRealtimeNanos()
                                var requestId = 1L
                                fun translate(text: String, segment: Long): String = engine.translate(
                                    text, requestId++, sessionId, segment, 0) { }

                                val coldStarted = android.os.SystemClock.elapsedRealtime()
                                val coldOutput = translate(sources.first(), 1)
                                val coldRequestMs = android.os.SystemClock.elapsedRealtime() - coldStarted
                                val coldTimings = engine.timings
                                engine.warmUp()
                                val durations = mutableListOf<Long>()
                                var prefill = 0L; var decode = 0L
                                var inputTokens = 0L; var outputTokens = 0L; var cacheTokens = 0L
                                var firstVisible = 0L; var complete = 0L; var prefillSync = 0L
                                var prefillDecode = 0L; var decodeCompute = 0L
                                var sampling = 0L; var decodeSync = 0L
                                var offloadedLayers = 0L; var totalLayers = 0L
                                var deviceBytes = 0L; var fallbacks = 0L
                                var output = listOf(coldOutput)
                                repeat(5) { repeatIndex ->
                                    output = sources.mapIndexed { sourceIndex, sentence ->
                                        val start = android.os.SystemClock.elapsedRealtime()
                                        val translated = translate(sentence, 2L + repeatIndex * sources.size + sourceIndex)
                                        durations += android.os.SystemClock.elapsedRealtime() - start
                                        prefill += engine.timings.prefillMs; decode += engine.timings.decodeMs
                                        inputTokens += engine.timings.inputTokens; outputTokens += engine.timings.outputTokens
                                        cacheTokens += engine.timings.cacheReusedTokens
                                        firstVisible += engine.timings.firstVisibleMs; complete += engine.timings.completeMs
                                        prefillDecode += engine.timings.prefillDecodeUs; prefillSync += engine.timings.prefillSyncUs
                                        decodeCompute += engine.timings.decodeComputeUs; sampling += engine.timings.samplingUs
                                        decodeSync += engine.timings.decodeSyncUs
                                        offloadedLayers = engine.timings.offloadedLayers
                                        totalLayers = engine.timings.totalLayers
                                        deviceBytes = engine.timings.deviceBytesAllocated
                                        fallbacks = engine.timings.fallbackCount
                                        translated
                                    }
                                }
                                val memory = MemoryUsage.sample(getApplication())
                                val sorted = durations.sorted()
                                val p50 = sorted[((sorted.size * .50).toInt().coerceAtLeast(1) - 1).coerceAtMost(sorted.lastIndex)]
                                val p95 = sorted[((sorted.size * .95).toInt().coerceAtLeast(1) - 1).coerceAtMost(sorted.lastIndex)]
                                val outputText = output.joinToString("\n")
                                val referenceWer = if (targetReferences.isEmpty()) null else {
                                    val expected = com.asr.live.asr.WordErrorScorer.normalize(targetReferences.joinToString(" "))
                                    val actual = com.asr.live.asr.WordErrorScorer.normalize(output.joinToString(" "))
                                    com.asr.live.asr.WordErrorScorer.count(expected, actual).wer
                                }
                                TranslationBenchmark(requestLabel, durations.average().toLong(), outputText,
                                    appProcessGroupPssMiB = memory.appPssKb / 1024, availableMiB = memory.availableKb / 1024,
                                    backend = engine.backend, requestedOpenCl = run.openCl, p50Ms = p50, p95Ms = p95,
                                    coldRequestMs = coldRequestMs, prefillMs = prefill / durations.size,
                                    decodeMs = decode / durations.size, loadMs = loadMs,
                                    batch = run.batch, ubatch = run.ubatch, threads = run.threads,
                                    inputTokens = (inputTokens + coldTimings.inputTokens) / (durations.size + 1),
                                    outputTokens = (outputTokens + coldTimings.outputTokens) / (durations.size + 1),
                                    cacheReusedTokens = (cacheTokens + coldTimings.cacheReusedTokens) / (durations.size + 1),
                                    firstVisibleMs = firstVisible / durations.size, completeMs = complete / durations.size,
                                    prefillDecodeUs = prefillDecode / durations.size, prefillSyncUs = prefillSync / durations.size,
                                    decodeComputeUs = decodeCompute / durations.size, samplingUs = sampling / durations.size,
                                    decodeSyncUs = decodeSync / durations.size, offloadedLayers = offloadedLayers,
                                    totalLayers = totalLayers, deviceBytesAllocated = deviceBytes,
                                    fallbackCount = fallbacks, quality = quality, humanReferenceWer = referenceWer,
                                    thermalStatus = memory.thermalStatus, thermalMaxC = memory.thermalMaxC)
                            }
                        }.getOrElse { TranslationBenchmark(requestLabel, 0, "", it.message ?: "Benchmark failed",
                            requestedOpenCl = run.openCl, batch = run.batch, ubatch = run.ubatch,
                            threads = run.threads, quality = quality) }
                    }
                    results += result
                    _benchmark.value = results.toList()
                }
            } catch (t: IllegalArgumentException) {
                _benchmark.value = listOf(TranslationBenchmark(stage.name, 0, "", t.message ?: "Invalid benchmark stage"))
            } finally { _benchmarkBusy.value = false }
        }
    }

    init { refreshPresence() }
    fun model() = ModelCatalog.byId(_config.value.modelId) ?: ModelCatalog.DEFAULT
    fun models() = ModelCatalog.ALL.filter { it.supports(_config.value.profile.source) }
    fun selectMode(mode: PerformanceMode) = update(_config.value.withMode(mode))
    fun update(config: SessionConfig, userChangedQnn: Boolean = false, userChangedCorrection: Boolean = false,
               userChangedOpenCl: Boolean = false, isAutomaticCalibration: Boolean = false) {
        if (shouldRejectConfigUpdate(_busy.value, _benchmarkBusy.value, CaptionState.running.value,
                isAutomaticCalibration)) return
        val info = ModelCatalog.byId(config.modelId)
        var adjusted = if (config.profile != _config.value.profile) config.copy(modelId = ModelCatalog.defaultFor(config.profile.source).id, correction = config.correction && config.profile.correctionSupported)
            else if (info?.supports(config.profile.source) == true) config else config.copy(modelId = ModelCatalog.defaultFor(config.profile.source).id)
        if (adjusted.qnn && ModelCatalog.byId(adjusted.modelId)?.kind == EngineKind.NEMOTRON)
            adjusted = adjusted.copy(modelId = ModelCatalog.NEMOTRON.id)
        if (adjusted.modelId != _config.value.modelId && adjusted.modelId != ModelCatalog.NEMOTRON.id &&
            ModelCatalog.byId(adjusted.modelId)?.kind == EngineKind.NEMOTRON &&
            _managed.value.none { it.id == adjusted.modelId && it.selectable }) return
        val previous = _config.value
        val translationChoiceChanged = previous.profile != adjusted.profile || previous.quality != adjusted.quality ||
            previous.performanceMode != adjusted.performanceMode || previous.gpuTranslation != adjusted.gpuTranslation
        if (!isAutomaticCalibration && translationChoiceChanged)
            adjusted = adjusted.copy(primaryEngineId = null)
        _config.value = adjusted
        // A calibration result is not a manual choice: it must not freeze out future
        // recalibration the way a real user override intentionally does.
        if (previous.quality != adjusted.quality && !isAutomaticCalibration)
            prefs.edit().putBoolean("quality.userOverride", true).apply()
        if (isAutomaticCalibration) prefs.edit()
            .putBoolean("quality.userOverride", false).putBoolean("opencl.userOverride", false).apply()
        prefs.edit().putString("profile", adjusted.profile.name).putString("model", adjusted.modelId)
            .putInt("threads", adjusted.threads).putInt("correctionThreads", adjusted.correctionThreads).putString("quality", adjusted.quality.name)
            .putString("performanceMode", adjusted.performanceMode.name)
            .putString("glossary", adjusted.glossary).putBoolean("correction", adjusted.correction)
            .putBoolean("qnn", adjusted.qnn).putBoolean("gpuTranslation", adjusted.gpuTranslation).apply()
        if (userChangedQnn) prefs.edit().putBoolean("qnn.userOverride", true).apply()
        if (userChangedCorrection) prefs.edit().putBoolean("correction.userOverride", true).apply()
        if (userChangedOpenCl) prefs.edit().putBoolean("opencl.userOverride", true).apply()
        prefs.edit().putInt("translationThreads", adjusted.translationThreads)
            .putInt("translationBatch", adjusted.translationBatch).putInt("translationUbatch", adjusted.translationUbatch).apply()
        if (previous.profile != adjusted.profile || previous.modelId != adjusted.modelId || previous.quality != adjusted.quality ||
            previous.primaryEngineId != adjusted.primaryEngineId ||
            previous.performanceMode != adjusted.performanceMode || previous.correction != adjusted.correction || previous.qnn != adjusted.qnn) refreshPresence()
    }

    private fun selectedEngine(cfg: SessionConfig): EngineCandidate? = cfg.primaryEngineId?.let { id ->
        EngineRegistry.all.firstOrNull { it.engineId == id && it.supports(cfg.profile) }
    }

    private fun primaryBundleId(cfg: SessionConfig): String? = when (selectedEngine(cfg)?.family) {
        EngineFamily.ML_KIT -> null
        EngineFamily.OPUS -> cfg.profile.fastBundle
        EngineFamily.HY_MT2 -> selectedEngine(cfg)?.quality?.bundleId
        else -> if (cfg.quality == TranslationQuality.ML_KIT) null else cfg.quality.bundleId
    }

    private suspend fun translationAssetsPresent(cfg: SessionConfig): Boolean {
        val primary = selectedEngine(cfg)
        val primaryReady = when (primary?.family) {
            EngineFamily.ML_KIT -> runCatching {
                ModelRepository.translationPresent(cfg.profile.source, cfg.profile.target)
            }.getOrDefault(false)
            EngineFamily.OPUS -> cfg.profile.fastBundle?.let { TranslationModels.present(getApplication(), it) } ?: false
            EngineFamily.HY_MT2 -> primary.quality?.bundleId?.let { TranslationModels.present(getApplication(), it) } ?: false
            else -> if (cfg.quality == TranslationQuality.ML_KIT)
                runCatching { ModelRepository.translationPresent(cfg.profile.source, cfg.profile.target) }.getOrDefault(false)
            else cfg.quality.bundleId?.let { TranslationModels.present(getApplication(), it) } ?: false
        }
        val opusReady = !cfg.opusBenchmarkEnabled ||
            cfg.profile.fastBundle?.let { TranslationModels.present(getApplication(), it) } == true
        return primaryReady && opusReady
    }

    fun refreshPresence() {
        val cfg = _config.value
        _ready.value = false
        viewModelScope.launch {
            refreshManaged()
            val ready = withContext(Dispatchers.IO) {
                val info = ModelCatalog.byId(cfg.modelId) ?: ModelCatalog.DEFAULT
                ModelStore.isPresent(getApplication(), info) && (info.kind != EngineKind.NEMOTRON || info == ModelCatalog.NEMOTRON || loaded(info)) && (!info.requiresVad || TranslationModels.present(getApplication(), "silero-vad")) && (!cfg.qnn || info.kind != EngineKind.NEMOTRON || ModelStore.isPresent(getApplication(), ModelCatalog.NEMOTRON_QNN)) && (!cfg.correction || !cfg.profile.correctionSupported || ModelStore.isPresent(getApplication(), ModelCatalog.PARAKEET)) && translationAssetsPresent(cfg)
            }
            if (_config.value == cfg) _ready.value = ready
        }
    }
    fun downloadRequired() {
        if (_busy.value || _benchmarkBusy.value || CaptionState.running.value) return
        val cfg = _config.value
        _busy.value = true
        viewModelScope.launch {
            try {
                val info = ModelCatalog.byId(cfg.modelId) ?: ModelCatalog.DEFAULT
                if (info.requiresVad && !ModelRepository.downloadBundle(getApplication(), "silero-vad")) return@launch
                if (!ModelRepository.download(getApplication(), info)) return@launch
                if (info.kind == EngineKind.NEMOTRON) validateChunk(info)
                if (cfg.qnn && info.kind == EngineKind.NEMOTRON && !ModelRepository.download(getApplication(), ModelCatalog.NEMOTRON_QNN)) return@launch
                if (cfg.correction && cfg.profile.correctionSupported && !ModelRepository.download(getApplication(), ModelCatalog.PARAKEET)) return@launch
                val primary = selectedEngine(cfg)
                if (primary?.family == EngineFamily.ML_KIT || (primary == null && cfg.quality == TranslationQuality.ML_KIT)) {
                    if (!ModelRepository.translationPresent(cfg.profile.source, cfg.profile.target) &&
                        !ModelRepository.prepareTranslation(getApplication(), "ml-kit-${cfg.profile.name.lowercase()}",
                            cfg.profile.source, cfg.profile.target)) return@launch
                } else {
                    val bundleId = primaryBundleId(cfg) ?: error("Selected translation engine has no model bundle")
                    if (!ModelRepository.downloadBundle(getApplication(), bundleId)) return@launch
                }
                if (cfg.opusBenchmarkEnabled) {
                    val opusBundle = cfg.profile.fastBundle
                    if (opusBundle != null && !ModelRepository.downloadBundle(getApplication(), opusBundle)) return@launch
                }
            } catch (e: Exception) {
                if (e is CancellationException) throw e
                CaptionState.setError("Model preparation failed: ${e.message}")
            } finally { _busy.value = false; refreshPresence() }
        }
    }
    private fun stamp(info: ModelInfo) = "${com.asr.live.BuildConfig.VERSION_CODE}:${info.sha256}"
    private fun loaded(info: ModelInfo) = prefs.getString("loaded.${info.id}", null) == stamp(info)
    private val asrAssets get() = (ModelCatalog.ALL + ModelCatalog.NEMOTRON_PROFILES + ModelCatalog.PARAKEET + ModelCatalog.NEMOTRON_QNN).distinctBy { it.id }
    private val bundleIds get() = (TranslationQuality.entries.mapNotNull { it.bundleId } + Profile.entries.mapNotNull { it.fastBundle } + "silero-vad").distinct()
    private suspend fun refreshManaged() {
        _managed.value = withContext(Dispatchers.IO) {
            asrAssets.map { info ->
                val installed = ModelStore.isPresent(getApplication(), info)
                ManagedModel(info.id, if (info in ModelCatalog.NEMOTRON_PROFILES) "Nemotron CPU · ${ModelCatalog.chunkLabel(info.chunkMs)}" else info.displayName,
                    info.archiveBytes, installed, installed && loaded(info), ModelStore.dir(getApplication(), info.id).exists())
            } + bundleIds.map { id -> val b = TranslationModels.bundle(getApplication(), id)
                ManagedModel(id, b.label, b.size, TranslationModels.present(getApplication(), id), stored = ModelStore.dir(getApplication(), id).exists()) }
        }
    }
    private suspend fun validateChunk(info: ModelInfo) = withContext(Dispatchers.IO) {
        ModelStore.verify(getApplication(), info)
        val engine = com.asr.live.asr.EngineFactory.create(getApplication(), info, _config.value.profile.source,
            "transcribe", {}, {}, _config.value.threads)
        try {
            // Exercise at least one complete encoder chunk; no microphone or persisted audio.
            repeat(20) { engine.accept(FloatArray(1600)) }
            engine.finish()
            prefs.edit().putString("loaded.${info.id}", stamp(info)).apply()
        } finally { engine.release() }
    }
    fun downloadModel(id: String) {
        if (_busy.value || _benchmarkBusy.value || CaptionState.running.value || _managed.value.none { it.id == id }) return
        _busy.value = true
        viewModelScope.launch {
            try {
                val info = asrAssets.firstOrNull { it.id == id }
                if (info != null) {
                    if (ModelRepository.download(getApplication(), info) && info in ModelCatalog.NEMOTRON_PROFILES) validateChunk(info)
                } else ModelRepository.downloadBundle(getApplication(), id)
            } catch (e: Exception) {
                if (e is CancellationException) throw e
                CaptionState.setError("Model load test failed: ${e.message}")
            } finally { _busy.value = false; refreshPresence() }
        }
    }
    fun removeModel(id: String) {
        if (_busy.value || _benchmarkBusy.value || CaptionState.running.value || _managed.value.none { it.id == id }) return
        _busy.value = true
        viewModelScope.launch {
            try {
                ModelRepository.remove(getApplication(), id)
                prefs.edit().remove("loaded.$id").apply()
            } catch (e: Exception) {
                if (e is CancellationException) throw e
                CaptionState.setError("Could not remove model: ${e.message}")
            } finally { _busy.value = false; refreshPresence() }
        }
    }
    fun downloadMegabytes(): Long {
        val cfg = _config.value
        var bytes = model().archiveBytes
        if (model().requiresVad) bytes += TranslationModels.bundle(getApplication(), "silero-vad").size
        if (cfg.qnn && model().kind == EngineKind.NEMOTRON) bytes += ModelCatalog.NEMOTRON_QNN.archiveBytes
        if (cfg.correction && cfg.profile.correctionSupported) bytes += ModelCatalog.PARAKEET.archiveBytes
        primaryBundleId(cfg)?.let { bytes += TranslationModels.bundle(getApplication(), it).size }
        if (cfg.quality == TranslationQuality.ML_KIT || selectedEngine(cfg)?.family == EngineFamily.ML_KIT)
            bytes += 30_000_000L
        if (cfg.opusBenchmarkEnabled) cfg.profile.fastBundle?.let { bytes += TranslationModels.bundle(getApplication(), it).size }
        return bytes / 1_000_000
    }
    fun toggle() {
        if (CaptionState.running.value) CaptionService.stop(getApplication())
        else if (_ready.value && !_busy.value && !_benchmarkBusy.value) {
            val cfg = _config.value
            val check = runCatching { com.asr.live.i18n.TranslationPrompt.build(cfg.profile, "", cfg.glossary) }
            if (check.isFailure) CaptionState.setError(check.exceptionOrNull()?.message)
            else CaptionService.start(getApplication(), cfg)
        }
    }
    fun clear() = CaptionState.clear()
    fun acknowledgeDisplayed(key: SegmentKey) {
        CaptionState.acknowledgeDisplayed(key.session, key, System.nanoTime())
    }
    fun rateComparison(key: SegmentKey, vote: TranslationVote) { CaptionState.vote(key, vote) }
    fun dismissError() = CaptionState.setError(null)
}
