package com.asr.live.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.ui.Modifier
import androidx.compose.ui.Alignment
import androidx.compose.ui.unit.dp
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.text.font.FontWeight
import com.asr.live.model.*
import com.asr.live.pipeline.*
import com.asr.live.i18n.TranslationBenchmarkStage
import com.asr.live.asr.AsrWavBenchmarkStage
import com.asr.live.service.*
import com.asr.live.pipeline.engine.EngineRegistry
import com.asr.live.pipeline.engine.EngineFamily

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CaptionScreen(vm: CaptionViewModel, hasAudioPermission: Boolean, onRequestPermission: () -> Unit,
                  hasOverlayPermission: Boolean, onRequestOverlayPermission: () -> Unit) {
    val config by vm.config.collectAsState()
    val ready by vm.ready.collectAsState()
    val busy by vm.busy.collectAsState()
    val lines by vm.lines.collectAsState()
    val lifecycle by vm.lifecycle.collectAsState()
    val error by vm.error.collectAsState()
    val download by vm.download.collectAsState()
    val metrics by vm.metrics.collectAsState()
    val overlay by vm.overlay.collectAsState()
    val managed by vm.managed.collectAsState()
    val benchmark by vm.benchmark.collectAsState()
    val comparisons by vm.comparisons.collectAsState()
    val benchmarkBusy by vm.benchmarkBusy.collectAsState()
    val asrBenchmark by vm.asrBenchmark.collectAsState()
    val asrBenchmarkError by vm.asrBenchmarkError.collectAsState()
    val calibrationProgress by vm.calibrationProgress.collectAsState()
    var benchmarkText by remember { mutableStateOf("Goedemorgen, dit is een test van de live vertaling.") }
    var benchmarkReference by remember { mutableStateOf("") }
    var asrReference by remember { mutableStateOf("") }
    var asrStage by remember { mutableStateOf(AsrWavBenchmarkStage.CPU_VS_QNN) }
    var asrStageMenu by remember { mutableStateOf(false) }
    val wavPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) vm.benchmarkAsrWav(uri, asrStage, asrReference)
    }
    var settings by remember { mutableStateOf(false) }
    var profiles by remember { mutableStateOf(false) }
    var diagnostics by remember { mutableStateOf(false) }
    var modelManager by remember { mutableStateOf(false) }
    val stopped = lifecycle == ListeningState.STOPPED
    val info = ModelCatalog.byId(config.modelId) ?: ModelCatalog.DEFAULT
    Scaffold(topBar = { TopAppBar(title = { Text("LiveTranslate") }, actions = {
        TextButton(onClick = { settings = true }) { Text("Settings") }
    }) }, bottomBar = {
        Surface(tonalElevation = 3.dp) {
            Row(Modifier.fillMaxWidth().navigationBarsPadding().padding(12.dp), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                if (!hasAudioPermission) Button(onClick = onRequestPermission, modifier = Modifier.weight(1f)) { Text("Allow microphone") }
                else if (!ready && stopped) Button(onClick = vm::downloadRequired, enabled = !busy, modifier = Modifier.weight(1f)) { Text(if (busy) "Downloading…" else "Download required models") }
                else Button(onClick = vm::toggle, enabled = lifecycle != ListeningState.STOPPING && (!stopped || (!busy && !benchmarkBusy)), modifier = Modifier.weight(1f)) {
                    Text(when (lifecycle) { ListeningState.STOPPED -> "Listen"; ListeningState.STOPPING -> "Stopping…"; else -> "Stop" })
                }
                TextButton(onClick = vm::clear) { Text("Clear") }
            }
        }
    }) { padding ->
        Column(Modifier.fillMaxSize().padding(padding).padding(horizontal = 16.dp)) {
            Box {
                TextButton(onClick = { profiles = true }, enabled = stopped && !busy) { Text(config.profile.label, style = MaterialTheme.typography.titleMedium) }
                DropdownMenu(expanded = profiles, onDismissRequest = { profiles = false }) {
                    Profile.entries.forEach { profile -> DropdownMenuItem(text = { Text(profile.label) }, onClick = {
                        vm.update(config.copy(profile = profile)); profiles = false
                    }) }
                }
            }
            Text("Performance: ${config.performanceMode.label}", style = MaterialTheme.typography.titleSmall)
            Text("ASR: ${info.shortName} · ${info.chunkMs?.let { "$it ms" } ?: "VAD phrases"} · ${if (!stopped) metrics.backend else if (config.qnn && info.kind == EngineKind.NEMOTRON) "QNN requested · experimental / CPU fallback" else "CPU"} · ${config.threads} threads", style = MaterialTheme.typography.bodySmall)
            val selectedEngine = config.primaryEngineId?.let { id -> EngineRegistry.all.firstOrNull { it.engineId == id } }
            Text("Translation: ${selectedEngine?.displayName ?: config.quality.label}", style = MaterialTheme.typography.bodySmall)
            val liveTranslator = when {
                selectedEngine?.family == EngineFamily.OPUS -> "OPUS CPU · automatic selection"
                selectedEngine?.family == EngineFamily.ML_KIT -> "ML Kit on-device · automatic selection"
                config.opusBenchmarkEnabled -> "OPUS CPU + ${config.quality.label} A/B"
                config.quality == TranslationQuality.ML_KIT -> "ML Kit on-device"
                else -> "${config.quality.label} · ${if (!stopped) metrics.translationBackend else if (config.gpuTranslation && com.asr.live.BuildConfig.OPENCL_ENABLED) "OpenCL requested" else "CPU"}"
            }
            Text("Live provisional: $liveTranslator", style = MaterialTheme.typography.bodySmall)
            Text(if (ready) "Offline ready" else "Required pinned models: ~${vm.downloadMegabytes()} MB${if (config.quality == TranslationQuality.ML_KIT) " + ML Kit language pack" else ""}", style = MaterialTheme.typography.labelMedium)
            if (download is ModelRepository.DownloadState.Running) {
                val d = download as ModelRepository.DownloadState.Running
                LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                Text("${d.phase.name.lowercase()} · ${d.pct}%${if (d.total > 0) " · ${d.bytes / 1_000_000}/${d.total / 1_000_000} MB" else ""}")
            }
            if (download is ModelRepository.DownloadState.Failed) Text((download as ModelRepository.DownloadState.Failed).message, color = MaterialTheme.colorScheme.error)
            error?.let { Text(it, color = MaterialTheme.colorScheme.error) }
            TextButton(onClick = { diagnostics = !diagnostics }) { Text(if (diagnostics) "Hide performance" else "Performance") }
            if (diagnostics) PerformancePanel(metrics)
            val list = rememberLazyListState()
            LaunchedEffect(lines.lastOrNull()?.key) { if (lines.isNotEmpty()) list.scrollToItem(lines.lastIndex) }
            LazyColumn(Modifier.weight(1f).fillMaxWidth(), state = list, verticalArrangement = Arrangement.spacedBy(12.dp), contentPadding = PaddingValues(bottom = 12.dp)) {
                if (lines.isEmpty()) item { Text(if (stopped) "Download models, then tap Listen." else "${lifecycle.name.lowercase()}…", Modifier.padding(vertical = 24.dp)) }
                items(lines, key = { it.key.id }) { caption ->
                    if (caption.translation.isNotBlank()) LaunchedEffect(caption.translationPortion,
                        caption.translationRequestId, caption.translation) {
                        vm.acknowledgeDisplayed(caption.key)
                    }
                    Card(colors = CardDefaults.cardColors(containerColor = if (caption.stage == CaptionStage.PROVISIONAL)
                        MaterialTheme.colorScheme.secondaryContainer else MaterialTheme.colorScheme.surfaceVariant)) {
                        Column(Modifier.fillMaxWidth().padding(14.dp)) {
                            Text(caption.translation.ifBlank { "…" }, style = MaterialTheme.typography.headlineSmall)
                            val suffixColor = MaterialTheme.colorScheme.onSurfaceVariant
                            Text(buildAnnotatedString {
                                val covered = caption.translatedSource.takeIf {
                                    it.isNotBlank() && caption.source.startsWith(it)
                                } ?: caption.stableSource.takeIf {
                                    it.isNotBlank() && caption.source.startsWith(it)
                                }
                                if (covered != null) {
                                    withStyle(SpanStyle(fontWeight = FontWeight.SemiBold)) { append(covered) }
                                    withStyle(SpanStyle(color = suffixColor)) {
                                        append(caption.source.removePrefix(covered))
                                    }
                                } else append(caption.source)
                            }, style = MaterialTheme.typography.bodyLarge)
                            Text(caption.detail, style = MaterialTheme.typography.labelSmall)
                        }
                    }
                }
            }
        }
    }
    if (settings) AlertDialog(onDismissRequest = { settings = false }, confirmButton = { TextButton(onClick = { settings = false }) { Text("Done") } }, title = { Text("Advanced settings") }, text = {
        Column(Modifier.verticalScroll(rememberScrollState())) {
            Text("Floating captions", style = MaterialTheme.typography.titleMedium)
            if (!hasOverlayPermission) {
                Text("Optional: allow display over other apps. In-app captions work without this permission.")
                TextButton(onClick = onRequestOverlayPermission) { Text("Allow floating captions") }
            }
            Row(verticalAlignment = Alignment.CenterVertically) {
                Switch(checked = overlay.enabled, onCheckedChange = { vm.updateOverlay(overlay.copy(enabled = it)) }, enabled = hasOverlayPermission)
                Text("Show while listening")
            }
            Text("Background opacity: ${(overlay.opacity * 100).toInt()}%")
            Slider(value = overlay.opacity, onValueChange = { vm.updateOverlay(overlay.copy(opacity = it)) }, valueRange = 0f..0.9f)
            Text("Font size: ${overlay.fontSp} sp")
            Slider(value = overlay.fontSp.toFloat(), onValueChange = { vm.updateOverlay(overlay.copy(fontSp = it.toInt())) }, valueRange = 16f..40f, steps = 23)
            Text("Translation lines: ${overlay.lines}")
            Slider(value = overlay.lines.toFloat(), onValueChange = { vm.updateOverlay(overlay.copy(lines = it.toInt())) }, valueRange = 1f..4f, steps = 2)
            Row(verticalAlignment = Alignment.CenterVertically) {
                Switch(checked = overlay.source, onCheckedChange = { vm.updateOverlay(overlay.copy(source = it)) })
                Text("Show source transcript")
            }
            Row(verticalAlignment = Alignment.CenterVertically) {
                Switch(checked = overlay.touchThrough, onCheckedChange = { vm.updateOverlay(overlay.copy(touchThrough = it)) })
                Text("Touch-through")
            }
            Text("Turn touch-through off to drag. Android limits window opacity in touch-through mode. Use the listening notification to show or hide captions.", style = MaterialTheme.typography.bodySmall)
            HorizontalDivider(Modifier.padding(vertical = 12.dp))
            TextButton(onClick = vm::downloadRequired, enabled = stopped && !busy) { Text("Verify / repair required models") }
            TextButton(onClick = { modelManager = true }) { Text("Download / remove models") }
            Text("Performance profile")
            PerformanceMode.entries.forEach { mode ->
                TextButton(onClick = { vm.selectMode(mode) }, enabled = stopped && !busy) {
                    Text((if (mode == config.performanceMode) "✓ " else "") + mode.label)
                }
            }
            Text("Fast uses Hy-MT2 1.8B Q4_K_M; for Dutch → English it also runs the disclosed OPUS/Hy A/B and OPUS supplies provisional captions. Balanced uses Hy-MT2 1.8B Q8_0. Max uses one Hy-MT2 7B Q4_K_M engine; Parakeet correction is separate and optional. No profile silently benchmarks the 7B model. QNN and OpenCL remain experimental until measured on this phone.", style = MaterialTheme.typography.bodySmall)
            HorizontalDivider(Modifier.padding(vertical = 12.dp))
            Text("Automatic — optimized for this device", style = MaterialTheme.typography.titleSmall)
            val calibration by vm.calibration.collectAsState()
            TextButton(onClick = { vm.optimizeForThisDevice() }, enabled = stopped && !busy && !benchmarkBusy) {
                Text(if (benchmarkBusy) "Calibrating…" else "Optimize for this phone")
            }
            TextButton(onClick = { vm.optimizeForThisDevice(includeExtendedTier = true) }, enabled = stopped && !busy && !benchmarkBusy) {
                Text("Full calibration · include Hy-MT2 7B")
            }
            calibrationProgress?.let { progress ->
                Text("${progress.phase}${if (progress.total > 0) " · ${progress.index}/${progress.total}" else ""}" +
                    (progress.candidate?.let { " · $it" } ?: ""), style = MaterialTheme.typography.bodySmall)
            }
            Text("Runs local compatibility, smoke and source-text quality screening over the " +
                "curated candidates for ${config.profile.label}, downloads missing candidate models in sequence, " +
                "and applies the best validated result. Nothing leaves this device.",
                style = MaterialTheme.typography.bodySmall)
            calibration?.let { outcome ->
                outcome.reports.forEach { report ->
                    Text("${report.candidate.displayName}: ${report.status} — ${report.detail}",
                        style = MaterialTheme.typography.bodySmall)
                }
                Text(outcome.reason, style = MaterialTheme.typography.bodySmall)
            }
            if (config.opusBenchmarkEnabled) {
                Text("Live OPUS vs ${config.quality.label} A/B from the same microphone audio", style = MaterialTheme.typography.titleMedium)
                Text("Both translators receive the identical stable Nemotron transcript prefix. Latency is measured per output; compare translation quality side by side and record a preference. Votes are human judgments, not reference-scored accuracy.", style = MaterialTheme.typography.bodySmall)
                comparisons.takeLast(5).forEach { row ->
                    HorizontalDivider(Modifier.padding(vertical = 6.dp))
                    Text("${config.profile.label} source: ${row.source}", style = MaterialTheme.typography.bodySmall)
                    Text("OPUS · ${row.opusMs?.let { "$it ms" } ?: "waiting"}: ${row.opusText ?: "—"}", style = MaterialTheme.typography.bodySmall)
                    Text("${config.quality.label} · ${row.hyMs?.let { "$it ms" } ?: "waiting"}: ${row.hyText ?: "—"}", style = MaterialTheme.typography.bodySmall)
                    if (row.opusText != null && row.hyText != null) {
                        Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                            listOf(TranslationVote.OPUS, TranslationVote.HY_MT2, TranslationVote.TIE).forEach { vote ->
                                TextButton(onClick = { vm.rateComparison(row.key, vote) }) {
                                    Text((if (row.vote == vote) "✓ " else "") + when (vote) {
                                        TranslationVote.OPUS -> "OPUS better"
                                        TranslationVote.HY_MT2 -> "Hy better"
                                        TranslationVote.TIE -> "Tie"
                                    })
                                }
                            }
                        }
                    }
                }
            }
            Text("Recognition model")
            vm.models().forEach { model -> TextButton(onClick = { vm.update(config.copy(modelId = model.id)) }, enabled = stopped && !busy) { Text((if (model.kind == info.kind) "✓ " else "") + model.displayName) } }
            Text("ASR backend")
            Row(verticalAlignment = Alignment.CenterVertically) {
                Switch(checked = config.qnn, onCheckedChange = { vm.update(config.copy(qnn = it), userChangedQnn = true) },
                    enabled = stopped && !busy && info.kind == EngineKind.NEMOTRON &&
                        android.os.Build.VERSION.SDK_INT >= 31 && BackendPolicy.qnnEligible(android.os.Build.SOC_MODEL, com.asr.live.BuildConfig.QNN_ENABLED))
                Text("QNN/NPU · experimental SM8750")
            }
            Text("CPU fallback is always downloaded. QNN has not been tested on this Honor phone.", style = MaterialTheme.typography.bodySmall)
            Row(verticalAlignment = Alignment.CenterVertically) {
                Switch(checked = config.gpuTranslation, onCheckedChange = { vm.update(config.copy(gpuTranslation = it), userChangedOpenCl = true) },
                    enabled = stopped && !busy && com.asr.live.BuildConfig.OPENCL_ENABLED)
                Text("Prefer Adreno 830 OpenCL translation")
            }
            Text(if (com.asr.live.BuildConfig.OPENCL_ENABLED)
                "The app probes for an Adreno 830 through the Qualcomm vendor ICD. The visible backend changes only after the Hy-MT2 model loads; unavailable or failed OpenCL returns to CPU."
                else "This build has no OpenCL backend. Hy-MT2 runs on CPU.", style = MaterialTheme.typography.bodySmall)
            if (info.kind == EngineKind.NEMOTRON) {
                Text("Nemotron model chunk")
                ModelCatalog.chunkProfiles(config.qnn).forEach { model ->
                    val installed = managed.firstOrNull { it.id == model.id }
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        TextButton(onClick = { vm.update(config.copy(modelId = model.id)) },
                            modifier = Modifier.weight(1f),
                            enabled = stopped && !busy && (model.id == config.modelId || installed?.selectable == true)) {
                            Text((if (model.id == config.modelId) "✓ " else "") + ModelCatalog.chunkLabel(model.chunkMs))
                        }
                        if (!config.qnn && installed?.selectable != true) TextButton(onClick = { vm.downloadModel(model.id) }, enabled = stopped && !busy) {
                            Text(if (installed?.installed == true) "Test load" else "Download")
                        }
                    }
                }
                Text(if (config.qnn) "Only the 560 ms context is integrated. Other QNN chunk profiles are unavailable until hardware validation."
                    else "Each size downloads a separate ~475 MB model and passes a CPU load/decode test before selection. 560 ms is the initial default.", style = MaterialTheme.typography.bodySmall)
            }
            Text("CPU threads: ${config.threads}")
            Slider(value = config.threads.toFloat(), onValueChange = { vm.update(config.copy(threads = it.toInt())) }, valueRange = 1f..8f, steps = 6, enabled = stopped && !busy)
            if (config.correction && config.profile.correctionSupported) {
                Text("Parakeet endpoint threads: ${config.correctionThreads} (2 / 4 / 6 / 8)")
                Slider(value = ((config.correctionThreads - 2) / 2).toFloat(),
                    onValueChange = { vm.update(config.copy(correctionThreads = 2 + it.toInt() * 2)) },
                    valueRange = 0f..3f, steps = 2, enabled = stopped && !busy)
            }
            Text("Translation quality")
            TranslationQuality.entries.forEach { quality -> TextButton(onClick = { vm.update(config.copy(quality = quality)) }, enabled = stopped && !busy) { Text((if (quality == config.quality) "✓ " else "") + quality.label) } }
            if (config.quality != TranslationQuality.ML_KIT) OutlinedTextField(value = config.glossary,
                onValueChange = { vm.update(config.copy(glossary = it.take(2000))) },
                enabled = stopped && !busy, label = { Text("Glossary: source -> target") },
                placeholder = { Text("晶圆 -> wafer\n套刻 -> overlay\n压印 -> imprint\n母模 -> master\n光刻胶 -> resist") }, minLines = 3)
            Row(verticalAlignment = Alignment.CenterVertically) {
                Switch(checked = config.correction && config.profile.correctionSupported,
                    onCheckedChange = { vm.update(config.copy(correction = it), userChangedCorrection = true) },
                    enabled = stopped && !busy && config.profile.correctionSupported && info.kind == EngineKind.NEMOTRON)
                Text("Parakeet v3 endpoint correction")
            }
            Text(if (config.profile.correctionSupported) "Optional second hypothesis; skipped under load. CPU performance is device dependent." else "Parakeet correction is unavailable for Mandarin.", style = MaterialTheme.typography.bodySmall)
            HorizontalDivider(Modifier.padding(vertical = 12.dp))
            Text("On-device WAV benchmark · injection", style = MaterialTheme.typography.titleMedium)
            Text("Choose one 16 kHz mono PCM16 WAV. A runs the identical file through Nemotron CPU and QNN; D runs it with Parakeet off and on, with Parakeet decoding on its own worker. This is an isolated injected stream, not microphone or translated-subtitle latency. The WAV SHA-256, human-reference WER, number-token errors, actual decode-call distribution, schedule lag, PCM signal statistics, process-group PSS and thermal status are logged and shown below.", style = MaterialTheme.typography.bodySmall)
            Box {
                TextButton(onClick = { asrStageMenu = true }, enabled = stopped && !busy && !benchmarkBusy) {
                    Text(if (asrStage == AsrWavBenchmarkStage.CPU_VS_QNN) "A · Nemotron CPU vs QNN" else "D · Parakeet off vs on")
                }
                DropdownMenu(expanded = asrStageMenu, onDismissRequest = { asrStageMenu = false }) {
                    DropdownMenuItem(text = { Text("A · Nemotron CPU vs QNN") }, onClick = {
                        asrStage = AsrWavBenchmarkStage.CPU_VS_QNN; asrStageMenu = false
                    })
                    DropdownMenuItem(text = { Text("D · Parakeet off vs on") }, onClick = {
                        asrStage = AsrWavBenchmarkStage.PARAKEET_OFF_VS_ON; asrStageMenu = false
                    })
                }
            }
            OutlinedTextField(asrReference, { asrReference = it.take(2000) },
                label = { Text("Human verbatim Dutch reference") }, minLines = 2,
                enabled = stopped && !busy && !benchmarkBusy)
            TextButton(onClick = { wavPicker.launch(arrayOf("audio/*")) },
                enabled = stopped && !busy && !benchmarkBusy) { Text(if (benchmarkBusy) "Running isolated WAV benchmark…" else "Choose WAV and run stage") }
            asrBenchmarkError?.let { Text(it, color = MaterialTheme.colorScheme.error) }
            asrBenchmark.forEach { result -> Text(result.reportLine(), style = MaterialTheme.typography.bodySmall) }
            if (!com.asr.live.BuildConfig.OPENCL_ENABLED)
                Text("This build omits OpenCL, so stage B cannot compare CPU with Adreno OpenCL.", style = MaterialTheme.typography.bodySmall)
            HorizontalDivider(Modifier.padding(vertical = 12.dp))
            Text("On-device translation benchmark · staged", style = MaterialTheme.typography.titleMedium)
            Text("B compares only Hy-MT2 1.8B Q4 on CPU and OpenCL with identical text and the same selected threads/batch. C compares 1.8B Q4 and 7B Q4 on the currently selected backend; review B first and select the backend in settings. Each model runs alone. Results include cold first request, warm p50/p95, prefill/decode, first visible and complete translation, token counts, actual layer placement, fallback and total app-process-group PSS (including :qnn when present). Nothing is auto-selected or saved from a benchmark. Text-output quality needs human review; this panel does not claim ASR accuracy or sustained pipeline results.", style = MaterialTheme.typography.bodySmall)
            OutlinedTextField(benchmarkText, { benchmarkText = it.take(1000) }, label = { Text("Source sentences, one per line") },
                placeholder = { Text("Use identical Dutch phrases with numbers, negations, and spontaneous wording") }, minLines = 2, enabled = stopped && !benchmarkBusy)
            OutlinedTextField(benchmarkReference, { benchmarkReference = it.take(1000) },
                label = { Text("Human English reference, one matching line per source") }, minLines = 2,
                enabled = stopped && !benchmarkBusy)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                TextButton(onClick = { vm.benchmarkTranslation(benchmarkText, benchmarkReference, TranslationBenchmarkStage.BACKEND_18B_Q4) },
                    enabled = stopped && !busy && !benchmarkBusy) { Text("B · CPU vs OpenCL · 1.8B Q4") }
                TextButton(onClick = { vm.benchmarkTranslation(benchmarkText, benchmarkReference, TranslationBenchmarkStage.MODEL_SIZE) },
                    enabled = stopped && !busy && !benchmarkBusy) { Text("C · 1.8B vs 7B") }
            }
            if (benchmarkBusy) Text("Benchmark running…")
            benchmark.forEach { result -> Text("${result.model} · ${if (result.requestedOpenCl) "OpenCL requested" else "CPU requested"}: ${result.error ?: "${result.backend} · load ${result.loadMs} ms · cold first request ${result.coldRequestMs} ms · warm avg ${result.elapsedMs} / p50 ${result.p50Ms} / p95 ${result.p95Ms} ms · prefill ${result.prefillMs} / decode ${result.decodeMs} ms · first visible ${result.firstVisibleMs} / complete ${result.completeMs} ms · human-reference WER ${result.humanReferenceWer?.let { "%.3f".format(it) } ?: "not supplied"} · tokens in/out ${result.inputTokens}/${result.outputTokens} · KV prefix ${result.cacheReusedTokens} · prefill decode/sync ${result.prefillDecodeUs}/${result.prefillSyncUs} µs · generation decode/sync/sample ${result.decodeComputeUs}/${result.decodeSyncUs}/${result.samplingUs} µs · placement ${result.offloadedLayers}/${result.totalLayers} layers (${result.deviceBytesAllocated / (1024 * 1024)} MiB net device free-memory delta), fallbacks ${result.fallbackCount} · ${result.threads} threads, batch ${result.batch}/${result.ubatch} · app process-group PSS ${result.appProcessGroupPssMiB} MiB · available ${result.availableMiB} MiB · thermal ${result.thermalStatus}/${result.thermalMaxC?.let { "%.1f°C".format(it) } ?: "unavailable"}\n${result.output}"}", style = MaterialTheme.typography.bodySmall) }
            Text("Audio stays in memory and is never uploaded or saved.", style = MaterialTheme.typography.bodySmall)
        }
    })
    if (modelManager) AlertDialog(onDismissRequest = { modelManager = false },
        confirmButton = { TextButton(onClick = { modelManager = false }) { Text("Done") } },
        title = { Text("Downloaded models") }, text = {
            Column(Modifier.verticalScroll(rememberScrollState())) {
                Text("Stop listening to change model files. Removing a required model disables Listen until downloaded again.")
                if (busy) { LinearProgressIndicator(Modifier.fillMaxWidth()); Text("Downloading / verifying / test loading…") }
                if (download is ModelRepository.DownloadState.Running) {
                    val d = download as ModelRepository.DownloadState.Running
                    Text("${d.phase.name.lowercase()} · ${d.pct}% · ${d.bytes / 1_000_000}/${d.total / 1_000_000} MB")
                }
                if (download is ModelRepository.DownloadState.Failed) Text((download as ModelRepository.DownloadState.Failed).message)
                error?.let { Text(it, color = MaterialTheme.colorScheme.error) }
                managed.forEach { model ->
                    HorizontalDivider(Modifier.padding(vertical = 8.dp))
                    Text(model.label)
                    Text("~${model.bytes / 1_000_000} MB download · ${if (model.installed) "Installed" else if (model.stored) "Incomplete / needs repair" else "Not installed"}", style = MaterialTheme.typography.bodySmall)
                    Row {
                        TextButton(onClick = { vm.downloadModel(model.id) }, enabled = stopped && !busy) { Text(if (model.installed) "Verify" else "Download") }
                        TextButton(onClick = { vm.removeModel(model.id) }, enabled = stopped && !busy && model.stored) { Text("Remove") }
                    }
                }
                Text("ML Kit language packs, when selected, are managed by Google Play services.", style = MaterialTheme.typography.bodySmall)
            }
        })
}

@Composable
private fun PerformancePanel(m: Performance) {
    val text = "${m.performanceMode} · ${m.profile}\nASR: ${m.asr.ifBlank { "Not running" }} · ${m.backend} · chunk ${m.chunk}\n" +
        "Translator: ${m.translator} · ${m.translationBackend} · ${m.translationThreads} threads\nASR last input callback ${m.asrMs} ms · RTF ${"%.2f".format(m.asrRtf)}\n" +
        "ASR compute ${m.asrComputeMs} ms total · phrase-end wait ${m.endpointWaitMs?.let { "$it ms" } ?: "—"}\n" +
        "Native decode calls ${m.decodeCalls} · mean ${"%.2f".format(m.decodeMeanMs)} ms · " +
        "p50/p95 ${m.decodeP50Ms}/${m.decodeP95Ms} ms · max ${m.decodeMaxMs} ms\n" +
        "Hy wait ${m.hyWaitMs} ms · compute ${m.hyComputeMs} ms · state publish ${m.hyDisplayMs} ms · " +
        "prefill ${m.translationPrefillMs} ms · generation ${m.translationDecodeMs} ms · " +
        "input/output ${m.translationInputTokens}/${m.translationOutputTokens} tokens · KV prefix ${m.translationCacheTokens} · " +
        "TTFT/first visible/complete ${m.translationFirstTokenMs}/${m.translationFirstVisibleMs}/${m.translationCompleteMs} ms · " +
        "prefill decode/sync ${m.translationPrefillDecodeUs}/${m.translationPrefillSyncUs} µs · " +
        "generation decode/sync/sample ${m.translationDecodeComputeUs}/${m.translationDecodeSyncUs}/${m.translationSamplingUs} µs · " +
        "${"%.1f".format(m.translationTokensPerSecond)} tok/s\n" +
        "OpenCL placement ${m.translationOffloadedLayers}/${m.translationTotalLayers} layers · " +
        "${m.translationDeviceBytes / (1024 * 1024)} MiB net device free-memory delta · fallback count ${m.translationFallbackCount}\n" +
        "OPUS (${m.opusBackend}) wait ${m.opusWaitMs} ms · compute ${m.opusComputeMs} ms · prefill ${m.opusPrefillMs} ms · " +
        "generation ${m.opusGenerationMs} ms · TTFT ${m.opusFirstTokenMs} ms · " +
        "${"%.1f".format(m.opusTokensPerSecond)} tok/s\n" +
        "Correction ${m.correctionMs} ms / RTF ${"%.2f".format(m.correctionRtf)} · ${m.correctionThreads} threads · skipped ${m.skippedCorrections}\n" +
        "Correction wait ${m.correctionWaitMs} ms · caption results computed/displayed/rejected " +
        "${m.resultsComputed}/${m.resultsDisplayed}/${m.resultsRejected} · display ${m.displayLatencyMs} ms · " +
        "tap → first useful subtitle ${m.firstUsefulCaptionMs?.let { "$it ms" } ?: "—"}\n" +
        (if (m.rejectionReasons.isEmpty()) "" else "Rejected: ${m.rejectionReasons.entries.joinToString { "${it.key}=${it.value}" }}\n") +
        "Segment audio → provisional ${m.audioToProvisionalMs?.let { "$it ms" } ?: "—"} · " +
        "stable prefix → provisional ${m.stableToProvisionalMs?.let { "$it ms" } ?: "—"}\n" +
        "Segment audio → final ${m.audioToFinalMs?.let { "$it ms" } ?: "—"} · " +
        "ASR endpoint → final ${m.finalLatencyMs?.let { "$it ms" } ?: "—"}\n" +
        "Audio queue ${m.audioDepth} · translation ${m.provisionalDepth}+${m.finalDepth}\n" +
        "Microphone input ${m.audioSampleRateHz} Hz · ${m.audioSamplesCaptured} samples · RMS ${"%.4f".format(m.audioRms)} · " +
        "peak ${"%.4f".format(m.audioPeak)} · clipped samples ${m.audioClippedSamples} · timestamp gaps ${m.audioTimestampGaps}\n" +
        "Capture monotonic ns ${m.audioFirstMonotonicNs ?: "—"} → ${m.audioLastMonotonicNs ?: "—"}\n" +
        "Caption backlog ${m.captionBacklogMs} ms · capture backlog ${m.backlogMs} ms\nDropped audio ${m.droppedAudioMs} ms · skipped translations ${m.skippedTranslations}\n" +
        "App process-group RAM ${m.appPssKb / 1024} MiB PSS · :qnn ${if (m.qnnProcessPresent) "${m.qnnPssKb / 1024} MiB PSS" else "not running"} · main-process RSS ${m.rssKb / 1024} MiB\n" +
        "Native heap ${m.nativeHeapKb / 1024} MiB · Java heap ${m.javaHeapKb / 1024} MiB\n" +
        "Model files on disk ≈ ${m.modelFilesDiskKb / 1024} MiB (not RAM) · system available ${m.availableKb / 1024} MiB · " +
        "thermal ${m.thermalStatus} · max readable sensor ${m.thermalMaxC?.let { "%.1f°C".format(it) } ?: "unavailable"} " +
        "(${m.readableThermalSensors} zones)\n" +
        "ADPF work-duration hints ${if (m.adpfActive) "active" else "unavailable"} · sampled every 5 s\n" +
        "Translation runtime: ${m.translatorState.name.lowercase()} · Hy load ${m.hyModelLoadMs} ms" +
        (if (m.hyWarmupMs > 0) " · warm-up ${m.hyWarmupMs} ms" else "") +
        " · resident model reused: ${if (m.residentTranslatorReused) "yes" else "no"}\n" +
        "Listen → ASR ready ${m.timeFromListenToAsrReadyMs?.let { "$it ms" } ?: "—"} · " +
        "→ translator ready ${m.timeFromListenToTranslatorReadyMs?.let { "$it ms" } ?: "—"} · " +
        "→ first translated caption ${m.timeFromListenToFirstTranslatedCaptionMs?.let { "$it ms" } ?: "—"}\n" +
        "Startup backlog: buffered ${m.startupFinalBuffered} · expired ${m.startupFinalDropped} · " +
        "oldest ${m.startupFinalOldestMs} ms · OPUS startup fallbacks used ${m.opusStartupFallbackCount}"
    Text(text, style = MaterialTheme.typography.bodySmall, modifier = Modifier.fillMaxWidth().heightIn(max = 220.dp).verticalScroll(rememberScrollState()).padding(bottom = 8.dp))
}
