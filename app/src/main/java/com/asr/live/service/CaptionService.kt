package com.asr.live.service

import android.app.*
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.IBinder
import android.os.Build
import android.util.Log
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat
import com.asr.live.MainActivity
import com.asr.live.i18n.ResidentTranslatorManager
import com.asr.live.model.ModelCatalog
import com.asr.live.pipeline.*
import java.util.concurrent.atomic.AtomicLong
import android.provider.Settings
import android.net.Uri
import com.asr.live.overlay.*
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*

class CaptionService : Service() {
    @Volatile private var displayedGeneration = 0L
    @Volatile private var current: CaptionSession? = null
    @Volatile private var destroyed = false
    @Volatile private var latestStartId = 0
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private val shutdownScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val terminationWaiters = mutableMapOf<CaptionSession, MutableList<() -> Unit>>()
    private lateinit var overlayPrefs: OverlayPreferences
    private lateinit var overlay: FloatingCaptions
    private var notificationModel = "Live captions"
    override fun onCreate() {
        super.onCreate()
        overlayPrefs = OverlayPreferences(this)
        overlay = FloatingCaptions(this, overlayPrefs)
        scope.launch {
            combine(CaptionState.lines, CaptionState.lifecycle, overlayPrefs.state) { lines, lifecycle, options -> Triple(lines, lifecycle, options) }
                .collect { (lines, lifecycle, _) ->
                    overlay.render(lines, lifecycle, if (CaptionState.metrics.value.profile == Profile.ENGLISH_FRENCH.label) "French" else "English")
                }
        }
        scope.launch {
            overlayPrefs.state.collect {
                if (CaptionState.running.value) getSystemService(NotificationManager::class.java).notify(NOTIF_ID, notification(notificationModel))
            }
        }
        scope.launch {
            while (isActive) {
                delay(5000)
                val generation = displayedGeneration
                if (CaptionState.running.value) {
                    val memory = withContext(Dispatchers.IO) { MemoryUsage.sample(this@CaptionService) }
                    CaptionState.metrics(generation) { it.copy(appPssKb = memory.appPssKb, rssKb = memory.rssKb,
                        qnnPssKb = memory.qnnPssKb, qnnProcessPresent = memory.qnnProcessPresent,
                        nativeHeapKb = memory.nativeHeapKb, javaHeapKb = memory.javaHeapKb, availableKb = memory.availableKb,
                        thermalStatus = memory.thermalStatus, thermalMaxC = memory.thermalMaxC,
                        readableThermalSensors = memory.readableThermalSensors) }
                    logRunSample(generation)
                    // Recheck permission and lock-screen visibility even during a silent phrase.
                    overlay.render(CaptionState.lines.value, CaptionState.lifecycle.value,
                        if (CaptionState.metrics.value.profile == Profile.ENGLISH_FRENCH.label) "French" else "English")
                    if (CaptionState.running.value) getSystemService(NotificationManager::class.java).notify(NOTIF_ID, notification(notificationModel))
                }
            }
        }
    }

    /** Low-rate, transcript-free snapshots make sustained device runs reproducible from logcat. */
    private fun logRunSample(generation: Long) {
        val atNs = System.nanoTime()
        val metrics = CaptionState.metrics.value
        val sessionStartNs = CaptionState.sessionStartNs(generation) ?: atNs
        val reasons = metrics.rejectionReasons.entries.sortedBy { it.key }
            .joinToString(",") { "${it.key.replace(Regex("[^A-Za-z0-9_-]"), "_")}:${it.value}" }
            .ifBlank { "none" }
        Log.i("LiveCaptionsRun", "session=$generation atNs=$atNs elapsedNs=${(atNs - sessionStartNs).coerceAtLeast(0)} " +
            "state=${CaptionState.lifecycle.value} profile=${metrics.profile.replace(' ', '_')} " +
            "asrBackend=${metrics.backend.replace(' ', '_')} translationBackend=${metrics.translationBackend.replace(' ', '_')} " +
            "opusBackend=${metrics.opusBackend.replace(' ', '_')} decodeCalls=${metrics.decodeCalls} " +
            "decodeP50Ms=${metrics.decodeP50Ms} decodeP95Ms=${metrics.decodeP95Ms} decodeMaxMs=${metrics.decodeMaxMs} " +
            "asrComputeMs=${metrics.asrComputeMs} endpointWaitMs=${metrics.endpointWaitMs ?: -1} " +
            "hyWaitMs=${metrics.hyWaitMs} hyComputeMs=${metrics.hyComputeMs} hyDisplayMs=${metrics.hyDisplayMs} " +
            "hyPrefillMs=${metrics.translationPrefillMs} hyGenerationMs=${metrics.translationDecodeMs} " +
            "hyFirstTokenMs=${metrics.translationFirstTokenMs} hyFirstVisibleMs=${metrics.translationFirstVisibleMs} " +
            "hyCompleteMs=${metrics.translationCompleteMs} hyInputTokens=${metrics.translationInputTokens} " +
            "hyOutputTokens=${metrics.translationOutputTokens} hyCacheTokens=${metrics.translationCacheTokens} " +
            "hyOffloadedLayers=${metrics.translationOffloadedLayers} hyTotalLayers=${metrics.translationTotalLayers} " +
            "hyFallbacks=${metrics.translationFallbackCount} opusWaitMs=${metrics.opusWaitMs} " +
            "opusComputeMs=${metrics.opusComputeMs} opusPrefillMs=${metrics.opusPrefillMs} " +
            "opusGenerationMs=${metrics.opusGenerationMs} opusFirstTokenMs=${metrics.opusFirstTokenMs} " +
            "correctionWaitMs=${metrics.correctionWaitMs} correctionComputeMs=${metrics.correctionMs} " +
            "correctionSkipped=${metrics.skippedCorrections} resultsComputed=${metrics.resultsComputed} " +
            "resultsDisplayed=${metrics.resultsDisplayed} resultsRejected=${metrics.resultsRejected} " +
            "rejectionReasons=$reasons firstUsefulMs=${metrics.firstUsefulCaptionMs ?: -1} " +
            "audioToProvisionalMs=${metrics.audioToProvisionalMs ?: -1} audioToFinalMs=${metrics.audioToFinalMs ?: -1} " +
            "finalLatencyMs=${metrics.finalLatencyMs ?: -1} audioHz=${metrics.audioSampleRateHz} " +
            "audioSamples=${metrics.audioSamplesCaptured} audioRms=${metrics.audioRms} audioPeak=${metrics.audioPeak} " +
            "audioClipped=${metrics.audioClippedSamples} audioTimestampGaps=${metrics.audioTimestampGaps} " +
            "audioFirstNs=${metrics.audioFirstMonotonicNs ?: -1} audioLastNs=${metrics.audioLastMonotonicNs ?: -1} " +
            "audioDepth=${metrics.audioDepth} provisionalDepth=${metrics.provisionalDepth} finalDepth=${metrics.finalDepth} " +
            "captionBacklogMs=${metrics.captionBacklogMs} captureBacklogMs=${metrics.backlogMs} " +
            "droppedAudioMs=${metrics.droppedAudioMs} skippedTranslations=${metrics.skippedTranslations} " +
            "appGroupPssKb=${metrics.appPssKb} qnnPssKb=${metrics.qnnPssKb} qnnPresent=${metrics.qnnProcessPresent} " +
            "availableKb=${metrics.availableKb} thermal=${metrics.thermalStatus} " +
            "thermalMaxC=${metrics.thermalMaxC ?: -1.0} thermalSensors=${metrics.readableThermalSensors}")
    }

    override fun onBind(intent: Intent?): IBinder? = null
    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        latestStartId = startId
        if (intent?.action == ACTION_OVERLAY) {
            if (CaptionState.running.value && Settings.canDrawOverlays(this))
                overlayPrefs.update(overlayPrefs.state.value.copy(enabled = !overlayPrefs.state.value.enabled))
            else if (!CaptionState.running.value) stopSelfResult(startId)
            return START_NOT_STICKY
        }
        val generation = requests.incrementAndGet()
        if (intent?.action == ACTION_STOP || intent == null) {
            val stoppedGeneration = displayedGeneration
            val session = current
            lifecycleLog(stoppedGeneration, "stop-requested", "state=${CaptionState.lifecycle.value}")
            if (CaptionState.lifecycle.value == ListeningState.LISTENING) {
                CaptionState.stopping(stoppedGeneration)
                session?.stop()
            } else {
                CaptionState.cancel(stoppedGeneration)
                session?.cancel()
            }
            if (session == null) {
                finishStop(stoppedGeneration, generation)
            } else {
                afterTermination(session) {
                    if (current === session) current = null
                    finishStop(stoppedGeneration, generation)
                }
                escalateShutdown(session, stoppedGeneration, generation)
            }
            return START_NOT_STICKY
        }
        lifecycleLog(generation, "listen-requested", "model=${intent.getStringExtra("model")}")
        val config = SessionConfig(
            profile = Profile.fromId(intent.getStringExtra("profile")),
            performanceMode = PerformanceMode.entries.firstOrNull { it.name == intent.getStringExtra("performanceMode") } ?: PerformanceMode.BALANCED,
            modelId = intent.getStringExtra("model") ?: ModelCatalog.DEFAULT.id,
            threads = intent.getIntExtra("threads", 6).coerceIn(1, 8),
            correctionThreads = intent.getIntExtra("correctionThreads", 4).let { if (it in setOf(2, 4, 6, 8)) it else 4 },
            quality = TranslationQuality.entries.firstOrNull { it.name == intent.getStringExtra("quality") } ?: TranslationQuality.HY_Q8,
            primaryEngineId = intent.getStringExtra("primaryEngineId"),
            compareTranslations = intent.getBooleanExtra("compareTranslations", false),
            memoryBudgetGiB = intent.getIntExtra("memoryBudgetGiB", 0).let { if (it == 10) 10 else 0 },
            qnn = intent.getBooleanExtra("qnn", false),
            gpuTranslation = intent.getBooleanExtra("gpuTranslation", com.asr.live.BuildConfig.OPENCL_ENABLED),
            translationThreads = intent.getIntExtra("translationThreads", 4).let { if (it in setOf(2, 4)) it else 4 },
            translationBatch = intent.getIntExtra("translationBatch", 256).let { if (it in setOf(128, 256, 512)) it else 256 },
            translationUbatch = intent.getIntExtra("translationUbatch", 128).let { if (it in setOf(64, 128, 256)) it else 128 },
            correction = intent.getBooleanExtra("correction", false),
            glossary = intent.getStringExtra("glossary") ?: "",
        )
        startForegroundNotification(ModelCatalog.byId(config.modelId)?.displayName ?: "Loading captions")
        displayedGeneration = generation
        CaptionState.begin(generation, config, ModelCatalog.byId(config.modelId)?.shortName ?: "Nemotron")
        val previous = current
        if (previous == null) {
            createAndStartSession(generation, config)
        } else {
            // A replacement session waits for the previous one's native cleanup so the two never
            // own the same recognizer, translator or QNN process at the same time.
            lifecycleLog(generation, "previous-session-cancelled", "previous=${previous.generation}")
            previous.cancel()
            afterTermination(previous) {
                if (current === previous) current = null
                if (!destroyed && generation == requests.get()) createAndStartSession(generation, config)
            }
            scope.launch {
                delay(ShutdownDeadline.HARD_MS)
                if (!destroyed && generation == requests.get() && current === previous && !previous.isTerminated()) {
                    CaptionState.error(generation, "Previous model workers are still stopping. Retry Listen after cleanup finishes.")
                    releaseUi(generation, generation)
                }
            }
        }
        return START_NOT_STICKY
    }
    override fun onDestroy() {
        destroyed = true; requests.incrementAndGet()
        scope.cancel(); shutdownScope.cancel(); terminationWaiters.clear()
        overlay.close(); overlayPrefs.close()
        current?.let { CaptionState.cancel(it.generation); it.cancel() }
        current = null
        // A foreground Service is routinely destroyed after Stop. Android can create another
        // service instance in the same process on the next Listen, so ordinary service teardown
        // must leave the process-wide resident models intact.
        super.onDestroy()
    }

    /**
     * Trim unused models without invalidating a live worker's lease. UI_HIDDEN is also handled
     * because modern Android no longer guarantees legacy critical-memory notifications.
     */
    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        if (level == android.content.ComponentCallbacks2.TRIM_MEMORY_RUNNING_CRITICAL ||
            level >= android.content.ComponentCallbacks2.TRIM_MEMORY_BACKGROUND ||
            level == android.content.ComponentCallbacks2.TRIM_MEMORY_UI_HIDDEN) {
            Thread({ ResidentTranslatorManager.closeIdle() }, "resident-translator-trim")
                .apply { isDaemon = true }.start()
        }
    }

    private fun createAndStartSession(generation: Long, config: SessionConfig) {
        if (destroyed || generation != requests.get()) return
        lateinit var session: CaptionSession
        session = CaptionSession(applicationContext, generation, config) { message ->
            CaptionState.error(generation, message)
            CaptionState.cancel(generation)
            scope.launch {
                if (!destroyed && generation == requests.get() && current === session) {
                    afterTermination(session) {
                        if (current === session) current = null
                        finishStop(generation, generation)
                    }
                    escalateShutdown(session, generation, generation)
                }
            }
        }
        current = session
        try {
            session.start()
        } catch (t: Throwable) {
            CaptionState.error(generation, "Could not start workers: ${t.message}")
            CaptionState.cancel(generation)
            session.cancel()
            afterTermination(session) {
                if (current === session) current = null
                finishStop(generation, generation)
            }
        }
    }

    /** Wait for worker-owned native cleanup off the main and service control threads. */
    private fun afterTermination(session: CaptionSession, action: () -> Unit) {
        terminationWaiters[session]?.let { it += action; return }
        terminationWaiters[session] = mutableListOf(action)
        val startedAtNs = System.nanoTime()
        shutdownScope.launch {
            try {
                while (isActive && !session.join(JOIN_SLICE_MS)) { }
            } catch (_: InterruptedException) {
                Thread.currentThread().interrupt()
                return@launch
            }
            if (!isActive) return@launch
            lifecycleLog(session.generation, "session-fully-stopped",
                "elapsedMs=${(System.nanoTime() - startedAtNs) / 1_000_000L}")
            withContext(Dispatchers.Main.immediate) {
                val actions = terminationWaiters.remove(session).orEmpty()
                actions.forEach { it() }
            }
        }
    }

    /**
     * Bounds how long the UI may stay in STOPPING.
     *
     * Translation work is dropped first so the ASR flush still gets to publish trailing speech;
     * only then is the session force-cancelled, and only after the hard deadline is the UI
     * released. `current` is deliberately *not* cleared here: the session's workers own their
     * native objects and a replacement session must still wait for them, so a stuck worker can
     * never produce two owners of the same recognizer, translator or QNN process.
     */
    private fun escalateShutdown(session: CaptionSession, stoppedGeneration: Long, requestGeneration: Long) {
        shutdownScope.launch {
            val deadline = ShutdownDeadline(startedAtNs = System.nanoTime())
            delay(deadline.remainingUntilAbandonTranslationsMs())
            if (!isActive || session.isTerminated()) return@launch
            lifecycleLog(session.generation, "translation-work-abandoned",
                "graceMs=${deadline.graceMs} elapsedMs=${deadline.elapsedMs()}")
            runCatching { session.abandonTranslations() }

            delay(deadline.remainingUntilForceCancelMs())
            if (!isActive || session.isTerminated()) return@launch
            lifecycleLog(session.generation, "forced-cancellation-triggered",
                "cancelMs=${deadline.cancelMs} elapsedMs=${deadline.elapsedMs()}")
            runCatching { session.cancel() }

            delay(deadline.remainingUntilReleaseUiMs())
            if (!isActive || session.isTerminated()) return@launch
            lifecycleLog(session.generation, "shutdown-deadline-exceeded",
                "hardMs=${deadline.hardMs}; releasing UI while workers unwind")
            withContext(Dispatchers.Main.immediate) { releaseUi(stoppedGeneration, requestGeneration) }
        }
    }

    private fun finishStop(stoppedGeneration: Long, requestGeneration: Long) {
        if (destroyed || requestGeneration != requests.get()) return
        logRunSample(stoppedGeneration)
        releaseUi(stoppedGeneration, requestGeneration)
    }

    private fun releaseUi(stoppedGeneration: Long, requestGeneration: Long) {
        if (destroyed || requestGeneration != requests.get()) return
        CaptionState.stopped(stoppedGeneration)
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelfResult(latestStartId)
    }

    /** Transcript-free lifecycle trail; pairs with CaptionPipeline traces from the session. */
    private fun lifecycleLog(generation: Long, event: String, details: String = "") {
        val atNs = System.nanoTime()
        val sessionStartNs = CaptionState.sessionStartNs(generation) ?: atNs
        Log.i("CaptionLifecycle", "session=$generation atNs=$atNs " +
            "sessionElapsedNs=${(atNs - sessionStartNs).coerceAtLeast(0)} event=$event $details")
    }

    private fun startForegroundNotification(modelName: String) {
        notificationModel = modelName
        val nm = getSystemService(NotificationManager::class.java)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            nm.createNotificationChannel(
                NotificationChannel(CHANNEL_ID, "Live captions", NotificationManager.IMPORTANCE_LOW)
            )
        }

        ServiceCompat.startForeground(this, NOTIF_ID, notification(modelName), ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE)
    }
    private fun notification(modelName: String): Notification {
        val open = PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE,
        )
        val stop = PendingIntent.getService(
            this, 1, Intent(this, CaptionService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_IMMUTABLE,
        )

        val allowed = Settings.canDrawOverlays(this)
        val toggle = if (allowed) PendingIntent.getService(this, 2,
            Intent(this, CaptionService::class.java).setAction(ACTION_OVERLAY), PendingIntent.FLAG_IMMUTABLE)
        else PendingIntent.getActivity(this, 3,
            Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION, Uri.parse("package:$packageName")), PendingIntent.FLAG_IMMUTABLE)
        @Suppress("DEPRECATION") // int-icon Action.Builder keeps us off the icons dependency
        val notification: Notification = Notification.Builder(this, CHANNEL_ID)
            .setContentTitle("Listening")
            .setContentText(modelName)
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setContentIntent(open)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .addAction(Notification.Action.Builder(android.R.drawable.ic_menu_view,
                if (!allowed) "Overlay permission" else if (overlayPrefs.state.value.enabled) "Hide captions" else "Show captions", toggle).build())
            .addAction(Notification.Action.Builder(android.R.drawable.ic_delete, "Stop", stop).build())
            .build()

        return notification
    }

    companion object {
        private val requests = AtomicLong()
        private const val ACTION_STOP = "com.asr.live.action.STOP"
        private const val ACTION_OVERLAY = "com.asr.live.action.OVERLAY"
        private const val CHANNEL_ID = "captions"
        private const val NOTIF_ID = 1
        /** Short slices keep the escalation timers responsive without busy-waiting. */
        private const val JOIN_SLICE_MS = 100L
        fun start(ctx: Context, config: SessionConfig) {
            ContextCompat.startForegroundService(ctx, Intent(ctx, CaptionService::class.java)
                .putExtra("profile", config.profile.name).putExtra("performanceMode", config.performanceMode.name).putExtra("model", config.modelId)
                .putExtra("threads", config.threads).putExtra("correctionThreads", config.correctionThreads)
                .putExtra("gpuTranslation", config.gpuTranslation).putExtra("quality", config.quality.name)
                .putExtra("primaryEngineId", config.primaryEngineId)
                .putExtra("compareTranslations", config.compareTranslations)
                .putExtra("memoryBudgetGiB", config.memoryBudgetGiB)
                .putExtra("translationThreads", config.translationThreads)
                .putExtra("translationBatch", config.translationBatch).putExtra("translationUbatch", config.translationUbatch)
                .putExtra("qnn", config.qnn).putExtra("correction", config.correction).putExtra("glossary", config.glossary))
        }
        fun stop(ctx: Context) { ctx.startService(Intent(ctx, CaptionService::class.java).setAction(ACTION_STOP)) }
    }
}
