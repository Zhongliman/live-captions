package com.asr.live.pipeline

enum class Profile(val source: String, val target: String, val label: String) {
    DUTCH_ENGLISH("nl", "en", "Dutch → English"),
    CHINESE_ENGLISH("zh", "en", "Chinese → English"),
    ENGLISH_FRENCH("en", "fr", "English → French");

    val fastBundle get() = when (this) { DUTCH_ENGLISH -> "opus-nl-en"; ENGLISH_FRENCH -> "opus-en-fr"; CHINESE_ENGLISH -> null }
    val correctionSupported get() = source in setOf("nl", "en")
    companion object {
        fun fromId(id: String?) = entries.firstOrNull { it.name == id } ?: DUTCH_ENGLISH
    }
}

enum class PerformanceMode(val label: String, val presetName: String, val defaultQuality: TranslationQuality) {
    FAST("Fast · Hy 1.8B Q4", "Fast", TranslationQuality.HY_Q4),
    BALANCED("Balanced · Hy 1.8B Q8", "Balanced", TranslationQuality.HY_Q8),
    MAX_QUALITY("Max · Hy 7B Q4", "Max", TranslationQuality.HY_7B_Q4);

    companion object {
        fun defaultFor(soc: String?, manufacturer: String?, model: String?): PerformanceMode =
            if (manufacturer?.contains("Honor", ignoreCase = true) == true &&
                (model?.contains("MBH-N49", ignoreCase = true) == true ||
                 (soc?.contains("SM8750", ignoreCase = true) == true &&
                  model?.contains("Magic V5", ignoreCase = true) == true))) MAX_QUALITY else BALANCED
    }
}

enum class TranslationQuality(val label: String, val bundleId: String? = null) {
    HY_7B_Q6("Hy-MT2 7B Q6_K", "hymt2-7b-Q6_K"),
    HY_7B_Q5("Hy-MT2 7B Q5_K_M", "hymt2-7b-Q5_K_M"),
    HY_7B_Q4("Hy-MT2 7B Q4_K_M", "hymt2-7b-Q4_K_M"),
    HY_7B_Q8("Hy-MT2 7B Q8_0 reference", "hymt2-7b-Q8_0"),
    HY_Q8("Hy-MT2 1.8B Q8_0", "hymt2-Q8_0"),
    HY_Q6("Hy-MT2 Q6_K", "hymt2-Q6_K"),
    HY_Q4("Hy-MT2 Q4_K_M", "hymt2-Q4_K_M"),
    ML_KIT("ML Kit · local fast translation")
}

data class SessionConfig(
    val profile: Profile = Profile.DUTCH_ENGLISH,
    val performanceMode: PerformanceMode = PerformanceMode.BALANCED,
    val modelId: String = "nemotron-3.5-560ms-int8",
    val threads: Int = 6,
    val correctionThreads: Int = 4,
    val quality: TranslationQuality = TranslationQuality.HY_Q8,
    val correction: Boolean = false,
    val glossary: String = "",
    val qnn: Boolean = false,
    val gpuTranslation: Boolean = false,
    val translationThreads: Int = 4,
    val translationBatch: Int = 256,
    val translationUbatch: Int = 128,
    /** Automatic calibrated engine identity; null keeps the manual/preset quality behavior. */
    val primaryEngineId: String? = null,
) {
    init {
        require(threads in 1..8); require(correctionThreads in setOf(2, 4, 6, 8))
        require(translationThreads in setOf(2, 4))
        require(translationBatch in setOf(128, 256, 512))
        require(translationUbatch in setOf(64, 128, 256) && translationUbatch <= translationBatch)
    }
}

/** Presets are explicit; manual model/backend choices remain available afterward. */
fun SessionConfig.withMode(mode: PerformanceMode): SessionConfig {
    val recognizer = if (profile == Profile.CHINESE_ENGLISH) "qwen3-asr-0.6b-int8" else "nemotron-3.5-560ms-int8"
    return copy(performanceMode = mode, modelId = recognizer, quality = mode.defaultQuality,
        primaryEngineId = null,
        correction = mode == PerformanceMode.MAX_QUALITY && profile.correctionSupported)
}

/** OPUS is loaded only for the explicit low-latency A/B profile, never for Max Quality. */
val SessionConfig.opusBenchmarkEnabled: Boolean
    get() = performanceMode == PerformanceMode.FAST && profile.fastBundle != null && quality != TranslationQuality.ML_KIT &&
        primaryEngineId != "ml-kit" && primaryEngineId?.startsWith("opus-") != true

/**
 * Whether an OPUS model can stand in for Hy-MT2 while it is still loading. Independent of the
 * A/B benchmark preset: any profile with a fast bundle and a non-ML-Kit quality can fall back to
 * it during startup, since OPUS loads in a fraction of the time Hy-MT2 needs.
 */
val SessionConfig.opusStartupFallbackEnabled: Boolean
    get() = profile.fastBundle != null && quality != TranslationQuality.ML_KIT &&
        primaryEngineId != "ml-kit" && primaryEngineId?.startsWith("opus-") != true

/** True once a manual override no longer matches the preset's own translation quality. */
val SessionConfig.qualityOverridesPreset: Boolean
    get() = quality != performanceMode.defaultQuality
