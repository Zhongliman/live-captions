package com.asr.live.pipeline.engine

import android.content.Context
import android.content.SharedPreferences
import android.os.Build
import com.asr.live.BuildConfig
import com.asr.live.pipeline.Profile
import com.asr.live.pipeline.TranslationQuality
import org.json.JSONObject

/** Everything that, if it changes, could invalidate a saved selection. */
data class DeviceFingerprint(
    val soc: String, val abi: String, val osRelease: String, val buildFingerprint: String,
    val qnnBuild: Boolean, val openClBuild: Boolean,
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("soc", soc); put("abi", abi); put("osRelease", osRelease); put("buildFingerprint", buildFingerprint)
        put("qnnBuild", qnnBuild); put("openClBuild", openClBuild)
    }
    companion object {
        fun current(): DeviceFingerprint = DeviceFingerprint(
            soc = if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else "unknown",
            abi = Build.SUPPORTED_ABIS.firstOrNull() ?: "unknown",
            osRelease = Build.VERSION.RELEASE ?: "unknown",
            buildFingerprint = Build.FINGERPRINT ?: "unknown",
            qnnBuild = BuildConfig.QNN_ENABLED, openClBuild = BuildConfig.OPENCL_ENABLED)
        fun fromJson(j: JSONObject) = DeviceFingerprint(j.getString("soc"), j.getString("abi"),
            j.getString("osRelease"), j.getString("buildFingerprint"), j.getBoolean("qnnBuild"), j.getBoolean("openClBuild"))
    }
}

/** One persisted per-language selection, atomic and versioned so a later release can tell it apart from a stale one. */
data class CalibrationRecord(
    val profile: Profile, val engineId: String, val quality: TranslationQuality?,
    val chrfScore: Double?, val fingerprint: DeviceFingerprint,
    val policyVersion: String, val corpusVersion: String, val calibratedAtEpochMs: Long,
    val backend: Backend = Backend.CPU,
) {
    /** True when the runtime this record was calibrated against no longer matches the current build/device. */
    fun isStale(): Boolean {
        val nowFingerprint = DeviceFingerprint.current()
        return fingerprint != nowFingerprint || policyVersion != EngineSelectionPolicy.POLICY_VERSION ||
            corpusVersion != BenchmarkCorpus.VERSION
    }
    fun toJson(): JSONObject = JSONObject().apply {
        put("engineId", engineId); put("quality", quality?.name ?: JSONObject.NULL)
        put("chrfScore", chrfScore ?: JSONObject.NULL); put("fingerprint", fingerprint.toJson())
        put("policyVersion", policyVersion); put("corpusVersion", corpusVersion); put("calibratedAtEpochMs", calibratedAtEpochMs)
        put("backend", backend.name)
    }
    companion object {
        fun fromJson(profile: Profile, j: JSONObject): CalibrationRecord = CalibrationRecord(
            profile = profile, engineId = j.getString("engineId"),
            quality = if (j.isNull("quality")) null else
                TranslationQuality.entries.firstOrNull { it.name == j.getString("quality") },
            chrfScore = if (j.isNull("chrfScore")) null else j.getDouble("chrfScore"),
            fingerprint = DeviceFingerprint.fromJson(j.getJSONObject("fingerprint")),
            policyVersion = j.getString("policyVersion"), corpusVersion = j.getString("corpusVersion"),
            calibratedAtEpochMs = j.getLong("calibratedAtEpochMs"),
            backend = runCatching { Backend.valueOf(j.optString("backend", Backend.CPU.name)) }.getOrDefault(Backend.CPU))
    }
}

/**
 * Persists the automatic-selection result per language pair so it survives process restarts and
 * applies immediately without another APK release (this is exactly the "apply in the same
 * build" requirement — the running app reads this store, nothing is compiled in).
 */
class CalibrationStore internal constructor(private val prefs: SharedPreferences) {
    constructor(context: Context) : this(context.getSharedPreferences("engine_calibration_v1", Context.MODE_PRIVATE))

    @Synchronized fun save(record: CalibrationRecord): Boolean {
        return runCatching {
            prefs.edit().putString(key(record.profile), record.toJson().toString()).commit()
        }.getOrDefault(false)
    }

    @Synchronized fun get(profile: Profile): CalibrationRecord? {
        val raw = prefs.getString(key(profile), null) ?: return null
        return runCatching { CalibrationRecord.fromJson(profile, JSONObject(raw)) }.getOrNull()
    }

    /** A record that no longer matches the current runtime is treated as absent, not silently reused. */
    @Synchronized fun getFresh(profile: Profile): CalibrationRecord? =
        runCatching { get(profile)?.takeUnless { it.isStale() } }.getOrNull()

    fun selectedQuality(profile: Profile): TranslationQuality? = getFresh(profile)?.quality

    @Synchronized fun clear(profile: Profile) { prefs.edit().remove(key(profile)).apply() }

    private fun key(profile: Profile) = "selection.${profile.name}"
}
