package com.asr.live.service

import android.app.ActivityManager
import android.content.Context
import android.os.Debug
import android.os.Process
import android.system.Os
import android.system.OsConstants

/** All values are snapshots in KiB. Model bytes are file sizes, not additional RSS. */
data class MemorySnapshot(
    val appPssKb: Long,
    val rssKb: Long,
    val nativeHeapKb: Long,
    val javaHeapKb: Long,
    val availableKb: Long,
    val thermalStatus: String,
    val thermalMaxC: Double?,
    val readableThermalSensors: Int,
    val qnnPssKb: Long,
    val qnnProcessPresent: Boolean,
)

object MemoryUsage {
    fun sample(context: Context): MemorySnapshot {
        val manager = context.getSystemService(ActivityManager::class.java)
        val own = Debug.MemoryInfo().also { Debug.getMemoryInfo(it) }.totalPss.toLong()
        val children = manager.runningAppProcesses.orEmpty().filter {
                it.uid == Process.myUid() && it.pid != Process.myPid()
            }
        val childPss = runCatching {
            if (children.isEmpty()) emptyList()
            else manager.getProcessMemoryInfo(children.map { it.pid }.toIntArray()).map { it.totalPss.toLong() }
        }.getOrDefault(emptyList())
        val processPssKb = children.zip(childPss).groupBy({ it.first.processName }, { it.second })
            .mapValues { (_, values) -> values.sum() }
        val additional = childPss.sum()
        val qnnPss = processPssKb.filterKeys { it.endsWith(":qnn") }.values.sum()
        val rss = runCatching {
            val pages = java.io.File("/proc/self/statm").readText().trim().split(Regex("\\s+"))[1].toLong()
            pages * Os.sysconf(OsConstants._SC_PAGESIZE) / 1024
        }.getOrDefault(0L)
        val runtime = Runtime.getRuntime()
        val system = ActivityManager.MemoryInfo().also { manager.getMemoryInfo(it) }
        val thermal = if (android.os.Build.VERSION.SDK_INT >= 29) {
            val status = context.getSystemService(android.os.PowerManager::class.java)?.currentThermalStatus
            when (status) {
                android.os.PowerManager.THERMAL_STATUS_NONE -> "None"
                android.os.PowerManager.THERMAL_STATUS_LIGHT -> "Light"
                android.os.PowerManager.THERMAL_STATUS_MODERATE -> "Moderate"
                android.os.PowerManager.THERMAL_STATUS_SEVERE -> "Severe"
                android.os.PowerManager.THERMAL_STATUS_CRITICAL -> "Critical"
                android.os.PowerManager.THERMAL_STATUS_EMERGENCY -> "Emergency"
                android.os.PowerManager.THERMAL_STATUS_SHUTDOWN -> "Shutdown"
                else -> "Unknown"
            }
        } else "Unavailable"
        val temperatures = runCatching {
            java.io.File("/sys/class/thermal").listFiles().orEmpty()
                .filter { it.name.startsWith("thermal_zone") }
                .take(64)
                .mapNotNull { zone ->
                    val raw = zone.resolve("temp").readText().trim().toDoubleOrNull() ?: return@mapNotNull null
                    val celsius = if (kotlin.math.abs(raw) > 1000.0) raw / 1000.0 else raw
                    celsius.takeIf { it in -30.0..150.0 }
                }
        }.getOrDefault(emptyList())
        return MemorySnapshot(own + additional, rss, Debug.getNativeHeapAllocatedSize() / 1024,
            (runtime.totalMemory() - runtime.freeMemory()) / 1024, system.availMem / 1024, thermal,
            temperatures.maxOrNull(), temperatures.size,
            qnnPss, processPssKb.keys.any { it.endsWith(":qnn") })
    }

    /** Assess only the memory this new translator adds; existing residents are already in availMem. */
    fun assessTranslatorLoad(context: Context, modelWeightBytes: Long, largeModel: Boolean = false,
                            batch: Int = 256, ubatch: Int = 128, openCl: Boolean = false,
                            appBudgetBytes: Long? = null): MemoryAdmission =
        assessAdditionalLoad(context, MemoryAdmissionPolicy.translatorEstimate(
            modelWeightBytes, largeModel, batch, ubatch, openCl).additionalBytes,
            appBudgetBytes = appBudgetBytes)

    fun assessAdditionalLoad(context: Context, additionalBytes: Long, reserveBytes: Long? = null,
                             appBudgetBytes: Long? = null): MemoryAdmission {
        // Same-UID PSS includes the QNN worker. GPU memory is shared with system RAM, so a
        // backend free-memory delta is not another private allocation to add to this snapshot.
        val currentAppBytes = if (appBudgetBytes != null) sample(context).appPssKb * 1024 else 0L
        val info = ActivityManager.MemoryInfo().also { context.getSystemService(ActivityManager::class.java).getMemoryInfo(it) }
        return MemoryAdmissionPolicy.assess(info.availMem, info.totalMem, info.threshold, info.lowMemory,
            additionalBytes, reserveBytes ?: MemoryAdmissionPolicy.systemReserve(info.totalMem, info.threshold),
            currentAppBytes = currentAppBytes, appBudgetBytes = appBudgetBytes)
    }

    /** Compatibility for callers that already supply an estimate of additional memory, not disk totals. */
    fun canLoad(context: Context, modelBytes: Long, reserveBytes: Long? = null): Boolean =
        assessAdditionalLoad(context, modelBytes, reserveBytes).allowed
}
