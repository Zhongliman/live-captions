package com.asr.live.service

private const val MIB = 1024L * 1024

/** An estimate of one new translator allocation, never the sum of installed model files. */
data class TranslatorAllocationEstimate(
    val weightsBytes: Long,
    val contextBytes: Long,
    val computeBytes: Long,
    val backendStagingBytes: Long,
) {
    init {
        require(listOf(weightsBytes, contextBytes, computeBytes, backendStagingBytes).all { it >= 0 })
    }
    val additionalBytes: Long get() = saturatedMemorySum(weightsBytes, contextBytes, computeBytes, backendStagingBytes)
}

data class MemoryAdmission(
    val availableBytes: Long,
    val additionalBytes: Long,
    val reserveBytes: Long,
    val systemLowMemory: Boolean,
    val allowed: Boolean,
    val currentAppBytes: Long = 0,
    val appBudgetBytes: Long? = null,
    val effectiveAppBudgetBytes: Long? = null,
) {
    val requiredBytes: Long get() = saturatedMemorySum(additionalBytes, reserveBytes)
    val description: String get() {
        val mode = if (appBudgetBytes == null) "available-RAM mode" else
            "app-budget mode: ${appBudgetBytes / MIB} MiB allowance, " +
                "${effectiveAppBudgetBytes?.div(MIB)} MiB physical-RAM-limited allowance, " +
                "${currentAppBytes / MIB} MiB current app PSS; Android may reclaim cached processes; " +
                "this estimate is not a guaranteed peak or allocation"
        return "${if (allowed) "Memory check passed" else "Memory admission refused"} ($mode): " +
            "${availableBytes / MIB} MiB available, ${additionalBytes / MIB} MiB estimated additional allocation, " +
            "${reserveBytes / MIB} MiB system headroom; Android low-memory=${if (systemLowMemory) "yes" else "no"}"
    }
}

/**
 * Pure admission policy shared by live sessions, benchmarks and calibration. Available RAM
 * already reflects existing ASR, QNN and translator allocations: never add those a second time.
 * An explicit app allowance may permit Android to reclaim cached processes instead of requiring
 * all new memory to be available already. It is soft admission, not a hard peak-memory limit:
 * PSS and allocator estimates are snapshots, and allocation can still fail. Shared GPU counters
 * must not be added to application PSS. No path allocates memory just to fill the allowance.
 */
object MemoryAdmissionPolicy {
    /**
     * GGUF weights are budgeted once using the selected weight file's bytes. The runtime has
     * a fixed 2048-token context; reserve 256/384 MiB for its KV cache and runtime metadata,
     * plus a compute budget that grows with the configured batches. OpenCL gets a separate
     * bounded staging allowance; its shared-memory weight buffers are already in weightsBytes.
     */
    fun translatorEstimate(modelWeightBytes: Long, largeModel: Boolean = false,
                           batch: Int = 256, ubatch: Int = 128,
                           openCl: Boolean = false): TranslatorAllocationEstimate {
        require(modelWeightBytes > 0) { "Selected translator weight size must be positive" }
        require(batch > 0 && ubatch in 1..batch) { "Invalid translation batch sizes" }
        val context = (if (largeModel) 384L else 256L) * MIB
        val compute = saturatedMemorySum(128L * MIB, batch.toLong() * MIB / 4, ubatch.toLong() * MIB / 2)
        val staging = if (openCl) (modelWeightBytes / 8).coerceIn(128L * MIB, 512L * MIB) else 0L
        return TranslatorAllocationEstimate(modelWeightBytes, context, compute, staging)
    }

    /**
     * Keep Android's own low-memory threshold plus 256 MiB, or 1/16 of physical RAM
     * bounded to 512–1024 MiB, whichever is larger. This scales without assuming a 16 GB phone.
     */
    fun systemReserve(totalBytes: Long, lowMemoryThresholdBytes: Long): Long {
        require(totalBytes >= 0 && lowMemoryThresholdBytes >= 0)
        val deviceHeadroom = (totalBytes / 16).coerceIn(512L * MIB, 1024L * MIB)
        return maxOf(deviceHeadroom, saturatedMemorySum(lowMemoryThresholdBytes, 256L * MIB))
    }

    fun assess(availableBytes: Long, totalBytes: Long, lowMemoryThresholdBytes: Long,
               systemLowMemory: Boolean, additionalBytes: Long,
               reserveBytes: Long = systemReserve(totalBytes, lowMemoryThresholdBytes),
               currentAppBytes: Long = 0, appBudgetBytes: Long? = null): MemoryAdmission {
        require(availableBytes >= 0 && totalBytes >= 0 && lowMemoryThresholdBytes >= 0 &&
            additionalBytes >= 0 && reserveBytes >= 0 && currentAppBytes >= 0)
        require(appBudgetBytes == null || appBudgetBytes > 0) { "Application memory allowance must be positive" }
        val effectiveBudget = appBudgetBytes?.let { minOf(it, (totalBytes - reserveBytes).coerceAtLeast(0)) }
        // Subtraction avoids overflow and equality is allowed: the requested reserve remains.
        val allowed = !systemLowMemory && availableBytes >= reserveBytes &&
            if (effectiveBudget == null) additionalBytes <= availableBytes - reserveBytes
            else totalBytes >= reserveBytes && currentAppBytes <= effectiveBudget &&
                additionalBytes <= effectiveBudget - currentAppBytes
        return MemoryAdmission(availableBytes, additionalBytes, reserveBytes, systemLowMemory, allowed,
            currentAppBytes, appBudgetBytes, effectiveBudget)
    }
}

internal fun saturatedMemorySum(vararg values: Long): Long = values.fold(0L) { total, value ->
    require(value >= 0)
    if (total > Long.MAX_VALUE - value) Long.MAX_VALUE else total + value
}
