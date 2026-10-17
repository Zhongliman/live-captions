package com.asr.live.service

/** One monotonic deadline covers verification, native loading and warm-up together. */
internal class TranslatorStartupDeadline(
    private val startedAtNs: Long,
    private val timeoutMs: Long = 60_000L,
    private val nowNs: () -> Long = System::nanoTime,
) {
    private var settled = false
    init { require(timeoutMs > 0 && timeoutMs <= Long.MAX_VALUE / 1_000_000L) }

    fun elapsedMs(): Long = ((nowNs() - startedAtNs).coerceAtLeast(0) / 1_000_000L)

    /** A timed-out initializer cannot subsequently announce READY. */
    @Synchronized fun ready(): Boolean {
        if (settled || elapsedMs() >= timeoutMs) return false
        settled = true
        return true
    }

    @Synchronized fun timeoutDue(): Boolean {
        if (settled || elapsedMs() < timeoutMs) return false
        settled = true
        return true
    }
}
