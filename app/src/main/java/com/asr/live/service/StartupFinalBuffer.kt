package com.asr.live.service

/** One buffered item plus the time it was queued, for age-based eviction. */
internal data class Aged<T>(val value: T, val queuedAtNs: Long)

internal data class StartupBufferSnapshot(val buffered: Int, val dropped: Int, val oldestAgeMs: Long)

/**
 * Holds final translation requests recognized while Hy-MT2 is still loading.
 *
 * Capture starts as soon as ASR is ready, so endpoints can arrive well before the translation
 * model does. The steady-state `finals` mailbox is only 2 deep — sized for normal backpressure,
 * not for a multi-second model load — so routing loading-time finals through it would silently
 * and permanently drop recognized speech (`SegmentLedger` never revisits a SKIPPED caption).
 * This buffer exists solely to bridge that startup window: bounded by count and by age, with
 * every eviction reported rather than silently discarded. Observing the buffer never expires
 * work: the owning worker must call [evictExpired], [offerReportingStale], or
 * [pollReportingStale] and settle every returned eviction outside this buffer's lock.
 */
internal class StartupFinalBuffer<T>(
    private val capacity: Int = 16,
    private val maxAgeNs: Long = 15_000_000_000L,
    private val nowNs: () -> Long = System::nanoTime,
) {
    init {
        require(capacity > 0) { "capacity must be positive" }
        require(maxAgeNs >= 0) { "maxAgeNs must not be negative" }
    }

    private val items = ArrayDeque<Aged<T>>()
    private var droppedCount = 0

    /** Capacity-only admission. Age-based removal always requires an explicit reporting API. */
    @Synchronized fun offer(value: T): T? {
        val dropped = if (items.size >= capacity) items.removeFirst() else null
        if (dropped != null) droppedCount++
        items.addLast(Aged(value, nowNs()))
        return dropped?.value
    }

    /** FIFO removal without an age check. Use [pollReportingStale] to enforce the age limit. */
    @Synchronized fun poll(): T? = items.removeFirstOrNull()?.value

    @Synchronized fun isEmpty(): Boolean = items.isEmpty()
    @Synchronized fun size(): Int = items.size

    /** Includes expired work until an owner explicitly removes and settles it. */
    @Synchronized fun snapshot(): StartupBufferSnapshot {
        val oldest = items.firstOrNull()?.let { (nowNs() - it.queuedAtNs) / 1_000_000L } ?: 0L
        return StartupBufferSnapshot(items.size, droppedCount, oldest)
    }

    /** Everything still queued when the model finally becomes ready, or the session ends. */
    @Synchronized fun drainAll(): List<T> {
        val all = items.map { it.value }
        items.clear()
        return all
    }

    /** Returns every expired item exactly once, for the owning worker to settle. */
    @Synchronized fun evictExpired(): List<T> = evictStaleLocked(nowNs())

    private fun evictStaleLocked(now: Long): List<T> {
        val evicted = mutableListOf<T>()
        while (items.isNotEmpty() && now - items.first().queuedAtNs > maxAgeNs) {
            evicted += items.removeFirst().value
            droppedCount++
        }
        return evicted
    }

    /** Same as offer, but also returns anything the age check evicted in the same pass. */
    @Synchronized fun offerReportingStale(value: T): Pair<T?, List<T>> {
        val now = nowNs()
        val stale = evictStaleLocked(now)
        val dropped = if (items.size >= capacity) items.removeFirst() else null
        if (dropped != null) droppedCount++
        items.addLast(Aged(value, now))
        return dropped?.value to stale
    }

    /** Atomically expires old work and polls the next live item, reporting every expiration. */
    @Synchronized fun pollReportingStale(): Pair<T?, List<T>> {
        val stale = evictStaleLocked(nowNs())
        return items.removeFirstOrNull()?.value to stale
    }
}
