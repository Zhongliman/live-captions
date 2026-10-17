package com.asr.live.service

import org.junit.Assert.*
import org.junit.Test

class StartupFinalBufferTest {
    private class FakeClock(private var nanos: Long = 0) {
        fun nowNs() = nanos
        fun advanceMs(ms: Long) { nanos += ms * 1_000_000L }
        fun advanceNs(ns: Long) { nanos += ns }
    }

    /** Mirrors the real defect: recognized finals that arrive while Hy-MT2 is loading must never
     * be silently and permanently dropped, only bounded and reported. */
    @Test fun fiveFinalsArrivingWhileLoadingAreAllRetainedUnderCapacity() {
        val buffer = StartupFinalBuffer<String>(capacity = 16, maxAgeNs = 15_000_000_000L)
        val dropped = (1..5).mapNotNull { buffer.offerReportingStale("final-$it").first }
        assertTrue("no capacity drop with 5 items under a capacity of 16", dropped.isEmpty())
        assertEquals(5, buffer.snapshot().buffered)
        assertEquals((1..5).map { "final-$it" }, generateSequence { buffer.poll() }.toList())
        assertTrue(buffer.isEmpty())
    }

    @Test fun overflowEvictsOldestFirstAndReportsIt() {
        val buffer = StartupFinalBuffer<Int>(capacity = 3, maxAgeNs = 15_000_000_000L)
        buffer.offer(1); buffer.offer(2); buffer.offer(3)
        val dropped = buffer.offer(4)
        assertEquals("oldest item evicted, not newest", 1, dropped)
        assertEquals(listOf(2, 3, 4), generateSequence { buffer.poll() }.toList())
        assertEquals(1, buffer.snapshot().dropped)
    }

    @Test fun entriesOlderThanMaxAgeAreEvictedAndReportedAsStale() {
        val clock = FakeClock()
        val buffer = StartupFinalBuffer<String>(capacity = 16, maxAgeNs = 10_000_000_000L, nowNs = clock::nowNs)
        buffer.offer("stale")
        clock.advanceMs(10_001)
        val (_, stale) = buffer.offerReportingStale("fresh")
        assertEquals(listOf("stale"), stale)
        assertEquals(listOf("fresh"), generateSequence { buffer.poll() }.toList())
        assertEquals(1, buffer.snapshot().dropped)
    }

    @Test fun snapshotReportsOldestAgeOfWhatIsStillBuffered() {
        val clock = FakeClock()
        val buffer = StartupFinalBuffer<String>(nowNs = clock::nowNs)
        buffer.offer("a")
        clock.advanceMs(500)
        buffer.offer("b")
        assertEquals(500, buffer.snapshot().oldestAgeMs)
    }

    @Test fun drainAllReturnsEverythingQueuedRightNowRegardlessOfAge() {
        // drainAll() is the session-end path (Stop's translation-abandon step, hard cancel):
        // the owner settles every returned request, including work past its age limit.
        val clock = FakeClock()
        val buffer = StartupFinalBuffer<String>(maxAgeNs = 1_000_000_000L, nowNs = clock::nowNs)
        buffer.offer("a")
        clock.advanceMs(5_000)
        assertEquals(listOf("a"), buffer.drainAll())
        assertTrue(buffer.isEmpty())
        assertEquals(0, buffer.snapshot().dropped)
        assertTrue(buffer.evictExpired().isEmpty())
    }

    @Test fun observationalQueriesNeverConsumeExpiredRequests() {
        val clock = FakeClock()
        val buffer = StartupFinalBuffer<String>(maxAgeNs = 1_000_000_000L, nowNs = clock::nowNs)
        buffer.offer("a")
        clock.advanceMs(5_000)
        repeat(3) {
            assertFalse(buffer.isEmpty())
            assertEquals(1, buffer.size())
            assertEquals(StartupBufferSnapshot(buffered = 1, dropped = 0, oldestAgeMs = 5_000), buffer.snapshot())
        }
        assertEquals(listOf("a"), buffer.evictExpired())
        assertEquals(StartupBufferSnapshot(buffered = 0, dropped = 1, oldestAgeMs = 0), buffer.snapshot())
    }

    @Test fun snapshotThenOfferStillReportsExpiredRequests() {
        val clock = FakeClock()
        val buffer = StartupFinalBuffer<String>(maxAgeNs = 1_000_000_000L, nowNs = clock::nowNs)
        buffer.offer("expired")
        clock.advanceMs(1_001)
        buffer.snapshot()
        val (capacityDrop, stale) = buffer.offerReportingStale("fresh")
        assertNull(capacityDrop)
        assertEquals(listOf("expired"), stale)
        assertEquals(listOf("fresh"), buffer.drainAll())
        assertEquals(1, buffer.snapshot().dropped)
        assertTrue(buffer.evictExpired().isEmpty())
    }

    @Test fun snapshotThenReportingPollReturnsExpiredRequestsAndNextFreshRequest() {
        val clock = FakeClock()
        val buffer = StartupFinalBuffer<String>(maxAgeNs = 1_000_000_000L, nowNs = clock::nowNs)
        buffer.offer("expired")
        clock.advanceMs(900)
        buffer.offer("fresh")
        clock.advanceMs(101)
        buffer.snapshot()
        val (next, stale) = buffer.pollReportingStale()
        assertEquals("fresh", next)
        assertEquals(listOf("expired"), stale)
        assertEquals(null to emptyList<String>(), buffer.pollReportingStale())
        assertEquals(1, buffer.snapshot().dropped)
        assertTrue(buffer.drainAll().isEmpty())
    }

    @Test fun reportingPollReturnsExpiredRequestsEvenWithoutLiveWork() {
        val clock = FakeClock()
        val buffer = StartupFinalBuffer<String>(maxAgeNs = 1_000_000_000L, nowNs = clock::nowNs)
        buffer.offer("expired")
        clock.advanceMs(1_001)
        assertEquals(null to listOf("expired"), buffer.pollReportingStale())
        assertEquals(null to emptyList<String>(), buffer.pollReportingStale())
        assertEquals(1, buffer.snapshot().dropped)
    }

    @Test fun simpleOfferAndPollNeverSilentlyDiscardExpiredRequests() {
        val clock = FakeClock()
        val buffer = StartupFinalBuffer<String>(capacity = 2, maxAgeNs = 1_000_000_000L, nowNs = clock::nowNs)
        buffer.offer("a")
        clock.advanceMs(5_000)
        assertNull(buffer.offer("b"))
        assertEquals("a", buffer.poll())
        assertEquals("b", buffer.poll())
        assertEquals(0, buffer.snapshot().dropped)
    }

    @Test fun capacityAndAgeEvictionsAreEachReportedAndCountedOnce() {
        val clock = FakeClock()
        val buffer = StartupFinalBuffer<String>(capacity = 2, maxAgeNs = 1_000_000_000L, nowNs = clock::nowNs)
        buffer.offer("capacity-drop")
        clock.advanceMs(500)
        buffer.offer("age-drop")
        clock.advanceMs(500)
        assertEquals("capacity-drop" to emptyList<String>(), buffer.offerReportingStale("fresh"))
        clock.advanceMs(501)
        assertEquals(StartupBufferSnapshot(buffered = 2, dropped = 1, oldestAgeMs = 1_001), buffer.snapshot())
        assertEquals("fresh" to listOf("age-drop"), buffer.pollReportingStale())
        assertTrue(buffer.evictExpired().isEmpty())
        assertTrue(buffer.drainAll().isEmpty())
        assertEquals(2, buffer.snapshot().dropped)
    }

    @Test fun ageEvictionLeavesCapacityForNewRequest() {
        val clock = FakeClock()
        val buffer = StartupFinalBuffer<String>(capacity = 2, maxAgeNs = 1_000_000_000L, nowNs = clock::nowNs)
        buffer.offer("expired")
        clock.advanceMs(900)
        buffer.offer("live")
        clock.advanceMs(101)
        assertEquals(null to listOf("expired"), buffer.offerReportingStale("new"))
        assertEquals(listOf("live", "new"), buffer.drainAll())
        assertEquals(1, buffer.snapshot().dropped)
    }

    @Test fun expiryUsesStrictAgeBoundary() {
        val clock = FakeClock()
        val buffer = StartupFinalBuffer<String>(maxAgeNs = 1_000_000_000L, nowNs = clock::nowNs)
        buffer.offer("a")
        clock.advanceMs(1_000)
        assertTrue(buffer.evictExpired().isEmpty())
        clock.advanceNs(1)
        assertEquals(listOf("a"), buffer.evictExpired())
        assertTrue(buffer.evictExpired().isEmpty())
        assertEquals(1, buffer.snapshot().dropped)
    }

    @Test fun repeatedExpiryAndOverflowStayBoundedAndAccountForEveryRequest() {
        val clock = FakeClock()
        val buffer = StartupFinalBuffer<Int>(capacity = 3, maxAgeNs = 10_000_000L, nowNs = clock::nowNs)
        val evicted = mutableListOf<Int>()
        var capacityDrops = 0
        var ageDrops = 0
        (1..1_000).forEach { value ->
            val (capacityDrop, stale) = buffer.offerReportingStale(value)
            capacityDrop?.let { evicted += it; capacityDrops++ }
            evicted += stale
            ageDrops += stale.size
            assertTrue(buffer.size() <= 3)
            assertEquals(evicted.size, buffer.snapshot().dropped)
            clock.advanceMs(if (value % 10 == 0) 20 else 2)
        }
        clock.advanceMs(20)
        buffer.snapshot()
        evicted += buffer.evictExpired()
        assertEquals((1..1_000).toList(), evicted)
        assertTrue(capacityDrops > 0)
        assertTrue(ageDrops > 0)
        assertTrue(buffer.isEmpty())
        assertTrue(buffer.evictExpired().isEmpty())
        assertEquals(1_000, buffer.snapshot().dropped)
    }

    @Test(expected = IllegalArgumentException::class)
    fun zeroCapacityIsRejected() {
        StartupFinalBuffer<String>(capacity = 0)
    }

    @Test(expected = IllegalArgumentException::class)
    fun negativeMaxAgeIsRejected() {
        StartupFinalBuffer<String>(maxAgeNs = -1)
    }

    @Test fun deterministicChronologicalOrderOnceDrained() {
        val buffer = StartupFinalBuffer<Int>()
        (1..8).forEach(buffer::offer)
        val order = generateSequence { buffer.poll() }.toList()
        assertEquals((1..8).toList(), order)
    }
}
