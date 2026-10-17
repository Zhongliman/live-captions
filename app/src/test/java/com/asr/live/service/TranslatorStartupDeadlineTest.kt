package com.asr.live.service

import org.junit.Assert.*
import org.junit.Test

class TranslatorStartupDeadlineTest {
    @Test fun lateInitializationCannotBecomeReadyAfterTimeout() {
        var now = 0L
        val deadline = TranslatorStartupDeadline(0, timeoutMs = 100, nowNs = { now })
        now = 100_000_000L
        assertTrue(deadline.timeoutDue())
        assertFalse(deadline.ready())
        assertFalse(deadline.timeoutDue())
    }

    @Test fun completedStartupCannotLaterTimeOut() {
        var now = 0L
        val deadline = TranslatorStartupDeadline(0, timeoutMs = 100, nowNs = { now })
        now = 99_000_000L
        assertTrue(deadline.ready())
        now = 200_000_000L
        assertFalse(deadline.timeoutDue())
    }

    @Test fun everyStartupPhaseSharesOneBudget() {
        var now = 0L
        val deadline = TranslatorStartupDeadline(0, timeoutMs = 100, nowNs = { now })
        now = 40_000_000L
        assertFalse(deadline.timeoutDue())
        now = 100_000_000L
        assertFalse(deadline.ready())
        assertTrue(deadline.timeoutDue())
    }
}
