package com.asr.live.service

import org.junit.Assert.*
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

class BenchmarkResourceGuardTest {
    @Test fun deferredCleanupKeepsTheBenchmarkGateAndRejectsAnotherModel() {
        val retained = AtomicInteger()
        val released = AtomicInteger()
        val resources = BenchmarkResourceGuard {
            retained.incrementAndGet()
            AutoCloseable { released.incrementAndGet() }
        }
        val workerOwnsModel = resources.acquire()
        assertFalse(resources.isIdle())
        assertThrows(IllegalStateException::class.java) { resources.acquire() }
        assertEquals(1, retained.get())
        assertEquals(0, released.get())
        val done = CountDownLatch(1)
        Thread { workerOwnsModel.close(); done.countDown() }.start()
        assertTrue(done.await(1, TimeUnit.SECONDS))
        assertTrue(resources.isIdle())
        resources.acquire().close()
        assertEquals(2, retained.get())
        assertEquals(2, released.get())
    }

    @Test fun duplicateCloseCannotReleaseAnotherWorkersOwnership() {
        val released = AtomicInteger()
        val resources = BenchmarkResourceGuard { AutoCloseable { released.incrementAndGet() } }
        val first = resources.acquire()
        first.close()
        val second = resources.acquire()
        first.close()
        assertFalse(resources.isIdle())
        assertEquals(1, released.get())
        second.close()
        assertTrue(resources.isIdle())
        assertEquals(2, released.get())
    }
}
