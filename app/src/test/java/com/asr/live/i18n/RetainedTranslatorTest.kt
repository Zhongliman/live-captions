package com.asr.live.i18n

import com.asr.live.service.BenchmarkResourceGuard
import org.junit.Assert.*
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

class RetainedTranslatorTest {
    @Test fun nativeCleanupMustActuallyFinishBeforeAnotherModelCanStart() {
        val closeEntered = CountDownLatch(1)
        val finishClose = CountDownLatch(1)
        val released = AtomicInteger()
        val resources = BenchmarkResourceGuard { AutoCloseable { released.incrementAndGet() } }
        val translator = resources.createTranslator {
            object : LocalTranslator {
                override fun translate(text: String) = text
                override fun close() { closeEntered.countDown(); finishClose.await() }
            }
        }
        val cleanup = Thread { translator.close() }.apply { start() }
        try {
            assertTrue(closeEntered.await(1, TimeUnit.SECONDS))
            assertFalse(resources.isIdle())
            assertEquals(0, released.get())
            assertThrows(IllegalStateException::class.java) { resources.acquire() }
        } finally {
            finishClose.countDown()
            cleanup.join(1000)
        }
        assertFalse(cleanup.isAlive)
        assertTrue(resources.isIdle())
        assertEquals(1, released.get())
    }

    @Test fun failedCloseKeepsAdmissionReserved() {
        val resources = BenchmarkResourceGuard()
        val translator = resources.createTranslator {
            object : LocalTranslator {
                override fun translate(text: String) = text
                override fun close() { error("native cleanup failed") }
            }
        }
        assertThrows(IllegalStateException::class.java) { translator.close() }
        assertFalse(resources.isIdle())
        assertThrows(IllegalStateException::class.java) { resources.acquire() }
    }

    @Test fun failedConstructionReleasesItsReservation() {
        val resources = BenchmarkResourceGuard()
        assertThrows(IllegalStateException::class.java) {
            resources.createTranslator { error("load failed before a translator was returned") }
        }
        assertTrue(resources.isIdle())
        resources.acquire().close()
    }
}
