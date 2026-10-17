package com.asr.live.i18n

import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import org.junit.Assert.*
import org.junit.Test

class ModelLoadScopeTest {
    /** Stands in for the native registry token: records ordering and any unsafe release. */
    private class RecordingControl(override val handle: Long = 7) : ModelLoadControl {
        val cancels = AtomicInteger()
        val releases = AtomicInteger()
        val loading = AtomicBoolean(false)
        val releasedWhileLoading = AtomicBoolean(false)
        val cancelledAfterRelease = AtomicBoolean(false)
        @Volatile var currentPhase = "Waiting for native model load"
        override fun phase() = currentPhase
        override fun cancel() {
            cancels.incrementAndGet()
            if (releases.get() > 0) cancelledAfterRelease.set(true)
        }
        override fun release() {
            releases.incrementAndGet()
            if (loading.get()) releasedWhileLoading.set(true)
        }
    }

    @Test fun stopDuringModelLoadCancelsItWithoutReleasingTheTokenInUse() {
        val control = RecordingControl()
        val scope = ModelLoadScope(control)
        val loadStarted = CountDownLatch(1)
        val loadMayReturn = CountDownLatch(1)
        val aborted = AtomicBoolean(false)

        val loader = Thread {
            val handle = scope.beginLoad()
            control.loading.set(true)
            try {
                assertEquals(7L, handle)
                loadStarted.countDown()
                // Models llama_model_load_from_file: the progress callback aborts once cancelled.
                loadMayReturn.await()
                aborted.set(scope.isCancelled())
            } finally {
                control.loading.set(false)
                scope.endLoad()
            }
        }.apply { isDaemon = true }

        loader.start()
        assertTrue(loadStarted.await(1, TimeUnit.SECONDS))
        assertTrue(scope.isLoading())

        // Stop arrives while the multi-GB read is still in flight.
        scope.cancel()
        scope.release()
        assertEquals(1, control.cancels.get())
        assertFalse("the loader still owns the token", scope.isReleased())
        assertEquals(0, control.releases.get())

        loadMayReturn.countDown()
        loader.join(2_000)
        assertFalse(loader.isAlive)
        assertTrue("the load observed the cancellation", aborted.get())
        assertTrue("the token is retired once the load returns", scope.isReleased())
        assertEquals(1, control.releases.get())
        assertFalse(control.releasedWhileLoading.get())
        assertFalse(control.cancelledAfterRelease.get())
    }

    @Test fun cancelBeforeTheLoadStartsStillAbortsIt() {
        val control = RecordingControl()
        val scope = ModelLoadScope(control)

        scope.cancel()
        assertEquals(1, control.cancels.get())
        assertTrue(scope.isCancelled())

        // The token predates the model, so the loader can see the cancellation immediately.
        assertEquals(7L, scope.beginLoad())
        scope.endLoad()
        scope.release()
        assertTrue(scope.isReleased())
        assertEquals(1, control.releases.get())
    }

    @Test fun releaseIsIdempotentAndRejectsReuseOfTheToken() {
        val control = RecordingControl()
        val scope = ModelLoadScope(control)

        scope.release()
        scope.release()
        assertEquals("release must never double-free the native token", 1, control.releases.get())

        try {
            scope.beginLoad()
            fail("a released token must not be handed to another load")
        } catch (expected: IllegalStateException) {
            assertTrue(expected.message!!.contains("already released"))
        }
    }

    @Test fun cancelAfterReleaseIsSafeAndDoesNotTouchTheNativeToken() {
        val control = RecordingControl()
        val scope = ModelLoadScope(control)

        scope.beginLoad()
        scope.endLoad()
        scope.release()
        scope.cancel()

        assertEquals("a released token is never cancelled again", 0, control.cancels.get())
        assertFalse(control.cancelledAfterRelease.get())
        assertEquals(1, control.releases.get())
    }

    @Test fun stoppingAfterLoadDoesNotPoisonAResidentModelsFutureReload() {
        val control = RecordingControl()
        val scope = ModelLoadScope(control)
        scope.beginLoad()
        scope.endLoad()

        scope.cancel()

        assertTrue("session cancellation is still recorded", scope.isCancelled())
        assertEquals("the completed resident load token must remain usable", 0, control.cancels.get())
        scope.release()
        assertEquals(1, control.releases.get())
    }

    @Test fun phaseOnlyDescribesAnInFlightNativeLoad() {
        val control = RecordingControl()
        val scope = ModelLoadScope(control)
        assertNull(scope.phase())
        scope.beginLoad()
        control.currentPhase = "Reading Hy-MT2 weights: 42%"
        assertEquals(control.currentPhase, scope.phase())
        scope.endLoad()
        assertNull("warm-up must not keep showing native constructor progress", scope.phase())
        scope.release()
        assertNull(scope.phase())
    }

    @Test fun deferredReleaseKeepsTheLoadPhaseAvailableForDiagnosis() {
        val control = RecordingControl()
        val scope = ModelLoadScope(control)
        scope.beginLoad()
        control.currentPhase = "Initializing llama backends and OpenCL driver"
        scope.cancel()
        scope.release()
        assertEquals(control.currentPhase, scope.phase())
        scope.endLoad()
        assertNull(scope.phase())
        assertEquals(1, control.releases.get())
    }

    @Test fun cancellationRacingLoadCompletionNeverReleasesTwice() {
        repeat(200) {
            val control = RecordingControl()
            val scope = ModelLoadScope(control)
            val ready = CountDownLatch(2)
            val loader = Thread {
                scope.beginLoad()
                ready.countDown(); ready.await()
                scope.endLoad()
                scope.release()
            }.apply { isDaemon = true }
            val stopper = Thread {
                ready.countDown(); ready.await()
                scope.cancel()
                scope.release()
            }.apply { isDaemon = true }

            loader.start(); stopper.start()
            loader.join(2_000); stopper.join(2_000)
            assertFalse(loader.isAlive); assertFalse(stopper.isAlive)
            assertEquals(1, control.releases.get())
            assertFalse(control.releasedWhileLoading.get())
            assertFalse(control.cancelledAfterRelease.get())
        }
    }
}
