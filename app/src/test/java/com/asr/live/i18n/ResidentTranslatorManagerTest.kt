package com.asr.live.i18n

import java.util.Collections
import java.util.concurrent.Callable
import java.util.concurrent.CancellationException
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import org.junit.After
import org.junit.Assert.*
import org.junit.Test

class ResidentTranslatorManagerTest {
    private class FakeTranslator(val id: Int, private val closing: () -> Unit = {}) : LocalTranslator {
        val closed = AtomicInteger(0)
        override fun translate(text: String) = text
        override fun close() { closed.incrementAndGet(); closing() }
    }

    private val executor = Executors.newCachedThreadPool { runnable -> Thread(runnable).apply { isDaemon = true } }
    private val ownership = Collections.synchronizedList(mutableListOf<AutoCloseable>())

    @After fun cleanUp() {
        executor.shutdownNow()
        assertTrue("test workers must terminate", executor.awaitTermination(3, TimeUnit.SECONDS))
        ownership.asReversed().forEach { it.close() }
        ResidentTranslatorManager.closeAll()
    }

    private fun identity(bundle: String = "hymt2-Q8_0") = TranslatorIdentity(
        modelBundle = bundle, sourceLanguage = "nl", targetLanguage = "en", glossary = "",
        backendRequestsOpenCl = false, batch = 256, ubatch = 128)

    private fun slot(suffix: String = "hy") = "$suffix-${System.identityHashCode(this)}"
    private fun acquire(slot: String = slot(), identity: TranslatorIdentity = identity(),
                        create: () -> LocalTranslator = { FakeTranslator(1) },
                        reused: (Boolean) -> Unit = {}): ResidentTranslatorManager.Lease =
        ResidentTranslatorManager.acquireLease(slot, identity, create, reused).also { ownership.add(it) }

    private fun CountDownLatch.mustArrive() = assertTrue("worker did not reach test barrier", await(3, TimeUnit.SECONDS))
    private fun <T> submit(block: () -> T) = executor.submit(Callable { block() })

    @Test fun secondAcquireWithTheSameIdentityReusesTheInstanceAndSkipsTheLoad() {
        var created = 0
        val first = acquire(create = { FakeTranslator(++created) })
        var reused = false
        val second = acquire(create = { FakeTranslator(++created) }, reused = { reused = it })

        assertSame(first.translator, second.translator)
        assertEquals(1, created)
        assertTrue(reused)
    }

    @Test fun listenStopListenNeverReloadsTheGgufWhenConfigurationIsUnchanged() {
        var loads = 0
        repeat(3) {
            acquire(create = { FakeTranslator(++loads) }).close()
        }
        assertEquals(1, loads)
    }

    @Test fun aDifferentIdentityClosesTheIdleResidentBeforeBuildingTheNewOne() {
        val first = acquire()
        val translator = first.translator as FakeTranslator
        first.close()
        var reused = true
        val second = acquire(identity = identity("hymt2-Q4_K_M"), create = {
            assertEquals("old native context must close before replacement load", 1, translator.closed.get())
            FakeTranslator(2)
        }, reused = { reused = it })

        assertFalse(reused)
        assertNotSame(first.translator, second.translator)
    }

    @Test fun aSlowHyLoadDoesNotBlockAnUnrelatedOpusSlot() {
        val loading = CountDownLatch(1)
        val complete = CountDownLatch(1)
        val hy = submit {
            acquire(create = { loading.countDown(); complete.await(); FakeTranslator(1) })
        }
        loading.mustArrive()
        try {
            val opus = submit { acquire(slot("opus"), identity("opus")) }.get(3, TimeUnit.SECONDS)
            assertNotNull(opus.translator)
            assertFalse(hy.isDone)
        } finally { complete.countDown() }
        hy.get(3, TimeUnit.SECONDS)
    }

    @Test fun concurrentAcquiresForOneSlotShareExactlyOneLoad() {
        val loading = CountDownLatch(1)
        val complete = CountDownLatch(1)
        val secondStarted = CountDownLatch(1)
        val loads = AtomicInteger()
        val first = submit {
            acquire(create = { loads.incrementAndGet(); loading.countDown(); complete.await(); FakeTranslator(1) })
        }
        loading.mustArrive()
        val second = submit {
            secondStarted.countDown()
            acquire(create = { loads.incrementAndGet(); FakeTranslator(2) })
        }
        secondStarted.mustArrive()
        try {
            assertFalse("same-slot acquire must wait for publication", second.isDone)
        } finally { complete.countDown() }
        assertSame(first.get(3, TimeUnit.SECONDS).translator, second.get(3, TimeUnit.SECONDS).translator)
        assertEquals(1, loads.get())
    }

    @Test fun aDifferentIdentityWaitsForActiveOwnershipAndNativeClose() {
        val closeStarted = CountDownLatch(1)
        val closeComplete = CountDownLatch(1)
        val loadStarted = CountDownLatch(1)
        val first = acquire(create = { FakeTranslator(1) { closeStarted.countDown(); closeComplete.await() } })
        val replacement = submit {
            acquire(identity = identity("replacement"), create = { loadStarted.countDown(); FakeTranslator(2) })
        }
        assertFalse("a replacement cannot load while the old worker owns the model", loadStarted.await(100, TimeUnit.MILLISECONDS))
        first.close()
        closeStarted.mustArrive()
        try {
            assertFalse("a replacement cannot load until native close returns", loadStarted.await(100, TimeUnit.MILLISECONDS))
            // A blocked native close must not hold the manager lock either.
            submit { acquire(slot("opus")) }.get(3, TimeUnit.SECONDS)
        } finally { closeComplete.countDown() }
        replacement.get(3, TimeUnit.SECONDS)
    }

    @Test fun teardownNeverPublishesAnObsoleteLoadOrLoadsTheSameSlotTwiceAtOnce() {
        val loadStarted = CountDownLatch(1)
        val loadComplete = CountDownLatch(1)
        val replacementStarted = CountDownLatch(1)
        val obsolete = FakeTranslator(1)
        val initial = submit {
            try {
                acquire(create = { loadStarted.countDown(); loadComplete.await(); obsolete })
                false
            } catch (_: CancellationException) { true }
        }
        loadStarted.mustArrive()
        ResidentTranslatorManager.closeAll()
        assertFalse(ResidentTranslatorManager.isResident(slot(), identity()))
        val replacement = submit {
            acquire(create = {
                assertEquals("obsolete model must finish closing before reloading", 1, obsolete.closed.get())
                replacementStarted.countDown()
                FakeTranslator(2)
            })
        }
        try {
            assertFalse(replacementStarted.await(100, TimeUnit.MILLISECONDS))
        } finally { loadComplete.countDown() }
        assertTrue(initial.get(3, TimeUnit.SECONDS))
        assertNotSame(obsolete, replacement.get(3, TimeUnit.SECONDS).translator)
        assertEquals(1, obsolete.closed.get())
    }

    @Test fun idleTrimmingPreservesPendingLoadsAndEveryActiveOwner() {
        val loading = CountDownLatch(1)
        val complete = CountDownLatch(1)
        val pending = submit { acquire(create = { loading.countDown(); complete.await(); FakeTranslator(1) }) }
        loading.mustArrive()
        try {
            assertEquals(0, ResidentTranslatorManager.closeIdle())
            assertNull(ResidentTranslatorManager.tryAcquireExclusive())
        } finally { complete.countDown() }
        val first = pending.get(3, TimeUnit.SECONDS)
        val second = acquire()
        val translator = first.translator as FakeTranslator
        first.close()
        first.close() // lease release is idempotent
        assertEquals(0, ResidentTranslatorManager.closeIdle())
        assertEquals(0, translator.closed.get())
        second.close()
        assertEquals(1, ResidentTranslatorManager.closeIdle())
        assertEquals(1, translator.closed.get())
        assertEquals(0, ResidentTranslatorManager.closeIdle())
    }

    @Test fun closeAllDefersActiveDestructionAndClosesIdleModelsExactlyOnce() {
        val active = acquire()
        val idle = acquire(slot("opus"))
        idle.close()
        ResidentTranslatorManager.closeAll()
        ResidentTranslatorManager.closeAll()

        assertEquals(0, (active.translator as FakeTranslator).closed.get())
        assertEquals(1, (idle.translator as FakeTranslator).closed.get())
        assertFalse(ResidentTranslatorManager.isResident(slot(), identity()))
        active.close()
        active.close()
        assertEquals(1, active.translator.closed.get())
    }

    @Test fun failedLoadsAndFailingTelemetryReleaseTheirReservations() {
        val error = IllegalStateException("load failed")
        try {
            acquire(create = { throw error })
            fail("load failure must propagate")
        } catch (actual: IllegalStateException) { assertSame(error, actual) }
        val translator = FakeTranslator(1)
        try {
            acquire(create = { translator }, reused = { throw error })
            fail("telemetry failure must propagate")
        } catch (actual: IllegalStateException) { assertSame(error, actual) }
        assertEquals(1, ResidentTranslatorManager.closeIdle())
        assertEquals(1, translator.closed.get())
        acquire().close()
    }

    @Test fun callbacksDoNotHoldTheManagerLock() {
        acquire(reused = {
            // Waiting for another worker inside telemetry would deadlock with a global monitor.
            submit { acquire(slot("opus")) }.get(3, TimeUnit.SECONDS)
        })
    }

    @Test fun legacyIdentityScopedReleaseCannotReleaseAReplacement() {
        val legacySlot = slot()
        val first = ResidentTranslatorManager.acquire(legacySlot, identity(), { FakeTranslator(1) })
        ResidentTranslatorManager.release(legacySlot, first)
        val second = ResidentTranslatorManager.acquire(legacySlot, identity("replacement"), { FakeTranslator(2) })
        try {
            ResidentTranslatorManager.release(legacySlot, first)
            assertEquals(0, ResidentTranslatorManager.closeIdle())
            assertEquals(0, (second as FakeTranslator).closed.get())
        } finally { ResidentTranslatorManager.release(legacySlot, second) }
        assertEquals(1, ResidentTranslatorManager.closeIdle())
    }

    @Test fun forgetTransfersResponsibilityWithoutClosingTheResource() {
        val owner = acquire()
        val translator = owner.translator as FakeTranslator
        ResidentTranslatorManager.forget(slot(), identity("different"))
        assertTrue(ResidentTranslatorManager.isResident(slot(), identity()))
        ResidentTranslatorManager.forget(slot(), identity())
        owner.close()
        ResidentTranslatorManager.closeAll()
        assertFalse(ResidentTranslatorManager.isResident(slot(), identity()))
        assertEquals(0, translator.closed.get())
        translator.close() // ownership was explicitly transferred back to the caller
    }

    @Test fun benchmarkAdmissionEvictsIdleModelsAndRetainsExclusivityThroughCleanup() {
        val idle = acquire()
        idle.close()
        val session = checkNotNull(ResidentTranslatorManager.tryAcquireSession()).also { ownership.add(it) }
        assertNull(ResidentTranslatorManager.tryAcquireSession())
        assertNull(ResidentTranslatorManager.tryAcquireExclusive())
        session.close()

        val benchmark = checkNotNull(ResidentTranslatorManager.tryAcquireExclusive()).also { ownership.add(it) }
        assertEquals(1, (idle.translator as FakeTranslator).closed.get())
        assertNull(ResidentTranslatorManager.tryAcquireSession())
        val cleanup = benchmark.retain().also { ownership.add(it) }
        benchmark.close()
        assertNull(ResidentTranslatorManager.tryAcquireExclusive())
        assertNull(ResidentTranslatorManager.tryAcquireSession())
        val workerStarted = CountDownLatch(1)
        val loadStarted = CountDownLatch(1)
        val pending = submit {
            workerStarted.countDown()
            acquire(create = { loadStarted.countDown(); FakeTranslator(2) })
        }
        workerStarted.mustArrive()
        assertFalse("live loads must wait for delayed benchmark cleanup", loadStarted.await(100, TimeUnit.MILLISECONDS))
        cleanup.close()
        cleanup.close()
        pending.get(3, TimeUnit.SECONDS).close()
        checkNotNull(ResidentTranslatorManager.tryAcquireSession()).also { ownership.add(it) }.close()
    }

    @Test fun interruptedBenchmarkWaiterDoesNotLeakAdmission() {
        val benchmark = checkNotNull(ResidentTranslatorManager.tryAcquireExclusive()).also { ownership.add(it) }
        val started = CountDownLatch(1)
        val cancelled = CountDownLatch(1)
        val worker = Thread {
            started.countDown()
            try { acquire() }
            catch (_: InterruptedException) { cancelled.countDown() }
        }
        worker.start()
        started.mustArrive()
        worker.interrupt()
        cancelled.mustArrive()
        worker.join(3000)
        benchmark.close()
        checkNotNull(ResidentTranslatorManager.tryAcquireExclusive()).also { ownership.add(it) }.close()
    }
}
