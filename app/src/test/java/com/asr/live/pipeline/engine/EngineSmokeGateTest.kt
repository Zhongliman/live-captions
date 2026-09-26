package com.asr.live.pipeline.engine

import com.asr.live.i18n.LocalTranslator
import com.asr.live.pipeline.Profile
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import org.junit.Assert.*
import org.junit.Test

class EngineSmokeGateTest {
    /** A well-behaved fake: translates, reports a backend, honours cancel(requestId), closes cleanly. */
    private class GoodFake(private val output: String, val closes: AtomicInteger = AtomicInteger()) : LocalTranslator {
        val cancelled = AtomicBoolean(false)
        override fun translate(text: String) = output
        override fun translate(text: String, requestId: Long): String {
            if (requestId != CANCEL_PROBE_ID) return output
            val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2)
            while (!cancelled.get() && System.nanoTime() < deadline) Thread.sleep(1)
            if (cancelled.get()) throw RuntimeException("cancelled")
            return output
        }
        override fun cancel(requestId: Long) { if (requestId == CANCEL_PROBE_ID) cancelled.set(true) }
        override val backend: String get() = "CPU"
        override fun close() { closes.incrementAndGet() }
        companion object { const val CANCEL_PROBE_ID = -424_242L }
    }

    private class EchoingFake : LocalTranslator {
        override fun translate(text: String) = text // bug: returns the source verbatim
        override fun close() {}
    }

    private class NeverStopsFake : LocalTranslator {
        val closeStarted = java.util.concurrent.CountDownLatch(1)
        override fun translate(text: String) = "translated output"
        override fun translate(text: String, requestId: Long): String {
            if (requestId != GoodFake.CANCEL_PROBE_ID) return "translated output"
            while (true) Thread.sleep(50)
        }
        override fun close() {
            closeStarted.countDown()
            CountDownLatch(1).await() // resembles NativeTranslator.close waiting on its context owner
        }
    }

    @Test fun aWellBehavedFakeQualifiesAndReportsItsBackend() {
        val fake = GoodFake("Vertaalde tekst naar het Engels zonder Chinese tekens")
        val outcome = EngineSmokeGate.run(Profile.DUTCH_ENGLISH, requestedOpenCl = false) { fake }
        val qualified = outcome as? SmokeOutcome.Qualified ?: fail("expected Qualified, got $outcome") as SmokeOutcome.Qualified
        assertEquals("CPU", qualified.actualBackend)
        assertFalse(qualified.backendMismatch)
        assertEquals(1, fake.closes.get())
    }

    @Test fun requestingOpenClButReportingCpuFallbackFailsQualification() {
        val fake = GoodFake("Vertaalde tekst zonder Chinese tekens")
        val outcome = EngineSmokeGate.run(Profile.DUTCH_ENGLISH, requestedOpenCl = true) { fake }
        val failed = outcome as? SmokeOutcome.Failed ?: fail("CPU fallback must fail OpenCL calibration") as SmokeOutcome.Failed
        assertTrue(failed.reason.contains("OpenCL was requested"))
    }

    @Test fun cpuFallbackLabelContainingOpenclIsNotMisclassifiedAsOpenclExecution() {
        val fake = object : LocalTranslator {
            override fun translate(text: String) = "Vertaalde tekst zonder Chinese tekens"
            override val backend get() = "CPU · Adreno OpenCL unavailable; fallback"
            override fun close() {}
        }
        val outcome = EngineSmokeGate.run(Profile.DUTCH_ENGLISH, requestedOpenCl = true) { fake }
        assertTrue("a fallback string containing the word OpenCL must remain a failure", outcome is SmokeOutcome.Failed)
    }

    @Test fun sdkManagedBackendIsNeverFlaggedAsAnOpenClMismatch() {
        val fake = object : LocalTranslator {
            override fun translate(text: String) = "Vertaalde tekst zonder Chinese tekens"
            override val backend get() = "SDK-managed"
            override fun close() {}
        }
        val outcome = EngineSmokeGate.run(Profile.DUTCH_ENGLISH, requestedOpenCl = true) { fake }
        assertFalse((outcome as SmokeOutcome.Qualified).backendMismatch)
    }

    @Test fun aTranslatorThatEchoesTheSourceFailsTheLanguageCheck() {
        val outcome = EngineSmokeGate.run(Profile.DUTCH_ENGLISH, requestedOpenCl = false) { EchoingFake() }
        val failed = outcome as? SmokeOutcome.Failed ?: fail("expected Failed, got $outcome") as SmokeOutcome.Failed
        assertTrue(failed.reason.contains("target language"))
    }

    @Test fun chineseOutputStillInSourceScriptFailsTheLanguageCheck() {
        val fake = GoodFake("这仍然是中文，翻译失败了")
        val outcome = EngineSmokeGate.run(Profile.CHINESE_ENGLISH, requestedOpenCl = false) { fake }
        assertTrue(outcome is SmokeOutcome.Failed)
    }

    @Test fun loadFailureIsReportedWithoutThrowingOutOfTheGate() {
        val outcome = EngineSmokeGate.run(Profile.DUTCH_ENGLISH, requestedOpenCl = false) {
            error("model file missing")
        }
        val failed = outcome as SmokeOutcome.Failed
        assertTrue(failed.reason.contains("load failed"))
    }

    /** A translator that ignores cancellation and a close waiting on native work stay bounded. */
    @Test fun cancellationTimeoutDoesNotWaitUnboundedlyForNativeClose() {
        val startedAt = System.nanoTime()
        val fake = NeverStopsFake()
        val outcome = EngineSmokeGate.run(Profile.DUTCH_ENGLISH, requestedOpenCl = false) { fake }
        val elapsedMs = (System.nanoTime() - startedAt) / 1_000_000L
        assertTrue(outcome is SmokeOutcome.Failed)
        assertTrue("cancellation and native close must be bounded, took ${elapsedMs}ms", elapsedMs < 8_000)
        assertTrue((outcome as SmokeOutcome.Failed).reason.contains("cancel"))
        assertTrue("cleanup runs off the calibration thread", fake.closeStarted.await(1, TimeUnit.SECONDS))
    }

    @Test fun cleanupFailureAfterAnOtherwiseSuccessfulTranslationIsSurfacedNotSwallowed() {
        val fake = object : LocalTranslator {
            override fun translate(text: String) = "Vertaalde tekst zonder Chinese tekens"
            override val backend get() = "CPU"
            override fun close() { throw RuntimeException("native handle already retired") }
        }
        val outcome = EngineSmokeGate.run(Profile.DUTCH_ENGLISH, requestedOpenCl = false) { fake }
        val failed = outcome as? SmokeOutcome.Failed ?: fail("expected Failed, got $outcome") as SmokeOutcome.Failed
        assertTrue(failed.reason.contains("cleanup"))
    }
}
