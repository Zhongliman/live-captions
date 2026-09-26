package com.asr.live.service

import com.asr.live.i18n.LocalTranslator
import com.asr.live.i18n.ResidentTranslatorManager
import com.asr.live.i18n.TranslatorIdentity
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class CaptionServiceResidencyTest {
    private class FakeTranslator : LocalTranslator {
        var closes = 0
        override fun translate(text: String) = text
        override fun close() { closes++ }
    }

    @Test fun ordinaryServiceDestructionLeavesProcessResidentTranslationModelsAvailable() {
        val slot = "service-destroy-${System.identityHashCode(this)}"
        val identity = TranslatorIdentity("hymt2-Q8_0", "nl", "en", "", false, 256, 128)
        val engine = FakeTranslator()
        ResidentTranslatorManager.acquire(slot, identity, { engine })

        Robolectric.buildService(CaptionService::class.java).create().destroy()

        assertTrue(ResidentTranslatorManager.isResident(slot, identity))
        assertEquals(0, engine.closes)
        ResidentTranslatorManager.closeAll()
    }
}
