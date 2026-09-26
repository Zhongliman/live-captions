package com.asr.live.pipeline.engine

import com.asr.live.pipeline.Profile
import com.asr.live.pipeline.TranslationQuality
import android.content.SharedPreferences
import java.lang.reflect.Proxy
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment

internal fun testSharedPreferences(readFailure: Boolean = false, commitResult: Boolean = true): SharedPreferences {
    lateinit var editor: SharedPreferences.Editor
    editor = Proxy.newProxyInstance(SharedPreferences.Editor::class.java.classLoader,
        arrayOf(SharedPreferences.Editor::class.java)) { _, method, _ ->
        when (method.name) {
            "putString" -> editor
            "commit" -> commitResult
            else -> null
        }
    } as SharedPreferences.Editor
    return Proxy.newProxyInstance(SharedPreferences::class.java.classLoader,
        arrayOf(SharedPreferences::class.java)) { _, method, _ ->
        when (method.name) {
            "getString" -> if (readFailure) error("simulated preference read failure") else null
            "edit" -> editor
            else -> null
        }
    } as SharedPreferences
}

@RunWith(RobolectricTestRunner::class)
class CalibrationStoreTest {
    private fun record(profile: Profile = Profile.DUTCH_ENGLISH, quality: TranslationQuality? = TranslationQuality.HY_Q8,
                       policyVersion: String = EngineSelectionPolicy.POLICY_VERSION,
                       corpusVersion: String = BenchmarkCorpus.VERSION,
                       fingerprint: DeviceFingerprint = DeviceFingerprint.current()) = CalibrationRecord(
        profile, "hy-mt2-hy_q8", quality, 82.5, fingerprint, policyVersion, corpusVersion, 1_000L)

    @Test fun savedSelectionRoundTripsThroughPersistence() {
        val store = CalibrationStore(RuntimeEnvironment.getApplication())
        store.save(record())
        val loaded = store.get(Profile.DUTCH_ENGLISH)
        assertEquals("hy-mt2-hy_q8", loaded?.engineId)
        assertEquals(TranslationQuality.HY_Q8, loaded?.quality)
        assertEquals(82.5, loaded?.chrfScore!!, 1e-9)
    }

    @Test fun eachLanguagePairIsPersistedIndependently() {
        val store = CalibrationStore(RuntimeEnvironment.getApplication())
        store.save(record(Profile.DUTCH_ENGLISH, TranslationQuality.HY_Q8))
        store.save(record(Profile.ENGLISH_FRENCH, TranslationQuality.HY_Q4))
        assertEquals(TranslationQuality.HY_Q8, store.get(Profile.DUTCH_ENGLISH)?.quality)
        assertEquals(TranslationQuality.HY_Q4, store.get(Profile.ENGLISH_FRENCH)?.quality)
        assertNull(store.get(Profile.CHINESE_ENGLISH))
    }

    @Test fun aRecordFromAnOlderPolicyVersionIsTreatedAsStaleNotSilentlyReused() {
        val store = CalibrationStore(RuntimeEnvironment.getApplication())
        store.save(record(policyVersion = "selection-policy-v0-ancient"))
        assertNotNull("the raw record is still readable", store.get(Profile.DUTCH_ENGLISH))
        assertNull("but must not be surfaced as fresh", store.getFresh(Profile.DUTCH_ENGLISH))
        assertNull(store.selectedQuality(Profile.DUTCH_ENGLISH))
    }

    @Test fun aRecordFromADifferentDeviceFingerprintIsStale() {
        val store = CalibrationStore(RuntimeEnvironment.getApplication())
        val otherDevice = DeviceFingerprint.current().copy(soc = "SM_SOME_OTHER_CHIP")
        store.save(record(fingerprint = otherDevice))
        assertNull(store.getFresh(Profile.DUTCH_ENGLISH))
    }

    @Test fun clearRemovesOnlyTheNamedProfile() {
        val store = CalibrationStore(RuntimeEnvironment.getApplication())
        store.save(record(Profile.DUTCH_ENGLISH))
        store.save(record(Profile.ENGLISH_FRENCH))
        store.clear(Profile.DUTCH_ENGLISH)
        assertNull(store.get(Profile.DUTCH_ENGLISH))
        assertNotNull(store.get(Profile.ENGLISH_FRENCH))
    }

    @Test fun opusWinnersPersistWithANullQualityRatherThanCrashing() {
        val store = CalibrationStore(RuntimeEnvironment.getApplication())
        store.save(record(quality = null))
        val loaded = store.get(Profile.DUTCH_ENGLISH)
        assertNotNull(loaded)
        assertNull(loaded?.quality)
    }

    @Test fun aPreferenceReadFailureMakesThePriorSelectionUnavailableWithoutThrowing() {
        val store = CalibrationStore(testSharedPreferences(readFailure = true))
        assertNull(store.getFresh(Profile.DUTCH_ENGLISH))
    }

    @Test fun aFailedPreferenceCommitIsReturnedToTheCaller() {
        val store = CalibrationStore(testSharedPreferences(commitResult = false))
        assertFalse(store.save(record()))
    }
}
