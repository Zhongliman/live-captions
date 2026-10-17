package com.asr.live

import com.asr.live.pipeline.*
import org.junit.Assert.*
import org.junit.Test

class CorrectionTest {
    @Test fun unsupportedAndOverloadedWorkIsRejected() {
        assertFalse(CorrectionPolicy.admit(Profile.CHINESE_ENGLISH, 16000, 0, 0, false, 0.0))
        assertFalse(CorrectionPolicy.admit(Profile.DUTCH_ENGLISH, 16000, 2, 0, false, 0.0))
        assertFalse(CorrectionPolicy.admit(Profile.DUTCH_ENGLISH, 16000, 0, 1, false, 0.0))
        assertFalse(CorrectionPolicy.admit(Profile.DUTCH_ENGLISH, 16000, 0, 0, true, 0.0))
        assertFalse(CorrectionPolicy.admit(Profile.DUTCH_ENGLISH, 16000, 0, 0, false, 0.6))
        assertFalse(CorrectionPolicy.admit(Profile.DUTCH_ENGLISH, 16000, -1, 0, false, 0.2))
        assertFalse(CorrectionPolicy.admit(Profile.DUTCH_ENGLISH, 16000, 0, 0, false, Double.NaN))
        assertFalse(CorrectionPolicy.admit(Profile.DUTCH_ENGLISH, 16000, 0, 0, false, -0.1))
        assertTrue(CorrectionPolicy.admit(Profile.DUTCH_ENGLISH, 16000, 0, 0, false, 0.2))
    }
    @Test fun secondHypothesisMustBeTimelyAndPlausible() {
        assertFalse(CorrectionPolicy.accept("Goedemorgen iedereen", "", 100, 0))
        assertFalse(CorrectionPolicy.accept("Goedemorgen iedereen", "Hallo iedereen", 3001, 0))
        assertFalse(CorrectionPolicy.accept("Hallo werelt", "Hallo wereld", 100, 2))
        assertFalse(CorrectionPolicy.accept("Hallo werelt", "Hallo wereld", -1, 0))
        assertTrue(CorrectionPolicy.accept("Hallo werelt", "Hallo wereld", 800, 0))
        assertTrue(CorrectionPolicy.accept("Hallo werelt", "Hallo wereld", 3000, 1))
        assertEquals(1_000_000_000L, CorrectionPolicy.remainingNanos(5_000_000_000L, 7_000_000_000L))
        assertEquals(0L, CorrectionPolicy.remainingNanos(5_000_000_000L, 8_000_000_000L))
    }
    @Test fun dutchPolarityCannotBeAddedRemovedOrSubstituted() {
        for ((original, candidate) in listOf(
            "Ik wil dit wel" to "Ik wil dit niet",
            "Ik wil dit niet" to "Ik wil dit",
            "Ik wil dit" to "Ik wil dit niet",
            "Er is geen probleem" to "Er is een probleem",
            "Dit gebeurt nooit meer" to "Dit gebeurt ooit meer",
            "Er is niemand aanwezig" to "Er is iemand aanwezig",
            "Ik hoorde niets buiten" to "Ik hoorde iets buiten",
            "We zijn nauwelijks begonnen" to "We zijn namelijk begonnen"
        )) {
            assertFalse("$original → $candidate", CorrectionPolicy.accept(original, candidate, 100, 0))
        }
        assertTrue(CorrectionPolicy.accept("IK WIL DIT NIET", "Ik wil dit niet.", 100, 0))
    }
    @Test fun englishPolarityUsesTheEnglishProfileAndPreservesContractions() {
        for ((original, candidate) in listOf(
            "I do not agree" to "I do now agree",
            "I can hardly believe it" to "I can hardly not believe it",
            "There is nobody outside" to "There is somebody outside",
            "It is rarely useful" to "It is nearly useful",
            "We can't accept this offer" to "We can accept this offer",
            "We won't accept this offer" to "We want accept this offer"
        )) {
            assertFalse("$original → $candidate", CorrectionPolicy.accept(original, candidate, 100, 0, "en"))
        }
        assertTrue(CorrectionPolicy.accept("I DONT KNOW", "I don't know.", 100, 0, "en-US"))
        assertTrue(CorrectionPolicy.accept("We can’t accept this offer", "We can't accept this offer.", 100, 0, "en"))
        assertTrue(CorrectionPolicy.accept("Everything is goiing really well", "Everything is going really well", 100, 0, "en"))
        assertFalse(CorrectionPolicy.accept("Hallo werelt", "Hallo wereld", 100, 0, "zh"))
    }
    @Test fun digitChangesIncludingSignsAndNumericPunctuationAreRejected() {
        for ((original, candidate) in listOf(
            "Neem 5 tabletten" to "Neem 50 tabletten",
            "De afstand is 1,5 km" to "De afstand is 15 km",
            "Het bedrag is 5 euro" to "Het bedrag is - 5 euro",
            "Het bedrag is 5 euro" to "Het bedrag is −5 euro",
            "Het bedrag is €5" to "Het bedrag is €50",
            "We beginnen om 12:30" to "We beginnen om 12:50",
            "De datum is 12/10" to "De datum is 10/12",
            "De groei was 5%" to "De groei was 5‰",
            "Versie B12 werkt prima" to "Versie B13 werkt prima"
        )) {
            assertFalse("$original → $candidate", CorrectionPolicy.accept(original, candidate, 100, 0))
        }
        assertTrue(CorrectionPolicy.accept("de afstand is 1,5 km", "De afstand is 1,5 km.", 100, 0))
    }
    @Test fun writtenNumbersAndAdjacentUnitsAreProtectedInBothLanguages() {
        for ((original, candidate, language) in listOf(
            Triple("Neem vijf tabletten", "Neem vijftig tabletten", "nl"),
            Triple("Het duurt tweeëntwintig minuten", "Het duurt tweeëndertig minuten", "nl"),
            Triple("Het duurt honderdvijf minuten", "Het duurt honderdzes minuten", "nl"),
            Triple("De afstand is 5 km", "De afstand is 5 cm", "nl"),
            Triple("De dosis is vijf milligram", "De dosis is vijf kilogram", "nl"),
            Triple("De termijn is vijf maanden", "De termijn is vijf manden", "nl"),
            Triple("Please bring five boxes", "Please bring fifty boxes", "en"),
            Triple("The first reading was useful", "The third reading was useful", "en"),
            Triple("The dose is five milligrams", "The dose is five kilograms", "en"),
            Triple("The load is 5 kg", "The load is 5 mg", "en")
        )) {
            assertFalse("$original → $candidate", CorrectionPolicy.accept(original, candidate, 100, 0, language))
        }
        assertTrue(CorrectionPolicy.accept("NEEM VIJF TABLETTEN", "Neem vijf tabletten.", 100, 0))
        assertTrue(CorrectionPolicy.accept("Please bring twenty one boxes", "Please bring twenty-one boxes.", 100, 0, "en"))
    }
    @Test fun onlySmallSpellingRepairsHaveEnoughLexicalAgreement() {
        assertTrue(CorrectionPolicy.accept("Hallo werelt", "Hallo wereld!", 100, 0))
        assertTrue(CorrectionPolicy.accept("We verwachten morgen hetzelfde berciht", "We verwachten morgen hetzelfde bericht", 100, 0))
        assertTrue(CorrectionPolicy.accept("Het resultaatt van deze controle verschijnt morgen", "Het resultaat van deze controle verschijnt morgen", 100, 0))
        assertFalse(CorrectionPolicy.accept("Goedemorgen iedereen", "Goedemorgen allemaal", 100, 0))
        assertFalse(CorrectionPolicy.accept("Luisteren", "Fluisteren", 100, 0))
        assertFalse(CorrectionPolicy.accept("I can walk", "I can talk", 100, 0, "en"))
        assertFalse(CorrectionPolicy.accept("Wij verwachten bericht", "Wij verwachten ander bericht", 100, 0))
        assertFalse(CorrectionPolicy.accept("Wij verwachten morgen hetzelfde bericht", "Wij verwachten morgen bericht", 100, 0))
        assertFalse(CorrectionPolicy.accept("We verwachten mrogen hetzelfde berciht", "We verwachten morgen hetzelfde bericht", 100, 0))
        assertFalse(CorrectionPolicy.accept("We verwachten morgen hetzelfde bericht", "We verwachten morgen een antwoord", 100, 0))
        assertTrue(CorrectionPolicy.accept(
            "We verwachten morgen hetzelfde berciht en willen daarna alle betrokken deelnemers een volledige samenvattng toesturen voor vertrek",
            "We verwachten morgen hetzelfde bericht en willen daarna alle betrokken deelnemers een volledige samenvatting toesturen voor vertrek",
            100, 0))
    }
    @Test fun formattingRepairsAndUnchangedProtectedValuesRemainEligible() {
        assertTrue(CorrectionPolicy.accept("hallo", "Hallo!", 100, 0))
        assertTrue(CorrectionPolicy.accept("we betalen 5 euro voor dit berciht", "We betalen 5 euro voor dit bericht.", 100, 0))
        assertTrue(CorrectionPolicy.accept("ik verwacht niet hetzelfde berciht", "Ik verwacht niet hetzelfde bericht.", 100, 0, "NL"))
        assertFalse(CorrectionPolicy.accept("Hallo wereld", "Hallo wereld", 100, 0))
        assertFalse(CorrectionPolicy.accept("", "Hallo wereld", 100, 0))
        assertFalse(CorrectionPolicy.accept("Hallo wereld", "!!!", 100, 0))
    }
    @Test fun correctionPublishesRevisedSourceAndHyTranslationAtomically() {
        val ledger = SegmentLedger(); ledger.start(1)
        val row = ledger.source(1, "Hallo werelt", true, 0)!!
        ledger.translate(row.key, "Hello", 2, true)
        val committed = ledger.snapshot().single()
        val revised = ledger.reviseTranslated(row.key, "Hallo wereld", "Hello world")!!
        assertEquals(row.key.id, revised.key.id)
        assertEquals("Hallo werelt", committed.source)
        assertEquals("Hello", committed.translation)
        assertEquals("Hallo wereld", revised.source)
        assertEquals("Hello world", revised.translation)
        assertEquals(revised.key.revision, revised.translatedRevision)
        assertEquals(CaptionStage.REVISED, revised.stage)
        assertFalse(ledger.translate(row.key, "Old result", 2, true))
        assertNull(ledger.reviseTranslated(row.key, "Late correction", "Too late"))
    }
    @Test fun skippedAndDiscontinuousSegmentsCannotBeRevived() {
        val ledger = SegmentLedger(); ledger.start(1)
        val row = ledger.source(1, "Hallo", false, 0)!!
        ledger.discontinuity(1)
        assertFalse(ledger.current(row.key))
        assertFalse(ledger.translate(row.key, "Late", 0, false))
        assertNull(ledger.reviseTranslated(row.key, "Late correction", "Too late"))
    }
}
