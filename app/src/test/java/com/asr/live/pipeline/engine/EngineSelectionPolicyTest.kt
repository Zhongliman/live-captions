package com.asr.live.pipeline.engine

import com.asr.live.pipeline.Profile
import com.asr.live.pipeline.TranslationQuality
import org.junit.Assert.*
import org.junit.Test

class EngineSelectionPolicyTest {
    private fun candidate(id: String, quality: TranslationQuality? = TranslationQuality.HY_Q8, extended: Boolean = false) =
        EngineCandidate(id, EngineFamily.HY_MT2, id, "test", quality, setOf(Profile.DUTCH_ENGLISH),
            "GGUF", "llama.cpp (bundled)", setOf(Backend.CPU), "n/a", QualificationTier.CANDIDATE, extended)

    private fun qualified(id: String, chrf: Double, loadMs: Long = 100, translateMs: Long = 50,
                          extended: Boolean = false, items: Int = 4) = CandidateEvaluation(
        candidate(id, extended = extended), SmokeOutcome.Qualified("CPU", loadMs, translateMs, false), chrf, items)

    @Test fun theHighestQualityQualifiedCandidateIsSelected() {
        val evaluations = listOf(qualified("low", 40.0), qualified("high", 80.0), qualified("mid", 60.0))
        val outcome = EngineSelectionPolicy.select(Profile.DUTCH_ENGLISH, evaluations, previousWinner = null)
        assertEquals("high", outcome.winner?.engineId)
        assertTrue(outcome.reason.contains("Best validated configuration"))
    }

    @Test fun opusCanWinAsACpuPrimaryEngineWithoutATranslationQualityEnum() {
        val opus = EngineCandidate("opus-nl-en", EngineFamily.OPUS, "OPUS", "75 MB", null,
            setOf(Profile.DUTCH_ENGLISH), "ONNX", "ONNX Runtime", setOf(Backend.CPU),
            "CC-BY 4.0", QualificationTier.CANDIDATE)
        val outcome = EngineSelectionPolicy.select(Profile.DUTCH_ENGLISH,
            listOf(CandidateEvaluation(opus, SmokeOutcome.Qualified("CPU", 50, 20, false), 92.0, 4)), null)
        assertEquals("opus-nl-en", outcome.winner?.engineId)
        assertNull(outcome.winner?.quality)
        assertEquals(Backend.CPU, outcome.selectedBackend)
    }

    @Test fun aCandidateBelowTheQualityThresholdIsNeverSelectedEvenIfItIsTheOnlyOne() {
        val evaluations = listOf(qualified("weak", EngineSelectionPolicy.MIN_CHRF_QUALITY_THRESHOLD - 1.0))
        val outcome = EngineSelectionPolicy.select(Profile.DUTCH_ENGLISH, evaluations, previousWinner = null)
        assertNull(outcome.winner)
        assertTrue(outcome.reports.single { it.candidate.engineId == "weak" }.status == SelectionStatus.QUALIFIED_NOT_SELECTED)
    }

    @Test fun noQualifiedCandidateKeepsThePreviousConfigurationRatherThanPickingTheLeastBadFailure() {
        val previous = candidate("previous-known-good")
        val evaluations = listOf(CandidateEvaluation(candidate("broken"), SmokeOutcome.Failed("native crash")))
        val outcome = EngineSelectionPolicy.select(Profile.DUTCH_ENGLISH, evaluations, previous)
        assertEquals(previous, outcome.winner)
        assertTrue(outcome.reason.contains("keeping the previous configuration"))
    }

    @Test fun noQualifiedCandidateAndNoPreviousConfigurationLeavesNoWinnerRatherThanGuessing() {
        val evaluations = listOf(CandidateEvaluation(candidate("broken"), SmokeOutcome.Failed("native crash")))
        val outcome = EngineSelectionPolicy.select(Profile.DUTCH_ENGLISH, evaluations, previousWinner = null)
        assertNull(outcome.winner)
        assertTrue(outcome.reason.contains("no previous configuration exists"))
    }

    @Test fun effectivelyTiedCandidatesPreferTheFasterOne() {
        val evaluations = listOf(
            qualified("slow-but-marginally-better", 80.5, loadMs = 5000, translateMs = 400),
            qualified("fast-and-almost-as-good", 80.0, loadMs = 500, translateMs = 100),
        )
        val outcome = EngineSelectionPolicy.select(Profile.DUTCH_ENGLISH, evaluations, previousWinner = null)
        assertEquals("within the tie epsilon, speed must break the tie",
            "fast-and-almost-as-good", outcome.winner?.engineId)
    }

    @Test fun candidatesOutsideTheTieEpsilonAreNotTreatedAsTied() {
        val evaluations = listOf(
            qualified("clearly-better", 90.0, loadMs = 5000, translateMs = 400),
            qualified("clearly-worse-but-fast", 50.0, loadMs = 100, translateMs = 50),
        )
        val outcome = EngineSelectionPolicy.select(Profile.DUTCH_ENGLISH, evaluations, previousWinner = null)
        assertEquals("clearly-better", outcome.winner?.engineId)
    }

    @Test fun theExtendedTierLosesATieToTheStandardTierEvenIfSlightlyFaster() {
        val evaluations = listOf(
            qualified("extended-7b", 80.0, loadMs = 50, translateMs = 20, extended = true),
            qualified("standard-1.8b", 79.5, loadMs = 200, translateMs = 80, extended = false),
        )
        val outcome = EngineSelectionPolicy.select(Profile.DUTCH_ENGLISH, evaluations, previousWinner = null)
        assertEquals("standard-1.8b", outcome.winner?.engineId)
    }

    @Test fun blockedResearchCandidatesAlwaysAppearInTheReportAsUnsupported() {
        val outcome = EngineSelectionPolicy.select(Profile.DUTCH_ENGLISH, emptyList(), previousWinner = null)
        val blockedIds = EngineRegistry.blocked().map { it.engineId }.toSet()
        val reportedIds = outcome.reports.filter { it.status == SelectionStatus.UNSUPPORTED }.map { it.candidate.engineId }.toSet()
        assertEquals("no failed candidate may be silently omitted from the report", blockedIds, reportedIds)
    }

    @Test fun aSmokeFailureIsReportedAsFailedNotSilentlyDropped() {
        val evaluations = listOf(CandidateEvaluation(candidate("crashes"), SmokeOutcome.Failed("load failed: out of memory")))
        val outcome = EngineSelectionPolicy.select(Profile.DUTCH_ENGLISH, evaluations, previousWinner = null)
        val report = outcome.reports.single { it.candidate.engineId == "crashes" }
        assertEquals(SelectionStatus.FAILED, report.status)
        assertTrue(report.detail.contains("out of memory"))
    }

    @Test fun aQualifiedCandidateWithNoQualityScoreIsReportedAsFailedNotSilentlySkipped() {
        val evaluations = listOf(CandidateEvaluation(candidate("no-score"),
            SmokeOutcome.Qualified("CPU", 10, 10, false), chrfScore = null, itemsScored = 0))
        val outcome = EngineSelectionPolicy.select(Profile.DUTCH_ENGLISH, evaluations, previousWinner = null)
        val report = outcome.reports.single { it.candidate.engineId == "no-score" }
        assertEquals(SelectionStatus.FAILED, report.status)
    }
}
