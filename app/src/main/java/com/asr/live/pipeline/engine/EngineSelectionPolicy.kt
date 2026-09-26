package com.asr.live.pipeline.engine

import com.asr.live.pipeline.Profile

/** Per-candidate Stage A + Stage B result, ready for the selection policy. */
data class CandidateEvaluation(
    val candidate: EngineCandidate,
    val smoke: SmokeOutcome,
    /** Mean chrF++ over the held-out corpus items; null when smoke failed (never got this far). */
    val chrfScore: Double? = null,
    val itemsScored: Int = 0,
    /** Diagnostic count only — see [SemanticChecks] for what this does and does not detect. */
    val negationMarkersDropped: Int = 0,
    val numbersMissing: Int = 0,
    /** Total held-out corpus items included in the score denominator. */
    val itemsAttempted: Int = itemsScored,
    /** A single failed output disqualifies the candidate for this calibration run. */
    val benchmarkFailure: String? = null,
)

enum class SelectionStatus { SELECTED, RETAINED, QUALIFIED_NOT_SELECTED, FAILED, UNSUPPORTED, NOT_TESTED }

data class CandidateReport(val candidate: EngineCandidate, val status: SelectionStatus, val detail: String)

data class SelectionOutcome(
    val profile: Profile,
    val reports: List<CandidateReport>,
    /** Null means: keep whatever configuration was already in effect for this profile. */
    val winner: EngineCandidate?,
    val reason: String,
    /** Verified execution backend for [winner], including a retained prior selection. */
    val selectedBackend: Backend? = null,
)

/**
 * Ordered-gate selection over Stage A/B results for one language pair.
 *
 * Every constant here is an explicit, versioned engineering target — not a claim about what is
 * objectively "good enough" for every use. [MIN_CHRF_QUALITY_THRESHOLD] in particular is a
 * starting point pending real held-out validation; see [BenchmarkCorpus] for why its scores are
 * not yet a quality certification.
 */
object EngineSelectionPolicy {
    const val POLICY_VERSION = "selection-policy-v2"
    const val MIN_CHRF_QUALITY_THRESHOLD = 30.0
    /** Candidates within this many chrF points of the best are treated as tied. */
    const val QUALITY_TIE_EPSILON = 1.0
    const val MIN_ITEMS_FOR_CONFIDENCE = 3

    fun select(profile: Profile, evaluations: List<CandidateEvaluation>, previousWinner: EngineCandidate?,
               previousBackend: Backend? = null): SelectionOutcome {
        val reports = mutableListOf<CandidateReport>()
        val qualified = mutableListOf<CandidateEvaluation>()

        for (eval in evaluations) {
            if (eval.benchmarkFailure != null) {
                reports += CandidateReport(eval.candidate, SelectionStatus.FAILED,
                    "Stage B failed (chrF++ ${eval.chrfScore?.let { "%.1f".format(it) } ?: "unavailable"}; " +
                    "${eval.itemsScored}/${eval.itemsAttempted} held-out items scored) " +
                        "benchmark item(s): ${eval.benchmarkFailure}")
                continue
            }
            when (val smoke = eval.smoke) {
                is SmokeOutcome.Unsupported -> reports += CandidateReport(eval.candidate, SelectionStatus.UNSUPPORTED, smoke.reason)
                is SmokeOutcome.Failed -> reports += CandidateReport(eval.candidate, SelectionStatus.FAILED, smoke.reason)
                is SmokeOutcome.Qualified -> {
                    val score = eval.chrfScore
                    if (score == null || eval.itemsScored == 0) {
                        reports += CandidateReport(eval.candidate, SelectionStatus.FAILED,
                            "passed Stage A but produced no Stage B quality score")
                    } else if (score < MIN_CHRF_QUALITY_THRESHOLD) {
                        reports += CandidateReport(eval.candidate, SelectionStatus.QUALIFIED_NOT_SELECTED,
                            "chrF++ ${"%.1f".format(score)} on ${eval.itemsScored} items, below the " +
                                "$MIN_CHRF_QUALITY_THRESHOLD quality threshold")
                    } else {
                        qualified += eval
                    }
                }
            }
        }
        EngineRegistry.blocked().forEach { reports += CandidateReport(it, SelectionStatus.UNSUPPORTED, it.blockerReason.orEmpty()) }

        if (qualified.isEmpty()) {
            val reason = if (previousWinner != null)
                "No candidate passed quality qualification for ${profile.label}; keeping the previous configuration (${previousWinner.displayName})."
            else "No candidate passed quality qualification for ${profile.label} and no previous configuration exists."
            val retainedReports = if (previousWinner == null) reports else {
                val existing = reports.indexOfFirst { it.candidate == previousWinner }
                val currentDetail = reports.getOrNull(existing)?.let {
                    " Current run: ${it.status} — ${it.detail}"
                }.orEmpty()
                val retained = CandidateReport(previousWinner, SelectionStatus.RETAINED,
                    "Retained the previously validated configuration because no candidate qualified this run." + currentDetail)
                if (existing >= 0) reports.toMutableList().also { it[existing] = retained }
                else reports + retained
            }
            return SelectionOutcome(profile, retainedReports, previousWinner, reason, previousBackend)
        }

        val best = qualified.maxOf { checkNotNull(it.chrfScore) }
        val tied = qualified.filter { checkNotNull(it.chrfScore) >= best - QUALITY_TIE_EPSILON }
        val ranked = tied.sortedWith(
            compareBy<CandidateEvaluation> { it.candidate.extendedTier }
                .thenBy { (it.smoke as SmokeOutcome.Qualified).loadMs + it.smoke.translateMs })
        val winnerEval = ranked.first()
        val winner = winnerEval.candidate

        for (eval in qualified) {
            val status = if (eval.candidate == winner) SelectionStatus.SELECTED else SelectionStatus.QUALIFIED_NOT_SELECTED
            val detail = if (eval.candidate == winner)
                "Best validated configuration among the tested candidates on this phone " +
                    "(chrF++ ${"%.1f".format(eval.chrfScore)} on ${eval.itemsScored} items)."
            else "Qualified (chrF++ ${"%.1f".format(eval.chrfScore)}) but not selected: " +
                if (tied.contains(eval)) "tied with the winner, which is simpler/faster" else "lower quality than the winner"
            reports += CandidateReport(eval.candidate, status, detail)
        }

        val confidenceCaveat = if (winnerEval.itemsScored < MIN_ITEMS_FOR_CONFIDENCE)
            " Based on only ${winnerEval.itemsScored} held-out item(s); treat as provisional." else ""
        return SelectionOutcome(profile, reports,
            winner, "Best validated configuration among the tested candidates on this phone.$confidenceCaveat",
            (winnerEval.smoke as SmokeOutcome.Qualified).backend)
    }
}
