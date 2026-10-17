package com.asr.live.pipeline

enum class CaptionStage { PROVISIONAL, FINAL, REVISED, SKIPPED, CANCELLED }
data class SegmentKey(val session: Long, val id: Long, val revision: Int)

/** Identity of the exact source span sent to a translator, independent of later segment revisions. */
data class TranslationPortionIdentity(val session: Long, val segmentId: Long, val source: String)

data class TranslationResultIdentity(val portion: TranslationPortionIdentity, val requestId: Long)

data class Caption(
    val key: SegmentKey,
    val source: String,
    val stableSource: String = "",
    val translation: String = "",
    val stage: CaptionStage = CaptionStage.PROVISIONAL,
    val endpointAtMs: Long? = null,
    val translatedRevision: Int = -1,
    val qualityRank: Int = -1,
    val detail: String = "",
    val corrected: Boolean = false,
    val translatedSource: String = "",
    val translationPortion: TranslationPortionIdentity? = null,
    val translationRequestId: Long = 0,
)

/** Consecutive hypotheses must agree on complete words, or complete Han characters. */
class StablePrefix {
    private var previous = ""
    private var committed = ""
    fun accept(hypothesis: String): String {
        val text = hypothesis.trim()
        val common = SourcePrefixBoundary.commonPrefixEnd(previous, text)
        var end = common
        while (end > 0 && (!SourcePrefixBoundary.isBoundary(text, end) ||
                !SourcePrefixBoundary.isBoundary(previous, end)))
            end = text.offsetByCodePoints(end, -1)
        val candidate = text.take(end).trimEnd()
        // A revised earlier word invalidates the old prefix; never fabricate a hybrid sentence.
        committed = if (text.startsWith(committed) && SourcePrefixBoundary.isBoundary(text, committed.length) &&
            candidate.length < committed.length) committed else candidate
        previous = text
        return committed
    }
    fun reset() { previous = ""; committed = "" }
}

/** Session and exact revision matching prevents old native work from updating a new session. */
class SegmentLedger(private val maxCaptions: Int = 150) {
    private var session = 0L
    private var nextId = 1L
    private var active: Long? = null
    private val captions = linkedMapOf<Long, Caption>()
    private val stable = StablePrefix()
    @Synchronized fun start(generation: Long) {
        cancel()
        session = generation
    }
    @Synchronized fun source(generation: Long, text: String, endpoint: Boolean, nowMs: Long): Caption? {
        if (generation != session || text.isBlank()) return null
        val id = active ?: nextId++.also { active = it }
        val old = captions[id]
        val source = text.trim()
        if (old?.source == source && !endpoint) return old
        val stableSource = if (endpoint) source else stable.accept(source)
        // A translation belongs to the source span it covered, not to this whole revision.
        // Keep it across append-only growth; an edit inside that span invalidates it.
        val advancesRevision = old == null || endpoint || old.stableSource != stableSource
        val key = if (advancesRevision) SegmentKey(session, id, (old?.key?.revision ?: -1) + 1) else checkNotNull(old).key
        val oldPortion = old?.translationPortion
        val keepTranslation = old != null && oldPortion != null &&
            sourceStillHasPrefix(stableSource, oldPortion.source)
        val row = Caption(key, source, stableSource,
            translation = if (keepTranslation) old!!.translation else "", endpointAtMs = if (endpoint) nowMs else null,
            translatedRevision = if (keepTranslation) old!!.translatedRevision else -1,
            qualityRank = if (keepTranslation) old!!.qualityRank else -1,
            detail = if (endpoint) "Awaiting final translation" else if (keepTranslation && old!!.translatedSource != stableSource)
                "Provisional · live suffix" else "Live hypothesis",
            translatedSource = if (keepTranslation) old!!.translatedSource else "",
            translationPortion = if (keepTranslation) oldPortion else null,
            translationRequestId = if (keepTranslation) old!!.translationRequestId else 0,
        )
        captions[id] = row
        if (endpoint) { active = null; stable.reset() }
        while (captions.size > maxCaptions) captions.remove(captions.keys.first())
        return row
    }
    @Synchronized fun translate(key: SegmentKey, text: String, rank: Int, final: Boolean): Boolean {
        val old = captions[key.id] ?: return false
        if (old.key != key) return false
        val source = if (final) old.source else old.stableSource
        if (source.isBlank()) return false
        return translatePortion(key, TranslationPortionIdentity(key.session, key.id, source), text, rank, final)
    }

    /** Append-only source growth is compatible with an in-flight translation of its prefix. */
    @Synchronized fun canTranslatePortion(
        key: SegmentKey,
        portion: TranslationPortionIdentity,
        final: Boolean = false,
    ): Boolean {
        val old = captions[key.id] ?: return false
        if (key.session != session || portion.session != session || portion.segmentId != key.id ||
            key.revision > old.key.revision || old.stage in setOf(CaptionStage.CANCELLED, CaptionStage.SKIPPED)) return false
        val currentSource = if (old.endpointAtMs != null) old.source else old.stableSource
        if (final) {
            if (old.endpointAtMs == null || old.key != key || portion.source != old.source) return false
        } else if (!sourceStillHasPrefix(currentSource, portion.source)) return false

        val currentPortion = old.translationPortion
        if (currentPortion != null) {
            // Results may advance coverage, never roll it back or jump across an internal edit.
            if (!sourceStillHasPrefix(portion.source, currentPortion.source)) return false
        }
        return true
    }

    @Synchronized fun translatePortion(
        key: SegmentKey,
        portion: TranslationPortionIdentity,
        text: String,
        rank: Int,
        final: Boolean,
        requestId: Long = 0,
    ): Boolean {
        val old = captions[key.id] ?: return false
        if (text.isBlank() || rank < old.qualityRank || !canTranslatePortion(key, portion, final)) return false
        captions[key.id] = old.copy(translation = text, qualityRank = rank, translatedRevision = key.revision,
            translatedSource = portion.source, translationPortion = portion, translationRequestId = requestId,
            stage = if (final && old.corrected) CaptionStage.REVISED else if (final) CaptionStage.FINAL else CaptionStage.PROVISIONAL,
            detail = if (final && old.corrected) "Revised · Parakeet second hypothesis" else if (final) "Final"
                else if (portion.source != old.stableSource) "Provisional · live suffix" else "Provisional")
        return true
    }

    @Synchronized fun reviseTranslated(key: SegmentKey, source: String, translation: String, requestId: Long = 0): Caption? {
        val old = captions[key.id] ?: return null
        if (!current(key) || old.endpointAtMs == null || source.isBlank() || translation.isBlank() || old.source == source) return null
        val revisedKey = key.copy(revision = key.revision + 1)
        val portion = TranslationPortionIdentity(key.session, key.id, source)
        val row = old.copy(key = revisedKey, source = source, stableSource = source,
            translation = translation, translatedRevision = revisedKey.revision, qualityRank = 2,
            stage = CaptionStage.REVISED, corrected = true,
            detail = "Revised · Parakeet second hypothesis", translatedSource = source,
            translationPortion = portion, translationRequestId = requestId)
        captions[key.id] = row
        return row
    }
    @Synchronized fun skip(key: SegmentKey, reason: String) {
        val old = captions[key.id] ?: return
        if (old.key == key && key.session == session && old.stage == CaptionStage.PROVISIONAL)
            captions[key.id] = old.copy(stage = CaptionStage.SKIPPED, detail = reason)
    }
    @Synchronized fun current(key: SegmentKey) = key.session == session && captions[key.id]?.key == key &&
        captions[key.id]?.stage !in setOf(CaptionStage.CANCELLED, CaptionStage.SKIPPED)
    @Synchronized fun snapshot() = captions.values.toList()
    @Synchronized fun caption(key: SegmentKey): Caption? = captions[key.id]?.takeIf { it.key == key }
    @Synchronized fun discontinuity(generation: Long) {
        if (generation != session) return
        active?.let { id -> captions[id]?.let { row ->
            captions[id] = row.copy(stage = CaptionStage.SKIPPED, detail = "Audio gap; recognition restarted")
        } }
        active = null; stable.reset()
    }
    @Synchronized fun cancel() {
        captions.replaceAll { _, c -> if (c.stage == CaptionStage.PROVISIONAL) c.copy(stage = CaptionStage.CANCELLED, detail = "Stopped") else c }
        active = null
        stable.reset()
        session = -1
    }
    @Synchronized fun clear() { captions.clear(); active = null; stable.reset() }

    private fun sourceStillHasPrefix(current: String, portion: String): Boolean {
        if (portion.isBlank() || !current.startsWith(portion)) return false
        return SourcePrefixBoundary.isBoundary(current, portion.length)
    }
}
