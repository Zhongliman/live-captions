package com.asr.live.pipeline

/** Word boundaries for spaced text, and character boundaries for unspaced Han text. */
internal object SourcePrefixBoundary {
    private const val PUNCTUATION = "。！？.!?,，、;:；："

    fun commonPrefixEnd(left: String, right: String): Int {
        var end = 0
        while (end < left.length && end < right.length) {
            val codePoint = left.codePointAt(end)
            if (codePoint != right.codePointAt(end)) break
            end += Character.charCount(codePoint)
        }
        return end
    }

    fun isBoundary(text: String, end: Int): Boolean {
        if (end !in 0..text.length) return false
        if (end > 0 && end < text.length && Character.isHighSurrogate(text[end - 1]) &&
            Character.isLowSurrogate(text[end])) return false
        if (end == 0 || end == text.length) return true
        val next = text.codePointAt(end)
        // A combining mark, variation selector or joiner still belongs to the preceding glyph.
        if (isMark(next) || next == 0x200D) return false
        val previous = text.codePointBefore(end)
        if (previous == 0x200D) return false
        var baseEnd = end
        while (baseEnd > 0 && isMark(text.codePointBefore(baseEnd)))
            baseEnd = text.offsetByCodePoints(baseEnd, -1)
        val previousBase = if (baseEnd > 0) text.codePointBefore(baseEnd) else previous
        return isSpace(previous) || isSpace(next) || isPunctuation(previous) || isPunctuation(next) ||
            isHan(previousBase) || isHan(next)
    }

    private fun isSpace(codePoint: Int) = Character.isWhitespace(codePoint) || Character.isSpaceChar(codePoint)
    private fun isPunctuation(codePoint: Int) = codePoint <= Char.MAX_VALUE.code && codePoint.toChar() in PUNCTUATION
    private fun isHan(codePoint: Int) = Character.UnicodeScript.of(codePoint) == Character.UnicodeScript.HAN
    private fun isMark(codePoint: Int) = when (Character.getType(codePoint)) {
        Character.NON_SPACING_MARK.toInt(), Character.COMBINING_SPACING_MARK.toInt(),
        Character.ENCLOSING_MARK.toInt() -> true
        else -> false
    }
}
