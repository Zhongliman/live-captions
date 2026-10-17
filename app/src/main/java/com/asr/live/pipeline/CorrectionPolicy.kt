package com.asr.live.pipeline

import java.text.Normalizer
import java.util.Locale

/** A second hypothesis is optional work, never an authority or an accumulating queue. */
object CorrectionPolicy {
    const val MAX_LATENCY_MS = 3_000L
    private const val MAX_LATENCY_NS = MAX_LATENCY_MS * 1_000_000L

    fun admit(profile: Profile, audioSamples: Int, audioDepth: Int, finalDepth: Int,
              busy: Boolean, previousRtf: Double): Boolean = profile.correctionSupported &&
        audioSamples in 8_000..320_000 && audioDepth in 0..1 && finalDepth == 0 && !busy &&
        previousRtf in 0.0..0.5

    /**
     * Only small, optional spelling/formatting repairs may replace an already translated source.
     * Polarity markers and quantities must have identical ordered signatures in the source language.
     * Word insertion/deletion and broad paraphrases are rejected, especially on short utterances.
     *
     * These are conservative lexical guards, not a semantic or acoustic confidence classifier:
     * an unlisted polarity expression, a changed name, or even a one-letter word change can still
     * change meaning. Passing this check is not evidence that the second recognizer is more accurate.
     */
    fun accept(original: String, candidate: String, elapsedMs: Long, audioDepth: Int,
               sourceLanguage: String = "nl"): Boolean {
        val source = original.trim()
        val revised = candidate.trim()
        if (source.isEmpty() || revised.isEmpty() || revised == source ||
            elapsedMs !in 0..MAX_LATENCY_MS || audioDepth !in 0..1 ||
            revised.length < source.length / 2 || revised.length > maxOf(24, source.length * 2)) return false

        val language = sourceLanguage.lowercase(Locale.ROOT).substringBefore('-').substringBefore('_')
        val polarityWords = when (language) {
            "nl" -> dutchPolarity
            "en" -> englishPolarity
            else -> return false // No automatic revision without language-specific guards.
        }
        val before = tokens(source)
        val after = tokens(revised)
        if (before.isEmpty() || before.size != after.size) return false
        if (before.filter { it in polarityWords } != after.filter { it in polarityWords }) return false
        if (quantities(before, language) != quantities(after, language)) return false

        // Exact lexical agreement permits case, whitespace, and surrounding punctuation repairs.
        if (before == after) return true
        if (after.size >= 6 && after.toSet().size <= 2) return false

        // A one-word answer has too little context to justify replacing any of its words.
        if (before.size == 1) return false
        val changed = before.indices.filter { before[it] != after[it] }
        val budget = (before.size / 8).coerceIn(1, 2)
        if (changed.size > budget) return false
        val minimumLength = if (before.size <= 3) 6 else 5
        return changed.all { index ->
            val from = before[index]
            val to = after[index]
            minOf(from.length, to.length) >= minimumLength && oneLetterRepair(from, to)
        }
    }

    // Keep signs, decimal/time/date separators, currency, and attached units with digit tokens.
    // Whitespace after a sign is significant too: "5" must not silently become "- 5".
    private val tokenPattern = Regex(
        "(?:[+\\-−‐‑‒–—]\\s*)?(?:\\p{Sc}\\s*)?[\\p{L}\\p{M}]*\\p{N}+" +
            "(?:[.,:/\\-‐‑–—]\\p{N}+)*[\\p{L}\\p{M}]*(?:[%‰°])?" +
            "|[\\p{L}\\p{M}]+(?:['’][\\p{L}\\p{M}]+)*|[%‰°℃℉\\p{Sc}]"
    )
    private val digit = Regex("\\p{N}")
    private val spaces = Regex("\\s+")
    private val combiningMarks = Regex("\\p{M}+")

    private fun tokens(text: String): List<String> = tokenPattern.findAll(
        Normalizer.normalize(text, Normalizer.Form.NFC).lowercase(Locale.ROOT).replace('’', '\'')
    ).map { match ->
        val token = match.value.replace(spaces, "")
        // Missing ASR apostrophes in negatives retain the same polarity signature.
        if (token in negativeContractions) token.replace("'", "") else token
    }.toList()

    private val negativeContractions = setOf(
        "aren't", "can't", "couldn't", "didn't", "doesn't", "don't", "hadn't", "hasn't", "haven't",
        "isn't", "mustn't", "needn't", "shan't", "shouldn't", "wasn't", "weren't", "won't", "wouldn't",
        "ain't"
    )
    private val dutchPolarity = setOf(
        "niet", "geen", "nooit", "niemand", "niets", "niks", "nergens", "noch", "zonder", "evenmin",
        "nimmer", "amper", "nauwelijks", "zelden", "wel", "jawel", "ja", "nee", "neen"
    )
    private val englishPolarity = setOf(
        "not", "no", "never", "none", "nothing", "nobody", "nowhere", "neither", "nor", "without",
        "hardly", "barely", "scarcely", "rarely", "seldom", "yes", "yeah", "cannot"
    ) + negativeContractions.map { it.replace("'", "") }

    private val englishNumbers = setOf(
        "zero", "oh", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
        "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
        "hundred", "thousand", "million", "billion", "trillion", "half", "quarter", "dozen",
        "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
        "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth",
        "eighteenth", "nineteenth", "twentieth", "thirtieth", "fortieth", "fiftieth", "sixtieth",
        "seventieth", "eightieth", "ninetieth", "hundredth", "thousandth", "millionth", "billionth"
    )
    private val dutchNumberPart = "nul|een|twee|drie|vier|vijf|zes|zeven|acht|negen|tien|elf|twaalf|" +
        "dertien|veertien|vijftien|zestien|zeventien|achttien|negentien|twintig|dertig|veertig|" +
        "vijftig|zestig|zeventig|tachtig|negentig|honderd|duizend|miljoen|miljard|biljoen"
    private val dutchNumber = Regex("(?:$dutchNumberPart)(?:(?:en)?(?:$dutchNumberPart))*(?:ste|de)?")
    private val dutchOtherNumbers = setOf("eerste", "tweede", "derde", "half", "halve", "kwart", "dozijn")
    private val commonUnits = setOf(
        "mm", "cm", "dm", "m", "km", "mg", "g", "kg", "t", "ml", "cl", "dl", "l", "oz", "lb", "lbs",
        "ms", "s", "sec", "min", "h", "hz", "khz", "mhz", "ghz", "w", "kw", "mw", "kwh", "v", "a",
        "pa", "hpa", "bar", "mb", "gb", "tb", "kb", "mbps", "gbps", "c", "f", "k", "°", "℃", "℉",
        "celsius", "fahrenheit", "kelvin", "ft", "yd", "mi", "mph",
        "%", "‰", "eur", "usd", "gbp", "euro", "euros", "dollar", "dollars", "€", "$", "£", "¥"
    )
    private val dutchUnits = setOf(
        "millimeter", "millimeters", "centimeter", "centimeters", "meter", "meters", "kilometer",
        "kilometers", "milligram", "milligrammen", "gram", "grammen", "kilogram", "kilogrammen",
        "kilo", "kilos", "liter", "liters", "milliliter", "milliliters", "seconde", "seconden", "secondes",
        "minuut", "minuten", "uur", "uren", "dag", "dagen", "week", "weken", "maand", "maanden",
        "jaar", "jaren", "procent", "graden", "graad", "cent", "centen"
    )
    private val englishUnits = setOf(
        "millimeter", "millimeters", "millimetre", "millimetres", "centimeter", "centimeters",
        "centimetre", "centimetres", "meter", "meters", "metre", "metres", "kilometer", "kilometers",
        "kilometre", "kilometres", "in", "inch", "inches", "foot", "feet", "yard", "yards", "mile", "miles",
        "milligram", "milligrams", "gram", "grams", "kilogram", "kilograms", "pound", "pounds",
        "ounce", "ounces", "liter", "liters", "litre", "litres", "milliliter", "milliliters", "gallon",
        "gallons", "second", "seconds", "minute", "minutes", "hour", "hours", "day", "days", "week",
        "weeks", "month", "months", "year", "years", "percent", "degree", "degrees", "cent", "cents"
    )
    private val quantityPrefixes = setOf("plus", "minus", "negative", "positive", "negatief", "positief")

    private fun quantities(tokens: List<String>, language: String): List<List<String>> {
        val units = commonUnits + if (language == "nl") dutchUnits else englishUnits
        return tokens.indices.filter { index ->
            val token = tokens[index]
            if (digit.containsMatchIn(token)) true else if (language == "en") token in englishNumbers else {
                val plain = Normalizer.normalize(token, Normalizer.Form.NFD).replace(combiningMarks, "")
                token in dutchOtherNumbers || dutchNumber.matches(plain)
            }
        }.map { index ->
            listOf(
                tokens[index],
                tokens.getOrNull(index - 1)?.takeIf { it in quantityPrefixes || it in commonUnits } ?: "",
                tokens.getOrNull(index + 1)?.takeIf { it in units } ?: ""
            )
        }
    }

    /** One insertion, deletion, substitution, or adjacent transposition within one word. */
    private fun oneLetterRepair(from: String, to: String): Boolean {
        if (kotlin.math.abs(from.length - to.length) > 1) return false
        val first = from.indices.firstOrNull { it >= to.length || from[it] != to[it] }
            ?: return to.length == from.length + 1
        return when {
            from.length == to.length -> from.substring(first + 1) == to.substring(first + 1) ||
                (first + 1 < from.length && from[first] == to[first + 1] && from[first + 1] == to[first] &&
                    from.substring(first + 2) == to.substring(first + 2))
            from.length > to.length -> from.substring(first + 1) == to.substring(first)
            else -> from.substring(first) == to.substring(first + 1)
        }
    }

    /** Deadline is anchored at the endpoint, so queue wait consumes the same correction budget. */
    fun remainingNanos(enqueuedAtNs: Long, nowNs: Long): Long =
        (enqueuedAtNs + MAX_LATENCY_NS - nowNs).coerceAtLeast(0)
}
