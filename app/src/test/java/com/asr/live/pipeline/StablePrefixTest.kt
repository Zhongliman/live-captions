package com.asr.live.pipeline

import org.junit.Assert.*
import org.junit.Test

class StablePrefixTest {
    @Test fun unpunctuatedMandarinAdvancesAcrossConsecutiveHypotheses() {
        val stable = StablePrefix()
        assertEquals("", stable.accept("今天我们去公园"))
        assertEquals("今天我们去公园", stable.accept("今天我们去公园散步"))
        assertEquals("今天我们去公园散步", stable.accept("今天我们去公园散步然后回家"))
    }

    @Test fun earlierMandarinEditInvalidatesTheCommittedPrefix() {
        val stable = StablePrefix()
        stable.accept("今天我们去公园")
        stable.accept("今天我们去公园散步")
        val revised = "今天他们去公园散步"
        val prefix = stable.accept(revised)
        assertEquals("今天", prefix)
        assertTrue(revised.startsWith(prefix))
    }

    @Test fun mixedMandarinAndLatinKeepsTheLatinWordWhole() {
        val stable = StablePrefix()
        stable.accept("今天使用trans")
        assertEquals("今天使用", stable.accept("今天使用translation模型"))
        assertEquals("今天使用translation模型", stable.accept("今天使用translation模型进行测试"))
    }

    @Test fun supplementaryHanCharactersAreNeverSplitAtTheirSharedSurrogate() {
        val stable = StablePrefix()
        stable.accept("今天\uD840\uDC00")
        assertEquals("今天", stable.accept("今天\uD840\uDC01继续"))
        assertEquals("今天\uD840\uDC01继续", stable.accept("今天\uD840\uDC01继续学习"))
    }

    @Test fun aChangingVariationSelectorDoesNotCommitPartOfTheGlyph() {
        val stable = StablePrefix()
        stable.accept("甲\uFE00乙")
        assertEquals("", stable.accept("甲\uFE01乙丙"))
    }

    @Test fun dutchStillRequiresCompleteWordsAndInvalidatesAnExtendedWord() {
        val stable = StablePrefix()
        assertEquals("", stable.accept("Hallo wer"))
        assertEquals("Hallo", stable.accept("Hallo wereld"))
        assertEquals("Hallo wereld", stable.accept("Hallo wereld morgen"))
        assertEquals("Hallo", stable.accept("Hallo wereldreis morgen"))
    }

    @Test fun aTruncatedHypothesisDoesNotTurnHalfADutchWordIntoStableText() {
        val stable = StablePrefix()
        stable.accept("Hallo wereld vandaag")
        assertEquals("Hallo", stable.accept("Hallo wer"))
    }

    @Test fun mandarinTranslationsSurviveAppendOnlyGrowthButRejectEarlierEdits() {
        val ledger = SegmentLedger()
        ledger.start(42)
        ledger.source(42, "今天我们去公园", false, 1)
        val first = ledger.source(42, "今天我们去公园散步", false, 2)!!
        val portion = TranslationPortionIdentity(42, first.key.id, first.stableSource)
        val grown = ledger.source(42, "今天我们去公园散步然后回家", false, 3)!!
        assertTrue(grown.key.revision > first.key.revision)
        assertTrue(ledger.translatePortion(first.key, portion, "Today we go to the park", 0, false, 5))
        assertEquals(portion.source, ledger.snapshot().single().translatedSource)
        assertEquals("Today we go to the park", ledger.snapshot().single().translation)

        val edited = ledger.source(42, "今天他们去公园散步然后回家", false, 4)!!
        assertEquals("今天", edited.stableSource)
        assertEquals("", edited.translation)
        assertFalse(ledger.canTranslatePortion(first.key, portion))
        assertFalse(ledger.translatePortion(first.key, portion, "Stale result", 0, false, 6))
    }
}
