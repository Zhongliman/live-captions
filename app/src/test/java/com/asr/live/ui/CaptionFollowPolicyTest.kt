package com.asr.live.ui

import org.junit.Assert.*
import org.junit.Test

class CaptionFollowPolicyTest {
    @Test fun sourceAndTranslationLayoutGrowthFollowTheBottomWithoutARevisionChange() {
        val policy = CaptionFollowPolicy()
        // Updating text within the same row does not change item count or its source key.
        assertFalse(policy.shouldScroll(canScrollForward = false, scrolling = false, dragging = false, itemCount = 2))
        assertTrue(policy.shouldScroll(canScrollForward = true, scrolling = false, dragging = false, itemCount = 2))
        assertFalse(policy.shouldScroll(canScrollForward = false, scrolling = false, dragging = false, itemCount = 2))
    }

    @Test fun tallSingleCaptionCanFollowItsSeparateTail() {
        assertTrue(CaptionFollowPolicy().shouldScroll(
            canScrollForward = true, scrolling = false, dragging = false, itemCount = 2))
        assertFalse(CaptionFollowPolicy().shouldScroll(
            canScrollForward = true, scrolling = false, dragging = false, itemCount = 1))
    }

    @Test fun manualScrollbackStaysDetachedThroughNewCaptionsAndTextGrowth() {
        val policy = CaptionFollowPolicy().userScrolled(verticalDelta = 12f).settled(atBottom = false)
        assertFalse(policy.following)
        assertFalse(policy.shouldScroll(canScrollForward = true, scrolling = false, dragging = false, itemCount = 10))
        assertFalse(policy.shouldScroll(canScrollForward = true, scrolling = false, dragging = false, itemCount = 11))
    }

    @Test fun scrollingBackToBottomResumesFollowingAfterTheGesture() {
        var policy = CaptionFollowPolicy().userScrolled(verticalDelta = 12f)
        policy = policy.userScrolled(verticalDelta = -12f).settled(atBottom = false)
        assertFalse(policy.following)
        policy = policy.settled(atBottom = true)
        assertTrue(policy.following)
        assertTrue(policy.shouldScroll(canScrollForward = true, scrolling = false, dragging = false, itemCount = 10))
    }

    @Test fun growthDuringADragOrFlingDoesNotFightTheGesture() {
        val policy = CaptionFollowPolicy()
        assertFalse(policy.shouldScroll(canScrollForward = true, scrolling = false, dragging = true, itemCount = 10))
        assertFalse(policy.shouldScroll(canScrollForward = true, scrolling = true, dragging = false, itemCount = 10))
        assertTrue(policy.shouldScroll(canScrollForward = true, scrolling = false, dragging = false, itemCount = 10))
    }
}
