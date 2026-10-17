package com.asr.live.ui

internal data class CaptionViewport(
    val canScrollForward: Boolean,
    val scrolling: Boolean,
    val dragging: Boolean,
    val itemCount: Int,
)

/** Layout changes may move the live tail; only a user scroll toward older rows detaches it. */
data class CaptionFollowPolicy(val following: Boolean = true) {
    fun userScrolled(verticalDelta: Float): CaptionFollowPolicy =
        if (verticalDelta > 0f) copy(following = false) else this

    fun settled(atBottom: Boolean): CaptionFollowPolicy =
        if (atBottom) copy(following = true) else this

    fun shouldScroll(canScrollForward: Boolean, scrolling: Boolean, dragging: Boolean, itemCount: Int): Boolean =
        following && canScrollForward && !scrolling && !dragging && itemCount > 1
}
