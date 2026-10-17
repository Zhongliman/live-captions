package com.asr.live.service

import java.util.concurrent.atomic.AtomicBoolean

/**
 * Keeps isolated benchmark ownership through deferred native cleanup. A timed-out worker may
 * still own a model, so subsequent candidates must wait for its actual close before allocating.
 */
class BenchmarkResourceGuard(private val retainResources: () -> AutoCloseable = { AutoCloseable {} }) {
    private var active = false

    @Synchronized fun acquire(): AutoCloseable {
        check(!active) { "The previous benchmark model is still in use or closing; wait for its worker to finish" }
        val retained = retainResources()
        active = true
        val released = AtomicBoolean(false)
        return AutoCloseable {
            if (released.compareAndSet(false, true)) {
                synchronized(this) {
                    retained.close()
                    active = false
                }
            }
        }
    }

    @Synchronized fun isIdle(): Boolean = !active
}
