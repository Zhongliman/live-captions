package com.asr.live.i18n

import com.asr.live.service.BenchmarkResourceGuard
import java.util.concurrent.atomic.AtomicBoolean

/** Keeps benchmark admission reserved until native close has actually completed. */
internal fun BenchmarkResourceGuard.createTranslator(create: () -> LocalTranslator): LocalTranslator {
    val ownership = acquire()
    val translator = try { create() }
    catch (t: Throwable) { ownership.close(); throw t }
    return object : LocalTranslator by translator {
        private val closing = AtomicBoolean(false)
        override fun close() {
            if (closing.compareAndSet(false, true)) {
                // If close throws or never returns, memory may still be in use. The retained
                // lease deliberately keeps sessions and subsequent benchmark loads excluded.
                translator.close()
                ownership.close()
            }
        }
    }
}
