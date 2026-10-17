package com.asr.live.i18n

import java.util.concurrent.CancellationException
import java.util.concurrent.CountDownLatch
import java.util.concurrent.atomic.AtomicBoolean

/** Configuration identity for one resident translator slot. */
data class TranslatorIdentity(
    val modelBundle: String,
    val sourceLanguage: String,
    val targetLanguage: String,
    val glossary: String,
    val backendRequestsOpenCl: Boolean,
    val batch: Int,
    val ubatch: Int,
)

/**
 * Keeps models resident between sessions. State changes use [lock], while native creation,
 * destruction, waiting, and telemetry callbacks always run outside it. Each slot has its own
 * completion signal, so loading Hy never prevents OPUS from becoming ready.
 *
 * A worker must retain its [Lease] until its final native call has returned. Idle trimming then
 * needs no inference from UI state, which may already say stopped while a worker is unwinding.
 */
object ResidentTranslatorManager {
    class Lease internal constructor(val translator: LocalTranslator, private val release: () -> Unit) : AutoCloseable {
        private val closed = AtomicBoolean()
        override fun close() { if (closed.compareAndSet(false, true)) release() }
    }

    /** Retained references keep benchmark admission exclusive through delayed native cleanup. */
    class ExclusiveLease internal constructor(
        private val release: () -> Unit,
        private val retainLease: () -> AutoCloseable,
    ) : AutoCloseable {
        private var closed = false
        @Synchronized fun retain(): AutoCloseable {
            check(!closed) { "Benchmark lease is already closed" }
            return retainLease()
        }
        @Synchronized override fun close() {
            if (!closed) { closed = true; release() }
        }
    }

    private class Slot {
        var resident: Resident? = null
        var loading = false
        var closing = 0
        var changed = CountDownLatch(1)
    }
    private class Resident(val identity: TranslatorIdentity, val translator: LocalTranslator, val slot: Slot) {
        var owners = 0
        var retired = false
        var closeOnRelease = true
        var closeClaimed = false
    }
    private class ExclusiveGate {
        var references = 1
        var changed = CountDownLatch(1)
    }
    private sealed interface Acquisition {
        class Ready(val lease: Lease) : Acquisition
        class Wait(val changed: CountDownLatch) : Acquisition
        class Load(val state: Slot, val stale: Resident?) : Acquisition
    }

    private val lock = Any()
    private val slots = mutableMapOf<String, Slot>()
    private val legacyLeases = mutableMapOf<String, MutableList<Lease>>()
    private var generation = 0L
    private var activeOwners = 0
    private var pendingAcquires = 0
    private var pendingLoads = 0
    private var pendingCloses = 0
    private var sessionOwner: Any? = null
    private var exclusive: ExclusiveGate? = null

    /**
     * Acquires active ownership before returning a model. Different identities in the same slot
     * wait for previous owners to release; identical identities share one load. Waiting is
     * interruptible. A teardown invalidates acquisitions that had not yet published their model.
     */
    fun acquireLease(slot: String, identity: TranslatorIdentity, create: () -> LocalTranslator,
                     reused: (Boolean) -> Unit = {}): Lease {
        val expectedGeneration = synchronized(lock) { pendingAcquires++; generation }
        try {
            while (true) {
                val acquisition = synchronized(lock) {
                    if (generation != expectedGeneration) throw CancellationException("Resident model acquisition was evicted")
                    val gate = exclusive
                    if (gate != null) Acquisition.Wait(gate.changed)
                    else {
                        val state = slots.getOrPut(slot) { Slot() }
                        val current = state.resident
                        when {
                            state.loading || state.closing > 0 -> Acquisition.Wait(state.changed)
                            current != null && !current.retired && current.identity == identity -> Acquisition.Ready(lease(current))
                            current != null && current.owners > 0 -> Acquisition.Wait(state.changed)
                            else -> {
                                state.resident = null
                                current?.retired = true
                                val stale = current?.let { claimClose(it) }
                                state.loading = true
                                pendingLoads++
                                Acquisition.Load(state, stale)
                            }
                        }
                    }
                }
                when (acquisition) {
                    is Acquisition.Wait -> acquisition.changed.await()
                    is Acquisition.Ready -> return notifyReused(acquisition.lease, true, reused)
                    is Acquisition.Load -> {
                        acquisition.stale?.let { closeTranslators(listOf(it)) }
                        val translator = try {
                            synchronized(lock) {
                                if (generation != expectedGeneration)
                                    throw CancellationException("Resident model acquisition was evicted")
                            }
                            create()
                        }
                        catch (t: Throwable) {
                            synchronized(lock) {
                                acquisition.state.loading = false
                                signal(acquisition.state)
                                pendingLoads--
                            }
                            throw t
                        }
                        var obsolete: Resident? = null
                        val loaded = synchronized(lock) {
                            acquisition.state.loading = false
                            pendingLoads--
                            signal(acquisition.state)
                            if (generation == expectedGeneration && slots[slot] === acquisition.state) {
                                Resident(identity, translator, acquisition.state).also { acquisition.state.resident = it }
                                    .let { lease(it) }
                            } else {
                                // closeAll won the race. Never publish this obsolete load, even
                                // if another acquisition has already installed a replacement.
                                obsolete = claimClose(Resident(identity, translator, acquisition.state))
                                null
                            }
                        }
                        if (loaded != null) return notifyReused(loaded, false, reused)
                        closeTranslators(listOf(checkNotNull(obsolete)))
                        throw CancellationException("Resident model load completed after teardown")
                    }
                }
            }
        } finally {
            synchronized(lock) { pendingAcquires-- }
        }
    }

    /** Compatibility API: every successful acquire must have one matching release. */
    fun acquire(slot: String, identity: TranslatorIdentity, create: () -> LocalTranslator,
                reused: (Boolean) -> Unit = {}): LocalTranslator {
        val lease = acquireLease(slot, identity, create, reused)
        synchronized(lock) { legacyLeases.getOrPut(slot) { mutableListOf() }.add(lease) }
        return lease.translator
    }

    /** Releases this acquisition only; an old worker cannot release a replacement model. */
    fun release(slot: String, translator: LocalTranslator) {
        val lease = synchronized(lock) {
            val leases = legacyLeases[slot] ?: return
            val index = leases.indexOfLast { it.translator === translator }
            if (index < 0) return
            leases.removeAt(index).also { if (leases.isEmpty()) legacyLeases.remove(slot) }
        }
        lease.close()
    }

    /** Legacy single-owner API; new callers should retain and close an explicit [Lease]. */
    fun release(slot: String) {
        val lease = synchronized(lock) {
            val leases = legacyLeases[slot] ?: return
            leases.removeAt(leases.lastIndex).also { if (leases.isEmpty()) legacyLeases.remove(slot) }
        }
        lease.close()
    }

    /**
     * Transfers resource responsibility to the caller without closing it. Existing ownership
     * still needs releasing. Failed loads are already removed automatically and need no forget.
     */
    fun forget(slot: String, identity: TranslatorIdentity) {
        synchronized(lock) {
            val state = slots[slot] ?: return
            val current = state.resident ?: return
            if (current.identity != identity) return
            current.retired = true
            current.closeOnRelease = false
            state.resident = null
            signal(state)
        }
    }

    fun isResident(slot: String, identity: TranslatorIdentity): Boolean = synchronized(lock) {
        slots[slot]?.resident?.let { !it.retired && it.identity == identity } == true
    }

    /** Closes only unowned models; pending worker loads and active models are always preserved. */
    fun closeIdle(): Int {
        val closing = synchronized(lock) {
            val result = mutableListOf<Resident>()
            val iterator = slots.iterator()
            while (iterator.hasNext()) {
                val state = iterator.next().value
                val current = state.resident
                if (!state.loading && state.closing == 0 && current != null && current.owners == 0) {
                    current.retired = true
                    claimClose(current)?.let { result.add(it) }
                    signal(state)
                }
            }
            result
        }
        closeTranslators(closing)
        return closing.size
    }

    /**
     * Explicit teardown: invalidates pending loads and defers active model destruction until its
     * last lease closes. Memory pressure should use [closeIdle] to preserve session startup.
     */
    fun closeAll() {
        val closing = synchronized(lock) {
            generation++
            val result = slots.values.mapNotNull { state ->
                signal(state)
                state.resident?.let { current ->
                    current.retired = true
                    if (current.owners == 0) claimClose(current) else null
                }
            }
            // Wake acquisitions waiting on a benchmark so they can observe teardown too.
            exclusive?.let { it.changed.countDown(); it.changed = CountDownLatch(1) }
            result
        }
        closeTranslators(closing)
    }

    /** A session holds this until every worker, including delayed native cleanup, has ended. */
    fun tryAcquireSession(): AutoCloseable? = synchronized(lock) {
        if (sessionOwner != null || exclusive != null) return@synchronized null
        val owner = Any()
        sessionOwner = owner
        val closed = AtomicBoolean()
        AutoCloseable {
            if (closed.compareAndSet(false, true)) synchronized(lock) {
                if (sessionOwner === owner) sessionOwner = null
            }
        }
    }

    fun acquireSession(): AutoCloseable = checkNotNull(tryAcquireSession()) {
        "Another session or benchmark is still using native models"
    }

    /**
     * Returns null while any session, model owner, load, or native close is active. On success,
     * idle models have finished closing before return and new session/model admission is held
     * until this lease and all [ExclusiveLease.retain] references close. Call on a worker thread.
     */
    fun tryAcquireExclusive(): ExclusiveLease? {
        val (gate, closing) = synchronized(lock) {
            if (sessionOwner != null || exclusive != null || activeOwners != 0 ||
                pendingAcquires != 0 || pendingLoads != 0 || pendingCloses != 0) return null
            val gate = ExclusiveGate()
            exclusive = gate
            val closing = slots.values.mapNotNull { state ->
                state.resident?.let { current -> current.retired = true; claimClose(current) }
            }
            gate to closing
        }
        closeTranslators(closing)
        return exclusiveLease(gate)
    }

    /** Must be called while holding [lock]. */
    private fun lease(resident: Resident): Lease {
        resident.owners++
        activeOwners++
        return Lease(resident.translator) {
            val closing = synchronized(lock) {
                resident.owners--
                activeOwners--
                if (resident.owners == 0) signal(resident.slot)
                if (resident.retired && resident.closeOnRelease && resident.owners == 0) claimClose(resident) else null
            }
            closing?.let { closeTranslators(listOf(it)) }
        }
    }

    private fun exclusiveLease(gate: ExclusiveGate): ExclusiveLease = ExclusiveLease(
        release = {
            synchronized(lock) {
                gate.references--
                if (gate.references == 0 && exclusive === gate) {
                    exclusive = null
                    gate.changed.countDown()
                }
            }
        },
        retainLease = {
            synchronized(lock) {
                check(exclusive === gate) { "Benchmark admission is no longer held" }
                gate.references++
                exclusiveLease(gate)
            }
        },
    )

    private fun notifyReused(lease: Lease, reused: Boolean, callback: (Boolean) -> Unit): Lease {
        try { callback(reused) }
        catch (t: Throwable) { lease.close(); throw t }
        return lease
    }

    /** Claims exactly one close while holding [lock]; the caller performs it outside the lock. */
    private fun claimClose(resident: Resident): Resident? {
        if (resident.closeClaimed) return null
        resident.closeClaimed = true
        pendingCloses++
        resident.slot.closing++
        return resident
    }

    private fun closeTranslators(residents: List<Resident>) {
        residents.forEach { resident ->
            try { runCatching { resident.translator.close() } }
            finally {
                synchronized(lock) {
                    pendingCloses--
                    resident.slot.closing--
                    if (resident.slot.resident === resident) resident.slot.resident = null
                    signal(resident.slot)
                }
            }
        }
    }

    /** A new signal lets future waiters sleep while all existing slot waiters wake. */
    private fun signal(state: Slot) {
        state.changed.countDown()
        state.changed = CountDownLatch(1)
    }
}
