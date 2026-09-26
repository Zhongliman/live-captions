package com.asr.live.model

import android.content.Context
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.common.model.RemoteModelManager
import com.google.mlkit.nl.translate.TranslateRemoteModel
import com.asr.live.i18n.MlKitTranslator
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.nio.file.Files
import java.nio.file.StandardCopyOption

object ModelRepository {
    enum class Phase { DOWNLOAD, VERIFY, EXTRACT, TRANSLATOR }
    sealed interface DownloadState {
        data object Idle : DownloadState
        data class Running(val id: String, val phase: Phase, val pct: Int, val bytes: Long = 0, val total: Long = 0) : DownloadState
        data class Failed(val id: String, val message: String) : DownloadState
    }
    private val _state = MutableStateFlow<DownloadState>(DownloadState.Idle)
    val state = _state.asStateFlow()
    private val mutex = Mutex()
    suspend fun remove(ctx: Context, id: String) = withContext(Dispatchers.IO) {
        require(id.matches(Regex("[a-zA-Z0-9_.-]+")) && id != "." && id != "..")
        mutex.withLock {
            check(!com.asr.live.service.CaptionState.running.value) { "Stop listening before removing models" }
            val dir = ModelStore.dir(ctx, id)
            check(!dir.exists() || dir.deleteRecursively()) { "Could not delete model files" }
        }
    }
    suspend fun downloadBundle(ctx: Context, id: String): Boolean = withContext(Dispatchers.IO) {
        mutex.withLock {
            val bundle = TranslationModels.bundle(ctx, id)
            val directory = ModelStore.dir(ctx, id).apply { mkdirs() }
            val job = currentCoroutineContext()[Job]
            try {
                for (asset in bundle.files) {
                    val file = File(directory, asset.path)
                    if (runCatching { VerifiedFiles.check(file, asset.size, asset.sha256) }.isSuccess) continue
                    downloadFile(id, asset.url, file, asset.size, asset.sha256, job)
                }
                val manifest = java.util.Properties()
                bundle.files.forEach { manifest.setProperty(it.path, it.sha256) }
                val temporary = File(directory, "verified.properties.part")
                temporary.outputStream().use { manifest.store(it, "Verified pinned translation assets") }
                Files.move(temporary.toPath(), File(directory, "verified.properties").toPath(),
                    StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
                _state.value = DownloadState.Idle
                true
            } catch (t: CancellationException) { _state.value = DownloadState.Idle; throw t }
            catch (t: Exception) { _state.value = DownloadState.Failed(id, t.message ?: "Model download failed"); false }
        }
    }
    suspend fun translationPresent(source: String, target: String): Boolean = withContext(Dispatchers.IO) {
        val installed = Tasks.await(RemoteModelManager.getInstance().getDownloadedModels(TranslateRemoteModel::class.java))
            .map { it.language }.toSet()
        setOf(source, target).filter { it != "en" }.all { it in installed }
    }
    suspend fun prepareTranslation(ctx: Context, info: ModelInfo, source: String, target: String): Boolean = withContext(Dispatchers.IO) {
        prepareTranslation(ctx, info.id, source, target)
    }
    suspend fun prepareTranslation(ctx: Context, id: String, source: String, target: String): Boolean = withContext(Dispatchers.IO) {
        mutex.withLock {
            val translator = MlKitTranslator(source, target)
            try {
                _state.value = DownloadState.Running(id, Phase.TRANSLATOR, 0)
                check(translator.supported) { "Unsupported translation direction" }
                translator.prepare()
                check(translationPresent(source, target)) { "Translation models not installed" }
                _state.value = DownloadState.Idle
                true
            } catch (t: CancellationException) { throw t }
            catch (t: Exception) { _state.value = DownloadState.Failed(id, t.message ?: "Translation download failed"); false }
            finally { translator.close() }
        }
    }
    suspend fun download(ctx: Context, info: ModelInfo): Boolean = withContext(Dispatchers.IO) {
        mutex.withLock {
            _state.value = DownloadState.Running(info.id, Phase.VERIFY, 0)
            if (runCatching { ModelStore.verify(ctx, info) }.isSuccess) {
                _state.value = DownloadState.Idle
                return@withLock true
            }
            val dir = ModelStore.dir(ctx, info.id)
            dir.parentFile!!.mkdirs()
            val archive = File(ctx.cacheDir, info.id + ".archive")
            val staging = File(dir.parentFile, info.id + ".installing")
            val job = currentCoroutineContext()[Job]
            try {
                downloadFile(info.id, info.url, archive, info.archiveBytes, info.sha256, job)
                _state.value = DownloadState.Running(info.id, Phase.EXTRACT, 100)
                staging.deleteRecursively(); staging.mkdirs()
                BZip2CompressorInputStream(archive.inputStream().buffered(), true).use { bz ->
                    TarArchiveInputStream(bz).use { tar ->
                        val seen = mutableSetOf<String>()
                        while (true) {
                            currentCoroutineContext().ensureActive()
                            val entry = tar.nextEntry ?: break
                            if (!entry.isFile || entry.isSymbolicLink || entry.isLink) continue
                            val name = entry.name.substringAfter('/')
                            if (name !in info.requiredFiles) continue
                            check(seen.add(name)) { "Duplicate archive file" }
                            val out = File(staging, name)
                            check(out.canonicalPath.startsWith(staging.canonicalPath + File.separator)) { "Invalid archive path" }
                            out.parentFile!!.mkdirs()
                            out.outputStream().use { tar.copyTo(it) }
                            check(out.length() == entry.size) { "Truncated extracted model" }
                        }
                    }
                }
                ModelStore.writeManifest(staging, info)
                check(ModelStore.isPresent(staging, info)) { "Incomplete model archive" }
                currentCoroutineContext().ensureActive()
                // Only discard an invalid installation after the replacement is fully verified.
                dir.deleteRecursively()
                Files.move(staging.toPath(), dir.toPath(), StandardCopyOption.ATOMIC_MOVE)
                _state.value = DownloadState.Idle
                true
            } catch (t: CancellationException) { _state.value = DownloadState.Idle; throw t }
            catch (t: Exception) { _state.value = DownloadState.Failed(info.id, t.message ?: "Download failed"); false }
            finally { archive.delete(); staging.deleteRecursively() }
        }
    }
    private suspend fun downloadFile(id: String, url: String, target: File, bytes: Long, sha: String, job: Job?) {
        var last: Exception? = null
        repeat(3) { attempt ->
            currentCoroutineContext().ensureActive()
            try {
                val connection = open(url)
                try {
                    connection.inputStream.use { input ->
                        VerifiedFiles.writeVerified(input, target, bytes, sha, { job?.isActive == false }) { read ->
                            _state.value = DownloadState.Running(id, Phase.DOWNLOAD,
                                if (bytes > 0) (100 * read / bytes).toInt() else 0, read, bytes)
                        }
                    }
                } finally { connection.disconnect() }
                return
            } catch (t: CancellationException) { throw t }
            catch (t: Exception) { last = t; if (attempt < 2) delay(1000L * (attempt + 1)) }
        }
        throw last ?: IllegalStateException("Download failed")
    }
    private fun open(address: String): HttpURLConnection {
        var url = URL(address)
        repeat(8) {
            check(url.protocol == "https") { "Model downloads require HTTPS" }
            val connection = (url.openConnection() as HttpURLConnection).apply {
                instanceFollowRedirects = false; connectTimeout = 30000; readTimeout = 30000
                setRequestProperty("User-Agent", "live-captions/2")
            }
            val code = connection.responseCode
            if (code == 200) return connection
            val location = connection.getHeaderField("Location")
            connection.disconnect()
            check(code in listOf(301, 302, 303, 307, 308) && location != null) { "HTTP $code" }
            url = URL(url, location)
        }
        error("Too many download redirects")
    }
}
