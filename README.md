# Live Captions for Honor Magic V5

Local microphone captions for arm64 Android (Android 10+, optimized for Snapdragon
8 Elite). Download the selected models, tap **Listen**, then **Stop**. No cloud
inference, external microphone, root, Shizuku or Termux is used. Audio stays in a
bounded in-memory buffer and is not saved or uploaded.

## Profiles

| Profile | Recognition | Translation |
|---|---|---|
| **Dutch → English (default)** | Nemotron 3.5 0.6B INT8, 560 ms | One Hy-MT2 engine translates stable prefixes and endpoints; Magic V5 defaults to 7B Q4_K_M |
| Mandarin → English | Qwen3-ASR 0.6B INT8, VAD phrases up to 4 s | Selected Hy-MT2 model translates stable text and endpoints; Parakeet is never used |
| English → French | Nemotron 3.5 0.6B INT8, 560 ms | Selected Hy-MT2 model translates stable prefixes and endpoints |

The Honor Magic V5 (including model identifier MBH-N49) starts in **MAX QUALITY** mode:
Nemotron 560 ms and Hy-MT2 7B Q4_K_M. This path uses a single resident Hy engine
for provisional and final output, with optional OPUS startup fallback when that
bundle is installed. **Fast** uses OPUS previews and Hy 1.8B Q4 phrase finals for
Dutch → English and English → French. Live OPUS/Hy prefix comparison is a separate,
explicit setting; normal Fast operation avoids duplicating that inference work.
Comparison votes are human judgments, not reference-scored accuracy.
The separate quantization autotune compares 7B Q4, Q5, Q6 and Q8 model outputs.
Selecting a preset never silently enables an untested accelerator.
Memory admission estimates the selected translator's additional weights, context,
compute buffers and optional OpenCL staging, with system headroom derived from
physical RAM and Android's low-memory threshold. Already loaded ASR/OPUS/QNN
memory is not counted again. A refusal reports available RAM, additional allocation
and headroom separately; it does not equate installed files with resident memory.

Hy-MT2 uses the selected native CPU/OpenCL backend and reuses only the identical
prompt-token KV prefix within the same session/configuration; generated and divergent
tokens are removed before the next request. It streams UTF-8 word-boundary previews
and reports first-visible separately from EOS completion. It supports an optional
`source -> target` glossary. OPUS supplies Fast previews and, when installed,
startup fallback while Hy loads. It uses SentencePiece,
separate encoder execution and a merged decoder with cached self/cross attention.
Q6_K and Q4_K_M are selectable memory/speed alternatives; all three quantizations
have local model execution evidence. ML Kit is an explicitly selected local
fallback, never a silent substitution. Whisper base is a compatibility ASR option;
it transcribes source text through the same translation pipeline.

Dutch and English can use Parakeet TDT 0.6B v3 as an optional endpoint second
hypothesis. It is limited to one pending utterance and skipped when work falls
behind. Original endpoint translation is queued immediately. A timely candidate
may be translated afterward, below original finals in priority; the source and
translation are then replaced together. Conservative guards reject changes to
protected numbers, units and polarity, broad rewrites and short-answer substitutions.
These lexical checks do not establish semantic correctness. Parakeet is never used for Mandarin.
This heuristic is not a claim that Parakeet is always more accurate.

## Reliability and controls

- Independent capture, recognition, translation and optional correction stages.
  One Hy worker handles live prefixes and endpoints in the default path; bounded
  queues prioritize endpoints and coalesce stale provisional requests.
- Stable source prefixes, caption segment IDs and monotonically increasing
  revisions reject stale results. Provisional, final, revised and skipped states
  are visible. Mandarin can stabilize at complete Han character boundaries.
  Source text appears below the prominent translation. The live view follows text
  growth until the user scrolls back, then resumes following at the bottom.
- Stop/restart invalidates old work. Native resources are released by their owning
  worker; cancellation does not free an in-use recognizer or translator.
- **Listen** starts the microphone as soon as recognition is ready. Translation and
  optional correction models keep loading concurrently, so a multi-GB translation
  model never delays capture; a required worker that fails later still ends the
  session with a visible error. Translation startup reports verification, loading
  and warm-up phases and has a 60-second total deadline. Late completion cannot
  publish READY after timeout or cancellation. Expired startup requests are settled
  explicitly; reading diagnostics never discards them.
- Resident models use leases and independent slot loading. Idle trimming cannot
  close an active translator. Benchmarks evict idle models and wait for exclusive
  admission, including after STOPPED while native workers are still unwinding.
- **Stop** is bounded. It immediately stops accepting input and discards work that
  only feeds the live caption line, while captured PCM still flushes through the
  recognizer. After ~1.5 s remaining translations are dropped but recognition keeps
  flushing; after ~2.5 s the session is force-cancelled; after ~4 s the UI returns to
  STOPPED regardless, so STOPPING is never open-ended. A translation model still
  loading when Stop arrives is cancelled mid-load through the llama.cpp model-load
  progress callback.
- Performance panel: recognition time/RTF, translation and correction time,
  endpoint-to-caption latency, queue depths, audio backlog and dropped work.
  Endpoint latency starts at the recognizer's endpoint event, not a measured
  acoustic end-of-speech timestamp.
- Model downloads use pinned URLs, sizes and SHA-256, temporary files, verified
  installation, retries and progress. Installed models are rehashed before use.
  Settings → **Verify / repair required models** repairs damaged installations.
- The displayed download estimate follows the selected ASR, translation, QNN and
  correction models. Model files require additional temporary space during
  verified installation.

## CPU, QNN and GPU

CPU is the default. The CI APK also includes a real **experimental QNN/NPU** path
for **SM8750 only**, using the released 560-ms contexts and matching QAIRT 2.40 /
HTP v79 libraries. It is opt-in, visibly labeled experimental and isolated in a
private process. Native failure or timeout falls back to the downloaded CPU model.
QNN startup is bounded for interactive use — 3 s to bind, 1 s for the process
round trip and 8 s to initialize — so an unhealthy vendor stack costs at most a few
seconds before CPU is selected instead of delaying Listen.
Only a successfully initialized QNN session is labeled QNN; CPU fallback is
explicit. No additional QNN chunk option is exposed. CPU has separately verified
160/320/560/1120-ms model profiles; 560 ms remains the default.

Hy-MT2 GPU/OpenCL offload is **not enabled**. See [accelerator investigation,
provenance and limitations](docs/ACCELERATION.md). No thermal safeguards are
changed. No physical Honor Magic V5 test or real-time performance guarantee is
claimed; phone benchmarking is still required, especially under simultaneous ASR,
translation and correction load.

## Build and validation

Optional floating captions use Android's **Display over other apps** permission;
no Accessibility Service is used. Settings include opacity, font size, 1–4
translation lines, source text and touch-through. Turn touch-through off to drag;
the listening notification can show/hide the window. Position is remembered
separately for compact and expanded screens. In-app captions require no overlay
permission. See [overlay and chunk-profile validation](docs/OVERLAY.md).

Settings → Download / remove models manages pinned model installations. New CPU
chunk profiles require a successful on-device load/decode test before selection.
Performance includes model/backend/chunk, latencies, RTF, bounded-queue backlog
and PSS (including QNN worker), main-process RSS, native and Java heaps,
estimated model file sizes and available system RAM.

CI builds a persistently signed arm64 APK and runs the unit tests, ZIP integrity,
signature, manifest, native ABI and 16 KB alignment checks. APK and SHA-256 are
uploaded as workflow artifacts. A separate **Model execution smoke** workflow
runs the shared native translation core and pinned ASR models on a CPU runner.
Those results establish model execution on a host, not an Android microphone test.
[Model pins and provenance](docs/MODELS.md), [recorded validation](docs/validation/).

Build requirements: JDK 17, Android SDK/build-tools 35, NDK 28.2.13676358,
CMake 3.22.1. A CPU development APK can be built with:

```sh
./gradlew :app:testDebugUnitTest :app:assembleDebug
```

For the QNN variant, follow `.github/workflows/ci.yml`: fetch the verified original
AAR, extract its arm64 ORT library, run `scripts/build_qnn.sh` with ANDROID_NDK set,
then build with `-Pqnn=true`. Proprietary SDK headers/runtime binaries and model
weights are downloaded during builds or app setup and are kept out of Git.

Release signing uses repository secrets `LIVE_CAPTIONS_KEYSTORE_B64`,
`LIVE_CAPTIONS_STORE_PASSWORD`, and `LIVE_CAPTIONS_KEY_PASSWORD`. The persistent
key is backed up outside the repository on the operator's machine. Never commit
the key or its passwords. Use APKs signed with this identity for subsequent updates.
Both workflows verify the pinned public signing certificate and upload
`LiveTranslate-MagicV5.apk` with its `.apk.sha256` in one artifact.
See [persistent signing and secret setup](docs/SIGNING.md) for recovery commands
and the version-code rule for future releases.

## MAX QUALITY status

Qwen3-ASR 1.7B Dutch endpoint recognition, a Dutch Qwen-versus-Parakeet A/B
accuracy winner, Hy-MT2 7B Q5, Adreno OpenCL offload and physical-device QNN
validation are not included in this APK. The 7B Q4 and Q6 models' host load/output smoke
and the on-device translation A/B control do not establish real-time performance
on the Honor phone. The app retains bounded queues and the CPU fallback.
