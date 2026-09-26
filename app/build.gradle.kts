import java.net.URI
import java.security.MessageDigest

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

val qnnEnabled = providers.gradleProperty("qnn").orNull == "true"
val openclEnabled = providers.gradleProperty("opencl").orNull == "true"
val openclRoot = layout.buildDirectory.dir("opencl").get().asFile

android {
    namespace = "com.asr.live"
    compileSdk = 35
    ndkVersion = "28.2.13676358"

    defaultConfig {
        applicationId = "com.asr.live"
        minSdk = 29
        targetSdk = 35
        buildConfigField("boolean", "QNN_ENABLED", qnnEnabled.toString())
        buildConfigField("boolean", "OPENCL_ENABLED", openclEnabled.toString())
        versionCode = 10
        versionName = "2.7"

        // Honor Magic V5 uses arm64-v8a.
        ndk { abiFilters += "arm64-v8a" }
        externalNativeBuild {
            cmake {
                arguments += "-DANDROID_STL=c++_shared"
                if (openclEnabled) arguments += listOf(
                    "-DTRANSLATION_OPENCL=ON",
                    "-DOpenCL_LIBRARY=${openclRoot.resolve("lib/libOpenCL_loader.a").absolutePath}",
                    "-DOpenCL_INCLUDE_DIR=${openclRoot.resolve("deps/headers").absolutePath}",
                )
            }
        }
    }

    signingConfigs {
        create("persistent") {
            val path = System.getenv("LIVE_CAPTIONS_KEYSTORE")
            if (!path.isNullOrEmpty()) {
                storeFile = file(path)
                storePassword = System.getenv("LIVE_CAPTIONS_STORE_PASSWORD")
                keyAlias = "live-captions"
                keyPassword = System.getenv("LIVE_CAPTIONS_KEY_PASSWORD")
            }
        }
    }

    buildTypes {
        release {
            signingConfig = signingConfigs.getByName("persistent")
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    buildFeatures {
        compose = true
        aidl = true
        buildConfig = true
    }
    testOptions { unitTests.isIncludeAndroidResources = true }
    externalNativeBuild { cmake { path = file("src/main/cpp/CMakeLists.txt"); version = "3.22.1" } }
    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
        // ONNX Runtime / sherpa-onnx .so files must stay loadable.
        jniLibs {
            useLegacyPackaging = qnnEnabled
            pickFirsts += "**/libonnxruntime.so"
        }
    }
}

dependencies {
    // Fully-offline on-device ASR runtime (ONNX Runtime + JNI + Kotlin API).
    implementation(files(if (qnnEnabled) "libs/sherpa-onnx-qnn-1.13.8.aar" else "libs/sherpa-onnx-1.13.8.aar"))

    implementation("androidx.core:core-ktx:1.13.1")
    // Provides the Theme.Material3.DayNight.* XML themes for the Activity window.
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7")
    implementation("androidx.activity:activity-compose:1.9.3")

    implementation(platform("androidx.compose:compose-bom:2024.10.01"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-extended")

    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")

    // Model archives ship as .tar.bz2; commons-compress decodes both (pure Java).
    implementation("org.apache.commons:commons-compress:1.27.1")

    // On-device (offline) text translation to any of 50+ languages.
    implementation("com.google.mlkit:translate:17.0.3")

    debugImplementation("androidx.compose.ui:ui-tooling")
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.robolectric:robolectric:4.14.1")
}

// The sherpa-onnx runtime AAR (~56 MB) is fetched on demand instead of being committed,
// so fresh clones and CI build without storing a large binary in git.
val sherpaAarUrl = "https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.8/sherpa-onnx-1.13.8.aar"
val sherpaAarSha256 = "633c24321e06b1fe79feafa03ea16cbc0f8a286641e2da3559bac91bdb13bd96"
val downloadSherpaAar by tasks.registering {
    val out = layout.projectDirectory.file("libs/sherpa-onnx-1.13.8.aar").asFile
    outputs.file(out)
    doLast {
        if (!out.exists() || out.length() == 0L) {
            out.parentFile.mkdirs()
            logger.lifecycle("Downloading sherpa-onnx AAR from $sherpaAarUrl")
            URI(sherpaAarUrl).toURL().openStream().use { input ->
                out.outputStream().use { output -> input.copyTo(output) }
            }
        }
        val actual = MessageDigest.getInstance("SHA-256").digest(out.readBytes())
            .joinToString("") { "%02x".format(it) }
        check(actual == sherpaAarSha256) { "sherpa-onnx AAR checksum mismatch" }
    }
}
tasks.named("preBuild") { dependsOn(downloadSherpaAar) }

val extractSherpaOrt by tasks.registering(Copy::class) {
    dependsOn(downloadSherpaAar)
    from(zipTree(layout.projectDirectory.file("libs/sherpa-onnx-1.13.8.aar"))) {
        include("jni/arm64-v8a/libonnxruntime.so")
    }
    into(layout.buildDirectory.dir("ort"))
}
tasks.configureEach {
    if (name.startsWith("configureCMake") || name.startsWith("buildCMake")) dependsOn(extractSherpaOrt)
}
