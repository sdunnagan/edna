# Edna

Edna is a fully local, GPU-accelerated voice assistant built for low latency, reproducibility, and user control. Speech recognition and language model inference run **entirely on the local machine**, using **CUDA** when available. There are no cloud APIs, hosted LLMs, or external services involved.

Edna is implemented primarily in **C++** as a long-running interactive application, coordinated by a small internal state machine so it does not transcribe its own voice or overlap audio and inference stages.

---

## Overview

Edna implements a streamlined local voice-assistant pipeline using modern open-source components:

- Voice activity detection (VAD) via **libfvad**  
- Speech-to-text using **whisper.cpp** (CUDA-accelerated)  
- Local LLM inference using **llama.cpp** with **GGUF** models  
- Neural text-to-speech using **Coqui TTS**

Wake-word detection is intentionally omitted. On systems with sufficient GPU capacity (e.g. RTX 3070), continuous listening and transcription with VAD gating provides acceptable latency without the complexity of a dedicated keyword-spotting model.

All models are stored locally (GGML/GGUF). Third-party dependencies are built from source, version-pinned, and installed into a local prefix under `deps/install` for reproducibility.

---

## Architecture at a Glance

Runtime flow:

```
[ Microphone ]
      |
      v
Voice activity detection (libfvad)
      |
      v
Speech-to-text (whisper.cpp)
      |
      v
Language model inference (llama.cpp)
      |
      v
Text-to-speech (Coqui TTS worker)
      |
      v
[ Speakers ]
```

Audio input is continuously monitored using VAD to detect speech boundaries. Once speech is detected, Whisper performs synchronous transcription. The resulting prompt is passed to a local LLM via llama.cpp. Text-to-speech synthesis runs in a dedicated worker process, keeping the main assistant responsive and restartable.

A small internal state machine coordinates these stages to prevent feedback loops and serialize access to audio and GPU resources.

---

## Design Goals and Non-Goals

**Goals**
- Fully local execution (offline-capable)
- Low end-to-end latency for interactive use
- Reproducible builds (pinned versions; local install prefix)
- Explicit, inspectable behavior (debuggable systems project)

**Non-Goals**
- Cloud-hosted or multi-user deployment
- Dedicated wake-word models
- CPU-only inference
- Turn-key consumer assistant features

---

## Supported Platforms

Primary development and testing target:
- x86_64 Linux
- NVIDIA GPU with CUDA support
- Ubuntu 25.04 (or similar)

---

## Table of Contents

- [Project Layout](#project-layout)
- [Set up USB mic and speakers](#set-up-usb-mic-and-speakers)
- [Preliminary](#preliminary)
- [Whisper](#whisper)
- [libfvad](#libfvad)
- [LLM Models](#llm-models)
- [Llama](#llama)
- [Neural TTS for voice (via Coqui TTS)](#neural-tts-for-voice-via-coqui-tts)
- [Build Edna voice assistant application](#build-edna-voice-assistant-application)

---

## Project Layout

After cloning, the repository is organized like this:

```
$EDNA_TOP_DIR/
  app/          # C++ voice assistant CMake project
  third_party/  # whisper.cpp, llama.cpp, libfvad, etc. (cloned repos)
  models/       # ggml/gguf models
  resources/    # runtime assets
  deps/         # local install prefix (generated)
  build/        # Edna build output (generated)
```

---

## Set up USB mic and speakers

Connect USB mic and speakers to the target machine, such as:

- ReSpeaker USB mic array v3.1
- Creative Pebble V3

Locate the hardware device for the Creative Pebble USB speakers:

```bash
aplay -l | grep Pebble
```

Play a test sound through the USB speakers:

```bash
speaker-test -D plughw:CARD=V3,DEV=0 -c 2
```

Record audio from the ReSpeaker mic array (16 kHz mono for Whisper):

```bash
arecord -D plughw:CARD=ArrayUAC10,DEV=0 -f S16_LE -r 16000 -c 1 -d 4 /tmp/in_mono.wav
```

Play back the recorded audio:

```bash
aplay -D plughw:CARD=V3,DEV=0 /tmp/in_mono.wav
```

---

## Preliminary

Install required packages:

```bash
sudo apt-get install -y   build-essential   cmake   ninja-build   git   pkg-config   ccache   nvidia-cuda-toolkit   libasound2-dev
```

Verify CUDA and NVIDIA driver:

```bash
nvcc --version
nvidia-smi
```

Add to `.bashrc`:

```bash
export EDNA_TOP_DIR="$HOME/projects/edna"
```

Clone the Edna repository:

```bash
cd "$HOME/projects"
git clone https://github.com/sdunnagan/edna
```

---

## Whisper

Clone and build whisper.cpp (v1.8.3):

```bash
cd "$EDNA_TOP_DIR/third_party"
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp
git checkout v1.8.3

bash models/download-ggml-model.sh base.en

cmake -S . -B build   -DCMAKE_BUILD_TYPE=Release   -DBUILD_SHARED_LIBS=ON   -DGGML_CUDA=ON   -DWHISPER_BUILD_EXAMPLES=OFF   -DWHISPER_BUILD_TESTS=OFF   -DCMAKE_INSTALL_PREFIX="$EDNA_TOP_DIR/deps/install"   -DCMAKE_INSTALL_RPATH="$EDNA_TOP_DIR/deps/install/lib"   -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON

cmake --build build -j "$(nproc)"
cmake --install build
```

Verify CUDA linkage:

```bash
ldd "$EDNA_TOP_DIR/deps/install/lib/libwhisper.so" | grep -i cuda
```

---

## libfvad

Clone and build libfvad:

```bash
cd "$EDNA_TOP_DIR/third_party"
git clone https://github.com/dpirch/libfvad.git
cd libfvad

cmake -S . -B build   -DCMAKE_BUILD_TYPE=Release   -DBUILD_SHARED_LIBS=OFF   -DCMAKE_INSTALL_PREFIX="$EDNA_TOP_DIR/deps/install"

cmake --build build -j "$(nproc)"
cmake --install build
```

---

## LLM Models

Download a GGUF model, for example:

- **Qwen2.5-2B-Instruct-Q6_K.gguf**  
  https://huggingface.co/mradermacher/Qwen2.5-2B-Instruct-GGUF

Place it in:

```
$EDNA_TOP_DIR/models
```

---

## Llama

Clone and build llama.cpp (commit b7782):

```bash
cd "$EDNA_TOP_DIR/third_party"
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout b7782

cmake -S . -B build   -DCMAKE_BUILD_TYPE=Release   -DBUILD_SHARED_LIBS=ON   -DGGML_CUDA=ON   -DCMAKE_CUDA_ARCHITECTURES=86   -DLLAMA_BUILD_TESTS=OFF   -DLLAMA_BUILD_EXAMPLES=OFF   -DCMAKE_INSTALL_PREFIX="$EDNA_TOP_DIR/deps/install"   -DCMAKE_INSTALL_RPATH="$EDNA_TOP_DIR/deps/install/lib"   -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON

cmake --build build -j "$(nproc)"
cmake --install build
```

Test inference:

```bash
"$EDNA_TOP_DIR/third_party/llama.cpp/build/bin/llama-cli"   -m "$EDNA_TOP_DIR/models/Qwen2.5-2B-Instruct-Q6_K.gguf"   -ngl 999 -c 2048 -b 256   -p "Write one sentence about blue sky."
```

---

## Neural TTS for voice (via Coqui TTS)

Install Coqui TTS:

```bash
sudo apt-get install -y espeak-ng
python3 -m pip install --user --break-system-packages coqui-tts
```

Set environment variables:

```bash
export EDNA_TTS_COQUI_BIN="$HOME/.local/bin/tts"
export EDNA_TTS_MODEL="tts_models/en/ljspeech/vits"
export EDNA_TTS_DEVICE="plughw:CARD=V3,DEV=0"
```

Test TTS:

```bash
$EDNA_TTS_COQUI_BIN   --model_name "$EDNA_TTS_MODEL"   --text "Edna is online."   --out_path /tmp/edna.wav   --use_cuda

aplay -D "$EDNA_TTS_DEVICE" /tmp/edna.wav
```

---

## Build Edna voice assistant application

```bash
cd "$EDNA_TOP_DIR"
rm -rf build
cmake -B build -DCMAKE_BUILD_TYPE=Release -DEDNA_USE_CUDA=ON
cmake --build build -j "$(nproc)"
./build/edna
```

---

## License

Add a license file (for example `LICENSE`) before publishing on GitHub.
