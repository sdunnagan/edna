// asr_whisper.hpp
#pragma once

#include <cstdint>
#include <string>
#include <vector>

class WhisperASR {
public:
    struct Params {
        // Model/backend behavior
        bool use_gpu = true;      // whisper.cpp: enables GPU (e.g., cuBLAS) if built with it
        int  gpu_device = 0;      // kept for future/multi-GPU sanity (harmless on Jetson)

        // Performance/latency
        int  n_threads = 6;       // Orin Nano: 6 CPU cores; tune alongside LLM threads

        // Decoding / output behavior
        bool single_segment = true;
        bool no_context = true;
        std::string language = "en";   // safer than const char*

        // Optional hygiene knobs
        int  max_len = 0;         // 0 = unlimited (whisper default). If supported by your whisper version.
    };

    WhisperASR(const std::string& model_path, const Params& p);
    ~WhisperASR();

    WhisperASR(const WhisperASR&) = delete;
    WhisperASR& operator=(const WhisperASR&) = delete;

    // Input: 16-bit mono PCM at 16kHz
    // Output: trimmed transcript (possibly empty)
    std::string transcribe_16k_mono_s16(const std::vector<int16_t>& pcm);

private:
    struct Impl;
    Impl* impl_;
};

