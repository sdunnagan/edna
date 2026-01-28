// tts_coqui.hpp
#pragma once

#include <string>
#include <mutex>

class CoquiTTS {
public:
    struct Params {
        // Playback device (aplay -D <device>)
        std::string out_device = "default";

        // Python executable to run the worker (e.g. "python3")
        std::string python_bin = "python3";

        // Coqui TTS model name
        // Example: "tts_models/en/ljspeech/vits"
        std::string model_name = "tts_models/en/ljspeech/vits";

        // Try to use CUDA in the worker (best effort)
        bool use_cuda = false;

        // Directory for temporary wav files
        std::string tmp_dir = "/tmp";

        // aplay binary (or "paplay" if you swap implementation)
        std::string aplay_bin = "aplay";

        // Extra aplay args (optional)
        std::string aplay_extra_args = "";
    };

    explicit CoquiTTS(const Params& p);
    ~CoquiTTS();

    bool is_enabled() const;
    std::string last_error() const;

    // Synthesize and play. Returns true if it *played* audio.
    bool speak(const std::string& text);

    // Optional: explicitly (re)start the worker.
    bool ensure_worker();

    // Optional: stop worker now.
    void shutdown();

private:
    struct Worker {
        int   to_child   = -1;   // parent -> child (stdin)
        int   from_child = -1;   // child -> parent (stdout)
        pid_t pid        = -1;
        bool  ready      = false;
    };

    bool start_worker_locked();
    void stop_worker_locked();
    bool worker_handshake_locked();
    bool write_all_locked(const char* data, size_t n);
    bool read_line_locked(std::string& out_line, int timeout_ms);

    bool play_wav_locked(const std::string& wav_path);

    Params p_;
    mutable std::mutex m_;
    Worker worker_;
    bool enabled_ = true;
    std::string last_err_;
};

