#pragma once

#include <string>
#include <mutex>

class LlamaBrain {
public:
    struct Params {
        // GPU offload: Orin Nano has limited VRAM, so keep this conservative.
        int n_gpu_layers     = 16;

        // Context and performance knobs
        int n_ctx            = 512;
        int n_threads        = 6;
        int n_batch          = 64;

        // Generation controls
        int max_new_tokens   = 128;

        // Prompt controls (helps prevent blowing context on long user text)
        int max_prompt_tokens = 384;   // should be <= n_ctx - safety margin

        // Assistant behavior
        std::string system_prompt =
            "You are Edna, a concise voice assistant. Answer in 1-2 sentences.";

        // Stop early for voice UX (useful when models ramble)
        bool stop_on_newline = true;
    };

    LlamaBrain(const std::string& model_path, const Params& p);
    ~LlamaBrain();

    LlamaBrain(const LlamaBrain&) = delete;
    LlamaBrain& operator=(const LlamaBrain&) = delete;

    std::string reply(const std::string& user_text);

    std::mutex reply_mu_;

private:
    struct Impl;
    Impl* impl_;
};

