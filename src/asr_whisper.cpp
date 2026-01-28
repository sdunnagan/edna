#include "asr_whisper.hpp"

#include <dlfcn.h>

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

extern "C" {
// Forward-declare opaque types so we don't need whisper.h at link time.
struct whisper_context;

// Minimal struct layout is ABI-sensitive, so we do NOT re-declare structs here.
// Instead we only traffic in pointers and plain-old-data returned by functions.
}

// ------------------------------------------------------------

static std::string trim_ws(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) a++;
    while (b > a && std::isspace((unsigned char)s[b - 1])) b--;
    return s.substr(a, b - a);
}

// ------------------------------------------------------------
// Dynamically-loaded whisper API
// ------------------------------------------------------------

// We *do* include whisper.h at compile time for type definitions (params structs),
// but we do NOT link libwhisper. That header must match the shared library version.
#include "whisper.h"

struct WhisperApi {
    void* handle = nullptr;

    whisper_context_params (*context_default_params)();
    whisper_context* (*init_from_file_with_params)(const char*, whisper_context_params);
    void (*free_ctx)(whisper_context*);

    whisper_full_params (*full_default_params)(whisper_sampling_strategy);
    int (*full)(whisper_context*, whisper_full_params, const float*, int);
    int (*full_n_segments)(whisper_context*);
    const char* (*full_get_segment_text)(whisper_context*, int);

    bool loaded() const {
        return handle &&
               context_default_params &&
               init_from_file_with_params &&
               free_ctx &&
               full_default_params &&
               full &&
               full_n_segments &&
               full_get_segment_text;
    }
};

static void* must_sym(void* h, const char* name) {
    void* p = dlsym(h, name);
    if (!p) {
        std::fprintf(stderr, "WhisperASR: missing symbol %s: %s\n", name, dlerror());
        std::exit(1);
    }
    return p;
}

static WhisperApi load_whisper_api() {
    WhisperApi api{};

    // Let the dynamic loader find it via RPATH/RUNPATH (you set EDNA_LIB_DIR in RPATH).
    // If you want to hardcode, replace with full path to libwhisper.so.
    const char* soname = "libwhisper.so";

    // RTLD_LOCAL is the whole point: don't leak ggml symbols globally.
    api.handle = dlopen(soname, RTLD_NOW | RTLD_LOCAL);
    if (!api.handle) {
        std::fprintf(stderr, "WhisperASR: dlopen(%s) failed: %s\n", soname, dlerror());
        std::exit(1);
    }

    api.context_default_params =
        (whisper_context_params (*)()) must_sym(api.handle, "whisper_context_default_params");
    api.init_from_file_with_params =
        (whisper_context* (*)(const char*, whisper_context_params)) must_sym(api.handle, "whisper_init_from_file_with_params");
    api.free_ctx =
        (void (*)(whisper_context*)) must_sym(api.handle, "whisper_free");

    api.full_default_params =
        (whisper_full_params (*)(whisper_sampling_strategy)) must_sym(api.handle, "whisper_full_default_params");
    api.full =
        (int (*)(whisper_context*, whisper_full_params, const float*, int)) must_sym(api.handle, "whisper_full");
    api.full_n_segments =
        (int (*)(whisper_context*)) must_sym(api.handle, "whisper_full_n_segments");
    api.full_get_segment_text =
        (const char* (*)(whisper_context*, int)) must_sym(api.handle, "whisper_full_get_segment_text");

    if (!api.loaded()) {
        std::fprintf(stderr, "WhisperASR: failed to load whisper API\n");
        std::exit(1);
    }

    return api;
}

// ------------------------------------------------------------

struct WhisperASR::Impl {
    WhisperApi api{};
    whisper_context* ctx = nullptr;
    Params p{};
    std::string language_stable;
};

WhisperASR::WhisperASR(const std::string& model_path, const Params& p)
    : impl_(new Impl) {
    impl_->p = p;
    impl_->language_stable = p.language;

    impl_->api = load_whisper_api();

    whisper_context_params wp = impl_->api.context_default_params();
    wp.use_gpu = p.use_gpu;

    impl_->ctx = impl_->api.init_from_file_with_params(model_path.c_str(), wp);
    if (!impl_->ctx) {
        std::fprintf(stderr, "WhisperASR: failed to init model: %s\n", model_path.c_str());
        std::exit(1);
    }
}

WhisperASR::~WhisperASR() {
    if (!impl_) return;

    if (impl_->ctx) {
        impl_->api.free_ctx(impl_->ctx);
        impl_->ctx = nullptr;
    }

    if (impl_->api.handle) {
        dlclose(impl_->api.handle);
        impl_->api.handle = nullptr;
    }

    delete impl_;
    impl_ = nullptr;
}

std::string WhisperASR::transcribe_16k_mono_s16(const std::vector<int16_t>& pcm16) {
    if (!impl_ || !impl_->ctx) return "";
    if (pcm16.empty()) return "";

    std::vector<float> pcmf;
    pcmf.reserve(pcm16.size());
    for (int16_t s : pcm16) {
        pcmf.push_back((float) s / 32768.0f);
    }

    whisper_full_params fp = impl_->api.full_default_params(WHISPER_SAMPLING_GREEDY);

    fp.print_realtime   = false;
    fp.print_progress   = false;
    fp.print_timestamps = false;
    fp.print_special    = false;

    fp.translate      = false;
    fp.no_context     = impl_->p.no_context;
    fp.single_segment = impl_->p.single_segment;
    fp.n_threads      = impl_->p.n_threads;

    if (!impl_->language_stable.empty()) {
        fp.language = impl_->language_stable.c_str();
    } else {
        fp.language = nullptr;
    }

    const int rc = impl_->api.full(impl_->ctx, fp, pcmf.data(), (int)pcmf.size());
    if (rc != 0) return "";

    std::string out;
    const int nseg = impl_->api.full_n_segments(impl_->ctx);
    for (int i = 0; i < nseg; i++) {
        const char* t = impl_->api.full_get_segment_text(impl_->ctx, i);
        if (t) out += t;
    }

    out = trim_ws(out);
    if (out == "[BLANK_AUDIO]") out.clear();
    return out;
}

