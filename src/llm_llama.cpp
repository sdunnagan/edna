#include "llm_llama.hpp"

// llama.h drags ggml headers too. Keep it quarantined in this TU.
#include <llama.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

/* ------------------------------------------------------------ */
/* Helpers                                                      */
/* ------------------------------------------------------------ */

static std::string trim_ws(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) a++;
    while (b > a && std::isspace((unsigned char)s[b - 1])) b--;
    return s.substr(a, b - a);
}

static void batch_reset(llama_batch& b) { b.n_tokens = 0; }

static void batch_add(llama_batch& b, llama_token id, llama_pos pos, bool logits) {
    const int32_t i = b.n_tokens;
    b.token[i]     = id;
    b.pos[i]       = pos;
    b.n_seq_id[i]  = 1;
    b.seq_id[i][0] = 0;
    b.logits[i]    = logits;
    b.n_tokens++;
}

// Two-pass tokenization (works across many llama.cpp revisions)
static std::vector<llama_token> tokenize_prompt(const llama_vocab* vocab,
                                                const std::string& text,
                                                bool add_special) {
    int32_t n = llama_tokenize(vocab,
                               text.c_str(),
                               (int32_t)text.size(),
                               nullptr,
                               0,
                               add_special,
                               /*parse_special=*/true);
    if (n < 0) n = -n;

    std::vector<llama_token> toks((size_t)n);
    int32_t n2 = llama_tokenize(vocab,
                                text.c_str(),
                                (int32_t)text.size(),
                                toks.data(),
                                (int32_t)toks.size(),
                                add_special,
                                /*parse_special=*/true);
    if (n2 < 0) n2 = -n2;
    toks.resize((size_t)n2);
    return toks;
}

static std::string token_to_piece(const llama_vocab* vocab, llama_token tok) {
    std::string out;
    out.resize(256);

    int32_t n = llama_token_to_piece(vocab, tok,
                                     out.data(),
                                     (int32_t)out.size(),
                                     /*lstrip=*/0,
                                     /*special=*/true);
    if (n < 0) {
        out.resize((size_t)(-n));
        n = llama_token_to_piece(vocab, tok,
                                 out.data(),
                                 (int32_t)out.size(),
                                 0, true);
    }

    if (n > 0) out.resize((size_t)n);
    else out.clear();
    return out;
}

// Backend init once-per-process (avoid repeated init/free thrash)
static std::once_flag g_backend_once;
static std::mutex     g_backend_mu;
static std::atomic<int> g_backend_refcnt{0};

static void backend_acquire() {
    std::call_once(g_backend_once, []() {
        llama_backend_init();
    });
    g_backend_refcnt.fetch_add(1, std::memory_order_relaxed);
}

static void backend_release() {
    const int n = g_backend_refcnt.fetch_sub(1, std::memory_order_relaxed) - 1;
    if (n == 0) {
        std::lock_guard<std::mutex> lk(g_backend_mu);
        llama_backend_free();
    }
}

static llama_sampler* make_sampler() {
    const uint32_t seed = 0xC0FFEEu;  // or time-based if you prefer
    const float temp    = 0.7f;
    const int   top_k   = 40;
    const float top_p   = 0.9f;

    llama_sampler* chain =
        llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (!chain) return nullptr;

    // Order: penalties (optional) -> temp -> top-k -> top-p -> dist
    // Keep it simple until stable.
    llama_sampler_chain_add(chain, llama_sampler_init_temp(temp));
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(chain, llama_sampler_init_top_p(top_p, 1));

    // REQUIRED: actually select a token
    llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));

    return chain;
}

static std::string build_prompt(const LlamaBrain::Params& p, const std::string& user_text) {
    // Keep this simple and predictable (fast, low-token).
    // You can swap to chat templates later if you want model-specific formatting.
    std::string s;
    s.reserve(p.system_prompt.size() + user_text.size() + 64);

    s += p.system_prompt;
    if (!s.empty() && s.back() != '\n') s += "\n";
    s += "User: ";
    s += user_text;
    s += "\nEdna:";

    return s;
}

/* ------------------------------------------------------------ */
/* LlamaBrain Impl                                              */
/* ------------------------------------------------------------ */

struct LlamaBrain::Impl {
    Params p{};

    llama_model* model = nullptr;
    const llama_vocab* vocab = nullptr;

    llama_context_params cparams{};
    llama_context* ctx = nullptr;

    llama_sampler* sampler = nullptr;
};

LlamaBrain::LlamaBrain(const std::string& model_path, const Params& p) : impl_(new Impl) {
    impl_->p = p;

    backend_acquire();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = p.n_gpu_layers;

    impl_->model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!impl_->model) {
        std::fprintf(stderr, "LlamaBrain: failed to load model: %s\n", model_path.c_str());
        std::exit(1);
    }

    impl_->vocab = llama_model_get_vocab(impl_->model);
    if (!impl_->vocab) {
        std::fprintf(stderr, "LlamaBrain: failed to get vocab\n");
        std::exit(1);
    }

    impl_->cparams = llama_context_default_params();
    impl_->cparams.n_ctx     = p.n_ctx;
    impl_->cparams.n_threads = p.n_threads;
    impl_->cparams.n_batch   = p.n_batch;

    impl_->ctx = llama_init_from_model(impl_->model, impl_->cparams);
    if (!impl_->ctx) {
        std::fprintf(stderr, "LlamaBrain: failed to create context\n");
        std::exit(1);
    }

    impl_->sampler = make_sampler();
    if (!impl_->sampler) {
        std::fprintf(stderr, "LlamaBrain: failed to init sampler\n");
        std::exit(1);
    }
}

LlamaBrain::~LlamaBrain() {
    if (!impl_) return;

    if (impl_->sampler) llama_sampler_free(impl_->sampler);
    if (impl_->ctx)     llama_free(impl_->ctx);
    if (impl_->model)   llama_model_free(impl_->model);

    backend_release();

    delete impl_;
    impl_ = nullptr;
}

std::string LlamaBrain::reply(const std::string& user_text) {
    // Serialize ALL access to impl_ / ctx / sampler. llama.cpp contexts are not thread-safe.
    std::lock_guard<std::mutex> lock(reply_mu_);

    if (!impl_ || !impl_->model || !impl_->vocab) {
        return "(no model)";
    }

    const int32_t n_ctx   = std::max<int32_t>(64, impl_->p.n_ctx);
    const int32_t n_batch = std::max<int32_t>(8,  impl_->p.n_batch);

    const std::string prompt = build_prompt(impl_->p, user_text);

    // Reset between requests (portable):
    // Older llama.h doesn't expose KV-cache APIs, so we recreate ONLY the context.
    // This keeps the model loaded and avoids full reload cost.
    if (impl_->ctx) {
        llama_free(impl_->ctx);
        impl_->ctx = nullptr;
    }
    impl_->ctx = llama_init_from_model(impl_->model, impl_->cparams);
    if (!impl_->ctx) {
        return "(failed to recreate context)";
    }
    
    // Samplers are stateful: recreate per request.
    if (impl_->sampler) {
        llama_sampler_free(impl_->sampler);
        impl_->sampler = nullptr;
    }
    impl_->sampler = make_sampler();
    if (!impl_->sampler) {
        return "(failed to recreate sampler)";
    }

    // Tokenize full prompt
    std::vector<llama_token> toks =
        tokenize_prompt(impl_->vocab, prompt, /*add_special=*/true);

    if (toks.empty()) {
        return "(empty prompt)";
    }

    // Decide how many prompt tokens we allow:
    // - prefer user-specified max_prompt_tokens
    // - clamp to fit in context with a safety margin for generation
    const int32_t safety = std::max<int32_t>(32, impl_->p.max_new_tokens + 8);
    int32_t max_prompt =
        (impl_->p.max_prompt_tokens > 0) ? impl_->p.max_prompt_tokens : (n_ctx - safety);

    max_prompt = std::min<int32_t>(max_prompt, n_ctx - safety);
    if (max_prompt < 16) max_prompt = 16;

    if ((int32_t)toks.size() > max_prompt) {
        // Keep tail so we preserve the recent turns near the end.
        toks.erase(toks.begin(), toks.end() - max_prompt);

        // Best-effort: ensure BOS if used
        const llama_token bos = llama_vocab_bos(impl_->vocab);
        if (!toks.empty() && toks.front() != bos) {
            toks.insert(toks.begin(), bos);
        }
    }

    llama_pos pos = 0;

    // Batch init
    llama_batch batch = llama_batch_init(n_batch, 0, 1);

    // --------------------
    // Prompt decode
    // --------------------
    for (size_t i = 0; i < toks.size(); i++) {
        if (pos >= n_ctx) break;

        batch_reset(batch);

        // Only the last prompt token needs logits
        const bool want_logits = (i + 1 == toks.size());
        batch_add(batch, toks[i], pos++, want_logits);

        if (llama_decode(impl_->ctx, batch) != 0) {
            llama_batch_free(batch);
            return "(decode failed on prompt)";
        }
    }

    // REQUIRED: reset sampler after prompt decode and before first sampling.
    llama_sampler_reset(impl_->sampler);

    std::string out;
    out.reserve(256);

    const llama_token eos = llama_vocab_eos(impl_->vocab);

    // --------------------
    // Generation loop
    // --------------------
    for (int i = 0; i < impl_->p.max_new_tokens; i++) {
        if (pos >= n_ctx - 1) break;

        // Must have logits right now.
        if (!llama_get_logits_ith(impl_->ctx, 0)) {
            break;
        }

        // REQUIRED: reset sampler BEFORE EVERY SAMPLE in modern llama.cpp.
        llama_sampler_reset(impl_->sampler);

        llama_token tok = llama_sampler_sample(impl_->sampler, impl_->ctx, 0);
        llama_sampler_accept(impl_->sampler, tok);

        if (tok == eos) break;

        out += token_to_piece(impl_->vocab, tok);

        if (impl_->p.stop_on_newline) {
            auto nl = out.find('\n');
            if (nl != std::string::npos) {
                out.resize(nl);
                break;
            }
        }

        // Decode generated token WITH logits enabled so we can sample next.
        batch_reset(batch);
        batch_add(batch, tok, pos++, /*want_logits=*/true);

        if (llama_decode(impl_->ctx, batch) != 0) {
            out += " (decode failed)";
            break;
        }
    }

    llama_batch_free(batch);

    out = trim_ws(out);
    if (out.empty()) out = "(no response)";
    return out;
}

