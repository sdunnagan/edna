// main.cpp
#include <alsa/asoundlib.h>
#include <fvad.h>

#include "asr_whisper.hpp"
#include "llm_llama.hpp"
#include "tts_coqui.hpp"
#include "state_machine.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <cctype>
#include <iostream>
#include <cmath>
#include <chrono>

static constexpr const char* COLOR_RESET = "\033[0m";
static constexpr const char* COLOR_ASR   = "\033[1;32m"; // bright green
static constexpr const char* COLOR_EDNA  = "\033[1;35m"; // bright magenta

static std::string require_env(const char *name) {
    const char *v = std::getenv(name);
    if (!v || !*v) {
        std::fprintf(stderr, "Required environment variable %s is not set\n", name);
        std::exit(1);
    }
    return std::string(v);
}

static void die(const char* msg, int err = 0) {
    if (err) std::fprintf(stderr, "%s: %s\n", msg, snd_strerror(err));
    else     std::fprintf(stderr, "%s\n", msg);
    std::exit(1);
}

static std::atomic<bool> g_running{true};
static void on_sigint(int) { g_running.store(false); }

static std::string trim_ws(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) a++;
    while (b > a && std::isspace((unsigned char)s[b - 1])) b--;
    return s.substr(a, b - a);
}

static std::string normalize(const std::string& in) {
    std::string s;
    s.reserve(in.size());
    for (unsigned char c : in) {
        if (std::isalnum(c) || std::isspace(c)) s.push_back((char)std::tolower(c));
        else s.push_back(' ');
    }
    // collapse spaces
    std::string out;
    out.reserve(s.size());
    bool prev_space = true;
    for (unsigned char c : s) {
        bool sp = std::isspace(c);
        if (sp) {
            if (!prev_space) out.push_back(' ');
        } else {
            out.push_back((char)c);
        }
        prev_space = sp;
    }
    return trim_ws(out);
}

static bool strip_invocation(std::string& text) {
    std::string t = normalize(text);

    auto strip_prefix = [&](const std::string& pfx) -> bool {
        if (t.rfind(pfx, 0) == 0) {
            t = trim_ws(t.substr(pfx.size()));
            return true;
        }
        return false;
    };

    // Strip longer prefixes first. Include aliases for common Whisper mishears.
    bool invoked =
        strip_prefix("hey edna")  ||
        strip_prefix("okay edna") ||
        strip_prefix("ok edna")   ||
        strip_prefix("edna")      ||
        strip_prefix("etna")      ||
        strip_prefix("ewa")       ||
        strip_prefix("ed")        ||
        strip_prefix("ed nah")    ||
        strip_prefix("ed na");

    if (!invoked) return false;

    text = t;   // leave only the remainder (may be empty)
    return true;
}

static std::vector<std::string> split_sentences(const std::string& in) {
    // Cheap splitter to reduce TTS latency by synthesizing smaller chunks.
    std::vector<std::string> out;
    std::string cur;
    cur.reserve(in.size());

    auto flush = [&]() {
        std::string s = trim_ws(cur);
        cur.clear();
        if (!s.empty()) out.push_back(std::move(s));
    };

    for (size_t i = 0; i < in.size(); ++i) {
        char c = in[i];
        cur.push_back(c);

        const bool end_punct = (c == '.' || c == '!' || c == '?');
        if (!end_punct) continue;

        const bool at_end = (i + 1 >= in.size());
        const bool next_space = (!at_end && std::isspace((unsigned char)in[i + 1]));
        if (at_end || next_space) flush();
    }
    flush();

    // If we didn't find punctuation, fall back to a soft wrap.
    if (out.size() <= 1 && !out.empty() && out[0].size() > 180) {
        std::string s = out[0];
        out.clear();
        size_t pos = 0;
        while (pos < s.size()) {
            size_t take = std::min<size_t>(180, s.size() - pos);
            size_t cut = s.rfind(' ', pos + take);
            if (cut == std::string::npos || cut <= pos) cut = pos + take;
            out.push_back(trim_ws(s.substr(pos, cut - pos)));
            pos = cut;
            while (pos < s.size() && std::isspace((unsigned char)s[pos])) pos++;
        }
        // Remove empties.
        std::vector<std::string> clean;
        clean.reserve(out.size());
        for (auto &p : out) if (!p.empty()) clean.push_back(std::move(p));
        out.swap(clean);
    }

    return out;
}


int main() {
    std::signal(SIGINT, on_sigint);

    // Audio capture settings
    const char* device = "plughw:0,0";   // ReSpeaker (card 0, dev 0)
    const unsigned sr = 16000;
    const int frame_ms = 20;
    const int frame_samples = (sr * frame_ms) / 1000; // 320

    // Mic gate: while speaking, ignore mic; after speaking, ignore for a short cooldown
    const int tts_cooldown_ms = 600; // tune: 300..800
    const int cooldown_frames = (tts_cooldown_ms + frame_ms - 1) / frame_ms;

    const std::string TOP = require_env("EDNA_TOP_DIR");
    const std::string whisper_model_path =
        TOP + "/third_party/whisper.cpp/models/ggml-base.en.bin";
    const std::string llama_model_path =
        TOP + "/models/Qwen2.5-2B-Instruct.Q6_K.gguf";

    /* ===================== State Machine ===================== */
    EdnaStateMachine::Config sm_cfg;
    EdnaStateMachine sm(sm_cfg);

    sm.set_observer([](EdnaStateMachine::State from,
                       EdnaStateMachine::State to,
                       EdnaStateMachine::Event why,
                       const std::string& note) {
        std::fprintf(stderr, "[SM] %s --(%s)--> %s%s%s\n",
                     EdnaStateMachine::state_name(from),
                     EdnaStateMachine::event_name(why),
                     EdnaStateMachine::state_name(to),
                     note.empty() ? "" : " : ",
                     note.empty() ? "" : note.c_str());
    });

    sm.start();

    /* ===================== Queues ===================== */
    std::mutex q_m, b_m;
    std::condition_variable q_cv, b_cv;

    std::deque<std::vector<int16_t>> audio_q; // audio -> ASR
    std::deque<std::string> text_q;           // transcript -> brain

    /* ===================== Init ASR + LLM + TTS ===================== */
    WhisperASR::Params asr_p;
    asr_p.use_gpu = true;
    asr_p.n_threads = 4;
    asr_p.single_segment = true;
    asr_p.no_context = true;
    asr_p.language = "en";
    WhisperASR asr(whisper_model_path, asr_p);

    // Tuned for Qwen2.5-2B-Instruct (fast voice assistant)
    LlamaBrain::Params llm_p;
    llm_p.n_gpu_layers = 999; // offload everything that fits
    llm_p.n_ctx = 1024; // keep context short for latency
    llm_p.n_threads = 4;
    llm_p.n_batch = 256;
    llm_p.max_new_tokens = 96; // short spoken replies
    LlamaBrain brain(llama_model_path, llm_p);

    CoquiTTS::Params tts_p;
    tts_p.out_device = "plughw:CARD=V3,DEV=0";
    CoquiTTS tts(tts_p);

    /* ===================== Brain Thread ===================== */
    std::thread brain_thread([&](){
        while (true) {
            std::string text;

            {
                std::unique_lock<std::mutex> lk(b_m);
                b_cv.wait(lk, [&]{ return !g_running.load() || !text_q.empty(); });

                if (!text_q.empty()) {
                    text = std::move(text_q.front());
                    text_q.pop_front();
                } else if (!g_running.load()) {
                    break;
                } else {
                    continue;
                }
            }

            text = trim_ws(text);
            if (text.empty() || text == "[BLANK_AUDIO]") continue;
            
            auto llm0 = std::chrono::steady_clock::now();
            std::string reply = brain.reply(text);
            
            auto strip_after_any = [&](std::string& s, const std::vector<std::string>& toks) {
                size_t cut = std::string::npos;
                for (const auto& t : toks) {
                    size_t p = s.find(t);
                    if (p != std::string::npos)
                        cut = std::min(cut, p);
                }
                if (cut != std::string::npos)
                    s.resize(cut);
            
                // full trim, not just tail
                s = trim_ws(s);
            };
            
            strip_after_any(reply, {
                "<|endoftext|>",
                "<|im_end|>",
                "\nHuman:",
                "\nUSER:",
                "\nUser:",
                "\n### Human:",
                "\n### Instruction:"
            });
            
            // optional safety net
            if (reply.empty()) {
                sm.dispatch(EdnaStateMachine::Event::NoCommand, "empty reply");
                continue;
            }
            
            auto llm1 = std::chrono::steady_clock::now();
            std::fprintf(stderr, "[perf] llm_ms=%lld\n",
                (long long)std::chrono::duration_cast<std::chrono::milliseconds>(llm1 - llm0).count());
            std::fflush(stderr);
            
            sm.dispatch(EdnaStateMachine::Event::ReplyReady);

            std::printf("%sEDNA: %s%s\n", COLOR_EDNA, reply.c_str(), COLOR_RESET);
            std::fflush(stdout);

            // TTS (always print status + timing so we know what happened)
            std::fprintf(stderr, "[tts] enabled=%d device='%s' err='%s'\n",
                         tts.is_enabled() ? 1 : 0,
                         tts_p.out_device.c_str(),
                         tts.last_error().c_str());
            std::fflush(stderr);

            auto tts0 = std::chrono::steady_clock::now();
            bool tts_ok = true;

            if (tts.is_enabled()) {
                // Sentence-by-sentence synthesis: start audio sooner, especially for long replies.
                const auto parts = split_sentences(reply);
                tts_ok = true;

                for (size_t i = 0; i < parts.size(); ++i) {
                    const std::string chunk = trim_ws(parts[i]);
                    if (chunk.empty()) continue;

                    const bool ok = tts.speak(chunk);
                    if (!ok) {
                        tts_ok = false;
                        std::fprintf(stderr, "[tts] speak() FAILED: %s\n", tts.last_error().c_str());
                        std::fflush(stderr);
                        break;
                    }
                }

                if (tts_ok) {
                    std::fprintf(stderr, "[tts] speak() OK\n");
                    std::fflush(stderr);
                }
            }

            auto tts1 = std::chrono::steady_clock::now();
            std::fprintf(stderr, "[perf] tts_ms=%lld ok=%d\n",
                         (long long)std::chrono::duration_cast<std::chrono::milliseconds>(tts1 - tts0).count(),
                         tts_ok ? 1 : 0);
            std::fflush(stderr);

            sm.dispatch(EdnaStateMachine::Event::TtsDone);
        }
    });

    /* ===================== ASR Thread ===================== */
    std::thread asr_thread([&](){
        while (true) {
            std::vector<int16_t> audio;

            {
                std::unique_lock<std::mutex> lk(q_m);
                q_cv.wait(lk, [&]{ return !g_running.load() || !audio_q.empty(); });

                if (!audio_q.empty()) {
                    audio = std::move(audio_q.back());  // newest only
                    audio_q.clear();
                } else if (!g_running.load()) {
                    break;
                } else {
                    continue;
                }
            }

            if (audio.empty()) continue;

            auto asr0 = std::chrono::steady_clock::now();
            std::string txt = asr.transcribe_16k_mono_s16(audio);
            auto asr1 = std::chrono::steady_clock::now();
            std::fprintf(stderr, "[perf] asr_ms=%lld\n",
                         (long long)std::chrono::duration_cast<std::chrono::milliseconds>(asr1 - asr0).count());
            std::fflush(stderr);

            txt = trim_ws(txt);
            const double asr_secs = (double)audio.size() / 16000.0;
            std::fprintf(stderr, "[asr] secs=%.2f raw='%s' norm='%s'\n",
                         asr_secs, txt.c_str(), normalize(txt).c_str());
            std::fflush(stderr);

            if (txt.size() < 2 || txt == "[BLANK_AUDIO]") {
                sm.dispatch(EdnaStateMachine::Event::NoCommand, "blank audio");
                continue;
            }

            std::string cmd = txt;
            if (!strip_invocation(cmd)) {
                sm.dispatch(EdnaStateMachine::Event::NoCommand, "ignored transcript");
                continue;
            }

            cmd = trim_ws(cmd);
            if (cmd.empty()) {
                sm.dispatch(EdnaStateMachine::Event::NoCommand, "invocation only");
                continue;
            }

            std::cout << COLOR_ASR << "ASR: " << txt << COLOR_RESET << std::endl;
            sm.dispatch(EdnaStateMachine::Event::TranscriptReady);
            std::fflush(stdout);

            {
                std::lock_guard<std::mutex> lk(b_m);
                text_q.emplace_back(std::move(cmd));   // enqueue COMMAND, not raw transcript
            }
            b_cv.notify_one();
        }
    });

    /* ===================== Audio + VAD ===================== */
    Fvad *vad = fvad_new();
    if (!vad) die("fvad_new failed");
    if (fvad_set_sample_rate(vad, sr) != 0) die("fvad_set_sample_rate failed");
    fvad_set_mode(vad, 2); // 0..3 higher = more aggressive

    snd_pcm_t *pcm = nullptr;
    int err = snd_pcm_open(&pcm, device, SND_PCM_STREAM_CAPTURE, 0);
    if (err < 0) die("snd_pcm_open failed", err);

    err = snd_pcm_set_params(pcm,
                             SND_PCM_FORMAT_S16_LE,
                             SND_PCM_ACCESS_RW_INTERLEAVED,
                             1,
                             sr,
                             1,
                             20000);
    if (err < 0) die("snd_pcm_set_params failed", err);

    std::vector<int16_t> frame(frame_samples);

    std::vector<int16_t> utterance;
    utterance.reserve(sr * 10); // ~10s max

    // Pre-roll (for ASR)
    const int preroll_frames = 15;
    const size_t max_preroll_samples = (size_t)preroll_frames * (size_t)frame_samples;
    std::vector<int16_t> preroll;
    preroll.reserve(max_preroll_samples);

    bool in_speech = false;
    int voiced_run = 0;
    int unvoiced_run = 0;

    const int start_trigger = 3;  // 60 ms
    const int stop_trigger  = 20; // 400 ms

    // Mic gate state
    int ignore_frames = 0;
    bool last_was_speaking = false;

    std::puts("Listening (Ctrl-C to stop) ...");

    while (g_running.load()) {
        snd_pcm_sframes_t got = snd_pcm_readi(pcm, frame.data(), frame_samples);
        if (got < 0) {
            got = snd_pcm_recover(pcm, (int)got, 1);
            if (got < 0) die("snd_pcm_readi failed", (int)got);
            continue;
        }
        if (got != frame_samples) continue;

        const auto st = sm.state();
        const bool speaking_now = (st == EdnaStateMachine::State::Speaking);

        // Detect transition out of Speaking -> start cooldown
        if (last_was_speaking && !speaking_now) {
            ignore_frames = cooldown_frames;
        }
        last_was_speaking = speaking_now;

        // While speaking or in cooldown: keep ALSA flowing but ignore mic input.
        if (speaking_now || ignore_frames > 0) {
            if (ignore_frames > 0) ignore_frames--;

            // Hard reset capture-side accumulators so we don't queue nonsense later.
            in_speech = false;
            voiced_run = 0;
            unvoiced_run = 0;
            utterance.clear();
            preroll.clear();

            // Also drop any pending ASR audio so it doesn't "catch up" late.
            {
                std::lock_guard<std::mutex> lk(q_m);
                audio_q.clear();
            }
            continue;
        }

        // Update pre-roll
        preroll.insert(preroll.end(), frame.begin(), frame.end());
        if (preroll.size() > max_preroll_samples) {
            const size_t extra = preroll.size() - max_preroll_samples;
            preroll.erase(preroll.begin(), preroll.begin() + (long)extra);
        }

        int is_speech = fvad_process(vad, frame.data(), frame_samples);
        if (is_speech < 0) die("fvad_process failed");

        if (!in_speech) {
            if (is_speech) voiced_run++;
            else voiced_run = 0;

            if (voiced_run >= start_trigger) {
                in_speech = true;
                voiced_run = 0;
                unvoiced_run = 0;

                utterance.clear();
                utterance.insert(utterance.end(), preroll.begin(), preroll.end());

                sm.dispatch(EdnaStateMachine::Event::SpeechStart, "VAD start_trigger");

                std::puts(">>> speech start");
                std::fflush(stdout);
            }
        } else {
            utterance.insert(utterance.end(), frame.begin(), frame.end());

            if (!is_speech) unvoiced_run++;
            else unvoiced_run = 0;

            if (unvoiced_run >= stop_trigger) {
                in_speech = false;
                unvoiced_run = 0;

                sm.dispatch(EdnaStateMachine::Event::SpeechEndQueued, "VAD stop_trigger");

                std::puts("<<< speech end (queued)");
                std::fflush(stdout);

                const double secs = (double)utterance.size() / (double)sr;
                if (secs >= 0.20) {
                    {
                        std::lock_guard<std::mutex> lk(q_m);
                        audio_q.clear();
                        audio_q.emplace_back(std::move(utterance));
                    }
                    q_cv.notify_one();
                }

                utterance = std::vector<int16_t>();
                utterance.reserve(sr * 10);
            }
        }
    }

    std::puts("\nStopping...");
    sm.dispatch(EdnaStateMachine::Event::Stop, "SIGINT");

    snd_pcm_close(pcm);
    fvad_free(vad);

    g_running.store(false);
    q_cv.notify_all();
    b_cv.notify_all();

    if (asr_thread.joinable()) asr_thread.join();
    if (brain_thread.joinable()) brain_thread.join();

    return 0;
}

