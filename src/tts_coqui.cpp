// tts_coqui.cpp
#include "tts_coqui.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <string>
#include <vector>

#include <unistd.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <sys/select.h>
#include <signal.h>

static std::string shell_escape_single_quotes(const std::string& s) {
    // Not used for command-line (we execve), but handy if you later do popen().
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    return out;
}

CoquiTTS::CoquiTTS(const Params& p)
    : p_(p) {
    // Lazy-start by default (start on first speak()).
}

CoquiTTS::~CoquiTTS() {
    shutdown();
}

bool CoquiTTS::is_enabled() const {
    std::lock_guard<std::mutex> lk(m_);
    return enabled_;
}

std::string CoquiTTS::last_error() const {
    std::lock_guard<std::mutex> lk(m_);
    return last_err_;
}

void CoquiTTS::shutdown() {
    std::lock_guard<std::mutex> lk(m_);
    stop_worker_locked();
}

bool CoquiTTS::ensure_worker() {
    std::lock_guard<std::mutex> lk(m_);
    if (!enabled_) return false;
    if (worker_.pid > 0 && worker_.ready) return true;
    return start_worker_locked();
}

bool CoquiTTS::start_worker_locked() {
    // If already running, don’t double-start.
    if (worker_.pid > 0) {
        if (worker_.ready) return true;
        // Not ready? kill and restart.
        stop_worker_locked();
    }

    int in_pipe[2]  = {-1, -1};  // parent writes to [1], child reads [0]
    int out_pipe[2] = {-1, -1};  // child writes to [1], parent reads [0]

    if (::pipe(in_pipe) != 0) {
        last_err_ = "pipe(in_pipe) failed";
        enabled_ = false;
        return false;
    }
    if (::pipe(out_pipe) != 0) {
        ::close(in_pipe[0]); ::close(in_pipe[1]);
        last_err_ = "pipe(out_pipe) failed";
        enabled_ = false;
        return false;
    }

    pid_t pid = ::fork();
    if (pid < 0) {
        ::close(in_pipe[0]); ::close(in_pipe[1]);
        ::close(out_pipe[0]); ::close(out_pipe[1]);
        last_err_ = "fork() failed";
        enabled_ = false;
        return false;
    }

    if (pid == 0) {
        // Child
        ::dup2(in_pipe[0], STDIN_FILENO);
        ::dup2(out_pipe[1], STDOUT_FILENO);

        // Keep stderr as-is (terminal). Important: warnings won’t corrupt stdout protocol.

        ::close(in_pipe[0]);
        ::close(in_pipe[1]);
        ::close(out_pipe[0]);
        ::close(out_pipe[1]);

        // Python worker script. Prints "READY" once model is loaded, then prints a wav path per line.
        // Uses -u for unbuffered stdout.
        const std::string script =
R"PY(
import os, sys, time, warnings
warnings.filterwarnings("ignore")
from TTS.api import TTS

model = os.environ.get("EDNA_TTS_MODEL", "tts_models/en/ljspeech/vits")
tmpdir = os.environ.get("EDNA_TTS_TMP", "/tmp")
use_cuda = os.environ.get("EDNA_TTS_CUDA", "0") == "1"

tts = TTS(model_name=model)
if use_cuda:
    try:
        tts = tts.to("cuda")
    except Exception:
        pass

print("READY", flush=True)

counter = 0
pid = os.getpid()

for line in sys.stdin:
    line = line.strip()
    if not line:
        print("ERR empty", flush=True)
        continue
    if line == "__quit__":
        break

    counter += 1
    out = os.path.join(tmpdir, f"edna_tts_{pid}_{counter}.wav")
    try:
        tts.tts_to_file(text=line, file_path=out)
        print(out, flush=True)
    except Exception as e:
        print("ERR " + str(e), flush=True)
)PY";

        // Build argv for execvp
        std::vector<const char*> argv;
        argv.push_back(p_.python_bin.c_str());
        argv.push_back("-u");
        argv.push_back("-c");
        argv.push_back(script.c_str());
        argv.push_back(nullptr);

        // Environment for worker
        ::setenv("EDNA_TTS_MODEL", p_.model_name.c_str(), 1);
        ::setenv("EDNA_TTS_TMP", p_.tmp_dir.c_str(), 1);
        ::setenv("EDNA_TTS_CUDA", p_.use_cuda ? "1" : "0", 1);

        ::execvp(argv[0], const_cast<char* const*>(argv.data()));
        _exit(127);
    }

    // Parent
    ::close(in_pipe[0]);   // parent keeps write end
    ::close(out_pipe[1]);  // parent keeps read end

    worker_.pid = pid;
    worker_.to_child = in_pipe[1];
    worker_.from_child = out_pipe[0];
    worker_.ready = false;

    // Handshake
    if (!worker_handshake_locked()) {
        stop_worker_locked();
        enabled_ = false;
        if (last_err_.empty()) last_err_ = "TTS worker bad hello";
        return false;
    }

    worker_.ready = true;
    enabled_ = true;
    last_err_.clear();
    return true;
}

bool CoquiTTS::worker_handshake_locked() {
    // Wait up to ~10s for READY
    std::string line;
    if (!read_line_locked(line, 10000)) {
        last_err_ = "TTS worker handshake timeout";
        return false;
    }
    if (line != "READY") {
        last_err_ = "TTS worker bad hello: '" + line + "'";
        return false;
    }
    return true;
}

void CoquiTTS::stop_worker_locked() {
    if (worker_.pid <= 0) {
        worker_ = Worker{};
        return;
    }

    // Ask it to quit politely
    const char* quit = "__quit__\n";
    (void)write_all_locked(quit, std::strlen(quit));

    if (worker_.to_child >= 0) ::close(worker_.to_child);
    if (worker_.from_child >= 0) ::close(worker_.from_child);

    // Reap
    int status = 0;
    pid_t r = ::waitpid(worker_.pid, &status, WNOHANG);
    if (r == 0) {
        // Still alive, give it a moment then kill.
        ::usleep(200 * 1000);
        r = ::waitpid(worker_.pid, &status, WNOHANG);
        if (r == 0) {
            ::kill(worker_.pid, SIGKILL);
            (void)::waitpid(worker_.pid, &status, 0);
        }
    }

    worker_ = Worker{};
}

bool CoquiTTS::write_all_locked(const char* data, size_t n) {
    if (worker_.to_child < 0) return false;
    size_t off = 0;
    while (off < n) {
        ssize_t w = ::write(worker_.to_child, data + off, n - off);
        if (w < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        off += (size_t)w;
    }
    return true;
}

bool CoquiTTS::read_line_locked(std::string& out_line, int timeout_ms) {
    out_line.clear();
    if (worker_.from_child < 0) return false;

    std::string buf;
    buf.reserve(256);

    const int fd = worker_.from_child;
    auto deadline_ms = timeout_ms;

    while (true) {
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(fd, &rfds);

        struct timeval tv;
        tv.tv_sec = deadline_ms / 1000;
        tv.tv_usec = (deadline_ms % 1000) * 1000;

        int rc = ::select(fd + 1, &rfds, nullptr, nullptr, &tv);
        if (rc < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (rc == 0) {
            return false; // timeout
        }

        char tmp[256];
        ssize_t r = ::read(fd, tmp, sizeof(tmp));
        if (r < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (r == 0) return false; // EOF

        buf.append(tmp, tmp + r);

        // Look for newline
        size_t pos = buf.find('\n');
        if (pos != std::string::npos) {
            out_line = buf.substr(0, pos);
            // trim CR
            if (!out_line.empty() && out_line.back() == '\r') out_line.pop_back();
            return true;
        }

        // Keep waiting; subtracting exact elapsed is overkill here.
        // If you care, measure time properly. Humans rarely do.
        deadline_ms = timeout_ms;
    }
}

bool CoquiTTS::play_wav_locked(const std::string& wav_path) {
    // Use aplay -D <device> <wav>
    std::vector<std::string> args;
    args.push_back(p_.aplay_bin);
    if (!p_.out_device.empty()) {
        args.push_back("-D");
        args.push_back(p_.out_device);
    }

    // Optional extra args (space-separated)
    if (!p_.aplay_extra_args.empty()) {
        // naive split, good enough for your own config strings
        std::string s = p_.aplay_extra_args;
        std::string cur;
        for (char c : s) {
            if (c == ' ') {
                if (!cur.empty()) { args.push_back(cur); cur.clear(); }
            } else cur.push_back(c);
        }
        if (!cur.empty()) args.push_back(cur);
    }

    args.push_back(wav_path);

    std::vector<char*> argv;
    argv.reserve(args.size() + 1);
    for (auto& a : args) argv.push_back(a.data());
    argv.push_back(nullptr);

    pid_t pid = ::fork();
    if (pid < 0) {
        last_err_ = "fork() failed for aplay";
        return false;
    }
    if (pid == 0) {
        ::execvp(argv[0], argv.data());
        _exit(127);
    }
    int status = 0;
    if (::waitpid(pid, &status, 0) < 0) {
        last_err_ = "waitpid() failed for aplay";
        return false;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        last_err_ = "aplay failed";
        return false;
    }
    return true;
}

bool CoquiTTS::speak(const std::string& text) {
    std::string wav_path;

    {
        std::lock_guard<std::mutex> lk(m_);

        if (!enabled_) return false;

        if (worker_.pid <= 0 || !worker_.ready) {
            if (!start_worker_locked()) return false;
        }

        std::string line = text;
        line.push_back('\n');
        if (!write_all_locked(line.c_str(), line.size())) {
            last_err_ = "Failed writing to TTS worker";
            enabled_ = false;
            stop_worker_locked();
            return false;
        }

        std::string resp;
        if (!read_line_locked(resp, 30000)) {
            last_err_ = "TTS worker timeout";
            enabled_ = false;
            stop_worker_locked();
            return false;
        }

        if (resp.rfind("ERR", 0) == 0) {
            last_err_ = "TTS worker: " + resp;
            return false;
        }

        wav_path = std::move(resp);
    } // unlock before playing audio

    // Playback can take seconds. Do not hold the mutex for it.
    return play_wav_locked(wav_path);  // make this non-locked version
}
