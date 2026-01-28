#pragma once

#include <functional>
#include <mutex>
#include <string>

/*
 * EdnaStateMachine
 */

class EdnaStateMachine
{
public:
    enum class State {
        Boot,
        AwaitSpeech,
        CapturingSpeech,
        Transcribing,
        Thinking,
        Speaking,
        Error,
        Shutdown,
    };

    enum class Event {
        Start,
        SpeechStart,
        SpeechEndQueued,
        TranscriptReady,
        ReplyReady,
        TtsDone,
        Stop,
        NoCommand,
        Fail,
    };

    struct Config {
    };

    using Observer = std::function<void(State from, State to, Event why, const std::string& note)>;

    EdnaStateMachine();
    explicit EdnaStateMachine(const Config& cfg);

    ~EdnaStateMachine() = default;

    // Current state snapshot.
    State state() const;

    // Begin operation
    void start();

    // Dispatch an event. Optional note is for debugging/logging.
    void dispatch(Event ev, const std::string& note = "");

    // Subscribe to transitions (called on every transition).
    void set_observer(Observer obs);

    static const char* state_name(State s);
    static const char* event_name(Event e);

private:
    State transition_locked(State cur, Event ev, const std::string& note, bool& did_transition);

    Config cfg_;
    mutable std::mutex m_;
    State st_ = State::Boot;
    Observer obs_;
};
