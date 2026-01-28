#include "state_machine.hpp"

// Default constructor: delegate to the Config constructor with defaults.
EdnaStateMachine::EdnaStateMachine()
    : EdnaStateMachine(Config{}) {}

EdnaStateMachine::EdnaStateMachine(const Config& cfg) : cfg_(cfg) {}

EdnaStateMachine::State EdnaStateMachine::state() const {
    std::lock_guard<std::mutex> lk(m_);
    return st_;
}

void EdnaStateMachine::set_observer(Observer obs) {
    std::lock_guard<std::mutex> lk(m_);
    obs_ = std::move(obs);
}

void EdnaStateMachine::start() {
    dispatch(Event::Start, "start()");
}

void EdnaStateMachine::dispatch(Event ev, const std::string& note) {
    Observer obs_copy;
    State from, to;
    bool did = false;

    {
        std::lock_guard<std::mutex> lk(m_);
        from = st_;
        to = transition_locked(st_, ev, note, did);
        if (did) st_ = to;
        obs_copy = obs_;
    }

    // Invoke observer outside lock.
    if (did && obs_copy) {
        obs_copy(from, to, ev, note);
    }
}

EdnaStateMachine::State
EdnaStateMachine::transition_locked(EdnaStateMachine::State cur,
                                    EdnaStateMachine::Event ev,
                                    const std::string& note,
                                    bool& did_transition)
{
    did_transition = false;

    switch (cur) {
        case State::Boot:
            if (ev == Event::Start) { did_transition = true; return State::AwaitSpeech; }
            break;
    
        case State::AwaitSpeech:
            if (ev == Event::SpeechStart) { did_transition = true; return State::CapturingSpeech; }
            break;
    
        case State::CapturingSpeech:
            if (ev == Event::SpeechEndQueued) { did_transition = true; return State::Transcribing; }
            break;
    
        case State::Transcribing:
            if (ev == Event::TranscriptReady) {
                did_transition = true;
                return State::Thinking;
            }
            if (ev == Event::NoCommand) {
                did_transition = true;
                return State::AwaitSpeech;
            }
            break;

        case State::Thinking:
            if (ev == Event::ReplyReady) {
                did_transition = true;
                return State::Speaking;
            }
            if (ev == Event::NoCommand) {
                did_transition = true;
                return State::AwaitSpeech;
            }
            break;
        
        case State::Speaking:
            if (ev == Event::TtsDone) { did_transition = true; return State::AwaitSpeech; }
            break;
    
        case State::Error:
            // optional: let Start reset it
            if (ev == Event::Start) { did_transition = true; return State::AwaitSpeech; }
            break;
    
        case State::Shutdown:
            break;
    }

    return cur;
}

const char* EdnaStateMachine::state_name(State s) {
    switch (s) {
        case State::Boot:            return "Boot";
        case State::AwaitSpeech:     return "AwaitSpeech";
        case State::CapturingSpeech: return "CapturingSpeech";
        case State::Transcribing:    return "Transcribing";
        case State::Thinking:        return "Thinking";
        case State::Speaking:        return "Speaking";
        case State::Error:           return "Error";
        case State::Shutdown:        return "Shutdown";
    }
    return "Unknown";
}

const char* EdnaStateMachine::event_name(Event e) {
    switch (e) {
        case Event::Start:          return "Start";
        case Event::SpeechStart:    return "SpeechStart";
        case Event::SpeechEndQueued:return "SpeechEndQueued";
        case Event::TranscriptReady:return "TranscriptReady";
        case Event::ReplyReady:     return "ReplyReady";
        case Event::TtsDone:        return "TtsDone";
        case Event::Stop:           return "Stop";
        case Event::NoCommand:      return "NoCommand";
        case Event::Fail:           return "Fail";
    }
    return "Unknown";
}
