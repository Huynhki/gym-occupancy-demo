"""
Occupied/Free State Machine (V.5) + Time Accumulation (V.6)
"""


class EquipmentStateMachine:
    """
    One instance tracks one piece of equipment across the frame sequence.
    Implements the count_in / count_out hysteresis logic + the start/end
    backdating correction described in V.5.a-b.
    """

    def __init__(self, equip_id, equip_type, t_in=3, t_out=5, score_threshold=0.5):
        self.id = equip_id
        self.type = equip_type
        self.t_in = t_in
        self.t_out = t_out
        self.threshold = score_threshold

        self.state = "Free"
        self.count_in = 0
        self.count_out = 0
        self.sessions = []  # list of [start_frame, end_frame)  (end exclusive)
        self._pending_start = None

    def step(self, frame_idx, max_score_this_frame):
        E = 1 if max_score_this_frame >= self.threshold else 0

        if self.state == "Free":
            if E == 1:
                self.count_in += 1
                if self.count_in >= self.t_in:
                    self.state = "Occupied"
                    self._pending_start = frame_idx - self.t_in + 1
                    self.count_out = 0
            else:
                self.count_in = 0
        else:  # Occupied
            if E == 1:
                self.count_out = 0
            else:
                self.count_out += 1
                if self.count_out >= self.t_out:
                    end_frame = frame_idx - self.t_out + 1
                    self.sessions.append([self._pending_start, end_frame])
                    self._pending_start = None
                    self.state = "Free"
                    self.count_in = 0

    def finalize(self, total_frames):
        """Close any still-open session at the end of the video (V.6: end = N+1)."""
        if self.state == "Occupied" and self._pending_start is not None:
            self.sessions.append([self._pending_start, total_frames + 1])
            self._pending_start = None

    def total_occupied_frames(self):
        return sum(max(0, end - start) for start, end in self.sessions)


def accumulate_time(state_machine, fps_sample_rate=1.0):
    """
    V.6 Time Computation: frames -> seconds.
    Since the pipeline samples at `fps_sample_rate` frames/sec (default 1
    FPS per the proposal), each sampled frame represents 1/fps_sample_rate
    seconds.
    """
    frames = state_machine.total_occupied_frames()
    seconds = frames / fps_sample_rate
    return seconds
