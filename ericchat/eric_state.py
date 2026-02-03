from pathlib import Path
from typing import List

from erictransformer import CHATStreamResult

from .util import ChatMessage, TPSTracker, available_model_factory


class EricUIState:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir

        self.available_models, self.chosen_hf_model = available_model_factory(model_dir)
        self.current_short_name = ""

        self.available_models_names = self.available_models.keys()

        self.tps_tracker = TPSTracker()
        self.tps = 0
        self.model_ready = False

        self.previous_marker_type = ""
        self.current_marker_stream: ChatMessage = ChatMessage()

        # example data for now
        self.convo_history: List[ChatMessage] = []
        self.stream_marker_i = 0

        self.should_update_ui = False
        self.in_inference = False
        self.cancel_inference = False

        self.convo_histories: List[List[ChatMessage]]= []

        self.current_convo_index = 0

        self.new_convo()

        #text gen params
        self.max_len = 2048
        self.top_p = 0.8
        self.temp = 0.7
        self.top_k = 0 # we don't adjust this


    def update_available_models_datasets(self):
        self.available_models, _ = available_model_factory(self.model_dir)
        self.available_models_names = self.available_models.keys()

    def _reset_state(self):
        self.current_marker_stream = ChatMessage()
        self.previous_marker_type = ""
        self.stream_marker_i = 0

    def user_input(self, text: str):
        # reset current_messages again just in-case finish_chat() is skipped due to an error
        self._reset_state()
        self.convo_history.append(ChatMessage(text=text, marker="", expanded_text="", role="user"))

        out = []
        for msg in self.convo_history:
            if msg.role == "assistant" and msg.marker == "text":
                out.append({"role": "assistant", "content": msg.text})
            elif msg.role == "user":
                out.append({"role": "user", "content": msg.text})

        return out


    def _submit_chat(self):
        if self.cancel_inference:
            if self.current_marker_stream.marker in ("thinking", "think_start", "think_end", "special"):
                self.current_marker_stream.text = f"**Cancelled. Thinking tokens:**\n\n {self.current_marker_stream.expanded_text}"
            else:
                self.current_marker_stream.text = self.current_marker_stream.text

        elif self.current_marker_stream.marker == "thinking":
            self.current_marker_stream.text = f"**Ran out of tokens while thinking:**\n\n {self.current_marker_stream.expanded_text}"

        self.convo_history.append(self.current_marker_stream)
        self._reset_state()

    def stream_step(self, step: CHATStreamResult):
        update_ui_marker = False
        self.tps = self.tps_tracker.step()

        if step.marker == "think_start":
            self.current_marker_stream.text="Thinking..."
            self.current_marker_stream.marker="thinking"
            self.current_marker_stream.expanded_text+=""
            self.current_marker_stream.role="assistant"
            self.current_marker_stream.tps = self.tps
            update_ui_marker = True

        elif step.marker == "thinking":
            self.current_marker_stream.expanded_text += step.text
            self.current_marker_stream.tps = self.tps

        elif step.marker == "think_end":
            update_ui_marker = True
            self.current_marker_stream.tps = self.tps

        elif step.marker == "text":
            if self.previous_marker_type != "text":
                self.should_update_ui = True
                self.current_marker_stream = ChatMessage(text=step.text,
                                                         marker="text",
                                                         expanded_text="",
                                                         role="assistant")
            else:
                self.current_marker_stream.text += step.text
                self.current_marker_stream.tps = self.tps

        elif step.marker == "think_end":
            update_ui_marker = True

        if update_ui_marker or (self.stream_marker_i % 32 ==0):
            self.should_update_ui = True
        else:
            self.should_update_ui = False
        self.previous_marker_type = self.current_marker_stream.marker

        self.stream_marker_i +=1

    def finish_chat(self):

        self._submit_chat()
        self.tps_tracker.reset() # this way if text or thinking are first we have a fresh state
        self.cancel_inference = False
        self.in_inference = False

    def new_convo(self):
        self.convo_histories.append([])
        self.current_convo_index = len(self.convo_histories) - 1
        self.convo_history = self.convo_histories[self.current_convo_index]

    def delete_convo(self, index: int):
        self.convo_histories.pop(index)
        if self.current_convo_index >= index:
            self.current_convo_index -= 1

    def change_convo(self, index: int):
        self.convo_history = self.convo_histories[index]
        self.current_convo_index = index

    def update_convo(self, index: int, convo: ChatMessage):
        self.convo_histories[index].append(convo)

    def set_token_length(self, max_len: float):
        # back-load from 1 to 8096
        gamma =  2.0002642 # at 0.5 it's 4096
        self.max_len = int(1 + (16384 - 1) * (float(max_len) ** gamma))

    def set_creativity(self, creativity: float):
        # just in-case there's a bug we restrict its value
        c = max(1.0, min(100.0, float(creativity)))

        if c <= 1.0:
            self.top_p = 0
            self.top_k = 0

        else:
            # normalize to [0, 1]
            normalized_c = (c - 1.0) / 99.0
            # back loaded with range 0.2 to 1.2
            temp_gamma = 1.3
            self.temp = 0.2 + (normalized_c ** temp_gamma) * (1.2 - 0.2)

            # front loaded with range 0.4 to 1.0
            self.top_p = 0.4 + 0.6 * (normalized_c ** 0.55)
            # safety clamps
            self.top_p = min(max(self.top_p, 0.0), 1.0)

