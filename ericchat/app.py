import gc
import json
import threading
import webbrowser
from functools import partial
from pathlib import Path
from importlib.metadata import version

import toga
from erictransformer import CHATCallArgs
from huggingface_hub import HfFileSystem
from huggingface_hub.utils import disable_progress_bars
from toga.constants import WindowState
from toga.style import Pack
from toga.style.pack import CENTER, COLUMN, LEFT, ROW

from .eric_state import EricUIState
from .message_html import render_html
from .style import EricColours
from .util import BytesCallback, ModelDetails, get_eric_chat_mlx, get_memory

VERSION = version("ericchat")

class EricChat(toga.App):
    def startup(self):
        self.resources_path = Path(self.paths.app) / "resources"
        self.icon = toga.Icon(self.resources_path / "icon.icns")

        self.model_dir = self.paths.data / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.available_gb = get_memory()
        self.state = EricUIState(self.model_dir)
        self.eric = None
        self.eric_lock = threading.Lock()
        self.current_selection = ""

        self.ui_loop = None
        self.cancel_download = False

        self.show_message_count = 32

        self.left_inner = toga.Box(direction=COLUMN, style=Pack(background_color=EricColours.DARK_RED))

        close_button = toga.Button("◀️", on_press=self.on_close, style=Pack(flex=1, margin=8, background_color=EricColours.ERIC_RED))

        self.chat_column = toga.Box(direction=COLUMN, style=Pack(flex=1, background_color=EricColours.DARK_RED))

        self.build_convo_history()

        self.left_inner.add(close_button)

        self.left_inner.add(self.chat_column)

        self.left_sc = toga.ScrollContainer(
            content=self.left_inner, horizontal=False, vertical=True, style=Pack(flex=2, background_color=EricColours.DARK_RED)
        )

        self.left_rail = toga.Box(direction=COLUMN, style=Pack(flex=0, width=40, background_color=EricColours.DARK_RED))

        self.open_btn = toga.Button(
            "▶️", on_press=self.on_open, style=Pack(flex=0, margin=0, background_color=EricColours.ERIC_RED)
        )
        self.left_rail.add(self.open_btn)
        self.left_rail.add(toga.Box(style=Pack(flex=1)))  # bottom spacer

        # WebView that shows the chat transcript as HTML
        self.web = toga.WebView(style=Pack(flex=1))

        # Input row (type + submit)
        self.input_field = toga.MultilineTextInput(
            placeholder="", style=Pack(flex=1, height=80))

        self.send_btn = toga.Button("Submit", on_press=self.on_submit, style=Pack(width=80, height=80, margin_left=4, color=EricColours.LIGHT_RED, background_color=EricColours.ERIC_RED))

        input_row = toga.Box(direction=ROW,
                             style=Pack())

        input_row.add(self.input_field)

        input_row.add(self.send_btn)

        self.loaded_model_label = toga.Label(
            "", style=Pack(text_align=LEFT, margin=(8, 8, 4, 8), color=EricColours.LIGHT_RED)
        )
        self.status_label = toga.Label(
            "", style=Pack(text_align=LEFT, margin=(0, 8, 8, 8), color=EricColours.LIGHT_RED)
        )

        self.load_hf_btn = toga.Button(
            "Model", on_press=self.on_load_model,
            style=Pack(width=180, margin=(0, 4, 0, 0), background_color=EricColours.ERIC_RED, color=EricColours.BG_LIGHT),
            enabled=True,
        )

        self.model_settings_btn = toga.Button(
            "⚙️", on_press=self.on_model_settings_btn_press,
            style=Pack(width=32, margin=0, background_color=EricColours.ERIC_RED, color=EricColours.BG_LIGHT),
            enabled=True,
        )

        self.select_model_row = toga.Box(direction=ROW, style=Pack(margin=8), children=[self.load_hf_btn, self.model_settings_btn])

        self.model_column = toga.Box(direction=COLUMN, style=Pack(margin=8))

        self.model_column.add(self.select_model_row)

        self.model_column.add(self.loaded_model_label)

        self.model_column.add(self.status_label)

        self.news_label = toga.Label(
            f"Current Version: {VERSION}", style=Pack(text_align=LEFT, margin=(0, 0, 4, 0), color=EricColours.LIGHT_RED)
        )

        self.releases_btn = toga.Button(
            "Releases 🔗", on_press=self.on_update,
            style=Pack(width=180, margin=0, background_color=EricColours.ERIC_RED, color=EricColours.BG_LIGHT),
            enabled=True
        )

        self.releases_row = toga.Box(direction=COLUMN, style=Pack(margin=0, align_items=CENTER), children=[self.news_label, self.releases_btn])


        self.button_header_row = toga.Box(direction=ROW, style=Pack(margin=8))
        self.button_header_row.add(self.model_column)
        self.button_header_row.add(toga.Box(style=Pack(flex=1)))
        self.button_header_row.add(self.releases_row)


        self.right_pane = toga.Box(direction=COLUMN, style=Pack(flex=20, background_color=EricColours.DARK_RED))

        self.load_btn = toga.Button(
            "Load",
            on_press=self._load_model_press,
            style=Pack(margin=(0, 0, 0, 8), width=128, background_color=EricColours.ERIC_RED, color=EricColours.BG_LIGHT),
        )

        self.sel = toga.Selection(items=self.state.available_models_names, style=Pack(flex=1, margin=8), on_change=self.update_sel_notice)
        self.sel.value = self.state.chosen_hf_model

        row = toga.Box(children=[self.sel], style=Pack(direction=ROW, margin=8))

        self.cancel_btn = toga.Button("Cancel", on_press=self._change_header, style=Pack(margin=0, width=128))
        self.memory_label = toga.Label(
            "", style=Pack(text_align=LEFT, margin=0, color=EricColours.LIGHT_RED)
        )

        spacer = toga.Box(style=Pack(flex=1))

        actions = toga.Box(
            children=[toga.Box(style=Pack(flex=1)),  self.cancel_btn, self.load_btn],
            style=Pack(direction=ROW, margin=8),
        )

        actions_with_memory = toga.Box(children=[self.memory_label, spacer,   actions],
                                       style=Pack(direction=ROW, align_items="center",  margin=8))

        self.notice_label = toga.Label("")
        license_box = toga.Box(children=[ self.notice_label],  flex=1,  style=Pack(margin=10))

        notice_view = toga.ScrollContainer(content=license_box, style=Pack(flex=1) )

        self.select_model_drop_down = toga.Box(children=[row, notice_view , actions_with_memory], style=Pack(direction=COLUMN, margin=8))

        # MODEL SETTINGS DROP DOWN
        # creativity slider
        starting_creativity_value = 50.0
        creativity_slider = toga.Slider(min=1, flex=10, max=100, value=starting_creativity_value, on_change=self.on_creativity_slider)
        self.creativity_label = toga.Label(
            f"Creativity: {round(starting_creativity_value)}", style=Pack(flex=0, text_align=LEFT, margin=0, color=EricColours.LIGHT_RED)
        )
        creativity_row = toga.Box(children=[self.creativity_label, creativity_slider], style=Pack(direction=ROW, margin=16))

        # token length slider
        token_length_slider = toga.Slider(min=0, flex=10, max=1, value=0.5, on_change=self.on_token_length_slider)
        self.state.set_token_length(token_length_slider.value)

        self.token_length_spaces = 0

        self.token_length_label = toga.Label(
            f"Length: {self.state.max_len}" + " " * self.token_length_spaces,
            style=Pack(flex=0, text_align=LEFT, margin=0, color=EricColours.LIGHT_RED)
        )

        length_row = toga.Box(children=[self.token_length_label, token_length_slider], style=Pack(direction=ROW, margin=(0, 16, 16, 16)))

        # cancel button
        model_settings_cancel_btn = toga.Button("Cancel", on_press=self.on_model_settings_btn_press, style=Pack(margin_left=16, width=128,  margin_bottom=8))

        self.model_settings_drop_down =  toga.Box(children=[creativity_row, length_row, model_settings_cancel_btn], style=Pack(direction=COLUMN, margin=0))

        self.progress = toga.Box(direction=COLUMN, style=Pack(margin_right=16, margin_left=24, margin_bottom=8))

        # actual fill bar (neon red)
        self.progress_fill = toga.Box(style=Pack(flex=0, background_color=EricColours.DARK_RED_L, height=8, margin_top=-8))
        self.progress_rest = toga.Box(style=Pack(flex=100, background_color=EricColours.DARK_RED, height=8, margin_top=-8))

        bar_row = toga.Box(direction=ROW)
        bar_row.add(self.progress_fill)
        bar_row.add(self.progress_rest)
        self._set_progress(0)
        self.progress.add(bar_row)

        self.right_pane.add(self.button_header_row)
        self.right_pane.add(self.progress)
        self.right_pane.add(self.web)
        self.right_pane.add(input_row)

        # Container row with sidebar
        self.inference_tab = toga.Box(direction=ROW, style=Pack(flex=1, background_color=EricColours.DARK_RED))
        self.inference_tab.add(self.left_rail)
        self.inference_tab.add(self.right_pane)

        root = toga.Box(style=Pack(flex=1, background_color=EricColours.DARK_RED))

        root.add(self.inference_tab)

        self.main_window = toga.MainWindow(title=self.formal_name)
        self.main_window.content = root
        self.main_window.show()
        self.main_window.state = WindowState.MAXIMIZED

        self.ui_loop = self.main_window.app.loop

        self._update_webview()

        mlx_cls = get_eric_chat_mlx()

        if mlx_cls is not None:
            self.eric_chat_class = mlx_cls
        else:
            self._set_status("ERROR: EricChatMLX is not compatible. Please ensure you have installed 'mlx-lm'.")
            self._set_buttons(False, False)

        self.state.new_tokens = token_length_slider.value
        self.state.set_creativity(creativity_slider.value)

        self._customize_about_command()

        self.about_window = None

    def _customize_about_command(self):
        try:
            about_cmd = self.commands[toga.Command.ABOUT]
        except KeyError:
            return

        about_cmd.action = self.action_about

    def action_about(self, widget):

        if self.about_window is not None:
            self.about_window.show()
            return

        license_path = self.resources_path / "LICENSE"
        license_text = license_path.read_text(encoding="utf-8")

        title_label = toga.Label(
            f"{self.formal_name}",
            style=Pack(
                margin_bottom=4,
                text_align=CENTER,
                font_size=14),
        )
        subtitle_label = toga.Label(
            f"Version {VERSION}",
            style=Pack(
                margin_bottom=8,
                text_align=CENTER
            ),
        )

        repository_label =  toga.Label(
            "Repository: https://github.com/EricFillion/ericchat",
            style=Pack(
                margin_bottom=8,
                text_align=CENTER
            ),
        )

        license_label = toga.Label(license_text)
        license_box = toga.Box(children=[license_label], flex=1, style=Pack(margin=10))
        license_view = toga.ScrollContainer(content=license_box, style=Pack(flex=1))

        about_box = toga.Box(
            children=[title_label, subtitle_label, repository_label, license_view],
            style=Pack(direction=COLUMN, margin=4, flex=1),
        )

        about_window = toga.Window(title="About", resizable=False)
        about_window.content = about_box
        about_window.size = (600, 400)

        # hide, don't close. This fixes a bug that was causing crashes.
        def _on_close(window):
            window.hide()
            return False

        about_window.on_close = _on_close

        self.about_window = about_window
        self.windows.add(about_window)
        about_window.show()

    def _render_chat_html(self) -> str:
        return render_html(self.state)

    def _update_webview(self):
        html = self._render_chat_html()
        self.web.set_content("http://127.0.0.1/", html)

    def _run_in_thread(self, target, *args, **kwargs):
        t = threading.Thread(target=target, args=args, kwargs=kwargs, daemon=True)
        t.start()
        return t

    def _with_ui(self, fn, *args, **kwargs):
        if self.ui_loop is not None:
            self.ui_loop.call_soon_threadsafe(lambda: fn(*args, **kwargs))

    def _set_status(self, text: str):
        self.status_label.text = text or ""

    def _set_model_name(self, model_name: str):
        self.loaded_model_label.text = model_name

    def _set_notice_label(self, notice: str):
        self.notice_label.text = notice

    def update_sel_notice(self, widget):
        model_details = self.state.available_models[self.sel.value]
        self._with_ui(self._set_notice_label, model_details.notice)

    def _set_buttons(self, enabled: bool, select_model: bool = False):
        try:
            if select_model:
                self.load_hf_btn.enabled = enabled

            for item in self.chat_column.children:
                if isinstance(item, toga.Button):
                    if item.text != "⬅️":
                        item.enabled = enabled
                if isinstance(item, toga.Box):
                    for button in item.children:
                        if isinstance(button, toga.Button):
                            button.enabled = enabled

        except Exception as e:
            self._with_ui(self._error_ui, e)
            pass

    def _set_progress(self, pct: int, text: str = ""):
        pct = max(0, min(100, int(pct)))
        self.progress_fill.style.flex = pct
        self.progress_rest.style.flex = 100 - pct
        if text:
            self._set_status(text)

    def _reset_progress(self):
        self.progress_fill.style.flex = 0
        self.progress_rest.style.flex = 100

    def _apply_stream_piece_ui(self, piece):
        self.state.stream_step(piece)
        if self.state.should_update_ui:
            self._update_webview()

    def _finish_stream_ui(self):
        self.state.finish_chat()
        self._update_webview()
        self._set_status("Ready.")
        self._set_buttons(True, True)
        self._with_ui(self._adjust_send_button_text, "Submit")


    def _error_ui(self, e):
        self._update_webview()
        self._set_status(f"Error: {e}")
        self._set_buttons(True, True)

    def remove_button_header_row(self):
        self.button_header_row.remove(self.releases_row)

    def _unload_model(self):
        old = self.eric
        self.eric = None

        if old is not None:
            try:
                old.model = None
                old.tokenizer = None
                old.text_streamer_handler = None
                del old
            except Exception as e:
                self._with_ui(self._error_ui, e)
        gc.collect()
        gc.collect()

    def _do_inference(self, messages_snapshot):
        # Grab the model pointer safely, then release the lock

        def _remove_releases():
            if self.releases_row in self.button_header_row.children:
                self.remove_button_header_row()

        self._with_ui(_remove_releases)

        with self.eric_lock:
            model = self.eric

        if model is None:
            self._with_ui(self._set_status, "Select a model.")
            self._with_ui(self._set_buttons, True, True)
            return

        try:
            # Do NOT hold eric_lock while streaming; just use the model
            for piece in model.stream(messages_snapshot, args=CHATCallArgs(max_len=self.state.max_len,
                                                                           top_k=self.state.top_k,  # always 0. We only adjust temperature and top_p
                                                                           temp=self.state.temp,
                                                                           top_p=self.state.top_p
                                                                           )):
                if self.state.cancel_inference:
                    return
                # Schedule each piece to the UI thread
                self._with_ui(self._apply_stream_piece_ui, piece)

        except Exception as e:
            self._with_ui(self._error_ui, e)
            return
        finally:
            # Always finalize on the UI thread
            self._with_ui(self._finish_stream_ui)

    def _switch_to_cancel_button(self):
        self.load_hf_btn.text = "Cancel"
        self.load_hf_btn.on_press = self.on_cancel_download

    def _switch_to_select_model_button(self):
        self.load_hf_btn.text = "Model"
        self.load_hf_btn.on_press = self.on_load_model
        self.cancel_download = False

    def _load_model(self, model_details: ModelDetails):

        self.state.current_short_name = model_details.short_name

        self._with_ui(self._set_model_name, model_details.short_name)

        self.check_redownload = False # for debugging

        if not model_details.is_downloaded or self.check_redownload:

            fs = HfFileSystem()
            # list remote files (use a fixed revision if you want determinism)
            entries = fs.find(model_details.hf_id, revision="main", detail=True)  # dict[rpath] -> info

            exists, missing, wrong_size = [], [], []
            fetch_total = 0

            for rpath, info in entries.items():
                base = Path(rpath).name
                file_path = model_details.save_path / base
                expected_size = int((info or {}).get("size", 0) or 0)

                if file_path.is_file():
                    actual_size = file_path.stat().st_size
                    if expected_size and expected_size != actual_size:
                        wrong_size.append([rpath, base, expected_size])  # keep remote path for fetch
                        fetch_total += expected_size
                    else:
                        exists.append([rpath, base, expected_size])
                else:
                    missing.append([rpath, base, expected_size])
                    fetch_total += expected_size

            fetch_files = missing + wrong_size

            fetch_total_gb = round(fetch_total/(1024*1024*1024), 3)
            try:
                self._with_ui(self._switch_to_cancel_button)
                disable_progress_bars()
                self._with_ui(self._reset_progress)
                self._with_ui(self._set_status, "Downloading: ")

                lock = threading.Lock()
                download_state = {"downloaded": 0}
                per_file_prev = {}

                for rpath, base, size in fetch_files:
                    lpath = str(model_details.save_path / base)

                    per_file_prev[base] = 0  # reset per-file cursor

                    def set_progress(n, total, label=base):
                        if self.cancel_download:
                            self._with_ui(self._set_progress, 0, "Cancelled download")
                            raise Exception("")

                        # accumulate deltas across all files
                        with lock:
                            prev = per_file_prev.get(label, 0)
                            delta = n - prev if n >= prev else n  # guard if n resets
                            per_file_prev[label] = n
                            download_state["downloaded"] += delta

                            pct = int(download_state["downloaded"] * 100 / fetch_total) if fetch_total else 0

                            gb = round(download_state["downloaded"]/(1024*1024*1024), 3)

                        self._with_ui(self._set_progress, pct, f"Downloading: {pct}%. {gb} GB / {fetch_total_gb} GB")

                    fs.get_file(
                        rpath=rpath,  # full remote path
                        lpath=lpath,  # local file (basename)
                        callback=BytesCallback(partial(set_progress, label=base)),
                        revision="main",
                        chunk_size=8 * 1024 * 1024,
                    )

                # Ensure progress bar reaches 100%
                if fetch_total and not self.cancel_download:
                    self._with_ui(
                        self._set_progress, 100,
                        f"Downloading: 100% — {fetch_total_gb} GB / {fetch_total_gb} GB"
                    )

            except Exception as e:
                if self.cancel_download:
                    self._with_ui(self._set_status, "Cancelled Download")
                else:
                    self._with_ui(self._set_status, f"Failed to download: {e}")

                self._with_ui(self._set_buttons, True, True)

                return
            finally:
                self._with_ui(self._switch_to_select_model_button)
                pass
        else:
            pass


        try:
            self._set_buttons(False, True)

            with self.eric_lock:
                self._with_ui(self._set_status, "Initializing...")

                self._unload_model()
                self.eric = self.eric_chat_class(model_name=str(model_details.save_path))

                if not model_details.is_downloaded:
                    details_payload = {"model_name": model_details.hf_id}
                    model_details.details_path.write_text(json.dumps(details_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                    self.state.update_available_models_datasets()

            self.state.chosen_hf_model = model_details.name.replace("🔗", "💾")
            self._with_ui(self._set_progress, 100, "Ready")


        except Exception as e:
            self._with_ui(self._set_status, f"Failed to initialize: {e}")
        finally:
            self._with_ui(self._set_buttons, True, True)

    def on_submit(self, widget):
        if self.state.in_inference:
            self.state.cancel_inference = True
            return

        text = (self.input_field.value or "").strip()
        if not text:
            return

        if self.eric is not None:
            self._with_ui(self._adjust_send_button_text, "Cancel")
            self.state.in_inference = True

        else:
            self._set_status("Please select a model.")

        # Update UI (UI thread)
        messages = self.state.user_input(text)  # build stable snapshot
        self.build_convo_history()
        self.input_field.value = ""

        self._update_webview()
        if self.eric is None:
            self._set_status("Error: no model loaded")
            return

        self._set_status("Generating...")
        self._set_buttons(False, True)

        # Background thread only does model.stream, not UI
        self._run_in_thread(self._do_inference, messages)

    def on_cancel_download(self, widget):
        self.cancel_download = True

    def _load_model_press(self, widget):
        self.cancel_download = False
        self._with_ui(lambda: self.on_load_model(""))

        model_name = self.sel.value
        if not model_name:
            self._set_status("Please pick a model.")
            return

        if model_name not in self.state.available_models.keys():
            self._set_status("Invalid model")
            return

        model_details = self.state.available_models[model_name]
        self._with_ui(self._set_progress, 0, f"Loading model: {model_name}...")
        self._set_buttons(False, False)

        def _run():
            try:
                self._load_model(model_details)
            finally:
                pass
        threading.Thread(target=_run, daemon=True).start()


    def on_load_model(self, widget):
        self.current_selection = "MODEl"

        self._with_ui(self._set_notice_label, "Loading...")

        models = self.state.available_models
        model_names = []

        for model_name, model_details in models.items():
            if model_details.required_memory < self.available_gb:
                model_names.append(model_name)
            else:
                pass

        if len(model_names) == 0:
            minimal_required_gb = min(md.required_memory for md in models.values())
            self._set_status(f"Not enough memory. Only {round(self.available_gb, 2)} GB are available. {minimal_required_gb} GB are required.")
            return

        if self.state.chosen_hf_model not in model_names:
            model_names.insert(0, self.state.chosen_hf_model)

        self.sel.items = model_names

        self.sel.value = self.state.chosen_hf_model

        notice = self.state.available_models[self.state.chosen_hf_model].notice
        self._with_ui(self._set_notice_label, notice)


        self.load_btn.on_press = self._load_model_press

        if self.eric is None:
            self.available_gb = get_memory()
            self.memory_label.text = f"{round(self.available_gb, 2)} GB of available memory"
        else:
            # no longer accurate after changes the model
            self.memory_label.text = ""


        self._change_header("")

    def _change_header(self, widget):
        self.state.update_available_models_datasets()

        if self.select_model_drop_down in self.right_pane.children:
            self.right_pane.remove(self.select_model_drop_down)
            self.right_pane.insert(0, self.button_header_row)
            self._with_ui(self._set_notice_label, "")

        else:
            self.right_pane.insert(0, self.select_model_drop_down)
            self.right_pane.remove(self.button_header_row)

    def on_close(self, widget):
        if self.left_sc in self.inference_tab.children:
            self.inference_tab.remove(self.left_sc)
            self.inference_tab.insert(0, self.left_rail)

    def on_open(self, widget):
        if self.left_rail in self.inference_tab.children:
            self.inference_tab.remove(self.left_rail)
            self.inference_tab.insert(0, self.left_sc)


    def on_model_settings_btn_press(self, widget):
        # get rid of model settings
        if self.model_settings_drop_down in self.right_pane.children:
            self.right_pane.remove(self.model_settings_drop_down)
            self.right_pane.insert(0, self.button_header_row)
            self.right_pane.insert(1, self.progress)

        # bring up model settings
        else:
            self.right_pane.insert(0, self.model_settings_drop_down)
            self.right_pane.remove(self.button_header_row)
            self.right_pane.remove(self.progress)

    def on_update(self, widget):
        url = "https://github.com/EricFillion/ericchat/releases"
        webbrowser.open_new_tab(url)

    def build_convo_history(self):
        self.chat_column.clear()
        self.chat_column.add(toga.Button("New Convo", on_press=self.new_convo, style=Pack(flex=1, margin_top=8, margin_bottom=8, margin_left=4, margin_right=4,
                                                                                          background_color=EricColours.ERIC_RED)))
        buttons = []
        for i, convo in enumerate(self.state.convo_histories[:self.show_message_count]):
            current_chat = self.state.current_convo_index ==i
            if current_chat:
                user_text = "CURRENT"
            elif convo:
                user_text = convo[0].text[:10]
            else:
                user_text = "Empty"

            button_row = toga.Box(direction=ROW,
                                 style=Pack(flex=0))
            chat_button = toga.Button(user_text, on_press=partial(self.change_convo, i), style=Pack(flex=4, margin_top=8, margin_left=4, margin_right=0,
                                                                                                    background_color=EricColours.ERIC_RED if not current_chat else EricColours.DARK_RED_L))

            delete_button = toga.Button("🗑️", on_press=partial(self.delete_convo, i),
                                        style=Pack(flex=1, margin_top=8, margin_left=4, margin_right=4,
                                                   background_color=EricColours.ERIC_RED if not current_chat else EricColours.DARK_RED_L))

            button_row.add(chat_button)
            button_row.add(delete_button)

            buttons.append(button_row)

        for button in reversed(buttons):
            self.chat_column.add(button)

        if len(self.state.convo_histories) > self.show_message_count:
            increase_count_button = toga.Button("See more", on_press=self.see_more, style=Pack(flex=4, margin_top=8, margin_left=4, margin_right=0,
                                                                                               background_color=EricColours.ERIC_DARK_SILVER))
            self.chat_column.add(increase_count_button)

    def new_convo(self, widget):
        self.state.new_convo()
        self.build_convo_history()
        self._with_ui(self._update_webview)

    def change_convo(self, index, widget):
        self.state.change_convo(index)
        self.build_convo_history()
        self._with_ui(self._update_webview)

    def delete_convo(self, index, widget):
        self.state.delete_convo(index)
        if self.state.current_convo_index +1 == index:
            if len(self.state.convo_histories) == 0:
                self.state.new_convo()

            self.state.change_convo(len(self.state.convo_histories)-1)
        self.build_convo_history()
        self._with_ui(self._update_webview)

    def see_more(self, widget):
        self.show_message_count += 32
        self.build_convo_history()

    def on_creativity_slider(self, slider):
        self.state.set_creativity(slider.value)
        self.creativity_label.text = f"Creativity: {round(slider.value)}"

    def on_token_length_slider(self, slider):
        self.state.set_token_length(slider.value)
        self.token_length_label.text = f"Length: {self.state.max_len}" +  " " * self.token_length_spaces

    def _adjust_send_button_text(self, text: str):
        self.send_btn.text = text


def main():
    return EricChat(
        formal_name="Eric Chat",
        app_id="com.ericchat.app",
    )


def run():
    main().main_loop()


if __name__ == "__main__":
    run()