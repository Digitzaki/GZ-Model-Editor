"""Tkinter front end for GZModel Editor."""
from __future__ import annotations

import contextlib
import io
import os
import queue
import sys
import threading
import tkinter as tk
import traceback
import webbrowser
from argparse import Namespace
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

import bridge
from update_checker import APP_VERSION, check_for_updates


HELP_URL = "https://docs.google.com/document/d/1Oe7O1mZ-LYBH_4bLOiNAoCP4P42aeKuYhTO_qcTBdfI/edit?usp=sharing"

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    BaseTk = TkinterDnD.Tk
    HAS_DND = True
except Exception:
    DND_FILES = None
    BaseTk = tk.Tk
    HAS_DND = False


class QueueWriter(io.TextIOBase):
    def __init__(self, emit) -> None:
        super().__init__()
        self.emit = emit

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        if text:
            self.emit(str(text))
        return len(text)

    def flush(self) -> None:
        return None


class Tooltip:
    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self.tip: tk.Toplevel | None = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, _event=None) -> None:
        if self.tip is not None:
            return
        x = self.widget.winfo_rootx() + 18
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        label = ttk.Label(self.tip, text=self.text, padding=(8, 5), relief="solid", borderwidth=1)
        label.grid()

    def hide(self, _event=None) -> None:
        if self.tip is not None:
            self.tip.destroy()
            self.tip = None


class BridgeGui(BaseTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("GZModel Editor")
        self._set_window_icon()
        self.geometry("760x400")
        self.minsize(680, 340)
        self.messages: queue.Queue[tuple[str, str]] = queue.Queue()
        self.completions: queue.Queue[
            tuple[str, ttk.Button, str, str | None, Callable[[], None] | None]
        ] = queue.Queue()
        self.update_results: queue.Queue[tuple[dict | None, str | None]] = queue.Queue()
        self.update_buttons: list[ttk.Button] = []
        self.status_boxes: dict[str, scrolledtext.ScrolledText] = {}
        self.export_button: ttk.Button | None = None
        self.import_button: ttk.Button | None = None
        self.quick_export_button: ttk.Button | None = None
        self.quick_export_args: Namespace | None = None
        self.custom_button: ttk.Button | None = None
        self._build()
        self.after(100, self._drain_messages)

    def _set_window_icon(self) -> None:
        base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
        icon_path = base / "gz.ico"
        if icon_path.exists():
            self.iconbitmap(default=str(icon_path))

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        tabs = ttk.Notebook(self)
        tabs.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.export_mode = tk.StringVar(value="BDG")
        self.import_mode = tk.StringVar(value="BDG")
        self.export_vars = {"input": tk.StringVar(), "anim": tk.StringVar(), "out": tk.StringVar()}
        self.import_vars = {"fbx": tk.StringVar(), "original": tk.StringVar(), "anim": tk.StringVar(), "out": tk.StringVar()}
        self.import_keep_mesh = tk.BooleanVar(value=False)
        self.custom_vars = {"fbx": tk.StringVar(), "template": tk.StringVar(), "out": tk.StringVar()}
        self.custom_bind_root = tk.BooleanVar(value=False)
        tabs.add(self._export_tab(tabs), text="Export")
        tabs.add(self._import_tab(tabs), text="Import")
        tabs.add(self._custom_tab(tabs), text="Custom Model")

    def _open_help(self) -> None:
        if not webbrowser.open_new_tab(HELP_URL):
            messagebox.showerror(
                "Help",
                f"The help page could not be opened.\n\n{HELP_URL}",
            )

    def _check_for_updates(self) -> None:
        self._set_update_running(True)

        def worker() -> None:
            try:
                result = check_for_updates()
            except Exception as exc:
                self.update_results.put((None, str(exc)))
                return
            self.update_results.put((result, None))

        threading.Thread(target=worker, daemon=True).start()

    def _set_update_running(self, running: bool) -> None:
        for button in self.update_buttons:
            button.configure(
                state="disabled" if running else "normal",
                text="Checking..." if running else "Check for Updates",
            )

    def _finish_update_check(self, result: dict | None, error: str | None) -> None:
        self._set_update_running(False)
        if error:
            messagebox.showerror("Check for Updates", error)
            return
        if result is None:
            messagebox.showerror("Check for Updates", "No update information was returned.")
            return

        status = result.get("status")
        if status == "update":
            latest = result.get("latest_tag") or "the latest release"
            if result.get("reason") == "checksum":
                summary = f"A newer {latest} mini-fix is available."
            else:
                summary = f"GZModel Editor {latest} is available."
            if messagebox.askyesno(
                "Update Available",
                f"{summary}\n\nCurrent version: v{APP_VERSION}\n\nOpen the download page?",
            ):
                webbrowser.open_new_tab(str(result.get("download_url") or result.get("release_url")))
            return
        if status == "ahead":
            messagebox.showinfo(
                "Check for Updates",
                f"This v{APP_VERSION} build is newer than the latest published release "
                f"({result.get('latest_tag') or 'none'}).",
            )
            return
        if status == "no_release":
            messagebox.showinfo(
                "Check for Updates",
                "No published GitHub release is available yet. Draft releases are not visible to public builds.",
            )
            return
        if status == "unknown_tag":
            latest = result.get("latest_tag") or "unknown"
            if messagebox.askyesno(
                "Check for Updates",
                f"The latest published release uses tag {latest}, which cannot be compared with "
                f"v{APP_VERSION}. No matching executable checksum was available.\n\nOpen the release page?",
            ):
                webbrowser.open_new_tab(str(result.get("release_url")))
            return
        checksum_text = " The executable checksum also matches." if result.get("checksum_checked") else ""
        messagebox.showinfo(
            "Check for Updates",
            f"GZModel Editor v{APP_VERSION} is up to date.{checksum_text}",
        )

    def _row(self, parent: ttk.Frame, row: int, label: str, var: tk.StringVar, browse, drop_kind: str = "file", tooltip: str | None = None):
        label_text = label if label.endswith(":") else f"{label}:"
        label_frame = ttk.Frame(parent)
        label_frame.grid(row=row, column=0, sticky="w", pady=3)
        label_widget = ttk.Label(label_frame, text=label_text)
        label_widget.grid(row=0, column=0, sticky="w")
        if tooltip:
            help_label = tk.Label(
                label_frame,
                text="?",
                width=2,
                cursor="question_arrow",
                relief="solid",
                borderwidth=1,
                bg="#f5f5f5",
            )
            help_label.grid(row=0, column=1, sticky="w", padx=(5, 0))
            Tooltip(help_label, tooltip)
        entry = ttk.Entry(parent, textvariable=var)
        entry.grid(row=row, column=1, sticky="ew", padx=6, pady=3)
        self._enable_drop(entry, var, drop_kind)
        button = ttk.Button(parent, text="Browse", command=browse)
        button.grid(row=row, column=2, sticky="ew", pady=3)
        return label_frame, entry, button, label_widget

    def _mode_box(self, parent: ttk.Frame, row: int, var: tk.StringVar, command) -> ttk.Frame:
        box = ttk.Frame(parent)
        box.grid(row=row, column=0, sticky="nw", pady=(8, 2))
        ttk.Label(box, text="Mode").grid(row=0, column=0, sticky="w")
        combo = ttk.Combobox(box, textvariable=var, values=("BDG", "CMP/CMG"), state="readonly", width=12)
        combo.grid(row=1, column=0, sticky="w", pady=(2, 0))
        combo.bind("<<ComboboxSelected>>", lambda _event: command())
        ttk.Button(box, text="Help", command=self._open_help).grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(6, 0),
        )
        update_button = ttk.Button(box, text="Check for Updates", command=self._check_for_updates)
        update_button.grid(row=3, column=0, sticky="ew", pady=(6, 0))
        self.update_buttons.append(update_button)
        return box

    def _export_tab(self, tabs: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(tabs, padding=8)
        frame.columnconfigure(1, weight=1)
        self.export_input_widgets = self._row(
            frame,
            0,
            "_Shapes File",
            self.export_vars["input"],
            self._pick_export_input,
        )
        self.export_anim_widgets = self._row(
            frame,
            1,
            "Character.BDG",
            self.export_vars["anim"],
            lambda: self._pick_file(self.export_vars["anim"], [("BDG files", "*.BDG *.bdg"), ("All files", "*.*")]),
            tooltip="Optional, but required for Skeleton Edits & Animation.",
        )
        self._row(frame, 2, "Output folder", self.export_vars["out"], lambda: self._pick_folder(self.export_vars["out"]), "folder")
        frame.rowconfigure(3, weight=1)
        self._mode_box(frame, 3, self.export_mode, self._refresh_export_mode)
        self.status_boxes["export"] = self._status_box(frame, 3)
        self.export_button = ttk.Button(frame, text="Export to FBX", command=self._run_export)
        self.export_button.grid(row=3, column=2, sticky="n", pady=(8, 2))
        self._refresh_export_mode()
        return frame

    def _import_tab(self, tabs: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(tabs, padding=8)
        frame.columnconfigure(1, weight=1)
        self._row(frame, 0, "Edited FBX", self.import_vars["fbx"], lambda: self._pick_file(self.import_vars["fbx"], [("FBX files", "*.fbx"), ("All files", "*.*")]))
        self.import_original_widgets = self._row(frame, 1, "_Shapes File", self.import_vars["original"], self._pick_import_original)
        self.import_anim_widgets = self._row(
            frame,
            2,
            "Character.BDG",
            self.import_vars["anim"],
            lambda: self._pick_file(self.import_vars["anim"], [("BDG files", "*.BDG *.bdg"), ("All files", "*.*")]),
            tooltip="Optional, but required for Skeleton Edits & Animation.",
        )
        self._row(frame, 3, "Output folder", self.import_vars["out"], lambda: self._pick_folder(self.import_vars["out"]), "folder")
        ttk.Checkbutton(
            frame,
            text="Keep weighted mesh in place when moving bones",
            variable=self.import_keep_mesh,
        ).grid(row=4, column=1, sticky="w", padx=6, pady=(3, 0))
        frame.rowconfigure(5, weight=1)
        self._mode_box(frame, 5, self.import_mode, self._refresh_import_mode)
        self.status_boxes["import"] = self._status_box(frame, 5)
        import_actions = ttk.Frame(frame)
        import_actions.grid(row=5, column=2, sticky="n", pady=(8, 2))
        self.import_button = ttk.Button(import_actions, text="Import from FBX", command=self._run_import)
        self.import_button.grid(row=0, column=0, sticky="ew")
        self.quick_export_button = ttk.Button(
            import_actions,
            text="Re-Export",
            command=self._run_quick_export,
            state="disabled",
        )
        self.quick_export_button.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        Tooltip(
            self.quick_export_button,
            "Export the files created by the last successful import back to FBX.",
        )
        for var in (*self.import_vars.values(), self.import_mode, self.import_keep_mesh):
            var.trace_add("write", self._invalidate_quick_export)
        self._refresh_import_mode()
        return frame

    def _custom_tab(self, tabs: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(tabs, padding=8)
        frame.columnconfigure(1, weight=1)
        self._row(frame, 0, "Replacement FBX", self.custom_vars["fbx"], lambda: self._pick_file(self.custom_vars["fbx"], [("FBX files", "*.fbx *.FBX"), ("All files", "*.*")]))
        self._row(frame, 1, "Template CMP / BDG", self.custom_vars["template"], lambda: self._pick_file(self.custom_vars["template"], [("Template files", "*.cmp *.CMP *.bdg *.BDG"), ("All files", "*.*")]))
        self._row(frame, 2, "Output folder", self.custom_vars["out"], lambda: self._pick_folder(self.custom_vars["out"]), "folder")
        ttk.Checkbutton(
            frame,
            text="Temporarily bind unweighted vertices (testing only)",
            variable=self.custom_bind_root,
        ).grid(row=3, column=1, sticky="w", padx=6, pady=(3, 0))
        frame.rowconfigure(4, weight=1)
        self.status_boxes["custom"] = self._status_box(frame, 4)
        self.custom_button = ttk.Button(frame, text="Build Custom Model", command=self._run_custom)
        self.custom_button.grid(row=4, column=2, sticky="n", pady=(8, 2))
        return frame

    def _status_box(self, parent: ttk.Frame, row: int) -> scrolledtext.ScrolledText:
        status = scrolledtext.ScrolledText(
            parent,
            height=6,
            wrap="word",
            state="disabled",
            font="TkFixedFont",
            borderwidth=1,
            relief="sunken",
        )
        status.grid(row=row, column=1, sticky="nsew", padx=6, pady=(8, 2))
        return status

    def _enable_drop(self, entry: ttk.Entry, var: tk.StringVar, drop_kind: str) -> None:
        if not HAS_DND or DND_FILES is None:
            return

        entry.drop_target_register(DND_FILES)
        entry.dnd_bind("<<Drop>>", lambda event: self._handle_drop(event, var, drop_kind))

    def _handle_drop(self, event, var: tk.StringVar, drop_kind: str) -> str:
        paths = self.tk.splitlist(event.data)
        if not paths:
            return "break"
        path = Path(paths[0])
        if drop_kind == "folder" and path.is_file():
            path = path.parent
        var.set(str(path))
        return "break"

    def _pick_file(self, var: tk.StringVar, filetypes) -> None:
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            var.set(path)

    def _pick_export_input(self) -> None:
        if self.export_mode.get() == "BDG":
            filetypes = [("_Shapes BDG files", "*_Shapes.BDG *_Shapes.bdg"), ("BDG files", "*.BDG *.bdg"), ("All files", "*.*")]
        else:
            filetypes = [("CMP/CMG files", "*.CMP *.cmp *.CMG *.cmg"), ("All files", "*.*")]
        self._pick_file(self.export_vars["input"], filetypes)

    def _pick_import_original(self) -> None:
        if self.import_mode.get() == "BDG":
            filetypes = [("_Shapes BDG files", "*_Shapes.BDG *_Shapes.bdg"), ("BDG files", "*.BDG *.bdg"), ("All files", "*.*")]
        else:
            filetypes = [("CMP/CMG files", "*.CMP *.cmp *.CMG *.cmg *.ZIP *.zip"), ("All files", "*.*")]
        self._pick_file(self.import_vars["original"], filetypes)

    def _refresh_export_mode(self) -> None:
        is_bdg = self.export_mode.get() == "BDG"
        self.export_input_widgets[3].configure(text="_Shapes File:" if is_bdg else "Input File:")
        for widget in self.export_anim_widgets:
            if is_bdg:
                widget.grid()
            else:
                widget.grid_remove()

    def _refresh_import_mode(self) -> None:
        is_bdg = self.import_mode.get() == "BDG"
        self.import_original_widgets[3].configure(text="_Shapes File:" if is_bdg else "Original file:")
        for widget in self.import_anim_widgets:
            if is_bdg:
                widget.grid()
            else:
                widget.grid_remove()

    def _pick_folder(self, var: tk.StringVar) -> None:
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    def _append(self, target: str, text: str) -> None:
        status = self.status_boxes.get(target)
        if status is None or not text:
            return
        status.configure(state="normal")
        status.insert("end", text)
        status.see("end")
        status.configure(state="disabled")

    def _clear_status(self, target: str) -> None:
        status = self.status_boxes.get(target)
        if status is None:
            return
        status.configure(state="normal")
        status.delete("1.0", "end")
        status.configure(state="disabled")

    def _drain_messages(self) -> None:
        while True:
            try:
                target, text = self.messages.get_nowait()
                self._append(target, text)
            except queue.Empty:
                break
        while True:
            try:
                title, active_button, active_text, error, on_success = self.completions.get_nowait()
                self._finish_background(title, active_button, active_text, error, on_success)
            except queue.Empty:
                break
        while True:
            try:
                result, error = self.update_results.get_nowait()
                self._finish_update_check(result, error)
            except queue.Empty:
                break
        self.after(100, self._drain_messages)

    def _set_running(self, active_button: ttk.Button, active_text: str, running: bool) -> None:
        buttons = [
            b
            for b in (self.export_button, self.import_button, self.quick_export_button, self.custom_button)
            if b is not None
        ]
        for button in buttons:
            button.configure(state="disabled" if running else "normal")
        if not running and self.quick_export_button is not None and self.quick_export_args is None:
            self.quick_export_button.configure(state="disabled")
        active_button.configure(text="Please Wait" if running else active_text)
        if running:
            self.update_idletasks()

    def _finish_background(
        self,
        title: str,
        active_button: ttk.Button,
        active_text: str,
        error: str | None = None,
        on_success: Callable[[], None] | None = None,
    ) -> None:
        if error:
            messagebox.showerror(title, error)
        else:
            if on_success is not None:
                on_success()
            messagebox.showinfo(title, "Done.")
        self._set_running(active_button, active_text, False)

    def _run_background(
        self,
        target: str,
        title: str,
        func,
        args: Namespace,
        active_button: ttk.Button,
        active_text: str,
        on_success: Callable[[], None] | None = None,
    ) -> None:
        self._clear_status(target)
        self._set_running(active_button, active_text, True)

        def worker() -> None:
            self.messages.put((target, f"{title} started.\n"))
            writer = QueueWriter(lambda text: self.messages.put((target, text)))
            try:
                with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                    func(args)
            except BaseException as exc:
                error = str(exc)
                self.messages.put((target, traceback.format_exc()))
                self.messages.put((target, f"{title} failed: {error}\n"))
                self.completions.put((title, active_button, active_text, error, None))
                return
            self.messages.put((target, f"{title} finished.\n"))
            self.completions.put((title, active_button, active_text, None, on_success))

        threading.Thread(target=worker, daemon=True).start()

    def _run_export(self) -> None:
        if self.export_button is None:
            return
        anim = self.export_vars["anim"].get() or None
        if self.export_mode.get() != "BDG":
            anim = None
        args = Namespace(input=self.export_vars["input"].get(), anim=anim, out=self.export_vars["out"].get() or None, force=True)
        self._run_background("export", "Export", bridge.export_bdg, args, self.export_button, "Export to FBX")

    def _run_import(self) -> None:
        if self.import_button is None:
            return
        self._invalidate_quick_export()
        anim = self.import_vars["anim"].get() or None
        if self.import_mode.get() != "BDG":
            anim = None
        args = Namespace(
            fbx=self.import_vars["fbx"].get(),
            project=None,
            original=self.import_vars["original"].get(),
            anim=anim,
            out=self.import_vars["out"].get() or None,
            force=True,
            bundle_mode=self.import_mode.get(),
            keep_mesh_in_place=self.import_keep_mesh.get(),
        )
        self._run_background(
            "import",
            "Import",
            bridge.import_fbx,
            args,
            self.import_button,
            "Import from FBX",
            lambda: self._remember_import(args),
        )

    def _invalidate_quick_export(self, *_args) -> None:
        self.quick_export_args = None
        if self.quick_export_button is not None:
            self.quick_export_button.configure(state="disabled")

    def _remember_import(self, args: Namespace) -> None:
        self.quick_export_args = Namespace(**vars(args))

    def _run_quick_export(self) -> None:
        if self.quick_export_button is None or self.quick_export_args is None:
            return
        self._run_background(
            "import",
            "Re-Export",
            bridge.quick_export_import,
            self.quick_export_args,
            self.quick_export_button,
            "Re-Export",
        )

    def _run_custom(self) -> None:
        if self.custom_button is None:
            return
        args = Namespace(
            fbx=self.custom_vars["fbx"].get(),
            template=self.custom_vars["template"].get(),
            out=self.custom_vars["out"].get() or None,
            bind_unweighted_root=self.custom_bind_root.get(),
        )
        self._run_background(
            "custom",
            "Custom Model",
            bridge.custom_model,
            args,
            self.custom_button,
            "Build Custom Model",
        )


def _run_frozen_tool() -> int | None:
    if len(sys.argv) < 3 or sys.argv[1] != bridge.FROZEN_TOOL_FLAG:
        return None
    # PyInstaller windowed applications may initialize these as None. The
    # parent converter supplies pipes for worker output, so attach Python text
    # streams to those handles before a bundled tool prints progress/errors.
    for stream_name, descriptor in (("stdout", 1), ("stderr", 2)):
        if getattr(sys, stream_name) is not None:
            continue
        try:
            stream = open(
                descriptor,
                "w",
                encoding="utf-8",
                errors="replace",
                buffering=1,
                closefd=False,
            )
        except OSError:
            stream = open(os.devnull, "w", encoding="utf-8")
        setattr(sys, stream_name, stream)
    script = sys.argv[2]
    try:
        bridge.run_tool_in_process(script, *sys.argv[3:])
    except SystemExit as exc:
        if isinstance(exc.code, str):
            print(exc.code, file=sys.stderr, flush=True)
            return 1
        return int(exc.code or 0)
    return 0


def main() -> int:
    worker_result = _run_frozen_tool()
    if worker_result is not None:
        return worker_result
    BridgeGui().mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
