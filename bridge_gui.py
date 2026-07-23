"""Tkinter front end for the BDG Blender Converter."""
from __future__ import annotations

import queue
import sys
import threading
import tkinter as tk
from argparse import Namespace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import bridge

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    BaseTk = TkinterDnD.Tk
    HAS_DND = True
except Exception:
    DND_FILES = None
    BaseTk = tk.Tk
    HAS_DND = False


class BridgeGui(BaseTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("GZ Blender Converter")
        self._set_window_icon()
        self.geometry("680x190")
        self.minsize(620, 170)
        self.messages: queue.Queue[str] = queue.Queue()
        self.export_button: ttk.Button | None = None
        self.import_button: ttk.Button | None = None
        self._build()
        self.after(100, self._drain_messages)

    def _set_window_icon(self) -> None:
        base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
        icon_path = base / "gz.ico"
        if icon_path.exists():
            self.iconbitmap(default=str(icon_path))

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        tabs = ttk.Notebook(self)
        tabs.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.export_vars = {"input": tk.StringVar(), "out": tk.StringVar()}
        self.import_vars = {"fbx": tk.StringVar(), "original": tk.StringVar(), "out": tk.StringVar()}
        tabs.add(self._export_tab(tabs), text="Export")
        tabs.add(self._import_tab(tabs), text="Import")

    def _row(self, parent: ttk.Frame, row: int, label: str, var: tk.StringVar, browse, drop_kind: str = "file") -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        entry = ttk.Entry(parent, textvariable=var)
        entry.grid(row=row, column=1, sticky="ew", padx=6, pady=3)
        self._enable_drop(entry, var, drop_kind)
        ttk.Button(parent, text="Browse", command=browse).grid(row=row, column=2, sticky="ew", pady=3)

    def _export_tab(self, tabs: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(tabs, padding=8)
        frame.columnconfigure(1, weight=1)
        self._row(
            frame,
            0,
            "Input File",
            self.export_vars["input"],
            lambda: self._pick_file(
                self.export_vars["input"],
                [("Kaiju model files", "*.BDG *.bdg *.CMG *.cmg"), ("All files", "*.*")],
            ),
        )
        self._row(frame, 1, "Output folder", self.export_vars["out"], lambda: self._pick_folder(self.export_vars["out"]), "folder")
        self.export_button = ttk.Button(frame, text="Export to FBX", command=self._run_export)
        self.export_button.grid(row=2, column=1, sticky="e", pady=(8, 2))
        return frame

    def _import_tab(self, tabs: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(tabs, padding=8)
        frame.columnconfigure(1, weight=1)
        self._row(frame, 0, "Edited FBX", self.import_vars["fbx"], lambda: self._pick_file(self.import_vars["fbx"], [("FBX files", "*.fbx"), ("All files", "*.*")]))
        self._row(frame, 1, "Original file", self.import_vars["original"], lambda: self._pick_file(self.import_vars["original"], [("Kaiju model files", "*.BDG *.bdg *.CMG *.cmg *.ZIP *.zip"), ("All files", "*.*")]))
        self._row(frame, 2, "Output folder", self.import_vars["out"], lambda: self._pick_folder(self.import_vars["out"]), "folder")
        self.import_button = ttk.Button(frame, text="Import from FBX", command=self._run_import)
        self.import_button.grid(row=3, column=1, sticky="e", pady=10)
        return frame

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

    def _pick_folder(self, var: tk.StringVar) -> None:
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    def _append(self, text: str) -> None:
        # Keep the worker queue for thread handoff, but do not render a status
        # log in the compact GUI. Errors and completion use message boxes.
        return None

    def _drain_messages(self) -> None:
        while True:
            try:
                self._append(self.messages.get_nowait())
            except queue.Empty:
                break
        self.after(100, self._drain_messages)

    def _set_running(self, active_button: ttk.Button, active_text: str, running: bool) -> None:
        buttons = [b for b in (self.export_button, self.import_button) if b is not None]
        for button in buttons:
            button.configure(state="disabled" if running else "normal")
        active_button.configure(text="Please Wait" if running else active_text)
        if running:
            self.update_idletasks()

    def _finish_background(self, title: str, active_button: ttk.Button, active_text: str, error: str | None = None) -> None:
        if error:
            messagebox.showerror(title, error)
        else:
            messagebox.showinfo(title, "Done.")
        self._set_running(active_button, active_text, False)

    def _run_background(self, title: str, func, args: Namespace, active_button: ttk.Button, active_text: str) -> None:
        self._set_running(active_button, active_text, True)

        def worker() -> None:
            self.messages.put(f"{title} started.")
            try:
                func(args)
            except BaseException as exc:
                error = str(exc)
                self.messages.put(f"{title} failed: {error}")
                self.after(0, lambda error=error: self._finish_background(title, active_button, active_text, error))
                return
            self.messages.put(f"{title} finished.")
            self.after(0, lambda: self._finish_background(title, active_button, active_text))

        threading.Thread(target=worker, daemon=True).start()

    def _run_export(self) -> None:
        if self.export_button is None:
            return
        args = Namespace(input=self.export_vars["input"].get(), out=self.export_vars["out"].get() or None, force=True)
        self._run_background("Export", bridge.export_bdg, args, self.export_button, "Export to FBX")

    def _run_import(self) -> None:
        if self.import_button is None:
            return
        args = Namespace(fbx=self.import_vars["fbx"].get(), project=None, original=self.import_vars["original"].get(), out=self.import_vars["out"].get() or None, force=True)
        self._run_background("Import", bridge.import_fbx, args, self.import_button, "Import from FBX")


def main() -> None:
    BridgeGui().mainloop()


if __name__ == "__main__":
    main()
