"""Bone reconstruction from landmarks - desktop app.

  1. choose the bone (femur or tibia)
  2. enter the x y z coordinates (mm) of its landmarks; the 3D view and the
     description show where each landmark lies on the bone
  3. the reconstructed bone is shown and saved as an STL file

    python bone_reconstruction_app.py
    python bone_reconstruction_app.py --bone Tibia --landmarks examples/tibia_example.txt --auto

Reconstructions are saved to ../results/reconstructions/ (STL + the entered
landmarks). The bones offered and the landmark descriptions come from
landmark_guide.json.
"""
from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np

from mesh_view import MeshView
from reconstruction_engine import BoneReconstructor, load_guide, output_paths, parse_landmark_text
from ssm_io import ROOT

APP_TITLE = "Bone Reconstruction from Landmarks"
ICON_FILE = ROOT / "files_for_gui" / "thesis_gui_icon.ico"
OUTPUT_DIR = ROOT / "results" / "reconstructions"

BG, PANEL, TEXT, MUTED = "#f4f5f7", "#ffffff", "#1d2433", "#5b6475"
ACCENT, ACCENT_DARK = "#2563eb", "#1d4ed8"
LANDMARK, SELECTED, ENTERED, FITTED = "#f2a33a", "#ff4d2e", "#e5484d", "#30c48d"
WARN_BG, NOTE_BG = "#fff4e5", "#e8f1ff"


# ---------------------------------------------------------------------- helpers
def setup_style(root: tk.Tk) -> None:
    style = ttk.Style(root)
    style.theme_use("clam")
    root.configure(background=BG)
    style.configure(".", background=BG, foreground=TEXT, font=("Segoe UI", 10))
    style.configure("Panel.TFrame", background=PANEL)
    style.configure("TLabel", background=BG)
    style.configure("Panel.TLabel", background=PANEL)
    style.configure("Muted.TLabel", background=PANEL, foreground=MUTED)
    style.configure("Title.TLabel", background=BG, font=("Segoe UI Semibold", 18))
    style.configure("Heading.TLabel", background=PANEL, font=("Segoe UI Semibold", 13))
    style.configure("Code.TLabel", background=PANEL, font=("Segoe UI Semibold", 10))
    style.configure("Step.TLabel", background=BG, foreground="#9aa1ad", font=("Segoe UI", 10))
    style.configure("StepOn.TLabel", background=BG, foreground=ACCENT, font=("Segoe UI Semibold", 10))
    style.configure("Ok.TLabel", background=PANEL, foreground="#16a34a", font=("Segoe UI Semibold", 11))
    style.configure("Todo.TLabel", background=PANEL, foreground="#c0c5ce", font=("Segoe UI Semibold", 11))
    style.configure("Bad.TLabel", background=PANEL, foreground="#dc2626", font=("Segoe UI Semibold", 11))
    style.configure("TButton", padding=(10, 6))
    style.configure("Accent.TButton", background=ACCENT, foreground="#ffffff", padding=(16, 9),
                    font=("Segoe UI Semibold", 11), borderwidth=0)
    style.map("Accent.TButton", background=[("disabled", "#a9b8d9"), ("active", ACCENT_DARK)],
              foreground=[("disabled", "#eef2fa")])
    style.configure("Card.TButton", background=PANEL, padding=(24, 22), font=("Segoe UI Semibold", 16),
                    borderwidth=1, relief="solid")
    style.map("Card.TButton", background=[("active", "#eef3ff")])
    style.configure("TEntry", padding=(4, 3))
    style.configure("Treeview", rowheight=24)


def set_icon(root: tk.Tk) -> None:
    if not ICON_FILE.is_file():
        print(f"note: window icon not found: {ICON_FILE}", file=sys.stderr)
        return
    try:
        root.iconbitmap(default=str(ICON_FILE))
    except tk.TclError as e:   # e.g. .ico not supported on this platform
        print(f"note: could not use the window icon ({e})", file=sys.stderr)


def open_folder(path: Path) -> None:
    if sys.platform.startswith("win"):
        os.startfile(path)  # noqa: S606 - opening the output folder for the user
    else:
        subprocess.run(["open" if sys.platform == "darwin" else "xdg-open", str(path)], check=False)


def short_path(path: Path) -> str:
    """Path relative to the WP5 folder when inside it (shorter to read)."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def fmt(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------------- pages
class ChoosePage(ttk.Frame):
    def __init__(self, app: "App"):
        super().__init__(app.body)
        self.app = app
        ttk.Label(self, text="Which bone do you want to reconstruct?", style="Title.TLabel").pack(pady=(40, 6))
        ttk.Label(self, text="You will enter a few landmark coordinates measured on the bone; "
                             "the full 3D bone is reconstructed from them with the statistical shape model.",
                  foreground=MUTED).pack(pady=(0, 30))
        cards = ttk.Frame(self)
        cards.pack()
        for col, (bone, cfg) in enumerate(app.guide.items()):
            self.card(cards, bone, cfg).grid(row=0, column=col, padx=18, sticky="nsew")
        steps = ("1. Choose the bone.", "2. Enter the x y z coordinates of its landmarks (mm).",
                 "3. Get the reconstructed 3D bone as an STL file.")
        ttk.Label(self, text="     ".join(steps), foreground=MUTED).pack(pady=(28, 0))
        self.status = ttk.Label(self, text="", foreground=MUTED)
        self.status.pack(pady=(30, 6))
        self.progress = ttk.Progressbar(self, mode="indeterminate", length=280)

    def card(self, parent, bone: str, cfg: dict) -> tk.Frame:
        """Clickable card: bone name, model description and its landmarks."""
        frame = tk.Frame(parent, background=PANEL, highlightthickness=1, highlightbackground="#d5d9e0",
                         cursor="hand2", padx=26, pady=20)
        tk.Frame(frame, width=int(frame.winfo_fpixels("2.8i")), height=1, background=PANEL).pack()  # min width
        labels = [
            tk.Label(frame, text=bone, background=PANEL, foreground=TEXT, font=("Segoe UI Semibold", 20)),
            tk.Label(frame, text=cfg.get("description", ""), background=PANEL, foreground=MUTED,
                     font=("Segoe UI", 10)),
            tk.Label(frame, text=f"{len(cfg['landmarks'])} landmarks", background=PANEL, foreground=TEXT,
                     font=("Segoe UI Semibold", 10)),
            tk.Label(frame, text="\n".join(f"{c}  {v['name']}" for c, v in cfg["landmarks"].items()),
                     background=PANEL, foreground=MUTED, font=("Segoe UI", 9), justify="left"),
        ]
        for i, lbl in enumerate(labels):
            lbl.pack(anchor="w", pady=(8 if i == 2 else 0, 0))
        widgets = [frame] + labels

        def hover(on):
            color = "#eef3ff" if on else PANEL
            frame.configure(highlightbackground=ACCENT if on else "#d5d9e0", background=color)
            for lbl in labels:
                lbl.configure(background=color)
        for w in widgets:
            w.bind("<Enter>", lambda e: hover(True))
            w.bind("<Leave>", lambda e: hover(False))
            w.bind("<Button-1>", lambda e, bone=bone: self.app.load_bone(bone))
        return frame

    def busy(self, text: str | None) -> None:
        self.status.configure(text=text or "")
        if text:
            self.progress.pack()
            self.progress.start(12)
        else:
            self.progress.stop()
            self.progress.pack_forget()


class LandmarkPage(ttk.Frame):
    def __init__(self, app: "App"):
        super().__init__(app.body)
        self.app = app
        self.rec: BoneReconstructor | None = None
        self.vars: list[list[tk.StringVar]] = []
        self.status: list[ttk.Label] = []
        self.entries: list[list[ttk.Entry]] = []
        self.selected = 0
        self.input_name: str | None = None   # stem of the loaded landmark file, used in the output name
        self.sides: dict[str, str] = {}

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.view = MeshView(self, width=640, height=600)
        self.view.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        self.panel = ttk.Frame(self, style="Panel.TFrame", padding=16)
        self.panel.grid(row=0, column=1, sticky="ns")

    def show_bone(self, rec: BoneReconstructor, keep_values: bool = False) -> None:
        old = [[v.get() for v in row] for row in self.vars] if keep_values and self.rec is rec else None
        if old is None:
            self.input_name = None
        self.rec = rec
        for w in self.panel.winfo_children():
            w.destroy()
        p = self.panel
        ttk.Label(p, text=f"{rec.bone}: enter the landmarks", style="Heading.TLabel").pack(anchor="w")
        ttk.Label(p, text="Coordinates in millimetres, all measured in the same coordinate system\n"
                          "(e.g. one CT scan). Click a row to see where the landmark lies.",
                  style="Muted.TLabel", justify="left").pack(anchor="w", pady=(2, 12))

        grid = ttk.Frame(p, style="Panel.TFrame")
        grid.pack(anchor="w", fill="x")
        for c, head in enumerate(["", "Landmark", "x", "y", "z", ""]):
            ttk.Label(grid, text=head, style="Muted.TLabel").grid(row=0, column=c, sticky="w", padx=2)
        self.vars, self.status, self.entries = [], [], []
        for i, lm in enumerate(rec.landmarks):
            ttk.Label(grid, text=f"{i + 1}", style="Muted.TLabel").grid(row=i + 1, column=0, padx=(0, 4))
            name = ttk.Label(grid, text=f"{lm.code}  {lm.name}", style="Code.TLabel", width=24, cursor="hand2")
            name.grid(row=i + 1, column=1, sticky="w", pady=3)
            name.bind("<Button-1>", lambda e, i=i: self.select(i))
            row_vars, row_entries = [], []
            for k in range(3):
                v = tk.StringVar(value=old[i][k] if old else "")
                v.trace_add("write", lambda *a, i=i: self.validate_row(i))
                e = ttk.Entry(grid, textvariable=v, width=10, justify="right")
                e.grid(row=i + 1, column=2 + k, padx=2)
                e.bind("<FocusIn>", lambda ev, i=i: self.select(i))
                e.bind("<Return>", lambda ev, i=i, k=k: self.focus_next(i, k))
                e.bind("<<Paste>>", lambda ev, i=i, k=k: self.paste(ev, i, k))
                row_vars.append(v)
                row_entries.append(e)
            st = ttk.Label(grid, text="●", style="Todo.TLabel")
            st.grid(row=i + 1, column=5, padx=(6, 0))
            self.vars.append(row_vars)
            self.entries.append(row_entries)
            self.status.append(st)

        tools = ttk.Frame(p, style="Panel.TFrame")
        tools.pack(anchor="w", pady=(10, 0))
        ttk.Button(tools, text="Paste all…", command=self.paste_dialog).pack(side="left")
        ttk.Button(tools, text="Load file…", command=self.load_file).pack(side="left", padx=6)
        ttk.Button(tools, text="Clear", command=self.clear).pack(side="left")

        self.where = ttk.LabelFrame(p, text="Where to place it", padding=10)
        self.where.pack(fill="x", pady=(16, 0))
        self.where_title = ttk.Label(self.where, text="", style="Code.TLabel")
        self.where_title.pack(anchor="w")
        self.where_text = ttk.Label(self.where, text="", wraplength=380, justify="left")
        self.where_text.pack(anchor="w", pady=(4, 0))

        side = ttk.Frame(p, style="Panel.TFrame")
        side.pack(anchor="w", pady=(16, 0), fill="x")
        ttk.Label(side, text="Body side:", style="Panel.TLabel").pack(side="left")
        self.sides = {"Auto-detect": "auto", f"Left {rec.bone.lower()}": "L", f"Right {rec.bone.lower()}": "R"}
        self.side = tk.StringVar(value="Auto-detect")
        ttk.Combobox(side, textvariable=self.side, values=list(self.sides), state="readonly", width=16).pack(
            side="left", padx=8)
        ttk.Label(side, text=f"(model: {rec.side_label(rec.model_side)})", style="Muted.TLabel").pack(side="left")

        buttons = ttk.Frame(p, style="Panel.TFrame")
        buttons.pack(fill="x", pady=(20, 0))
        ttk.Button(buttons, text="← Back", command=self.app.show_choose).pack(side="left")
        self.go = ttk.Button(buttons, text="Reconstruct bone", style="Accent.TButton", command=self.reconstruct)
        self.go.pack(side="right")
        self.hint = ttk.Label(p, text="", style="Muted.TLabel")
        self.hint.pack(anchor="e", pady=(6, 0))

        self.view.set_mesh(rec.mean_mesh.points, rec.mean_mesh.faces)
        self.view.set_view(*rec.view_axes())
        for i in range(len(self.vars)):
            self.validate_row(i)
        self.select(0)
        self.entries[0][0].focus_set()

    # ------------------------------------------------------------------ rows
    def row_values(self, i: int) -> list[float] | None:
        try:
            vals = [float(v.get().replace(",", ".")) for v in self.vars[i]]
        except ValueError:
            return None
        return vals if all(np.isfinite(vals)) else None

    def validate_row(self, i: int) -> None:
        texts = [v.get().strip() for v in self.vars[i]]
        if not any(texts):
            self.status[i].configure(text="●", style="Todo.TLabel")
        elif self.row_values(i) is None:
            self.status[i].configure(text="✗", style="Bad.TLabel")
        else:
            self.status[i].configure(text="✓", style="Ok.TLabel")
        done = sum(self.row_values(k) is not None for k in range(len(self.vars)))
        total = len(self.vars)
        self.go.state(["!disabled"] if done == total else ["disabled"])
        self.hint.configure(text="" if done == total else f"{done} of {total} landmarks entered")

    def select(self, i: int) -> None:
        self.selected = i
        lm = self.rec.landmarks[i]
        self.where.configure(text=f"Where to place landmark {i + 1} of {len(self.rec.landmarks)}")
        self.where_title.configure(text=f"{lm.code}: {lm.name}")
        self.where_text.configure(text=lm.description)
        self.view.set_markers([{"position": l.position, "label": l.code,
                                "color": SELECTED if k == i else LANDMARK, "highlight": k == i}
                               for k, l in enumerate(self.rec.landmarks)])

    def focus_next(self, i: int, k: int) -> None:
        if k < 2:
            self.entries[i][k + 1].focus_set()
        elif i + 1 < len(self.entries):
            self.entries[i + 1][0].focus_set()
        elif str(self.go.cget("state")) != "disabled":
            self.reconstruct()

    def set_values(self, coords: np.ndarray, start: int = 0) -> None:
        for r, row in enumerate(coords):
            if start + r >= len(self.vars):
                break
            for k in range(3):
                self.vars[start + r][k].set(fmt(row[k]))

    def paste(self, event, i: int, k: int):
        """Pasting several numbers into one field fills that row and the following ones."""
        try:
            text = self.clipboard_get()
        except tk.TclError:
            return None
        if len(text.replace(",", " ").replace(";", " ").split()) < 3:
            return None   # a single value: normal paste
        try:
            coords = parse_landmark_text(text)
        except ValueError as e:
            messagebox.showerror("Cannot paste", str(e), parent=self)
            return "break"
        self.set_values(coords, start=i)
        self.input_name = None
        return "break"

    def paste_dialog(self) -> None:
        dlg = tk.Toplevel(self)
        dlg.title("Paste landmark coordinates")
        dlg.transient(self.winfo_toplevel())
        dlg.configure(background=PANEL)
        order = ", ".join(self.rec.codes)
        ttk.Label(dlg, text=f"One landmark per line, x y z in mm, in this order:\n{order}",
                  style="Panel.TLabel", justify="left").pack(anchor="w", padx=14, pady=(12, 6))
        txt = tk.Text(dlg, width=48, height=len(self.rec.codes) + 2, font=("Consolas", 11))
        txt.pack(padx=14)
        txt.focus_set()

        def ok():
            try:
                coords = parse_landmark_text(txt.get("1.0", "end"), self.rec.n_landmarks)
            except ValueError as e:
                messagebox.showerror("Cannot use these coordinates", str(e), parent=dlg)
                return
            self.set_values(coords)
            self.input_name = None
            dlg.destroy()

        row = ttk.Frame(dlg, style="Panel.TFrame")
        row.pack(fill="x", padx=14, pady=12)
        ttk.Button(row, text="Use these values", style="Accent.TButton", command=ok).pack(side="right")
        ttk.Button(row, text="Cancel", command=dlg.destroy).pack(side="right", padx=8)
        dlg.grab_set()

    def load_file(self) -> None:
        path = filedialog.askopenfilename(
            parent=self, title="Load landmark coordinates",
            filetypes=[("Landmark files", "*.particles *.txt *.csv *.xyz"), ("All files", "*.*")])
        if not path:
            return
        try:
            coords = parse_landmark_text(Path(path).read_text(encoding="utf-8", errors="replace"),
                                         self.rec.n_landmarks)
        except (OSError, ValueError) as e:
            messagebox.showerror("Cannot load landmarks", f"{Path(path).name}:\n{e}", parent=self)
            return
        self.set_values(coords)
        self.input_name = Path(path).stem

    def clear(self) -> None:
        for row in self.vars:
            for v in row:
                v.set("")
        self.input_name = None
        self.select(0)
        self.entries[0][0].focus_set()

    def coordinates(self) -> np.ndarray:
        return np.array([self.row_values(i) for i in range(len(self.vars))], dtype=float)

    def reconstruct(self) -> None:
        if any(self.row_values(i) is None for i in range(len(self.vars))):
            messagebox.showwarning("Missing values", "Enter x, y and z for every landmark first.", parent=self)
            return
        self.app.reconstruct(self.rec, self.coordinates(), self.sides[self.side.get()], self.input_name)


class ResultPage(ttk.Frame):
    def __init__(self, app: "App"):
        super().__init__(app.body)
        self.app = app
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.view = MeshView(self, width=640, height=600)
        self.view.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        self.panel = ttk.Frame(self, style="Panel.TFrame", padding=16)
        self.panel.grid(row=0, column=1, sticky="ns")
        self.result = None
        self.saved: Path | None = None

    def show_result(self, rec: BoneReconstructor, result, saved: Path) -> None:
        self.result, self.saved = result, saved
        for w in self.panel.winfo_children():
            w.destroy()
        p = self.panel
        ttk.Label(p, text=f"Reconstructed {rec.side_label(result.side)}", style="Heading.TLabel").pack(anchor="w")
        pts = result.mesh.points
        up, front = rec.view_axes(result.fitted_landmarks)
        length = np.ptp(pts @ up)
        ttk.Label(p, text=f"{len(pts):,} vertices, {len(result.mesh.faces):,} triangles, "
                          f"length {length:.0f} mm", style="Muted.TLabel").pack(anchor="w", pady=(2, 8))
        ttk.Label(p, text=f"Saved: {short_path(saved)}", style="Panel.TLabel", wraplength=420,
                  justify="left").pack(anchor="w")

        buttons = ttk.Frame(p, style="Panel.TFrame")
        buttons.pack(fill="x", pady=(10, 0))
        ttk.Button(buttons, text="Save STL as…", style="Accent.TButton", command=self.save_as).pack(side="left")
        ttk.Button(buttons, text="Open folder", command=lambda: open_folder(self.saved.parent)).pack(
            side="left", padx=8)
        ttk.Button(buttons, text="New bone", command=self.app.show_choose).pack(side="right")
        ttk.Button(buttons, text="← Edit landmarks", command=self.app.show_landmarks).pack(side="right", padx=8)

        for text, bg in [(n, NOTE_BG) for n in result.notes] + [(w, WARN_BG) for w in result.warnings]:
            tk.Label(p, text=text, background=bg, foreground=TEXT, wraplength=410, justify="left",
                     anchor="w", padx=10, pady=6, font=("Segoe UI", 9)).pack(fill="x", pady=(10, 0))
        if not result.warnings:
            tk.Label(p, text="No problems found in the landmarks.", background="#e9f8ef", foreground=TEXT,
                     anchor="w", padx=10, pady=6, font=("Segoe UI", 9)).pack(fill="x", pady=(10, 0))

        ttk.Label(p, text="Fit at the landmarks", style="Code.TLabel").pack(anchor="w", pady=(14, 4))
        tree = ttk.Treeview(p, columns=("code", "x", "y", "z", "dist"), show="headings",
                            height=len(result.codes), selectmode="none")
        for col, head, w in (("code", "Landmark", 76), ("x", "x", 66), ("y", "y", 66), ("z", "z", 66),
                             ("dist", "Distance (mm)", 112)):
            tree.heading(col, text=head)
            tree.column(col, width=w, anchor="e" if col != "code" else "w")
        for c, x, d in zip(result.codes, result.landmarks, result.residuals):
            tree.insert("", "end", values=(c, f"{x[0]:.1f}", f"{x[1]:.1f}", f"{x[2]:.1f}", f"{d:.1f}"))
        tree.pack(anchor="w")
        ttk.Label(p, text="Distance between each entered landmark and the same landmark on the "
                          "reconstructed bone (typically 1-7 mm).", style="Muted.TLabel",
                  wraplength=420, justify="left").pack(anchor="w", pady=(4, 0))

        legend = ttk.Frame(p, style="Panel.TFrame")
        legend.pack(anchor="w", pady=(10, 0))
        for color, text in ((ENTERED, "entered landmarks"), (FITTED, "landmarks on the reconstructed bone")):
            tk.Label(legend, text="●", foreground=color, background=PANEL, font=("Segoe UI", 12)).pack(side="left")
            ttk.Label(legend, text=text, style="Muted.TLabel").pack(side="left", padx=(2, 12))

        self.view.set_mesh(result.mesh.points, result.mesh.faces)
        self.view.set_view(up, front)
        markers = [{"position": x, "label": c, "color": ENTERED, "radius": 7}
                   for c, x in zip(result.codes, result.landmarks)]
        markers += [{"position": x, "color": FITTED, "radius": 4} for x in result.fitted_landmarks]
        self.view.set_markers(markers)

    def save_as(self) -> None:
        path = filedialog.asksaveasfilename(parent=self, title="Save reconstructed bone",
                                            defaultextension=".stl", initialfile=self.saved.name,
                                            initialdir=str(self.saved.parent),
                                            filetypes=[("STL mesh", "*.stl")])
        if not path:
            return
        try:
            self.result.save_stl(path)
        except OSError as e:
            messagebox.showerror("Could not save", str(e), parent=self)
            return
        messagebox.showinfo("Saved", f"Saved {Path(path).name}", parent=self)


# ---------------------------------------------------------------------- app
class App(tk.Tk):
    def __init__(self, background_tasks: bool = True, output_dir: Path = OUTPUT_DIR):
        super().__init__()
        self.title(APP_TITLE)
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h = min(1320, sw - 40), min(860, sh - 90)      # fit the screen, above the taskbar
        self.geometry(f"{w}x{h}+{(sw - w) // 2}+{max((sh - h) // 2 - 30, 0)}")
        self.minsize(min(1100, w), min(660, h))
        set_icon(self)
        setup_style(self)
        self.background_tasks = background_tasks
        self.output_dir = Path(output_dir)
        self.guide = load_guide()
        self.models: dict[str, BoneReconstructor] = {}
        self.current: BoneReconstructor | None = None
        self.last_result = None
        self.last_saved: Path | None = None

        header = ttk.Frame(self, padding=(20, 14, 20, 6))
        header.pack(fill="x")
        ttk.Label(header, text=APP_TITLE, style="Title.TLabel").pack(side="left")
        steps = ttk.Frame(header)
        steps.pack(side="right")
        self.step_labels = []
        for i, name in enumerate(["1  Choose bone", "2  Enter landmarks", "3  Result"]):
            lbl = ttk.Label(steps, text=name, style="Step.TLabel")
            lbl.pack(side="left", padx=10)
            self.step_labels.append(lbl)
        ttk.Separator(self).pack(fill="x", padx=20)

        self.body = ttk.Frame(self, padding=20)
        self.body.pack(fill="both", expand=True)
        self.pages = {"choose": ChoosePage(self), "landmarks": LandmarkPage(self), "result": ResultPage(self)}
        self.show_choose()

    # ------------------------------------------------------------------ navigation
    def _show(self, name: str, step: int) -> None:
        for page in self.pages.values():
            page.pack_forget()
        self.pages[name].pack(fill="both", expand=True)
        for i, lbl in enumerate(self.step_labels):
            lbl.configure(style="StepOn.TLabel" if i == step else "Step.TLabel")

    def show_choose(self) -> None:
        self.pages["choose"].busy(None)
        self._show("choose", 0)

    def show_landmarks(self, keep_values: bool = True) -> None:
        self._show("landmarks", 1)
        page = self.pages["landmarks"]
        if page.rec is not self.current or not keep_values:
            page.show_bone(self.current)

    # ------------------------------------------------------------------ work
    def run_task(self, work, on_done, busy_text: str) -> None:
        """Run ``work`` off the UI thread (unless disabled), then ``on_done(result)``; errors are shown."""
        if not self.background_tasks:
            try:
                result = work()
            except Exception as e:  # noqa: BLE001 - every error goes to the user
                self.show_error(e)
                return
            on_done(result)
            return
        self.configure(cursor="watch")
        q: queue.Queue = queue.Queue()

        def target():
            try:
                q.put((True, work()))
            except Exception as e:  # noqa: BLE001
                q.put((False, e))
        threading.Thread(target=target, daemon=True).start()

        def poll():
            try:
                ok, value = q.get_nowait()
            except queue.Empty:
                self.after(50, poll)
                return
            self.configure(cursor="")
            on_done(value) if ok else self.show_error(value)
        self.title(f"{APP_TITLE} - {busy_text}")
        self.after(50, poll)

    def show_error(self, e: Exception) -> None:
        self.title(APP_TITLE)
        self.pages["choose"].busy(None)
        kind = {FileNotFoundError: "A required file is missing", ValueError: "Cannot use these landmarks"}.get(
            type(e), "Something went wrong")
        messagebox.showerror(kind, str(e), parent=self)

    def load_bone(self, bone: str) -> None:
        if bone in self.models:
            self.current = self.models[bone]
            self.show_landmarks(keep_values=False)
            return
        self.pages["choose"].busy(f"Loading the {bone.lower()} model…")

        def done(rec):
            self.title(APP_TITLE)
            self.models[bone] = rec
            self.current = rec
            self.pages["choose"].busy(None)
            self.show_landmarks(keep_values=False)
        self.run_task(lambda: BoneReconstructor(bone, guide=self.guide), done, "loading model")

    def reconstruct(self, rec: BoneReconstructor, coords: np.ndarray, side: str = "auto",
                    input_name: str | None = None) -> None:
        def work():
            result = rec.reconstruct(coords, side=side)
            stl, lm_file = output_paths(self.output_dir, rec.bone, result.side, input_name)
            path = result.save_stl(stl)
            header = (f"# {rec.side_label(result.side)} landmarks ({', '.join(rec.codes)}), x y z in mm"
                      + (f", from {input_name}" if input_name else "") + "\n")
            lm_file.write_text(header + "\n".join(" ".join(f"{v:.4f}" for v in row) for row in coords) + "\n")
            return result, path

        def done(value):
            self.title(APP_TITLE)
            self.last_result, self.last_saved = value
            self._show("result", 2)
            self.pages["result"].show_result(rec, *value)
        self.run_task(work, done, "reconstructing")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bone", help="open this bone directly (e.g. Femur, Tibia)")
    ap.add_argument("--landmarks", type=Path, help="fill in the landmarks from this file")
    ap.add_argument("--auto", action="store_true", help="reconstruct right away (needs --bone and --landmarks)")
    ap.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = ap.parse_args()

    try:  # sharp text on high-DPI Windows screens
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass
    try:
        app = App(output_dir=args.output_dir)
    except (FileNotFoundError, ValueError) as e:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(APP_TITLE, f"The app cannot start:\n\n{e}")
        sys.exit(1)

    if args.bone:
        bone = next((b for b in app.guide if b.lower() == args.bone.lower()), None)
        if bone is None:
            ap.error(f"unknown bone {args.bone!r}; available: {', '.join(app.guide)}")

        def after_load(_=None):
            if app.current is None:
                app.after(100, after_load)
                return
            page = app.pages["landmarks"]
            if args.landmarks:
                try:
                    coords = parse_landmark_text(args.landmarks.read_text(), app.current.n_landmarks)
                except (OSError, ValueError) as e:
                    app.show_error(ValueError(f"{args.landmarks}: {e}"))
                    return
                page.set_values(coords)
                page.input_name = args.landmarks.stem
                if args.auto:
                    page.reconstruct()
        app.after(100, lambda: app.load_bone(bone))
        app.after(200, after_load)
    app.mainloop()


if __name__ == "__main__":
    main()
