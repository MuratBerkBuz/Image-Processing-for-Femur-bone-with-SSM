"""Interactive 3D mesh viewer on a tkinter Canvas (no external 3D library).

Shaded triangles drawn back to front, with markers for landmarks. Markers
hidden behind the surface are drawn hollow.

  left drag: rotate   right drag: pan   wheel: zoom   double-click: reset view
"""
from __future__ import annotations

import tkinter as tk

import numpy as np

LIGHT = np.array([-0.35, 0.45, 0.82]) / np.linalg.norm([-0.35, 0.45, 0.82])
FINE_FACES, COARSE_FACES = 6000, 1800     # display resolution at rest / while dragging


def decimate(points: np.ndarray, faces: np.ndarray, max_faces: int) -> tuple[np.ndarray, np.ndarray]:
    """Vertex-clustering decimation to at most ``max_faces`` triangles (display only)."""
    if len(faces) <= max_faces:
        return points, faces
    lo, hi = points.min(0), points.max(0)
    cells = 200
    while True:
        size = (hi - lo).max() / cells
        key = np.floor((points - lo) / size).astype(np.int64)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.ravel()
        f = inv[faces]
        f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
        f = f[np.unique(np.sort(f, axis=1), axis=0, return_index=True)[1]]
        if len(f) <= max_faces or cells < 10:
            break
        cells = int(cells * 0.8)
    n = inv.max() + 1
    pts = np.zeros((n, 3))
    np.add.at(pts, inv, points)
    pts /= np.bincount(inv, minlength=n)[:, None]
    return pts, f


def orient_outward(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = points[faces]
    volume = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum()
    return faces if volume >= 0 else faces[:, ::-1]


def _rot(axis: int, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    i, j = [k for k in range(3) if k != axis]
    R = np.eye(3)
    R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
    return R


class MeshView(tk.Canvas):
    def __init__(self, master, background: str = "#1f2633", **kw):
        super().__init__(master, background=background, highlightthickness=0, **kw)
        self.background = background
        self._lods = None            # {"fine": (pts, faces, normals), "coarse": ...}
        self._color = np.array([233, 223, 201]) / 255
        self._markers: list[dict] = []
        self._center = np.zeros(3)
        self._radius = 1.0
        self._home = np.eye(3)
        self._R = np.eye(3)
        self._zoom, self._pan = 1.0, np.zeros(2)
        self._drag = None
        self._pending = False
        self._palette = None
        self.message = ""
        self.bind("<Configure>", lambda e: self.redraw())
        self.bind("<ButtonPress-1>", lambda e: self._start(e, "rotate"))
        self.bind("<ButtonPress-3>", lambda e: self._start(e, "pan"))
        self.bind("<B1-Motion>", self._move)
        self.bind("<B3-Motion>", self._move)
        self.bind("<ButtonRelease-1>", self._end)
        self.bind("<ButtonRelease-3>", self._end)
        self.bind("<MouseWheel>", self._wheel)
        self.bind("<Button-4>", lambda e: self._zoom_by(1.1))   # Linux
        self.bind("<Button-5>", lambda e: self._zoom_by(1 / 1.1))
        self.bind("<Double-Button-1>", lambda e: self.reset_view())

    # ------------------------------------------------------------------ content
    def set_mesh(self, points: np.ndarray, faces: np.ndarray, color: str = "#e9dfc9") -> None:
        points = np.asarray(points, float)
        faces = orient_outward(points, np.asarray(faces))
        self._lods = {}
        for name, n in (("fine", FINE_FACES), ("coarse", COARSE_FACES)):
            p, f = decimate(points, faces, n)
            nrm = np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]])
            nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
            self._lods[name] = (p, f, nrm)
        self._center = (points.min(0) + points.max(0)) / 2
        self._radius = float(np.linalg.norm(points - self._center, axis=1).max())
        rgb = np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)]) / 255
        levels = np.linspace(0.22, 1.0, 64)
        self._palette = ["#%02x%02x%02x" % tuple((255 * np.clip(rgb * l, 0, 1)).astype(int)) for l in levels]
        self.redraw()

    def clear(self) -> None:
        self._lods, self._markers = None, []
        self.redraw()

    def set_markers(self, markers: list[dict]) -> None:
        """Each marker: position (3,), label, color, optional radius (px) and highlight (bool)."""
        self._markers = list(markers)
        self.redraw()

    def set_view(self, up, front) -> None:
        """Orient the camera so that ``up`` points up and ``front`` faces the viewer."""
        y = np.asarray(up, float) / np.linalg.norm(up)
        z = np.asarray(front, float) - np.dot(front, y) * y
        z /= np.linalg.norm(z)
        self._home = np.stack([np.cross(y, z), y, z])
        self.reset_view()

    def reset_view(self) -> None:
        self._R, self._zoom, self._pan = self._home.copy(), 1.0, np.zeros(2)
        self.redraw()

    # ------------------------------------------------------------------ interaction
    def _start(self, e, mode):
        self._drag = (mode, e.x, e.y)

    def _move(self, e):
        if not self._drag:
            return
        mode, x0, y0 = self._drag
        dx, dy = e.x - x0, e.y - y0
        if mode == "rotate":
            self._R = _rot(0, dy * 0.01) @ _rot(1, dx * 0.01) @ self._R
        else:
            self._pan += [dx, dy]
        self._drag = (mode, e.x, e.y)
        self._schedule(coarse=True)

    def _end(self, e):
        self._drag = None
        self.redraw()

    def _wheel(self, e):
        self._zoom_by(1.1 if e.delta > 0 else 1 / 1.1)

    def _zoom_by(self, f):
        self._zoom = float(np.clip(self._zoom * f, 0.2, 20))
        self.redraw()

    def _schedule(self, coarse=False):
        if not self._pending:
            self._pending = True
            self.after_idle(lambda: (setattr(self, "_pending", False), self.redraw(coarse=coarse)))

    # ------------------------------------------------------------------ drawing
    def _project(self, pts, w, h, scale):
        cam = (np.asarray(pts, float) - self._center) @ self._R.T
        sx = w / 2 + self._pan[0] + scale * cam[:, 0]
        sy = h / 2 + self._pan[1] - scale * cam[:, 1]
        return sx, sy, cam[:, 2]

    def redraw(self, coarse: bool = False) -> None:
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 20 or h < 20:
            return
        if self._lods is None:
            if self.message:
                self.create_text(w / 2, h / 2, text=self.message, fill="#9aa4b5", font=("Segoe UI", 12))
            return
        scale = self._zoom * 0.46 * min(w, h) / self._radius
        pts, faces, nrm = self._lods["coarse" if coarse else "fine"]
        sx, sy, sz = self._project(pts, w, h, scale)
        n_cam = nrm @ self._R.T
        visible = np.flatnonzero(n_cam[:, 2] > 0)
        depth = sz[faces[visible]].mean(axis=1)
        order = visible[np.argsort(depth)]
        shade = np.clip(n_cam[order] @ LIGHT, 0, 1)
        colors = np.minimum((shade * 63).astype(int), 63)
        xy = np.stack([sx[faces[order]], sy[faces[order]]], axis=-1).reshape(len(order), 6).tolist()
        pal = self._palette
        for c, ci in zip(xy, colors):
            col = pal[ci]
            self.create_polygon(c, fill=col, outline=col)
        if not coarse:
            self._draw_markers(sx, sy, sz, faces, visible, scale, w, h)
        self.create_text(10, h - 10, anchor="sw", fill="#7d879a", font=("Segoe UI", 9),
                         text="drag: rotate   right-drag: pan   wheel: zoom   double-click: reset")

    def _draw_markers(self, sx, sy, sz, faces, visible, scale, w, h):
        if not self._markers:
            return
        # coarse depth buffer of the front-most surface, to tell hidden markers
        cell = 6
        gw, gh = w // cell + 1, h // cell + 1
        zbuf = np.full(gw * gh, -np.inf)
        tri = faces[visible]
        for k in range(3):   # vertices and centroids of visible faces
            gx = np.clip((sx[tri[:, k]] // cell).astype(int), 0, gw - 1)
            gy = np.clip((sy[tri[:, k]] // cell).astype(int), 0, gh - 1)
            np.maximum.at(zbuf, gy * gw + gx, sz[tri[:, k]])
        cx, cy = sx[tri].mean(1), sy[tri].mean(1)
        gx = np.clip((cx // cell).astype(int), 0, gw - 1)
        gy = np.clip((cy // cell).astype(int), 0, gh - 1)
        np.maximum.at(zbuf, gy * gw + gx, sz[tri].mean(1))

        mx, my, mz = self._project(np.stack([m["position"] for m in self._markers]), w, h, scale)
        for m, x, y, z in sorted(zip(self._markers, mx, my, mz), key=lambda t: (t[0].get("highlight", False), t[3])):
            gx, gy = int(np.clip(x // cell, 0, gw - 1)), int(np.clip(y // cell, 0, gh - 1))
            near = zbuf.reshape(gh, gw)[max(gy - 1, 0):gy + 2, max(gx - 1, 0):gx + 2]
            hidden = near.size and near.max() > z + 3.0
            r = m.get("radius", 7) * (1.5 if m.get("highlight") else 1.0)
            col = m.get("color", "#f2a33a")
            if m.get("highlight"):
                self.create_oval(x - r - 5, y - r - 5, x + r + 5, y + r + 5, outline=col, width=2)
            if hidden:
                self.create_oval(x - r, y - r, x + r, y + r, outline=col, width=2, dash=(3, 2))
            else:
                self.create_oval(x - r, y - r, x + r, y + r, fill=col, outline="#111111", width=1)
            label = m.get("label")
            if label:
                font = ("Segoe UI", 11 if m.get("highlight") else 10, "bold")
                self.create_text(x + r + 5, y - r, anchor="w", text=label, fill="#000000", font=font)
                self.create_text(x + r + 4, y - r - 1, anchor="w", text=label,
                                 fill="#ffffff" if not hidden else "#c9d1dc", font=font)
