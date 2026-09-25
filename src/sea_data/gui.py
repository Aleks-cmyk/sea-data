"""Interactive Tkinter viewer for a rendered dataset.

Shows the original photo, the ground-truth overlay and the horizon/
atmosphere/object metadata side by side, and lets a person page through
every image in a dataset with the keyboard.
"""

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any

from PIL import ImageTk

from sea_data.annotations import iter_frames
from sea_data.visualize import render_comparison

MAX_DISPLAY_HEIGHT = 900


class DatasetViewer(tk.Tk):
    """Top-level window that pages through a dataset's comparison images."""

    def __init__(self, output_dir: Path) -> None:
        """Open the viewer on a dataset.

        Args:
            output_dir: Dataset root directory containing ``meta/``.
        """
        super().__init__()
        self.title("sea-data viewer")
        self.geometry("1400x900")

        self.status = tk.Label(self, anchor="w")
        self.status.pack(side="top", fill="x")

        self.image_label = tk.Label(self, bg="black")
        self.image_label.pack(side="top", fill="both", expand=True)

        controls = tk.Frame(self)
        controls.pack(side="bottom", fill="x")
        tk.Button(controls, text="< Prev", command=self.prev).pack(side="left")
        tk.Button(controls, text="Next >", command=self.next).pack(side="left")
        tk.Button(controls, text="Open...", command=self.open_dataset).pack(side="left")
        self.bind("<Left>", lambda _event: self.prev())
        self.bind("<Right>", lambda _event: self.next())

        self._photo: ImageTk.PhotoImage | None = None
        self.output_dir = output_dir
        self.frames: list[dict[str, Any]] = []
        self.index = 0
        self.load_dataset(output_dir)

    def load_dataset(self, output_dir: Path) -> None:
        """Load a dataset's frames and show the first one.

        Args:
            output_dir: Dataset root directory containing ``meta/``.
        """
        frames = list(iter_frames(output_dir))
        if not frames:
            messagebox.showerror(
                "sea-data viewer", f"no metadata found in {output_dir}"
            )
            return
        self.output_dir, self.frames, self.index = output_dir, frames, 0
        self.show(0)

    def open_dataset(self) -> None:
        """Prompt for a dataset folder and load it."""
        chosen = filedialog.askdirectory(initialdir=str(self.output_dir))
        if chosen:
            self.load_dataset(Path(chosen))

    def show(self, index: int) -> None:
        """Render and display the frame at ``index``.

        Args:
            index: Frame index, wrapped into range.
        """
        self.index = index % len(self.frames)
        frame = self.frames[self.index]
        image = render_comparison(self.output_dir, frame)
        if image.height > MAX_DISPLAY_HEIGHT:
            scale = MAX_DISPLAY_HEIGHT / image.height
            image = image.resize(
                (round(image.width * scale), round(image.height * scale))
            )
        self._photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=self._photo)
        self.status.config(
            text=f"{self.index + 1}/{len(self.frames)}  "
            f"{frame['image']}  ({self.output_dir})"
        )

    def prev(self) -> None:
        """Show the previous frame, wrapping around at the start."""
        if self.frames:
            self.show(self.index - 1)

    def next(self) -> None:
        """Show the next frame, wrapping around at the end."""
        if self.frames:
            self.show(self.index + 1)


def run_viewer(output_dir: Path) -> None:
    """Launch the interactive dataset viewer and block until it is closed.

    Args:
        output_dir: Dataset root directory containing ``meta/``.
    """
    DatasetViewer(output_dir).mainloop()
