import math
import os

import cv2
import numpy as np

from ftl import EMPTY_BLOCK, GC_BLOCK, PAGE_INVALID, shannon_entropy


# Colors are BGR (OpenCV order), dark theme
BACKGROUND = (24, 18, 18)
FRAME_LINE = (72, 62, 60)
TEXT = (235, 235, 235)
TEXT_DIM = (150, 140, 140)
EMPTY_COLOR = (66, 54, 50)
VALID_COLOR = (182, 196, 46)
INVALID_COLOR = (107, 107, 255)
OPEN_BLOCK_COLOR = (102, 209, 255)
GC_BLOCK_COLOR = (229, 93, 155)
GC_SOURCE_COLOR = (245, 245, 245)
WA_COLOR = OPEN_BLOCK_COLOR
ENTROPY_COLOR = VALID_COLOR
ERASE_COLOR = GC_BLOCK_COLOR

FRAME_SIZE = (1280, 720)  # width, height
GRID_PX = 576             # target size of the block grid
GAP_PX = 2                # gap between blocks, also holds the open/GC block outline
MARGIN_PX = 48
FONT = cv2.FONT_HERSHEY_SIMPLEX
STATS_PERIOD_FRAMES = 60  # refresh L2P entropy and the charts once per video second

# Panel layout: (title baseline, box top, box bottom)
WA_CHART = (300, 310, 400)
ENTROPY_CHART = (430, 440, 530)
ERASE_CHART = (560, 570, 648)


class FTLVideo:
    """Renders the FTL state into an H.264 video, one frame per `frame_every` page writes (host and GC).

    Attach with `ftl.on_page_write = video.capture`.
    """

    def __init__(self, ftl, path, fps=60, frame_every=1):
        self.ftl = ftl
        self.frame_every = frame_every
        self.phase = ""
        self.page_writes = 0
        self.frame_count = 0
        self.entropy = 0.0
        self.wa_history = []
        self.entropy_history = []

        # Each block is drawn as a tile_h x tile_w square of pages, blocks laid out in a rows x cols grid
        block_count, ppb = ftl.block_count_total, ftl.pages_per_block
        self.tile_w = 2 ** math.ceil(math.log2(ppb) / 2)
        self.tile_h = math.ceil(ppb / self.tile_w)
        self.cols = math.ceil(math.sqrt(block_count))
        self.rows = math.ceil(block_count / self.cols)
        self.cell = max(1, (GRID_PX // max(self.rows, self.cols) - GAP_PX) // max(self.tile_w, self.tile_h))
        self.block_h = self.tile_h * self.cell + GAP_PX
        self.block_w = self.tile_w * self.cell + GAP_PX
        self.grid_x = MARGIN_PX
        self.grid_y = (FRAME_SIZE[1] - self.rows * self.block_h) // 2
        self.panel_x = 2 * MARGIN_PX + self.cols * self.block_w
        self.panel_right = FRAME_SIZE[0] - MARGIN_PX

        # Page codes: 0 empty, 1 invalid, 2 valid, 3 padding (when ppb is not tile_h * tile_w)
        self.palette = np.array([EMPTY_COLOR, INVALID_COLOR, VALID_COLOR, BACKGROUND], dtype=np.uint8)
        self.codes = np.full((block_count, self.tile_h * self.tile_w), 3, dtype=np.uint8)
        self.tiles = np.empty((self.rows * self.cols, self.block_h, self.block_w, 3), dtype=np.uint8)
        self.tiles[...] = BACKGROUND

        self.background = self.render_background()

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"avc1"), fps, FRAME_SIZE)
        assert self.writer.isOpened(), f"Cannot open video writer for {path}"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.writer.release()

    def capture(self, _ftl=None):
        self.page_writes += 1
        if self.page_writes % self.frame_every == 0:
            self.write_frame()

    def write_frame(self):
        if self.frame_count % STATS_PERIOD_FRAMES == 0:
            self.update_stats()
        self.writer.write(self.render())
        self.frame_count += 1

    def update_stats(self):
        self.entropy = shannon_entropy(np.diff(self.ftl.L2P.astype(np.uint16)), bins=2**16)
        self.wa_history.append(self.write_amplification())
        self.entropy_history.append(self.entropy)

    def write_amplification(self):
        return self.ftl.write_storage / self.ftl.write_lba if self.ftl.write_lba else 1.0

    def gc_active(self):
        ftl = self.ftl
        return ftl.gc_block_index is not None and ftl.block_status[ftl.gc_block_index] == GC_BLOCK

    # --- drawing ---

    def render_background(self):
        ftl = self.ftl
        frame = np.empty((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
        frame[...] = BACKGROUND

        x = self.panel_x
        text(frame, "FLASH FTL SIMULATOR", (x, 88), 0.9, TEXT, 2)
        op = ftl.block_count_op / ftl.block_count_lba
        text(frame, f"{ftl.config['name']}  |  {ftl.block_count_total} blocks x {ftl.pages_per_block} pages  |  "
                    f"{op:.0%} OP", (x, 116), 0.45, TEXT_DIM)
        cv2.line(frame, (x, 168), (self.panel_right, 168), FRAME_LINE, 1)

        for title, (title_y, top, bottom) in (("WRITE AMPLIFICATION", WA_CHART),
                                               ("L2P ENTROPY, BITS/ENTRY", ENTROPY_CHART),
                                               ("ERASE COUNT PER BLOCK", ERASE_CHART)):
            text(frame, title, (x, title_y), 0.45, TEXT_DIM)
            cv2.rectangle(frame, (x, top), (self.panel_right, bottom), FRAME_LINE, 1)

        legend_y = self.grid_y + self.rows * self.block_h + 28
        legend_x = self.grid_x
        for label, color in (("Valid", VALID_COLOR), ("Invalid", INVALID_COLOR), ("Empty", EMPTY_COLOR)):
            cv2.rectangle(frame, (legend_x, legend_y - 11), (legend_x + 12, legend_y + 1), color, -1)
            legend_x = text(frame, label, (legend_x + 20, legend_y), 0.45, TEXT_DIM) + 24
        for label, color in (("Open block", OPEN_BLOCK_COLOR), ("GC open block", GC_BLOCK_COLOR),
                             ("GC source", GC_SOURCE_COLOR)):
            cv2.rectangle(frame, (legend_x + 1, legend_y - 10), (legend_x + 11, legend_y), color, 2)
            legend_x = text(frame, label, (legend_x + 20, legend_y), 0.45, TEXT_DIM) + 24

        return frame

    def render(self):
        frame = self.background.copy()
        self.draw_grid(frame)
        self.draw_metrics(frame)
        self.draw_line_chart(frame, self.wa_history + [self.write_amplification()], WA_CHART, WA_COLOR, 1.0, "{:.2f}")
        self.draw_line_chart(frame, self.entropy_history, ENTROPY_CHART, ENTROPY_COLOR, 0.0, "{:.2f}")
        self.draw_erase_histogram(frame)
        return frame

    def draw_grid(self, frame):
        ftl = self.ftl
        block_count, ppb, cell = ftl.block_count_total, ftl.pages_per_block, self.cell

        self.codes[:, :ppb] = 2 * (ftl.P2L >= 0) + (ftl.P2L == PAGE_INVALID)
        pages = self.palette[self.codes].reshape(block_count, self.tile_h, self.tile_w, 3)
        self.tiles[:block_count, :self.tile_h * cell, :self.tile_w * cell] = pages.repeat(cell, 1).repeat(cell, 2)

        grid = (self.tiles.reshape(self.rows, self.cols, self.block_h, self.block_w, 3)
                .transpose(0, 2, 1, 3, 4)
                .reshape(self.rows * self.block_h, self.cols * self.block_w, 3))
        frame[self.grid_y:self.grid_y + grid.shape[0], self.grid_x:self.grid_x + grid.shape[1]] = grid

        for block_index in ftl.gc_source_blocks:
            self.outline_block(frame, block_index, GC_SOURCE_COLOR)
        if ftl.open_block_index is not None:
            self.outline_block(frame, ftl.open_block_index, OPEN_BLOCK_COLOR)
        if self.gc_active():
            self.outline_block(frame, ftl.gc_block_index, GC_BLOCK_COLOR)

    def outline_block(self, frame, block_index, color):
        row, col = divmod(int(block_index), self.cols)
        x = self.grid_x + col * self.block_w
        y = self.grid_y + row * self.block_h
        cv2.rectangle(frame, (x - 1, y - 1), (x + self.tile_w * self.cell, y + self.tile_h * self.cell), color, 2)

    def draw_metrics(self, frame):
        ftl = self.ftl
        x = self.panel_x
        end = text(frame, self.phase, (x, 150), 0.6, OPEN_BLOCK_COLOR, 1)
        if self.gc_active():
            text(frame, "GC", (end + 14, 150), 0.6, GC_BLOCK_COLOR, 2)

        free_blocks = int((ftl.block_status == EMPTY_BLOCK).sum())
        cells = (
            ("Host writes", f"{ftl.write_lba:,}"),
            ("NAND writes", f"{ftl.write_storage:,}"),
            ("Write amp.", f"{self.write_amplification():.3f}"),
            ("L2P entropy", f"{self.entropy:.2f}"),
            ("Utilization", f"{ftl.utilization:.1%}"),
            ("Free blocks", f"{free_blocks}"),
        )
        column_w = (self.panel_right - x) // 2
        for i, (label, value) in enumerate(cells):
            row, col = divmod(i, 2)
            left = x + col * column_w
            right = left + column_w - (24 if col == 0 else 0)
            y = 200 + row * 30
            text(frame, label, (left, y), 0.5, TEXT_DIM)
            right_text(frame, value, (right, y), 0.5, TEXT)

    def draw_line_chart(self, frame, history, layout, color, y_min, value_format):
        title_y, top, bottom = layout
        left, right = self.panel_x + 1, self.panel_right - 1
        y_max = max(y_min + 1.0, max(history, default=y_min))
        right_text(frame, "now " + value_format.format(history[-1]) if history else "", (self.panel_right, title_y),
                   0.45, color)
        right_text(frame, "max " + value_format.format(y_max), (self.panel_right - 120, title_y), 0.45, TEXT_DIM)
        if len(history) < 2:
            return
        xs = np.linspace(left, right, len(history))
        ys = (bottom - 6) - (np.array(history) - y_min) / (y_max - y_min) * (bottom - top - 12)
        points = np.stack([xs, ys], axis=1).astype(np.int32)
        cv2.polylines(frame, [points], False, color, 2, cv2.LINE_AA)

    def draw_erase_histogram(self, frame):
        title_y, top, bottom = ERASE_CHART
        erases = self.ftl.block_erase_counter.astype(np.int64)
        stats = f"min {erases.min()}  mean {erases.mean():.2f}  max {erases.max()}  std {erases.std():.2f}"
        right_text(frame, stats, (self.panel_right, title_y), 0.45, ERASE_COLOR)

        counts = np.bincount(erases)
        left, right = self.panel_x + 2, self.panel_right - 2
        edges = np.linspace(left, right, len(counts) + 1).astype(int)
        heights = (counts / counts.max() * (bottom - top - 6)).astype(int)
        for i, h in enumerate(heights):
            if h:
                cv2.rectangle(frame, (edges[i], bottom - 1 - h), (max(edges[i], edges[i + 1] - 2), bottom - 1),
                              ERASE_COLOR, -1)
        text(frame, "0", (self.panel_x, bottom + 18), 0.4, TEXT_DIM)
        right_text(frame, f"{erases.max()} erases", (self.panel_right, bottom + 18), 0.4, TEXT_DIM)


def text(frame, value, origin, scale, color, thickness=1):
    """Draw text at origin (left, baseline); return the x coordinate where it ends."""
    cv2.putText(frame, value, origin, FONT, scale, color, thickness, cv2.LINE_AA)
    return origin[0] + cv2.getTextSize(value, FONT, scale, thickness)[0][0]


def right_text(frame, value, origin, scale, color, thickness=1):
    """Draw text right-aligned to origin (right, baseline)."""
    width = cv2.getTextSize(value, FONT, scale, thickness)[0][0]
    cv2.putText(frame, value, (origin[0] - width, origin[1]), FONT, scale, color, thickness, cv2.LINE_AA)
