from contextlib import nullcontext

from ftl import FTL
import numpy as np
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from parameters import SLC_SMALL as config_json
from visualization import FTLVideo
from workloads import full_disk, random_write


CHUNK_SIZE_MIN = 1
CHUNK_SIZE_MAX = 16
RANDOM_DRIVE_WRITES = 7
LOG_PERIOD_BLOCKS = 32      # log metrics after every N blocks of host writes
GC_ORDER = "lba"            # one of ftl.GC_ORDERS: source, lba, address, block, page
VIDEO_PATH = "__dev/ftl.mp4"  # None disables video writing
VIDEO_FPS = 60
VIDEO_FRAME_EVERY = 1       # capture a frame after every N page writes (host and GC)


def log_metrics(ftl):
    ftl.log_wa()
    ftl.log_entropy()
    ftl.log_utilization()
    ftl.log_erase_counts()


def run_phase(ftl, video, workload, page_total, description):
    log_period = LOG_PERIOD_BLOCKS * ftl.pages_per_block
    pages_since_log = 0
    if video is not None:
        video.phase = description

    with logging_redirect_tqdm(), tqdm(total=page_total, desc=description, unit="page") as progress:
        for lba_index, lba_chunk in workload:
            ftl.write(lba_index, lba_chunk)
            progress.update(lba_chunk)

            pages_since_log += lba_chunk
            if pages_since_log >= log_period:
                log_metrics(ftl)
                pages_since_log = 0

        progress.total = progress.n  # random workloads only estimate their page total
        progress.refresh()

        log_metrics(ftl)


if __name__ == "__main__":
    np.random.seed(7)

    ftl = FTL(config_json, gc_order=GC_ORDER)
    ftl.log_entropy()

    page_count = ftl.page_count_logical

    video_context = FTLVideo(ftl, VIDEO_PATH, fps=VIDEO_FPS, frame_every=VIDEO_FRAME_EVERY) if VIDEO_PATH else nullcontext()

    with video_context as video:
        if video is not None:
            ftl.on_page_write = video.capture
            video.write_frame()  # initial empty-disk frame

        # 1. Fill the whole disk sequentially
        run_phase(ftl, video, full_disk(CHUNK_SIZE_MIN, CHUNK_SIZE_MAX, page_count), page_count, "Sequential fill")

        # 2. Random overwrites, one disk's worth of pages per pass
        mean_chunk = (CHUNK_SIZE_MIN + CHUNK_SIZE_MAX) / 2
        chunk_count = round(page_count / mean_chunk)
        for drive_write in range(1, RANDOM_DRIVE_WRITES + 1):
            workload = random_write(CHUNK_SIZE_MIN, CHUNK_SIZE_MAX, page_count, chunk_count)
            run_phase(ftl, video, workload, round(chunk_count * mean_chunk),
                      f"Random write {drive_write}/{RANDOM_DRIVE_WRITES}")

    ftl.check()
    print("Done")
