
from ftl import FTL
import numpy as np
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from parameters import SLC_SMALL as config_json
from workloads import full_disk, random_write


CHUNK_SIZE_MIN = 1
CHUNK_SIZE_MAX = 16
RANDOM_DRIVE_WRITES = 3
SNAPSHOT_PERIOD_BLOCKS = 4  # snapshot after every N blocks of host writes
LOG_PERIOD_BLOCKS = 32      # log metrics after every N blocks of host writes


def log_metrics(ftl):
    ftl.log_wa()
    ftl.log_entropy()
    ftl.log_utilization()


def run_phase(ftl, workload, page_total, description):
    snapshot_period = SNAPSHOT_PERIOD_BLOCKS * ftl.pages_per_block
    log_period = LOG_PERIOD_BLOCKS * ftl.pages_per_block
    pages_since_snapshot = 0
    pages_since_log = 0

    with logging_redirect_tqdm(), tqdm(total=page_total, desc=description, unit="page") as progress:
        for lba_index, lba_chunk in workload:
            ftl.write(lba_index, lba_chunk)
            progress.update(lba_chunk)

            pages_since_snapshot += lba_chunk
            if pages_since_snapshot >= snapshot_period:
                ftl.to_image()
                pages_since_snapshot = 0

            pages_since_log += lba_chunk
            if pages_since_log >= log_period:
                log_metrics(ftl)
                pages_since_log = 0

        progress.total = progress.n  # random workloads only estimate their page total
        progress.refresh()

        ftl.to_image()
        log_metrics(ftl)


if __name__ == "__main__":
    np.random.seed(7)

    ftl = FTL(config_json)
    ftl.log_entropy()
    ftl.to_image()

    page_count = ftl.page_count_logical

    # 1. Fill the whole disk sequentially
    run_phase(ftl, full_disk(CHUNK_SIZE_MIN, CHUNK_SIZE_MAX, page_count), page_count, "Sequential fill")

    # 2. Random overwrites, one disk's worth of pages per pass
    mean_chunk = (CHUNK_SIZE_MIN + CHUNK_SIZE_MAX) / 2
    chunk_count = round(page_count / mean_chunk)
    for drive_write in range(1, RANDOM_DRIVE_WRITES + 1):
        workload = random_write(CHUNK_SIZE_MIN, CHUNK_SIZE_MAX, page_count, chunk_count)
        run_phase(ftl, workload, round(chunk_count * mean_chunk), f"Random write {drive_write}/{RANDOM_DRIVE_WRITES}")

    ftl.check()
    print("Done")
