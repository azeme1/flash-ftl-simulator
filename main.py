
from ftl import FTL
import numpy as np
from tqdm import tqdm
from parameters import SLC_SMALL as config_json


if __name__ == "__main__":
    np.random.seed(7)

    ftl = FTL(config_json)
    ftl.log_entropy()

    # lba_range_workload = 32 * ftl.pages_per_block
    lba_range_workload = ftl.block_count_lba * ftl.pages_per_block

    ftl.to_image()

    lba_index = 0
    lba_chunk = 1
    for _ in tqdm(range(ftl.block_count_lba)):
        for _ in range(ftl.pages_per_block):
            ftl.write(lba_index, lba_chunk)
            lba_index += 1

        ftl.to_image()
        ftl.log_wa()
        ftl.log_entropy()
        ftl.log_utilization()

    lba_index = 0
    lba_chunk = 1
    ftl.write(lba_index, lba_chunk)

    ftl.to_image()

    chunk_size_min = 1
    chunk_size_max = 16

    for k in tqdm(range(16 * ftl.block_count_lba)):
        for _ in range(ftl.pages_per_block):
            lba_chunk = np.random.randint(chunk_size_min, chunk_size_max + 1)
            lba_index = np.random.randint(lba_range_workload - lba_chunk + 1)
            ftl.write(lba_index, lba_chunk)

        if k % 64 == 0:
            ftl.to_image()
            ftl.log_wa()
            ftl.log_entropy()

    print("Done")


