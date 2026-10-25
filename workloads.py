import numpy as np


def full_disk_single(chunk_min, chunk_max, page_count):
    assert page_count > 0
    assert chunk_min > 1
    assert chunk_min < chunk_max < page_count

    start_lba = 0
    for _ in range(2 * int(page_count / chunk_min)):
        chunk_size = np.random.randint(chunk_min, chunk_max + 1)
        if start_lba >= page_count:
            yield start_lba, page_count - start_lba
            break
        else:
            yield start_lba, chunk_size

