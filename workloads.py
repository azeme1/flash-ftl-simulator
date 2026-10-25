import numpy as np


def full_disk(chunk_min, chunk_max, page_count):
    assert page_count > 0
    assert 1 <= chunk_min <= chunk_max <= page_count

    start_lba = 0
    while start_lba < page_count:
        chunk_size = np.random.randint(chunk_min, chunk_max + 1)
        chunk_size = min(chunk_size, page_count - start_lba)
        yield start_lba, chunk_size
        start_lba += chunk_size


def random_write(chunk_min, chunk_max, page_count, chunk_count):
    assert page_count > 0
    assert chunk_count >= 0
    assert 1 <= chunk_min <= chunk_max <= page_count

    for _ in range(chunk_count):
        chunk_size = np.random.randint(chunk_min, chunk_max + 1)
        start_lba = np.random.randint(page_count - chunk_size + 1)
        yield start_lba, chunk_size

