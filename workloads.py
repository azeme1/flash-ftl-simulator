import numpy as np


# JESD219 enterprise endurance workload: transfer size in bytes -> % of accesses
JESD219_ENTERPRISE_SIZES = {
    512: 4, 1024: 1, 1536: 1, 2048: 1, 2560: 1, 3072: 1, 3584: 1,
    4096: 67, 8192: 10, 16384: 7, 32768: 3, 65536: 3,
}
# (% of accesses, % of LBA space), zones laid out from the start of the LBA range
JESD219_ENTERPRISE_ZONES = ((50, 5), (30, 15), (20, 80))
JESD219_ALIGNMENT = 4096  # bytes


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


def jesd219_enterprise(page_count, chunk_count, page_size=4096):
    """JESD219 enterprise workload: random writes, 4 KB aligned, skewed to the first 20% of LBAs.
    One page is one 4 KB host unit by default, so a 4 KB transfer writes exactly one page."""
    # Transfer sizes rounded up to whole pages: a sub-page write still programs a full page
    sizes = np.array(list(JESD219_ENTERPRISE_SIZES))
    chunk_sizes = -(-sizes // page_size)
    p_sizes = np.array(list(JESD219_ENTERPRISE_SIZES.values())) / 100
    align = max(1, JESD219_ALIGNMENT // page_size)

    p_zones = np.array([zone[0] for zone in JESD219_ENTERPRISE_ZONES]) / 100
    zone_bounds = np.cumsum([0] + [zone[1] for zone in JESD219_ENTERPRISE_ZONES]) * page_count // 100

    assert chunk_count >= 0
    assert np.all(np.diff(zone_bounds) >= align), "page_count too small for the JESD219 zones"
    assert chunk_sizes.max() <= page_count

    sizes_drawn = np.random.choice(chunk_sizes, size=chunk_count, p=p_sizes)
    zones_drawn = np.random.choice(len(p_zones), size=chunk_count, p=p_zones)
    starts = np.random.randint(zone_bounds[zones_drawn], zone_bounds[zones_drawn + 1])

    for start_lba, chunk_size in zip(starts, sizes_drawn):
        start_lba = min(start_lba // align, (page_count - chunk_size) // align) * align
        yield int(start_lba), int(chunk_size)


def jesd219_client(trace, page_count, page_size=4096):
    """JESD219A client workload: replay (byte_offset, byte_length) writes from the JEDEC trace
    (JESD219A_MT master trace or JESD219A_TT test trace), mapped onto pages."""
    for offset, length in trace:
        first_lba = offset // page_size
        last_lba = (offset + length - 1) // page_size
        assert 0 <= first_lba <= last_lba < page_count, f"Trace write at byte {offset} is outside the device"
        yield first_lba, last_lba - first_lba + 1

