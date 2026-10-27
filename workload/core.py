"""Workloads: iterables of (start_lba, chunk_size) host writes, measured in pages.

Arguments are checked when a workload is created. Random values are drawn while iterating, from np.random,
so np.random.seed makes runs repeatable."""

import numpy as np


# JESD219 enterprise endurance workload: transfer size in bytes -> % of accesses
JESD219_ENTERPRISE_SIZES = {
    512: 4, 1024: 1, 1536: 1, 2048: 1, 2560: 1, 3072: 1, 3584: 1,
    4096: 67, 8192: 10, 16384: 7, 32768: 3, 65536: 3,
}
# (% of accesses, % of LBA space), zones laid out from the start of the LBA range
JESD219_ENTERPRISE_ZONES = ((50, 5), (30, 15), (20, 80))
JESD219_ALIGNMENT = 4096  # bytes


class Workload:
    """Base class: iterating yields (start_lba, chunk_size) writes."""

    def __iter__(self):
        raise NotImplementedError


class RandomChunkWorkload(Workload):
    """Writes in chunks of a uniformly random size between chunk_min and chunk_max pages."""

    def __init__(self, chunk_min, chunk_max, page_count):
        assert page_count > 0
        assert 1 <= chunk_min <= chunk_max <= page_count
        self.chunk_min = chunk_min
        self.chunk_max = chunk_max
        self.page_count = page_count

    def random_chunk_size(self):
        return np.random.randint(self.chunk_min, self.chunk_max + 1)


class FullDisk(RandomChunkWorkload):
    """Writes every LBA once, sequentially, in random-sized chunks (the last chunk is clamped)."""

    def __iter__(self):
        start_lba = 0
        while start_lba < self.page_count:
            chunk_size = min(self.random_chunk_size(), self.page_count - start_lba)
            yield start_lba, chunk_size
            start_lba += chunk_size


class FullDiskRandom(FullDisk):
    """Writes every LBA once, in the same random-sized chunks as FullDisk, but in random chunk order."""

    def __iter__(self):
        chunks = list(super().__iter__())
        for i in np.random.permutation(len(chunks)):
            yield chunks[i]


class RandomWrite(RandomChunkWorkload):
    """chunk_count random-sized chunks at uniformly random positions inside the device."""

    def __init__(self, chunk_min, chunk_max, page_count, chunk_count):
        assert chunk_count >= 0
        super().__init__(chunk_min, chunk_max, page_count)
        self.chunk_count = chunk_count

    def __iter__(self):
        for _ in range(self.chunk_count):
            chunk_size = self.random_chunk_size()
            start_lba = np.random.randint(self.page_count - chunk_size + 1)
            yield start_lba, chunk_size


class Jesd219Enterprise(Workload):
    """JESD219 enterprise workload: random writes, 4 KB aligned, skewed to the first 20% of LBAs.
    One page is one 4 KB host unit by default, so a 4 KB transfer writes exactly one page."""

    def __init__(self, page_count, chunk_count, page_size=4096):
        # Transfer sizes rounded up to whole pages: a sub-page write still programs a full page
        sizes = np.array(list(JESD219_ENTERPRISE_SIZES))
        self.chunk_sizes = -(-sizes // page_size)
        self.p_sizes = np.array(list(JESD219_ENTERPRISE_SIZES.values())) / 100
        self.align = max(1, JESD219_ALIGNMENT // page_size)

        self.p_zones = np.array([zone[0] for zone in JESD219_ENTERPRISE_ZONES]) / 100
        self.zone_bounds = np.cumsum([0] + [zone[1] for zone in JESD219_ENTERPRISE_ZONES]) * page_count // 100

        assert chunk_count >= 0
        assert np.all(np.diff(self.zone_bounds) >= self.align), "page_count too small for the JESD219 zones"
        assert self.chunk_sizes.max() <= page_count
        self.page_count = page_count
        self.chunk_count = chunk_count

    def __iter__(self):
        sizes_drawn = np.random.choice(self.chunk_sizes, size=self.chunk_count, p=self.p_sizes)
        zones_drawn = np.random.choice(len(self.p_zones), size=self.chunk_count, p=self.p_zones)
        starts = np.random.randint(self.zone_bounds[zones_drawn], self.zone_bounds[zones_drawn + 1])

        for start_lba, chunk_size in zip(starts, sizes_drawn):
            start_lba = min(start_lba // self.align, (self.page_count - chunk_size) // self.align) * self.align
            yield int(start_lba), int(chunk_size)


class Jesd219Client(Workload):
    """JESD219A client workload: replay (byte_offset, byte_length) writes from the JEDEC trace
    (JESD219A_MT master trace or JESD219A_TT test trace), mapped onto pages."""

    def __init__(self, trace, page_count, page_size=4096):
        self.trace = trace
        self.page_count = page_count
        self.page_size = page_size

    def __iter__(self):
        for offset, length in self.trace:
            first_lba = offset // self.page_size
            last_lba = (offset + length - 1) // self.page_size
            assert 0 <= first_lba <= last_lba < self.page_count, f"Trace write at byte {offset} is outside the device"
            yield first_lba, last_lba - first_lba + 1
