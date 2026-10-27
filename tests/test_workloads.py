import numpy as np
import pytest

from workload import (
    JESD219_ENTERPRISE_SIZES,
    FullDisk,
    FullDiskRandom,
    Jesd219Client,
    Jesd219Enterprise,
    RandomWrite,
)


@pytest.fixture(autouse=True)
def seed():
    np.random.seed(7)


def expand(chunks):
    """Flatten (start_lba, chunk_size) chunks into the list of LBAs written."""
    return [lba for start, size in chunks for lba in range(start, start + size)]


def split(chunks):
    starts = np.array([start for start, _ in chunks])
    sizes = np.array([size for _, size in chunks])
    return starts, sizes


# --- full_disk ---

@pytest.mark.parametrize("chunk_min, chunk_max, page_count", [
    (1, 16, 1000),
    (4, 4, 10),     # fixed size, last chunk clamped
    (1, 1, 5),      # single-page chunks
    (3, 50, 50),    # chunk_max == page_count
])
def test_full_disk_writes_every_lba_once_in_order(chunk_min, chunk_max, page_count):
    chunks = list(FullDisk(chunk_min, chunk_max, page_count))
    assert expand(chunks) == list(range(page_count))


def test_full_disk_chunk_sizes_within_bounds():
    _, sizes = split(list(FullDisk(2, 9, 10000)))
    assert sizes[:-1].min() >= 2 and sizes[:-1].max() <= 9
    assert 1 <= sizes[-1] <= 9  # last chunk may be clamped below chunk_min


@pytest.mark.parametrize("args", [(0, 4, 10), (5, 4, 10), (1, 11, 10), (1, 4, 0)])
def test_full_disk_rejects_invalid_arguments(args):
    with pytest.raises(AssertionError):
        list(FullDisk(*args))


# --- full_disk_random ---

@pytest.mark.parametrize("chunk_min, chunk_max, page_count", [
    (1, 16, 1000),
    (4, 4, 10),
    (1, 1, 5),
    (3, 50, 50),
])
def test_full_disk_random_writes_every_lba_exactly_once(chunk_min, chunk_max, page_count):
    chunks = list(FullDiskRandom(chunk_min, chunk_max, page_count))
    assert sorted(expand(chunks)) == list(range(page_count))


def test_full_disk_random_uses_the_full_disk_chunks_in_random_order():
    sequential = list(FullDisk(1, 16, 10000))
    np.random.seed(7)
    shuffled = list(FullDiskRandom(1, 16, 10000))
    assert sorted(shuffled) == sequential   # the same chunks
    assert shuffled != sequential           # in a different order


@pytest.mark.parametrize("args", [(0, 4, 10), (5, 4, 10), (1, 11, 10), (1, 4, 0)])
def test_full_disk_random_rejects_invalid_arguments(args):
    with pytest.raises(AssertionError):
        list(FullDiskRandom(*args))


# --- random_write ---

def test_random_write_count_and_bounds():
    starts, sizes = split(list(RandomWrite(1, 16, 1000, 50000)))
    assert len(starts) == 50000
    assert sizes.min() == 1 and sizes.max() == 16
    assert starts.min() == 0 and (starts + sizes).max() == 1000  # both device edges reachable, never past


def test_random_write_zero_chunks():
    assert list(RandomWrite(1, 4, 10, 0)) == []


def test_random_write_full_device_chunk():
    assert list(RandomWrite(5, 5, 5, 3)) == [(0, 5)] * 3


def test_random_write_is_reproducible_with_seed():
    first = list(RandomWrite(1, 16, 1000, 100))
    np.random.seed(7)
    assert list(RandomWrite(1, 16, 1000, 100)) == first


@pytest.mark.parametrize("args", [(0, 4, 10, 1), (5, 4, 10, 1), (1, 11, 10, 1), (1, 4, 10, -1)])
def test_random_write_rejects_invalid_arguments(args):
    with pytest.raises(AssertionError):
        list(RandomWrite(*args))


# --- jesd219_enterprise ---

PAGE_COUNT = 61248
CHUNK_COUNT = 200000


def test_jesd219_enterprise_size_mix_4k_pages():
    _, sizes = split(list(Jesd219Enterprise(PAGE_COUNT, CHUNK_COUNT)))
    # 512 B .. 4 KB -> 1 page (77%), 8K -> 2 (10%), 16K -> 4 (7%), 32K -> 8 (3%), 64K -> 16 (3%)
    expected = {1: 0.77, 2: 0.10, 4: 0.07, 8: 0.03, 16: 0.03}
    assert set(np.unique(sizes)) == set(expected)
    for size, share in expected.items():
        assert (sizes == size).mean() == pytest.approx(share, abs=0.005)


def test_jesd219_enterprise_zone_split():
    starts, _ = split(list(Jesd219Enterprise(PAGE_COUNT, CHUNK_COUNT)))
    hot, warm = PAGE_COUNT * 5 // 100, PAGE_COUNT * 20 // 100
    assert (starts < hot).mean() == pytest.approx(0.50, abs=0.005)
    assert ((starts >= hot) & (starts < warm)).mean() == pytest.approx(0.30, abs=0.005)
    assert (starts >= warm).mean() == pytest.approx(0.20, abs=0.005)


def test_jesd219_enterprise_stays_inside_device():
    starts, sizes = split(list(Jesd219Enterprise(PAGE_COUNT, CHUNK_COUNT)))
    assert starts.min() >= 0
    assert (starts + sizes).max() <= PAGE_COUNT


@pytest.mark.parametrize("page_size, expected_sizes, align", [
    (4096, {1, 2, 4, 8, 16}, 1),
    (1024, {1, 2, 3, 4, 8, 16, 32, 64}, 4),   # 4 KB alignment = 4 pages
    (16384, {1, 2, 4}, 1),                    # transfers below 16 KB round up to one page
])
def test_jesd219_enterprise_adapts_to_page_size(page_size, expected_sizes, align):
    starts, sizes = split(list(Jesd219Enterprise(PAGE_COUNT, 50000, page_size=page_size)))
    assert set(np.unique(sizes)) == expected_sizes
    assert np.all(starts % align == 0)


def test_jesd219_enterprise_size_table_sums_to_100():
    assert sum(JESD219_ENTERPRISE_SIZES.values()) == 100


def test_jesd219_enterprise_zero_chunks():
    assert list(Jesd219Enterprise(PAGE_COUNT, 0)) == []


def test_jesd219_enterprise_is_reproducible_with_seed():
    first = list(Jesd219Enterprise(PAGE_COUNT, 100))
    np.random.seed(7)
    assert list(Jesd219Enterprise(PAGE_COUNT, 100)) == first


def test_jesd219_enterprise_rejects_too_small_device():
    with pytest.raises(AssertionError):
        list(Jesd219Enterprise(10, 1))  # 5% zone would be empty


# --- jesd219_client ---

@pytest.mark.parametrize("record, expected", [
    ((0, 512), (0, 1)),          # sub-page write programs one page
    ((4096, 4096), (1, 1)),      # aligned 4 KB write = one page
    ((4000, 200), (0, 2)),       # unaligned write crossing a page boundary
    ((8192, 65536), (2, 16)),    # aligned 64 KB write
])
def test_jesd219_client_maps_bytes_to_pages(record, expected):
    assert list(Jesd219Client([record], page_count=100)) == [expected]


def test_jesd219_client_custom_page_size():
    assert list(Jesd219Client([(3000, 2000)], page_count=100, page_size=1024)) == [(2, 3)]


def test_jesd219_client_empty_trace():
    assert list(Jesd219Client([], page_count=100)) == []


@pytest.mark.parametrize("record", [(100 * 4096, 512), (99 * 4096, 8192), (-4096, 512)])
def test_jesd219_client_rejects_writes_outside_device(record):
    with pytest.raises(AssertionError):
        list(Jesd219Client([record], page_count=100))
