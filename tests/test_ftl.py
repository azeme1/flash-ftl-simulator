"""FTL tests: greedy GC victim selection, L2P entropy, GC correctness and regressions, block erasure,
and the mapping invariant checks."""

from types import SimpleNamespace

import numpy as np
import pytest

from ftl import CLOSED_BLOCK, EMPTY_BLOCK, FTL, GC_POLICIES, PAGE_INVALID, LbaGC
from ftl.parameters import SLC_SMALL
from workload import FullDisk, RandomWrite


@pytest.fixture(autouse=True)
def seed():
    np.random.seed(7)


def make_ftl(spare_blocks=None, gc="lba"):
    """FTL on SLC_SMALL; spare_blocks overrides the over-provisioning (default 67 spare blocks)."""
    ftl = FTL(SLC_SMALL, gc=gc)
    if spare_blocks is not None:
        ftl.block_count_lba = ftl.block_count_total - spare_blocks
        ftl.block_count_op = spare_blocks
        ftl.page_count_logical = ftl.block_count_lba * ftl.pages_per_block
        ftl.L2P = np.full(ftl.page_count_logical, -1)
    return ftl


def write_all(ftl, workload):
    for lba, size in workload:
        ftl.write(lba, size)


def fill_and_overwrite(ftl, random_chunks):
    write_all(ftl, FullDisk(1, 16, ftl.page_count_logical))
    write_all(ftl, RandomWrite(1, 16, ftl.page_count_logical, random_chunks))


def drive_write_chunks(ftl):
    """Number of 1..16-page random chunks that writes about one drive's worth of pages."""
    return round(ftl.page_count_logical / 8.5)


def leaked_blocks(ftl):
    """Closed blocks without valid pages: GC should have erased them."""
    return int(((ftl.block_status == CLOSED_BLOCK) & (ftl.block_valid_pages == 0)).sum())


def write_amplification(ftl):
    return ftl.write_storage / ftl.write_lba


def record_gc_victims(ftl):
    """Wrap ftl.gc to record every GC run as (victim valid page counts, valid page counts of the closed
    blocks GC left alone), both taken when the run starts."""
    gc_runs = []
    original_gc = ftl.gc

    def recording_gc():
        closed = np.where(ftl.block_status == CLOSED_BLOCK)[0]
        valid = ftl.block_valid_pages[closed].copy()
        victims = []
        previous_hook = ftl.on_page_write

        def hook(f):  # gc_source_blocks is set before GC moves its first page and cleared when it ends
            if not victims:
                victims.extend(int(b) for b in f.gc_source_blocks)
            if previous_hook is not None:
                previous_hook(f)

        ftl.on_page_write = hook
        try:
            original_gc()
        finally:
            ftl.on_page_write = previous_hook
        chosen = np.isin(closed, victims)
        gc_runs.append((valid[chosen], valid[~chosen]))

    ftl.gc = recording_gc
    return gc_runs


@pytest.fixture(scope="module")
def reference_runs():
    """One reference run per GC order, shared by the tests below: default device, sequential fill + one
    random drive write, seed 7, with every GC victim selection recorded."""
    runs = {}
    for gc in GC_POLICIES:
        np.random.seed(7)
        ftl = make_ftl(gc=gc)
        gc_runs = record_gc_victims(ftl)
        write_all(ftl, FullDisk(1, 16, ftl.page_count_logical))
        entropy_after_fill = ftl.l2p_entropy
        write_all(ftl, RandomWrite(1, 16, ftl.page_count_logical, drive_write_chunks(ftl)))
        runs[gc] = SimpleNamespace(ftl=ftl, gc_runs=gc_runs, entropy_after_fill=entropy_after_fill)
    return runs


# --- GC ---

@pytest.mark.parametrize("gc", GC_POLICIES)
def test_gc_policies_keep_mapping_consistent(gc, reference_runs):
    """Every GC page order survives a sequential fill plus one random drive write with consistent
    L2P/P2L mappings and valid-page counters, no leaked blocks, and every LBA still mapped."""
    ftl = reference_runs[gc].ftl
    ftl.check()
    assert leaked_blocks(ftl) == 0
    assert ftl.utilization == 1.0


@pytest.mark.parametrize("gc", GC_POLICIES)
def test_gc_victim_selection_is_greedy(gc, reference_runs):
    """Every GC run takes its victims from the closed blocks with the fewest valid pages: no victim holds
    more valid pages than any closed block that GC left alone. The GC page order must not affect this."""
    gc_runs = reference_runs[gc].gc_runs
    assert len(gc_runs) > 0
    for victims, others in gc_runs:
        assert len(victims) > 0
        if len(others):
            assert victims.max() <= others.min()


def test_write_amplification_regression(reference_runs):
    """Pins the write amplification of the gc="lba" reference run, so any change to GC behavior is
    noticed. Update the expected value deliberately when the GC policy changes."""
    assert write_amplification(reference_runs["lba"].ftl) == pytest.approx(3.0276, abs=1e-4)


# --- L2P entropy ---

@pytest.mark.parametrize("gc", GC_POLICIES)
def test_l2p_entropy_after_sequential_fill_is_zero(gc, reference_runs):
    """After a sequential fill every L2P delta is the same, so the table has zero entropy."""
    assert reference_runs[gc].entropy_after_fill == 0.0


@pytest.mark.parametrize("gc", GC_POLICIES)
def test_l2p_entropy_after_random_writes_is_in_range(gc, reference_runs):
    """Random overwrites scatter the mapping: entropy rises above zero but stays below the 16-bit maximum."""
    assert 0.0 < reference_runs[gc].ftl.l2p_entropy < 16.0


def test_l2p_entropy_regression(reference_runs):
    """Pins the L2P entropy of the gc="lba" reference run, so any change to GC page placement is
    noticed. Update the expected value deliberately when the GC policy changes."""
    assert reference_runs["lba"].ftl.l2p_entropy == pytest.approx(3.0956, abs=1e-4)


def test_lba_gc_gives_lowest_l2p_entropy(reference_runs):
    """Relocating pages in LBA order keeps the mapping most sequential: lower L2P entropy than every GC policy
    that does not write in LBA order."""
    lba_entropy = reference_runs["lba"].ftl.l2p_entropy
    for gc, policy in GC_POLICIES.items():
        if not issubclass(policy, LbaGC):
            assert lba_entropy < reference_runs[gc].ftl.l2p_entropy, gc


def test_fully_invalid_blocks_are_reclaimed():
    """Blocks whose pages were all invalidated by host overwrites are erased by GC.

    A sequential rewrite turns whole blocks invalid. GC used to select such blocks as victims but never
    erase them, so they were lost for good (29 blocks in this scenario)."""
    ftl = make_ftl()
    fill_and_overwrite(ftl, 3000)
    write_all(ftl, FullDisk(64, 64, 100 * ftl.pages_per_block))
    write_all(ftl, RandomWrite(1, 16, ftl.page_count_logical, 20000))
    ftl.check()
    assert leaked_blocks(ftl) == 0


def test_gc_with_few_empty_blocks_does_not_crash():
    """GC limits its destination blocks to the empty blocks available.

    With only 16 spare blocks, GC used to request 15 destination blocks when fewer were empty and crashed
    with a ValueError in open_gc_block after 67,649 host writes. Slow (~14 s) because write amplification
    is very high with so little spare space."""
    ftl = make_ftl(spare_blocks=16)
    fill_and_overwrite(ftl, 400)
    assert ftl.write_lba > 67649
    ftl.check()
    assert leaked_blocks(ftl) == 0


@pytest.mark.parametrize("hot_pages", [16, 500, 959, 960])
def test_gc_with_little_valid_data_does_not_crash(hot_pages):
    """GC works when the drive holds too little valid data to fill its destination blocks.

    Random writes confined to fewer than 961 LBAs on a fresh drive used to crash on the first GC
    (after 64,513 host writes) with a ValueError, because GC insisted on relocating 15 full blocks."""
    ftl = make_ftl()
    write_all(ftl, RandomWrite(1, 16, hot_pages, 9000))
    assert ftl.write_lba > 64513
    ftl.check()


def test_invalid_gc_policy_is_rejected():
    """An unknown gc fails when the FTL is created, not later inside GC."""
    with pytest.raises(AssertionError, match="Unknown gc "):
        FTL(SLC_SMALL, gc="bogus")


# --- write ---

@pytest.mark.parametrize("request_for", [
    lambda pages: (-1, 1),         # negative LBA, used to wrap silently to the last LBA
    lambda pages: (-2, 4),         # chunk starting before LBA 0
    lambda pages: (pages - 2, 4),  # chunk crossing the end, used to write 2 pages, then raise IndexError
    lambda pages: (pages, 1),      # first LBA past the end
], ids=["negative", "starts-before-0", "crosses-end", "past-end"])
def test_write_rejects_lbas_outside_the_device(request_for):
    """A write touching any LBA outside 0..page_count_logical-1 is rejected before anything is written."""
    ftl = make_ftl()
    lba, chunk_size = request_for(ftl.page_count_logical)
    with pytest.raises(AssertionError, match="outside the device"):
        ftl.write(lba, chunk_size)
    assert ftl.write_lba == 0
    assert np.all(ftl.L2P == -1)


def test_write_accepts_the_full_lba_range():
    """The first and last LBAs, and a chunk ending exactly at the end of the device, are valid writes."""
    ftl = make_ftl()
    ftl.write(0, 1)
    ftl.write(ftl.page_count_logical - 4, 4)
    assert ftl.write_lba == 5
    ftl.check()


# --- erase_block_if_possible ---

def test_erase_block_if_possible():
    """erase_block_if_possible leaves a block that still holds valid pages untouched and returns False.
    Once every page is invalid it erases the block (now empty, erase count 1) and returns True."""
    ftl = make_ftl()
    pages = ftl.pages_per_block
    ftl.write(0, pages + 1)  # fills block 0, the extra page closes it and opens block 1
    block = int(np.where(ftl.block_status == CLOSED_BLOCK)[0][0])

    assert ftl.erase_block_if_possible(block) is False  # still holds valid pages
    assert ftl.block_status[block] == CLOSED_BLOCK

    ftl.write(0, pages)  # overwrite every LBA of the block, its pages become invalid
    assert np.all(ftl.P2L[block] == PAGE_INVALID)
    assert ftl.erase_block_if_possible(block) is True
    assert ftl.block_status[block] == EMPTY_BLOCK
    assert ftl.block_erase_counter[block] == 1
    ftl.check()


# --- check ---

@pytest.fixture
def written_ftl():
    """A small, valid FTL state: 20 blocks of LBAs written sequentially, then partly overwritten
    (so there are invalid pages), and verified by check()."""
    ftl = make_ftl()
    write_all(ftl, FullDisk(1, 16, 20 * ftl.pages_per_block))
    write_all(ftl, RandomWrite(1, 16, 20 * ftl.pages_per_block, 100))
    ftl.check()
    return ftl


def corrupt_l2p(ftl):
    """L2P entry points to a page that holds a different LBA."""
    ftl.L2P[7] = ftl.L2P[8]


def corrupt_p2l(ftl):
    """A stale page is marked valid again for an LBA that lives elsewhere."""
    block, page = map(int, np.argwhere(ftl.P2L == PAGE_INVALID)[0])
    ftl.P2L[block, page] = 5  # stale page claims LBA 5, which lives elsewhere
    ftl.block_valid_pages[block] += 1


def corrupt_valid_pages(ftl):
    """A block's valid-page counter no longer matches its P2L content."""
    block =int(np.argmax(ftl.block_valid_pages))
    ftl.block_valid_pages[block] -= 1


@pytest.mark.parametrize("corrupt, method, message", [
    (corrupt_l2p, "check_l2p", "L2P entry of LBA 7"),
    (corrupt_p2l, "check_p2l", "P2L page not referenced by L2P"),
    (corrupt_valid_pages, "check_valid_pages", "Wrong block_valid_pages"),
])
def test_check_detects_corruption(written_ftl, corrupt, method, message):
    """Each invariant check catches its own kind of corruption with a specific message, and check()
    with default flags catches it too."""
    corrupt(written_ftl)
    with pytest.raises(AssertionError, match=message):
        getattr(written_ftl, method)()
    with pytest.raises(AssertionError, match=message):
        written_ftl.check()


def test_check_flags_skip_checks(written_ftl):
    """A check turned off by its flag is skipped, while the default check() still runs it."""
    corrupt_valid_pages(written_ftl)
    written_ftl.check(valid_pages=False)
    with pytest.raises(AssertionError):
        written_ftl.check()
