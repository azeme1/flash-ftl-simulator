"""FTL tests: GC correctness and regressions, block erasure, and the mapping invariant checks."""

import numpy as np
import pytest

from ftl import CLOSED_BLOCK, EMPTY_BLOCK, FTL, GC_ORDERS, PAGE_INVALID
from parameters import SLC_SMALL
from workloads import full_disk, random_write


@pytest.fixture(autouse=True)
def seed():
    np.random.seed(7)


def make_ftl(spare_blocks=None, gc_order="lba"):
    """FTL on SLC_SMALL; spare_blocks overrides the over-provisioning (default 67 spare blocks)."""
    ftl = FTL(SLC_SMALL, gc_order=gc_order)
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
    write_all(ftl, full_disk(1, 16, ftl.page_count_logical))
    write_all(ftl, random_write(1, 16, ftl.page_count_logical, random_chunks))


def drive_write_chunks(ftl):
    """Number of 1..16-page random chunks that writes about one drive's worth of pages."""
    return round(ftl.page_count_logical / 8.5)


def leaked_blocks(ftl):
    """Closed blocks without valid pages: GC should have erased them."""
    return int(((ftl.block_status == CLOSED_BLOCK) & (ftl.block_valid_pages == 0)).sum())


def write_amplification(ftl):
    return ftl.write_storage / ftl.write_lba


# --- GC ---

@pytest.mark.parametrize("gc_order", GC_ORDERS)
def test_gc_orders_keep_mapping_consistent(gc_order):
    """Every GC page order survives a sequential fill plus one random drive write with consistent
    L2P/P2L mappings and valid-page counters, no leaked blocks, and every LBA still mapped."""
    ftl = make_ftl(gc_order=gc_order)
    fill_and_overwrite(ftl, drive_write_chunks(ftl))
    ftl.check()
    assert leaked_blocks(ftl) == 0
    assert ftl.utilization == 1.0


def test_write_amplification_regression():
    """Pins the write amplification of a reference run (default device, gc_order="lba", sequential
    fill + one random drive write, seed 7), so any change to GC behavior is noticed. Update the
    expected value deliberately when the GC policy changes."""
    ftl = make_ftl()
    fill_and_overwrite(ftl, drive_write_chunks(ftl))
    assert write_amplification(ftl) == pytest.approx(2.9284, abs=1e-4)


def test_fully_invalid_blocks_are_reclaimed():
    """Blocks whose pages were all invalidated by host overwrites are erased by GC.

    A sequential rewrite turns whole blocks invalid. GC used to select such blocks as victims but never
    erase them, so they were lost for good (29 blocks in this scenario)."""
    ftl = make_ftl()
    fill_and_overwrite(ftl, 3000)
    write_all(ftl, full_disk(64, 64, 100 * ftl.pages_per_block))
    write_all(ftl, random_write(1, 16, ftl.page_count_logical, 20000))
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


def test_invalid_gc_order_is_rejected():
    """An unknown gc_order fails when the FTL is created, not later inside GC."""
    with pytest.raises(AssertionError, match="Unknown gc_order"):
        FTL(SLC_SMALL, gc_order="bogus")


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
    write_all(ftl, full_disk(1, 16, 20 * ftl.pages_per_block))
    write_all(ftl, random_write(1, 16, 20 * ftl.pages_per_block, 100))
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
