import numpy as np

import logging
from ftl.parameters import BLOCK_BITS, PAGE_BITS, decode_address, encode_address
from ftl.gc_policies import CLEAN_BLOCKS, GC_POLICIES, MONITOR_BLOCKS

# L2P entropy treats physical addresses as uint16
assert PAGE_BITS + BLOCK_BITS <= 16, "Physical address does not fit in 16 bits"


PAGE_INVALID = -1
PAGE_EMPTY = -2

LBA_EMPTY = -1

EMPTY_BLOCK = -1
OPENED_BLOCK = 0
CLOSED_BLOCK = 1
GC_BLOCK = 2
WA_WINDOW_BLOCKS = 16  # write_lba and write_storage restart every N blocks of host writes (windowed WA)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def shannon_entropy(data: np.ndarray, bins: int = 256) -> float:
    """Shannon entropy (bits/symbol) of a 1-D array of non-negative ints < bins."""
    if data.size == 0:
        return 0.0
    counts = np.bincount(data, minlength=bins)
    p = counts[counts > 0] / data.size  # drop zero-probability symbols
    return float(np.sum(p * np.log2(1 / p)))


class FTL:
    def __init__(self, config, gc="locality", monitor_blocks=MONITOR_BLOCKS, clean_blocks=CLEAN_BLOCKS):
        assert gc in GC_POLICIES, f"Unknown gc {gc!r}, expected one of {tuple(GC_POLICIES)}"
        # GC starts with at least monitor_blocks - 1 empty blocks, so it always has room for its destinations
        assert 1 <= clean_blocks < monitor_blocks, \
            f"Need 1 <= clean_blocks < monitor_blocks, got clean_blocks={clean_blocks}, monitor_blocks={monitor_blocks}"
        self.config = config
        self.gc_policy = GC_POLICIES[gc]()
        self.monitor_blocks = monitor_blocks
        self.clean_blocks = clean_blocks
        self.over_provisioning = 0.07
        self.block_count_total = config["total_blocks"]
        self.pages_per_block = config["pages_per_block"]
        self.block_count_lba = int(self.block_count_total / (1 + self.over_provisioning))
        self.block_count_op = int(self.block_count_total - self.block_count_lba)

        self.page_count_physical = self.block_count_total * self.pages_per_block
        self.page_count_logical = self.block_count_lba * self.pages_per_block

        self.block_erase_counter = np.zeros(self.block_count_total)
        self.block_status = np.full((self.block_count_total, ), EMPTY_BLOCK)
        self.block_valid_pages = np.full((self.block_count_total, ), 0)

        self.L2P = np.full((self.page_count_logical, ), LBA_EMPTY)
        self.P2L = np.full((self.block_count_total, self.pages_per_block), PAGE_EMPTY)

        self.open_block_index = None
        self.gc_block_index = None
        self.gc_source_blocks = []  # blocks being collected by the running GC, empty outside GC
        self.page_index = None

        # Metrics
        self.write_lba = 0
        self.write_storage = 0

        # Visualization hook: called as on_page_write(ftl) after every host and GC page write
        self.on_page_write = None

    def open_block(self):
        selection = (self.block_status == EMPTY_BLOCK) & (self.block_status != GC_BLOCK) & (self.block_status != OPENED_BLOCK)
        self.open_block_index = np.where(selection)[0].min()
        self.block_status[self.open_block_index] = OPENED_BLOCK
        logger.debug(f"Open Block {self.open_block_index}")

    def open_gc_block(self):
        selection = (self.block_status == EMPTY_BLOCK) & (self.block_status != GC_BLOCK) & (self.block_status != OPENED_BLOCK)
        self.gc_block_index = np.where(selection)[0].min()
        self.block_status[self.gc_block_index] = GC_BLOCK
        logger.debug(f"Open GC Block {self.gc_block_index}")

    def close_block(self):
        if self.open_block_index is not None:
            self.block_status[self.open_block_index] = CLOSED_BLOCK

    def new_physical_address(self):
        if self.page_index is None:
            self.page_index = 0
        else:
            self.page_index = (self.page_index + 1) % self.pages_per_block

        if self.page_index == 0:
            self.close_block()
            self.open_block()

        return encode_address(self.open_block_index, self.page_index)

    def erase_block(self, block_index):
        assert np.all(self.P2L[block_index] < 0)  # invalid, or never written in a partly filled GC block
        self.P2L[block_index, ...] = PAGE_EMPTY
        self.block_erase_counter[block_index] += 1
        self.block_status[block_index] = EMPTY_BLOCK
        self.block_valid_pages[block_index] = 0
        logger.debug(f"Erase Block {block_index}")

    def erase_block_if_possible(self, block_index):
        """Erase a closed block once it holds no valid pages. Returns True if the block was erased."""
        if self.block_valid_pages[block_index] != 0:
            return False
        self.erase_block(block_index)
        return True

    def write(self, lba_index, chuck_size):
        # Validate the whole chunk up front, so a bad request never leaves a write half done
        assert 0 <= lba_index and lba_index + chuck_size <= self.page_count_logical, \
            f"Write of LBAs [{lba_index}, {lba_index + chuck_size}) is outside the device (0..{self.page_count_logical - 1})"
        for _lba in range(lba_index, lba_index + chuck_size):
            # WA is measured per window: both counters restart every WA_WINDOW_BLOCKS blocks of host writes
            if self.write_lba == WA_WINDOW_BLOCKS * self.pages_per_block:
                self.write_lba = 0
                self.write_storage = 0

            # TODO: erase empty blocks if possible

            if self.gc_policy.need_gc(self):
                self.gc()

            address_physical = self.new_physical_address()

            if self.L2P[_lba] >= 0:
                address_physical_current = self.L2P[_lba]
                block_current, page_current = decode_address(address_physical_current)
                assert self.P2L[block_current, page_current] == _lba
                self.P2L[block_current, page_current] = PAGE_INVALID
                self.block_valid_pages[block_current] -= 1
                assert self.block_valid_pages[block_current] >= 0

            self.L2P[_lba] = address_physical

            assert self.P2L[self.open_block_index, self.page_index] == PAGE_EMPTY

            self.P2L[self.open_block_index, self.page_index] = _lba
            self.block_valid_pages[self.open_block_index] += 1
            assert self.block_valid_pages[self.open_block_index] <= self.pages_per_block

            self.write_lba += 1
            self.write_storage += 1

            if self.on_page_write is not None:
                self.on_page_write(self)

    def gc(self):
        closed_blocks = np.where(self.block_status == CLOSED_BLOCK)[0]

        # Check that the closed blocks are consistent with the P2L and block_valid_pages
        for _block_index in closed_blocks:
            assert (self.P2L[_block_index, :] >= 0).sum() == self.block_valid_pages[_block_index]

        # Stable sort: blocks with equal valid page counts are taken in block index order
        closed_blocks_indexes = np.argsort(self.block_valid_pages[closed_blocks], kind="stable")
        gc_block_indexes = closed_blocks[closed_blocks_indexes]
        # Skip Open Block and GC Block, which are never victims
        gc_block_indexes = gc_block_indexes[(gc_block_indexes != self.open_block_index) & (gc_block_indexes != self.gc_block_index)]

        # Check the block and page order of the victims: fewest valid pages first, and pages in physical order within each block
        assert np.all((np.diff(self.block_valid_pages[gc_block_indexes]) >= 0)), "Incorrect sort order"

        gc_block_pages = self.block_valid_pages[gc_block_indexes]
        cutout = np.argwhere(np.cumsum(gc_block_pages) > self.gc_policy.clean_blocks * self.pages_per_block)[0].item() + 1

        self.gc_source_blocks = gc_block_indexes[:cutout]
        
        assert self.open_block_index not in self.gc_source_blocks, "GC selected the host open block"
        assert self.gc_block_index not in self.gc_source_blocks, "GC selected the GC open block"

        # Victims without valid pages need no relocation: free them right away
        for _block_index in self.gc_source_blocks:
            self.erase_block_if_possible(_block_index)

        if not self.gc_policy.need_gc(self):
            # all victims were empty, nothing to do
            return

        # Compute source and target items for the relocation
        lba_items_amount = []
        pages_src_amount = []
        blocks_src_amount = []
        for _block_index in self.gc_source_blocks:
            block_content = self.P2L[_block_index, :]
            _selection = block_content >= 0
            lba_items_amount += block_content[_selection].tolist()
            pages_src_amount += np.where(_selection)[0].tolist()
            blocks_src_amount += [_block_index] * sum(_selection)

        assert len(pages_src_amount) >= self.gc_policy.clean_blocks * self.pages_per_block, "Not enough pages to fill a destination blocks"

        for i in range(self.gc_policy.clean_blocks):
            # Limit items to to number of blocks
            slice_item = slice(i * self.pages_per_block, (i + 1) * self.pages_per_block)
            lba_items = lba_items_amount[slice_item]
            pages_src = pages_src_amount[slice_item]
            blocks_src = blocks_src_amount[slice_item]
            pages_dst = np.arange(self.pages_per_block) 

            self.open_gc_block()
            _block_dst = self.gc_block_index

            for k in range(len(lba_items)):
                _lba = lba_items[k]
                _page_src = pages_src[k]
                _block_src = blocks_src[k]
                _page_dst = pages_dst[k]

                address_physical_dst = encode_address(_block_dst, _page_dst)
                address_physical_src = self.L2P[_lba]
                self.L2P[_lba] = address_physical_dst

                _block_src_check, _page_src_check = decode_address(address_physical_src)
                assert (_block_src, _page_src) == (_block_src_check, _page_src_check), "Wrong storage content"

                self.P2L[_block_src, _page_src] = PAGE_INVALID
                self.P2L[_block_dst, _page_dst] = _lba

                self.block_valid_pages[_block_src] -= 1
                self.block_valid_pages[_block_dst] += 1

                assert 0 <= self.block_valid_pages[_block_src] <= self.pages_per_block
                assert 0 <= self.block_valid_pages[_block_dst] <= self.pages_per_block

                self.write_storage += 1

                # Free a victim as soon as its last valid page has moved
                self.erase_block_if_possible(_block_src)

                if self.on_page_write is not None:
                    self.on_page_write(self)

            self.block_status[_block_dst] = CLOSED_BLOCK

            assert np.all(self.P2L[_block_dst, :len(lba_items)] >= 0)

        self.gc_source_blocks = []
        logger.debug("Done GC")

    def check(self, l2p=True, p2l=True, valid_pages=True):
        if l2p:
            self.check_l2p()
        if p2l:
            self.check_p2l()
        if valid_pages:
            self.check_valid_pages()
        logger.debug("Done Check")

    def check_l2p(self):
        """Every mapped LBA points to a physical page that holds it."""
        for _lba, _address in enumerate(self.L2P):
            if _address >= 0:
                _block, _page = decode_address(_address)
                assert self.P2L[_block, _page] == _lba, f"L2P entry of LBA {_lba} points to a wrong page"

    def check_p2l(self):
        """Every valid physical page is the current location of its LBA
        (this also catches an LBA stored as valid in two places)."""
        _blocks, _pages = np.nonzero(self.P2L >= 0)
        _lbas = self.P2L[_blocks, _pages]
        assert np.all(self.L2P[_lbas] == (_blocks << PAGE_BITS) | _pages), "P2L page not referenced by L2P"

    def check_valid_pages(self):
        """Per-block valid page counters match the P2L content."""
        assert np.all((self.P2L >= 0).sum(axis=1) == self.block_valid_pages), "Wrong block_valid_pages"

    def log_wa(self):
        if self.write_lba == 0:
            wa = 0.0
        else:
            wa = self.write_storage / self.write_lba
        logger.info(f"Write Amplification : {wa}")

    @property
    def utilization(self):
        return (self.L2P >= 0).mean()

    def log_utilization(self):
        logger.info(f"Disk Usage: {self.utilization}")

    @property
    def l2p_entropy(self):
        """Order-0 entropy of consecutive L2P deltas, in bits per L2P entry."""
        deltas = np.diff(self.L2P.astype(np.uint16))  # wraps mod 2**16, lossless
        return shannon_entropy(deltas, bins=2**16)

    def log_entropy(self):
        H = self.l2p_entropy
        logger.info(f"L2P entropy: {H:.4f} bits/entry ({H / 16:.4f} of raw)")
        return H

    def log_erase_counts(self):
        counts = self.block_erase_counter
        logger.info(f"Erase counts: min {counts.min():.0f}, mean {counts.mean():.2f}, "
                    f"max {counts.max():.0f}, std {counts.std():.2f}")
