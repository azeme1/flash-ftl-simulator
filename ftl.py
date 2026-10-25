import numpy as np

import logging
from parameters import BLOCK_BITS, PAGE_BITS, decode_address, encode_address

# L2P entropy treats physical addresses as uint16
assert PAGE_BITS + BLOCK_BITS <= 16, "Physical address does not fit in 16 bits"


PAGE_INVALID = -1
PAGE_EMPTY = -2

LBA_EMPTY = -1

EMPTY_BLOCK = -1
OPENED_BLOCK = 0
CLOSED_BLOCK = 1
GC_BLOCK = 2
MONITOR_BLOCKS = 16  # GC starts when fewer blocks than this are empty
CLEAN_BLOCKS = 15    # destination blocks filled by one GC run, strictly fewer than MONITOR_BLOCKS

# Order in which GC writes the collected valid pages into its destination blocks
GC_ORDERS = (
    "greedy",   # baseline, as collected: victim blocks (fewest valid pages first), pages in physical order
    "lba",      # by logical address
    "address",  # by current physical address
    "block",    # by source block index
    "page",     # by page index inside the source block
)


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
    def __init__(self, config, gc_order="lba", monitor_blocks=MONITOR_BLOCKS, clean_blocks=CLEAN_BLOCKS):
        assert gc_order in GC_ORDERS, f"Unknown gc_order {gc_order!r}, expected one of {GC_ORDERS}"
        # GC starts with at least monitor_blocks - 1 empty blocks, so it always has room for its destinations
        assert 1 <= clean_blocks < monitor_blocks, \
            f"Need 1 <= clean_blocks < monitor_blocks, got clean_blocks={clean_blocks}, monitor_blocks={monitor_blocks}"
        self.config = config
        self.gc_order = gc_order
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
        self.open_block_index = np.where(self.block_status == EMPTY_BLOCK)[0].min()
        self.block_status[self.open_block_index] = OPENED_BLOCK
        logger.debug(f"Open Block {self.open_block_index}")

    def open_gc_block(self):
        self.gc_block_index = np.where(self.block_status == EMPTY_BLOCK)[0].min()
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
            if (self.block_status == EMPTY_BLOCK).sum() < self.monitor_blocks:
                self.gc()
                # ftl.check()

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
        clean_blocks = self.clean_blocks
        closed_blocks = np.where(self.block_status == CLOSED_BLOCK)[0]

        for _block_index in closed_blocks:
            # print(_block_index, self.block_valid_pages[_block_index], (self.P2L[_block_index, :] >= 0).sum())
            assert (self.P2L[_block_index, :] >= 0).sum() == self.block_valid_pages[_block_index]
        # Stable sort: blocks with equal valid page counts are taken in block index order
        closed_blocks_indexes = np.argsort(self.block_valid_pages[closed_blocks], kind="stable")
        gc_block_indexes = closed_blocks[closed_blocks_indexes]

        assert np.all((np.diff(self.block_valid_pages[gc_block_indexes]) >= 0)), "Incorrect sort order"

        # Free block strategy
        # invalid_pages = self.pages_per_block - self.block_valid_pages[gc_block_indexes]
        # min_gc_block_count = 1 + np.where(np.cumsum(invalid_pages) > self.pages_per_block)[0].min()
        # gc_move_block_list = gc_block_indexes[:min_gc_block_count]

        # Victims: the clean_blocks greedy blocks among the monitor_blocks candidates. All their valid pages
        # are copied; they fit, because GC starts with at least clean_blocks empty blocks
        gc_move_block_list = gc_block_indexes[:self.monitor_blocks][:clean_blocks]
        self.gc_source_blocks = gc_move_block_list

        # Victims without valid pages need no relocation: free them right away
        for _block_index in gc_move_block_list:
            self.erase_block_if_possible(_block_index)

        lba_items = []
        pages_src = []
        blocks_src = []
        for _block_index in gc_move_block_list:
            block_content = self.P2L[_block_index, :]
            _selection = block_content >= 0
            lba_items += block_content[_selection].tolist()
            pages_src += np.where(_selection)[0].tolist()
            blocks_src += [_block_index] * sum(_selection)

        pages_dst = np.arange(self.pages_per_block)

        # Precomputed destination order: one chunk of page_order per destination block, the last may be partial
        page_order = self.gc_page_order(lba_items)
        page_order = [page_order[s:s + self.pages_per_block] for s in range(0, len(page_order), self.pages_per_block)]
        assert len(page_order) <= (self.block_status == EMPTY_BLOCK).sum(), "Not enough empty blocks for GC"

        for _order in page_order:
            self.open_gc_block()    
            _block_dst = self.gc_block_index
            for k, i in enumerate(_order):
                _lba = lba_items[i]
                _page_src = pages_src[i]
                _block_src = blocks_src[i]
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

            assert np.all(self.P2L[_block_dst, :len(_order)] >= 0)

        self.gc_source_blocks = []
        logger.debug("Done GC")

    def gc_page_order(self, lba_items):
        """Permutation of the collected pages, by the configured gc_order. Stable, so ties keep collection order."""
        if self.gc_order == "greedy":
            return np.arange(len(lba_items))
        if self.gc_order == "lba":
            keys = np.asarray(lba_items)
        else:
            addresses = self.L2P[lba_items]
            blocks, pages = decode_address(addresses)
            keys = {"address": addresses, "block": blocks, "page": pages}[self.gc_order]
        return np.argsort(keys, kind="stable")

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
        logger.info(f"Disk Ussage: {self.utilization}")

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
