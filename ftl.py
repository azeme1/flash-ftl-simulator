import numpy as np

import cv2
import os
import logging
from parameters import decode_address, encode_address


PAGE_INVALID = -1
PAGE_EMPTY = -2

PAGE_EMPTY_COLOR = (128, 128, 128)
PAGE_INVALID_COLOR = (0, 0, 255)
PAGE_LBA_COLOR = (0, 255, 0)
OPEN_BLOCK_COLOR = (0, 255, 255)

LBA_EMPTY = -1

EMPTY_BLOCK = -1
OPENED_BLOCK = 0
CLOSED_BLOCK = 1
GC_BLOCK = 2
FTL_SNAPSHOT_FOLDER = "__dev/snapshots/"
MAX_GC_BLOCKS = 15


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def shannon_entropy(data: np.ndarray, bins: int = 256) -> float:
    """Shannon entropy (bits/symbol) of a 1-D uint8 array."""
    if data.size == 0:
        return 0.0
    counts = np.bincount(data.astype(np.uint8), minlength=bins)
    p = counts / counts.sum()
    p = p[p > 0]                       # drop zero-probability symbols
    return float(-np.sum(p * np.log2(p)))


class FTL:
    def __init__(self, config):
        self.config = config
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
        self.page_index = None

        self.min_empty_blocks = MAX_GC_BLOCKS + 1

        # Metrics
        self.write_lba = 0
        self.write_storage = 0

        # Visualization section
        self.image_counter = 0
        self.image_template = np.full((self.block_count_total, self.pages_per_block, 3), np.nan)

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
        assert np.all(self.P2L[block_index] == PAGE_INVALID)
        self.P2L[block_index, ...] = PAGE_EMPTY
        self.block_erase_counter[block_index] += 1
        self.block_status[block_index] = EMPTY_BLOCK
        self.block_valid_pages[block_index] = 0
        logger.debug(f"Erase Block {block_index}")

    def write(self, lba_index, chuck_size):
        for _lba in range(lba_index, lba_index + chuck_size):
            if (self.block_status == EMPTY_BLOCK).sum() < self.min_empty_blocks:
                self.gc(MAX_GC_BLOCKS)
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

    def gc(self, clean_blocks=MAX_GC_BLOCKS):
        closed_blocks = np.where(self.block_status == CLOSED_BLOCK)[0]
        for _block_index in closed_blocks:
            # print(_block_index, self.block_valid_pages[_block_index], (self.P2L[_block_index, :] >= 0).sum())
            assert (self.P2L[_block_index, :] >= 0).sum() == self.block_valid_pages[_block_index]
        closed_blocks_indexes = np.argsort(self.block_valid_pages[closed_blocks])
        gc_block_indexes = closed_blocks[closed_blocks_indexes]

        assert np.all((np.diff(self.block_valid_pages[gc_block_indexes]) >= 0)), "Incorrect sort order"

        # Free block strategy
        # invalid_pages = self.pages_per_block - self.block_valid_pages[gc_block_indexes]
        # min_gc_block_count = 1 + np.where(np.cumsum(invalid_pages) > self.pages_per_block)[0].min()
        # gc_move_block_list = gc_block_indexes[:min_gc_block_count]

        # Write block strategy
        pages_to_move = clean_blocks * self.pages_per_block
        _valid_pages = self.block_valid_pages[gc_block_indexes]
        min_gc_block_count = 1 + np.where(np.cumsum(_valid_pages) > pages_to_move)[0].min()
        gc_move_block_list = gc_block_indexes[:min_gc_block_count]

        lba_items = []
        pages_src = []
        blocks_src = []
        for _block_index in gc_move_block_list:
            block_content = self.P2L[_block_index, :]
            _selection = block_content >= 0
            lba_items += block_content[_selection].tolist()
            pages_src += np.where(_selection)[0].tolist()
            blocks_src += [_block_index] * sum(_selection)

        assert len(pages_src) >= pages_to_move, "Not enough pages for GC"

        pages_dst = np.array(range(pages_to_move))

        lba_items = lba_items[:pages_to_move]
        pages_src = pages_src[:pages_to_move]
        blocks_src = blocks_src[:pages_to_move]
        pages_dst = pages_dst[:pages_to_move]

        page_order = np.arange(pages_to_move)
        page_order = np.argsort(lba_items)
        # page_order = np.argsort(self.L2P[lba_items])
        # page_order = np.argsort([decode_address(item)[0] for item in self.L2P[lba_items]])
        # page_order = np.argsort([decode_address(item)[1] for item in self.L2P[lba_items]])
        page_order = page_order.reshape(clean_blocks, self.pages_per_block)

        for _block, _order in zip(range(clean_blocks), page_order):
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

            self.block_status[_block_dst] = CLOSED_BLOCK

            assert np.all(self.P2L[_block_dst, ...] >= 0)

        for _block in set(blocks_src):
            if self.block_valid_pages[_block] == 0:
                self.erase_block(_block)

        logger.debug("Done GC")

    def check(self):
        for _lba, _address in enumerate(self.L2P):
            if _address >= 0:
                _block, _page = decode_address(_address)
                assert self.P2L[_block, _page] == _lba
        logger.debug("Done Check")

    def log_wa(self):
        wa = self.write_storage / self.write_lba
        logger.info(f"Write Amplification : {wa}")

    @property
    def utilization(self):
        return (self.L2P >= 0).mean()

    def log_utilization(self):
        logger.info(f"Disk Ussage: {self.utilization}")

    def log_entropy(self):
        b = np.diff(self.L2P.astype(np.uint16)).view(np.uint8)

        # print(f"dtype              : {L2P.dtype}")
        # print(f"shape / size       : {L2P.shape}  ({L2P.size} entries)")

        # 1) Byte view (little-endian: low byte, high byte)
        # b = np.ascontiguousarray(L2P.astype(np.uint16)).view(np.uint8)
        # print(f"byte stream length : {b.size} bytes")

        # 2) Global byte entropy
        H = shannon_entropy(b)
        # print(f"Shannon entropy    : {H:.4f} bits/byte")
        # print(f"Max entropy        : 8.0000 bits/byte")
        # print(f"Normalized (H/8)   : {H/8:.4f}")

        # 3) Per-position entropy (low byte vs high byte)
        low  = b[0::2]
        high = b[1::2]
        # print(f"Entropy (low  byte): {shannon_entropy(low):.4f} bits/byte")
        # print(f"Entropy (high byte): {shannon_entropy(high):.4f} bits/byte")

        # 4) Byte histogram highlights
        counts = np.bincount(b, minlength=256)
        total  = counts.sum()
        top = np.argsort(counts)[::-1][:5]
        # print("Top-5 byte values  :")
        # for v in top:
        #     print(f"   0x{v:02X} ({v:3d}) : {counts[v]:>10d}  ({counts[v]/total:6.2%})")

        # 5) Zero-byte fraction and sentinel-byte fraction
        zero_frac     = counts[0x00] / total
        sentinel_frac = (counts[0xFF]) / total
        # print(f"Zero-byte fraction : {zero_frac:.2%}")
        # print(f"0xFF-byte fraction : {sentinel_frac:.2%}")

        # 6) RLE-style lower bound: only runs of identical bytes matter
        #    Count run boundaries — a crude proxy for compressibility.
        if b.size > 0:
            runs = 1 + int(np.sum(b[1:] != b[:-1]))
            rle_ratio = runs / b.size
            # print(f"RLE run count      : {runs}")
            # print(f"RLE ratio          : {rle_ratio:.4f}   (lower = more compressible)")

        # 7) Theoretical bits saved by entropy coding
        raw_bits     = b.size * 8
        entropy_bits = b.size * H
        # print(f"Raw bits           : {raw_bits}")
        # print(f"Entropy-coded bits : {entropy_bits:.0f}")

        logger.info(f"Compression ratio  : {entropy_bits / raw_bits:.4f}")

        return {
            "H": H,
            "H_norm": H / 8,
            "H_low": shannon_entropy(low),
            "H_high": shannon_entropy(high),
            "zero_frac": zero_frac,
            "sentinel_frac": sentinel_frac,
            "rle_ratio": rle_ratio,
            "compression_ratio": entropy_bits / raw_bits,
        }

    def to_image(self, repeat_period=5):
        snapshot = self.image_template.copy()

        snapshot[self.P2L == PAGE_EMPTY, ...] = PAGE_EMPTY_COLOR
        if self.open_block_index is not None:
            snapshot[self.open_block_index, ...] = OPEN_BLOCK_COLOR
        snapshot[self.P2L == PAGE_INVALID, ...] = PAGE_INVALID_COLOR
        snapshot[self.P2L >= 0, ...] = PAGE_LBA_COLOR

        assert ~np.any(np.isnan(snapshot)), "Color mapping issue"
        assert ~np.any(np.isnan(snapshot < 0))

        snapshot_path = os.path.join(FTL_SNAPSHOT_FOLDER, str(self.image_counter).zfill(repeat_period) + ".png")
        os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
        cv2.imwrite(snapshot_path, snapshot.astype(np.uint8))

        self.image_counter += 1
