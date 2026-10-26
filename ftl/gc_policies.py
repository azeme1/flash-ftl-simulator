"""GC policies: which victims GC takes and the order in which it writes their valid pages.

Every policy inherits from GreedyGC and overrides only the decision it changes."""

import numpy as np

from ftl.parameters import decode_address


class GreedyGC:
    """Baseline: the clean_blocks candidates with the fewest valid pages (ties by block index); pages copied
    as collected: victim by victim (fewest valid pages first), pages in physical order."""
    name = "greedy"

    def select_victims(self, ftl, candidates, clean_blocks):
        """Victims among the candidates, which are sorted by valid page count, fewest first."""
        return candidates[:clean_blocks]

    def page_order(self, ftl, lba_items):
        """Permutation of the collected pages giving the order in which they are written."""
        return np.arange(len(lba_items))


class SortedGC(GreedyGC):
    """Greedy victims; pages written sorted by a key (stable, so ties keep collection order)."""

    def sort_keys(self, ftl, lba_items):
        raise NotImplementedError

    def page_order(self, ftl, lba_items):
        return np.argsort(self.sort_keys(ftl, lba_items), kind="stable")


class LbaGC(SortedGC):
    """Pages written in logical address order."""
    name = "lba"

    def sort_keys(self, ftl, lba_items):
        return np.asarray(lba_items)


class AddressGC(SortedGC):
    """Pages written in current physical address order."""
    name = "address"

    def sort_keys(self, ftl, lba_items):
        return ftl.L2P[lba_items]


class BlockGC(SortedGC):
    """Pages written in source block index order."""
    name = "block"

    def sort_keys(self, ftl, lba_items):
        return decode_address(ftl.L2P[lba_items])[0]


class PageGC(SortedGC):
    """Pages written in order of their page index inside the source block (interleaves all victims)."""
    name = "page"

    def sort_keys(self, ftl, lba_items):
        return decode_address(ftl.L2P[lba_items])[1]


class LocalityGC(LbaGC):
    """LBA page order; candidates tied with the cut-off's valid count are ranked by how many of their LBAs
    neighbour (LBA +-1) the LBAs of the victims below the cut-off."""
    name = "locality"

    def select_victims(self, ftl, candidates, clean_blocks):
        valid = ftl.block_valid_pages[candidates]
        cutoff = valid[min(clean_blocks, len(candidates)) - 1]
        sure, tied = candidates[valid < cutoff], candidates[valid == cutoff]
        content = ftl.P2L[sure]
        chosen = set(content[content >= 0].tolist())
        scores = [sum((x - 1 in chosen) + (x + 1 in chosen) for x in ftl.P2L[b][ftl.P2L[b] >= 0].tolist())
                  for b in tied]
        best = tied[np.argsort(-np.array(scores), kind="stable")][:clean_blocks - len(sure)]
        return np.concatenate([sure, best])


GC_POLICIES = {cls.name: cls for cls in (GreedyGC, LbaGC, AddressGC, BlockGC, PageGC, LocalityGC)}
