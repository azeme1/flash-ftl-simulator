# Flash FTL Simulator

A page-mapped flash translation layer (FTL) simulator for studying how garbage collection (GC) policies affect
**write amplification**, **L2P mapping-table entropy** (how compressible the table is) and **block wear**.
It can render the full state of the flash array into a video, one frame per page write.

## Quick start

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python main.py
```

`main.py` fills the drive sequentially, then overwrites it randomly `RANDOM_DRIVE_WRITES` times, logging metrics
as it goes and writing a video to `__dev/ftl.mp4`.

## Simulated device

The default geometry is `SLC_SMALL` in `ftl/parameters.py`:

| Parameter | Value |
|---|---|
| Blocks | 1024 |
| Pages per block | 64 |
| Page size | 4 KB (one page = one host LBA) |
| Over-provisioning | 7% (957 blocks of user space, 67 spare) |
| Physical address | 16 bits: 10 block bits + 6 page bits |

The FTL keeps a full page-level map in both directions:

- **L2P** maps each logical page (LBA) to its physical address, or `-1` if it was never written.
- **P2L** stores, for each physical page, the LBA it holds, `-1` for an invalid (stale) page or `-2` for an empty one.

Host writes go to one open block, page by page. Overwriting an LBA marks its old physical page invalid.

## Garbage collection

GC starts when fewer than `monitor_blocks` blocks are empty (default 16) and each run fills `clean_blocks`
destination blocks (default 15). Both are `FTL` parameters, and `clean_blocks` must be strictly smaller than
`monitor_blocks`, so GC always starts with at least as many empty blocks as it needs. Each run:

1. **Sizing:** relocates `clean_blocks` blocks' worth of valid pages.
2. **Victim selection (greedy):** sorts closed blocks by valid page count, fewest first; blocks with equal
   counts are taken in block index order.
3. **Collection:** takes valid pages from those victims until it has filled the planned destination blocks'
   worth. The last victim may be only partly collected. Victims with no valid pages are erased straight away.
4. **Ordering:** arranges the collected pages using the configured **GC order** (below).
5. **Relocation and erase:** writes the pages into freshly opened GC blocks and updates L2P and P2L. Each victim is
   erased as soon as its last valid page has moved, so the block can be reused immediately, even as a later
   destination in the same run.

### GC policies

A GC policy decides two things: which victims GC takes, and the order in which it writes their valid pages, which
decides which data ends up together in the new blocks. That affects later write amplification and how sequential the
L2P table stays. Policies are classes in `ftl/gc_policies.py`; each inherits from `GreedyGC` and overrides only
the decision it changes. Select one with `FTL(config, gc=...)`, or `GC` in `main.py`:

| `gc` | Class | Victims | GC writes the collected pages in order of |
|---|---|---|---|
| `greedy` | `GreedyGC` | fewest valid pages | baseline, collection order: victim blocks (fewest valid pages first), pages in physical order |
| `lba` | `LbaGC` | fewest valid pages | logical address, which keeps neighbouring LBAs together |
| `address` | `AddressGC` | fewest valid pages | current physical address |
| `block` | `BlockGC` | fewest valid pages | source block index |
| `page` | `PageGC` | fewest valid pages | page index inside the source block, which interleaves pages from all victims |
| `locality` *(default)* | `LocalityGC` | fewest valid pages; ties at the cut-off go to blocks whose LBAs neighbour the other victims' LBAs | logical address |

A new policy is one more subclass, registered in `GC_POLICIES`.

Sorting is stable, so pages with equal keys keep their collection order. Because pages are collected in ascending
page order within each block, `block` currently produces the same order as `address`.

Results for a sequential fill plus 3 random drive writes (1–16 page chunks, `np.random.seed(7)`):

| `gc` | Write amplification | L2P entropy (bits/entry) | Erases per block, mean / std |
|---|---|---|---|
| `lba` | **4.498** | **4.16** | 15.88 / 3.22 |
| `greedy` | 4.584 | 4.54 | 16.20 / 3.47 |
| `address` | 4.588 | 4.52 | 16.21 / 3.46 |
| `block` | 4.588 | 4.52 | 16.21 / 3.46 |
| `page` | 5.971 | 13.07 | 21.40 / 2.72 |

## Workloads

Workloads are classes in `workload/core.py`. Iterating a workload yields `(start_lba, chunk_size)` writes, measured in pages; arguments are checked when it is created, and random values are drawn while iterating, so `np.random.seed` makes runs repeatable. `RandomChunkWorkload` is the shared base of `FullDisk` and `RandomWrite`:

| Workload | Description |
|---|---|
| `FullDisk(chunk_min, chunk_max, page_count)` | Writes every LBA once, sequentially, in random-sized chunks |
| `RandomWrite(chunk_min, chunk_max, page_count, chunk_count)` | Random-sized chunks at uniformly random positions |
| `Jesd219Enterprise(page_count, chunk_count, page_size=4096)` | JEDEC JESD219 enterprise endurance workload (below) |
| `Jesd219Client(trace, page_count, page_size=4096)` | Replays `(byte_offset, byte_length)` writes from a JEDEC client trace |

### JESD219 enterprise workload

Random writes aligned to 4 KB, using the JESD219 transfer-size mix, rounded up to whole pages:

| Transfer size | Share of writes |
|---|---|
| 512 B – 3.5 KB (7 sizes) | 10% |
| 4 KB | 67% |
| 8 KB | 10% |
| 16 KB | 7% |
| 32 KB | 3% |
| 64 KB | 3% |

Writes are concentrated at the start of the LBA range: 50% go to the first 5% of LBAs, 30% to the next 15%, and
20% to the remaining 80%.

The parameters follow fio's [`jesd219.fio`](https://github.com/axboe/fio/blob/master/examples/jesd219.fio)
reference job. The standard itself is available from [JEDEC](https://www.jedec.org/node/8295).

### JESD219 client workload

The client workload is defined by JEDEC's trace files (JESD219A_MT master trace and JESD219A_TT test trace,
free from jedec.org after logging in). `Jesd219Client` maps trace records onto pages. There is no parser for the
trace file format yet.

## Metrics

| Metric | Meaning |
|---|---|
| Write amplification | NAND page writes (host + GC) ÷ host page writes |
| L2P entropy | Order-0 Shannon entropy of the differences between consecutive L2P entries, in bits per entry (0 = perfectly sequential, 16 = random). A lower value means a more compressible mapping table. |
| Utilization | Share of LBAs that are mapped |
| Erase counts | Min, mean, max and std of erases per block |

`FTL.check()` verifies the mapping invariants: L2P and P2L agree in both directions, and the per-block valid
page counters are correct. Each check can be turned off, e.g. `check(l2p=False)`.

## Video

`visualization/video.py` renders every page write (host and GC) into a 1280×720 H.264 video. Each frame shows:

- the flash array as a grid of blocks: valid, invalid and empty pages, with the host open block, the GC open
  block and the GC source blocks outlined;
- the current phase, with a GC indicator while a collection is running;
- live values for host and NAND writes, write amplification, L2P entropy, utilization and free blocks;
- charts of write amplification and L2P entropy over time, and a histogram of erase counts.

Settings in `main.py`:

| Setting | Effect |
|---|---|
| `VIDEO_PATH` | Output file; `None` disables video writing |
| `VIDEO_FPS` | Frame rate (default 60) |
| `VIDEO_FRAME_EVERY` | Capture one frame every N page writes |

With one frame per page write, the default scenario produces about 2.6 million frames, roughly a 12-hour video.
Use `VIDEO_FRAME_EVERY = 16` or higher for something watchable.

## Tests

```sh
.venv/bin/pip install pytest
.venv/bin/python -m pytest -q
```

| File | Covers |
|---|---|
| `tests/test_workloads.py` | Workloads: LBA coverage, bounds, JESD219 size and zone distributions, page-size mapping, argument validation |
| `tests/test_ftl.py` | GC correctness for every GC order, a write-amplification regression value, reclaiming fully invalid blocks, GC with very few empty blocks, `erase_block_if_possible`, and the invariant checks |

The full suite takes about 30 seconds; `test_gc_with_few_empty_blocks_does_not_crash` accounts for about half.

## Project layout

```
ftl/core.py         FTL core: mapping tables, host writes, GC, metrics, invariant checks
ftl/gc_policies.py  GC policies: victim selection and page order
ftl/parameters.py   Device geometry and physical address encoding
workload/core.py    Workload classes
main.py             Simulation scenario
visualization/      Video renderer
tests/              Workload and FTL tests
```

## License

[PolyForm Noncommercial 1.0.0](LICENSE): free for research and other noncommercial use. Commercial use requires
a separate license; contact azemel@gmail.com.
