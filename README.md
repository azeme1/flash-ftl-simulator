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

The default geometry is `SLC_SMALL` in `parameters.py`:

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

GC starts when fewer than 16 blocks are empty. Each run:

1. **Victim selection (greedy):** sorts closed blocks by valid page count, fewest first.
2. **Collection:** takes valid pages from those victims until it has 15 blocks' worth (960 pages). The last victim
   may be only partly collected.
3. **Ordering:** arranges the collected pages using the configured **GC order** (below).
4. **Relocation:** writes them into 15 freshly opened GC blocks and updates L2P and P2L.
5. **Erase:** erases every victim that no longer holds valid pages.

### GC page orders

The order in which GC writes the collected pages decides which data ends up together in the new blocks. That
affects later write amplification and how sequential the L2P table stays. Select it with
`FTL(config, gc_order=...)`, or `GC_ORDER` in `main.py`:

| `gc_order` | GC writes the collected pages in order of |
|---|---|
| `source` | collection order: victim blocks (fewest valid pages first), pages in physical order |
| `lba` *(default)* | logical address, which keeps neighbouring LBAs together |
| `address` | current physical address |
| `block` | source block index |
| `page` | page index inside the source block, which interleaves pages from all victims |

Sorting is stable, so pages with equal keys keep their collection order. Because pages are collected in ascending
page order within each block, `block` currently produces the same order as `address`.

Results for a sequential fill plus 3 random drive writes (1–16 page chunks, `np.random.seed(7)`):

| `gc_order` | Write amplification | L2P entropy (bits/entry) | Erases per block, mean / std |
|---|---|---|---|
| `lba` | **4.490** | **4.13** | 15.85 / 2.82 |
| `source` | 4.568 | 4.52 | 16.14 / 2.67 |
| `address` | 4.588 | 4.53 | 16.21 / 2.75 |
| `block` | 4.588 | 4.53 | 16.21 / 2.75 |
| `page` | 5.956 | 13.03 | 21.34 / 2.64 |

## Workloads

`workloads.py` contains generators that yield `(start_lba, chunk_size)` writes, measured in pages:

| Workload | Description |
|---|---|
| `full_disk(chunk_min, chunk_max, page_count)` | Writes every LBA once, sequentially, in random-sized chunks |
| `random_write(chunk_min, chunk_max, page_count, chunk_count)` | Random-sized chunks at uniformly random positions |
| `jesd219_enterprise(page_count, chunk_count, page_size=4096)` | JEDEC JESD219 enterprise endurance workload (below) |
| `jesd219_client(trace, page_count, page_size=4096)` | Replays `(byte_offset, byte_length)` writes from a JEDEC client trace |

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
free from jedec.org after logging in). `jesd219_client` maps trace records onto pages. There is no parser for the
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

## Project layout

```
ftl.py              FTL core: mapping tables, host writes, GC, metrics, invariant checks
parameters.py       Device geometry and physical address encoding
workloads.py        Workload generators
main.py             Simulation scenario
visualization/      Video renderer
tests/              Workload tests
```

## License

[PolyForm Noncommercial 1.0.0](LICENSE): free for research and other noncommercial use. Commercial use requires
a separate license; contact azemel@gmail.com.
