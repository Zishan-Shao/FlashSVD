# FlashSVD-v1.5 Notes

Start here.

## Active Notes

- [`CURRENT_STATUS.md`](./CURRENT_STATUS.md)
  Current production path, verified experiments, and the headline numbers that
  should be quoted today.

- [`quick_commands.md`](./quick_commands.md)
  Short runbook for the benchmark and correctness commands we still use.

- [`ARCHITECTURE.md`](./ARCHITECTURE.md)
  Runtime ownership and folder boundaries.

## Local Results Archive

Older experiment logs, tables, and dated result snapshots were moved out of
`docs/` and into the local `results/docs_notes_archive/` folder.

That bucket now holds:

- historical experiment writeups
- dated comparison tables and JSON dumps
- older archive notes that are useful for local forensics only

Rule of thumb for this tracked `docs/notes/` folder:

- use `CURRENT_STATUS.md` for the answer to "what is the current winner?"
- use `quick_commands.md` for "what should I run now?"
- use `results/docs_notes_archive/` only for local historical context
