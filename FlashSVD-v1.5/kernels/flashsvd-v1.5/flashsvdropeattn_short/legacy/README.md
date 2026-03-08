## legacy

This subfolder holds older exploratory scripts that are no longer the primary short-context decode entrypoints.

They were moved out of the top level to keep the active workflow focused on:

- `decode_compare.py`
- `bench_decode_stack_compare.py`
- `flashsvdropeattn_dense_decode.py`
- `flashsvdropeattn_v1.5_decode.py`
- `flashsvdropeattn_v1.6_decode_opt.py`

Some legacy scripts still assume the old top-level layout and may need path fixes if you want to run them again.
