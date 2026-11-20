# Inter font (vendored)

Provenance and reproduction steps for the vendored `Inter-4.1-latin-var.woff2`, the admin console's font. This file is not deployed (`SITE_EXCLUDE` in `infra/owlbank_infra/admin_stack.py` excludes `*.md`).

## Source

- Release: Inter v4.1, <https://github.com/rsms/inter/releases/tag/v4.1> (published 2024-11-16).
- Asset: `https://github.com/rsms/inter/releases/download/v4.1/Inter-4.1.zip`
  - size 33,707,794 bytes
  - SHA-256 `9883fdd4a49d4fb66bd8177ba6625ef9a64aa45899767dde3d36aa425756b11e`
- Input file from the zip: `InterVariable.ttf` (879,708 bytes, SHA-256 `4989b125924991b90d05b2d16e0e388c48f7d5bb8b30539bbf9c755278d0ccaf`).
- Licence: the zip's `LICENSE.txt`, copied unchanged as `OFL.txt` (SHA-256 `262481e844521b326f5ecd053e59b98c8b2da78c8ee1bdbb6e8174305e54935a`). SIL Open Font License 1.1. The copyright line names no Reserved Font Name, so the subset can keep the name "Inter".

## Output

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `Inter-4.1-latin-var.woff2` | 28,236 | `15245ca859666c5d28b938a24ffc95aa225ca4fb51af81ebff892fee5185d33c` |

- Variable `wght` axis 400–700 (default 400); `opsz` pinned at 14 and removed.
- 216 code points, 369 glyphs. OpenType features kept: `calt`, `dnom`, `frac`, `kern`, `locl`, `numr`, `tnum`.

## Tooling

- Versions used to build the shipped file: fonttools 4.67.0 with the `woff` extra (brotli), run one-off through `uvx` (uv 0.12.23). Not a project dependency.

## Commands

Run from an empty working directory outside the extracted zip, in one shell (the later commands use `$FT` and `$U`). `SOURCE_DATE_EPOCH` fixes the `head.modified` timestamp so the output is byte-for-byte reproducible; without it the SHA-256 changes on every run.

```sh
FT='fonttools[woff]==4.67.0'
U='U+0000-00FF,U+0131,U+0152-0153,U+02C6,U+02DA,U+02DC,U+2013-2014,U+2018-201A,U+201C-201E,U+2020-2022,U+2026,U+2030,U+2039-203A,U+2044,U+20AC,U+2122,U+2212'
export SOURCE_DATE_EPOCH=1731628800   # 2024-11-15, the zip's file date

curl -sSL -o Inter-4.1.zip https://github.com/rsms/inter/releases/download/v4.1/Inter-4.1.zip
shasum -a 256 Inter-4.1.zip           # must match the SHA-256 above
mkdir x && unzip -q Inter-4.1.zip InterVariable.ttf LICENSE.txt -d x

uvx --from "$FT" fonttools varLib.instancer x/InterVariable.ttf wght=400:700 opsz=14 -o inter-inst.ttf
uvx --from "$FT" pyftsubset inter-inst.ttf --unicodes="$U" --layout-features+=tnum \
  --flavor=woff2 --output-file=Inter-4.1-latin-var.woff2
shasum -a 256 Inter-4.1-latin-var.woff2   # must match the table above
```

`pyftsubset` keeps its default layout features (including `kern`); `+=tnum` adds tabular figures.

## unicode-range (for the `@font-face` rule)

```css
unicode-range: U+0000-00FF, U+0131, U+0152-0153, U+02C6, U+02DA, U+02DC, U+2013-2014, U+2018-201A, U+201C-201E, U+2020-2022, U+2026, U+2030, U+2039-203A, U+2044, U+20AC, U+2122, U+2212;
```

Latin-1 (covers Spanish: á é í ó ú ñ ü ¿ ¡), plus ı, Œ œ, ˆ ˚ ˜, en/em dashes, curly quotes, † ‡ •, …, ‰, ‹ ›, ⁄, €, ™ and −. U+00AD (soft hyphen) is in the range but has no glyph in Inter; the browser falls back per glyph for it and for anything outside the range.

## Inspection check

Tables and axes, with the CLI only:

```sh
uvx --from "$FT" fonttools ttx -l Inter-4.1-latin-var.woff2
uvx --from "$FT" fonttools ttx -q -t fvar -o - Inter-4.1-latin-var.woff2 | grep -E 'AxisTag|MinValue|DefaultValue|MaxValue'
```

Expected: one axis, `wght` 400 / 400 / 700; no `opsz`.

cmap and features: save this as `check.py` in a directory of its own and run it with `python -I`:

```python
import sys
from fontTools.ttLib import TTFont
f = TTFont(sys.argv[1])
allowed = set()
for part in sys.argv[2].split(","):
    a, _, b = part.strip()[2:].partition("-")
    allowed.update(range(int(a, 16), int(b or a, 16) + 1))
cmap = set(f.getBestCmap())
print("axes", [(a.axisTag, a.minValue, a.defaultValue, a.maxValue) for a in f["fvar"].axes])
print("missing", [c for c in "áéíóúñü¿¡" if ord(c) not in cmap])
print("outside range", sorted(hex(c) for c in cmap - allowed))
print("features", sorted({r.FeatureTag for t in ("GSUB", "GPOS") for r in f[t].table.FeatureList.FeatureRecord}))
```

```sh
uv run --no-project --with "$FT" python -I check.py Inter-4.1-latin-var.woff2 "$U"
```

Expected: `axes [('wght', 400.0, 400.0, 700.0)]`, `missing []`, `outside range []`, and `features` containing `kern` and `tnum`.
