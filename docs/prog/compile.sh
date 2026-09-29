#!/usr/bin/env bash
# Render prog.md as an A4 PDF. Run from any directory:
#   docs/prog/compile.sh

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
input="$script_dir/prog.md"
output="$script_dir/prog.pdf"
html="$(mktemp --suffix=.html)"
trap 'rm -f -- "$html"' EXIT

for command in npx google-chrome; do
    if ! command -v "$command" >/dev/null 2>&1; then
        printf 'Error: %s is required but was not found in PATH.\n' "$command" >&2
        exit 1
    fi
done

npx --yes marked --gfm -i "$input" | {
    cat <<'HTML'
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>prog</title>
  <style>
    @page { size: A4; margin: 18mm 16mm; }
    body { font-family: "Noto Sans CJK SC", "Noto Sans SC", "Microsoft YaHei", sans-serif; color: #1f2937; font-size: 10.5pt; line-height: 1.55; }
    h1, h2, h3 { color: #111827; line-height: 1.25; margin: 1.25em 0 .45em; break-after: avoid; }
    h1 { font-size: 22pt; }
    h2 { font-size: 16pt; border-bottom: 1px solid #d1d5db; padding-bottom: .2em; }
    h3 { font-size: 13pt; }
    p { margin: .6em 0; }
    table { border-collapse: collapse; width: 100%; margin: .8em 0; font-size: 9pt; break-inside: avoid; }
    th, td { border: 1px solid #9ca3af; padding: 5px 7px; vertical-align: top; }
    th { background: #f3f4f6; }
    code { font-family: monospace; font-size: 8.8pt; background: #f3f4f6; padding: 1px 3px; border-radius: 3px; }
    pre { white-space: pre-wrap; break-inside: avoid; }
    blockquote { border-left: 3px solid #9ca3af; margin-left: 0; padding-left: 12px; color: #4b5563; }
    a { color: #1d4ed8; text-decoration: none; }
    img { max-width: 100%; }
  </style>
</head>
<body>
HTML
    cat
    printf '%s\n' '</body></html>'
} > "$html"

google-chrome \
    --headless \
    --no-sandbox \
    --disable-gpu \
    --no-pdf-header-footer \
    --print-to-pdf="$output" \
    "file://$html" >/dev/null 2>&1

printf 'Wrote %s\n' "$output"
