---
name: doc-convert
description: "Convert local PDF/DOCX documents to text for analysis"
metadata: {"aeloon": {"emoji": "📄", "requires": {"bins": ["python3"]}}}
---

# Document Conversion

When a user references local documents or directories (e.g., "refer to ~/papers/report.pdf",
"according to these files", "use the documents in ~/data/"),
convert the referenced documents to text format, then read and analyze them.

## Document Detection

Recognize when the user mentions:
- File paths (e.g., `~/papers/report.pdf`, `/home/user/docs/notes.docx`)
- Directories (e.g., `~/papers/`, `./docs/`)
- "these documents" / "these files" / "use these papers" / "refer to" / "according to"

## Conversion Workflow

### Step 1: Convert PDF to text

```bash
exec(command="python3 -c \"
import fitz, sys, os, pathlib
src = '{source_path}'
out_dir = os.path.expanduser('~/.aeloon/converted_docs')
os.makedirs(out_dir, exist_ok=True)
stem = pathlib.Path(src).stem
out_path = os.path.join(out_dir, stem + '.txt')
doc = fitz.open(src)
text = ''
for page in doc:
    text += page.get_text() + '\n\n'
doc.close()
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(text.strip())
print(f'Converted: {out_path}')
print(f'Pages: {doc.page_count}, Characters: {len(text)}')
\"")
```

### Step 2: Convert DOCX to text

```bash
exec(command="python3 -c \"
import docx, os, pathlib
src = '{source_path}'
out_dir = os.path.expanduser('~/.aeloon/converted_docs')
os.makedirs(out_dir, exist_ok=True)
stem = pathlib.Path(src).stem
out_path = os.path.join(out_dir, stem + '.txt')
doc = docx.Document(src)
lines = []
for para in doc.paragraphs:
    if para.style.name.startswith('Heading'):
        level = int(para.style.name.split()[-1]) if para.style.name[-1].isdigit() else 1
        lines.append('#' * level + ' ' + para.text)
    else:
        lines.append(para.text)
text = '\n\n'.join(lines)
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(text.strip())
print(f'Converted: {out_path}')
\"")
```

### Step 3: Read converted text

```bash
read_file(path="{converted_path}")
```

### Step 4: Process directory of documents

When the user specifies a directory:

```bash
exec(command="python3 -c \"
import fitz, docx, os, pathlib, glob
src_dir = '{directory_path}'
out_dir = os.path.expanduser('~/.aeloon/converted_docs')
os.makedirs(out_dir, exist_ok=True)
supported = {'.pdf', '.docx', '.md', '.txt', '.csv'}
converted = 0
skipped = 0
for f in sorted(pathlib.Path(src_dir).rglob('*')):
    if f.suffix.lower() not in supported:
        continue
    out_path = pathlib.Path(out_dir) / (f.stem + '.txt')
    if f.suffix.lower() == '.pdf':
        doc = fitz.open(str(f))
        text = '\\n\\n'.join(p.get_text() for p in doc)
        doc.close()
    elif f.suffix.lower() == '.docx':
        doc = docx.Document(str(f))
        text = '\\n\\n'.join(p.text for p in doc.paragraphs)
    else:
        text = f.read_text(encoding='utf-8', errors='replace')
    out_path.write_text(text.strip(), encoding='utf-8')
    converted += 1
    print(f'OK: {f.name} -> {out_path.name}')
print(f'\\nTotal: {converted} converted, {skipped} skipped')
\"")
```

## Output Directory

- Converted text files are saved to `~/.aeloon/converted_docs/`
- Files are named `<original_stem>.txt` (e.g., `report.txt` from `report.pdf`)
- **Never auto-delete** converted files — they serve as a local cache for future reference
- If the directory grows too large, the user may manually clean old files

## Error Handling

If conversion fails:
1. Report which file failed: `"Failed to convert {filename}: {error}"`
2. Suggest installing dependencies: `pip install PyMuPDF pdfminer.six python-docx`
3. Continue with available text-based documents (`.md`, `.txt`, `.csv`)

## Important Notes
- Only convert referenced documents — do NOT auto-scan entire file systems
- If a file is already converted (exists in `~/.aeloon/converted_docs/`), skip conversion and read the cached version
- Large PDFs may take time — inform the user about processing time
- Always use `exec` tool for file operations, never call `read_file` on binary files
- For `.md` / `.txt` / `.csv` files, use `read_file` directly — no conversion needed
