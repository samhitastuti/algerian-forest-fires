"""Patch build_notebook.py to fix the data loading cell."""
import re

with open('build_notebook.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the position of the two problematic cells and replace them
# Find start of first load cell
start_marker = '    # ── Load & Parse Raw CSV ─'
end_marker_after = 'df.head(3)\n""")]'

start_idx = content.find(start_marker)
if start_idx == -1:
    print("ERROR: start marker not found")
    exit(1)

# Go back to find the cells += [code(""" opening
open_marker = 'cells += [code("""'
cell_open = content.rfind(open_marker, 0, start_idx)
print(f"Found cell open at char {cell_open}")

# Find the closing of the SECOND cell (df.head(3))
end_idx = content.find(end_marker_after, start_idx)
if end_idx == -1:
    print("ERROR: end marker not found")
    print("Searching for alternative...")
    end_marker_after = 'df.head(3)\n\"\"\")]'
    end_idx = content.find(end_marker_after, start_idx)

end_idx += len(end_marker_after)
print(f"Found end at char {end_idx}")

new_cell = '''cells += [code("""
    # ── Load & Parse Raw CSV ─────────────────────────────────────────────────
    # The UCI CSV has the structure:
    #   Line 0: 'Bejaia Region Dataset'  (title)
    #   Line 1: column headers
    #   Lines 2-123: Bejaia data (122 rows)
    #   Line 124: blank
    #   Line 125: 'Sidi-Bel Abbes Region Dataset' (title)
    #   Line 126: column headers
    #   Lines 127-248: Sidi data (122 rows)

    import os, glob, io

    csv_files = glob.glob("data/*.csv")
    if not csv_files:
        raise FileNotFoundError(
            "No CSV found in data/.  Run: python fetch_dataset.py")

    raw_path = csv_files[0]
    print(f"Loading: {raw_path}")

    col_names = ['day','month','year','Temperature','RH','Ws','Rain',
                 'FFMC','DMC','DC','ISI','BUI','FWI','Classes']

    with open(raw_path, 'r', encoding='utf-8', errors='replace') as fh:
        all_lines = fh.readlines()
    print(f"Total raw lines in file: {len(all_lines)}")

    def is_data_row(line):
        return line.count(',') >= 13   # 14 columns => 13 commas minimum

    bejaia_rows, sidi_rows = [], []
    current_region = None
    for line in all_lines:
        s = line.strip()
        if not s:
            continue
        if 'bejaia' in s.lower():
            current_region = 'Bejaia'; continue
        if 'sidi' in s.lower():
            current_region = 'Sidi-Bel-Abbes'; continue
        if s.lower().startswith('day,'):
            continue      # skip repeated header rows
        if not is_data_row(s):
            continue      # skip any other non-data rows
        if current_region == 'Bejaia':
            bejaia_rows.append(s)
        elif current_region == 'Sidi-Bel-Abbes':
            sidi_rows.append(s)

    def rows_to_df(rows, region):
        text = "\\n".join(rows)
        d = pd.read_csv(io.StringIO(text), header=None,
                        names=col_names, skipinitialspace=True)
        d['Region'] = region
        return d

    bejaia_df = rows_to_df(bejaia_rows, 'Bejaia')
    sidi_df   = rows_to_df(sidi_rows,   'Sidi-Bel-Abbes')
    df = pd.concat([bejaia_df, sidi_df], ignore_index=True)

    print(f"Bejaia rows: {len(bejaia_df)}  |  Sidi rows: {len(sidi_df)}")
    print(f"Combined shape: {df.shape}")
    df.head(5)
""")]'''

content = content[:cell_open] + new_cell + content[end_idx:]

with open('build_notebook.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Patch applied successfully!")
print(f"New file size: {len(content)} chars")
