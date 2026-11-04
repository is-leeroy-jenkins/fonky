# Local Automation (`loco.py`)

Fonky's `fonky.loco` module provides nine local-operation classes without cloud AI API keys.
The Python classes do not register themselves with an AI provider or authorize an agent to execute actions.

## Classes

| Class | Backend | Primary capabilities |
| --- | --- | --- |
| `File` | pathlib/shutil/zipfile | Path-limited read, write, list, move, copy, delete, hash, archive |
| `Data` | sqlite3, pandas | Parameterized queries, transactions, schema discovery, backups |
| `Calculator` | ast/math | Arithmetic, percentages, safe expression parsing |
| `Shell` | subprocess | Executable allowlist, timeouts, PowerShell, system info |
| `Excel` | openpyxl or COM | Cells, ranges, DataFrames, formulas, macros, queries, PDF |
| `Word` | python-docx or COM | Paragraphs, replace, tables, images, PDF |
| `PowerPoint` | python-pptx or COM | Slides, text boxes, images, PDF |
| `Outlook` | classic Outlook COM | Inbox, messages, drafts, approved send, appointments |
| `Windows` | pywinauto | Window discovery, controls, screenshots |

## Installation

Standard Fonky includes file-oriented Office libraries. Use optional extras for Windows desktop automation:

```powershell
python -m pip install 'fonky[office]'
python -m pip install 'fonky[desktop]'
python -m pip install 'fonky[local]'
```

These extras are available from PyPI only after this change has been published. For repository development, install an editable checkout with `python -m pip install -e '.[local]'`.

## Examples

```python
from fonky.loco import Calculator, Data, Excel, File, Shell

files = File(root='C:/Work')
files.write_text('notes.txt', 'Local automation')

print(Calculator().evaluate_expression('(12 + 8) / 4'))

database = Data('C:/Work/sample.db')
database.execute_nonquery('CREATE TABLE IF NOT EXISTS items(name TEXT)')
database.execute_nonquery('INSERT INTO items(name) VALUES (?)', ('demo',))
database.commit()
print(database.execute_query('SELECT name FROM items'))
database.close()

excel = Excel(backend='file')
excel.create_workbook()
excel.write_cell('Sheet', 'A1', 'Budget')
excel.save_workbook('C:/Work/example.xlsx')
excel.close_workbook()

shell = Shell(allowed_commands=['python.exe'])
```

Use `Excel(backend='com')`, `Word(backend='com')`, or `PowerPoint(backend='com')` for live Microsoft Office. Outlook automation requires classic Outlook for Windows, not the new Outlook.

## Operational boundaries

- `Shell` denies execution until the caller supplies an explicit executable allowlist. The allowlist is not an OS-level sandbox.
- `File(root=...)` enforces an allowed directory after path resolution. Agents should not receive destructive tools without additional approval.
- `Data.execute_query` permits read-only SELECT statements and SQLite integrity checks. Write operations require explicit transaction management.
- `Outlook.create_draft` only saves a draft. `Outlook.send_message` requires `approved=True`.
- `Excel.run_macro` requires `approved=True` and the COM backend.
- COM automation starts dedicated application instances and requires an installed Office application and an interactive Windows desktop.
- Python tools remain available independently of hosted AI providers. An external model needs explicitly registered, approved tool wrappers.
- The file-backed Office libraries cannot refresh Excel queries or reliably preserve every document formatting edge case.

## Testing

```powershell
python -m unittest discover -s tests -p test_loco.py -v
```

CI runs Python 3.11 and 3.13 on Ubuntu and Windows. Live Office COM and Outlook profile testing requires a workstation with an Office installation.
