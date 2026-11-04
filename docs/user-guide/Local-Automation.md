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


## AI tool integrations (29 approved operations)

The nine classes in `fonky.loco` remain standalone and do not import an AI SDK. The
`fonky.loco_tools` module exposes **29 complete, one-call operations** over those classes.
Every provider module exports the same tool names without duplicating the execution logic.

| Group | Local tool functions |
| --- | --- |
| Calculator | `local_calculate`, `local_percentage` |
| Files | `local_file_list`, `local_file_read`, `local_file_exists`, `local_file_info`, `local_file_hash` |
| SQLite | `local_sqlite_tables`, `local_sqlite_schema`, `local_sqlite_query` |
| Excel | `local_excel_sheets`, `local_excel_cell`, `local_excel_range`, `local_excel_write_range` |
| Word | `local_word_read`, `local_word_create`, `local_word_append`, `local_word_replace` |
| PowerPoint | `local_slides_create`, `local_slides_add_slide`, `local_slides_add_text` |
| Outlook | `local_outlook_messages`, `local_outlook_search`, `local_outlook_read`, `local_outlook_draft` |
| Windows | `local_windows_list`, `local_windows_controls` |
| Shell | `local_system_info`, `local_disk_usage` |

These are **atomic operations**. For example, an Excel read tool opens the workbook,
retrieves the requested content, and closes the workbook within a single invocation.
File-backed Word, Excel, and PowerPoint operations use the existing `backend='file'`
implementation. Live Office COM sessions, executing shell commands, VBA macros,
sending email, and destructive file/database operations are **not** AI-exposed in
this first integration.

### Host-side configuration

Set an approved local workspace before registering file-based tools:

```powershell
$env:FONKY_LOCAL_ROOT = "C:\\Work\\Fonky"
```

The following optional settings must be defined by the **host application or operator**,
not supplied as model-visible tool arguments:

| Environment variable | Default | Effect when set to `1` |
| --- | --- | --- |
| `FONKY_LOCAL_WRITE_ENABLED` | Disabled | Permit the selected Office document writes and Outlook draft creation |
| `FONKY_LOCAL_OUTLOOK_ENABLED` | Disabled | Permit reading/searching the local classic Outlook profile |
| `FONKY_LOCAL_DESKTOP_ENABLED` | Disabled | Permit inspecting local Windows windows and controls |

Filesystem paths exposed to the model are relative to `FONKY_LOCAL_ROOT`, with
resolved-path containment enforced. Read-only SQLite connections are used for database tools.
Treat the environment-variable gates as host configuration, **not** as a substitute for
per-user approval, OS access controls, isolation, or workstation permissions.
In a multi-user service, use separate isolated executors rather than sharing one
process-level environment configuration across users.

### Provider registration

OpenAI Agents SDK:

```python
from agents import Agent
from fonky.gpt.tools import local_calculate, local_file_read

agent = Agent(
    name='Fonky Local',
    instructions='Use only the available approved local tools.',
    tools=[ local_calculate, local_file_read ],
)
```

Anthropic uses the corresponding `fonky.claude.tools` names decorated with `@beta_tool`.
Google ADK uses the plain Python functions exported by `fonky.gemini.tools`.
LangChain exports its local operations as `@tool(parse_docstring=True)` instances.

xAI and Mistral require provider-native function definitions and a client-side tool call
execution loop. Their modules export:
- `local_tool_definitions`: provider-native tool definitions for all 29 operations.
- `local_*` functions: matching local Python callables to execute when the model requests them.

Register **only the subset of tools needed for the application**. Importing a tool definition
does not grant a remote AI model access to a workstation: the tool executor must run on
the local machine with the necessary operating-system and Office permissions.

### Testing

```powershell
python -m unittest discover -s tests -p test_loco_tools.py -v
```

CI runs these tests alongside the existing `loco.py` regression suite under
Python 3.11 and 3.13 on Windows and Ubuntu. Live classic Outlook and desktop UI
interactions still require separate workstation-based integration testing.
