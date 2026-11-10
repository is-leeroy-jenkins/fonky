# Local AI Tool Functions — API Reference

This GitHub-readable reference documents the **29 public local AI tool operations** implemented in [`loco_tools.py`](../../loco_tools.py). It includes their Python signatures and purpose statements. For the operational setup, provider integration, and permission controls, consult the [Local Automation guide](../user-guide/Local-Automation.md) or the [tool index](loco-tools.md).

> This is a Markdown snapshot of the Python API. Check the linked source for subsequent signature changes.

## Calculator

### `local_calculate`

Evaluate a restricted arithmetic expression.

```python
local_calculate(expression: str) -> float
```

### `local_percentage`

Calculate a percentage of a numeric value.

```python
local_percentage(value: float, percent: float) -> float
```

## File operations

### `local_file_list`

List files inside the approved local workspace.

```python
local_file_list(directory: str = '.', pattern: str = '*', recursive: bool = False) -> list[str]
```

### `local_file_read`

Read a small text file from the approved local workspace.

```python
local_file_read(path: str) -> str
```

### `local_file_exists`

Check whether an approved workspace path exists.

```python
local_file_exists(path: str) -> bool
```

### `local_file_info`

Return JSON metadata for an approved workspace path.

```python
local_file_info(path: str) -> str
```

### `local_file_hash`

Calculate the SHA-256 digest of a local file.

```python
local_file_hash(path: str) -> str
```

## SQLite

### `local_sqlite_tables`

List tables in an existing SQLite database.

```python
local_sqlite_tables(path: str) -> list[str]
```

### `local_sqlite_schema`

Return JSON column metadata for an SQLite table.

```python
local_sqlite_schema(path: str, table: str) -> str
```

### `local_sqlite_query`

Execute a read-only SELECT query against an SQLite database.

```python
local_sqlite_query(path: str, sql: str, parameters_json: str = '[]', max_rows: int = 50) -> str
```

## Excel

### `local_excel_sheets`

List worksheets in an existing Excel workbook.

```python
local_excel_sheets(path: str) -> list[str]
```

### `local_excel_cell`

Read a single Excel cell as JSON.

```python
local_excel_cell(path: str, sheet: str, cell: str) -> str
```

### `local_excel_range`

Read a bounded Excel range as JSON.

```python
local_excel_range(path: str, sheet: str, address: str) -> str
```

### `local_excel_write_range`

Write an Excel range and save the existing workbook.

```python
local_excel_write_range(path: str, sheet: str, start_cell: str, rows: list[list[str | int | float | bool | None]]) -> str
```

## Word

### `local_word_read`

Read up to 100000 characters from an existing Word document.

```python
local_word_read(path: str) -> str
```

### `local_word_create`

Create a new Word document without overwriting another file.

```python
local_word_create(path: str, text: str) -> str
```

### `local_word_append`

Append text to an existing Word document and save it.

```python
local_word_append(path: str, text: str) -> str
```

### `local_word_replace`

Replace text in an existing Word document and save changes.

```python
local_word_replace(path: str, old: str, new: str) -> int
```

## PowerPoint

### `local_slides_create`

Create a new PowerPoint presentation without overwriting an existing file.

```python
local_slides_create(path: str) -> str
```

### `local_slides_add_slide`

Add a titled slide to an existing PowerPoint presentation.

```python
local_slides_add_slide(path: str, title: str) -> int
```

### `local_slides_add_text`

Add text to an existing PowerPoint slide and save.

```python
local_slides_add_text(path: str, slide_number: int, text: str) -> str
```

## Outlook

### `local_outlook_messages`

List recent messages from the local Outlook profile.

```python
local_outlook_messages(limit: int = 20) -> str
```

### `local_outlook_search`

Search recent Outlook messages for a phrase.

```python
local_outlook_search(phrase: str, limit: int = 50) -> str
```

### `local_outlook_read`

Read a message from the local Outlook profile.

```python
local_outlook_read(entry_id: str) -> str
```

### `local_outlook_draft`

Create an Outlook draft without sending it.

```python
local_outlook_draft(recipient: str, subject: str, body: str) -> str
```

## Windows desktop

### `local_windows_list`

List windows in the active local Windows desktop session.

```python
local_windows_list() -> list[str]
```

### `local_windows_controls`

List controls of a selected local Windows application window.

```python
local_windows_controls(title: str) -> list[str]
```

## System

### `local_system_info`

Get local operating-system information as JSON.

```python
local_system_info() -> str
```

### `local_disk_usage`

Read disk-capacity statistics for a permitted workspace path.

```python
local_disk_usage(path: str = '.') -> str
```

## Permissions and execution

- Set `FONKY_LOCAL_ROOT` to an approved directory before using file-backed operations.
- Use `FONKY_LOCAL_WRITE_ENABLED=1` for enabled document writes and Outlook draft creation.
- Use `FONKY_LOCAL_OUTLOOK_ENABLED=1` to enable local classic Outlook access.
- Use `FONKY_LOCAL_DESKTOP_ENABLED=1` to enable Windows desktop inspection.
- The host application must register only approved tools and enforce appropriate user permissions. These settings are not an operating-system sandbox.

For provider-specific tool registration examples and platform limitations, see the [Local Automation guide](../user-guide/Local-Automation.md).
