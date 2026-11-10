# Local AI Tool Functions

Fonky's [`loco_tools.py`](../../loco_tools.py) exposes **29 provider-independent local AI tool functions** backed by the standalone [`loco.py`](../../loco.py) classes. These operations execute locally without an AI provider API key. Register only the tools that a host application needs.

For setup, provider registration, host permissions, and examples, see the [Local Automation guide](../user-guide/Local-Automation.md).

## Tool index

| Group | Functions | Purpose |
| --- | --- | --- |
| Calculator | `local_calculate`, `local_percentage` | Evaluate arithmetic expressions and calculate percentages. |
| Files | `local_file_list`, `local_file_read`, `local_file_exists`, `local_file_info`, `local_file_hash` | List and inspect approved workspace files. |
| SQLite | `local_sqlite_tables`, `local_sqlite_schema`, `local_sqlite_query` | Discover tables and schemas and run read-only queries. |
| Excel | `local_excel_sheets`, `local_excel_cell`, `local_excel_range`, `local_excel_write_range` | Read workbook data and perform permitted range writes. |
| Word | `local_word_read`, `local_word_create`, `local_word_append`, `local_word_replace` | Read, create, append to, and replace text in documents. |
| PowerPoint | `local_slides_create`, `local_slides_add_slide`, `local_slides_add_text` | Create presentations, add slides, and add text. |
| Outlook | `local_outlook_messages`, `local_outlook_search`, `local_outlook_read`, `local_outlook_draft` | Inspect messages and create drafts using classic Outlook. |
| Windows | `local_windows_list`, `local_windows_controls` | Inspect windows and their controls. |
| System | `local_system_info`, `local_disk_usage` | Report local system information and disk usage. |

## Usage

```python
from fonky.loco_tools import local_calculate, local_percentage

result = local_calculate('(12 + 8) / 4')
percentage = local_percentage(200, 15)
```

File-based operations require an approved workspace configured through `FONKY_LOCAL_ROOT`. Selected document-writing, Outlook, and desktop-inspection operations also require explicit host-side opt-in. Tool registration alone does not authorize access to the workstation.

## API reference

- [Complete function signatures and docstrings (MkDocs)](loco-tools-generated.md)
- [Python implementation on GitHub](../../loco_tools.py)
- [Local automation and provider integration guide](../user-guide/Local-Automation.md)

The generated reference is rendered by MkDocstrings on the MkDocs documentation site. GitHub's Markdown preview does not execute MkDocstrings directives.
