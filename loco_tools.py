"""Provider-neutral, self-contained local AI tool functions.

Purpose:
    Provides one-call functions over the standalone fonky.loco implementations.
    Every function runs locally; no AI SDK or cloud API key is needed.
    Paths are confined to FONKY_LOCAL_ROOT. Writes, Outlook, and desktop
    inspection require explicit host-side environment opt-in.
"""
from __future__ import annotations

import json
import os
from inspect import Parameter, signature
from pathlib import Path
from typing import Callable
from pydantic import TypeAdapter
from .loco import Calculator, Data, Excel, File, Outlook, PowerPoint, Shell, Windows, Word


def _files( ) -> File:
    """Return an instance restricted to the configured workspace."""
    root = os.environ.get( 'FONKY_LOCAL_ROOT', '' )
    if not root:
        raise PermissionError( 'FONKY_LOCAL_ROOT must identify an approved workspace.' )
    return File( root=root )


def _path( path: str ) -> str:
    """Resolve a relative path inside the trusted workspace."""
    if Path( path ).is_absolute( ):
        raise PermissionError( 'Only workspace-relative paths are allowed.' )
    return str( _files( ).resolve( path ) )


def _existing( path: str ) -> str:
    """Resolve an existing workspace file."""
    resolved = _path( path )
    if not Path( resolved ).is_file( ):
        raise FileNotFoundError( resolved )
    return resolved


def _new( path: str ) -> str:
    """Resolve a new workspace file without replacing an existing file."""
    resolved = _path( path )
    if Path( resolved ).exists( ):
        raise FileExistsError( resolved )
    return resolved


def _require_write( ) -> None:
    """Require host permission for document and mailbox changes."""
    if os.environ.get( 'FONKY_LOCAL_WRITE_ENABLED' ) != '1':
        raise PermissionError( 'Local writes require host opt-in.' )


def _require_outlook( ) -> None:
    """Require host permission to access the local Outlook profile."""
    if os.environ.get( 'FONKY_LOCAL_OUTLOOK_ENABLED' ) != '1':
        raise PermissionError( 'Outlook access requires host opt-in.' )


def _require_desktop( ) -> None:
    """Require host permission to inspect desktop windows."""
    if os.environ.get( 'FONKY_LOCAL_DESKTOP_ENABLED' ) != '1':
        raise PermissionError( 'Desktop inspection requires host opt-in.' )


def local_calculate( expression: str ) -> float:
    """Evaluate a restricted arithmetic expression.

    Purpose:
        Evaluate a restricted arithmetic expression.

    Args:
        expression: A restricted arithmetic expression.

    Returns:
        float: Result of the local operation.
    """
    return Calculator( ).evaluate_expression( expression )


def local_percentage( value: float, percent: float ) -> float:
    """Calculate a percentage of a numeric value.

    Purpose:
        Calculate a percentage of a numeric value.

    Args:
        value: Base numeric value.
        percent: Percentage of the base value.

    Returns:
        float: Result of the local operation.
    """
    return Calculator( ).percentage( value, percent )


def local_file_list( directory: str = '.', pattern: str = '*', recursive: bool = False ) -> list[str]:
    """List files inside the approved local workspace.

    Purpose:
        List files inside the approved local workspace.

    Args:
        directory: Directory within the approved workspace.
        pattern: File-matching glob pattern.
        recursive: Whether to search subdirectories.

    Returns:
        list[str]: Result of the local operation.
    """
    return _files( ).list_files( directory, pattern, recursive )


def local_file_read( path: str ) -> str:
    """Read a small text file from the approved local workspace.

    Purpose:
        Read a small text file from the approved local workspace.

    Args:
        path: Path relative to the host-approved workspace.

    Returns:
        str: Result of the local operation.
    """
    files = _files( )
    resolved = _path( path )
    if Path( resolved ).stat( ).st_size > 1048576:
        raise ValueError( 'Text file exceeds 1 MiB.' )
    return files.read_text( path )


def local_file_exists( path: str ) -> bool:
    """Check whether an approved workspace path exists.

    Purpose:
        Check whether an approved workspace path exists.

    Args:
        path: Path relative to the host-approved workspace.

    Returns:
        bool: Result of the local operation.
    """
    return _files( ).exists( path )


def local_file_info( path: str ) -> str:
    """Return JSON metadata for an approved workspace path.

    Purpose:
        Return JSON metadata for an approved workspace path.

    Args:
        path: Path relative to the host-approved workspace.

    Returns:
        str: Result of the local operation.
    """
    return json.dumps( _files( ).get_info( path ), default=str )


def local_file_hash( path: str ) -> str:
    """Calculate the SHA-256 digest of a local file.

    Purpose:
        Calculate the SHA-256 digest of a local file.

    Args:
        path: Path relative to the host-approved workspace.

    Returns:
        str: Result of the local operation.
    """
    return _files( ).calculate_hash( path )


def local_sqlite_tables( path: str ) -> list[str]:
    """List tables in an existing SQLite database.

    Purpose:
        List tables in an existing SQLite database.

    Args:
        path: Path relative to the host-approved workspace.

    Returns:
        list[str]: Result of the local operation.
    """
    database = Data( _existing( path ), read_only=True )
    try:
        return database.list_tables( )
    finally:
        database.close( )


def local_sqlite_schema( path: str, table: str ) -> str:
    """Return JSON column metadata for an SQLite table.

    Purpose:
        Return JSON column metadata for an SQLite table.

    Args:
        path: Path relative to the host-approved workspace.
        table: SQLite table name.

    Returns:
        str: Result of the local operation.
    """
    database = Data( _existing( path ), read_only=True )
    try:
        return json.dumps( database.get_schema( table ), default=str )
    finally:
        database.close( )


def local_sqlite_query( path: str, sql: str, parameters_json: str = '[]', max_rows: int = 50 ) -> str:
    """Execute a read-only SELECT query against an SQLite database.

    Purpose:
        Execute a read-only SELECT query against an SQLite database.

    Args:
        path: Path relative to the host-approved workspace.
        sql: Read-only SELECT statement.
        parameters_json: JSON list of SQLite positional parameters.
        max_rows: Maximum returned rows (1–200).

    Returns:
        str: Result of the local operation.
    """
    if not 1 <= max_rows <= 200:
        raise ValueError( 'max_rows must be between 1 and 200.' )
    parameters = json.loads( parameters_json )
    if not isinstance( parameters, list ):
        raise ValueError( 'parameters_json must be an array.' )
    database = Data( _existing( path ), read_only=True )
    try:
        rows = database.execute_query( sql, parameters )
        return json.dumps( rows[ :max_rows ], default=str )
    finally:
        database.close( )


def local_excel_sheets( path: str ) -> list[str]:
    """List worksheets in an existing Excel workbook.

    Purpose:
        List worksheets in an existing Excel workbook.

    Args:
        path: Path relative to the host-approved workspace.

    Returns:
        list[str]: Result of the local operation.
    """
    excel = Excel( backend='file' )
    try:
        excel.open_workbook( _existing( path ) )
        return excel.list_worksheets( )
    finally:
        if excel.workbook is not None:
            excel.close_workbook( )


def local_excel_cell( path: str, sheet: str, cell: str ) -> str:
    """Read a single Excel cell as JSON.

    Purpose:
        Read a single Excel cell as JSON.

    Args:
        path: Path relative to the host-approved workspace.
        sheet: Excel worksheet name.
        cell: Excel A1 cell address.

    Returns:
        str: Result of the local operation.
    """
    excel = Excel( backend='file' )
    try:
        excel.open_workbook( _existing( path ) )
        return json.dumps( excel.read_cell( sheet, cell ), default=str )
    finally:
        if excel.workbook is not None:
            excel.close_workbook( )


def local_excel_range( path: str, sheet: str, address: str ) -> str:
    """Read a bounded Excel range as JSON.

    Purpose:
        Read a bounded Excel range as JSON.

    Args:
        path: Path relative to the host-approved workspace.
        sheet: Excel worksheet name.
        address: Excel A1 range address.

    Returns:
        str: Result of the local operation.
    """
    excel = Excel( backend='file' )
    try:
        excel.open_workbook( _existing( path ) )
        rows = excel.read_range( sheet, address )
        if len( rows ) > 500 or any( len( row ) > 50 for row in rows ):
            raise ValueError( 'Excel range exceeds 500 rows or 50 columns.' )
        return json.dumps( rows, default=str )
    finally:
        if excel.workbook is not None:
            excel.close_workbook( )


def local_excel_write_range( path: str, sheet: str, start_cell: str, rows: list[list[str | int | float | bool | None]] ) -> str:
    """Write an Excel range and save the existing workbook.

    Purpose:
        Write an Excel range and save the existing workbook.

    Args:
        path: Path relative to the host-approved workspace.
        sheet: Excel worksheet name.
        start_cell: Top-left destination cell.
        rows: Rows of primitive Excel cell values.

    Returns:
        str: Result of the local operation.
    """
    _require_write( )
    if not rows or len( rows ) > 500 or any( len( row ) > 50 for row in rows ):
        raise ValueError( 'Use 1–500 rows of at most 50 columns.' )
    excel = Excel( backend='file' )
    try:
        excel.open_workbook( _existing( path ) )
        excel.write_range( sheet, start_cell, rows )
        return excel.save_workbook( )
    finally:
        if excel.workbook is not None:
            excel.close_workbook( )


def local_word_read( path: str ) -> str:
    """Read up to 100000 characters from an existing Word document.

    Purpose:
        Read up to 100000 characters from an existing Word document.

    Args:
        path: Path relative to the host-approved workspace.

    Returns:
        str: Result of the local operation.
    """
    word = Word( backend='file' )
    try:
        word.open_document( _existing( path ) )
        return word.read_document( )[ :100000 ]
    finally:
        if word.document is not None:
            word.close_document( )


def local_word_create( path: str, text: str ) -> str:
    """Create a new Word document without overwriting another file.

    Purpose:
        Create a new Word document without overwriting another file.

    Args:
        path: Path relative to the host-approved workspace.
        text: Text to insert into the target document or slide.

    Returns:
        str: Result of the local operation.
    """
    _require_write( )
    destination = _new( path )
    word = Word( backend='file' )
    try:
        word.create_document( )
        word.append_text( text )
        return word.save_document( destination )
    finally:
        if word.document is not None:
            word.close_document( )


def local_word_append( path: str, text: str ) -> str:
    """Append text to an existing Word document and save it.

    Purpose:
        Append text to an existing Word document and save it.

    Args:
        path: Path relative to the host-approved workspace.
        text: Text to insert into the target document or slide.

    Returns:
        str: Result of the local operation.
    """
    _require_write( )
    word = Word( backend='file' )
    try:
        word.open_document( _existing( path ) )
        word.append_text( text )
        return word.save_document( )
    finally:
        if word.document is not None:
            word.close_document( )


def local_word_replace( path: str, old: str, new: str ) -> int:
    """Replace text in an existing Word document and save changes.

    Purpose:
        Replace text in an existing Word document and save changes.

    Args:
        path: Path relative to the host-approved workspace.
        old: Original text to replace.
        new: Replacement text.

    Returns:
        int: Result of the local operation.
    """
    _require_write( )
    word = Word( backend='file' )
    try:
        word.open_document( _existing( path ) )
        count = word.replace_text( old, new )
        if count:
            word.save_document( )
        return count
    finally:
        if word.document is not None:
            word.close_document( )


def local_slides_create( path: str ) -> str:
    """Create a new PowerPoint presentation without overwriting an existing file.

    Purpose:
        Create a new PowerPoint presentation without overwriting an existing file.

    Args:
        path: Path relative to the host-approved workspace.

    Returns:
        str: Result of the local operation.
    """
    _require_write( )
    destination = _new( path )
    slides = PowerPoint( backend='file' )
    try:
        slides.create_presentation( )
        return slides.save_presentation( destination )
    finally:
        if slides.presentation is not None:
            slides.close_presentation( )


def local_slides_add_slide( path: str, title: str ) -> int:
    """Add a titled slide to an existing PowerPoint presentation.

    Purpose:
        Add a titled slide to an existing PowerPoint presentation.

    Args:
        path: Path relative to the host-approved workspace.
        title: PowerPoint slide title or window title.

    Returns:
        int: Result of the local operation.
    """
    _require_write( )
    slides = PowerPoint( backend='file' )
    try:
        slides.open_presentation( _existing( path ) )
        number = slides.add_slide( title )
        slides.save_presentation( )
        return number
    finally:
        if slides.presentation is not None:
            slides.close_presentation( )


def local_slides_add_text( path: str, slide_number: int, text: str ) -> str:
    """Add text to an existing PowerPoint slide and save.

    Purpose:
        Add text to an existing PowerPoint slide and save.

    Args:
        path: Path relative to the host-approved workspace.
        slide_number: One-based slide number.
        text: Text to insert into the target document or slide.

    Returns:
        str: Result of the local operation.
    """
    _require_write( )
    slides = PowerPoint( backend='file' )
    try:
        slides.open_presentation( _existing( path ) )
        slides.add_text( slide_number, text )
        return slides.save_presentation( )
    finally:
        if slides.presentation is not None:
            slides.close_presentation( )


def local_outlook_messages( limit: int = 20 ) -> str:
    """List recent messages from the local Outlook profile.

    Purpose:
        List recent messages from the local Outlook profile.

    Args:
        limit: Maximum number of Outlook messages to scan.

    Returns:
        str: Result of the local operation.
    """
    _require_outlook( )
    if not 1 <= limit <= 50:
        raise ValueError( 'limit must be between 1 and 50.' )
    outlook = Outlook( )
    outlook.connect( )
    return json.dumps( outlook.list_messages( limit ), default=str )


def local_outlook_search( phrase: str, limit: int = 50 ) -> str:
    """Search recent Outlook messages for a phrase.

    Purpose:
        Search recent Outlook messages for a phrase.

    Args:
        phrase: Outlook message search phrase.
        limit: Maximum number of Outlook messages to scan.

    Returns:
        str: Result of the local operation.
    """
    _require_outlook( )
    if not 1 <= limit <= 100:
        raise ValueError( 'limit must be between 1 and 100.' )
    outlook = Outlook( )
    outlook.connect( )
    return json.dumps( outlook.search_messages( phrase, limit ), default=str )


def local_outlook_read( entry_id: str ) -> str:
    """Read a message from the local Outlook profile.

    Purpose:
        Read a message from the local Outlook profile.

    Args:
        entry_id: Outlook message identifier.

    Returns:
        str: Result of the local operation.
    """
    _require_outlook( )
    outlook = Outlook( )
    outlook.connect( )
    return json.dumps( outlook.read_message( entry_id ), default=str )


def local_outlook_draft( recipient: str, subject: str, body: str ) -> str:
    """Create an Outlook draft without sending it.

    Purpose:
        Create an Outlook draft without sending it.

    Args:
        recipient: Draft recipient address.
        subject: Draft subject line.
        body: Draft message body.

    Returns:
        str: Result of the local operation.
    """
    _require_outlook( )
    _require_write( )
    outlook = Outlook( )
    outlook.connect( )
    return outlook.create_draft( recipient, subject, body )


def local_windows_list(  ) -> list[str]:
    """List windows in the active local Windows desktop session.

    Purpose:
        List windows in the active local Windows desktop session.

    Returns:
        list[str]: Result of the local operation.
    """
    _require_desktop( )
    return Windows( ).list_windows( )


def local_windows_controls( title: str ) -> list[str]:
    """List controls of a selected local Windows application window.

    Purpose:
        List controls of a selected local Windows application window.

    Args:
        title: PowerPoint slide title or window title.

    Returns:
        list[str]: Result of the local operation.
    """
    _require_desktop( )
    windows = Windows( )
    windows.find_window( title )
    return windows.list_controls( )


def local_system_info(  ) -> str:
    """Get local operating-system information as JSON.

    Purpose:
        Get local operating-system information as JSON.

    Returns:
        str: Result of the local operation.
    """
    return json.dumps( Shell( ).get_system_info( ), default=str )


def local_disk_usage( path: str = '.' ) -> str:
    """Read disk-capacity statistics for a permitted workspace path.

    Purpose:
        Read disk-capacity statistics for a permitted workspace path.

    Args:
        path: Path relative to the host-approved workspace.

    Returns:
        str: Result of the local operation.
    """
    return json.dumps( Shell( ).get_disk_usage( _path( path ) ), default=str )


def local_tool_schema( operation: Callable ) -> dict[str, object]:
    """Create a JSON Schema for a standalone local tool callable.

    Purpose:
        Extracts typed parameters and defaults for xAI and Mistral native tool definitions.

    Args:
        operation (Callable): One of the approved local tool functions.

    Returns:
        dict[str, object]: JSON Schema object describing the tool's input parameters.
    """
    properties = {}
    required = []
    for parameter in signature( operation ).parameters.values( ):
        details = TypeAdapter( parameter.annotation ).json_schema( )
        if parameter.default is Parameter.empty:
            required.append( parameter.name )
        else:
            details[ 'default' ] = parameter.default
        properties[ parameter.name ] = details
    return { 'type': 'object', 'properties': properties, 'required': required,
        'additionalProperties': False }


__all__ = [
    'local_calculate',
    'local_percentage',
    'local_file_list',
    'local_file_read',
    'local_file_exists',
    'local_file_info',
    'local_file_hash',
    'local_sqlite_tables',
    'local_sqlite_schema',
    'local_sqlite_query',
    'local_excel_sheets',
    'local_excel_cell',
    'local_excel_range',
    'local_excel_write_range',
    'local_word_read',
    'local_word_create',
    'local_word_append',
    'local_word_replace',
    'local_slides_create',
    'local_slides_add_slide',
    'local_slides_add_text',
    'local_outlook_messages',
    'local_outlook_search',
    'local_outlook_read',
    'local_outlook_draft',
    'local_windows_list',
    'local_windows_controls',
    'local_system_info',
    'local_disk_usage',
]
