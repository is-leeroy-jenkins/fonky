'''
  ******************************************************************************************
      Assembly:                fonky
      Filename:                loco.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="loco.py" company="Terry D. Eppler">

	     loco.py
	     Copyright ©  2022  Terry Eppler

     Permission is hereby granted, free of charge, to any person obtaining a copy
     of this software and associated documentation files (the “Software”),
     to deal in the Software without restriction,
     including without limitation the rights to use,
     copy, modify, merge, publish, distribute, sublicense,
     and/or sell copies of the Software,
     and to permit persons to whom the Software is furnished to do so,
     subject to the following conditions:

     The above copyright notice and this permission notice shall be included in all
     copies or substantial portions of the Software.

     THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
     INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
     FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
     IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
     ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
     DEALINGS IN THE SOFTWARE.

     You can contact me at:  terryeppler@gmail.com or eppler.terry@epa.gov

  </copyright>
  <summary>
    Local automation for Microsoft Office, SQLite, the filesystem, and Windows.
  </summary>
  ******************************************************************************************
  '''


from __future__ import annotations

import ast
import hashlib
import math
import operator
import os
import platform
import shutil
import sqlite3
import subprocess
import sys
import zipfile
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Sequence


# ==========================================================================================
# VALIDATION AND PLATFORM SUPPORT
# ==========================================================================================

def throw_if( name: str, value: object ) -> None:
    """Validate a required argument.

    Purpose:
        Validate a required argument.

    Args:
        name (str): Parameter name.
        value (object): Parameter value.

    Raises:
        ValueError: If the argument is missing or empty.
    """
    if value is None or (isinstance( value, str ) and not value.strip( )):
        raise ValueError( f'Argument "{name}" cannot be empty!' )
    if isinstance( value, (list, tuple, dict, set) ) and not value:
        raise ValueError( f'Argument "{name}" cannot be empty!' )



def sql_identifier( name: str ) -> str:
    """Quote a SQLite identifier without treating it as SQL syntax.

    Purpose:
        Quote a SQLite identifier without treating it as SQL syntax.

    Args:
        name (str): Database table or column name.

    Returns:
        str: Escaped and quoted identifier.
    """
    throw_if( 'name', name )
    return '"' + name.replace( '"', '""' ) + '"'


def require_windows( ) -> None:
    """Require a Windows desktop environment.

    Purpose:
        Require a Windows desktop environment.

    Raises:
        OSError: If the current platform is not Windows.
    """
    if sys.platform != 'win32':
        raise OSError( 'This operation requires Windows and a desktop session.' )


def office_application( prog_id: str, visible: bool = False ) -> Any:
    """Create a dedicated Office COM application instance.

    Purpose:
        Create a dedicated Office COM application instance.

    Args:
        prog_id (str): Registered Office COM identifier.
        visible (bool): Initial visibility.

    Returns:
        Any: COM application proxy.

    Raises:
        OSError: If Windows is unavailable.
        ImportError: If pywin32 is not installed.
    """
    throw_if( 'prog_id', prog_id )
    require_windows( )
    import win32com.client
    application = win32com.client.DispatchEx( prog_id )
    if prog_id != 'Outlook.Application':
        application.Visible = visible
    return application


# ==========================================================================================
# LOCAL FILESYSTEM
# ==========================================================================================

class File:
    """Local filesystem operations with an optional root-directory boundary.

    Purpose:
        Exposes the public File automation API through local Python or supported
        desktop-application backends without requiring a cloud-model API key.

    Attributes:
        root (Path | None): Configured filesystem access boundary, if supplied.
        path (str | Path): Active local file, document, or database path.
        directory (Path): Selected filesystem directory.
        pattern (str): Glob expression used for file discovery.
        recursive (bool): Whether the current operation traverses subdirectories.
        entries (Any): Filesystem iterator used by the active search.
        encoding (str): Character encoding for text file operations.
        content (str | bytes): Content supplied for the active write operation.
        overwrite (bool): Whether an existing target may be replaced.
        source (Path): Resolved source of a copy or move.
        destination (str | Path): Resolved destination of the active operation.
        algorithm (str): Selected file hashing algorithm.
        digest (Any): Incremental cryptographic hash state.
        archive (Path): ZIP archive selected for compression or extraction.
        stat (Any): Filesystem metadata returned by Path.stat().
        new_name (str): Requested destination filename.

    Methods:
        resolve(): Resolve a path within the configured root.
        exists(): Check whether a local path exists.
        list_files(): List files matching a glob pattern.
        list_directories(): List immediate child directories.
        read_text(): Read a text file.
        write_text(): Write text without overwriting an existing file by default.
        copy_file(): Copy a file to a new destination.
        move_file(): Move a file without overwriting by default.
        delete_file(): Remove a specified file.
        create_directory(): Create a directory and missing parents.
        calculate_hash(): Calculate a supported file digest.
        compress(): Create a ZIP archive without following out-of-root symlinks.
        extract(): Extract a ZIP archive into an allowed directory.
        get_info(): Return filesystem metadata.
        read_bytes(): Read binary file contents.
        write_bytes(): Write binary file contents.
        rename_file(): Rename a file within its existing directory.
        get_size(): Get a file size in bytes.
        delete_directory(): Remove a directory, optionally including its contents.
    """


    # Public instance members (annotations only; values are assigned by operations).
    root: Path | None  # Configured filesystem access boundary, if supplied.
    path: str | Path  # Active local file, document, or database path.
    directory: Path  # Selected filesystem directory.
    pattern: str  # Glob expression used for file discovery.
    recursive: bool  # Whether the current operation traverses subdirectories.
    entries: Any  # Filesystem iterator used by the active search.
    encoding: str  # Character encoding for text file operations.
    content: str | bytes  # Content supplied for the active write operation.
    overwrite: bool  # Whether an existing target may be replaced.
    source: Path  # Resolved source of a copy or move.
    destination: str | Path  # Resolved destination of the active operation.
    algorithm: str  # Selected file hashing algorithm.
    digest: Any  # Incremental cryptographic hash state.
    archive: Path  # ZIP archive selected for compression or extraction.
    stat: Any  # Filesystem metadata returned by Path.stat().
    new_name: str  # Requested destination filename.

    def __init__( self, root: str = '' ) -> None:
        """Configure permitted filesystem scope.

        Purpose:
            Configure permitted filesystem scope.

        Args:
            root (str): Optional directory limiting access.

        Returns:
            None: Initializes the filesystem service.
        """
        self.root = Path( root ).resolve( ) if root else None

    def resolve( self, path: str ) -> Path:
        """Resolve a path within the configured root.

        Purpose:
            Resolve a path within the configured root.

        Args:
            path (str): File or directory path.

        Returns:
            Path: Resolved filesystem path.

        Raises:
            ValueError: If the path is missing or outside the permitted root.
        """
        try:
            throw_if( 'path', path )
            self.path = Path( path )
            if self.root is not None and not self.path.is_absolute( ):
                self.path = self.root / self.path
            self.path = self.path.resolve( )
            if self.root is not None and not self.path.is_relative_to( self.root ):
                raise PermissionError( f'Path is outside the permitted directory: {self.path}' )
            return self.path
        except Exception as e:
            e.add_note( 'fonky.loco.File.resolve' )
            raise

    def exists( self, path: str ) -> bool:
        """Check whether a local path exists.

        Purpose:
            Check whether a local path exists.

        Args:
            path (str): Target path.

        Returns:
            bool: True if the path exists.
        """
        try:
            self.path = path
            return self.resolve( self.path ).exists( )
        except Exception as e:
            e.add_note( 'fonky.loco.File.exists' )
            raise

    def list_files( self, directory: str = '.', pattern: str = '*', recursive: bool = False ) -> list[ str ]:
        """List files matching a glob pattern.

        Purpose:
            List files matching a glob pattern.

        Args:
            directory (str): Starting directory.
            pattern (str): Glob pattern.
            recursive (bool): Include subdirectories.

        Returns:
            list[str]: Matching file paths.
        """
        self.directory = self.resolve( directory )
        self.pattern = pattern
        self.recursive = recursive
        self.entries = self.directory.rglob( self.pattern ) if self.recursive else self.directory.glob( self.pattern )
        return sorted( str( entry ) for entry in self.entries if entry.is_file( ) and
            (self.root is None or entry.resolve( ).is_relative_to( self.root )) )

    def list_directories( self, directory: str = '.' ) -> list[ str ]:
        """List immediate child directories.

        Purpose:
            List immediate child directories.

        Args:
            directory (str): Parent directory.

        Returns:
            list[str]: Child directory paths.
        """
        self.directory = self.resolve( directory )
        return sorted( str( entry ) for entry in self.directory.iterdir( ) if entry.is_dir( ) and
            (self.root is None or entry.resolve( ).is_relative_to( self.root )) )

    def read_text( self, path: str, encoding: str = 'utf-8' ) -> str:
        """Read a text file.

        Purpose:
            Read a text file.

        Args:
            path (str): Source file.
            encoding (str): Text encoding.

        Returns:
            str: File contents.
        """
        try:
            self.path = path
            self.encoding = encoding
            return self.resolve( self.path ).read_text( encoding=self.encoding )
        except Exception as e:
            e.add_note( 'fonky.loco.File.read_text' )
            raise

    def write_text( self, path: str, content: str, overwrite: bool = False,
        encoding: str = 'utf-8' ) -> int:
        """Write text without overwriting an existing file by default.

        Purpose:
            Write text without overwriting an existing file by default.

        Args:
            path (str): Output file.
            content (str): Text content.
            overwrite (bool): Allow replacing an existing file.
            encoding (str): Text encoding.

        Returns:
            int: Characters written.
        """
        try:
            self.path = self.resolve( path )
            self.content = content
            self.overwrite = overwrite
            self.encoding = encoding
            with self.path.open( 'w' if self.overwrite else 'x', encoding=self.encoding ) as handle:
                return handle.write( self.content )
        except Exception as e:
            e.add_note( 'fonky.loco.File.write_text' )
            raise

    def copy_file( self, source: str, destination: str, overwrite: bool = False ) -> str:
        """Copy a file to a new destination.

        Purpose:
            Copy a file to a new destination.

        Args:
            source (str): Source file.
            destination (str): Destination file.
            overwrite (bool): Permit replacing an existing destination.

        Returns:
            str: Destination path.
        """
        try:
            self.source = self.resolve( source )
            self.destination = self.resolve( destination )
            self.overwrite = overwrite
            if not self.overwrite and self.destination.exists( ):
                raise FileExistsError( str( self.destination ) )
            shutil.copy2( self.source, self.destination )
            return str( self.destination )
        except Exception as e:
            e.add_note( 'fonky.loco.File.copy_file' )
            raise

    def move_file( self, source: str, destination: str, overwrite: bool = False ) -> str:
        """Move a file without overwriting by default.

        Purpose:
            Move a file without overwriting by default.

        Args:
            source (str): Source file.
            destination (str): Destination path.
            overwrite (bool): Permit destination replacement.

        Returns:
            str: Destination path.
        """
        try:
            self.source = self.resolve( source )
            self.destination = self.resolve( destination )
            self.overwrite = overwrite
            if self.destination.exists( ) and not self.overwrite:
                raise FileExistsError( str( self.destination ) )
            if self.destination.exists( ):
                self.destination.unlink( )
            shutil.move( str( self.source ), str( self.destination ) )
            return str( self.destination )
        except Exception as e:
            e.add_note( 'fonky.loco.File.move_file' )
            raise

    def delete_file( self, path: str ) -> None:
        """Remove a specified file.

        Purpose:
            Remove a specified file.

        Args:
            path (str): File to delete.

        Returns:
            None: Deletes the file.
        """
        try:
            self.path = self.resolve( path )
            self.path.unlink( )
        except Exception as e:
            e.add_note( 'fonky.loco.File.delete_file' )
            raise

    def create_directory( self, path: str ) -> str:
        """Create a directory and missing parents.

        Purpose:
            Create a directory and missing parents.

        Args:
            path (str): Directory path.

        Returns:
            str: Created directory path.
        """
        try:
            self.path = self.resolve( path )
            self.path.mkdir( parents=True, exist_ok=True )
            return str( self.path )
        except Exception as e:
            e.add_note( 'fonky.loco.File.create_directory' )
            raise

    def calculate_hash( self, path: str, algorithm: str = 'sha256' ) -> str:
        """Calculate a supported file digest.

        Purpose:
            Calculate a supported file digest.

        Args:
            path (str): Source file.
            algorithm (str): Hash algorithm accepted by hashlib.

        Returns:
            str: Hex digest.
        """
        try:
            self.path = self.resolve( path )
            self.algorithm = algorithm
            self.digest = hashlib.new( self.algorithm )
            with self.path.open( 'rb' ) as handle:
                for chunk in iter( lambda: handle.read( 1024 * 1024 ), b'' ):
                    self.digest.update( chunk )
            return self.digest.hexdigest( )
        except Exception as e:
            e.add_note( 'fonky.loco.File.calculate_hash' )
            raise

    def compress( self, directory: str, destination: str ) -> str:
        """Create a ZIP archive without following out-of-root symlinks.

        Purpose:
            Create a ZIP archive without following out-of-root symlinks.

        Args:
            directory (str): Directory to archive.
            destination (str): New ZIP path.

        Returns:
            str: Archive path.
        """
        try:
            self.directory = self.resolve( directory )
            self.destination = self.resolve( destination )
            with zipfile.ZipFile( self.destination, 'x', zipfile.ZIP_DEFLATED ) as archive:
                for entry in self.directory.rglob( '*' ):
                    if entry.is_file( ) and entry.resolve( ) != self.destination and (
                        self.root is None or entry.resolve( ).is_relative_to( self.root )):
                        archive.write( entry, entry.relative_to( self.directory ) )
            return str( self.destination )
        except Exception as e:
            e.add_note( 'fonky.loco.File.compress' )
            raise

    def extract( self, archive: str, destination: str ) -> str:
        """Extract a ZIP archive into an allowed directory.

        Purpose:
            Extract a ZIP archive into an allowed directory.

        Args:
            archive (str): Source ZIP archive.
            destination (str): Extraction directory.

        Returns:
            str: Extraction directory.

        Raises:
            PermissionError: If an entry escapes the extraction boundary.
        """
        try:
            self.archive = self.resolve( archive )
            self.destination = self.resolve( destination )
            with zipfile.ZipFile( self.archive ) as source:
                for member in source.infolist( ):
                    target = (self.destination / member.filename).resolve( )
                    if not target.is_relative_to( self.destination ) or (
                        self.root is not None and not target.is_relative_to( self.root )):
                        raise PermissionError( f'Unsafe archive entry: {member.filename}' )
                source.extractall( self.destination )
            return str( self.destination )


    # ==========================================================================================
    # SQLITE DATABASE
    # ==========================================================================================
        except Exception as e:
            e.add_note( 'fonky.loco.File.extract' )
            raise


    def get_info( self, path: str ) -> dict[str, Any]:
        """Return filesystem metadata.

        Purpose:
            Return filesystem metadata.

        Args:
            path (str): Target path.

        Returns:
            dict[str, Any]: Name, path, type, size, and modification time.
        """
        try:
            self.path = self.resolve( path )
            self.stat = self.path.stat( )
            return { 'name': self.path.name, 'path': str( self.path ),
                'is_file': self.path.is_file( ), 'is_directory': self.path.is_dir( ),
                'size': self.stat.st_size, 'modified': self.stat.st_mtime }
        except Exception as e:
            e.add_note( 'fonky.loco.File.get_info' )
            raise

    def read_bytes( self, path: str ) -> bytes:
        """Read binary file contents.

        Purpose:
            Read binary file contents.

        Args:
            path (str): Source file.

        Returns:
            bytes: Binary content.
        """
        try:
            self.path = self.resolve( path )
            return self.path.read_bytes( )
        except Exception as e:
            e.add_note( 'fonky.loco.File.read_bytes' )
            raise

    def write_bytes( self, path: str, content: bytes, overwrite: bool = False ) -> int:
        """Write binary file contents.

        Purpose:
            Write binary file contents.

        Args:
            path (str): Destination file.
            content (bytes): Binary content.
            overwrite (bool): Permit file replacement.

        Returns:
            int: Number of bytes written.
        """
        try:
            throw_if( 'content', content )
            self.path = self.resolve( path )
            self.content = content
            self.overwrite = overwrite
            with self.path.open( 'wb' if self.overwrite else 'xb' ) as handle:
                return handle.write( self.content )
        except Exception as e:
            e.add_note( 'fonky.loco.File.write_bytes' )
            raise

    def rename_file( self, path: str, new_name: str ) -> str:
        """Rename a file within its existing directory.

        Purpose:
            Rename a file within its existing directory.

        Args:
            path (str): Existing path.
            new_name (str): New filename without directory separators.

        Returns:
            str: Renamed file path.
        """
        try:
            throw_if( 'new_name', new_name )
            self.path = self.resolve( path )
            self.new_name = new_name
            if Path( self.new_name ).name != self.new_name or self.new_name in ('.', '..'):
                raise ValueError( 'new_name must be a filename, not a path.' )
            self.destination = self.resolve( str( self.path.with_name( self.new_name ) ) )
            if self.destination.exists( ):
                raise FileExistsError( str( self.destination ) )
            return str( self.path.rename( self.destination ) )
        except Exception as e:
            e.add_note( 'fonky.loco.File.rename_file' )
            raise

    def get_size( self, path: str ) -> int:
        """Get a file size in bytes.

        Purpose:
            Get a file size in bytes.

        Args:
            path (str): Source file path.

        Returns:
            int: Size in bytes.
        """
        try:
            self.path = self.resolve( path )
            return self.path.stat( ).st_size
        except Exception as e:
            e.add_note( 'fonky.loco.File.get_size' )
            raise

    def delete_directory( self, path: str, recursive: bool = False ) -> None:
        """Remove a directory, optionally including its contents.

        Purpose:
            Remove a directory, optionally including its contents.

        Args:
            path (str): Directory path.
            recursive (bool): Permit recursive deletion.

        Returns:
            None: Removes the directory.

        Raises:
            PermissionError: When attempting to remove the configured root.
        """
        try:
            self.path = self.resolve( path )
            self.recursive = recursive
            if self.root is not None and self.path == self.root:
                raise PermissionError( 'Cannot delete the configured filesystem root.' )
            if self.recursive:
                shutil.rmtree( self.path )
            else:
                self.path.rmdir( )
        except Exception as e:
            e.add_note( 'fonky.loco.File.delete_directory' )
            raise


class Data:
    """SQLite connection and transaction operations.

    Purpose:
        Exposes the public SQLite automation API through local Python or supported
        desktop-application backends without requiring a cloud-model API key.

    Attributes:
        path (str | Path): Active local file, document, or database path.
        read_only (bool): Whether SQLite permits write operations.
        connection (sqlite3.Connection | None): Managed SQLite connection, initialized on demand.
        uri (str): SQLite URI used for a read-only database connection.
        sql (str): SQL statement supplied to the current operation.
        parameters (Sequence[Any]): Bound parameters for the active SQL statement.
        statement (str): SQL statement prefix used to enforce read operations.
        cursor (sqlite3.Cursor): Cursor for the most recent SQLite command.
        rows (Any): Rows used for a database, spreadsheet, or presentation operation.
        table (str): Selected SQLite table or Office table name.
        destination (str | Path): Resolved destination of the active operation.
        columns (Any): Column definitions or table dimensions.
        approved (bool): Explicit authorization for a sensitive operation.
        values (dict[str, Any]): Values mapped to database columns or document elements.
        where (str): Trusted SQL WHERE clause supplied by the caller.
        where_parameters (Sequence[Any]): Bound parameters used by the SQL WHERE clause.
        df_data (Any): Pandas DataFrame supplied to the active data operation.
        if_exists (str): SQLite DataFrame table replacement policy.

    Methods:
        connect(): Connect to SQLite.
        close(): Close the database connection.
        execute_query(): Execute parameterized SQL and return rows.
        execute_nonquery(): Execute a parameterized write statement without committing.
        execute_many(): Execute a parameterized write for multiple records.
        list_tables(): List non-system SQLite tables.
        get_schema(): Read a table schema.
        commit(): Commit pending changes.
        rollback(): Rollback pending changes.
        backup_database(): Copy a consistent SQLite database snapshot.
        integrity_check(): Verify SQLite file integrity.
        read_dataframe(): Read a query into a pandas DataFrame.
        create_table(): Create a table with explicitly supported SQLite column types.
        drop_table(): Drop a table when expressly approved.
        insert_record(): Insert a parameterized SQLite record.
        update_record(): Update selected records using bound values.
        delete_record(): Delete selected SQLite records with explicit permission.
        write_dataframe(): Write a DataFrame to SQLite.
    """


    # Public instance members (annotations only; values are assigned by operations).
    path: str | Path  # Active local file, document, or database path.
    read_only: bool  # Whether SQLite permits write operations.
    connection: sqlite3.Connection | None  # Managed SQLite connection, initialized on demand.
    uri: str  # SQLite URI used for a read-only database connection.
    sql: str  # SQL statement supplied to the current operation.
    parameters: Sequence[Any]  # Bound parameters for the active SQL statement.
    statement: str  # SQL statement prefix used to enforce read operations.
    cursor: sqlite3.Cursor  # Cursor for the most recent SQLite command.
    rows: Any  # Rows used for a database, spreadsheet, or presentation operation.
    table: str  # Selected SQLite table or Office table name.
    destination: str | Path  # Resolved destination of the active operation.
    columns: Any  # Column definitions or table dimensions.
    approved: bool  # Explicit authorization for a sensitive operation.
    values: dict[str, Any]  # Values mapped to database columns or document elements.
    where: str  # Trusted SQL WHERE clause supplied by the caller.
    where_parameters: Sequence[Any]  # Bound parameters used by the SQL WHERE clause.
    df_data: Any  # Pandas DataFrame supplied to the active data operation.
    if_exists: str  # SQLite DataFrame table replacement policy.

    def __init__( self, path: str, read_only: bool = False ) -> None:
        """Initialize a database wrapper without connecting.

        Purpose:
            Initialize a database wrapper without connecting.

        Args:
            path (str): Database file path.
            read_only (bool): Open using SQLite read-only mode.

        Returns:
            None: Stores database configuration.
        """
        self.path = path
        self.read_only = read_only
        self.connection = None

    def connect( self ) -> None:
        """Connect to SQLite.

        Purpose:
            Connect to SQLite.

        Returns:
            None: Creates a connection held on the instance.
        """
        if self.connection is not None:
            return
        if self.read_only:
            self.uri = Path( self.path ).resolve( ).as_uri( ) + '?mode=ro'
            self.connection = sqlite3.connect( self.uri, uri=True )
        else:
            self.connection = sqlite3.connect( self.path )
        self.connection.row_factory = sqlite3.Row

    def close( self ) -> None:
        """Close the database connection.

        Purpose:
            Close the database connection.

        Returns:
            None: Closes the connection if present.
        """
        if self.connection is not None:
            self.connection.close( )
            self.connection = None

    def execute_query( self, sql: str, parameters: Sequence[Any] = () ) -> list[ dict[ str, Any ] ]:
        """Execute parameterized SQL and return rows.

        Purpose:
            Execute parameterized SQL and return rows.

        Args:
            sql (str): SQL query.
            parameters (Sequence[Any]): Positional SQL parameters.

        Returns:
            list[dict[str, Any]]: Query rows.
        """
        try:
            throw_if( 'sql', sql )
            self.sql = sql
            self.parameters = parameters
            self.statement = self.sql.lstrip( ).upper( )
            if not (self.statement.startswith( 'SELECT ' ) or
                self.statement.startswith( 'SELECT\n' ) or
                self.statement.startswith( 'PRAGMA INTEGRITY_CHECK' )):
                raise PermissionError( 'execute_query permits SELECT and integrity-check only.' )
            self.connect( )
            self.cursor = self.connection.execute( self.sql, self.parameters )
            if self.cursor.description is None:
                raise ValueError( 'execute_query requires a result-producing statement.' )
            return [ dict( record ) for record in self.cursor.fetchall( ) ]
        except Exception as e:
            e.add_note( 'fonky.loco.Data.execute_query' )
            raise

    def execute_nonquery( self, sql: str, parameters: Sequence[Any] = () ) -> int:
        """Execute a parameterized write statement without committing.

        Purpose:
            Execute a parameterized write statement without committing.

        Args:
            sql (str): SQL command.
            parameters (Sequence[Any]): Positional SQL parameters.

        Returns:
            int: Number of affected rows.
        """
        try:
            throw_if( 'sql', sql )
            self.sql = sql
            self.parameters = parameters
            if self.read_only:
                raise PermissionError( 'Database is read-only.' )
            self.connect( )
            self.cursor = self.connection.execute( self.sql, self.parameters )
            return self.cursor.rowcount
        except Exception as e:
            e.add_note( 'fonky.loco.Data.execute_nonquery' )
            raise

    def execute_many( self, sql: str, rows: Sequence[Sequence[Any]] ) -> int:
        """Execute a parameterized write for multiple records.

        Purpose:
            Execute a parameterized write for multiple records.

        Args:
            sql (str): SQL command.
            rows (Sequence[Sequence[Any]]): Parameter groups.

        Returns:
            int: Number of affected rows.
        """
        try:
            throw_if( 'sql', sql )
            self.sql = sql
            self.rows = rows
            if self.read_only:
                raise PermissionError( 'Database is read-only.' )
            self.connect( )
            self.cursor = self.connection.executemany( self.sql, self.rows )
            return self.cursor.rowcount
        except Exception as e:
            e.add_note( 'fonky.loco.Data.execute_many' )
            raise

    def list_tables( self ) -> list[ str ]:
        """List non-system SQLite tables.

        Purpose:
            List non-system SQLite tables.

        Returns:
            list[str]: Table names.
        """
        return [ row[ 'name' ] for row in self.execute_query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name" ) ]

    def get_schema( self, table: str ) -> list[ dict[ str, Any ] ]:
        """Read a table schema.

        Purpose:
            Read a table schema.

        Args:
            table (str): Existing table name.

        Returns:
            list[dict[str, Any]]: Column metadata.
        """
        try:
            throw_if( 'table', table )
            self.table = table
            self.connect( )
            self.cursor = self.connection.execute( 'SELECT name FROM sqlite_master WHERE type=? AND name=?',
                ('table', self.table) )
            if self.cursor.fetchone( ) is None:
                raise ValueError( f'Unknown table: {self.table}' )
            self.cursor = self.connection.execute( 'PRAGMA table_info(' +
                '"' + self.table.replace( '"', '""' ) + '"' + ')' )
            return [ dict( row ) for row in self.cursor.fetchall( ) ]
        except Exception as e:
            e.add_note( 'fonky.loco.Data.get_schema' )
            raise

    def commit( self ) -> None:
        """Commit pending changes.

        Purpose:
            Commit pending changes.

        Returns:
            None: Commits the active transaction.
        """
        self.connect( )
        self.connection.commit( )

    def rollback( self ) -> None:
        """Rollback pending changes.

        Purpose:
            Rollback pending changes.

        Returns:
            None: Rolls back the transaction.
        """
        self.connect( )
        self.connection.rollback( )

    def backup_database( self, destination: str ) -> str:
        """Copy a consistent SQLite database snapshot.

        Purpose:
            Copy a consistent SQLite database snapshot.

        Args:
            destination (str): New SQLite database path.

        Returns:
            str: Backup path.
        """
        try:
            throw_if( 'destination', destination )
            self.destination = destination
            self.connect( )
            if Path( self.destination ).resolve( ) == Path( self.path ).resolve( ):
                raise ValueError( 'Destination must differ from the source database.' )
            target = sqlite3.connect( self.destination )
            try:
                self.connection.backup( target )
            finally:
                target.close( )
            return self.destination
        except Exception as e:
            e.add_note( 'fonky.loco.Data.backup_database' )
            raise

    def integrity_check( self ) -> list[ str ]:
        """Verify SQLite file integrity.

        Purpose:
            Verify SQLite file integrity.

        Returns:
            list[str]: SQLite integrity-check messages.
        """
        return [ row[ 'integrity_check' ] for row in self.execute_query( 'PRAGMA integrity_check' ) ]

    def read_dataframe( self, sql: str, parameters: Sequence[Any] = () ) -> Any:
        """Read a query into a pandas DataFrame.

        Purpose:
            Read a query into a pandas DataFrame.

        Args:
            sql (str): SQL query.
            parameters (Sequence[Any]): Positional parameters.

        Returns:
            pandas.DataFrame: Query results.
        """
        try:
            throw_if( 'sql', sql )
            self.sql = sql
            self.parameters = parameters
            import pandas as pd
            return pd.DataFrame( self.execute_query( self.sql, self.parameters ) )
        except Exception as e:
            e.add_note( 'fonky.loco.Data.read_dataframe' )
            raise



    def create_table( self, table: str, columns: dict[str, str] ) -> None:
        """Create a table with explicitly supported SQLite column types.

        Purpose:
            Create a table with explicitly supported SQLite column types.

        Args:
            table (str): New table name.
            columns (dict[str, str]): Column names and SQLite type declarations.

        Returns:
            None: Adds the table in the current transaction.
        """
        try:
            throw_if( 'table', table )
            throw_if( 'columns', columns )
            self.table = table
            self.columns = columns
            allowed = { 'INTEGER', 'REAL', 'TEXT', 'BLOB', 'NUMERIC',
                'INTEGER PRIMARY KEY', 'INTEGER PRIMARY KEY AUTOINCREMENT' }
            if any( declaration.upper( ) not in allowed for declaration in self.columns.values( ) ):
                raise ValueError( 'Unsupported SQLite column declaration.' )
            definitions = ', '.join( sql_identifier( name ) + ' ' + declaration
                for name, declaration in self.columns.items( ) )
            self.execute_nonquery( 'CREATE TABLE ' + sql_identifier( self.table ) + '(' + definitions + ')' )
        except Exception as e:
            e.add_note( 'fonky.loco.Data.create_table' )
            raise

    def drop_table( self, table: str, approved: bool = False ) -> None:
        """Drop a table when expressly approved.

        Purpose:
            Drop a table when expressly approved.

        Args:
            table (str): Table name.
            approved (bool): Explicit permission to drop the table.

        Returns:
            None: Drops the table in the current transaction.
        """
        try:
            throw_if( 'table', table )
            self.table = table
            self.approved = approved
            if not self.approved:
                raise PermissionError( 'Dropping a table requires approved=True.' )
            self.execute_nonquery( 'DROP TABLE ' + sql_identifier( self.table ) )
        except Exception as e:
            e.add_note( 'fonky.loco.Data.drop_table' )
            raise

    def insert_record( self, table: str, values: dict[str, Any] ) -> int:
        """Insert a parameterized SQLite record.

        Purpose:
            Insert a parameterized SQLite record.

        Args:
            table (str): Table name.
            values (dict[str, Any]): Column/value mapping.

        Returns:
            int: Affected record count.
        """
        try:
            throw_if( 'table', table )
            throw_if( 'values', values )
            self.table = table
            self.values = values
            names = ', '.join( sql_identifier( name ) for name in self.values )
            placeholders = ', '.join( '?' for name in self.values )
            statement = f'INSERT INTO {sql_identifier( self.table )} ({names}) VALUES ({placeholders})'
            return self.execute_nonquery( statement, tuple( self.values.values( ) ) )
        except Exception as e:
            e.add_note( 'fonky.loco.Data.insert_record' )
            raise

    def update_record( self, table: str, values: dict[str, Any],
        where: str, where_parameters: Sequence[Any] ) -> int:
        """Update selected records using bound values.

        Purpose:
            Update selected records using bound values.

        Args:
            table (str): Table name.
            values (dict[str, Any]): New field values.
            where (str): Trusted SQL WHERE clause supplied by the application.
            where_parameters (Sequence[Any]): Bound WHERE clause arguments.

        Returns:
            int: Number of affected records.
        """
        try:
            throw_if( 'table', table )
            throw_if( 'values', values )
            throw_if( 'where', where )
            self.table = table
            self.values = values
            self.where = where
            self.where_parameters = where_parameters
            changes = ', '.join( sql_identifier( name ) + '=?' for name in self.values )
            statement = f'UPDATE {sql_identifier( self.table )} SET {changes} WHERE {self.where}'
            return self.execute_nonquery( statement, tuple( self.values.values( ) ) + tuple( self.where_parameters ) )
        except Exception as e:
            e.add_note( 'fonky.loco.Data.update_record' )
            raise

    def delete_record( self, table: str, where: str,
        where_parameters: Sequence[Any], approved: bool = False ) -> int:
        """Delete selected SQLite records with explicit permission.

        Purpose:
            Delete selected SQLite records with explicit permission.

        Args:
            table (str): Table name.
            where (str): Trusted SQL WHERE clause.
            where_parameters (Sequence[Any]): Bound WHERE arguments.
            approved (bool): Explicit deletion approval.

        Returns:
            int: Number of deleted records.
        """
        try:
            throw_if( 'table', table )
            throw_if( 'where', where )
            self.table = table
            self.where = where
            self.where_parameters = where_parameters
            self.approved = approved
            if not self.approved:
                raise PermissionError( 'Record deletion requires approved=True.' )
            return self.execute_nonquery( f'DELETE FROM {sql_identifier( self.table )} WHERE {self.where}',
                self.where_parameters )
        except Exception as e:
            e.add_note( 'fonky.loco.Data.delete_record' )
            raise

    def write_dataframe( self, table: str, df_data: Any, if_exists: str = 'fail' ) -> int:
        """Write a DataFrame to SQLite.

        Purpose:
            Write a DataFrame to SQLite.

        Args:
            table (str): Destination table.
            df_data (pandas.DataFrame): Input records.
            if_exists (str): 'fail', 'append', or 'replace'.

        Returns:
            int: Number of rows inserted.
        """
        try:
            throw_if( 'table', table )
            throw_if( 'df_data', df_data )
            self.table = table
            self.df_data = df_data
            self.if_exists = if_exists
            if self.if_exists not in ('fail', 'append', 'replace'):
                raise ValueError( 'if_exists must be fail, append, or replace.' )
            if self.read_only:
                raise PermissionError( 'Database is read-only.' )
            self.connect( )
            return int( self.df_data.to_sql( self.table, self.connection, if_exists=self.if_exists, index=False ) )


    # ==========================================================================================
    # DETERMINISTIC CALCULATOR
    # ==========================================================================================
        except Exception as e:
            e.add_note( 'fonky.loco.Data.write_dataframe' )
            raise

class Calculator:
    """Safe arithmetic and optional Windows Calculator control.

    Purpose:
        Exposes the public Calculator automation API through local Python or supported
        desktop-application backends without requiring a cloud-model API key.

    Attributes:
        left (float): Left arithmetic operand or horizontal slide position.
        right (float): Right arithmetic operand.
        value (Any): Value being read, written, calculated, or searched.
        percent (float): Percentage applied to a base numeric value.
        expression (str): Arithmetic expression supplied for parsing.
        tree (ast.Expression): Parsed abstract syntax tree of an expression.
        node (ast.AST): Current AST node during restricted arithmetic evaluation.
        base (float): Base value for exponentiation.
        exponent (float): Power used in exponentiation.
        digits (int): Decimal places used for rounding.

    Methods:
        add(): Add two numbers.
        subtract(): Subtract two numbers.
        multiply(): Multiply two numbers.
        divide(): Divide two numbers.
        square_root(): Calculate a square root.
        percentage(): Calculate a percentage of a number.
        evaluate_expression(): Evaluate arithmetic syntax without executing Python code.
        evaluate_node(): Evaluate a restricted arithmetic node.
        launch_calculator(): Launch Windows Calculator.
        power(): Raise a base to a power.
        round_value(): Round a number to a specified precision.
    """


    # Public instance members (annotations only; values are assigned by operations).
    left: float  # Left arithmetic operand or horizontal slide position.
    right: float  # Right arithmetic operand.
    value: Any  # Value being read, written, calculated, or searched.
    percent: float  # Percentage applied to a base numeric value.
    expression: str  # Arithmetic expression supplied for parsing.
    tree: ast.Expression  # Parsed abstract syntax tree of an expression.
    node: ast.AST  # Current AST node during restricted arithmetic evaluation.
    base: float  # Base value for exponentiation.
    exponent: float  # Power used in exponentiation.
    digits: int  # Decimal places used for rounding.

    # Private, shared operator dispatch tables (not instance state).
    _binary: dict[type, Callable[[Any, Any], Any]] = {  # AST binary operations.
        ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod
    }
    _unary: dict[type, Callable[[Any], Any]] = { ast.UAdd: operator.pos, ast.USub: operator.neg }  # AST unary operations.

    def add( self, left: float, right: float ) -> float:
        """Add two numbers.

        Purpose:
            Add two numbers.

        Args:
            left (float): First number.
            right (float): Second number.

        Returns:
            float: Sum.
        """
        try:
            self.left = left
            self.right = right
            return self.left + self.right
        except Exception as e:
            e.add_note( 'fonky.loco.Calculator.add' )
            raise

    def subtract( self, left: float, right: float ) -> float:
        """Subtract two numbers.

        Purpose:
            Subtract two numbers.

        Args:
            left (float): First number.
            right (float): Second number.

        Returns:
            float: Difference.
        """
        try:
            self.left = left
            self.right = right
            return self.left - self.right
        except Exception as e:
            e.add_note( 'fonky.loco.Calculator.subtract' )
            raise

    def multiply( self, left: float, right: float ) -> float:
        """Multiply two numbers.

        Purpose:
            Multiply two numbers.

        Args:
            left (float): First number.
            right (float): Second number.

        Returns:
            float: Product.
        """
        try:
            self.left = left
            self.right = right
            return self.left * self.right
        except Exception as e:
            e.add_note( 'fonky.loco.Calculator.multiply' )
            raise

    def divide( self, left: float, right: float ) -> float:
        """Divide two numbers.

        Purpose:
            Divide two numbers.

        Args:
            left (float): Numerator.
            right (float): Denominator.

        Returns:
            float: Quotient.
        """
        try:
            self.left = left
            self.right = right
            return self.left / self.right
        except Exception as e:
            e.add_note( 'fonky.loco.Calculator.divide' )
            raise

    def square_root( self, value: float ) -> float:
        """Calculate a square root.

        Purpose:
            Calculate a square root.

        Args:
            value (float): Radicand.

        Returns:
            float: Square root.
        """
        try:
            self.value = value
            return math.sqrt( self.value )
        except Exception as e:
            e.add_note( 'fonky.loco.Calculator.square_root' )
            raise

    def percentage( self, value: float, percent: float ) -> float:
        """Calculate a percentage of a number.

        Purpose:
            Calculate a percentage of a number.

        Args:
            value (float): Base value.
            percent (float): Percentage.

        Returns:
            float: Percentage result.
        """
        try:
            self.value = value
            self.percent = percent
            return self.value * self.percent / 100
        except Exception as e:
            e.add_note( 'fonky.loco.Calculator.percentage' )
            raise

    def evaluate_expression( self, expression: str ) -> float:
        """Evaluate arithmetic syntax without executing Python code.

        Purpose:
            Evaluate arithmetic syntax without executing Python code.

        Args:
            expression (str): Numeric expression.

        Returns:
            float: Calculated value.

        Raises:
            ValueError: If the expression contains unsupported nodes.
        """
        try:
            throw_if( 'expression', expression )
            self.expression = expression
            self.tree = ast.parse( self.expression, mode='eval' )
            return float( self.evaluate_node( self.tree.body ) )
        except Exception as e:
            e.add_note( 'fonky.loco.Calculator.evaluate_expression' )
            raise

    def evaluate_node( self, node: ast.AST ) -> float:
        """Evaluate a restricted arithmetic node.

        Purpose:
            Evaluate a restricted arithmetic node.

        Args:
            node (ast.AST): Validated expression subtree.

        Returns:
            float: Calculated value.
        """
        try:
            self.node = node
            if isinstance( node, ast.Constant ) and type( node.value ) in (int, float):
                return node.value
            if isinstance( node, ast.UnaryOp ) and type( node.op ) in self._unary:
                return self._unary[ type( node.op ) ]( self.evaluate_node( node.operand ) )
            if isinstance( node, ast.BinOp ) and type( node.op ) in self._binary:
                left = self.evaluate_node( node.left )
                right = self.evaluate_node( node.right )
                if type( node.op ) is ast.Pow and abs( right ) > 100:
                    raise ValueError( 'Exponent is too large.' )
                return self._binary[ type( node.op ) ]( left, right )
            raise ValueError( 'Only numeric literals and arithmetic operators are permitted.' )
        except Exception as e:
            e.add_note( 'fonky.loco.Calculator.evaluate_node' )
            raise

    def launch_calculator( self ) -> subprocess.Popen:
        """Launch Windows Calculator.

        Purpose:
            Launch Windows Calculator.

        Returns:
            subprocess.Popen: Launched process handle.
        """
        require_windows( )
        return subprocess.Popen( [ 'calc.exe' ] )


# ==========================================================================================
# SYSTEM PROCESS EXECUTION
# ==========================================================================================


    def power( self, base: float, exponent: float ) -> float:
        """Raise a base to a power.

        Purpose:
            Raise a base to a power.

        Args:
            base (float): Base number.
            exponent (float): Exponent.

        Returns:
            float: Exponentiation result.
        """
        try:
            self.base = base
            self.exponent = exponent
            return math.pow( self.base, self.exponent )
        except Exception as e:
            e.add_note( 'fonky.loco.Calculator.power' )
            raise

    def round_value( self, value: float, digits: int = 2 ) -> float:
        """Round a number to a specified precision.

        Purpose:
            Round a number to a specified precision.

        Args:
            value (float): Input number.
            digits (int): Number of fractional digits.

        Returns:
            float: Rounded value.
        """
        try:
            self.value = value
            self.digits = digits
            return round( self.value, self.digits )
        except Exception as e:
            e.add_note( 'fonky.loco.Calculator.round_value' )
            raise


class Shell:
    """Subprocess execution with explicit permissions and timeouts.

    Purpose:
        Exposes the public Shell automation API through local Python or supported
        desktop-application backends without requiring a cloud-model API key.

    Attributes:
        allowed_commands (set[str]): Executable names explicitly permitted by the shell policy.
        command (list[str] | str): Executable command or program selected for execution.
        timeout (int): Maximum command execution duration in seconds.
        cwd (str): Working directory for the launched process.
        result (subprocess.CompletedProcess | Any): Most recent process execution result.
        script (str): PowerShell script submitted for execution.
        path (str | Path): Active local file, document, or database path.
        usage (Any): Disk usage statistics returned by shutil.
        process (subprocess.Popen): Process handle retained after launch.
        memory (Any): Memory statistics returned by psutil.
        interval (float): Duration for CPU utilization sampling.

    Methods:
        run_command(): Run a permitted executable without shell interpolation.
        run_powershell(): Run a PowerShell command only when explicitly allowed.
        which(): Find an executable on PATH.
        get_system_info(): Return non-sensitive platform information.
        get_disk_usage(): Read local disk capacity.
        launch_process(): Launch an allowlisted executable without a shell.
        list_processes(): Enumerate local processes using optional psutil.
        get_memory_usage(): Read system RAM utilization with psutil.
        get_cpu_usage(): Read CPU utilization with psutil.
    """


    # Public instance members (annotations only; values are assigned by operations).
    allowed_commands: set[str]  # Executable names explicitly permitted by the shell policy.
    command: list[str] | str  # Executable command or program selected for execution.
    timeout: int  # Maximum command execution duration in seconds.
    cwd: str  # Working directory for the launched process.
    result: subprocess.CompletedProcess | Any  # Most recent process execution result.
    script: str  # PowerShell script submitted for execution.
    path: str | Path  # Active local file, document, or database path.
    usage: Any  # Disk usage statistics returned by shutil.
    process: subprocess.Popen  # Process handle retained after launch.
    memory: Any  # Memory statistics returned by psutil.
    interval: float  # Duration for CPU utilization sampling.

    def __init__( self, allowed_commands: Sequence[str] = () ) -> None:
        """Configure an optional executable allowlist.

        Purpose:
            Configure an optional executable allowlist.

        Args:
            allowed_commands (Sequence[str]): Names of allowed executables; empty denies execution.

        Returns:
            None: Initializes execution policy.
        """
        self.allowed_commands = set( allowed_commands )

    def run_command( self, command: Sequence[str], timeout: int = 30,
        cwd: str = '' ) -> dict[ str, Any ]:
        """Run a permitted executable without shell interpolation.

        Purpose:
            Run a permitted executable without shell interpolation.

        Args:
            command (Sequence[str]): Executable and argument tokens.
            timeout (int): Timeout in seconds.
            cwd (str): Optional working directory.

        Returns:
            dict[str, Any]: Return code, stdout, and stderr.

        Raises:
            PermissionError: If the executable is not on the allowlist.
        """
        try:
            throw_if( 'command', command )
            self.command = list( command )
            self.timeout = timeout
            self.cwd = cwd
            if not self.allowed_commands or Path( self.command[ 0 ] ).name not in self.allowed_commands:
                raise PermissionError( 'Executable is not permitted by the shell policy.' )
            self.result = subprocess.run( self.command, capture_output=True, text=True,
                timeout=self.timeout, cwd=self.cwd or None, check=False, shell=False )
            return { 'returncode': self.result.returncode, 'stdout': self.result.stdout,
                'stderr': self.result.stderr }
        except Exception as e:
            e.add_note( 'fonky.loco.Shell.run_command' )
            raise

    def run_powershell( self, script: str, timeout: int = 30 ) -> dict[ str, Any ]:
        """Run a PowerShell command only when explicitly allowed.

        Purpose:
            Run a PowerShell command only when explicitly allowed.

        Args:
            script (str): PowerShell source.
            timeout (int): Maximum execution time.

        Returns:
            dict[str, Any]: Process result.
        """
        try:
            throw_if( 'script', script )
            self.script = script
            self.timeout = timeout
            require_windows( )
            return self.run_command( [ 'powershell.exe', '-NoProfile', '-NonInteractive',
                '-Command', self.script ], self.timeout )
        except Exception as e:
            e.add_note( 'fonky.loco.Shell.run_powershell' )
            raise

    def which( self, command: str ) -> str | None:
        """Find an executable on PATH.

        Purpose:
            Find an executable on PATH.

        Args:
            command (str): Executable name.

        Returns:
            str | None: Resolved path, if present.
        """
        try:
            throw_if( 'command', command )
            self.command = command
            return shutil.which( self.command )
        except Exception as e:
            e.add_note( 'fonky.loco.Shell.which' )
            raise

    def get_system_info( self ) -> dict[ str, str ]:
        """Return non-sensitive platform information.

        Purpose:
            Return non-sensitive platform information.

        Returns:
            dict[str, str]: OS, Python, and architecture.
        """
        return { 'platform': platform.system( ), 'release': platform.release( ),
            'architecture': platform.machine( ), 'python': platform.python_version( ) }

    def get_disk_usage( self, path: str = '.' ) -> dict[ str, int ]:
        """Read local disk capacity.

        Purpose:
            Read local disk capacity.

        Args:
            path (str): Filesystem path.

        Returns:
            dict[str, int]: Capacity, used, and free bytes.
        """
        self.path = path
        self.usage = shutil.disk_usage( self.path )
        return { 'total': self.usage.total, 'used': self.usage.used, 'free': self.usage.free }



    def launch_process( self, command: Sequence[str], cwd: str = '' ) -> int:
        """Launch an allowlisted executable without a shell.

        Purpose:
            Launch an allowlisted executable without a shell.

        Args:
            command (Sequence[str]): Executable and arguments.
            cwd (str): Optional working directory.

        Returns:
            int: New process identifier.
        """
        try:
            throw_if( 'command', command )
            self.command = list( command )
            self.cwd = cwd
            if not self.allowed_commands or Path( self.command[ 0 ] ).name not in self.allowed_commands:
                raise PermissionError( 'Executable is not permitted by the shell policy.' )
            self.process = subprocess.Popen( self.command, cwd=self.cwd or None, shell=False )
            return self.process.pid
        except Exception as e:
            e.add_note( 'fonky.loco.Shell.launch_process' )
            raise

    def list_processes( self ) -> list[dict[str, Any]]:
        """Enumerate local processes using optional psutil.

        Purpose:
            Enumerate local processes using optional psutil.

        Returns:
            list[dict[str, Any]]: Process IDs and names.
        """
        import psutil
        return [ { 'pid': process.info[ 'pid' ], 'name': process.info[ 'name' ] }
            for process in psutil.process_iter( attrs=['pid', 'name'] ) ]

    def get_memory_usage( self ) -> dict[str, Any]:
        """Read system RAM utilization with psutil.

        Purpose:
            Read system RAM utilization with psutil.

        Returns:
            dict[str, Any]: Available, total, and used bytes.
        """
        import psutil
        self.memory = psutil.virtual_memory( )
        return { 'total': self.memory.total, 'used': self.memory.used,
            'available': self.memory.available }

    def get_cpu_usage( self, interval: float = 0.1 ) -> float:
        """Read CPU utilization with psutil.

        Purpose:
            Read CPU utilization with psutil.

        Args:
            interval (float): Sampling interval in seconds.

        Returns:
            float: CPU utilization percentage.
        """
        self.interval = interval
        import psutil
        return float( psutil.cpu_percent( interval=self.interval ) )


# ==========================================================================================
# MICROSOFT OFFICE: EXCEL, WORD, POWERPOINT AND OUTLOOK
# ==========================================================================================

class Excel:
    """Workbook editing through openpyxl or dedicated Excel COM automation.

    Purpose:
        Exposes the public Excel automation API through local Python or supported
        desktop-application backends without requiring a cloud-model API key.

    Attributes:
        backend (str): Selected file-based, COM, or UI Automation backend.
        workbook (Any): Current open Excel workbook object.
        application (Any): COM application or UI automation object owned by this instance.
        path (str | Path): Active local file, document, or database path.
        name (str): Name of the current worksheet, table, or resource.
        sheet (str): Selected Excel worksheet name.
        cell (str): A1-style selected cell address.
        value (Any): Value being read, written, calculated, or searched.
        address (str): A1-style range address.
        start_cell (str): Top-left A1-style destination cell.
        rows (Any): Rows used for a database, spreadsheet, or presentation operation.
        save_changes (bool): Whether to save pending edits before closing.
        formula (str): Excel formula, including its leading equals sign.
        df_data (Any): Pandas DataFrame supplied to the active data operation.
        include_index (bool): Whether to include a DataFrame index in exported rows.
        approved (bool): Explicit authorization for a sensitive operation.
        pdf_path (str): Destination PDF file path.
        bold (bool): Whether text formatting uses bold weight.
        number_format (str): Excel display format string applied to cells.
        anchor (str): Excel destination cell for an embedded chart.

    Methods:
        open_workbook(): Open an existing workbook.
        create_workbook(): Create a new in-memory or COM workbook.
        list_worksheets(): List worksheet names.
        ensure_open(): Require an open workbook.
        create_worksheet(): Add a worksheet.
        read_cell(): Read a workbook cell.
        write_cell(): Assign a cell value or formula.
        read_range(): Read a rectangular range as rows.
        write_range(): Write a rectangular sequence of values.
        calculate(): Request recalculation with live Excel.
        save_workbook(): Save the workbook.
        close_workbook(): Close the workbook and any dedicated COM instance.
        set_formula(): Set an Excel formula.
        read_dataframe(): Read a rectangular Excel range into a DataFrame.
        write_dataframe(): Write a DataFrame and its column headers to a worksheet.
        refresh_queries(): Refresh all workbook data connections using Excel COM.
        run_macro(): Run an Excel macro only with explicit authorization.
        export_pdf(): Export an Excel workbook to PDF.
        create_table(): Create an Excel table over an existing rectangular range.
        format_range(): Apply bold text and number formats to a worksheet range.
        create_chart(): Create a column chart from headers and numeric rows.
    """


    # Public instance members (annotations only; values are assigned by operations).
    backend: str  # Selected file-based, COM, or UI Automation backend.
    workbook: Any  # Current open Excel workbook object.
    application: Any  # COM application or UI automation object owned by this instance.
    path: str | Path  # Active local file, document, or database path.
    name: str  # Name of the current worksheet, table, or resource.
    sheet: str  # Selected Excel worksheet name.
    cell: str  # A1-style selected cell address.
    value: Any  # Value being read, written, calculated, or searched.
    address: str  # A1-style range address.
    start_cell: str  # Top-left A1-style destination cell.
    rows: Any  # Rows used for a database, spreadsheet, or presentation operation.
    save_changes: bool  # Whether to save pending edits before closing.
    formula: str  # Excel formula, including its leading equals sign.
    df_data: Any  # Pandas DataFrame supplied to the active data operation.
    include_index: bool  # Whether to include a DataFrame index in exported rows.
    approved: bool  # Explicit authorization for a sensitive operation.
    pdf_path: str  # Destination PDF file path.
    bold: bool  # Whether text formatting uses bold weight.
    number_format: str  # Excel display format string applied to cells.
    anchor: str  # Excel destination cell for an embedded chart.

    def __init__( self, backend: str = 'file' ) -> None:
        """Choose the file or COM backend.

        Purpose:
            Choose the file or COM backend.

        Args:
            backend (str): Either 'file' or 'com'.

        Returns:
            None: Initializes workbook state.
        """
        self.backend = backend
        self.workbook = None
        self.application = None
        self.path = ''

    def open_workbook( self, path: str, visible: bool = False ) -> None:
        """Open an existing workbook.

        Purpose:
            Open an existing workbook.

        Args:
            path (str): Workbook file.
            visible (bool): Show the Excel COM instance.

        Returns:
            None: Holds an open workbook.
        """
        try:
            throw_if( 'path', path )
            self.path = str( Path( path ).resolve( ) )
            if self.backend == 'com':
                self.application = office_application( 'Excel.Application', visible )
                self.workbook = self.application.Workbooks.Open( self.path )
            else:
                from openpyxl import load_workbook
                self.workbook = load_workbook( self.path )
        except Exception as e:
            e.add_note( 'fonky.loco.Excel.open_workbook' )
            raise

    def create_workbook( self, visible: bool = False ) -> None:
        """Create a new in-memory or COM workbook.

        Purpose:
            Create a new in-memory or COM workbook.

        Args:
            visible (bool): Show the Excel COM instance.

        Returns:
            None: Holds the new workbook.
        """
        if self.backend == 'com':
            self.application = office_application( 'Excel.Application', visible )
            self.workbook = self.application.Workbooks.Add( )
        else:
            from openpyxl import Workbook
            self.workbook = Workbook( )

    def list_worksheets( self ) -> list[str]:
        """List worksheet names.

        Purpose:
            List worksheet names.

        Returns:
            list[str]: Worksheet names.
        """
        self.ensure_open( )
        if self.backend == 'com':
            return [ sheet.Name for sheet in self.workbook.Worksheets ]
        return list( self.workbook.sheetnames )

    def ensure_open( self ) -> None:
        """Require an open workbook.

        Purpose:
            Require an open workbook.

        Raises:
            RuntimeError: If no workbook is open.
        """
        if self.workbook is None:
            raise RuntimeError( 'No workbook is open.' )

    def create_worksheet( self, name: str ) -> None:
        """Add a worksheet.

        Purpose:
            Add a worksheet.

        Args:
            name (str): New sheet name.

        Returns:
            None: Adds a sheet.
        """
        try:
            throw_if( 'name', name )
            self.name = name
            self.ensure_open( )
            if self.backend == 'com':
                self.workbook.Worksheets.Add( ).Name = self.name
            else:
                self.workbook.create_sheet( self.name )
        except Exception as e:
            e.add_note( 'fonky.loco.Excel.create_worksheet' )
            raise

    def read_cell( self, sheet: str, cell: str ) -> Any:
        """Read a workbook cell.

        Purpose:
            Read a workbook cell.

        Args:
            sheet (str): Worksheet name.
            cell (str): A1 cell address.

        Returns:
            Any: Cell value.
        """
        try:
            throw_if( 'sheet', sheet )
            throw_if( 'cell', cell )
            self.sheet = sheet
            self.cell = cell
            self.ensure_open( )
            if self.backend == 'com':
                return self.workbook.Worksheets( self.sheet ).Range( self.cell ).Value
            return self.workbook[ self.sheet ][ self.cell ].value
        except Exception as e:
            e.add_note( 'fonky.loco.Excel.read_cell' )
            raise

    def write_cell( self, sheet: str, cell: str, value: Any ) -> None:
        """Assign a cell value or formula.

        Purpose:
            Assign a cell value or formula.

        Args:
            sheet (str): Worksheet name.
            cell (str): A1 cell address.
            value (Any): New cell value.

        Returns:
            None: Updates the cell.
        """
        try:
            throw_if( 'sheet', sheet )
            throw_if( 'cell', cell )
            self.sheet = sheet
            self.cell = cell
            self.value = value
            self.ensure_open( )
            if self.backend == 'com':
                self.workbook.Worksheets( self.sheet ).Range( self.cell ).Value = self.value
            else:
                self.workbook[ self.sheet ][ self.cell ] = self.value
        except Exception as e:
            e.add_note( 'fonky.loco.Excel.write_cell' )
            raise

    def read_range( self, sheet: str, address: str ) -> list[list[Any]]:
        """Read a rectangular range as rows.

        Purpose:
            Read a rectangular range as rows.

        Args:
            sheet (str): Worksheet name.
            address (str): A1-style rectangular address.

        Returns:
            list[list[Any]]: Cell values organized by row.
        """
        try:
            throw_if( 'sheet', sheet )
            throw_if( 'address', address )
            self.sheet = sheet
            self.address = address
            self.ensure_open( )
            if self.backend == 'com':
                values = self.workbook.Worksheets( self.sheet ).Range( self.address ).Value
                if not isinstance( values, tuple ):
                    return [ [ values ] ]
                if values and not isinstance( values[ 0 ], tuple ):
                    return [ list( values ) ]
                return [ list( row ) for row in values ]
            worksheet = self.workbook[ self.sheet ]
            return [ [ cell.value for cell in row ] for row in worksheet[ self.address ] ] if ':' in self.address else [
                [ worksheet[ self.address ].value ] ]
        except Exception as e:
            e.add_note( 'fonky.loco.Excel.read_range' )
            raise

    def write_range( self, sheet: str, start_cell: str, rows: Sequence[Sequence[Any]] ) -> None:
        """Write a rectangular sequence of values.

        Purpose:
            Write a rectangular sequence of values.

        Args:
            sheet (str): Worksheet name.
            start_cell (str): Top-left A1 address.
            rows (Sequence[Sequence[Any]]): Rectangular values.

        Returns:
            None: Updates the selected range.
        """
        try:
            throw_if( 'sheet', sheet )
            throw_if( 'start_cell', start_cell )
            throw_if( 'rows', rows )
            self.sheet = sheet
            self.start_cell = start_cell
            self.rows = [ list( row ) for row in rows ]
            self.ensure_open( )
            width = len( self.rows[ 0 ] )
            if width == 0 or any( len( row ) != width for row in self.rows ):
                raise ValueError( 'rows must form a nonempty rectangle.' )
            if self.backend == 'com':
                start = self.workbook.Worksheets( self.sheet ).Range( self.start_cell )
                destination = start.Resize( len( self.rows ), width )
                destination.Value = tuple( tuple( row ) for row in self.rows )
            else:
                from openpyxl.utils.cell import coordinate_from_string, column_index_from_string
                col, row = coordinate_from_string( self.start_cell )
                start_column = column_index_from_string( col )
                worksheet = self.workbook[ self.sheet ]
                for i, values in enumerate( self.rows ):
                    for j, value in enumerate( values ):
                        worksheet.cell( row=row + i, column=start_column + j, value=value )
        except Exception as e:
            e.add_note( 'fonky.loco.Excel.write_range' )
            raise

    def calculate( self ) -> None:
        """Request recalculation with live Excel.

        Purpose:
            Request recalculation with live Excel.

        Raises:
            NotImplementedError: If using the file backend.
        """
        self.ensure_open( )
        if self.backend != 'com':
            raise NotImplementedError( 'Recalculation requires the COM backend.' )
        self.application.CalculateFull( )

    def save_workbook( self, path: str = '' ) -> str:
        """Save the workbook.

        Purpose:
            Save the workbook.

        Args:
            path (str): Optional destination.

        Returns:
            str: Saved file path.
        """
        self.ensure_open( )
        self.path = str( Path( path or self.path ).resolve( ) ) if path or self.path else ''
        throw_if( 'path', self.path )
        if self.backend == 'com':
            self.workbook.SaveAs( self.path ) if path else self.workbook.Save( )
        else:
            self.workbook.save( self.path )
        return self.path

    def close_workbook( self, save_changes: bool = False ) -> None:
        """Close the workbook and any dedicated COM instance.

        Purpose:
            Close the workbook and any dedicated COM instance.

        Args:
            save_changes (bool): Commit modifications before closing.

        Returns:
            None: Clears instance references.
        """
        self.ensure_open( )
        self.save_changes = save_changes
        try:
            if self.backend == 'com':
                self.workbook.Close( SaveChanges=self.save_changes )
            elif self.save_changes:
                self.save_workbook( )
        finally:
            self.workbook = None
            if self.application is not None:
                self.application.Quit( )
                self.application = None



    def set_formula( self, sheet: str, cell: str, formula: str ) -> None:
        """Set an Excel formula.

        Purpose:
            Set an Excel formula.

        Args:
            sheet (str): Worksheet name.
            cell (str): Destination cell.
            formula (str): Formula beginning with '='.

        Returns:
            None: Updates the selected cell.
        """
        try:
            throw_if( 'formula', formula )
            throw_if( 'sheet', sheet )
            throw_if( 'cell', cell )
            self.sheet = sheet
            self.cell = cell
            self.formula = formula
            if not self.formula.startswith( '=' ):
                raise ValueError( 'Excel formula must start with =.' )
            self.write_cell( self.sheet, self.cell, self.formula )
        except Exception as e:
            e.add_note( 'fonky.loco.Excel.set_formula' )
            raise

    def read_dataframe( self, sheet: str, address: str ) -> Any:
        """Read a rectangular Excel range into a DataFrame.

        Purpose:
            Read a rectangular Excel range into a DataFrame.

        Args:
            sheet (str): Worksheet name.
            address (str): Range including its header row.

        Returns:
            pandas.DataFrame: Values with the first row as headers.
        """
        try:
            self.sheet = sheet
            self.address = address
            import pandas as pd
            self.rows = self.read_range( self.sheet, self.address )
            return pd.DataFrame( self.rows[ 1: ], columns=self.rows[ 0 ] ) if self.rows else pd.DataFrame( )
        except Exception as e:
            e.add_note( 'fonky.loco.Excel.read_dataframe' )
            raise

    def write_dataframe( self, sheet: str, start_cell: str, df_data: Any,
        include_index: bool = False ) -> None:
        """Write a DataFrame and its column headers to a worksheet.

        Purpose:
            Write a DataFrame and its column headers to a worksheet.

        Args:
            sheet (str): Worksheet name.
            start_cell (str): Top-left cell address.
            df_data (pandas.DataFrame): Input DataFrame.
            include_index (bool): Include the DataFrame index.

        Returns:
            None: Updates the selected range.
        """
        try:
            throw_if( 'df_data', df_data )
            self.df_data = df_data
            self.include_index = include_index
            self.rows = [ list( self.df_data.columns ) ]
            self.rows.extend( self.df_data.itertuples( index=False, name=None ) )
            if self.include_index:
                self.rows[ 0 ].insert( 0, str( self.df_data.index.name or 'Index') )
                for i, index in enumerate( self.df_data.index ):
                    self.rows[ i + 1 ] = [ index, *self.rows[ i + 1 ] ]
            self.write_range( sheet, start_cell, self.rows )
        except Exception as e:
            e.add_note( 'fonky.loco.Excel.write_dataframe' )
            raise

    def refresh_queries( self ) -> None:
        """Refresh all workbook data connections using Excel COM.

        Purpose:
            Refresh all workbook data connections using Excel COM.

        Returns:
            None: Requests refresh.

        Raises:
            NotImplementedError: If the COM backend is not selected.
        """
        self.ensure_open( )
        if self.backend != 'com':
            raise NotImplementedError( 'Query refresh requires Microsoft Excel COM.' )
        self.workbook.RefreshAll( )
        self.application.CalculateUntilAsyncQueriesDone( )

    def run_macro( self, name: str, approved: bool = False ) -> Any:
        """Run an Excel macro only with explicit authorization.

        Purpose:
            Run an Excel macro only with explicit authorization.

        Args:
            name (str): Qualified VBA macro name.
            approved (bool): Authorization to execute VBA.

        Returns:
            Any: Macro result.
        """
        try:
            throw_if( 'name', name )
            self.name = name
            self.approved = approved
            self.ensure_open( )
            if not self.approved:
                raise PermissionError( 'Macro execution requires approved=True.' )
            if self.backend != 'com':
                raise NotImplementedError( 'VBA execution requires Excel COM.' )
            return self.application.Run( self.name )
        except Exception as e:
            e.add_note( 'fonky.loco.Excel.run_macro' )
            raise

    def export_pdf( self, path: str ) -> str:
        """Export an Excel workbook to PDF.

        Purpose:
            Export an Excel workbook to PDF.

        Args:
            path (str): PDF destination.

        Returns:
            str: PDF path.
        """
        try:
            throw_if( 'path', path )
            self.pdf_path = str( Path( path ).resolve( ) )
            self.ensure_open( )
            if self.backend != 'com':
                raise NotImplementedError( 'PDF export requires Excel COM.' )
            self.workbook.ExportAsFixedFormat( 0, self.pdf_path )
            return self.pdf_path
        except Exception as e:
            e.add_note( 'fonky.loco.Excel.export_pdf' )
            raise



    def create_table( self, sheet: str, address: str, name: str ) -> None:
        """Create an Excel table over an existing rectangular range.

        Purpose:
            Create an Excel table over an existing rectangular range.

        Args:
            sheet (str): Worksheet name.
            address (str): A1-style source range.
            name (str): Unique table name.

        Returns:
            None: Adds the Excel table.
        """
        try:
            throw_if( 'sheet', sheet )
            throw_if( 'address', address )
            throw_if( 'name', name )
            self.sheet = sheet
            self.address = address
            self.name = name
            self.ensure_open( )
            if self.backend == 'com':
                worksheet = self.workbook.Worksheets( self.sheet )
                table = worksheet.ListObjects.Add( 1, worksheet.Range( self.address ), None, 1 )
                table.Name = self.name
            else:
                from openpyxl.worksheet.table import Table, TableStyleInfo
                table = Table( displayName=self.name, ref=self.address )
                table.tableStyleInfo = TableStyleInfo( name='TableStyleMedium2',
                    showFirstColumn=False, showLastColumn=False, showRowStripes=True,
                    showColumnStripes=False )
                self.workbook[ self.sheet ].add_table( table )
        except Exception as e:
            e.add_note( 'fonky.loco.Excel.create_table' )
            raise

    def format_range( self, sheet: str, address: str, bold: bool = False,
        number_format: str = 'General' ) -> None:
        """Apply bold text and number formats to a worksheet range.

        Purpose:
            Apply bold text and number formats to a worksheet range.

        Args:
            sheet (str): Worksheet name.
            address (str): A1-style range.
            bold (bool): Whether text should be bold.
            number_format (str): Excel number-format string.

        Returns:
            None: Formats the selected range.
        """
        try:
            throw_if( 'sheet', sheet )
            throw_if( 'address', address )
            self.sheet = sheet
            self.address = address
            self.bold = bold
            self.number_format = number_format
            self.ensure_open( )
            if self.backend == 'com':
                target = self.workbook.Worksheets( self.sheet ).Range( self.address )
                target.Font.Bold = self.bold
                target.NumberFormat = self.number_format
            else:
                from openpyxl.styles import Font
                worksheet = self.workbook[ self.sheet ]
                selected = worksheet[ self.address ]
                rows = selected if isinstance( selected, tuple ) else ((selected,),)
                if rows and not isinstance( rows[ 0 ], tuple ):
                    rows = (rows,)
                for row in rows:
                    for cell in row:
                        existing = cell.font.copy( )
                        existing.bold = self.bold
                        cell.font = existing
                        cell.number_format = self.number_format
        except Exception as e:
            e.add_note( 'fonky.loco.Excel.format_range' )
            raise

    def create_chart( self, sheet: str, address: str, anchor: str = 'E2' ) -> None:
        """Create a column chart from headers and numeric rows.

        Purpose:
            Create a column chart from headers and numeric rows.

        Args:
            sheet (str): Worksheet name.
            address (str): Rectangular chart-data range.
            anchor (str): Destination cell for file-based charts.

        Returns:
            None: Adds a column chart.
        """
        try:
            throw_if( 'sheet', sheet )
            throw_if( 'address', address )
            self.sheet = sheet
            self.address = address
            self.anchor = anchor
            self.ensure_open( )
            if self.backend == 'com':
                worksheet = self.workbook.Worksheets( self.sheet )
                chart = worksheet.ChartObjects( ).Add( 300, 40, 450, 250 ).Chart
                chart.ChartType = 51
                chart.SetSourceData( worksheet.Range( self.address ) )
            else:
                from openpyxl.chart import BarChart, Reference
                from openpyxl.utils.cell import range_boundaries
                worksheet = self.workbook[ self.sheet ]
                min_col, min_row, max_col, max_row = range_boundaries( self.address )
                if max_col - min_col < 1 or max_row - min_row < 1:
                    raise ValueError( 'Chart data requires headers and at least one data row.' )
                chart = BarChart( )
                values = Reference( worksheet, min_col=min_col + 1, max_col=max_col,
                    min_row=min_row, max_row=max_row )
                labels = Reference( worksheet, min_col=min_col, min_row=min_row + 1, max_row=max_row )
                chart.add_data( values, titles_from_data=True )
                chart.set_categories( labels )
                worksheet.add_chart( chart, self.anchor )
        except Exception as e:
            e.add_note( 'fonky.loco.Excel.create_chart' )
            raise


class Word:
    """Word document editing through python-docx or dedicated Word COM automation.

    Purpose:
        Exposes the public Word automation API through local Python or supported
        desktop-application backends without requiring a cloud-model API key.

    Attributes:
        backend (str): Selected file-based, COM, or UI Automation backend.
        document (Any): Open Word document object.
        application (Any): COM application or UI automation object owned by this instance.
        path (str | Path): Active local file, document, or database path.
        text (str): Text used in the current document or UI operation.
        style (str): Word paragraph style identifier.
        old (str): Text to search for replacement.
        new (str): Replacement text.
        path_pdf (str): Word PDF export destination.
        save_changes (bool): Whether to save pending edits before closing.
        rows (Any): Rows used for a database, spreadsheet, or presentation operation.
        columns (Any): Column definitions or table dimensions.
        image_path (str): Source path of an image inserted into an Office document.
        value (Any): Value being read, written, calculated, or searched.

    Methods:
        open_document(): Open a Word document.
        create_document(): Create a new Word document.
        read_document(): Read paragraph text from the document.
        append_text(): Append a paragraph.
        replace_text(): Replace text in normal paragraphs.
        save_document(): Save a Word document.
        export_pdf(): Export through installed Microsoft Word.
        close_document(): Close the document and the dedicated COM application.
        insert_table(): Insert a Word table.
        insert_image(): Insert a local image into Word.
        find_text(): Check whether the document contains text.
        set_header(): Set the primary document header.
        set_footer(): Set the primary document footer.
    """


    # Public instance members (annotations only; values are assigned by operations).
    backend: str  # Selected file-based, COM, or UI Automation backend.
    document: Any  # Open Word document object.
    application: Any  # COM application or UI automation object owned by this instance.
    path: str | Path  # Active local file, document, or database path.
    text: str  # Text used in the current document or UI operation.
    style: str  # Word paragraph style identifier.
    old: str  # Text to search for replacement.
    new: str  # Replacement text.
    path_pdf: str  # Word PDF export destination.
    save_changes: bool  # Whether to save pending edits before closing.
    rows: Any  # Rows used for a database, spreadsheet, or presentation operation.
    columns: Any  # Column definitions or table dimensions.
    image_path: str  # Source path of an image inserted into an Office document.
    value: Any  # Value being read, written, calculated, or searched.
    def __init__( self, backend: str = 'file' ) -> None:
        """Select the Word backend.

        Purpose:
            Select the Word backend.

        Args:
            backend (str): 'file' or 'com'.

        Returns:
            None: Initializes document state.
        """
        self.backend = backend
        self.document = None
        self.application = None
        self.path = ''

    def open_document( self, path: str, visible: bool = False ) -> None:
        """Open a Word document.

        Purpose:
            Open a Word document.

        Args:
            path (str): Document path.
            visible (bool): Show the COM application.

        Returns:
            None: Holds the opened document.
        """
        try:
            throw_if( 'path', path )
            self.path = str( Path( path ).resolve( ) )
            if self.backend == 'com':
                self.application = office_application( 'Word.Application', visible )
                self.document = self.application.Documents.Open( self.path )
            else:
                from docx import Document
                self.document = Document( self.path )
        except Exception as e:
            e.add_note( 'fonky.loco.Word.open_document' )
            raise

    def create_document( self, visible: bool = False ) -> None:
        """Create a new Word document.

        Purpose:
            Create a new Word document.

        Args:
            visible (bool): Show the COM application.

        Returns:
            None: Holds the new document.
        """
        if self.backend == 'com':
            self.application = office_application( 'Word.Application', visible )
            self.document = self.application.Documents.Add( )
        else:
            from docx import Document
            self.document = Document( )

    def read_document( self ) -> str:
        """Read paragraph text from the document.

        Purpose:
            Read paragraph text from the document.

        Returns:
            str: Document text.
        """
        if self.document is None:
            raise RuntimeError( 'No document is open.' )
        if self.backend == 'com':
            return str( self.document.Content.Text )
        return '\n'.join( paragraph.text for paragraph in self.document.paragraphs )

    def append_text( self, text: str, style: str = '' ) -> None:
        """Append a paragraph.

        Purpose:
            Append a paragraph.

        Args:
            text (str): Paragraph text.
            style (str): Optional paragraph style.

        Returns:
            None: Appends text.
        """
        try:
            throw_if( 'text', text )
            self.text = text
            self.style = style
            if self.document is None:
                raise RuntimeError( 'No document is open.' )
            if self.backend == 'com':
                self.document.Content.InsertAfter( self.text + '\n' )
            else:
                self.document.add_paragraph( self.text, style=self.style or None )
        except Exception as e:
            e.add_note( 'fonky.loco.Word.append_text' )
            raise

    def replace_text( self, old: str, new: str ) -> int:
        """Replace text in normal paragraphs.

        Purpose:
            Replace text in normal paragraphs.

        Args:
            old (str): Text to find.
            new (str): Replacement text.

        Returns:
            int: Number of replacements.
        """
        try:
            throw_if( 'old', old )
            self.old = old
            self.new = new
            if self.document is None:
                raise RuntimeError( 'No document is open.' )
            if self.backend == 'com':
                finder = self.document.Content.Find
                finder.ClearFormatting( )
                finder.Replacement.ClearFormatting( )
                finder.Execute( FindText=self.old, ReplaceWith=self.new, Replace=2 )
                return -1
            count = 0
            for paragraph in self.document.paragraphs:
                for run in paragraph.runs:
                    count += run.text.count( self.old )
                    if self.old in run.text:
                        run.text = run.text.replace( self.old, self.new )
            return count
        except Exception as e:
            e.add_note( 'fonky.loco.Word.replace_text' )
            raise

    def save_document( self, path: str = '' ) -> str:
        """Save a Word document.

        Purpose:
            Save a Word document.

        Args:
            path (str): Optional output path.

        Returns:
            str: Saved path.
        """
        if self.document is None:
            raise RuntimeError( 'No document is open.' )
        self.path = str( Path( path or self.path ).resolve( ) ) if path or self.path else ''
        throw_if( 'path', self.path )
        if self.backend == 'com':
            self.document.SaveAs2( self.path ) if path else self.document.Save( )
        else:
            self.document.save( self.path )
        return self.path

    def export_pdf( self, path: str ) -> str:
        """Export through installed Microsoft Word.

        Purpose:
            Export through installed Microsoft Word.

        Args:
            path (str): PDF output path.

        Returns:
            str: PDF path.

        Raises:
            NotImplementedError: If COM is not selected.
        """
        try:
            throw_if( 'path', path )
            self.path_pdf = str( Path( path ).resolve( ) )
            if self.backend != 'com' or self.document is None:
                raise NotImplementedError( 'PDF export requires an open COM Word document.' )
            self.document.ExportAsFixedFormat( self.path_pdf, 17 )
            return self.path_pdf
        except Exception as e:
            e.add_note( 'fonky.loco.Word.export_pdf' )
            raise

    def close_document( self, save_changes: bool = False ) -> None:
        """Close the document and the dedicated COM application.

        Purpose:
            Close the document and the dedicated COM application.

        Args:
            save_changes (bool): Save modifications before close.

        Returns:
            None: Releases document state.
        """
        if self.document is None:
            return
        self.save_changes = save_changes
        try:
            if self.backend == 'com':
                self.document.Close( SaveChanges=self.save_changes )
            elif self.save_changes:
                self.save_document( )
        finally:
            self.document = None
            if self.application is not None:
                self.application.Quit( )
                self.application = None



    def insert_table( self, rows: int, columns: int ) -> None:
        """Insert a Word table.

        Purpose:
            Insert a Word table.

        Args:
            rows (int): Number of rows.
            columns (int): Number of columns.

        Returns:
            None: Inserts a table.
        """
        try:
            self.rows = rows
            self.columns = columns
            if self.rows < 1 or self.columns < 1:
                raise ValueError( 'Table dimensions must be positive.' )
            if self.document is None:
                raise RuntimeError( 'No document is open.' )
            if self.backend == 'com':
                position = self.document.Content
                position.Collapse( 0 )
                self.document.Tables.Add( position, self.rows, self.columns )
            else:
                self.document.add_table( rows=self.rows, cols=self.columns )
        except Exception as e:
            e.add_note( 'fonky.loco.Word.insert_table' )
            raise

    def insert_image( self, path: str ) -> None:
        """Insert a local image into Word.

        Purpose:
            Insert a local image into Word.

        Args:
            path (str): Image file path.

        Returns:
            None: Adds the image.
        """
        try:
            throw_if( 'path', path )
            self.image_path = str( Path( path ).resolve( ) )
            if self.document is None:
                raise RuntimeError( 'No document is open.' )
            if self.backend == 'com':
                self.document.InlineShapes.AddPicture( self.image_path )
            else:
                self.document.add_picture( self.image_path )
        except Exception as e:
            e.add_note( 'fonky.loco.Word.insert_image' )
            raise



    def find_text( self, value: str ) -> bool:
        """Check whether the document contains text.

        Purpose:
            Check whether the document contains text.

        Args:
            value (str): Target substring.

        Returns:
            bool: True when a match is found.
        """
        try:
            throw_if( 'value', value )
            self.value = value
            return self.value in self.read_document( )
        except Exception as e:
            e.add_note( 'fonky.loco.Word.find_text' )
            raise

    def set_header( self, text: str ) -> None:
        """Set the primary document header.

        Purpose:
            Set the primary document header.

        Args:
            text (str): Header text.

        Returns:
            None: Updates the header.
        """
        try:
            throw_if( 'text', text )
            self.text = text
            if self.document is None:
                raise RuntimeError( 'No document is open.' )
            if self.backend == 'com':
                self.document.Sections( 1 ).Headers( 1 ).Range.Text = self.text
            else:
                section = self.document.sections[ 0 ]
                paragraph = section.header.paragraphs[ 0 ]
                paragraph.text = self.text
        except Exception as e:
            e.add_note( 'fonky.loco.Word.set_header' )
            raise

    def set_footer( self, text: str ) -> None:
        """Set the primary document footer.

        Purpose:
            Set the primary document footer.

        Args:
            text (str): Footer text.

        Returns:
            None: Updates the footer.
        """
        try:
            throw_if( 'text', text )
            self.text = text
            if self.document is None:
                raise RuntimeError( 'No document is open.' )
            if self.backend == 'com':
                self.document.Sections( 1 ).Footers( 1 ).Range.Text = self.text
            else:
                section = self.document.sections[ 0 ]
                paragraph = section.footer.paragraphs[ 0 ]
                paragraph.text = self.text
        except Exception as e:
            e.add_note( 'fonky.loco.Word.set_footer' )
            raise


class PowerPoint:
    """PowerPoint slide creation with python-pptx or dedicated COM automation.

    Purpose:
        Exposes the public PowerPoint automation API through local Python or supported
        desktop-application backends without requiring a cloud-model API key.

    Attributes:
        backend (str): Selected file-based, COM, or UI Automation backend.
        presentation (Any): Current open PowerPoint presentation object.
        application (Any): COM application or UI automation object owned by this instance.
        path (str | Path): Active local file, document, or database path.
        title (str): Current slide, window, or control title.
        layout (int): PowerPoint slide layout index.
        pdf_path (str): Destination PDF file path.
        slide_number (int): One-based slide index.
        text (str): Text used in the current document or UI operation.
        bounds (tuple[float, float, float, float]): Text-box position and dimensions measured in inches.
        image_path (str): Source path of an image inserted into an Office document.
        left (float): Left arithmetic operand or horizontal slide position.
        top (float): Vertical slide coordinate measured in inches.
        width (float): Object or window width.
        rows (Any): Rows used for a database, spreadsheet, or presentation operation.
        height (float): Object or window height.

    Methods:
        create_presentation(): Create a presentation.
        open_presentation(): Open an existing presentation.
        list_slides(): List one-based slide numbers.
        add_slide(): Add a titled slide.
        save_presentation(): Save the presentation.
        export_pdf(): Export a presentation to PDF with PowerPoint COM.
        close_presentation(): Release an open presentation and dedicated COM application.
        add_text(): Add positioned text to a slide, using inches for coordinates.
        add_image(): Insert a local image on a slide.
        delete_slide(): Delete a presentation slide.
        add_table(): Insert a rectangular table on a slide.
    """


    # Public instance members (annotations only; values are assigned by operations).
    backend: str  # Selected file-based, COM, or UI Automation backend.
    presentation: Any  # Current open PowerPoint presentation object.
    application: Any  # COM application or UI automation object owned by this instance.
    path: str | Path  # Active local file, document, or database path.
    title: str  # Current slide, window, or control title.
    layout: int  # PowerPoint slide layout index.
    pdf_path: str  # Destination PDF file path.
    slide_number: int  # One-based slide index.
    text: str  # Text used in the current document or UI operation.
    bounds: tuple[float, float, float, float]  # Text-box position and dimensions measured in inches.
    image_path: str  # Source path of an image inserted into an Office document.
    left: float  # Left arithmetic operand or horizontal slide position.
    top: float  # Vertical slide coordinate measured in inches.
    width: float  # Object or window width.
    rows: Any  # Rows used for a database, spreadsheet, or presentation operation.
    height: float  # Object or window height.
    def __init__( self, backend: str = 'file' ) -> None:
        """Select the presentation backend.

        Purpose:
            Select the presentation backend.

        Args:
            backend (str): 'file' or 'com'.

        Returns:
            None: Initializes presentation state.
        """
        self.backend = backend
        self.presentation = None
        self.application = None
        self.path = ''

    def create_presentation( self, visible: bool = False ) -> None:
        """Create a presentation.

        Purpose:
            Create a presentation.

        Args:
            visible (bool): Show the PowerPoint application.

        Returns:
            None: Holds the presentation.
        """
        if self.backend == 'com':
            self.application = office_application( 'PowerPoint.Application', visible )
            self.presentation = self.application.Presentations.Add( )
        else:
            from pptx import Presentation
            self.presentation = Presentation( )

    def open_presentation( self, path: str, visible: bool = False ) -> None:
        """Open an existing presentation.

        Purpose:
            Open an existing presentation.

        Args:
            path (str): Source path.
            visible (bool): Show the application.

        Returns:
            None: Holds the presentation.
        """
        try:
            throw_if( 'path', path )
            self.path = str( Path( path ).resolve( ) )
            if self.backend == 'com':
                self.application = office_application( 'PowerPoint.Application', visible )
                self.presentation = self.application.Presentations.Open( self.path )
            else:
                from pptx import Presentation
                self.presentation = Presentation( self.path )
        except Exception as e:
            e.add_note( 'fonky.loco.PowerPoint.open_presentation' )
            raise

    def list_slides( self ) -> list[int]:
        """List one-based slide numbers.

        Purpose:
            List one-based slide numbers.

        Returns:
            list[int]: Slide numbers.
        """
        if self.presentation is None:
            raise RuntimeError( 'No presentation is open.' )
        count = self.presentation.Slides.Count if self.backend == 'com' else len( self.presentation.slides )
        return list( range( 1, count + 1 ) )

    def add_slide( self, title: str, layout: int = 1 ) -> int:
        """Add a titled slide.

        Purpose:
            Add a titled slide.

        Args:
            title (str): Slide title.
            layout (int): Layout index.

        Returns:
            int: One-based slide number.
        """
        try:
            throw_if( 'title', title )
            self.title = title
            self.layout = layout
            if self.presentation is None:
                raise RuntimeError( 'No presentation is open.' )
            if self.backend == 'com':
                slide = self.presentation.Slides.Add( self.presentation.Slides.Count + 1, self.layout + 1 )
                slide.Shapes.Title.TextFrame.TextRange.Text = self.title
            else:
                slide = self.presentation.slides.add_slide( self.presentation.slide_layouts[ self.layout ] )
                slide.shapes.title.text = self.title
            return len( self.list_slides( ) )
        except Exception as e:
            e.add_note( 'fonky.loco.PowerPoint.add_slide' )
            raise

    def save_presentation( self, path: str = '' ) -> str:
        """Save the presentation.

        Purpose:
            Save the presentation.

        Args:
            path (str): Optional destination path.

        Returns:
            str: Saved path.
        """
        if self.presentation is None:
            raise RuntimeError( 'No presentation is open.' )
        self.path = str( Path( path or self.path ).resolve( ) ) if path or self.path else ''
        throw_if( 'path', self.path )
        if self.backend == 'com':
            self.presentation.SaveAs( self.path ) if path else self.presentation.Save( )
        else:
            self.presentation.save( self.path )
        return self.path

    def export_pdf( self, path: str ) -> str:
        """Export a presentation to PDF with PowerPoint COM.

        Purpose:
            Export a presentation to PDF with PowerPoint COM.

        Args:
            path (str): Destination PDF.

        Returns:
            str: Destination path.
        """
        try:
            throw_if( 'path', path )
            self.pdf_path = str( Path( path ).resolve( ) )
            if self.backend != 'com' or self.presentation is None:
                raise NotImplementedError( 'PDF export requires an open COM PowerPoint presentation.' )
            self.presentation.SaveAs( self.pdf_path, 32 )
            return self.pdf_path
        except Exception as e:
            e.add_note( 'fonky.loco.PowerPoint.export_pdf' )
            raise

    def close_presentation( self ) -> None:
        """Release an open presentation and dedicated COM application.

        Purpose:
            Release an open presentation and dedicated COM application.

        Returns:
            None: Clears presentation state.
        """
        if self.presentation is None:
            return
        try:
            if self.backend == 'com':
                self.presentation.Close( )
        finally:
            self.presentation = None
            if self.application is not None:
                self.application.Quit( )
                self.application = None



    def add_text( self, slide_number: int, text: str, left: float = 1.0,
        top: float = 1.0, width: float = 5.0, height: float = 1.0 ) -> None:
        """Add positioned text to a slide, using inches for coordinates.

        Purpose:
            Add positioned text to a slide, using inches for coordinates.

        Args:
            slide_number (int): One-based slide number.
            text (str): Text to insert.
            left (float): Left coordinate in inches.
            top (float): Top coordinate in inches.
            width (float): Width in inches.
            height (float): Height in inches.

        Returns:
            None: Adds a text box.
        """
        try:
            throw_if( 'text', text )
            self.slide_number = slide_number
            self.text = text
            self.bounds = (left, top, width, height)
            if self.presentation is None:
                raise RuntimeError( 'No presentation is open.' )
            if self.backend == 'com':
                shape = self.presentation.Slides( self.slide_number ).Shapes.AddTextbox(
                    1, *(value * 72 for value in self.bounds) )
                shape.TextFrame.TextRange.Text = self.text
            else:
                from pptx.util import Inches
                slide = self.presentation.slides[ self.slide_number - 1 ]
                shape = slide.shapes.add_textbox( *(Inches( value ) for value in self.bounds) )
                shape.text_frame.text = self.text
        except Exception as e:
            e.add_note( 'fonky.loco.PowerPoint.add_text' )
            raise

    def add_image( self, slide_number: int, path: str, left: float = 1.0,
        top: float = 1.0, width: float = 4.0 ) -> None:
        """Insert a local image on a slide.

        Purpose:
            Insert a local image on a slide.

        Args:
            slide_number (int): One-based slide index.
            path (str): Image filename.
            left (float): Horizontal position in inches.
            top (float): Vertical position in inches.
            width (float): Image width in inches.

        Returns:
            None: Adds the image.
        """
        try:
            throw_if( 'path', path )
            self.slide_number = slide_number
            self.image_path = str( Path( path ).resolve( ) )
            self.left = left
            self.top = top
            self.width = width
            if self.presentation is None:
                raise RuntimeError( 'No presentation is open.' )
            if self.backend == 'com':
                self.presentation.Slides( self.slide_number ).Shapes.AddPicture(
                    self.image_path, False, True, self.left * 72, self.top * 72, self.width * 72, -1 )
            else:
                from pptx.util import Inches
                self.presentation.slides[ self.slide_number - 1 ].shapes.add_picture(
                    self.image_path, Inches( self.left ), Inches( self.top ), width=Inches( self.width ) )
        except Exception as e:
            e.add_note( 'fonky.loco.PowerPoint.add_image' )
            raise



    def delete_slide( self, slide_number: int ) -> None:
        """Delete a presentation slide.

        Purpose:
            Delete a presentation slide.

        Args:
            slide_number (int): One-based slide number.

        Returns:
            None: Removes the selected slide.
        """
        try:
            self.slide_number = slide_number
            if self.presentation is None:
                raise RuntimeError( 'No presentation is open.' )
            if self.slide_number < 1 or self.slide_number > len( self.list_slides( ) ):
                raise IndexError( 'Slide number is out of range.' )
            if self.backend == 'com':
                self.presentation.Slides( self.slide_number ).Delete( )
            else:
                slide_id = self.presentation.slides._sldIdLst[ self.slide_number - 1 ]
                self.presentation.part.drop_rel( slide_id.rId )
                self.presentation.slides._sldIdLst.remove( slide_id )
        except Exception as e:
            e.add_note( 'fonky.loco.PowerPoint.delete_slide' )
            raise

    def add_table( self, slide_number: int, rows: Sequence[Sequence[Any]],
        left: float = 1.0, top: float = 1.0, width: float = 8.0,
        height: float = 3.0 ) -> None:
        """Insert a rectangular table on a slide.

        Purpose:
            Insert a rectangular table on a slide.

        Args:
            slide_number (int): One-based slide number.
            rows (Sequence[Sequence[Any]]): Rectangular table contents.
            left (float): Left position in inches.
            top (float): Top position in inches.
            width (float): Table width in inches.
            height (float): Table height in inches.

        Returns:
            None: Adds the table.
        """
        try:
            throw_if( 'rows', rows )
            self.slide_number = slide_number
            self.rows = [ list( row ) for row in rows ]
            self.left = left
            self.top = top
            self.width = width
            self.height = height
            columns = len( self.rows[ 0 ] )
            if columns < 1 or any( len( row ) != columns for row in self.rows ):
                raise ValueError( 'Table rows must have consistent nonzero width.' )
            if self.presentation is None:
                raise RuntimeError( 'No presentation is open.' )
            if self.backend == 'com':
                shape = self.presentation.Slides( self.slide_number ).Shapes.AddTable(
                    len( self.rows ), columns, self.left * 72, self.top * 72,
                    self.width * 72, self.height * 72 )
                for i, row in enumerate( self.rows ):
                    for j, value in enumerate( row ):
                        shape.Table.Cell( i + 1, j + 1 ).Shape.TextFrame.TextRange.Text = str( value )
            else:
                from pptx.util import Inches
                slide = self.presentation.slides[ self.slide_number - 1 ]
                shape = slide.shapes.add_table( len( self.rows ), columns,
                    Inches( self.left ), Inches( self.top ), Inches( self.width ),
                    Inches( self.height ) )
                for i, row in enumerate( self.rows ):
                    for j, value in enumerate( row ):
                        shape.table.cell( i, j ).text = str( value )
        except Exception as e:
            e.add_note( 'fonky.loco.PowerPoint.add_table' )
            raise


class Outlook:
    """Classic Microsoft Outlook desktop automation via COM.

    Purpose:
        Exposes the public Outlook automation API through local Python or supported
        desktop-application backends without requiring a cloud-model API key.

    Attributes:
        application (Any): COM application or UI automation object owned by this instance.
        namespace (Any): MAPI namespace obtained from the Outlook COM application.
        folder (str): Selected Outlook default mailbox folder.
        limit (int): Maximum number of returned or scanned Outlook items.
        entry_id (str): Outlook item identifier.
        recipient (str): Message recipient address.
        subject (str): Mail or calendar appointment subject.
        body (str): Plain-text Outlook message body.
        approved (bool): Explicit authorization for a sensitive operation.
        start (str): Appointment start timestamp.
        end (str): Appointment end timestamp.
        location (str): Appointment location.
        phrase (str): Case-folded term used for Outlook message searching.
        directory (Path): Selected filesystem directory.

    Methods:
        connect(): Connect to the default Outlook MAPI profile.
        list_folders(): List subfolders of the default mailbox folder.
        list_messages(): List recent messages in the default Inbox.
        read_message(): Read an Outlook mail item by entry ID.
        create_draft(): Create and save an Outlook email draft.
        send_message(): Send an existing draft only with explicit approval.
        create_appointment(): Save an Outlook calendar appointment without inviting attendees.
        search_messages(): Search recent Inbox messages by subject or sender.
        list_appointments(): List appointments in the default Outlook Calendar.
        save_attachments(): Save a message's attachments to an existing local directory.
    """


    # Public instance members (annotations only; values are assigned by operations).
    application: Any  # COM application or UI automation object owned by this instance.
    namespace: Any  # MAPI namespace obtained from the Outlook COM application.
    folder: str  # Selected Outlook default mailbox folder.
    limit: int  # Maximum number of returned or scanned Outlook items.
    entry_id: str  # Outlook item identifier.
    recipient: str  # Message recipient address.
    subject: str  # Mail or calendar appointment subject.
    body: str  # Plain-text Outlook message body.
    approved: bool  # Explicit authorization for a sensitive operation.
    start: str  # Appointment start timestamp.
    end: str  # Appointment end timestamp.
    location: str  # Appointment location.
    phrase: str  # Case-folded term used for Outlook message searching.
    directory: Path  # Selected filesystem directory.
    def __init__( self ) -> None:
        """Initialize Outlook state without launching the application."""
        self.application = None
        self.namespace = None

    def connect( self ) -> None:
        """Connect to the default Outlook MAPI profile.

        Purpose:
            Connect to the default Outlook MAPI profile.

        Returns:
            None: Holds a COM namespace.
        """
        self.application = office_application( 'Outlook.Application' )
        self.namespace = self.application.GetNamespace( 'MAPI' )

    def list_folders( self, folder: str = 'Inbox' ) -> list[str]:
        """List subfolders of the default mailbox folder.

        Purpose:
            List subfolders of the default mailbox folder.

        Args:
            folder (str): Supported default folder name.

        Returns:
            list[str]: Folder names.
        """
        self.folder = folder
        if self.namespace is None:
            self.connect( )
        ids = { 'Inbox': 6, 'Sent': 5, 'Drafts': 16, 'Calendar': 9 }
        if self.folder not in ids:
            raise ValueError( f'Unknown default Outlook folder: {self.folder}' )
        parent = self.namespace.GetDefaultFolder( ids[ self.folder ] )
        return [ child.Name for child in parent.Folders ]

    def list_messages( self, limit: int = 20 ) -> list[dict[str, str]]:
        """List recent messages in the default Inbox.

        Purpose:
            List recent messages in the default Inbox.

        Args:
            limit (int): Maximum number of messages.

        Returns:
            list[dict[str, str]]: Message summaries.
        """
        self.limit = limit
        if self.limit < 1:
            raise ValueError( 'limit must be positive.' )
        if self.namespace is None:
            self.connect( )
        items = self.namespace.GetDefaultFolder( 6 ).Items
        items.Sort( '[ReceivedTime]', True )
        output = []
        for item in items:
            if item.Class == 43:
                output.append( { 'entry_id': str( item.EntryID ), 'subject': str( item.Subject or '' ),
                    'sender': str( item.SenderName or '' ) } )
                if len( output ) >= self.limit:
                    break
        return output

    def read_message( self, entry_id: str ) -> dict[str, str]:
        """Read an Outlook mail item by entry ID.

        Purpose:
            Read an Outlook mail item by entry ID.

        Args:
            entry_id (str): Outlook message identifier.

        Returns:
            dict[str, str]: Subject, body, and sender.
        """
        try:
            throw_if( 'entry_id', entry_id )
            self.entry_id = entry_id
            if self.namespace is None:
                self.connect( )
            item = self.namespace.GetItemFromID( self.entry_id )
            if item.Class != 43:
                raise ValueError( 'Requested Outlook item is not a mail message.' )
            return { 'subject': str( item.Subject or '' ), 'sender': str( item.SenderName or '' ),
                'body': str( item.Body or '' ) }
        except Exception as e:
            e.add_note( 'fonky.loco.Outlook.read_message' )
            raise

    def create_draft( self, recipient: str, subject: str, body: str ) -> str:
        """Create and save an Outlook email draft.

        Purpose:
            Create and save an Outlook email draft.

        Args:
            recipient (str): Recipient address.
            subject (str): Message subject.
            body (str): Plain text message.

        Returns:
            str: Draft entry ID.
        """
        try:
            throw_if( 'recipient', recipient )
            throw_if( 'subject', subject )
            self.recipient = recipient
            self.subject = subject
            self.body = body
            if self.application is None:
                self.connect( )
            item = self.application.CreateItem( 0 )
            item.To = self.recipient
            item.Subject = self.subject
            item.Body = self.body
            item.Save( )
            return str( item.EntryID )
        except Exception as e:
            e.add_note( 'fonky.loco.Outlook.create_draft' )
            raise

    def send_message( self, entry_id: str, approved: bool = False ) -> None:
        """Send an existing draft only with explicit approval.

        Purpose:
            Send an existing draft only with explicit approval.

        Args:
            entry_id (str): Draft entry ID.
            approved (bool): Explicit authorization to send.

        Returns:
            None: Sends the message.

        Raises:
            PermissionError: If send approval is absent.
        """
        try:
            throw_if( 'entry_id', entry_id )
            self.entry_id = entry_id
            self.approved = approved
            if not self.approved:
                raise PermissionError( 'Sending Outlook mail requires approved=True.' )
            if self.namespace is None:
                self.connect( )
            item = self.namespace.GetItemFromID( self.entry_id )
            if item.Class != 43:
                raise ValueError( 'Requested Outlook item is not a mail message.' )
            item.Send( )


    # ==========================================================================================
    # WINDOWS UI AUTOMATION
    # ==========================================================================================
        except Exception as e:
            e.add_note( 'fonky.loco.Outlook.send_message' )
            raise


    def create_appointment( self, subject: str, start: str, end: str,
        location: str = '' ) -> str:
        """Save an Outlook calendar appointment without inviting attendees.

        Purpose:
            Save an Outlook calendar appointment without inviting attendees.

        Args:
            subject (str): Appointment subject.
            start (str): Start time, parseable by Outlook.
            end (str): End time, parseable by Outlook.
            location (str): Optional location.

        Returns:
            str: Saved appointment entry ID.
        """
        try:
            throw_if( 'subject', subject )
            throw_if( 'start', start )
            throw_if( 'end', end )
            self.subject = subject
            self.start = start
            self.end = end
            self.location = location
            if self.application is None:
                self.connect( )
            item = self.application.CreateItem( 1 )
            item.Subject = self.subject
            item.Start = self.start
            item.End = self.end
            item.Location = self.location
            item.Save( )
            return str( item.EntryID )
        except Exception as e:
            e.add_note( 'fonky.loco.Outlook.create_appointment' )
            raise



    def search_messages( self, phrase: str, limit: int = 100 ) -> list[dict[str, str]]:
        """Search recent Inbox messages by subject or sender.

        Purpose:
            Search recent Inbox messages by subject or sender.

        Args:
            phrase (str): Case-insensitive search phrase.
            limit (int): Maximum messages scanned.

        Returns:
            list[dict[str, str]]: Matching message summaries.
        """
        try:
            throw_if( 'phrase', phrase )
            self.phrase = phrase.casefold( )
            self.limit = limit
            messages = self.list_messages( self.limit )
            return [ message for message in messages if self.phrase in
                message[ 'subject' ].casefold( ) or self.phrase in message[ 'sender' ].casefold( ) ]
        except Exception as e:
            e.add_note( 'fonky.loco.Outlook.search_messages' )
            raise

    def list_appointments( self, limit: int = 20 ) -> list[dict[str, str]]:
        """List appointments in the default Outlook Calendar.

        Purpose:
            List appointments in the default Outlook Calendar.

        Args:
            limit (int): Maximum number of calendar items.

        Returns:
            list[dict[str, str]]: Calendar appointment summaries.
        """
        self.limit = limit
        if self.limit < 1:
            raise ValueError( 'limit must be positive.' )
        if self.namespace is None:
            self.connect( )
        items = self.namespace.GetDefaultFolder( 9 ).Items
        items.Sort( '[Start]' )
        output = []
        for item in items:
            if item.Class == 26:
                output.append( { 'entry_id': str( item.EntryID ),
                    'subject': str( item.Subject or '' ),
                    'start': str( item.Start ), 'end': str( item.End ) } )
                if len( output ) >= self.limit:
                    break
        return output

    def save_attachments( self, entry_id: str, directory: str ) -> list[str]:
        """Save a message's attachments to an existing local directory.

        Purpose:
            Save a message's attachments to an existing local directory.

        Args:
            entry_id (str): Outlook message identifier.
            directory (str): Destination directory.

        Returns:
            list[str]: Saved attachment paths.
        """
        try:
            throw_if( 'entry_id', entry_id )
            throw_if( 'directory', directory )
            self.entry_id = entry_id
            self.directory = Path( directory ).resolve( )
            if not self.directory.is_dir( ):
                raise NotADirectoryError( str( self.directory ) )
            if self.namespace is None:
                self.connect( )
            item = self.namespace.GetItemFromID( self.entry_id )
            if item.Class != 43:
                raise ValueError( 'Requested Outlook item is not a mail message.' )
            saved = []
            for attachment in item.Attachments:
                name = Path( str( attachment.FileName ) ).name
                if not name or name in ('.', '..'):
                    raise ValueError( 'Invalid attachment filename.' )
                target = self.directory / name
                if target.exists( ):
                    raise FileExistsError( str( target ) )
                attachment.SaveAsFile( str( target ) )
                saved.append( str( target ) )
            return saved
        except Exception as e:
            e.add_note( 'fonky.loco.Outlook.save_attachments' )
            raise


class Windows:
    """Windows UI Automation backed by pywinauto.

    Purpose:
        Exposes the public Windows GUI automation API through local Python or supported
        desktop-application backends without requiring a cloud-model API key.

    Attributes:
        backend (str): Selected file-based, COM, or UI Automation backend.
        application (Any): COM application or UI automation object owned by this instance.
        window (Any): Selected Windows top-level UI Automation window.
        title (str): Current slide, window, or control title.
        control_type (str): UI Automation category of the selected control.
        text (str): Text used in the current document or UI operation.
        path (str | Path): Active local file, document, or database path.
        width (float): Object or window width.
        height (float): Object or window height.

    Methods:
        list_windows(): List visible top-level window titles.
        find_window(): Attach to an existing window by exact title.
        activate_window(): Focus an attached window.
        click_control(): Click a named UI control.
        set_text(): Write to an identified edit control.
        capture_window(): Capture the attached window to an image.
        get_text(): Read text from a named UI Automation control.
        close_window(): Request that the selected window close.
        minimize_window(): Minimize the attached window.
        maximize_window(): Maximize the attached window.
        resize_window(): Resize the attached window.
        list_controls(): List textual descriptions of descendant controls.
    """


    # Public instance members (annotations only; values are assigned by operations).
    backend: str  # Selected file-based, COM, or UI Automation backend.
    application: Any  # COM application or UI automation object owned by this instance.
    window: Any  # Selected Windows top-level UI Automation window.
    title: str  # Current slide, window, or control title.
    control_type: str  # UI Automation category of the selected control.
    text: str  # Text used in the current document or UI operation.
    path: str | Path  # Active local file, document, or database path.
    width: float  # Object or window width.
    height: float  # Object or window height.
    def __init__( self, backend: str = 'uia' ) -> None:
        """Select a pywinauto backend.

        Purpose:
            Select a pywinauto backend.

        Args:
            backend (str): UIA or win32.

        Returns:
            None: Initializes automation state.
        """
        self.backend = backend
        self.application = None
        self.window = None

    def list_windows( self ) -> list[str]:
        """List visible top-level window titles.

        Purpose:
            List visible top-level window titles.

        Returns:
            list[str]: Window captions.
        """
        require_windows( )
        from pywinauto import Desktop
        return [ item.window_text( ) for item in Desktop( backend=self.backend ).windows( )
            if item.is_visible( ) ]

    def find_window( self, title: str ) -> None:
        """Attach to an existing window by exact title.

        Purpose:
            Attach to an existing window by exact title.

        Args:
            title (str): Window title.

        Returns:
            None: Stores the window proxy.
        """
        try:
            throw_if( 'title', title )
            self.title = title
            require_windows( )
            from pywinauto import Desktop
            self.window = Desktop( backend=self.backend ).window( title=self.title )
            self.window.wait( 'exists ready', timeout=10 )
        except Exception as e:
            e.add_note( 'fonky.loco.Windows.find_window' )
            raise

    def activate_window( self ) -> None:
        """Focus an attached window.

        Purpose:
            Focus an attached window.

        Returns:
            None: Activates the window.
        """
        if self.window is None:
            raise RuntimeError( 'No window has been selected.' )
        self.window.set_focus( )

    def click_control( self, title: str, control_type: str = 'Button' ) -> None:
        """Click a named UI control.

        Purpose:
            Click a named UI control.

        Args:
            title (str): Visible control title.
            control_type (str): UI Automation control type.

        Returns:
            None: Invokes the control.
        """
        try:
            throw_if( 'title', title )
            self.title = title
            self.control_type = control_type
            if self.window is None:
                raise RuntimeError( 'No window has been selected.' )
            self.window.child_window( title=self.title, control_type=self.control_type ).click_input( )
        except Exception as e:
            e.add_note( 'fonky.loco.Windows.click_control' )
            raise

    def set_text( self, title: str, text: str ) -> None:
        """Write to an identified edit control.

        Purpose:
            Write to an identified edit control.

        Args:
            title (str): Edit control name.
            text (str): Replacement text.

        Returns:
            None: Updates the control.
        """
        try:
            throw_if( 'title', title )
            self.title = title
            self.text = text
            if self.window is None:
                raise RuntimeError( 'No window has been selected.' )
            self.window.child_window( title=self.title, control_type='Edit' ).set_edit_text( self.text )
        except Exception as e:
            e.add_note( 'fonky.loco.Windows.set_text' )
            raise

    def capture_window( self, path: str ) -> str:
        """Capture the attached window to an image.

        Purpose:
            Capture the attached window to an image.

        Args:
            path (str): Destination image path.

        Returns:
            str: Saved image path.
        """
        try:
            throw_if( 'path', path )
            self.path = str( Path( path ).resolve( ) )
            if self.window is None:
                raise RuntimeError( 'No window has been selected.' )
            self.window.capture_as_image( ).save( self.path )
            return self.path
        except Exception as e:
            e.add_note( 'fonky.loco.Windows.capture_window' )
            raise



    def get_text( self, title: str, control_type: str = 'Edit' ) -> str:
        """Read text from a named UI Automation control.

        Purpose:
            Read text from a named UI Automation control.

        Args:
            title (str): Control title.
            control_type (str): UIA control type.

        Returns:
            str: Control text.
        """
        try:
            throw_if( 'title', title )
            self.title = title
            self.control_type = control_type
            if self.window is None:
                raise RuntimeError( 'No window has been selected.' )
            return str( self.window.child_window( title=self.title,
                control_type=self.control_type ).window_text( ) )
        except Exception as e:
            e.add_note( 'fonky.loco.Windows.get_text' )
            raise

    def close_window( self ) -> None:
        """Request that the selected window close.

        Purpose:
            Request that the selected window close.

        Returns:
            None: Closes the window.
        """
        if self.window is None:
            raise RuntimeError( 'No window has been selected.' )
        self.window.close( )
        self.window = None



    def minimize_window( self ) -> None:
        """Minimize the attached window.

        Purpose:
            Minimize the attached window.

        Returns:
            None: Minimizes the window.
        """
        if self.window is None:
            raise RuntimeError( 'No window has been selected.' )
        self.window.minimize( )

    def maximize_window( self ) -> None:
        """Maximize the attached window.

        Purpose:
            Maximize the attached window.

        Returns:
            None: Maximizes the window.
        """
        if self.window is None:
            raise RuntimeError( 'No window has been selected.' )
        self.window.maximize( )

    def resize_window( self, width: int, height: int ) -> None:
        """Resize the attached window.

        Purpose:
            Resize the attached window.

        Args:
            width (int): Width in pixels.
            height (int): Height in pixels.

        Returns:
            None: Resizes the window.
        """
        try:
            self.width = width
            self.height = height
            if self.width < 1 or self.height < 1:
                raise ValueError( 'Window dimensions must be positive.' )
            if self.window is None:
                raise RuntimeError( 'No window has been selected.' )
            self.window.move_window( width=self.width, height=self.height )
        except Exception as e:
            e.add_note( 'fonky.loco.Windows.resize_window' )
            raise

    def list_controls( self ) -> list[str]:
        """List textual descriptions of descendant controls.

        Purpose:
            List textual descriptions of descendant controls.

        Returns:
            list[str]: Window control descriptions.
        """
        if self.window is None:
            raise RuntimeError( 'No window has been selected.' )
        return [ f'{control.element_info.control_type}: {control.window_text( )}'
            for control in self.window.descendants( ) ]


__all__ = [ 'Outlook', 'Word', 'PowerPoint', 'Excel', 'Calculator', 'File', 'Data',
    'Windows', 'Shell', 'throw_if' ]
