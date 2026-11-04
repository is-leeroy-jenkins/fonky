'''
  ******************************************************************************************
      Assembly:                fonky
      Filename:                loco.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="guro.py" company="Terry D. Eppler">

	     name.py
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
    name.py
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


def require_windows( ) -> None:
    """Require a Windows desktop environment.

    Raises:
        OSError: If the current platform is not Windows.
    """
    if sys.platform != 'win32':
        raise OSError( 'This operation requires Windows and a desktop session.' )


def office_application( prog_id: str, visible: bool = False ) -> Any:
    """Create a dedicated Office COM application instance.

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
    """Local filesystem operations with an optional root-directory boundary."""

    root: Path | None

    def __init__( self, root: str = '' ) -> None:
        """Configure permitted filesystem scope.

        Args:
            root (str): Optional directory limiting access.

        Returns:
            None: Initializes the filesystem service.
        """
        self.root = Path( root ).resolve( ) if root else None

    def resolve( self, path: str ) -> Path:
        """Resolve a path within the configured root.

        Args:
            path (str): File or directory path.

        Returns:
            Path: Resolved filesystem path.

        Raises:
            ValueError: If the path is missing or outside the permitted root.
        """
        throw_if( 'path', path )
        self.path = Path( path )
        if self.root is not None and not self.path.is_absolute( ):
            self.path = self.root / self.path
        self.path = self.path.resolve( )
        if self.root is not None and not self.path.is_relative_to( self.root ):
            raise PermissionError( f'Path is outside the permitted directory: {self.path}' )
        return self.path

    def exists( self, path: str ) -> bool:
        """Check whether a local path exists.

        Args:
            path (str): Target path.

        Returns:
            bool: True if the path exists.
        """
        self.path = path
        return self.resolve( self.path ).exists( )

    def list_files( self, directory: str = '.', pattern: str = '*', recursive: bool = False ) -> list[ str ]:
        """List files matching a glob pattern.

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

        Args:
            path (str): Source file.
            encoding (str): Text encoding.

        Returns:
            str: File contents.
        """
        self.path = path
        self.encoding = encoding
        return self.resolve( self.path ).read_text( encoding=self.encoding )

    def write_text( self, path: str, content: str, overwrite: bool = False,
        encoding: str = 'utf-8' ) -> int:
        """Write text without overwriting an existing file by default.

        Args:
            path (str): Output file.
            content (str): Text content.
            overwrite (bool): Allow replacing an existing file.
            encoding (str): Text encoding.

        Returns:
            int: Characters written.
        """
        self.path = self.resolve( path )
        self.content = content
        self.overwrite = overwrite
        self.encoding = encoding
        with self.path.open( 'w' if self.overwrite else 'x', encoding=self.encoding ) as handle:
            return handle.write( self.content )

    def copy_file( self, source: str, destination: str, overwrite: bool = False ) -> str:
        """Copy a file to a new destination.

        Args:
            source (str): Source file.
            destination (str): Destination file.
            overwrite (bool): Permit replacing an existing destination.

        Returns:
            str: Destination path.
        """
        self.source = self.resolve( source )
        self.destination = self.resolve( destination )
        self.overwrite = overwrite
        if not self.overwrite and self.destination.exists( ):
            raise FileExistsError( str( self.destination ) )
        shutil.copy2( self.source, self.destination )
        return str( self.destination )

    def move_file( self, source: str, destination: str, overwrite: bool = False ) -> str:
        """Move a file without overwriting by default.

        Args:
            source (str): Source file.
            destination (str): Destination path.
            overwrite (bool): Permit destination replacement.

        Returns:
            str: Destination path.
        """
        self.source = self.resolve( source )
        self.destination = self.resolve( destination )
        self.overwrite = overwrite
        if self.destination.exists( ) and not self.overwrite:
            raise FileExistsError( str( self.destination ) )
        if self.destination.exists( ):
            self.destination.unlink( )
        shutil.move( str( self.source ), str( self.destination ) )
        return str( self.destination )

    def delete_file( self, path: str ) -> None:
        """Remove a specified file.

        Args:
            path (str): File to delete.

        Returns:
            None: Deletes the file.
        """
        self.path = self.resolve( path )
        self.path.unlink( )

    def create_directory( self, path: str ) -> str:
        """Create a directory and missing parents.

        Args:
            path (str): Directory path.

        Returns:
            str: Created directory path.
        """
        self.path = self.resolve( path )
        self.path.mkdir( parents=True, exist_ok=True )
        return str( self.path )

    def calculate_hash( self, path: str, algorithm: str = 'sha256' ) -> str:
        """Calculate a supported file digest.

        Args:
            path (str): Source file.
            algorithm (str): Hash algorithm accepted by hashlib.

        Returns:
            str: Hex digest.
        """
        self.path = self.resolve( path )
        self.algorithm = algorithm
        self.digest = hashlib.new( self.algorithm )
        with self.path.open( 'rb' ) as handle:
            for chunk in iter( lambda: handle.read( 1024 * 1024 ), b'' ):
                self.digest.update( chunk )
        return self.digest.hexdigest( )

    def compress( self, directory: str, destination: str ) -> str:
        """Create a ZIP archive without following out-of-root symlinks.

        Args:
            directory (str): Directory to archive.
            destination (str): New ZIP path.

        Returns:
            str: Archive path.
        """
        self.directory = self.resolve( directory )
        self.destination = self.resolve( destination )
        with zipfile.ZipFile( self.destination, 'x', zipfile.ZIP_DEFLATED ) as archive:
            for entry in self.directory.rglob( '*' ):
                if entry.is_file( ) and entry.resolve( ) != self.destination and (
                    self.root is None or entry.resolve( ).is_relative_to( self.root )):
                    archive.write( entry, entry.relative_to( self.directory ) )
        return str( self.destination )

    def extract( self, archive: str, destination: str ) -> str:
        """Extract a ZIP archive into an allowed directory.

        Args:
            archive (str): Source ZIP archive.
            destination (str): Extraction directory.

        Returns:
            str: Extraction directory.

        Raises:
            PermissionError: If an entry escapes the extraction boundary.
        """
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

class Data:
    """SQLite connection and transaction operations."""

    connection: sqlite3.Connection | None
    path: str
    read_only: bool

    def __init__( self, path: str, read_only: bool = False ) -> None:
        """Initialize a database wrapper without connecting.

        Args:
            path (str): Database file path.
            read_only (bool): Open using SQLite read-only mode.

        Returns:
            None: Stores database configuration.
        """
        throw_if( 'path', path )
        self.path = path
        self.read_only = read_only
        self.connection = None

    def connect( self ) -> None:
        """Connect to SQLite.

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

        Returns:
            None: Closes the connection if present.
        """
        if self.connection is not None:
            self.connection.close( )
            self.connection = None

    def execute_query( self, sql: str, parameters: Sequence[Any] = () ) -> list[ dict[ str, Any ] ]:
        """Execute parameterized SQL and return rows.

        Args:
            sql (str): SQL query.
            parameters (Sequence[Any]): Positional SQL parameters.

        Returns:
            list[dict[str, Any]]: Query rows.
        """
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

    def execute_nonquery( self, sql: str, parameters: Sequence[Any] = () ) -> int:
        """Execute a parameterized write statement without committing.

        Args:
            sql (str): SQL command.
            parameters (Sequence[Any]): Positional SQL parameters.

        Returns:
            int: Number of affected rows.
        """
        throw_if( 'sql', sql )
        self.sql = sql
        self.parameters = parameters
        if self.read_only:
            raise PermissionError( 'Database is read-only.' )
        self.connect( )
        self.cursor = self.connection.execute( self.sql, self.parameters )
        return self.cursor.rowcount

    def execute_many( self, sql: str, rows: Sequence[Sequence[Any]] ) -> int:
        """Execute a parameterized write for multiple records.

        Args:
            sql (str): SQL command.
            rows (Sequence[Sequence[Any]]): Parameter groups.

        Returns:
            int: Number of affected rows.
        """
        throw_if( 'sql', sql )
        self.sql = sql
        self.rows = rows
        if self.read_only:
            raise PermissionError( 'Database is read-only.' )
        self.connect( )
        self.cursor = self.connection.executemany( self.sql, self.rows )
        return self.cursor.rowcount

    def list_tables( self ) -> list[ str ]:
        """List non-system SQLite tables.

        Returns:
            list[str]: Table names.
        """
        return [ row[ 'name' ] for row in self.execute_query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name" ) ]

    def get_schema( self, table: str ) -> list[ dict[ str, Any ] ]:
        """Read a table schema.

        Args:
            table (str): Existing table name.

        Returns:
            list[dict[str, Any]]: Column metadata.
        """
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

    def commit( self ) -> None:
        """Commit pending changes.

        Returns:
            None: Commits the active transaction.
        """
        self.connect( )
        self.connection.commit( )

    def rollback( self ) -> None:
        """Rollback pending changes.

        Returns:
            None: Rolls back the transaction.
        """
        self.connect( )
        self.connection.rollback( )

    def backup_database( self, destination: str ) -> str:
        """Copy a consistent SQLite database snapshot.

        Args:
            destination (str): New SQLite database path.

        Returns:
            str: Backup path.
        """
        throw_if( 'destination', destination )
        self.destination = destination
        self.connect( )
        if Path( self.destination ).resolve( ) == Path( self.path ).resolve( ):
            raise ValueError( 'Destination must differ from the source database.' )
        with sqlite3.connect( self.destination ) as target:
            self.connection.backup( target )
        return self.destination

    def integrity_check( self ) -> list[ str ]:
        """Verify SQLite file integrity.

        Returns:
            list[str]: SQLite integrity-check messages.
        """
        return [ row[ 'integrity_check' ] for row in self.execute_query( 'PRAGMA integrity_check' ) ]

    def read_dataframe( self, sql: str, parameters: Sequence[Any] = () ) -> Any:
        """Read a query into a pandas DataFrame.

        Args:
            sql (str): SQL query.
            parameters (Sequence[Any]): Positional parameters.

        Returns:
            pandas.DataFrame: Query results.
        """
        throw_if( 'sql', sql )
        self.sql = sql
        self.parameters = parameters
        import pandas as pd
        return pd.DataFrame( self.execute_query( self.sql, self.parameters ) )


# ==========================================================================================
# DETERMINISTIC CALCULATOR
# ==========================================================================================

class Calculator:
    """Safe arithmetic and optional Windows Calculator control."""

    _binary: dict[type, Callable[[Any, Any], Any]] = {
        ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod
    }
    _unary: dict[type, Callable[[Any], Any]] = { ast.UAdd: operator.pos, ast.USub: operator.neg }

    def add( self, left: float, right: float ) -> float:
        """Add two numbers.

        Args:
            left (float): First number.
            right (float): Second number.

        Returns:
            float: Sum.
        """
        self.left = left
        self.right = right
        return self.left + self.right

    def subtract( self, left: float, right: float ) -> float:
        """Subtract two numbers.

        Args:
            left (float): First number.
            right (float): Second number.

        Returns:
            float: Difference.
        """
        self.left = left
        self.right = right
        return self.left - self.right

    def multiply( self, left: float, right: float ) -> float:
        """Multiply two numbers.

        Args:
            left (float): First number.
            right (float): Second number.

        Returns:
            float: Product.
        """
        self.left = left
        self.right = right
        return self.left * self.right

    def divide( self, left: float, right: float ) -> float:
        """Divide two numbers.

        Args:
            left (float): Numerator.
            right (float): Denominator.

        Returns:
            float: Quotient.
        """
        self.left = left
        self.right = right
        return self.left / self.right

    def square_root( self, value: float ) -> float:
        """Calculate a square root.

        Args:
            value (float): Radicand.

        Returns:
            float: Square root.
        """
        self.value = value
        return math.sqrt( self.value )

    def percentage( self, value: float, percent: float ) -> float:
        """Calculate a percentage of a number.

        Args:
            value (float): Base value.
            percent (float): Percentage.

        Returns:
            float: Percentage result.
        """
        self.value = value
        self.percent = percent
        return self.value * self.percent / 100

    def evaluate_expression( self, expression: str ) -> float:
        """Evaluate arithmetic syntax without executing Python code.

        Args:
            expression (str): Numeric expression.

        Returns:
            float: Calculated value.

        Raises:
            ValueError: If the expression contains unsupported nodes.
        """
        throw_if( 'expression', expression )
        self.expression = expression
        self.tree = ast.parse( self.expression, mode='eval' )
        return float( self.evaluate_node( self.tree.body ) )

    def evaluate_node( self, node: ast.AST ) -> float:
        """Evaluate a restricted arithmetic node.

        Args:
            node (ast.AST): Validated expression subtree.

        Returns:
            float: Calculated value.
        """
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

    def launch_calculator( self ) -> subprocess.Popen:
        """Launch Windows Calculator.

        Returns:
            subprocess.Popen: Launched process handle.
        """
        require_windows( )
        return subprocess.Popen( [ 'calc.exe' ] )


# ==========================================================================================
# SYSTEM PROCESS EXECUTION
# ==========================================================================================

class Shell:
    """Subprocess execution with explicit permissions and timeouts."""

    allowed_commands: set[str] | None

    def __init__( self, allowed_commands: Sequence[str] = () ) -> None:
        """Configure an optional executable allowlist.

        Args:
            allowed_commands (Sequence[str]): Names of allowed executables; empty denies execution.

        Returns:
            None: Initializes execution policy.
        """
        self.allowed_commands = set( allowed_commands )

    def run_command( self, command: Sequence[str], timeout: int = 30,
        cwd: str = '' ) -> dict[ str, Any ]:
        """Run a permitted executable without shell interpolation.

        Args:
            command (Sequence[str]): Executable and argument tokens.
            timeout (int): Timeout in seconds.
            cwd (str): Optional working directory.

        Returns:
            dict[str, Any]: Return code, stdout, and stderr.

        Raises:
            PermissionError: If the executable is not on the allowlist.
        """
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

    def run_powershell( self, script: str, timeout: int = 30 ) -> dict[ str, Any ]:
        """Run a PowerShell command only when explicitly allowed.

        Args:
            script (str): PowerShell source.
            timeout (int): Maximum execution time.

        Returns:
            dict[str, Any]: Process result.
        """
        throw_if( 'script', script )
        self.script = script
        self.timeout = timeout
        require_windows( )
        return self.run_command( [ 'powershell.exe', '-NoProfile', '-NonInteractive',
            '-Command', self.script ], self.timeout )

    def which( self, command: str ) -> str | None:
        """Find an executable on PATH.

        Args:
            command (str): Executable name.

        Returns:
            str | None: Resolved path, if present.
        """
        throw_if( 'command', command )
        self.command = command
        return shutil.which( self.command )

    def get_system_info( self ) -> dict[ str, str ]:
        """Return non-sensitive platform information.

        Returns:
            dict[str, str]: OS, Python, and architecture.
        """
        return { 'platform': platform.system( ), 'release': platform.release( ),
            'architecture': platform.machine( ), 'python': platform.python_version( ) }

    def get_disk_usage( self, path: str = '.' ) -> dict[ str, int ]:
        """Read local disk capacity.

        Args:
            path (str): Filesystem path.

        Returns:
            dict[str, int]: Capacity, used, and free bytes.
        """
        self.path = path
        self.usage = shutil.disk_usage( self.path )
        return { 'total': self.usage.total, 'used': self.usage.used, 'free': self.usage.free }


# ==========================================================================================
# MICROSOFT OFFICE: EXCEL, WORD, POWERPOINT AND OUTLOOK
# ==========================================================================================

class Excel:
    """Workbook editing through openpyxl or dedicated Excel COM automation."""

    workbook: Any
    application: Any
    path: str
    backend: str

    def __init__( self, backend: str = 'file' ) -> None:
        """Choose the file or COM backend.

        Args:
            backend (str): Either 'file' or 'com'.

        Returns:
            None: Initializes workbook state.
        """
        if backend not in ('file', 'com'):
            raise ValueError( "backend must be 'file' or 'com'." )
        self.backend = backend
        self.workbook = None
        self.application = None
        self.path = ''

    def open_workbook( self, path: str, visible: bool = False ) -> None:
        """Open an existing workbook.

        Args:
            path (str): Workbook file.
            visible (bool): Show the Excel COM instance.

        Returns:
            None: Holds an open workbook.
        """
        throw_if( 'path', path )
        self.path = str( Path( path ).resolve( ) )
        if self.backend == 'com':
            self.application = office_application( 'Excel.Application', visible )
            self.workbook = self.application.Workbooks.Open( self.path )
        else:
            from openpyxl import load_workbook
            self.workbook = load_workbook( self.path )

    def create_workbook( self, visible: bool = False ) -> None:
        """Create a new in-memory or COM workbook.

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

        Returns:
            list[str]: Worksheet names.
        """
        self.ensure_open( )
        if self.backend == 'com':
            return [ sheet.Name for sheet in self.workbook.Worksheets ]
        return list( self.workbook.sheetnames )

    def ensure_open( self ) -> None:
        """Require an open workbook.

        Raises:
            RuntimeError: If no workbook is open.
        """
        if self.workbook is None:
            raise RuntimeError( 'No workbook is open.' )

    def create_worksheet( self, name: str ) -> None:
        """Add a worksheet.

        Args:
            name (str): New sheet name.

        Returns:
            None: Adds a sheet.
        """
        throw_if( 'name', name )
        self.name = name
        self.ensure_open( )
        if self.backend == 'com':
            self.workbook.Worksheets.Add( ).Name = self.name
        else:
            self.workbook.create_sheet( self.name )

    def read_cell( self, sheet: str, cell: str ) -> Any:
        """Read a workbook cell.

        Args:
            sheet (str): Worksheet name.
            cell (str): A1 cell address.

        Returns:
            Any: Cell value.
        """
        throw_if( 'sheet', sheet )
        throw_if( 'cell', cell )
        self.sheet = sheet
        self.cell = cell
        self.ensure_open( )
        if self.backend == 'com':
            return self.workbook.Worksheets( self.sheet ).Range( self.cell ).Value
        return self.workbook[ self.sheet ][ self.cell ].value

    def write_cell( self, sheet: str, cell: str, value: Any ) -> None:
        """Assign a cell value or formula.

        Args:
            sheet (str): Worksheet name.
            cell (str): A1 cell address.
            value (Any): New cell value.

        Returns:
            None: Updates the cell.
        """
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

    def read_range( self, sheet: str, address: str ) -> list[list[Any]]:
        """Read a rectangular range as rows.

        Args:
            sheet (str): Worksheet name.
            address (str): A1-style rectangular address.

        Returns:
            list[list[Any]]: Cell values organized by row.
        """
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

    def write_range( self, sheet: str, start_cell: str, rows: Sequence[Sequence[Any]] ) -> None:
        """Write a rectangular sequence of values.

        Args:
            sheet (str): Worksheet name.
            start_cell (str): Top-left A1 address.
            rows (Sequence[Sequence[Any]]): Rectangular values.

        Returns:
            None: Updates the selected range.
        """
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

    def calculate( self ) -> None:
        """Request recalculation with live Excel.

        Raises:
            NotImplementedError: If using the file backend.
        """
        self.ensure_open( )
        if self.backend != 'com':
            raise NotImplementedError( 'Recalculation requires the COM backend.' )
        self.application.CalculateFull( )

    def save_workbook( self, path: str = '' ) -> str:
        """Save the workbook.

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


class Word:
    """Word document editing through python-docx or dedicated Word COM automation."""

    def __init__( self, backend: str = 'file' ) -> None:
        """Select the Word backend.

        Args:
            backend (str): 'file' or 'com'.

        Returns:
            None: Initializes document state.
        """
        if backend not in ('file', 'com'):
            raise ValueError( "backend must be 'file' or 'com'." )
        self.backend = backend
        self.document = None
        self.application = None
        self.path = ''

    def open_document( self, path: str, visible: bool = False ) -> None:
        """Open a Word document.

        Args:
            path (str): Document path.
            visible (bool): Show the COM application.

        Returns:
            None: Holds the opened document.
        """
        throw_if( 'path', path )
        self.path = str( Path( path ).resolve( ) )
        if self.backend == 'com':
            self.application = office_application( 'Word.Application', visible )
            self.document = self.application.Documents.Open( self.path )
        else:
            from docx import Document
            self.document = Document( self.path )

    def create_document( self, visible: bool = False ) -> None:
        """Create a new Word document.

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

        Args:
            text (str): Paragraph text.
            style (str): Optional paragraph style.

        Returns:
            None: Appends text.
        """
        throw_if( 'text', text )
        self.text = text
        self.style = style
        if self.document is None:
            raise RuntimeError( 'No document is open.' )
        if self.backend == 'com':
            self.document.Content.InsertAfter( self.text + '\n' )
        else:
            self.document.add_paragraph( self.text, style=self.style or None )

    def replace_text( self, old: str, new: str ) -> int:
        """Replace text in normal paragraphs.

        Args:
            old (str): Text to find.
            new (str): Replacement text.

        Returns:
            int: Number of replacements.
        """
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

    def save_document( self, path: str = '' ) -> str:
        """Save a Word document.

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

        Args:
            path (str): PDF output path.

        Returns:
            str: PDF path.

        Raises:
            NotImplementedError: If COM is not selected.
        """
        throw_if( 'path', path )
        self.path_pdf = str( Path( path ).resolve( ) )
        if self.backend != 'com' or self.document is None:
            raise NotImplementedError( 'PDF export requires an open COM Word document.' )
        self.document.ExportAsFixedFormat( self.path_pdf, 17 )
        return self.path_pdf

    def close_document( self, save_changes: bool = False ) -> None:
        """Close the document and the dedicated COM application.

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


class PowerPoint:
    """PowerPoint slide creation with python-pptx or dedicated COM automation."""

    def __init__( self, backend: str = 'file' ) -> None:
        """Select the presentation backend.

        Args:
            backend (str): 'file' or 'com'.

        Returns:
            None: Initializes presentation state.
        """
        if backend not in ('file', 'com'):
            raise ValueError( "backend must be 'file' or 'com'." )
        self.backend = backend
        self.presentation = None
        self.application = None
        self.path = ''

    def create_presentation( self, visible: bool = False ) -> None:
        """Create a presentation.

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

        Args:
            path (str): Source path.
            visible (bool): Show the application.

        Returns:
            None: Holds the presentation.
        """
        throw_if( 'path', path )
        self.path = str( Path( path ).resolve( ) )
        if self.backend == 'com':
            self.application = office_application( 'PowerPoint.Application', visible )
            self.presentation = self.application.Presentations.Open( self.path )
        else:
            from pptx import Presentation
            self.presentation = Presentation( self.path )

    def list_slides( self ) -> list[int]:
        """List one-based slide numbers.

        Returns:
            list[int]: Slide numbers.
        """
        if self.presentation is None:
            raise RuntimeError( 'No presentation is open.' )
        count = self.presentation.Slides.Count if self.backend == 'com' else len( self.presentation.slides )
        return list( range( 1, count + 1 ) )

    def add_slide( self, title: str, layout: int = 1 ) -> int:
        """Add a titled slide.

        Args:
            title (str): Slide title.
            layout (int): Layout index.

        Returns:
            int: One-based slide number.
        """
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

    def save_presentation( self, path: str = '' ) -> str:
        """Save the presentation.

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

        Args:
            path (str): Destination PDF.

        Returns:
            str: Destination path.
        """
        throw_if( 'path', path )
        self.pdf_path = str( Path( path ).resolve( ) )
        if self.backend != 'com' or self.presentation is None:
            raise NotImplementedError( 'PDF export requires an open COM PowerPoint presentation.' )
        self.presentation.SaveAs( self.pdf_path, 32 )
        return self.pdf_path

    def close_presentation( self ) -> None:
        """Release an open presentation and dedicated COM application.

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


class Outlook:
    """Classic Microsoft Outlook desktop automation via COM."""

    def __init__( self ) -> None:
        """Initialize Outlook state without launching the application."""
        self.application = None
        self.namespace = None

    def connect( self ) -> None:
        """Connect to the default Outlook MAPI profile.

        Returns:
            None: Holds a COM namespace.
        """
        self.application = office_application( 'Outlook.Application' )
        self.namespace = self.application.GetNamespace( 'MAPI' )

    def list_folders( self, folder: str = 'Inbox' ) -> list[str]:
        """List subfolders of the default mailbox folder.

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

        Args:
            entry_id (str): Outlook message identifier.

        Returns:
            dict[str, str]: Subject, body, and sender.
        """
        throw_if( 'entry_id', entry_id )
        self.entry_id = entry_id
        if self.namespace is None:
            self.connect( )
        item = self.namespace.GetItemFromID( self.entry_id )
        if item.Class != 43:
            raise ValueError( 'Requested Outlook item is not a mail message.' )
        return { 'subject': str( item.Subject or '' ), 'sender': str( item.SenderName or '' ),
            'body': str( item.Body or '' ) }

    def create_draft( self, recipient: str, subject: str, body: str ) -> str:
        """Create and save an Outlook email draft.

        Args:
            recipient (str): Recipient address.
            subject (str): Message subject.
            body (str): Plain text message.

        Returns:
            str: Draft entry ID.
        """
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

    def send_message( self, entry_id: str, approved: bool = False ) -> None:
        """Send an existing draft only with explicit approval.

        Args:
            entry_id (str): Draft entry ID.
            approved (bool): Explicit authorization to send.

        Returns:
            None: Sends the message.

        Raises:
            PermissionError: If send approval is absent.
        """
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

class Windows:
    """Windows UI Automation backed by pywinauto."""

    def __init__( self, backend: str = 'uia' ) -> None:
        """Select a pywinauto backend.

        Args:
            backend (str): UIA or win32.

        Returns:
            None: Initializes automation state.
        """
        if backend not in ('uia', 'win32'):
            raise ValueError( "backend must be 'uia' or 'win32'." )
        self.backend = backend
        self.application = None
        self.window = None

    def list_windows( self ) -> list[str]:
        """List visible top-level window titles.

        Returns:
            list[str]: Window captions.
        """
        require_windows( )
        from pywinauto import Desktop
        return [ item.window_text( ) for item in Desktop( backend=self.backend ).windows( )
            if item.is_visible( ) ]

    def find_window( self, title: str ) -> None:
        """Attach to an existing window by exact title.

        Args:
            title (str): Window title.

        Returns:
            None: Stores the window proxy.
        """
        throw_if( 'title', title )
        self.title = title
        require_windows( )
        from pywinauto import Desktop
        self.window = Desktop( backend=self.backend ).window( title=self.title )
        self.window.wait( 'exists ready', timeout=10 )

    def activate_window( self ) -> None:
        """Focus an attached window.

        Returns:
            None: Activates the window.
        """
        if self.window is None:
            raise RuntimeError( 'No window has been selected.' )
        self.window.set_focus( )

    def click_control( self, title: str, control_type: str = 'Button' ) -> None:
        """Click a named UI control.

        Args:
            title (str): Visible control title.
            control_type (str): UI Automation control type.

        Returns:
            None: Invokes the control.
        """
        throw_if( 'title', title )
        self.title = title
        self.control_type = control_type
        if self.window is None:
            raise RuntimeError( 'No window has been selected.' )
        self.window.child_window( title=self.title, control_type=self.control_type ).click_input( )

    def set_text( self, title: str, text: str ) -> None:
        """Write to an identified edit control.

        Args:
            title (str): Edit control name.
            text (str): Replacement text.

        Returns:
            None: Updates the control.
        """
        throw_if( 'title', title )
        self.title = title
        self.text = text
        if self.window is None:
            raise RuntimeError( 'No window has been selected.' )
        self.window.child_window( title=self.title, control_type='Edit' ).set_edit_text( self.text )

    def capture_window( self, path: str ) -> str:
        """Capture the attached window to an image.

        Args:
            path (str): Destination image path.

        Returns:
            str: Saved image path.
        """
        throw_if( 'path', path )
        self.path = str( Path( path ).resolve( ) )
        if self.window is None:
            raise RuntimeError( 'No window has been selected.' )
        self.window.capture_as_image( ).save( self.path )
        return self.path


__all__ = [ 'Outlook', 'Word', 'PowerPoint', 'Excel', 'Calculator', 'File', 'Data',
    'Windows', 'Shell', 'throw_if' ]
