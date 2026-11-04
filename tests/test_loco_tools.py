"""Regression tests for standalone Fonky local AI tool functions.

Purpose:
    Confirms workspace confinement, host authorization, file-backed Office tools,
    SQLite read access, and provider-native tool registration without cloud API keys.
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PACKAGE_PATH = Path( __file__ ).resolve( ).parents[ 1 ]
PACKAGE_SPEC = importlib.util.spec_from_file_location(
    'fonky', PACKAGE_PATH / '__init__.py', submodule_search_locations=[ str( PACKAGE_PATH ) ] )
package = importlib.util.module_from_spec( PACKAGE_SPEC )
sys.modules[ 'fonky' ] = package
PACKAGE_SPEC.loader.exec_module( package )
local = importlib.import_module( 'fonky.loco_tools' )


class LocalToolTests( unittest.TestCase ):
    """Validate one-call local automation and host-side access controls."""

    def setUp( self ) -> None:
        """Prepare an isolated root directory for every test."""
        self.temporary = tempfile.TemporaryDirectory( )
        self.workspace = Path( self.temporary.name )
        self.environment = patch.dict( os.environ, { 'FONKY_LOCAL_ROOT': str( self.workspace ),
            'FONKY_LOCAL_WRITE_ENABLED': '0',
            'FONKY_LOCAL_OUTLOOK_ENABLED': '0',
            'FONKY_LOCAL_DESKTOP_ENABLED': '0' } )
        self.environment.start( )

    def tearDown( self ) -> None:
        """Release the isolated filesystem and environment."""
        self.environment.stop( )
        self.temporary.cleanup( )

    def test_calculator_and_file_reads( self ) -> None:
        """Verify model-safe arithmetic and workspace file inspection."""
        self.assertEqual( local.local_calculate( '2 * (4 + 1)' ), 10 )
        self.assertEqual( local.local_percentage( 250, 20 ), 50 )
        (self.workspace / 'readme.txt').write_text( 'hello', encoding='utf-8' )
        self.assertEqual( local.local_file_read( 'readme.txt' ), 'hello' )
        self.assertTrue( local.local_file_exists( 'readme.txt' ) )
        self.assertEqual( len( local.local_file_hash( 'readme.txt' ) ), 64 )
        self.assertEqual( len( local.local_file_list( ) ), 1 )
        self.assertIn( 'size', local.local_file_info( 'readme.txt' ) )

    def test_paths_cannot_escape_workspace( self ) -> None:
        """Reject absolute paths and parent traversal from model-generated inputs."""
        with self.assertRaises( PermissionError ):
            local.local_file_read( str( self.workspace / 'readme.txt' ) )
        with self.assertRaises( Exception ):
            local.local_file_read( '../outside.txt' )

    def test_database_read_only( self ) -> None:
        """Read SQLite metadata and queries without permitting modifications."""
        path = self.workspace / 'data.db'
        connection = sqlite3.connect( path )
        try:
            connection.execute( 'CREATE TABLE items (value TEXT)' )
            connection.execute( 'INSERT INTO items VALUES (?)', ('demo',) )
            connection.commit( )
        finally:
            connection.close( )
        self.assertEqual( local.local_sqlite_tables( 'data.db' ), ['items'] )
        self.assertIn( 'value', local.local_sqlite_schema( 'data.db', 'items' ) )
        self.assertIn( 'demo', local.local_sqlite_query( 'data.db',
            'SELECT value FROM items WHERE value=?', '["demo"]' ) )
        with self.assertRaises( Exception ):
            local.local_sqlite_query( 'data.db', 'DELETE FROM items' )

    def test_mutations_require_host_opt_in( self ) -> None:
        """Reject writes, mailbox operations, and desktop inspection by default."""
        with self.assertRaises( PermissionError ):
            local.local_word_create( 'draft.docx', 'hello' )
        with self.assertRaises( PermissionError ):
            local.local_slides_create( 'slides.pptx' )
        with self.assertRaises( PermissionError ):
            local.local_outlook_messages( )
        with self.assertRaises( PermissionError ):
            local.local_outlook_draft( 'a@example.org', 'subject', 'body' )
        with self.assertRaises( PermissionError ):
            local.local_windows_list( )

    def test_file_backed_office_round_trip( self ) -> None:
        """Create and edit Word, Excel, and PowerPoint files in the workspace."""
        try:
            import openpyxl
            import docx
            import pptx
        except ImportError:
            self.skipTest( 'Optional Office file libraries not available' )
        os.environ[ 'FONKY_LOCAL_WRITE_ENABLED' ] = '1'
        self.assertTrue( local.local_word_create( 'report.docx', 'Budget' ) )
        self.assertIn( 'Budget', local.local_word_read( 'report.docx' ) )
        self.assertTrue( local.local_word_append( 'report.docx', 'Additional notes' ) )
        self.assertEqual( local.local_word_replace( 'report.docx', 'Budget', 'Revised' ), 1 )
        self.assertIn( 'Revised', local.local_word_read( 'report.docx' ) )
        self.assertTrue( local.local_slides_create( 'deck.pptx' ) )
        self.assertEqual( local.local_slides_add_slide( 'deck.pptx', 'Overview' ), 1 )
        self.assertTrue( local.local_slides_add_text( 'deck.pptx', 1, 'Budget update' ) )
        workbook = openpyxl.Workbook( )
        workbook.save( self.workspace / 'book.xlsx' )
        workbook.close( )
        self.assertEqual( local.local_excel_sheets( 'book.xlsx' ), ['Sheet'] )
        self.assertTrue( local.local_excel_write_range( 'book.xlsx', 'Sheet', 'A1',
            [['Amount', 100], ['Budget', 250]] ) )
        self.assertIn( '100', local.local_excel_cell( 'book.xlsx', 'Sheet', 'B1' ) )
        self.assertIn( 'Budget', local.local_excel_range( 'book.xlsx', 'Sheet', 'A1:B2' ) )
        with self.assertRaises( FileExistsError ):
            local.local_word_create( 'report.docx', 'Do not overwrite' )

    def test_host_workspace_mandatory( self ) -> None:
        """Require an explicitly configured local workspace."""
        os.environ.pop( 'FONKY_LOCAL_ROOT', None )
        with self.assertRaises( PermissionError ):
            local.local_file_list( )


class ProviderRegistrationTests( unittest.TestCase ):
    """Check all six provider modules expose the same selected local operations."""

    def test_provider_wiring( self ) -> None:
        """Verify provider-native decoration or direct callable exports."""
        expected = set( local.__all__ )
        self.assertEqual( len( expected ), 29 )
        for provider in ('gpt', 'claude', 'gemini', 'grok', 'mistral', 'langchain'):
            with self.subTest( provider=provider ):
                source = (PACKAGE_PATH / provider / 'tools.py').read_text( encoding='utf-8' )
                tree = ast.parse( source )
                assignments = { target.id: node.value
                    for node in ast.walk( tree ) if isinstance( node, ast.Assign )
                    for target in node.targets if isinstance( target, ast.Name ) }
                self.assertTrue( expected.issubset( assignments ) )
                for name in expected:
                    assignment = assignments[ name ]
                    if provider in ('gpt', 'claude', 'langchain'):
                        self.assertIsInstance( assignment, ast.Call )
                    else:
                        self.assertIsInstance( assignment, ast.Attribute )


if __name__ == '__main__':
    unittest.main( )
