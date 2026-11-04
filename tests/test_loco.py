"""Regression tests for Fonky local automation without cloud credentials."""
from __future__ import annotations

import importlib.util
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path( __file__ ).resolve( ).parents[ 1 ] / 'loco.py'
SPEC = importlib.util.spec_from_file_location( 'fonky_loco_test', MODULE_PATH )
loco = importlib.util.module_from_spec( SPEC )
SPEC.loader.exec_module( loco )


class LocalFileTests( unittest.TestCase ):
    """Verify safe local filesystem operations."""

    def test_file_lifecycle( self ) -> None:
        """Exercise creation, reading, copying, moving, hashing, and deletion."""
        with tempfile.TemporaryDirectory( ) as directory:
            service = loco.File( directory )
            service.create_directory( 'docs' )
            self.assertEqual( service.write_text( 'docs/a.txt', 'example' ), 7 )
            self.assertEqual( service.read_text( 'docs/a.txt' ), 'example' )
            self.assertTrue( service.exists( 'docs/a.txt' ) )
            self.assertEqual( len( service.calculate_hash( 'docs/a.txt' ) ), 64 )
            self.assertEqual( len( service.list_files( 'docs' ) ), 1 )
            service.copy_file( 'docs/a.txt', 'docs/b.txt' )
            with self.assertRaises( FileExistsError ):
                service.copy_file( 'docs/a.txt', 'docs/b.txt' )
            service.move_file( 'docs/b.txt', 'docs/c.txt' )
            service.delete_file( 'docs/c.txt' )
            self.assertFalse( service.exists( 'docs/c.txt' ) )

    def test_boundary_and_archive( self ) -> None:
        """Reject traversal and archive ZIP-slip attacks."""
        with tempfile.TemporaryDirectory( ) as directory:
            service = loco.File( directory )
            with self.assertRaises( PermissionError ):
                service.read_text( '../outside.txt' )
            service.create_directory( 'source' )
            service.write_text( 'source/data.txt', 'test' )
            service.compress( 'source', 'archive.zip' )
            service.extract( 'archive.zip', 'output' )
            self.assertEqual( service.read_text( 'output/data.txt' ), 'test' )
            import zipfile
            with zipfile.ZipFile( Path( directory ) / 'bad.zip', 'w' ) as archive:
                archive.writestr( '../outside.txt', 'unsafe' )
            with self.assertRaises( PermissionError ):
                service.extract( 'bad.zip', 'output2' )


class DatabaseTests( unittest.TestCase ):
    """Verify SQLite lifecycle and transaction contracts."""

    def test_transactions_and_backup( self ) -> None:
        """Verify parameter binding, commits, rollbacks, and backup."""
        with tempfile.TemporaryDirectory( ) as directory:
            path = str( Path( directory ) / 'main.db' )
            service = loco.Data( path )
            service.execute_nonquery( 'CREATE TABLE records(id INTEGER PRIMARY KEY, name TEXT)' )
            service.execute_nonquery( 'INSERT INTO records(name) VALUES (?)', ('A',) )
            service.rollback( )
            self.assertEqual( service.list_tables( ), ['records'] )
            self.assertEqual( service.execute_query( 'SELECT * FROM records' ), [] )
            service.commit( )
            service.execute_many( 'INSERT INTO records(name) VALUES (?)', [('A',), ('B',)] )
            service.commit( )
            self.assertEqual( len( service.execute_query( 'SELECT * FROM records' ) ), 2 )
            self.assertEqual( service.get_schema( 'records' )[ 1 ][ 'name' ], 'name' )
            self.assertEqual( service.integrity_check( ), ['ok'] )
            service.backup_database( str( Path( directory ) / 'backup.db' ) )
            service.close( )

    def test_readonly_database( self ) -> None:
        """Verify SQLite read-only connection compatibility."""
        with tempfile.TemporaryDirectory( ) as directory:
            path = str( Path( directory ) / 'readonly.db' )
            db = sqlite3.connect( path )
            try:
                db.execute( 'CREATE TABLE records(name TEXT)' )
                db.execute( 'INSERT INTO records VALUES (?)', ('demo',) )
                db.commit( )
            finally:
                db.close( )
            readonly = loco.Data( path, read_only=True )
            self.assertEqual( len( readonly.execute_query( 'SELECT * FROM records' ) ), 1 )
            with self.assertRaises( PermissionError ):
                readonly.execute_nonquery( 'DELETE FROM records' )
            readonly.close( )


    def test_database_crud( self ) -> None:
        """Create and change records using guarded SQL identifiers."""
        with tempfile.TemporaryDirectory( ) as directory:
            service = loco.Data( str( Path( directory ) / 'crud.db' ) )
            service.create_table( 'items', { 'id': 'INTEGER PRIMARY KEY',
                'name': 'TEXT', 'amount': 'REAL' } )
            service.insert_record( 'items', { 'name': 'A', 'amount': 4.0 } )
            service.commit( )
            self.assertEqual( len( service.execute_query( 'SELECT * FROM items' ) ), 1 )
            self.assertEqual( service.update_record( 'items', { 'amount': 5.0 },
                'name=?', ('A',) ), 1 )
            service.commit( )
            self.assertEqual( service.execute_query( 'SELECT amount FROM items' )[ 0 ][ 'amount' ], 5.0 )
            with self.assertRaises( PermissionError ):
                service.delete_record( 'items', 'name=?', ('A',) )
            self.assertEqual( service.delete_record( 'items', 'name=?', ('A',),
                approved=True ), 1 )
            service.commit( )
            with self.assertRaises( PermissionError ):
                service.drop_table( 'items' )
            service.close( )


class CalculatorTests( unittest.TestCase ):
    """Verify calculations use no arbitrary-code evaluation."""

    def test_arithmetic( self ) -> None:
        """Check supported deterministic arithmetic."""
        service = loco.Calculator( )
        self.assertEqual( service.add( 2, 3 ), 5 )
        self.assertEqual( service.divide( 10, 2 ), 5 )
        self.assertEqual( service.percentage( 200, 10 ), 20 )
        self.assertEqual( service.evaluate_expression( '(2+3)*4' ), 20 )
        with self.assertRaises( ValueError ):
            service.evaluate_expression( '__import__("os").system("echo unsafe")' )


class ShellTests( unittest.TestCase ):
    """Verify shell access requires explicit executable allowlisting."""

    def test_allowlist( self ) -> None:
        """Allow safe commands while rejecting arbitrary executables."""
        service = loco.Shell( )
        with self.assertRaises( PermissionError ):
            service.run_command( [ sys.executable, '-V' ] )
        service = loco.Shell( [ Path( sys.executable ).name ] )
        result = service.run_command( [ sys.executable, '-c', 'print(123)' ] )
        self.assertEqual( result[ 'returncode' ], 0 )
        self.assertEqual( result[ 'stdout' ].strip( ), '123' )


class OfficeFileTests( unittest.TestCase ):
    """Verify file-backed Office workflows when optional libraries are available."""

    def test_excel( self ) -> None:
        """Create, write, read, and reopen an Excel workbook."""
        try:
            import openpyxl
        except ImportError:
            self.skipTest( 'openpyxl not installed' )
        with tempfile.TemporaryDirectory( ) as directory:
            path = str( Path( directory ) / 'sample.xlsx' )
            excel = loco.Excel( )
            excel.create_workbook( )
            excel.write_cell( 'Sheet', 'A1', 'Value' )
            excel.write_range( 'Sheet', 'A2', [[1, 2], [3, 4]] )
            self.assertEqual( excel.read_range( 'Sheet', 'A2:B3' ), [[1, 2], [3, 4]] )
            excel.save_workbook( path )
            excel.close_workbook( )
            excel.open_workbook( path )
            self.assertEqual( excel.read_cell( 'Sheet', 'A1' ), 'Value' )
            excel.close_workbook( )

    def test_word( self ) -> None:
        """Create, modify, and reopen a document."""
        try:
            import docx
        except ImportError:
            self.skipTest( 'python-docx not installed' )
        with tempfile.TemporaryDirectory( ) as directory:
            path = str( Path( directory ) / 'sample.docx' )
            word = loco.Word( )
            word.create_document( )
            word.append_text( 'Hello world' )
            self.assertEqual( word.replace_text( 'world', 'Fonky' ), 1 )
            word.save_document( path )
            word.close_document( )
            word.open_document( path )
            self.assertIn( 'Hello Fonky', word.read_document( ) )
            word.close_document( )

    def test_powerpoint( self ) -> None:
        """Create a slide and reopen the presentation."""
        try:
            import pptx
        except ImportError:
            self.skipTest( 'python-pptx not installed' )
        with tempfile.TemporaryDirectory( ) as directory:
            path = str( Path( directory ) / 'sample.pptx' )
            slides = loco.PowerPoint( )
            slides.create_presentation( )
            self.assertEqual( slides.add_slide( 'Overview' ), 1 )
            slides.save_presentation( path )
            slides.close_presentation( )
            slides.open_presentation( path )
            self.assertEqual( slides.list_slides( ), [1] )
            slides.close_presentation( )


    def test_extended_office_operations( self ) -> None:
        """Verify supported file-backed charts, tables, headers, and slide text."""
        try:
            import openpyxl
            import docx
            import pptx
        except ImportError:
            self.skipTest( 'Office file libraries not installed' )
        with tempfile.TemporaryDirectory( ) as directory:
            excel = loco.Excel( )
            excel.create_workbook( )
            excel.write_range( 'Sheet', 'A1', [['Category', 'Amount'],
                ['A', 2], ['B', 4]] )
            excel.create_table( 'Sheet', 'A1:B3', 'BudgetTable' )
            excel.create_chart( 'Sheet', 'A1:B3' )
            excel.format_range( 'Sheet', 'B2:B3', number_format='#,##0.00' )
            excel.save_workbook( str( Path( directory ) / 'extended.xlsx' ) )
            excel.close_workbook( )

            word = loco.Word( )
            word.create_document( )
            word.append_text( 'Budget' )
            word.set_header( 'Internal Report' )
            word.set_footer( 'Footer' )
            word.insert_table( 2, 2 )
            self.assertTrue( word.find_text( 'Budget' ) )
            word.save_document( str( Path( directory ) / 'extended.docx' ) )
            word.close_document( )

            slides = loco.PowerPoint( )
            slides.create_presentation( )
            slides.add_slide( 'Overview' )
            slides.add_text( 1, 'Budget analysis' )
            slides.add_table( 1, [['Year', 'Amount'], ['2026', 100]] )
            slides.add_slide( 'Temporary' )
            slides.delete_slide( 2 )
            self.assertEqual( slides.list_slides( ), [1] )
            slides.save_presentation( str( Path( directory ) / 'extended.pptx' ) )
            slides.close_presentation( )



class PublicDocumentationTests( unittest.TestCase ):
    """Verify public class metadata and exception diagnostics."""

    def test_class_public_members_documented( self ) -> None:
        """Require public Attributes and Methods in every local class docstring."""
        for cls in (loco.Outlook, loco.Word, loco.PowerPoint, loco.Excel,
            loco.Calculator, loco.File, loco.Data, loco.Windows, loco.Shell):
            with self.subTest( cls=cls.__name__ ):
                self.assertIn( 'Attributes:', cls.__doc__ )
                self.assertIn( 'Methods:', cls.__doc__ )

    def test_contextual_exception_without_logging( self ) -> None:
        """Preserve an original ValueError and annotate the failing method."""
        files = loco.File( )
        with self.assertRaises( ValueError ) as caught:
            files.resolve( '' )
        self.assertTrue( any( 'fonky.loco.File.resolve' in note
            for note in caught.exception.__notes__ ) )

    def test_constructor_configuration_not_guard_checked( self ) -> None:
        """Allow constructors to capture configuration without argument guards."""
        self.assertEqual( loco.Data( '' ).path, '' )
        self.assertEqual( loco.Excel( 'custom' ).backend, 'custom' )
        self.assertEqual( loco.Windows( 'custom' ).backend, 'custom' )



if __name__ == '__main__':
    unittest.main( )
