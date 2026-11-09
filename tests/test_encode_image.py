"""Regression tests for image encoding input validation.

Purpose:
    Verify that encode_image uses the shared throw_if guard for missing and
    whitespace-only paths without changing successful base64 encoding behavior.
"""

from __future__ import annotations

import base64
import importlib
import importlib.util
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
fetchers = importlib.import_module( 'fonky.fetchers' )


class EncodeImageTests( unittest.TestCase ):
    """Verify the encode_image path guard and successful byte encoding."""

    def test_encode_image_returns_base64_for_existing_file( self ) -> None:
        """Verify valid image paths still encode the original file bytes."""
        image_bytes = b'\x89PNG\r\n\x1a\n\x00\x01\x02'
        with tempfile.TemporaryDirectory( ) as directory:
            image_path = Path( directory ) / 'sample.png'
            image_path.write_bytes( image_bytes )

            encoded = fetchers.encode_image( str( image_path ) )

        self.assertEqual( encoded, base64.b64encode( image_bytes ).decode( 'utf-8' ) )

    def test_encode_image_rejects_empty_paths_before_reading( self ) -> None:
        """Check that missing and blank paths raise ValueError without disk reads."""
        for path in (None, '', '   ', '\t\n'):
            with self.subTest( path=path ):
                with patch.object( fetchers.Path, 'read_bytes' ) as reader:
                    with self.assertRaises( ValueError ) as caught:
                        fetchers.encode_image( path )

                self.assertIn( 'path', str( caught.exception ) )
                reader.assert_not_called( )

    def test_encode_image_preserves_file_not_found( self ) -> None:
        """Verify that an absent nonblank file still raises FileNotFoundError."""
        with tempfile.TemporaryDirectory( ) as directory:
            image_path = Path( directory ) / 'missing.png'

            with self.assertRaises( FileNotFoundError ):
                fetchers.encode_image( str( image_path ) )


if __name__ == '__main__':
    unittest.main( )
