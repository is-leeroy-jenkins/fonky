"""Regression tests for GoogleMaps diagnostic method signatures.

Purpose:
    Verify that wrapped failures report concrete public method signatures
    rather than nonexistent variadic parameters.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
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
Error = importlib.import_module( 'fonky.boogr' ).Error


class GoogleMapsErrorMetadataTests( unittest.TestCase ):
    """Verify diagnostic signatures without making provider requests."""

    def test_geocode_coordinates_signature( self ) -> None:
        """Check reverse-geocoding error metadata for mandatory coordinates."""
        maps = fetchers.GoogleMaps( )
        maps.api_key = 'test-key'

        with patch( 'fonky.fetchers.requests.get' ) as request:
            with self.assertRaises( Error ) as caught:
                maps.geocode_coordinates( lat=None, long=-77.1 )

        self.assertEqual( caught.exception.method,
            'geocode_coordinates( self, lat: float, long: float ) -> str | None' )
        request.assert_not_called( )

    def test_validate_address_signature( self ) -> None:
        """Check address-validation error metadata for its list argument."""
        maps = fetchers.GoogleMaps( )
        maps.api_key = 'test-key'

        with patch( 'fonky.fetchers.requests.post' ) as request:
            with self.assertRaises( Error ) as caught:
                maps.validate_address( address=[ ] )

        self.assertEqual( caught.exception.method,
            'validate_address( self, address: List[ str ] ) -> Dict[ Any, Any ] | None' )
        request.assert_not_called( )

    def test_request_directions_signature( self ) -> None:
        """Check directions error metadata for both locations and mode."""
        maps = fetchers.GoogleMaps( )
        maps.api_key = 'test-key'

        with patch( 'fonky.fetchers.requests.get' ) as request:
            with self.assertRaises( Error ) as caught:
                maps.request_directions( origin='', destination='Arlington, VA' )

        self.assertEqual( caught.exception.method,
            "request_directions( self, origin: str, destination: str, mode: str='driving' ) "
            "-> Dict[ str, Any ] | None" )
        request.assert_not_called( )

    def test_create_schema_signature( self ) -> None:
        """Check schema-building error metadata for the explicit parameter list."""
        maps = fetchers.GoogleMaps( )

        with self.assertRaises( Error ) as caught:
            maps.create_schema( function='', tool='maps', description='description',
                parameters={ 'address': { 'type': 'string' } }, required=[ 'address' ] )

        self.assertEqual( caught.exception.method,
            'create_schema( self, function: str, tool: str, description: str, '
            'parameters: dict, required: list[ str ] ) -> Dict[ str, str ] | None' )


if __name__ == '__main__':
    unittest.main( )
