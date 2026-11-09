"""Regression tests for NASA GIBS Mercator CRS validation.

Purpose:
    Preserve the existing Web Mercator request contract, reject unsupported
    coordinate systems, and ensure no network request is made for invalid CRS.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


PACKAGE_PATH = Path( __file__ ).resolve( ).parents[ 1 ]
PACKAGE_SPEC = importlib.util.spec_from_file_location(
    'fonky', PACKAGE_PATH / '__init__.py', submodule_search_locations=[ str( PACKAGE_PATH ) ] )
package = importlib.util.module_from_spec( PACKAGE_SPEC )
sys.modules[ 'fonky' ] = package
PACKAGE_SPEC.loader.exec_module( package )
fetchers = importlib.import_module( 'fonky.fetchers' )
Error = importlib.import_module( 'fonky.boogr' ).Error


class MercatorMapTests( unittest.TestCase ):
    """Verify GlobalImagery handles explicit CRS specifications correctly."""

    def test_default_projection_preserves_existing_behavior( self ) -> None:
        """Confirm the no-argument call requests the existing EPSG:3857 imagery."""
        imagery = fetchers.GlobalImagery( )

        with patch.object( imagery, 'fetch_wms_map', return_value={ 'ok': True } ) as request:
            result = imagery.fetch_mercator_map( )

        self.assertEqual( result, { 'ok': True } )
        self.assertEqual( imagery.projection, 'epsg3857' )
        self.assertEqual( request.call_args.kwargs[ 'projection' ], 'epsg3857' )
        self.assertEqual( request.call_args.kwargs[ 'bbox' ],
            (-8000000.0, -8000000.0, 8000000.0, 8000000.0) )

    def test_equivalent_mercator_references_are_accepted( self ) -> None:
        """Accept EPSG:3857 strings, integers, and a matching CRS object."""
        supported = [ 'EPSG:3857', 'epsg3857', 3857 ]
        projected_crs = Mock( )
        projected_crs.to_epsg.return_value = 3857
        supported.append( projected_crs )

        for projection in supported:
            with self.subTest( projection=projection ):
                imagery = fetchers.GlobalImagery( )

                with patch.object( imagery, 'fetch_wms_map',
                        return_value={ 'ok': True } ) as request:
                    imagery.fetch_mercator_map( ccrs=projection )

                self.assertEqual( request.call_args.kwargs[ 'projection' ], 'epsg3857' )

    def test_unsupported_crs_is_rejected_before_request( self ) -> None:
        """Prevent mismatched projections from reaching the EPSG:3857 WMS endpoint."""
        unsupported = [ 'EPSG:4326', 4326 ]
        projected_crs = Mock( )
        projected_crs.to_epsg.return_value = 4326
        unsupported.append( projected_crs )

        for projection in unsupported:
            with self.subTest( projection=projection ):
                imagery = fetchers.GlobalImagery( )

                with patch.object( imagery, 'fetch_wms_map' ) as request:
                    with self.assertRaises( Error ) as caught:
                        imagery.fetch_mercator_map( ccrs=projection )

                self.assertIsInstance( caught.exception.exception, ValueError )
                request.assert_not_called( )


if __name__ == '__main__':
    unittest.main( )
