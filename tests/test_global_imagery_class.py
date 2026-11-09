"""Regression coverage for NASA GIBS GlobalImagery class.

Purpose:
    Verify WMS 1.1.1/1.3.0 projection contracts, GetCapabilities,
    image download and storage, validation, convenience methods,
    and tool-schema behavior without live NASA service requests.
"""

from __future__ import annotations

import importlib
import importlib.util
import tempfile
import unittest
import urllib.parse
from pathlib import Path
from unittest.mock import Mock, patch


PACKAGE_PATH = Path( __file__ ).resolve( ).parents[ 1 ]
PACKAGE_SPEC = importlib.util.spec_from_file_location(
    'fonky', PACKAGE_PATH / '__init__.py',
    submodule_search_locations=[ str( PACKAGE_PATH ) ] )
package = importlib.util.module_from_spec( PACKAGE_SPEC )
import sys
sys.modules[ 'fonky' ] = package
PACKAGE_SPEC.loader.exec_module( package )
fetchers = importlib.import_module( 'fonky.fetchers' )
Error = importlib.import_module( 'fonky.boogr' ).Error


class GlobalImageryTests( unittest.TestCase ):
    """Exercise every public GlobalImagery method and supported WMS projection."""

    def setUp( self ) -> None:
        """Prepare an isolated NASA GIBS client."""
        self.imagery = fetchers.GlobalImagery( )

    def test_get_capabilities_for_all_gibs_projections( self ) -> None:
        """Use the official four GIBS WMS projection endpoints."""
        for projection in ('epsg4326', 'epsg3857', 'epsg3413', 'epsg3031'):
            with self.subTest( projection=projection ):
                url = self.imagery.get_capabilities_url(
                    projection=projection, version='1.3.0' )
                parsed = urllib.parse.urlparse( url )
                params = urllib.parse.parse_qs( parsed.query )
                self.assertIn( '/wms/' + projection + '/best/wms.cgi', parsed.path )
                self.assertEqual( params[ 'REQUEST' ], [ 'GetCapabilities' ] )
                self.assertEqual( params[ 'VERSION' ], [ '1.3.0' ] )

    def test_wms_111_crs_and_bbox_are_correct( self ) -> None:
        """Keep SRS and west-south-east-north bbox ordering for WMS 1.1.1."""
        for projection in ('epsg4326', 'epsg3857', 'epsg3413', 'epsg3031'):
            with self.subTest( projection=projection ):
                url = self.imagery.build_wms_url(
                    layer='MODIS_Terra_CorrectedReflectance_TrueColor',
                    image_date='2021-09-21', bbox=(-30, -20, 30, 20),
                    projection=projection )
                params = urllib.parse.parse_qs( urllib.parse.urlparse( url ).query )
                self.assertEqual( params[ 'SRS' ],
                    [ 'EPSG:' + projection[ 4: ] ] )
                self.assertNotIn( 'CRS', params )
                self.assertEqual( params[ 'BBOX' ], [ '-30.0,-20.0,30.0,20.0' ] )
                self.assertEqual( self.imagery.params[ 'WIDTH' ], 1200 )

    def test_wms_130_geographic_axis_order( self ) -> None:
        """Apply WMS 1.3.0 latitude-first axis ordering only to EPSG:4326."""
        geographic = self.imagery.build_wms_url(
            layer='MODIS_Terra_CorrectedReflectance_TrueColor',
            image_date='2021-09-21', bbox=(-30, -20, 30, 20), version='1.3.0' )
        params = urllib.parse.parse_qs( urllib.parse.urlparse( geographic ).query )
        self.assertEqual( params[ 'CRS' ], [ 'EPSG:4326' ] )
        self.assertEqual( params[ 'BBOX' ], [ '-20.0,-30.0,20.0,30.0' ] )
        self.assertNotIn( 'SRS', params )

        projected = self.imagery.build_wms_url(
            layer='MODIS_Terra_CorrectedReflectance_TrueColor',
            image_date='2021-09-21', bbox=(-30, -20, 30, 20),
            projection='epsg3857', version='1.3.0' )
        params = urllib.parse.parse_qs( urllib.parse.urlparse( projected ).query )
        self.assertEqual( params[ 'BBOX' ], [ '-30.0,-20.0,30.0,20.0' ] )

    def test_bad_wms_options_are_rejected( self ) -> None:
        """Reject unsupported projections, versions and malformed bounding boxes."""
        arguments = [
            { 'projection': 'epsg9999' },
            { 'version': '1.0.0' },
            { 'quality': '../best' },
            { 'bbox': (-30, -20, 30) },
            { 'bbox': (30, -20, -30, 20) },
            { 'bbox': (-30, -20, float( 'nan' ), 20) },
            { 'width': 0 },
            { 'height': 0 },
            { 'image_format': 'text/html' },
            { 'layer': '../illegal' },
        ]
        standard = { 'layer': 'MODIS_Terra_CorrectedReflectance_TrueColor',
            'image_date': '2021-09-21', 'bbox': (-30, -20, 30, 20) }
        for arguments_override in arguments:
            with self.subTest( arguments=arguments_override ):
                options = standard.copy( )
                options.update( arguments_override )
                with self.assertRaises( Error ):
                    self.imagery.build_wms_url( **options )

    def test_fetch_wms_map_saves_valid_image_and_metadata( self ) -> None:
        """Download valid mocked bytes after response checks and preserve result shape."""
        response = Mock( )
        response.headers = { 'Content-Type': 'image/png; charset=binary' }
        response.content = b'\x89PNG\r\n\x1a\nDATA'
        response.status_code = 200

        with tempfile.TemporaryDirectory( ) as directory:
            with patch( 'fonky.fetchers.requests.get', return_value=response ) as request:
                result = self.imagery.fetch_wms_map(
                    layer='MODIS_Terra_CorrectedReflectance_TrueColor',
                    image_date='2021-09-21', bbox=(-30, -20, 30, 20),
                    output_dir=directory, output_name='image.png', time=7 )

            self.assertEqual( Path( result[ 'image_path' ] ).read_bytes( ),
                response.content )
            self.assertEqual( result[ 'bytes' ], len( response.content ) )
            self.assertEqual( result[ 'mode' ], 'wms_map' )
            self.assertEqual( result[ 'bbox' ][ 'west' ], -30.0 )
            self.assertEqual( request.call_args.kwargs[ 'timeout' ], 7 )
            self.assertEqual( self.imagery.result, result )

    def test_bad_image_and_path_are_rejected_without_file_writes( self ) -> None:
        """Block unsafe filename traversal, missing images and invalid timeouts."""
        with tempfile.TemporaryDirectory( ) as directory:
            arguments = { 'layer': 'MODIS_Terra_CorrectedReflectance_TrueColor',
                'image_date': '2021-09-21', 'bbox': (-30, -20, 30, 20),
                'output_dir': directory }
            with patch( 'fonky.fetchers.requests.get' ) as request:
                for output_name in ('../escape.png', '..\\escape.png', '/tmp/escape.png'):
                    with self.subTest( output_name=output_name ):
                        with self.assertRaises( Error ):
                            self.imagery.fetch_wms_map(
                                **arguments, output_name=output_name )
                with self.assertRaises( Error ):
                    self.imagery.fetch_wms_map( **arguments, time=0 )
                request.assert_not_called( )

            response = Mock( )
            response.headers = { 'Content-Type': 'text/xml' }
            response.text = '<ServiceExceptionReport />'
            response.content = b'<ServiceExceptionReport />'
            with patch( 'fonky.fetchers.requests.get', return_value=response ):
                with self.assertRaises( Error ):
                    self.imagery.fetch_wms_map(
                        **arguments, output_name='should-not-exist.png' )
            self.assertFalse( (Path( directory ) / 'should-not-exist.png').exists( ) )

    def test_map_service_convenience_methods( self ) -> None:
        """Preserve the established default and Web Mercator request contracts."""
        with patch.object( self.imagery, 'fetch_wms_map',
                return_value={ 'ok': True } ) as fetch:
            self.assertEqual( self.imagery.fetch_map_services( ), { 'ok': True } )
            self.assertEqual( fetch.call_args.kwargs[ 'projection' ], 'epsg4326' )
            self.assertEqual( self.imagery.fetch_mercator_map( ), { 'ok': True } )
            self.assertEqual( fetch.call_args.kwargs[ 'projection' ], 'epsg3857' )

    def test_create_schema_and_introspection( self ) -> None:
        """Retain AI tool schema output and advertise all public methods."""
        schema = self.imagery.create_schema(
            function='fetch_wms_map', tool='NASA', description='GIBS map',
            parameters={ 'layer': { 'type': 'string' } }, required=[ 'layer' ] )
        self.assertEqual( schema[ 'name' ], 'fetch_wms_map' )
        self.assertIn( 'get_capabilities_url', dir( self.imagery ) )
        self.assertIn( 'validate_wms_options', dir( self.imagery ) )


if __name__ == '__main__':
    unittest.main( )
