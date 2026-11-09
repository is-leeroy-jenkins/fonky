"""Regression coverage for the file-wide redundant-validation cleanup.

Purpose:
    Verify that PurpleAir uses throw_if for caller arguments, constructor
    credentials flow to requests without repeated validators, and the
    affected environmental-data classes retain their existing behavior.
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
    'fonky', PACKAGE_PATH / '__init__.py',
    submodule_search_locations=[ str( PACKAGE_PATH ) ] )
package = importlib.util.module_from_spec( PACKAGE_SPEC )
sys.modules[ 'fonky' ] = package
PACKAGE_SPEC.loader.exec_module( package )
fetchers = importlib.import_module( 'fonky.fetchers' )
Error = importlib.import_module( 'fonky.boogr' ).Error


class PurpleAirValidationTests( unittest.TestCase ):
    """Cover the entire PurpleAir request and dispatch flow."""

    def setUp( self ) -> None:
        """Prepare a configured, isolated sensor client."""
        self.air = fetchers.PurpleAir( )
        self.air.api_key = 'test-key'
        self.air.headers[ 'X-API-Key' ] = self.air.api_key

    def test_list_sensors_and_shape_rows( self ) -> None:
        """Preserve bounding-box request fields and normalized sensor rows."""
        response = Mock( )
        response.url = 'https://api.purpleair.com/v1/sensors'
        response.json.return_value = { 'fields': ['sensor_index', 'name', 'pm2.5'],
            'data': [[123, 'Sensor One', 7.5]] }

        with patch( 'fonky.fetchers.requests.get', return_value=response ) as request:
            result = self.air.fetch_sensors( nwlng=-78, nwlat=39,
                selng=-77, selat=38, location_type=0, max_age=0 )

        self.assertEqual( result[ 'mode' ], 'sensors' )
        self.assertEqual( result[ 'rows' ][0]['Sensor Index'], 123 )
        self.assertEqual( result[ 'rows' ][0]['PM2.5'], 7.5 )
        self.assertEqual( request.call_args.kwargs['headers']['X-API-Key'], 'test-key' )
        self.assertEqual( request.call_args.kwargs['params']['nwlng'], -78.0 )

    def test_single_sensor_and_public_dispatch( self ) -> None:
        """Keep the single-sensor response schema and mode routing."""
        response = Mock( )
        response.url = 'https://api.purpleair.com/v1/sensors/123'
        response.json.return_value = { 'sensor': {
            'sensor_index': 123, 'name': 'Sensor One', 'model': 'PA-II' } }
        with patch( 'fonky.fetchers.requests.get', return_value=response ) as request:
            result = self.air.fetch( mode='sensor', sensor_index=123 )

        self.assertEqual( result['rows'][0]['Sensor Index'], 123 )
        self.assertEqual( result['rows'][0]['Model'], 'PA-II' )
        self.assertTrue( request.call_args.kwargs['url'].endswith('/sensors/123') )

    def test_missing_and_invalid_user_arguments_do_not_request( self ) -> None:
        """Reject absent inputs and invalid geographic or sensor constraints."""
        invalid = [
            lambda: self.air.fetch_sensor( sensor_index=None ),
            lambda: self.air.fetch_sensor( sensor_index=0 ),
            lambda: self.air.fetch_sensors( nwlng=None, nwlat=39, selng=-77, selat=38 ),
            lambda: self.air.fetch_sensors( nwlng=-78, nwlat=39,
                selng=-77, selat=38, location_type=2 ),
            lambda: self.air.fetch_sensors( nwlng=-77, nwlat=39,
                selng=-78, selat=38 ),
            lambda: self.air.fetch_sensors( nwlng=-78, nwlat=38,
                selng=-77, selat=39 ),
            lambda: self.air.fetch_sensors( nwlng=-78, nwlat=39,
                selng=-77, selat=38, max_age=-1 ),
            lambda: self.air.request( endpoint='invalid' ),
            lambda: self.air.request( endpoint='sensors', time=0 ),
            lambda: self.air.fetch( mode='not-supported' ),
        ]
        with patch( 'fonky.fetchers.requests.get' ) as request:
            for operation in invalid:
                with self.subTest( operation=operation ):
                    with self.assertRaises( Error ):
                        operation( )
            request.assert_not_called( )

    def test_result_packaging_and_schema( self ) -> None:
        """Retain summary, raw payload and existing tool schema output."""
        self.air.mode = 'sensor'
        self.air.url = 'https://api.purpleair.com/v1/sensors/123'
        self.air.payload = { 'sensor': { 'sensor_index': 123 } }
        result = self.air.package_response(
            rows=[{ 'Sensor Index': 123, 'PM2.5': 9.1 }], params={ 'sensor_index': 123 } )
        self.assertEqual( result['summary']['max_pm25'], 9.1 )
        self.assertIs( result['raw'], self.air.payload )
        schema = self.air.create_schema( function='fetch', tool='PurpleAir',
            description='Air quality', parameters={'mode': {'type': 'string'}},
            required=['mode'] )
        self.assertEqual( schema['name'], 'fetch' )
        self.assertNotIn( 'validate_api_key', dir( self.air ) )
        self.assertNotIn( 'validate_bbox', dir( self.air ) )


class CredentialReuseTests( unittest.TestCase ):
    """Verify constructor-supplied key headers and URLs survive cleanup."""

    def test_openaq_uses_constructor_header( self ) -> None:
        """Use the configured header without invoking an extra key validator."""
        aq = fetchers.OpenAQ( )
        aq.api_key = 'test-key'
        aq.headers['X-API-Key'] = aq.api_key
        response = Mock( )
        response.url = 'https://api.openaq.org/v3/locations'
        response.json.return_value = { 'meta': {}, 'results': [] }

        with patch( 'fonky.fetchers.requests.get', return_value=response ) as request:
            aq.request( endpoint='locations', params={ 'limit': 1 } )

        self.assertEqual( request.call_args.kwargs['headers']['X-API-Key'], 'test-key' )
        self.assertNotIn( 'validate_api_key', dir( aq ) )

    def test_firms_uses_initialized_map_key( self ) -> None:
        """Build FIRMS API paths from stored configuration without revalidating it."""
        firms = fetchers.Firms( )
        firms.map_key = 'test-map-key'
        with patch.object( firms, 'request_csv', return_value={ 'ok': True } ):
            with patch.object( firms, 'csv_to_rows', return_value=[] ):
                firms.fetch_data_availability( sensor='ALL' )

        self.assertIn( '/test-map-key/ALL', firms.url )
        self.assertNotIn( 'validate_map_key', dir( firms ) )


if __name__ == '__main__':
    unittest.main( )
