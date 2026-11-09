"""Class-wide mocked regression tests for NASA OSDR and DONKI fetchers.

Purpose:
    Cover OpenScience and SpaceWeather endpoints, provider error handling,
    parameter validation, secret-free metadata, dispatch, and response state.
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


class OpenScienceTests( unittest.TestCase ):
    """Verify all public NASA OSDR query and dataset interfaces."""

    def setUp( self ) -> None:
        """Create a new OSDR fetcher."""
        self.science = fetchers.OpenScience( )

    def test_dataset_get_and_saved_response( self ) -> None:
        """Verify accession routing, parsed JSON, state and timeout."""
        response = Mock( )
        response.headers = { 'Content-Type': 'application/json' }
        response.json.return_value = { 'dataset': 'OSD-48' }
        with patch( 'fonky.fetchers.requests.get', return_value=response ) as request:
            result = self.science.fetch_dataset( accession='OSD-48', time=7 )
        self.assertEqual( result[ 'data' ], { 'dataset': 'OSD-48' } )
        self.assertIs( self.science.result, result )
        self.assertIs( self.science.payload, response.json.return_value )
        self.assertEqual( request.call_args.kwargs[ 'url' ],
            'https://visualization.osdr.nasa.gov/biodata/api/v2/dataset/OSD-48/' )
        self.assertEqual( request.call_args.kwargs[ 'timeout' ], 7 )

    def test_all_query_endpoints_submit_osdr_selectors( self ) -> None:
        """Ensure field-based filters are submitted as distinct query parameters."""
        for name in ('fetch_metadata', 'fetch_assays', 'fetch_data'):
            response = Mock( )
            response.headers = { 'Content-Type': 'application/json' }
            response.json.return_value = { 'columns': [ 'id.accession' ], 'data': [ ] }
            with self.subTest( name=name ):
                with patch( 'fonky.fetchers.requests.get', return_value=response ) as request:
                    result = vars( type( self.science ) )[ name ](
                        self.science, query='id.accession=OSD-48&file.data%20type=counts',
                        format_value='json', time=9 )
                parameters = request.call_args.kwargs[ 'params' ]
                self.assertEqual( parameters[ 'id.accession' ], 'OSD-48' )
                self.assertEqual( parameters[ 'file.data type' ], 'counts' )
                self.assertEqual( parameters[ 'format' ], 'json' )
                self.assertEqual( result[ 'data' ], response.json.return_value )
                self.assertEqual( self.science.result, result )

    def test_csv_text_response( self ) -> None:
        """Retain non-JSON data for tabular OSDR outputs."""
        response = Mock( )
        response.headers = { 'Content-Type': 'text/csv' }
        response.json.side_effect = ValueError( 'not JSON' )
        response.text = 'accession,assay\\nOSD-48,RNA-Seq'
        with patch( 'fonky.fetchers.requests.get', return_value=response ):
            result = self.science.fetch_assays(
                query='id.accession=OSD-48', format_value='csv' )
        self.assertEqual( result[ 'data' ], response.text )

    def test_invalid_inputs_and_api_errors( self ) -> None:
        """Reject bad accession, output format, timeout and API errors."""
        with patch( 'fonky.fetchers.requests.get' ) as request:
            for action in (
                lambda: self.science.fetch_dataset( accession='../etc/passwd' ),
                lambda: self.science.fetch_dataset( accession='OSD-48', time=0 ),
                lambda: self.science.fetch_metadata( query='' ),
                lambda: self.science.fetch_data( query='id.accession=OSD-48',
                    format_value='invalid' ),
            ):
                with self.assertRaises( Error ):
                    action( )
            request.assert_not_called( )
        response = Mock( )
        response.headers = { 'Content-Type': 'application/json' }
        response.json.return_value = { 'error': 'Invalid query' }
        with patch( 'fonky.fetchers.requests.get', return_value=response ):
            with self.assertRaises( Error ):
                self.science.fetch_metadata( query='id.accession=OSD-48' )

    def test_dispatcher_and_schema( self ) -> None:
        """Preserve existing mode routing and schema structure."""
        for mode, method in (('dataset', 'fetch_dataset'),
                ('metadata', 'fetch_metadata'), ('assays', 'fetch_assays'),
                ('data', 'fetch_data')):
            with self.subTest( mode=mode ):
                with patch.object( self.science, method,
                        return_value={ 'mode': mode } ):
                    self.assertEqual( self.science.fetch( mode=mode ),
                        { 'mode': mode } )
        with self.assertRaises( Error ):
            self.science.fetch( mode='invalid' )
        schema = self.science.create_schema(
            function='fetch_dataset', tool='OSDR', description='Dataset',
            parameters={ 'accession': { 'type': 'string' } },
            required=[ 'accession' ] )
        self.assertEqual( schema[ 'name' ], 'fetch_dataset' )


class SpaceWeatherTests( unittest.TestCase ):
    """Verify the complete NASA DONKI dispatcher and endpoint interface."""

    def setUp( self ) -> None:
        """Initialize the DONKI fetcher with a synthetic API key."""
        self.weather = fetchers.SpaceWeather( )
        self.weather.api_key = 'test-secret'

    def test_each_donki_mode_routes_to_provider( self ) -> None:
        """Confirm the documented endpoint mapping across all 11 modes."""
        endpoints = { 'cme': 'CME', 'cme_analysis': 'CMEAnalysis',
            'gst': 'GST', 'ips': 'IPS', 'flr': 'FLR', 'sep': 'SEP',
            'mpc': 'MPC', 'rbe': 'RBE', 'hss': 'HSS',
            'wsa_enlil': 'WSAEnlilSimulations', 'notifications': 'notifications' }
        response = Mock( )
        response.json.return_value = [ { 'activityID': 'test' } ]
        for mode, endpoint in endpoints.items( ):
            with self.subTest( mode=mode ):
                with patch( 'fonky.fetchers.requests.get', return_value=response ) as request:
                    result = self.weather.fetch(
                        mode=mode, start_date='2026-10-01', end_date='2026-10-03' )
                self.assertEqual( result[ 'endpoint' ], endpoint )
                self.assertTrue( request.call_args.kwargs[ 'url' ].endswith( '/' + endpoint ) )
                self.assertEqual( result[ 'data' ], response.json.return_value )
                self.assertEqual( self.weather.result, result )

    def test_donki_query_filters_and_credential_redaction( self ) -> None:
        """Send NASA key to provider but never include it in returned metadata."""
        response = Mock( )
        response.json.return_value = [ ]
        with patch( 'fonky.fetchers.requests.get', return_value=response ) as request:
            result = self.weather.fetch(
                mode='cme_analysis', start_date='2026-10-01',
                end_date='2026-10-03', speed=500, half_angle=30,
                catalog='SWRC_CATALOG', api_key='request-secret' )
        parameters = request.call_args.kwargs[ 'params' ]
        self.assertEqual( parameters[ 'api_key' ], 'request-secret' )
        self.assertEqual( parameters[ 'speed' ], 500 )
        self.assertEqual( parameters[ 'halfAngle' ], 30 )
        self.assertNotIn( 'api_key', result[ 'params' ] )
        self.assertNotIn( 'request-secret', str( result ) )

    def test_defaults_allow_nasa_date_defaults( self ) -> None:
        """Omit the optional dates so NASA can apply its documented defaults."""
        response = Mock( )
        response.json.return_value = [ ]
        with patch( 'fonky.fetchers.requests.get', return_value=response ) as request:
            self.weather.fetch( mode='cme' )
        self.assertNotIn( 'startDate', request.call_args.kwargs[ 'params' ] )
        self.assertNotIn( 'endDate', request.call_args.kwargs[ 'params' ] )

    def test_reject_invalid_inputs_without_network( self ) -> None:
        """Reject bad endpoints, dates, timeouts, and analysis constraints."""
        with patch( 'fonky.fetchers.requests.get' ) as request:
            for operation in (
                lambda: self.weather.fetch_endpoint( 'invalid', '', '' ),
                lambda: self.weather.fetch( mode='unknown' ),
                lambda: self.weather.fetch( mode='cme',
                    start_date='2026-10-03', end_date='2026-10-01' ),
                lambda: self.weather.fetch( mode='cme',
                    start_date='2026-10-01' ),
                lambda: self.weather.fetch( mode='cme', time=0 ),
                lambda: self.weather.fetch( mode='cme_analysis', speed=-1 ),
                lambda: self.weather.fetch( mode='cme_analysis', half_angle=181 ),
            ):
                with self.assertRaises( Error ):
                    operation( )
            request.assert_not_called( )

    def test_provider_error_and_schema( self ) -> None:
        """Reject embedded provider errors and preserve tool metadata."""
        response = Mock( )
        response.json.return_value = { 'error': 'Invalid API key' }
        with patch( 'fonky.fetchers.requests.get', return_value=response ):
            with self.assertRaises( Error ):
                self.weather.fetch( mode='cme' )
        schema = self.weather.create_schema(
            function='fetch', tool='DONKI', description='Space weather',
            parameters={ 'mode': { 'type': 'string' } }, required=[ 'mode' ] )
        self.assertEqual( schema[ 'name' ], 'fetch' )


if __name__ == '__main__':
    unittest.main( )
