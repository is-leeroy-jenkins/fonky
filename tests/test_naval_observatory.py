"""Regression tests for the NavalObservatory celestial-navigation fetcher.

Purpose:
    Exercise the entire class using mocked USNO responses, covering date/time,
    coordinate and timeout constraints, result persistence, dispatcher, schema,
    provider errors, and diagnostic metadata.
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


class NavalObservatoryTests( unittest.TestCase ):
    """Validate request construction and failure behavior across the class."""

    def setUp( self ) -> None:
        """Initialize a new fetcher for each regression test."""
        self.observatory = fetchers.NavalObservatory( )

    def test_validators_preserve_valid_values( self ) -> None:
        """Verify accepted date, time and decimal-degree coordinates."""
        self.assertEqual( self.observatory.validate_date( '2026-11-08' ), '2026-11-08' )
        self.assertEqual( self.observatory.validate_time( '15:03:04.123' ), '15:03:04.123' )
        self.assertEqual( self.observatory.validate_coordinates( 0.0, -77.0369 ),
            (0.0, -77.0369) )
        self.assertEqual( self.observatory.validate_coordinates( -90.0, 180.0 ),
            (-90.0, 180.0) )

    def test_invalid_dates_fail_before_request( self ) -> None:
        """Reject absent, malformed, impossible and unsupported celestial-navigation dates."""
        for date in (None, '', '   ', '2026-02-30', '1799-12-31', '2051-01-01'):
            with self.subTest( date=date ):
                with self.assertRaises( Error ):
                    self.observatory.validate_date( date )

    def test_invalid_times_fail_before_request( self ) -> None:
        """Reject absent, malformed and out-of-range UT1 values."""
        for time_value in (None, '', '   ', '24:00', '12:60', '12:30:61', 'now'):
            with self.subTest( time_value=time_value ):
                with self.assertRaises( Error ):
                    self.observatory.validate_time( time_value )

    def test_invalid_coordinates_are_rejected( self ) -> None:
        """Reject missing, out-of-range and non-finite coordinates."""
        invalid = [
            (None, -77.0), (38.0, None), (91.0, 0.0), (-91.0, 0.0),
            (0.0, 181.0), (0.0, -181.0), (float( 'nan' ), 0.0),
            (0.0, float( 'nan' )), (float( 'inf' ), 0.0),
        ]
        for latitude, longitude in invalid:
            with self.subTest( latitude=latitude, longitude=longitude ):
                with self.assertRaises( Error ):
                    self.observatory.validate_coordinates( latitude, longitude )

    def test_fetch_celnav_builds_expected_request_and_state( self ) -> None:
        """Verify documented USNO request parameters and persisted response state."""
        payload = { 'type': 'Feature', 'properties': { 'data': [ ] } }
        response = Mock( )
        response.json.return_value = payload

        with patch( 'fonky.fetchers.requests.get', return_value=response ) as request:
            result = self.observatory.fetch_celnav(
                date_value='2026-11-08', time_value='15:03', latitude=38.9072,
                longitude=-77.0369, location_label='Washington DC', time=12 )

        request.assert_called_once_with(
            url='https://aa.usno.navy.mil/api/celnav',
            params={ 'date': '2026-11-08', 'time': '15:03',
                'coords': '38.9072,-77.0369' },
            headers=self.observatory.headers, timeout=12 )
        self.assertEqual( result[ 'data' ], payload )
        self.assertEqual( result[ 'location_label' ], 'Washington DC' )
        self.assertIs( self.observatory.payload, payload )
        self.assertIs( self.observatory.result, result )
        self.assertEqual( self.observatory.timeout, 12 )

    def test_provider_errors_and_bad_json_raise( self ) -> None:
        """Ensure an HTTP 200 containing an API error never reports success."""
        for payload in ({ 'error': 'Invalid request' }, [ 'unexpected' ]):
            response = Mock( )
            response.json.return_value = payload
            with self.subTest( payload=payload ):
                with patch( 'fonky.fetchers.requests.get', return_value=response ):
                    with self.assertRaises( Error ) as caught:
                        self.observatory.fetch_celnav(
                            date_value='2026-11-08', time_value='15:03',
                            latitude=38.9, longitude=-77.0 )
                self.assertIsInstance( caught.exception.exception, ValueError )

    def test_invalid_timeout_does_not_call_provider( self ) -> None:
        """Validate the timeout before performing any HTTP request."""
        with patch( 'fonky.fetchers.requests.get' ) as request:
            with self.assertRaises( Error ) as caught:
                self.observatory.fetch_celnav(
                    date_value='2026-11-08', time_value='15:03',
                    latitude=38.9, longitude=-77.0, time=0 )

        self.assertIsInstance( caught.exception.exception, ValueError )
        request.assert_not_called( )

    def test_fetch_dispatches_celnav_and_rejects_other_modes( self ) -> None:
        """Verify the public dispatcher retains its established return contract."""
        with patch.object( self.observatory, 'fetch_celnav',
                return_value={ 'mode': 'celnav' } ) as fetcher:
            result = self.observatory.fetch(
                date_value='2026-11-08', time_value='15:03',
                latitude=38.9, longitude=-77.0 )

        self.assertEqual( result, { 'mode': 'celnav' } )
        self.assertEqual( fetcher.call_args.kwargs[ 'latitude' ], 38.9 )
        with self.assertRaises( Error ) as caught:
            self.observatory.fetch( mode='unsupported' )
        self.assertIn( 'mode: str', caught.exception.method )

    def test_create_schema_and_public_members( self ) -> None:
        """Verify the tool schema and public validation members remain accessible."""
        schema = self.observatory.create_schema(
            function='fetch_celnav', tool='USNO', description='Celestial navigation',
            parameters={ 'date_value': { 'type': 'string' } }, required=[ 'date_value' ] )
        self.assertEqual( schema[ 'name' ], 'fetch_celnav' )
        self.assertIn( 'validate_coordinates', dir( self.observatory ) )

    def test_diagnostics_report_actual_signatures( self ) -> None:
        """Reject bogus variadic signatures for all relevant public methods."""
        operations = [
            lambda: self.observatory.validate_coordinates( None, 0.0 ),
            lambda: self.observatory.fetch_celnav( '', '15:03', 0.0, 0.0 ),
            lambda: self.observatory.fetch( mode='invalid' ),
            lambda: self.observatory.create_schema(
                '', 'USNO', 'description', { 'a': { 'type': 'string' } }, [ 'a' ] ),
        ]
        for operation in operations:
            with self.subTest( operation=operation ):
                with self.assertRaises( Error ) as caught:
                    operation( )
                self.assertNotIn( '*params', caught.exception.method )
                self.assertNotIn( '**kwargs', caught.exception.method )


if __name__ == '__main__':
    unittest.main( )
