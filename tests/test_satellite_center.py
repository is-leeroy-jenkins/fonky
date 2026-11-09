"""Class-wide regression tests for NASA SSCWeb SatelliteCenter.

Purpose:
    Verify metadata retrieval, location request construction, input validation,
    provider status handling, dispatcher operation and error diagnostics with
    all external services mocked.
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


class SatelliteCenterTests( unittest.TestCase ):
    """Cover all SatelliteCenter retrieval modes and failure paths."""

    def setUp( self ) -> None:
        """Prepare the isolated SatelliteCenter fetcher."""
        self.center = fetchers.SatelliteCenter( )

    def test_metadata_methods_preserve_response_and_state( self ) -> None:
        """Exercise both supported sscws metadata methods."""
        observatories = { 'HttpStatus': 200, 'Observatory': [ { 'Id': 'iss' } ] }
        ground_stations = { 'HttpStatus': 200, 'GroundStation': [ { 'Id': 'SPA' } ] }
        client = Mock( )
        client.get_observatories.return_value = observatories
        client.get_ground_stations.return_value = ground_stations

        with patch( 'fonky.fetchers.SscWs', return_value=client ) as constructor:
            self.assertIs( self.center.fetch_observatories( ), observatories )
            self.assertEqual( self.center.observatories, observatories[ 'Observatory' ] )
            self.assertIs( self.center.fetch_ground_stations( ), ground_stations )
            self.assertEqual( self.center.ground_stations, ground_stations[ 'GroundStation' ] )

        self.assertEqual( constructor.call_count, 2 )
        self.assertEqual( constructor.call_args.kwargs[ 'timeout' ], 20 )
        self.assertIs( self.center.result, ground_stations )

    def test_metadata_methods_reject_provider_failure( self ) -> None:
        """Do not return failed requests from sscws as successful results."""
        client = Mock( )
        client.get_observatories.return_value = { 'HttpStatus': 503,
            'ErrorMessage': 'Service unavailable' }
        client.get_ground_stations.return_value = { 'HttpStatus': 429,
            'ErrorMessage': 'Rate limited' }

        with patch( 'fonky.fetchers.SscWs', return_value=client ):
            with self.assertRaises( Error ) as caught:
                self.center.fetch_observatories( )
            self.assertIsInstance( caught.exception.exception, ValueError )

            with self.assertRaises( Error ) as caught:
                self.center.fetch_ground_stations( )
            self.assertIsInstance( caught.exception.exception, ValueError )

    def test_location_url_and_params_follow_nasa_contract( self ) -> None:
        """Confirm basic UTC timestamps, multiple spacecraft and coordinate systems."""
        response = Mock( )
        payload = { 'HttpStatus': 200, 'Data': [ { 'Id': 'mms1' } ] }
        response.json.return_value = payload

        with patch( 'fonky.fetchers.requests.get', return_value=response ) as request:
            result = self.center.fetch_locations(
                observatories='mms1,mms2', start_time='20190101T000000Z',
                end_time='20190101T001000Z', coordinate_systems='gse,geo',
                resolution_factor=2, time=12 )

        self.assertIs( result, payload )
        self.assertIs( self.center.result, payload )
        request.assert_called_once_with(
            url='https://sscweb.gsfc.nasa.gov/WS/sscr/2/locations/mms1,mms2/'
                '20190101T000000Z,20190101T001000Z/gse,geo/',
            params={ 'resolutionFactor': 2 }, headers=self.center.headers, timeout=12 )

    def test_invalid_location_inputs_make_no_http_request( self ) -> None:
        """Reject malformed timestamps, invalid ranges and unsafe URL segments."""
        cases = [
            { 'observatories': '' },
            { 'observatories': 'iss/../other' },
            { 'observatories': 'iss?token=x' },
            { 'start_time': '2019-01-01T00:00:00Z' },
            { 'start_time': '20190230T000000Z' },
            { 'end_time': '20190101T000000Z' },
            { 'coordinate_systems': 'unknown' },
            { 'coordinate_systems': 'gse,unknown' },
            { 'resolution_factor': 0 },
            { 'time': 0 },
        ]
        standard = { 'observatories': 'iss', 'start_time': '20190101T000000Z',
            'end_time': '20190101T001000Z', 'coordinate_systems': 'gse',
            'resolution_factor': 1, 'time': 20 }

        with patch( 'fonky.fetchers.requests.get' ) as request:
            for override in cases:
                with self.subTest( override=override ):
                    values = standard.copy( )
                    values.update( override )
                    with self.assertRaises( Error ):
                        self.center.fetch_locations( **values )

            request.assert_not_called( )

    def test_locations_reject_provider_failure_or_invalid_json( self ) -> None:
        """Guard against non-200 provider statuses and unexpected response bodies."""
        for payload in ({ 'HttpStatus': 503, 'ErrorMessage': 'Unavailable' },
                [ 'unexpected' ]):
            with self.subTest( payload=payload ):
                response = Mock( )
                response.json.return_value = payload
                with patch( 'fonky.fetchers.requests.get', return_value=response ):
                    with self.assertRaises( Error ) as caught:
                        self.center.fetch_locations(
                            observatories='iss', start_time='20190101T000000Z',
                            end_time='20190101T001000Z' )
                self.assertIsInstance( caught.exception.exception, ValueError )

    def test_dispatcher_preserves_supported_modes( self ) -> None:
        """Confirm each dispatcher branch delegates to the expected public method."""
        with patch.object( self.center, 'fetch_observatories',
                return_value={ 'mode': 'observatories' } ):
            self.assertEqual( self.center.fetch( mode='observatories' ),
                { 'mode': 'observatories' } )

        with patch.object( self.center, 'fetch_ground_stations',
                return_value={ 'mode': 'ground_stations' } ):
            self.assertEqual( self.center.fetch( mode='ground_stations' ),
                { 'mode': 'ground_stations' } )

        with patch.object( self.center, 'fetch_locations',
                return_value={ 'mode': 'locations' } ) as locations:
            self.assertEqual( self.center.fetch( mode='locations', query='iss',
                start_time='20190101T000000Z', end_time='20190101T001000Z' ),
                { 'mode': 'locations' } )
            self.assertEqual( locations.call_args.kwargs[ 'observatories' ], 'iss' )

        with self.assertRaises( Error ) as caught:
            self.center.fetch( mode='unsupported' )
        self.assertIn( 'mode: str', caught.exception.method )

    def test_public_members_and_signatures( self ) -> None:
        """Check that instance state and explicit diagnostic parameters are exposed."""
        self.assertIn( 'observatories', dir( self.center ) )
        self.assertIn( 'ground_stations', dir( self.center ) )
        with self.assertRaises( Error ) as caught:
            self.center.fetch_locations( observatories='', start_time='',
                end_time='' )
        self.assertIn( 'observatories: str', caught.exception.method )
        self.assertNotIn( '*params', caught.exception.method )


if __name__ == '__main__':
    unittest.main( )
