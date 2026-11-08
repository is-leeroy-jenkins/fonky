"""Regression tests for the complete GoogleWeather fetcher class.

Purpose:
    Cover all public Weather API endpoints, geocoding, request validation,
    response packaging, secret redaction, and failure diagnostics without
    requiring live Google credentials or making network requests.
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


class GoogleWeatherTests( unittest.TestCase ):
    """Check all GoogleWeather public weather retrieval methods and helpers."""

    def setUp( self ) -> None:
        """Create a credentialed fetcher using fixed, non-secret test values."""
        self.weather = fetchers.GoogleWeather( )
        self.weather.api_key = 'test-weather-secret'
        self.weather.gmaps.api_key = 'test-geocoding-secret'

    def test_resolve_coordinates( self ) -> None:
        """Verify geocoding state and coordinate ordering."""
        with patch.object( self.weather.gmaps, 'geocode_location',
                return_value=(38.8799, -77.1068) ) as geocode:
            result = self.weather.resolve_coordinates( address='Arlington, VA' )

        self.assertEqual( result, (38.8799, -77.1068) )
        self.assertEqual( self.weather.coordinates, result )
        self.assertEqual( self.weather.address, 'Arlington, VA' )
        geocode.assert_called_once_with( address='Arlington, VA' )

    def test_all_weather_endpoints( self ) -> None:
        """Check URL, request mode, dimensions, and secret-free return values."""
        methods = [
            ('fetch_current', { }, 'currentConditions:lookup', 'current'),
            ('fetch_hourly_forecast', { 'hours': 6 }, 'forecast/hours:lookup',
                'hourly_forecast'),
            ('fetch_daily_forecast', { 'days': 3 }, 'forecast/days:lookup',
                'daily_forecast'),
            ('fetch_hourly_history', { 'hours': 4 }, 'history/hours:lookup',
                'hourly_history'),
            ('fetch_alerts', { }, 'publicAlerts:lookup', 'alerts'),
        ]

        for method, options, endpoint, mode in methods:
            with self.subTest( method=method ):
                response = Mock( )
                response.json.return_value = { 'test': 'weather-data' }
                with patch.object( self.weather, 'resolve_coordinates',
                        return_value=(38.8799, -77.1068) ):
                    with patch( 'fonky.fetchers.requests.get',
                            return_value=response ) as request:
                        result = vars( type( self.weather ) )[ method ](
                            self.weather, address='Arlington, VA', **options )

                self.assertEqual( result[ 'mode' ], mode )
                self.assertTrue( result[ 'url' ].endswith( endpoint ) )
                self.assertEqual( result[ 'data' ], { 'test': 'weather-data' } )
                self.assertEqual( result[ 'params' ][ 'location.latitude' ], 38.8799 )
                self.assertEqual( result[ 'params' ][ 'location.longitude' ], -77.1068 )
                self.assertNotIn( 'key', result[ 'params' ] )
                self.assertEqual( self.weather.params[ 'key' ], 'test-weather-secret' )
                self.assertEqual( request.call_args.kwargs[ 'params' ][ 'key' ],
                    'test-weather-secret' )

    def test_request_excludes_secret_from_package_response( self ) -> None:
        """Ensure both response packaging paths never disclose the API key."""
        response = Mock( )
        response.json.return_value = { 'currentTime': '2026-11-08T12:00:00Z' }

        with patch( 'fonky.fetchers.requests.get', return_value=response ):
            result = self.weather.request( path='currentConditions:lookup',
                params={ 'location.latitude': 38.8799, 'location.longitude': -77.1068 } )

        self.assertNotIn( 'key', result[ 'params' ] )
        self.assertNotIn( 'test-weather-secret', str( result ) )
        self.assertEqual( self.weather.package_response( ), result )

        self.weather.result = { }
        self.assertNotIn( 'key', self.weather.package_response( )[ 'params' ] )

    def test_request_rejects_invalid_payload( self ) -> None:
        """Reject non-object JSON instead of returning an invalid weather result."""
        response = Mock( )
        response.json.return_value = [ 'invalid' ]

        with patch( 'fonky.fetchers.requests.get', return_value=response ):
            with self.assertRaises( Error ) as caught:
                self.weather.request( path='currentConditions:lookup',
                    params={ 'location.latitude': 38.8799 } )

        self.assertIsInstance( caught.exception.exception, ValueError )
        self.assertIn( 'request( self, path: str', caught.exception.method )

    def test_request_rejects_non_dictionary_params( self ) -> None:
        """Verify malformed parameter collections never reach HTTP calls."""
        with patch( 'fonky.fetchers.requests.get' ) as request:
            with self.assertRaises( Error ) as caught:
                self.weather.request( path='currentConditions:lookup', params='wrong' )

        self.assertIsInstance( caught.exception.exception, TypeError )
        request.assert_not_called( )

    def test_forecast_ranges_fail_without_geocoding( self ) -> None:
        """Validate the established limits before geocoding or HTTP requests."""
        invalid = [
            ('fetch_hourly_forecast', { 'hours': 241 }),
            ('fetch_hourly_forecast', { 'hours': 0 }),
            ('fetch_daily_forecast', { 'days': 11 }),
            ('fetch_daily_forecast', { 'days': 0 }),
            ('fetch_hourly_history', { 'hours': 25 }),
            ('fetch_hourly_history', { 'hours': 0 }),
        ]

        with patch.object( self.weather, 'resolve_coordinates' ) as geocode:
            for method, options in invalid:
                with self.subTest( method=method, options=options ):
                    with self.assertRaises( Error ):
                        vars( type( self.weather ) )[ method ](
                            self.weather, address='Arlington, VA', **options )

            geocode.assert_not_called( )

    def test_all_fetch_methods_report_explicit_signatures( self ) -> None:
        """Verify failures no longer advertise variadic arguments."""
        methods = [ 'resolve_coordinates', 'fetch_current', 'fetch_hourly_forecast',
            'fetch_daily_forecast', 'fetch_hourly_history', 'fetch_alerts' ]

        for method in methods:
            with self.subTest( method=method ):
                with self.assertRaises( Error ) as caught:
                    vars( type( self.weather ) )[ method ]( self.weather, address='' )

                self.assertNotIn( '**kwargs', caught.exception.method )
                self.assertNotIn( '*args', caught.exception.method )
                self.assertIn( 'address: str', caught.exception.method )


if __name__ == '__main__':
    unittest.main( )
