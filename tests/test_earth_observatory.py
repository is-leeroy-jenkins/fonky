"""Regression tests for the NASA EONET EarthObservatory fetcher.

Purpose:
    Exercise events, categories, sources, layers, dispatch, input validation,
    response shape, error handling, and persisted state using mocked HTTP.
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


class EarthObservatoryTests( unittest.TestCase ):
    """Verify EarthObservatory's complete public retrieval interface."""

    def setUp( self ) -> None:
        """Create an isolated EONET fetcher."""
        self.observatory = fetchers.EarthObservatory( )

    def test_events_default_query( self ) -> None:
        """Preserve standard events request filters and result structure."""
        response = Mock( )
        response.json.return_value = { 'events': [ { 'id': 'EONET_1' } ],
            'title': 'EONET Events', 'description': 'Natural events' }

        with patch( 'fonky.fetchers.requests.get', return_value=response ) as request:
            result = self.observatory.fetch_events( )

        request.assert_called_once_with(
            url='https://eonet.gsfc.nasa.gov/api/v3/events',
            params={ 'status': 'open', 'limit': 20, 'days': 30 },
            headers=self.observatory.headers, timeout=20 )
        self.assertEqual( result[ 'events' ], [ { 'id': 'EONET_1' } ] )
        self.assertEqual( self.observatory.result, result )
        self.assertEqual( self.observatory.payload, response.json.return_value )

    def test_events_date_range_overrides_days( self ) -> None:
        """Include inclusive start/end filters instead of the relative-day filter."""
        response = Mock( )
        response.json.return_value = { 'events': [ ] }

        with patch( 'fonky.fetchers.requests.get', return_value=response ) as request:
            self.observatory.fetch_events( status='all', category='wildfires',
                source='InciWeb', limit=5, days=30, start_date='2026-10-01',
                end_date='2026-10-31', time=15 )

        self.assertEqual( request.call_args.kwargs[ 'params' ], {
            'status': 'all', 'limit': 5, 'category': 'wildfires',
            'source': 'InciWeb', 'start': '2026-10-01', 'end': '2026-10-31' } )
        self.assertEqual( request.call_args.kwargs[ 'timeout' ], 15 )

    def test_categories_and_sources( self ) -> None:
        """Verify both metadata endpoints preserve their public return shapes."""
        for method, key in [('fetch_categories', 'categories'), ('fetch_sources', 'sources')]:
            with self.subTest( method=method ):
                response = Mock( )
                response.json.return_value = { key: [ { 'id': 'test' } ] }
                with patch( 'fonky.fetchers.requests.get',
                        return_value=response ) as request:
                    result = vars( type( self.observatory ) )[ method ](
                        self.observatory, time=12 )

                self.assertEqual( result[ key ], [ { 'id': 'test' } ] )
                self.assertTrue( result[ 'url' ].endswith( '/' + key ) )
                self.assertEqual( request.call_args.kwargs[ 'timeout' ], 12 )

    def test_layers_extracts_nested_nasa_categories( self ) -> None:
        """Flatten the layer lists inside NASA's documented categories structure."""
        response = Mock( )
        response.json.return_value = {
            'title': 'Layers',
            'categories': [
                { 'id': 'wildfires', 'layers': [ { 'name': 'MODIS' }, { 'name': 'VIIRS' } ] },
                { 'id': 'volcanoes', 'layers': [ { 'name': 'SO2' } ] },
            ],
        }

        with patch( 'fonky.fetchers.requests.get', return_value=response ) as request:
            result = self.observatory.fetch_layers( )

        self.assertEqual( result[ 'layers' ],
            [ { 'name': 'MODIS' }, { 'name': 'VIIRS' }, { 'name': 'SO2' } ] )
        self.assertEqual( result[ 'category' ], '' )
        self.assertEqual( request.call_args.kwargs[ 'url' ],
            'https://eonet.gsfc.nasa.gov/api/v3/layers' )

    def test_layers_category_endpoint_and_legacy_payload( self ) -> None:
        """Preserve category routing and direct-layer response compatibility."""
        response = Mock( )
        response.json.return_value = { 'layers': [ { 'name': 'MODIS' } ] }

        with patch( 'fonky.fetchers.requests.get', return_value=response ) as request:
            result = self.observatory.fetch_layers( category='wildfires' )

        self.assertEqual( result[ 'layers' ], [ { 'name': 'MODIS' } ] )
        self.assertTrue( request.call_args.kwargs[ 'url' ].endswith(
            '/layers/wildfires' ) )

    def test_invalid_events_filters_are_rejected_without_http( self ) -> None:
        """Validate filter values and inclusive date bounds before network access."""
        invalid = [
            { 'status': 'unknown' }, { 'limit': 0 }, { 'days': -1 },
            { 'start_date': '2026-11-01' }, { 'end_date': '2026-11-01' },
            { 'start_date': '2026-02-30', 'end_date': '2026-03-01' },
            { 'start_date': '2026-11-02', 'end_date': '2026-11-01' },
        ]

        with patch( 'fonky.fetchers.requests.get' ) as request:
            for options in invalid:
                with self.subTest( options=options ):
                    with self.assertRaises( Error ):
                        self.observatory.fetch_events( **options )
            request.assert_not_called( )

    def test_request_rejects_invalid_timeout_and_provider_errors( self ) -> None:
        """Reject invalid timeouts, malformed JSON and provider-reported errors."""
        with patch( 'fonky.fetchers.requests.get' ) as request:
            with self.assertRaises( Error ):
                self.observatory.fetch_categories( time=0 )
            request.assert_not_called( )

        for payload in ([ 'unexpected' ], { 'error': 'Service failure' },
                { 'categories': 'wrong type' }):
            response = Mock( )
            response.json.return_value = payload
            with self.subTest( payload=payload ):
                with patch( 'fonky.fetchers.requests.get', return_value=response ):
                    with self.assertRaises( Error ):
                        self.observatory.fetch_categories( )

    def test_layers_rejects_malformed_category( self ) -> None:
        """Reject unexpected category contents instead of silently losing layers."""
        response = Mock( )
        response.json.return_value = {
            'categories': [ { 'id': 'wildfires', 'layers': 'not a list' } ] }
        with patch( 'fonky.fetchers.requests.get', return_value=response ):
            with self.assertRaises( Error ):
                self.observatory.fetch_layers( )

    def test_dispatcher_and_schema( self ) -> None:
        """Exercise each dispatcher mode and existing AI tool schema."""
        cases = [
            ('events', 'fetch_events'), ('categories', 'fetch_categories'),
            ('sources', 'fetch_sources'), ('layers', 'fetch_layers') ]

        for mode, method in cases:
            with self.subTest( mode=mode ):
                with patch.object( self.observatory, method,
                        return_value={ 'mode': mode } ):
                    self.assertEqual( self.observatory.fetch( mode=mode ),
                        { 'mode': mode } )

        with self.assertRaises( Error ) as caught:
            self.observatory.fetch( mode='unknown' )
        self.assertIn( 'mode: str', caught.exception.method )

        schema = self.observatory.create_schema(
            function='fetch_events', tool='NASA', description='EONET events',
            parameters={ 'status': { 'type': 'string' } }, required=[ 'status' ] )
        self.assertEqual( schema[ 'name' ], 'fetch_events' )

    def test_diagnostics_report_actual_signatures( self ) -> None:
        """Check erroneous variadic metadata is no longer emitted."""
        with self.assertRaises( Error ) as caught:
            self.observatory.fetch_events( status='invalid' )
        self.assertIn( 'status: str', caught.exception.method )
        self.assertNotIn( '**kwargs', caught.exception.method )

        with self.assertRaises( Error ) as caught:
            self.observatory.fetch_layers( category='../invalid' )
        self.assertIn( 'category: str', caught.exception.method )
        self.assertNotIn( '**kwargs', caught.exception.method )


if __name__ == '__main__':
    unittest.main( )
