"""Class-wide regression tests for NASA JPL NearbyObjects.

Purpose:
    Verify CAD, SBDB, NHATS summary/object, fireball, dispatcher,
    input constraints, HTTP response validation and error metadata.
    All provider calls are mocked.
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


class NearbyObjectsTests( unittest.TestCase ):
    """Test every NearbyObjects retrieval mode and its failure contract."""

    def setUp( self ) -> None:
        """Create an isolated JPL fetcher for each test."""
        self.objects = fetchers.NearbyObjects( )

    def test_close_approaches_request_and_result( self ) -> None:
        """Validate dates, sorting and normalized CAD response."""
        response = Mock( )
        response.json.return_value = { 'count': '1', 'fields': [ 'des' ],
            'data': [ [ '433' ] ], 'signature': { 'version': '1.5' } }

        with patch( 'fonky.fetchers.requests.get', return_value=response ) as request:
            result = self.objects.fetch_close_approaches(
                start_date='2026-11-01', end_date='2026-11-08',
                dist_max='10LD', sort='-dist', limit=10, time=8 )

        self.assertEqual( result[ 'data' ], [ [ '433' ] ] )
        self.assertEqual( result[ 'count' ], '1' )
        self.assertEqual( self.objects.result, result )
        self.assertIs( self.objects.payload, response.json.return_value )
        self.assertTrue( request.call_args.kwargs[ 'url' ].endswith( '/cad.api' ) )
        self.assertEqual( request.call_args.kwargs[ 'timeout' ], 8 )
        self.assertEqual( request.call_args.kwargs[ 'params' ][ 'sort' ], '-dist' )

    def test_object_lookup_request( self ) -> None:
        """Preserve SBDB selector and optional data section switches."""
        response = Mock( )
        response.json.return_value = { 'object': { 'fullname': '433 Eros' } }

        with patch( 'fonky.fetchers.requests.get', return_value=response ) as request:
            result = self.objects.fetch_object_lookup(
                query='Eros', query_type='sstr',
                include_physical=True, include_close_approaches=False,
                include_discovery=True )

        self.assertEqual( result[ 'data' ][ 'object' ][ 'fullname' ], '433 Eros' )
        self.assertEqual( request.call_args.kwargs[ 'params' ][ 'sstr' ], 'Eros' )
        self.assertEqual( request.call_args.kwargs[ 'params' ][ 'ca-data' ], '0' )
        self.assertNotIn( 'ca-body', request.call_args.kwargs[ 'params' ] )

    def test_nhats_summary_and_object_modes( self ) -> None:
        """Retain supported NHATS summary and designation-based query parameters."""
        response = Mock( )
        response.json.return_value = { 'signature': { 'source': 'NHATS' },
            'data': [ ] }

        with patch( 'fonky.fetchers.requests.get', return_value=response ) as request:
            summary = self.objects.fetch_nhats_summary(
                dv=6.0, dur=360, stay=8, launch='2020-2045', h=26.0, occ=7 )
            summary_params = request.call_args.kwargs[ 'params' ].copy( )
            details = self.objects.fetch_nhats_object( designation='433' )

        self.assertEqual( summary[ 'mode' ], 'nhats_summary' )
        self.assertEqual( summary_params[ 'h' ], 26.0 )
        self.assertEqual( details[ 'mode' ], 'nhats_object' )
        self.assertEqual( self.objects.params[ 'des' ], '433' )
        self.assertNotIn( 'h', self.objects.params )

    def test_fireballs_request_and_result( self ) -> None:
        """Preserve fireball count/fields/data payload and optional date filter."""
        response = Mock( )
        response.json.return_value = { 'count': 0, 'signature': { 'version': '1.2' } }

        with patch( 'fonky.fetchers.requests.get', return_value=response ) as request:
            result = self.objects.fetch_fireballs( date_min='2026-01-01', limit=5 )

        self.assertEqual( result[ 'count' ], 0 )
        self.assertEqual( result[ 'data' ], [ ] )
        self.assertEqual( request.call_args.kwargs[ 'params' ],
            { 'limit': 5, 'date-min': '2026-01-01' } )

    def test_invalid_inputs_do_not_issue_requests( self ) -> None:
        """Reject malformed time ranges, limits, mission filters and selectors."""
        invalid = [
            lambda: self.objects.fetch_close_approaches( '', '2026-11-08' ),
            lambda: self.objects.fetch_close_approaches(
                '2026-11-09', '2026-11-08' ),
            lambda: self.objects.fetch_close_approaches(
                '2026-11-01', '2026-11-08', limit=0 ),
            lambda: self.objects.fetch_close_approaches(
                '2026-11-01', '2026-11-08', sort='bad' ),
            lambda: self.objects.fetch_close_approaches(
                '2026-11-01', '2026-11-08', dist_max='10INVALID' ),
            lambda: self.objects.fetch_object_lookup( query='Eros', query_type='bad' ),
            lambda: self.objects.fetch_nhats_summary( dv=13 ),
            lambda: self.objects.fetch_nhats_summary( dur=35 ),
            lambda: self.objects.fetch_nhats_summary( h=31 ),
            lambda: self.objects.fetch_nhats_summary( occ=9 ),
            lambda: self.objects.fetch_nhats_object( designation='' ),
            lambda: self.objects.fetch_fireballs( limit=0 ),
            lambda: self.objects.fetch_fireballs( date_min='2026-02-30' ),
            lambda: self.objects.fetch_fireballs( time=0 ),
        ]
        with patch( 'fonky.fetchers.requests.get' ) as request:
            for operation in invalid:
                with self.subTest( operation=operation ):
                    with self.assertRaises( Error ):
                        operation( )
            request.assert_not_called( )

    def test_provider_errors_and_malformed_json( self ) -> None:
        """Treat a JPL error inside HTTP 200 as an error, not data."""
        for payload in ({ 'error': 'Object not found' }, [ 'unexpected' ]):
            with self.subTest( payload=payload ):
                response = Mock( )
                response.json.return_value = payload
                with patch( 'fonky.fetchers.requests.get', return_value=response ):
                    with self.assertRaises( Error ):
                        self.objects.fetch_object_lookup( query='Eros' )

    def test_dispatcher_routes_every_mode( self ) -> None:
        """Ensure the existing mode dispatcher still calls all five methods."""
        modes = [
            ('close_approaches', 'fetch_close_approaches'),
            ('object_lookup', 'fetch_object_lookup'),
            ('nhats_summary', 'fetch_nhats_summary'),
            ('nhats_object', 'fetch_nhats_object'),
            ('fireballs', 'fetch_fireballs'),
        ]
        for mode, method in modes:
            with self.subTest( mode=mode ):
                with patch.object( self.objects, method,
                        return_value={ 'mode': mode } ):
                    self.assertEqual( self.objects.fetch( mode=mode ), { 'mode': mode } )

        with self.assertRaises( Error ) as caught:
            self.objects.fetch( mode='not_supported' )
        self.assertIn( 'mode: str', caught.exception.method )

    def test_schema_and_public_members( self ) -> None:
        """Retain schema generation and publish the new validation helper."""
        schema = self.objects.create_schema(
            function='fetch_fireballs', tool='JPL', description='Meteor observations',
            parameters={ 'limit': { 'type': 'integer' } }, required=[ 'limit' ] )
        self.assertEqual( schema[ 'name' ], 'fetch_fireballs' )
        self.assertIn( 'validate_nhats_options', dir( self.objects ) )
        self.assertIn( 'request', dir( self.objects ) )


if __name__ == '__main__':
    unittest.main( )
