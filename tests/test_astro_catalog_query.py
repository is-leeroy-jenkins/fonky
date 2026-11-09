"""Class-wide regression coverage for AstroCatalog and AstroQuery.

Purpose:
    Verify initialization, astronomy service request construction, coordinate
    searches, result conversion, mode dispatch and validation without network.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch


PACKAGE_PATH = Path( __file__ ).resolve( ).parents[ 1 ]
PACKAGE_SPEC = importlib.util.spec_from_file_location(
    'fonky', PACKAGE_PATH / '__init__.py',
    submodule_search_locations=[ str( PACKAGE_PATH ) ] )
package = importlib.util.module_from_spec( PACKAGE_SPEC )
sys.modules[ 'fonky' ] = package
PACKAGE_SPEC.loader.exec_module( package )
fetchers = importlib.import_module( 'fonky.fetchers' )
Error = importlib.import_module( 'fonky.boogr' ).Error


class AstroCatalogTests( unittest.TestCase ):
    """Exercise both AstroCatalog query modes and request validation."""

    def setUp( self ) -> None:
        """Create an isolated catalog client."""
        self.catalog = fetchers.AstroCatalog( )

    def test_object_query_builds_encoded_path_and_parameters( self ) -> None:
        """Check path encoding, output format and query parameter conversion."""
        response = Mock( )
        response.headers = { 'Content-Type': 'application/json' }
        response.json.return_value = { 'event': 'SN 2014J' }

        with patch( 'fonky.fetchers.requests.get', return_value=response ) as request:
            result = self.catalog.fetch_object( name='SN 2014J',
                quantity='photometry', arguments='limit=3,sort=desc', time=11 )

        self.assertEqual( result, { 'event': 'SN 2014J' } )
        self.assertIn( '/SN%202014J/photometry', request.call_args.kwargs[ 'url' ] )
        self.assertEqual( request.call_args.kwargs[ 'params' ][ 'limit' ], '3' )
        self.assertEqual( request.call_args.kwargs[ 'params' ][ 'format' ], 'json' )
        self.assertEqual( request.call_args.kwargs[ 'timeout' ], 11 )
        self.assertIs( self.catalog.result, result )

    def test_cone_search_uses_coordinates_and_positive_radius( self ) -> None:
        """Ensure the established catalog route and selector parameters persist."""
        response = Mock( )
        response.headers = { 'Content-Type': 'text/plain' }
        response.text = 'catalog data'
        with patch( 'fonky.fetchers.requests.get', return_value=response ) as request:
            result = self.catalog.cone_search( ra='12:30:00', dec='+15:20:00',
                radius=2, attributes='ra,dec', data_format='csv' )
        self.assertEqual( result, 'catalog data' )
        self.assertEqual( request.call_args.kwargs[ 'params' ][ 'radius' ], '2' )
        self.assertEqual( request.call_args.kwargs[ 'params' ][ 'ra' ], '12:30:00' )
        self.assertIn( '/catalog/ra/dec', request.call_args.kwargs[ 'url' ] )

    def test_bad_request_inputs_make_no_network_calls( self ) -> None:
        """Reject blank names, invalid radii, formats and timeouts before HTTP."""
        with patch( 'fonky.fetchers.requests.get' ) as request:
            invalid = (
                lambda: self.catalog.fetch_object( name='' ),
                lambda: self.catalog.fetch_object( name='M31', data_format='binary' ),
                lambda: self.catalog.cone_search( ra='', dec='+15:00' ),
                lambda: self.catalog.cone_search( ra='12:00', dec='+15:00', radius=0 ),
                lambda: self.catalog.request( route='catalog', time=0 ),
            )
            for operation in invalid:
                with self.subTest( operation=operation ):
                    with self.assertRaises( Error ):
                        operation( )
            request.assert_not_called( )

    def test_dispatch_preserves_modes_and_response_error( self ) -> None:
        """Keep object/cone dispatching and report provider error dictionaries."""
        for mode, method in (('object_query', 'fetch_object'), ('cone_search', 'cone_search')):
            with self.subTest( mode=mode ):
                with patch.object( self.catalog, method, return_value={ 'ok': True } ):
                    self.assertEqual( self.catalog.fetch( mode=mode ), { 'ok': True } )

        response = Mock( )
        response.headers = { 'Content-Type': 'application/json' }
        response.json.return_value = { 'error': 'Unknown object' }
        with patch( 'fonky.fetchers.requests.get', return_value=response ):
            with self.assertRaises( Error ):
                self.catalog.fetch_object( name='Unknown' )
        with self.assertRaises( Error ):
            self.catalog.fetch( mode='invalid' )


class AstroQueryTests( unittest.TestCase ):
    """Exercise SIMBAD object, identifier and region queries with mocked clients."""

    def setUp( self ) -> None:
        """Verify construction initializes headers before membership checks."""
        self.query = fetchers.AstroQuery( )

    def test_constructor_and_table_conversion( self ) -> None:
        """Ensure constructor works and table records retain string values."""
        self.assertEqual( self.query.headers[ 'Accept' ] if 'Accept' in
            self.query.headers else 'default', 'default' )
        self.assertIn( 'User-Agent', self.query.headers )
        table = MagicMock( )
        table.colnames = [ 'main_id', 'ra' ]
        table.__iter__.return_value = iter([
            { 'main_id': 'M31', 'ra': 10.6847 } ]) )
        self.assertEqual( self.query.table_to_records( table ),
            [ { 'main_id': 'M31', 'ra': '10.6847' } ] )
        self.assertEqual( self.query.table_to_records( None ), [ ] )

    def test_object_queries(self ) -> None:
        """Verify SIMBAD object and identifier lookups preserve response schema."""
        table = MagicMock( )
        table.colnames = [ 'main_id' ]
        table.__iter__.return_value = iter([
            { 'main_id': 'M31' } ]) )
        simbad = Mock( )
        simbad.query_object.return_value = table
        simbad.query_objectids.return_value = table

        with patch( 'fonky.fetchers.Simbad', return_value=simbad ):
            found = self.query.object_search( name='M31', row_limit=5 )
            identifiers = self.query.object_ids( name='M31', row_limit=5 )

        self.assertEqual( found[ 'rows' ], [ { 'main_id': 'M31' } ] )
        self.assertEqual( identifiers[ 'mode' ], 'object_ids' )
        self.assertEqual( simbad.ROW_LIMIT, 5 )
        simbad.query_object.assert_called_once_with( 'M31' )
        simbad.query_objectids.assert_called_once_with( 'M31' )

    def test_region_search_and_unit_selection( self ) -> None:
        """Verify SkyCoord input, angular units and SIMBAD query_region call."""
        simbad = Mock( )
        simbad.query_region.return_value = None
        with patch( 'fonky.fetchers.Simbad', return_value=simbad ):
            result = self.query.region_search(
                ra='00h42m44s', dec='+41d16m09s', radius=5.0,
                radius_unit='arcmin', row_limit=25 )

        self.assertEqual( result[ 'rows' ], [ ] )
        self.assertEqual( result[ 'radius_unit' ], 'arcmin' )
        self.assertEqual( result[ 'row_limit' ], 25 )
        self.assertTrue( simbad.query_region.called )

    def test_invalid_regions_and_limits_do_not_use_simbad( self ) -> None:
        """Fail invalid radii, units and limits before constructing the client."""
        with patch( 'fonky.fetchers.Simbad' ) as client:
            invalid = (
                lambda: self.query.object_search( name='', row_limit=100 ),
                lambda: self.query.object_ids( name='M31', row_limit=0 ),
                lambda: self.query.region_search( ra='00h42m44s',
                    dec='+41d16m09s', radius=0 ),
                lambda: self.query.region_search( ra='00h42m44s',
                    dec='+41d16m09s', radius_unit='radians' ),
            )
            for operation in invalid:
                with self.subTest( operation=operation ):
                    with self.assertRaises( Error ):
                        operation( )
            client.assert_not_called( )

    def test_dispatch_preserves_all_three_modes( self ) -> None:
        """Retain every public SIMBAD fetch dispatch branch."""
        for mode, method in (('object_search', 'object_search'),
                ('object_ids', 'object_ids'), ('region_search', 'region_search')):
            with self.subTest( mode=mode ):
                with patch.object( self.query, method, return_value={ 'mode': mode } ):
                    self.assertEqual( self.query.fetch( mode=mode ), { 'mode': mode } )
        with self.assertRaises( Error ):
            self.query.fetch( mode='bad' )


if __name__ == '__main__':
    unittest.main( )
