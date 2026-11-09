"""
    ******************************************************************************************
        Assembly:                Fonky
        Filename:                test_astrocatalog.py
    ******************************************************************************************

    Purpose:
        Verify AstroCatalog dispatch, formatting, and error behavior without network access.
        The class is isolated from optional unrelated Fonky dependencies for focused tests.
"""
from __future__ import annotations

import ast
import io
import re
import urllib.parse
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest


class FakeError( Exception ):
    """Provide the project's error metadata contract for isolated method tests."""


class FakeTable:
    """Expose the minimal Astropy Table contract used by AstroCatalog."""

    colnames = [ 'main_id', 'magnitude' ]

    def __init__( self ) -> None:
        self.rows = [ { 'main_id': 'M 31', 'magnitude': np.float64(3.4) } ]

    def __iter__( self ):
        return iter( self.rows )

    def write( self, output: io.StringIO, format: str, overwrite: bool=True ) -> None:
        """Write representative delimited output for format contract tests."""
        output.write( 'main_id,magnitude\nM 31,3.4\n' )


def load_catalog( ) -> tuple[ Any, Mock, Mock ]:
    """Load the real AstroCatalog class from source with mocked external clients.

    Purpose:
        Executes the production class in isolation so tests do not require Fonky's many
        unrelated optional provider dependencies or live astronomical web services.

    Returns:
        tuple: Catalog class and mocked SIMBAD/VizieR client constructors.
    """
    root = Path( __file__ ).resolve( ).parents[ 1 ]
    module = ast.parse( (root / 'fetchers.py').read_text( encoding='utf-8' ) )
    class_node = next( item for item in module.body
        if isinstance( item, ast.ClassDef ) and item.name == 'AstroCatalog' )
    tree = ast.Module( body=[
        ast.ImportFrom( module='__future__',
            names=[ ast.alias( name='annotations' ) ], level=0 ),
        class_node ], type_ignores=[ ] )
    ast.fix_missing_locations( tree )

    simbad = Mock( )
    simbad.return_value.query_object.return_value = FakeTable( )
    simbad.return_value.query_region.return_value = FakeTable( )
    vizier = Mock( )
    vizier.return_value.query_object.return_value = { 'J/test/catalog': FakeTable( ) }
    vizier.return_value.query_region.return_value = { 'J/test/catalog': FakeTable( ) }

    namespace: dict[ str, Any ] = {
        'Fetcher': type( 'Fetcher', (object,), { } ),
        'Simbad': simbad,
        'Vizier': vizier,
        'SkyCoord': Mock( return_value='ICRS' ),
        'Table': FakeTable,
        'Error': FakeError,
        'np': np,
        'io': io,
        're': re,
        'urllib': urllib,
        'u': SimpleNamespace( deg=1, arcmin=1 ),
        'cfg': SimpleNamespace( AGENTS='Fonky tests' ),
        'throw_if': lambda name, value: (
            (_ for _ in [ ]).throw( ValueError( name ) ) if value is None or value == ''
            else None ),
    }
    exec( compile( tree, str( root / 'fetchers.py' ), 'exec' ), namespace )
    return namespace[ 'AstroCatalog' ], simbad, vizier


def test_simbad_object_lookup( ) -> None:
    """Confirm the default public lookup uses SIMBAD and serializes numeric values."""
    catalog_class, simbad, vizier = load_catalog( )
    result = catalog_class( ).fetch_object( 'M 31' )

    assert result[ 'provider' ] == 'simbad'
    assert result[ 'mode' ] == 'object_query'
    assert result[ 'count' ] == 1
    assert result[ 'tables' ][ 0 ][ 'rows' ][ 0 ][ 'magnitude' ] == 3.4
    simbad.return_value.query_object.assert_called_once_with( 'M 31' )
    vizier.assert_not_called( )


def test_vizier_catalog_and_cone_search( ) -> None:
    """Verify explicit catalog selection and the original cone-search entry point."""
    catalog_class, simbad, vizier = load_catalog( )
    result = catalog_class( ).cone_search(
        '10', '20', radius=3,
        arguments='provider=transient,catalog=J/test/catalog' )

    assert result[ 'provider' ] == 'vizier'
    assert result[ 'catalog' ] == 'J/test/catalog'
    assert result[ 'mode' ] == 'cone_search'
    assert result[ 'count' ] == 1
    vizier.return_value.query_region.assert_called_once_with(
        'ICRS', radius=3, catalog='J/test/catalog' )
    simbad.assert_not_called( )


def test_csv_format( ) -> None:
    """Retain structured result metadata when delimited data is requested."""
    catalog_class, simbad, vizier = load_catalog( )
    result = catalog_class( ).fetch_object( 'M 31', data_format='csv' )

    assert result[ 'formatted_tables' ][ 0 ][ 'data' ].startswith(
        'main_id,magnitude' )


def test_missing_vizier_catalog_rejected( ) -> None:
    """Reject an ambiguous catalog lookup instead of querying arbitrary catalogs."""
    catalog_class, simbad, vizier = load_catalog( )

    with pytest.raises( FakeError ):
        catalog_class( ).fetch_object( 'M 31', arguments='provider=vizier' )
    vizier.assert_not_called( )


def test_legacy_attribute_routes_are_not_silently_discarded( ) -> None:
    """Flag unsupported retired-API quantity paths explicitly."""
    catalog_class, simbad, vizier = load_catalog( )

    with pytest.raises( FakeError ):
        catalog_class( ).fetch_object( 'M 31', quantity='photometry' )
    simbad.assert_not_called( )
    vizier.assert_not_called( )
