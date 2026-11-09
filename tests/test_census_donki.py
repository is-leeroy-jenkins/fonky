"""
    ******************************************************************************************
        Assembly:                Fonky
        Filename:                test_census_donki.py
    ******************************************************************************************

    Purpose:
        Exercise Census and DONKI request security and parameter construction without
        network access, while preserving the production class methods and signatures.
"""
from __future__ import annotations

import ast
import datetime as dt
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest


class FakeError( Exception ):
    """Represent Fonky's exception wrapper for isolated integration tests."""


def throw_if( name: str, value: object ) -> None:
    """Reject missing required values as the production guard does."""
    if value is None or isinstance( value, str ) and not value.strip( ):
        raise ValueError( name )


def load_fetcher( name: str, census_key: str | None='TEST_CENSUS_KEY',
        nasa_key: str | None=None ) -> tuple[ Any, Mock ]:
    """Load one production class with mocked provider requests.

    Purpose:
        Verifies actual method request construction without importing unrelated
        integrations or contacting any external API.

    Args:
        name: Class name to isolate from fetchers.py.
        census_key: Optional configured Census credential.
        nasa_key: Optional configured NASA credential.

    Returns:
        tuple: Instantiable fetcher class and requests mock.
    """
    root = Path( __file__ ).resolve( ).parents[ 1 ]
    module = ast.parse( (root / 'fetchers.py' ).read_text( encoding='utf-8' ) )
    node = next( item for item in module.body
        if isinstance( item, ast.ClassDef ) and item.name == name )
    unit = ast.Module( body=[
        ast.ImportFrom( module='__future__',
            names=[ ast.alias( name='annotations' ) ], level=0 ), node ], type_ignores=[ ] )
    ast.fix_missing_locations( unit )

    response = Mock( )
    response.json.return_value = [
        [ 'NAME', 'B01001_001E', 'state' ],
        [ 'Virginia', '8000000', '51' ] ]
    response.url = 'https://api.census.gov/data/2022/acs/acs5?key=TEST_CENSUS_KEY'
    requests = Mock( )
    requests.get.return_value = response

    environment: dict[ str, Any ] = {
        'Fetcher': type( 'Fetcher', (object,), { } ),
        'cfg': SimpleNamespace( AGENTS='Fonky tests',
            CENSUS_API_KEY=census_key, NASA_API_KEY=nasa_key ),
        'requests': requests, 'dt': dt,
        'throw_if': throw_if, 'Error': FakeError,
    }
    exec( compile( unit, str( root / 'fetchers.py' ), 'exec' ), environment )
    return environment[ name ], requests


def test_census_data_key_sent_but_not_returned( ) -> None:
    """Keep Census authentication out of result URLs and parameters."""
    census_class, requests = load_fetcher( 'CensusData' )
    result = census_class( ).fetch_data(
        year='2022', dataset='acs/acs5', fields='NAME,B01001_001E',
        geography_for='state:*' )
    query = requests.get.call_args.kwargs[ 'params' ]

    assert query[ 'key' ] == 'TEST_CENSUS_KEY'
    assert result[ 'params' ][ 'get' ] == 'NAME,B01001_001E'
    assert 'key' not in result[ 'params' ]
    assert 'TEST_CENSUS_KEY' not in result[ 'url' ]
    assert result[ 'data' ][ 'count' ] == 1


def test_census_variables_need_no_key( ) -> None:
    """Verify metadata remains accessible without a configured credential."""
    census_class, requests = load_fetcher( 'CensusData', census_key=None )
    result = census_class( ).fetch_variables( year='2022', dataset='acs/acs5' )

    assert 'key' not in requests.get.call_args.kwargs[ 'params' ]
    assert result[ 'mode' ] == 'variables'


def test_census_data_requires_configured_key( ) -> None:
    """Prevent misleading upstream authorization failures when key is missing."""
    census_class, requests = load_fetcher( 'CensusData', census_key=None )

    with pytest.raises( FakeError ):
        census_class( ).fetch_data(
            year='2022', dataset='acs/acs5',
            fields='NAME', geography_for='state:*' )
    requests.get.assert_not_called( )


@pytest.mark.parametrize( 'mode,endpoint', [
    ( 'cme', 'CME' ),
    ( 'cme_analysis', 'CMEAnalysis' ),
    ( 'gst', 'GST' ),
    ( 'ips', 'IPS' ),
    ( 'flr', 'FLR' ),
    ( 'sep', 'SEP' ),
    ( 'mpc', 'MPC' ),
    ( 'rbe', 'RBE' ),
    ( 'hss', 'HSS' ),
    ( 'wsa_enlil', 'WSAEnlilSimulations' ),
    ( 'notifications', 'notifications' ) ] )
def test_donki_all_modes_allow_provider_default_dates(
        mode: str, endpoint: str ) -> None:
    """Check each documented DONKI route without imposing required dates."""
    space_class, requests = load_fetcher( 'SpaceWeather' )
    requests.get.return_value.json.return_value = [ ]
    result = space_class( ).fetch( mode=mode )

    query = requests.get.call_args.kwargs[ 'params' ]
    assert requests.get.call_args.kwargs[ 'url' ].endswith( '/' + endpoint )
    assert query[ 'api_key' ] == 'DEMO_KEY'
    assert 'startDate' not in query
    assert 'endDate' not in query
    assert result[ 'endpoint' ] == endpoint
    assert 'api_key' not in result[ 'params' ]


def test_donki_uses_configured_key_and_dates( ) -> None:
    """Ensure explicitly supplied dates and credentials are retained."""
    space_class, requests = load_fetcher(
        'SpaceWeather', nasa_key='TEST_NASA_KEY' )
    requests.get.return_value.json.return_value = [ ]
    space_class( ).fetch(
        mode='flr', start_date='2026-01-01', end_date='2026-01-02' )
    query = requests.get.call_args.kwargs[ 'params' ]

    assert query[ 'api_key' ] == 'TEST_NASA_KEY'
    assert query[ 'startDate' ] == '2026-01-01'
    assert query[ 'endDate' ] == '2026-01-02'


def test_donki_rejects_one_sided_date_range( ) -> None:
    """Prevent a caller from submitting an incomplete date interval."""
    space_class, requests = load_fetcher( 'SpaceWeather' )

    with pytest.raises( FakeError ):
        space_class( ).fetch( mode='cme', start_date='2026-01-01' )
    requests.get.assert_not_called( )
