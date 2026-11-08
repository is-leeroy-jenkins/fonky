"""Regression tests for Google Maps geocoding and its Google Weather dependency.

Purpose:
    Verifies forward-geocoding request construction, the existing coordinate tuple
    contract, provider failures, and Google Weather's dependent coordinate resolution.
    All HTTP calls are mocked; the tests do not consume API quota.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from fonky.boogr import Error
from fonky.fetchers import GoogleMaps, GoogleWeather


def test_geocode_location_builds_request_and_preserves_state() -> None:
    """Verify successful forward geocoding.

    Purpose:
        Confirms that the address and API key are sent to Google and that the public
        tuple return value and instance members retain their established shape.

    Returns:
        None: Assertions validate request construction and coordinate state.
    """
    maps = GoogleMaps( )
    maps.api_key = 'test-google-key'
    payload = {
        'status': 'OK',
        'results': [ { 'geometry': { 'location': { 'lat': 38.8799, 'lng': -77.1068 } } } ],
    }
    response = Mock( )
    response.json.return_value = payload

    with patch( 'fonky.fetchers.requests.get', return_value=response ) as request:
        coordinates = maps.geocode_location( address='Arlington, VA' )

    assert coordinates == (38.8799, -77.1068)
    assert maps.address == 'Arlington, VA'
    assert maps.mode == 'geocode_location'
    assert maps.latitude == 38.8799
    assert maps.longitude == -77.1068
    assert maps.coordinates == coordinates
    assert maps.result == coordinates
    assert maps.payload == payload
    request.assert_called_once_with(
        url='https://maps.googleapis.com/maps/api/geocode/json',
        params={ 'address': 'Arlington, VA', 'key': 'test-google-key' },
        headers=maps.headers,
        timeout=maps.timeout,
    )


@pytest.mark.parametrize(
    'address,api_key',
    [
        ('', 'test-google-key'),
        ('   ', 'test-google-key'),
        ('Arlington, VA', None),
    ],
)
def test_geocode_location_rejects_invalid_inputs(address: str, api_key: str | None) -> None:
    """Verify missing required inputs fail before contacting Google.

    Purpose:
        Confirms the project's Error wrapper preserves the underlying ValueError
        while preventing HTTP requests for blank addresses or missing credentials.

    Args:
        address (str): Address to validate.
        api_key (str | None): API credential to validate.

    Returns:
        None: Assertions validate the error and HTTP-call behavior.
    """
    maps = GoogleMaps( )
    maps.api_key = api_key

    with patch( 'fonky.fetchers.requests.get' ) as request:
        with pytest.raises( Error ) as caught:
            maps.geocode_location( address=address )

    assert isinstance( caught.value.exception, ValueError )
    request.assert_not_called( )


@pytest.mark.parametrize(
    'payload',
    [
        { 'status': 'ZERO_RESULTS', 'results': [ ] },
        { 'status': 'REQUEST_DENIED', 'error_message': 'Invalid key', 'results': [ ] },
        { 'status': 'OK', 'results': [ ] },
        { 'status': 'OK', 'results': [ { 'geometry': { 'location': { 'lng': -77.1 } } } ] },
        [ 'unexpected response' ],
    ],
)
def test_geocode_location_rejects_invalid_provider_responses(payload: object) -> None:
    """Verify invalid geocoding payloads do not generate fabricated coordinates.

    Purpose:
        Confirms provider errors and missing geometry are surfaced through the
        existing Error wrapper rather than returned as successful coordinates.

    Args:
        payload (object): Mocked Geocoding API response.

    Returns:
        None: Assertions validate rejection of malformed or unsuccessful responses.
    """
    maps = GoogleMaps( )
    maps.api_key = 'test-google-key'
    response = Mock( )
    response.json.return_value = payload

    with patch( 'fonky.fetchers.requests.get', return_value=response ):
        with pytest.raises( Error ) as caught:
            maps.geocode_location( address='Arlington, VA' )

    assert isinstance( caught.value.exception, ValueError )


def test_google_weather_resolves_coordinates_through_google_maps() -> None:
    """Verify Google Weather can use the repaired forward-geocoding path.

    Purpose:
        Exercises GoogleWeather.resolve_coordinates with a mocked provider response
        to guard against future regressions in the dependent GoogleMaps call.

    Returns:
        None: Assertions validate the dependent coordinate state.
    """
    weather = GoogleWeather( )
    weather.gmaps.api_key = 'test-google-key'
    response = Mock( )
    response.json.return_value = {
        'status': 'OK',
        'results': [ { 'geometry': { 'location': { 'lat': 38.8799, 'lng': -77.1068 } } } ],
    }

    with patch( 'fonky.fetchers.requests.get', return_value=response ) as request:
        coordinates = weather.resolve_coordinates( address='Arlington, VA' )

    assert coordinates == (38.8799, -77.1068)
    assert weather.address == 'Arlington, VA'
    assert weather.latitude == 38.8799
    assert weather.longitude == -77.1068
    assert weather.coordinates == coordinates
    assert request.call_args.kwargs[ 'params' ][ 'address' ] == 'Arlington, VA'
