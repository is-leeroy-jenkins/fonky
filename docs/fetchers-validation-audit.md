# Fetcher Validation and Constructor-State Audit

## Scope and decision rule

Reviewed the complete `fetchers.py` source (48 named fetcher classes, approximately 27,500 lines) for repeat guards of attributes initialized in `__init__`, redundant credential-only validators, unnecessary argument validation wrappers, and legitimate domain-specific constraints.

**Policy applied:** `throw_if(name, value)` is the shared missing/blank input guard. Do not re-check constructor-established configuration in every method. Inline required domain/range validation when consuming user input. Preserve provider-specific constraints, optional-credential semantics, and public fetch API behavior.

## Corrections included in this PR

- **PurpleAir**: remove nine duplicative validation helpers (`validate_api_key`, `validate_mode`, `validate_endpoint`, `validate_sensor_index`, `validate_longitude`, `validate_latitude`, `validate_bbox`, `validate_location_type`, `validate_non_negative_integer`). The call sites use `throw_if` for mandatory user inputs and retain local bbox, location-type, sensor-index and non-negative range rules.
- **OpenAQ**: remove credential-only `validate_api_key`; use the API-key header established in its constructor, without repeated request-time mutation. Retain its six provider-dependent validators.
- **Firms**: remove credential-only `validate_map_key`; use stored map key in its URL. Retain provider-specific source, sensor, date and area validation.
- **GoogleSearch, GoogleMaps, GoogleWeather, GovData, Congress, GoogleGeocoding, AirNow, and SpaceWeather**: remove constructor-state key/CSE guards in consuming methods.
- Add mocked regression coverage for PurpleAir sensor list, sensor detail, geographic bounds, dispatch, OpenAQ request header, and FIRMS URL construction.

## Remaining validators by class

These are **candidates for later class-wide simplification**, not a claim that they are all redundant. Several enforce real provider contracts and are intentionally retained.

| Class | Remaining `validate_*` methods |
|---|---:|
| `GoogleMaps` | 1 |
| `NavalObservatory` | 3 |
| `GlobalImagery` | 1 |
| `NearbyObjects` | 1 |
| `OpenScience` | 1 |
| `GovData` | 3 |
| `Congress` | 2 |
| `InternetArchive` | 2 |
| `Socrata` | 2 |
| `HealthData` | 2 |
| `WorldPopulation` | 2 |
| `USGSEarthquakes` | 7 |
| `USGSWaterData` | 3 |
| `USGSTheNationalMap` | 4 |
| `USGSScienceBase` | 3 |
| `ClimateData` | 3 |
| `EoNet` | 6 |
| `EnviroFacts` | 3 |
| `TidesAndCurrents` | 7 |
| `UvIndex` | 4 |
| `OpenAQ` | 6 |
| `Firms` | 6 |
| `OpenSky` | 9 |

Other classes have no `validate_*` methods. **Remaining total: 81** across 23 classes.

### Important distinctions

- `if self.api_key:` branches used to **conditionally include optional credentials** in requests or headers are not necessarily redundant validation and have not been blindly removed.
- `InternetArchive` guards `self.keywords` after deriving it from a caller parameter; it is **not** a constructor-credential guard.
- Methods validating ISO date syntax, coordinate bounds, API mode enumerations, pagination constraints, and identifier formats enforce requirements **beyond** `throw_if`; delete their wrappers only as part of a tested class-level migration.
- Public helper removal changes direct callers of the removed helper names. Fetch operations and the documented query modes are preserved; review any direct external helper usages before merging.

## Verification

This audit is based on the retrieved GitHub source and source-pattern verification. Mocked regression tests were added but **not executed** in this environment. Run Python compilation, existing tests, and new regression tests before merge. Further removal of remaining domain-specific helper methods should be reviewed in cohesive multi-class PRs, not blanket deletions.
