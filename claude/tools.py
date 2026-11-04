'''
  ******************************************************************************************
      Assembly:                Fonky
      Filename:                tools.py
      Author:                  Terry D. Eppler
      Created:                 09-02-2026

      Last Modified By:        Terry D. Eppler
      Last Modified On:        09-02-2026
  ******************************************************************************************
  <summary>
    Provides the Anthropic Claude function-tool interface for Fonky.

    Purpose:
        Exposes individual Anthropic Claude function tools over implementation classes in
        ``fetchers.py``, ``loaders.py``, ``scrapers.py``, and ``processors.py``. Each public
        operation is decorated with ``@beta_tool`` so the Anthropic SDK can derive its
        provider schema from the callable signature and documentation, execute the delegated Fonky
        implementation, and expose the result through Claude client-tool execution.
  </summary>
  ******************************************************************************************
'''
from __future__ import annotations

from anthropic import beta_tool
from typing import Any, Dict, List, Optional, Tuple
from sentence_transformers import SentenceTransformer
import datetime as dt
import numpy as np
from pandas import DataFrame, Series

# ==========================================================================================
# FETCHERS IMPORTS
# ==========================================================================================

from ..fetchers import AirNow
from ..fetchers import ArXiv
from ..fetchers import AstroCatalog
from ..fetchers import AstroQuery
from ..fetchers import CensusData
from ..fetchers import ClimateData
from ..fetchers import Congress
from ..fetchers import EarthObservatory
from ..fetchers import EnviroFacts
from ..fetchers import EoNet
from ..fetchers import Firms
from ..fetchers import GlobalHealthData
from ..fetchers import GlobalImagery
from ..fetchers import GoogleDrive
from ..fetchers import GoogleGeocoding
from ..fetchers import GoogleMaps
from ..fetchers import GoogleSearch
from ..fetchers import GoogleWeather
from ..fetchers import GovData
from ..fetchers import Grokipedia
from ..fetchers import HealthData
from ..fetchers import HistoricalWeather
from ..fetchers import InternetArchive
from ..fetchers import NavalObservatory
from ..fetchers import NearbyObjects
from ..fetchers import OpenAQ
from ..fetchers import OpenScience
from ..fetchers import OpenSky
from ..fetchers import OpenWeather
from ..fetchers import PurpleAir
from ..fetchers import SatelliteCenter
from ..fetchers import Socrata
from ..fetchers import SpaceWeather
from ..fetchers import StarChart
from ..fetchers import StarMap
from ..fetchers import TheNews
from ..fetchers import TidesAndCurrents
from ..fetchers import USGSEarthquakes
from ..fetchers import USGSScienceBase
from ..fetchers import USGSTheNationalMap
from ..fetchers import USGSWaterData
from ..fetchers import UnitedNations
from ..fetchers import UvIndex
from ..fetchers import WebCrawler
from ..fetchers import WebFetcher
from ..fetchers import Wikipedia
from ..fetchers import Wonder
from ..fetchers import WorldPopulation
from ..fetchers import encode_image as _encode_image

# --------- LOADERS IMPORTS ---------

from ..loaders import ArXivLoader
from ..loaders import AwsBucketLoader
from ..loaders import AwsFileLoader
from ..loaders import CsvLoader
from ..loaders import EmailLoader
from ..loaders import ExcelLoader
from ..loaders import GithubLoader
from ..loaders import GoogleBucketLoader
from ..loaders import GoogleCloudFileLoader
from ..loaders import GoogleDriveLoader
from ..loaders import GoogleSpeechToTextLoader
from ..loaders import HtmlLoader
from ..loaders import JsonLoader
from ..loaders import JupyterNotebookLoader
from ..loaders import MarkdownLoader
from ..loaders import OneDriveDocLoader
from ..loaders import OpenCityLoader
from ..loaders import OutlookLoader
from ..loaders import PdfLoader
from ..loaders import PdfReader
from ..loaders import PowerPointLoader
from ..loaders import PubMedSearchLoader
from ..loaders import SpfxLoader
from ..loaders import TextLoader
from ..loaders import WebLoader
from ..loaders import WikiLoader
from ..loaders import WordLoader
from ..loaders import XmlLoader

# --------- SCRAPERS IMPORTS ---------

from ..scrapers import WebExtractor

# --------- PROCESSORS IMPORTS ---------

from ..processors import NltkParser
from ..processors import TextParser


# ==========================================================================================
# RESEARCH, SEARCH, AND KNOWLEDGE TOOLS
# ==========================================================================================

@beta_tool
def fetch_arxiv( question: str, max_documents: int | None=None, full_documents: bool | None=None,
		include_metadata: bool | None=None ) -> Any:
    """Retrieve ArXiv research documents.

    Purpose:
        Retrieve ArXiv research documents through ArXiv. The query text determines the records or documents matched by the provider. Result-count arguments bound the amount of data requested. Boolean options control retrieval depth or supplemental content.

    Args:
        question (str): Search text, lookup value, or provider query submitted by the caller.
        max_documents (int | None): Maximum number of documents to retrieve.
        full_documents (bool | None): Whether to retrieve full document content instead of abbreviated search results.
        include_metadata (bool | None): Whether provider metadata should be included with retrieved content.

    Returns:
        Any: LangChain documents loaded from the requested source.

    Raises:
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = ArXiv( )
    return _instance.fetch( question=question, max_documents=max_documents,
	    full_documents=full_documents, include_metadata=include_metadata )

@beta_tool
def fetch_google_drive( question: str, folder_id: str='root', results: int=10,
		template: str='gdrive-query', mime_type: str | None=None, mode: str='documents' ) -> Any:
    """Retrieve Google Drive documents.

    Purpose:
        Retrieve Google Drive documents through Google Drive. The query text determines the records or documents matched by the provider. Result-count arguments bound the amount of data requested.

    Args:
        question (str): Search text, lookup value, or provider query submitted by the caller.
        folder_id (str): Provider folder identifier that scopes the operation.
        results (int): Maximum number of search results to request.
        template (str): Provider query template used to construct the request.
        mime_type (str | None): Optional MIME type used to restrict matching files.
        mode (str): Operation mode used to select the provider or processing workflow.

    Returns:
        Any: LangChain documents loaded from the requested source.

    Raises:
        ValueError: If a required value is missing, blank, or outside the supported range.
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = GoogleDrive( )
    return _instance.fetch( question=question, folder_id=folder_id, results=results,
	    template=template, mime_type=mime_type, mode=mode )

@beta_tool
def fetch_wikipedia( question: str, language: str | None=None, max_documents: int | None=None,
		include_metadata: bool | None=None ) -> Any:
    """Retrieve Wikipedia documents.

    Purpose:
        Retrieve Wikipedia documents through Wikipedia. The query text determines the records or documents matched by the provider. Result-count arguments bound the amount of data requested. Boolean options control retrieval depth or supplemental content.

    Args:
        question (str): Search text, lookup value, or provider query submitted by the caller.
        language (str | None): Language code used for provider results or parsing.
        max_documents (int | None): Maximum number of documents to retrieve.
        include_metadata (bool | None): Whether provider metadata should be included with retrieved content.

    Returns:
        Any: LangChain documents loaded from the requested source.

    Raises:
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = Wikipedia( )
    return _instance.fetch( question=question, language=language, max_documents=max_documents,
	    include_metadata=include_metadata )

@beta_tool
def fetch_news( endpoint: str='all', query: str='', language: str='en', categories: str='',
		exclude_categories: str='', locale: str='', domains: str='', exclude_domains: str='',
		source_ids: str='', exclude_source_ids: str='', published_after: str='',
		published_before: str='', published_on: str='', sort: str='published_at',
		limit: int=10, page: int=1, include_similar: bool=True,
		headlines_per_category: int=6, time: int=10, api_key: str | None=None ) -> Any:
    """Retrieve The News API article.

    Purpose:
        Retrieve The News API article through The News API. The query text determines the records
        or documents matched by the provider. Date and time arguments constrain the requested
        interval when supplied. Result-count arguments bound the amount of data requested.
        Boolean options control retrieval depth or supplemental content. When supplied, ``api_key``
        overrides the configured provider credential for this request.

    Args:
        endpoint (str): Provider endpoint or endpoint family to request.
        query (str): Search text, lookup value, or provider query submitted by the caller.
        language (str): Language code used for provider results or parsing.
        categories (str): Comma-separated news categories used to include matching articles.
        exclude_categories (str): Filter value used to exclude categories from provider results.
        locale (str): Locale filter applied to news results.
        domains (str): Comma-separated source domains used to include matching news articles.
        exclude_domains (str): Filter value used to exclude domains from provider results.
        source_ids (str): Provider identifiers for the selected source.
        exclude_source_ids (str): Filter value used to exclude source ids from provider results.
        published_after (str): Earliest publication timestamp accepted by the news query.
        published_before (str): Latest publication timestamp accepted by the news query.
        published_on (str): Specific publication date used to restrict news results.
        sort (str): Provider-supported result ordering expression.
        limit (int): Maximum number of records or items to return.
        page (int): One-based result page to request.
        include_similar (bool): Whether to include similar in the result.
        headlines_per_category (int): Maximum number of headlines returned for each category in headline mode.
        time (int): Request timeout in seconds.
        api_key (str | None): Optional credential override used for the active request.

    Returns:
        Any: Structured mapping produced by the operation.

    Raises:
        ValueError: If a required value is missing, blank, or outside the supported range.
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = TheNews( )
    return _instance.fetch( endpoint=endpoint, query=query, language=language,
	    categories=categories, exclude_categories=exclude_categories, locale=locale,
	    domains=domains, exclude_domains=exclude_domains, source_ids=source_ids,
	    exclude_source_ids=exclude_source_ids, published_after=published_after,
	    published_before=published_before, published_on=published_on, sort=sort, limit=limit,
	    page=page, include_similar=include_similar, headlines_per_category=headlines_per_category,
	    time=time, api_key=api_key )

@beta_tool
def fetch_cse_search( keywords: str, results: int=10, start: int=1, exact_terms: str='',
		exclude_terms: str='', file_type: str='', date_restrict: str='', gl: str='', lr: str='',
		safe: str='off', search_type: str='', site_search: str='', site_search_filter: str='',
		sort: str='', img_size: str='', img_type: str='', img_color_type: str='',
		img_dominant_color: str='', time: int=10, api_key: str | None=None, cse_id: str | None=None ) -> Any:
    """Retrieve Google Programmable Search Engine results.

    Purpose:
        Retrieve results through Google Programmable Search Engine (Custom Search JSON API). The query text determines the records or documents matched by the provider. Result-count arguments bound the amount of data requested. When supplied, ``api_key`` overrides the configured provider credential for this request.

    Args:
        keywords (str): Search text, lookup value, or provider query submitted by the caller.
        results (int): Maximum number of search results to request.
        start (int): Starting result position used for pagination.
        exact_terms (str): Phrase that must appear exactly in Google Custom Search results.
        exclude_terms (str): Terms that must not appear in Google Custom Search results.
        file_type (str): File-extension filter applied to Google Custom Search results.
        date_restrict (str): Google Custom Search date restriction expression.
        gl (str): Google country-code boost applied to search results.
        lr (str): Google language restriction expression.
        safe (str): Google SafeSearch setting.
        search_type (str): Google Custom Search result type; use the provider-supported image-search value when requesting images.
        site_search (str): Domain or site used to restrict Google Custom Search results.
        site_search_filter (str): Whether ``site_search`` is included or excluded by Google Custom Search.
        sort (str): Provider-supported result ordering expression.
        img_size (str): Image-size filter used for Google image search.
        img_type (str): Google image type filter.
        img_color_type (str): Google image color-type filter.
        img_dominant_color (str): Dominant-color filter used for Google image search.
        time (int): Request timeout in seconds.
        api_key (str | None): Optional credential override used for the active request.
        cse_id (str | None): Google Programmable Search Engine identifier.

    Returns:
        Any: Structured mapping produced by the operation.

    Raises:
        ValueError: If a required value is missing, blank, or outside the supported range.
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = GoogleSearch( )
    return _instance.fetch( keywords=keywords, results=results, start=start,
	    exact_terms=exact_terms, exclude_terms=exclude_terms, file_type=file_type,
	    date_restrict=date_restrict, gl=gl, lr=lr, safe=safe, search_type=search_type,
	    site_search=site_search, site_search_filter=site_search_filter, sort=sort,
	    img_size=img_size, img_type=img_type, img_color_type=img_color_type,
	    img_dominant_color=img_dominant_color, time=time, api_key=api_key, cse_id=cse_id )

@beta_tool
def fetch_gov_data( mode: str='search', query: str='', page_size: int=10, offset_mark: str='*',
		sort_field: str='score', sort_order: str='DESC', package_id: str='', collection: str='',
		start_date: str='', time: int=20 ) -> Any:
    """Retrieve Data.gov package and collection.

    Purpose:
        Retrieve Data.gov package and collection through Data.gov. Use ``mode`` to select among ``collection``, ``package_summary``, ``search``. The query text determines the records or documents matched by the provider. Date and time arguments constrain the requested interval when supplied. Result-count arguments bound the amount of data requested.

    Args:
        mode (str): Operation selector. Supported values detected in the implementation include ``collection``, ``package_summary``, ``search``.
        query (str): Search text, lookup value, or provider query submitted by the caller.
        page_size (int): Maximum number of records requested per page.
        offset_mark (str): Provider continuation marker used for paginated Data.gov search results.
        sort_field (str): Provider field used to order search results.
        sort_order (str): Sort direction applied to the provider search.
        package_id (str): Provider identifier for the selected package.
        collection (str): Provider collection identifier used to restrict results.
        start_date (str): Inclusive start date for the requested time range, in the provider-supported format.
        time (int): Request timeout in seconds.

    Returns:
        Any: Structured mapping produced by the operation.

    Raises:
        ValueError: If a required value is missing, blank, or outside the supported range.
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = GovData( )
    return _instance.fetch( mode=mode, query=query, page_size=page_size, offset_mark=offset_mark,
	    sort_field=sort_field, sort_order=sort_order, package_id=package_id, collection=collection,
	    start_date=start_date, time=time )

@beta_tool
def fetch_congress( mode: str='congresses', congress: int=0, bill_type: str='', bill_number: int=0,
		law_type: str='', law_number: int=0, report_type: str='', report_number: int=0,
		offset: int=0, limit: int=20, sort: str='updateDate+desc', from_date_time: str='',
		to_date_time: str='', conference: bool=False, time: int=20 ) -> Any:
    """Retrieve Congress.gov legislative data.

    Purpose:
        Retrieve Congress.gov legislative data through Congress.gov. Use ``mode`` to select among ``bill_detail``, ``bills``, ``congresses``, ``law_detail``, ``laws``, ``report_detail``, ``reports``. Date and time arguments constrain the requested interval when supplied. Result- count arguments bound the amount of data requested.

    Args:
        mode (str): Operation selector. Supported values detected in the implementation include ``bill_detail``, ``bills``, ``congresses``, ``law_detail``, ``laws``, ``report_detail``, ``reports``.
        congress (int): Congress number used to scope legislative records.
        bill_type (str): Provider type selector for bill.
        bill_number (int): Legislative bill number used with the selected Congress and bill type.
        law_type (str): Provider type selector for law.
        law_number (int): Public or private law number used with the selected law type.
        report_type (str): Provider type selector for report.
        report_number (int): Committee report number used with the selected Congress and report type.
        offset (int): Zero-based result offset used for pagination.
        limit (int): Maximum number of records or items to return.
        sort (str): Provider-supported result ordering expression.
        from_date_time (str): Earliest provider update timestamp to include.
        to_date_time (str): Latest provider update timestamp to include.
        conference (bool): Whether to restrict committee reports to conference reports.
        time (int): Request timeout in seconds.

    Returns:
        Any: Structured mapping produced by the operation.

    Raises:
        ValueError: If a required value is missing, blank, or outside the supported range.
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = Congress( )
    return _instance.fetch( mode=mode, congress=congress, bill_type=bill_type,
	    bill_number=bill_number, law_type=law_type, law_number=law_number, report_type=report_type,
	    report_number=report_number, offset=offset, limit=limit, sort=sort,
	    from_date_time=from_date_time, to_date_time=to_date_time, conference=conference, time=time )

@beta_tool
def fetch_internet_archive( keywords: str, fields: List[str] | None=None, rows: int=10, page: int=1,
		sort: str='downloads desc', media_type: str='', collection: str='', time: int=20 ) -> Any:
    """Retrieve Internet Archive search and metadata.

    Purpose:
        Retrieve Internet Archive search and metadata through Internet Archive. The query text determines the records or documents matched by the provider. Result-count arguments bound the amount of data requested.

    Args:
        keywords (str): Search text, lookup value, or provider query submitted by the caller.
        fields (List[str] | None): Comma-separated or provider-specific field selection.
        rows (int): Maximum number of rows to request.
        page (int): One-based result page to request.
        sort (str): Provider-supported result ordering expression.
        media_type (str): Provider type selector for media.
        collection (str): Provider collection identifier used to restrict results.
        time (int): Request timeout in seconds.

    Returns:
        Any: Structured mapping produced by the operation.

    Raises:
        ValueError: If a required value is missing, blank, or outside the supported range.
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = InternetArchive( )
    return _instance.fetch( keywords=keywords, fields=fields, rows=rows, page=page, sort=sort,
	    media_type=media_type, collection=collection, time=time )

@beta_tool
def fetch_grokipedia( mode: str='search', query: str='', page: str='', limit: int=12,
		offset: int=0, include_content: bool=True ) -> Any:
    """Retrieve Grokipedia search and page.

    Purpose:
        Retrieve Grokipedia search and page through Grokipedia. The query text determines the records or documents matched by the provider. Result-count arguments bound the amount of data requested.

    Args:
        mode (str): Operation mode used to select the provider or processing workflow.
        query (str): Search text, lookup value, or provider query submitted by the caller.
        page (str): One-based result page to request.
        limit (int): Maximum number of records or items to return.
        offset (int): Zero-based result offset used for pagination.
        include_content (bool): Whether to include content in the result.

    Returns:
        Any: Structured mapping produced by the operation.

    Raises:
        ValueError: If a required value is missing, blank, or outside the supported range.
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = Grokipedia( )
    return _instance.fetch( mode=mode, query=query, page=page, limit=limit, offset=offset,
	    include_content=include_content )

@beta_tool
def load_arxiv( question: str ) -> Any:
    """Load ArXiv research documents.

    Purpose:
        Load ArXiv research documents using the ArXiv loader. The query text determines the records or documents matched by the provider.

    Args:
        question (str): Search query or prompt submitted to the backing loader.

    Returns:
        Any: LangChain documents loaded from the requested source.

    Raises:
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = ArXivLoader( )
    return _instance.load( question=question )

@beta_tool
def load_wikipedia( question: str ) -> Any:
    """Load Wikipedia articles.

    Purpose:
        Load Wikipedia articles using the Wikipedia loader. The query text determines the records or documents matched by the provider.

    Args:
        question (str): Search query or prompt submitted to the backing loader.

    Returns:
        Any: LangChain documents loaded from the requested source.

    Raises:
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = WikiLoader( )
    return _instance.load( question=question )


# ==========================================================================================
# ASTRONOMY, SPACE, AND AVIATION TOOLS
# ==========================================================================================

@beta_tool
def fetch_naval_observatory( mode: str='celnav', date_value: str='', time_value: str='',
		latitude: float=0.0, longitude: float=0.0, location_label: str='', time: int=20 ) -> Any:
    """Retrieve U.S. Naval Observatory celestial-navigation data.

    Purpose:
        Retrieve U.S. Naval Observatory celestial-navigation data through U.S. Naval Observatory. Coordinate and bounding arguments constrain geographic scope when supported.

    Args:
        mode (str): Operation mode used to select the provider or processing workflow.
        date_value (str): Calendar date used by the selected provider operation.
        time_value (str): Clock time or timestamp used by the selected provider operation.
        latitude (float): Latitude in decimal degrees.
        longitude (float): Longitude in decimal degrees.
        location_label (str): Human-readable label associated with the supplied coordinates.
        time (int): Request timeout in seconds.

    Returns:
        Any: Structured mapping produced by the operation.

    Raises:
        ValueError: If a required value is missing, blank, or outside the supported range.
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = NavalObservatory( )
    return _instance.fetch( mode=mode, date_value=date_value, time_value=time_value,
	    latitude=latitude, longitude=longitude, location_label=location_label, time=time )

@beta_tool
def fetch_satellite_center( mode: str='observatories', query: str='', start_time: str='',
		end_time: str='', coordinate_systems: str='gse', resolution_factor: int=1, time: int=20 ) -> Any:
    """Retrieve SSC satellite observatory, ground-station, and location data.

    Purpose:
        Retrieve SSC satellite observatory, ground-station, and location data through NASA Satellite Situation Center. The query text determines the records or documents matched by the provider.

    Args:
        mode (str): Operation mode used to select the provider or processing workflow.
        query (str): Search text, lookup value, or provider query submitted by the caller.
        start_time (str): Beginning timestamp for the requested provider interval.
        end_time (str): Ending timestamp for the requested provider interval.
        coordinate_systems (str): Coordinate system or comma-separated coordinate systems requested from the satellite service.
        resolution_factor (int): Sampling resolution factor applied to returned satellite location data.
        time (int): Request timeout in seconds.

    Returns:
        Any: Structured mapping produced by the operation.

    Raises:
        ValueError: If a required value is missing, blank, or outside the supported range.
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = SatelliteCenter( )
    return _instance.fetch( mode=mode, query=query, start_time=start_time, end_time=end_time,
	    coordinate_systems=coordinate_systems, resolution_factor=resolution_factor, time=time )

@beta_tool
def fetch_nearby_objects( mode: str='close_approaches', start_date: str='', end_date: str='',
		query: str='', query_type: str='sstr', dist_max: str='10LD', body: str='Earth',
		sort: str='date', limit: int=20, dv: float=6.0, dur: int=360, stay: int=8,
		launch: str='2020-2045', h: float=26.0, occ: int=7, include_physical: bool=True,
		include_close_approaches: bool=True, ca_body: str='Earth',
		include_discovery: bool=True, time: int=20 ) -> Any:
    """Retrieve JPL SSD and CNEOS near-Earth object data.

    Purpose:
        Retrieve JPL SSD and CNEOS near-Earth object data through NASA/JPL near-Earth object services. The query text determines the records or documents matched by the provider. Date and time arguments constrain the requested interval when supplied. Result-count arguments bound the amount of data requested.

    Args:
        mode (str): Operation mode used to select the provider or processing workflow.
        start_date (str): Inclusive start date for the requested time range, in the provider-supported format.
        end_date (str): Inclusive end date for the requested time range, in the provider-supported format.
        query (str): Search text, lookup value, or provider query submitted by the caller.
        query_type (str): Provider type selector for query.
        dist_max (str): Maximum close-approach distance expression accepted by the JPL service.
        body (str): Solar-system body used as the reference object.
        sort (str): Provider-supported result ordering expression.
        limit (int): Maximum number of records or items to return.
        dv (float): Delta-v threshold or mission constraint used by the near-Earth object query.
        dur (int): Mission duration constraint, in days, used by the near-Earth object query.
        stay (int): Target stay-duration constraint, in days, used by the near-Earth object query.
        launch (str): Launch-year or launch-window expression used by the near-Earth object query.
        h (float): Absolute-magnitude threshold used by the near-Earth object query.
        occ (int): Opportunity-count or occurrence constraint used by the mission query.
        include_physical (bool): Whether to include physical in the result.
        include_close_approaches (bool): Whether to include close approaches in the result.
        ca_body (str): Reference body used for close-approach data.
        include_discovery (bool): Whether to include discovery in the result.
        time (int): Request timeout in seconds.

    Returns:
        Any: Structured mapping produced by the operation.

    Raises:
        ValueError: If a required value is missing, blank, or outside the supported range.
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = NearbyObjects( )
    return _instance.fetch( mode=mode, start_date=start_date, end_date=end_date, query=query,
	    query_type=query_type, dist_max=dist_max, body=body, sort=sort, limit=limit, dv=dv,
	    dur=dur, stay=stay, launch=launch, h=h, occ=occ, include_physical=include_physical,
	    include_close_approaches=include_close_approaches, ca_body=ca_body,
	    include_discovery=include_discovery, time=time )


@beta_tool
def fetch_open_science( mode: str='dataset', query: str='', accession: str='',
		format_value: str='json', time: int=20 ) -> Any:
    """Retrieve NASA Open Science Data Repository resources.

    Purpose:
        Retrieve NASA Open Science Data Repository resources through NASA Open Science Data Repository. The query text determines the records or documents matched by the provider.

    Args:
        mode (str): Operation mode used to select the provider or processing workflow.
        query (str): Search text, lookup value, or provider query submitted by the caller.
        accession (str): Dataset accession identifier used to retrieve a specific Open Science resource.
        format_value (str): Provider output format.
        time (int): Request timeout in seconds.

    Returns:
        Any: Structured mapping produced by the operation.

    Raises:
        ValueError: If a required value is missing, blank, or outside the supported range.
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = OpenScience( )
    return _instance.fetch( mode=mode, query=query, accession=accession,
	    format_value=format_value, time=time )

@beta_tool
def fetch_space_weather( mode: str='cme', start_date: str='', end_date: str='', time: int=20,
		location: str='ALL', catalog: str='ALL', notification_type: str='all',
		most_accurate_only: bool=True, complete_entry_only: bool=True, speed: int=0,
		half_angle: int=0, keyword: str='', api_key: str | None=None ) -> Any:
    """Retrieve NASA DONKI space weather endpoints.

    Purpose:
        Retrieve NASA DONKI space weather endpoints through NASA DONKI. Date and time arguments constrain the requested interval when supplied. When supplied, ``api_key`` overrides the configured provider credential for this request.

    Args:
        mode (str): Operation mode used to select the provider or processing workflow.
        start_date (str): Inclusive start date for the requested time range, in the provider-supported format.
        end_date (str): Inclusive end date for the requested time range, in the provider-supported format.
        time (int): Request timeout in seconds.
        location (str): Place name, address, or location description resolved by the provider.
        catalog (str): Provider catalog filter.
        notification_type (str): Provider type selector for notification.
        most_accurate_only (bool): Whether to restrict results to the provider-designated most accurate analyses.
        complete_entry_only (bool): Whether to restrict results to complete provider entries.
        speed (int): Minimum or target speed constraint used by the space-weather query.
        half_angle (int): Half-angle constraint used by the space-weather query.
        keyword (str): Keyword used to filter provider records.
        api_key (str | None): Optional credential override used for the active request.

    Returns:
        Any: Structured mapping produced by the operation.

    Raises:
        ValueError: If a required value is missing, blank, or outside the supported range.
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = SpaceWeather( )
    return _instance.fetch( mode=mode, start_date=start_date, end_date=end_date, time=time,
	    location=location, catalog=catalog, notification_type=notification_type,
	    most_accurate_only=most_accurate_only, complete_entry_only=complete_entry_only,
	    speed=speed, half_angle=half_angle, keyword=keyword, api_key=api_key )

@beta_tool
def fetch_astro_catalog( mode: str='object_query', query: str='', quantity: str='',
		attributes: str='', arguments: str='', ra: str='', dec: str='', radius: int=2,
		data_format: str='json', time: int=20 ) -> Any:
    """Retrieve Open Astronomy Catalog queries.

    Purpose:
        Retrieve Open Astronomy Catalog queries through Open Astronomy Catalog. The query text determines the records or documents matched by the provider.

    Args:
        mode (str): Operation mode used to select the provider or processing workflow.
        query (str): Search text, lookup value, or provider query submitted by the caller.
        quantity (str): Provider quantity or field requested from the catalog.
        attributes (str): Provider attributes requested for matching catalog records.
        arguments (str): Keyword arguments passed to the bound callable.
        ra (str): Right ascension value.
        dec (str): Declination value.
        radius (int): Search radius in the units specified by the operation.
        data_format (str): Provider output data format.
        time (int): Request timeout in seconds.

    Returns:
        Any: Provider-specific structured data produced by the retrieval operation.

    Raises:
        ValueError: If a required value is missing, blank, or outside the supported range.
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = AstroCatalog( )
    return _instance.fetch( mode=mode, query=query, quantity=quantity, attributes=attributes,
	    arguments=arguments, ra=ra, dec=dec, radius=radius, data_format=data_format, time=time )

@beta_tool
def fetch_astro_query( mode: str='object_search', query: str='', ra: str='', dec: str='',
		radius: float=0.5, radius_unit: str='deg', row_limit: int=100 ) -> Any:
    """Retrieve Simbad and astronomy object search operations.

    Purpose:
        Retrieve Simbad and astronomy object search operations through Astroquery/SIMBAD. The query text determines the records or documents matched by the provider. Result-count arguments bound the amount of data requested.

    Args:
        mode (str): Operation mode used to select the provider or processing workflow.
        query (str): Search text, lookup value, or provider query submitted by the caller.
        ra (str): Right ascension value.
        dec (str): Declination value.
        radius (float): Search radius in the units specified by the operation.
        radius_unit (str): Unit applied to the search radius.
        row_limit (int): Maximum number of rows returned by the astronomy query.

    Returns:
        Any: Structured mapping produced by the operation.

    Raises:
        ValueError: If a required value is missing, blank, or outside the supported range.
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = AstroQuery( )
    return _instance.fetch( mode=mode, query=query, ra=ra, dec=dec, radius=radius,
	    radius_unit=radius_unit, row_limit=row_limit )

@beta_tool
def fetch_star_map( mode: str='object_link', query: str='', ra: float=0.0, dec: float=0.0,
		zoom: int=5, image_source: str='DSS2', box_color: str='yellow', show_box: bool=True,
		show_grid: bool=True, show_lines: bool=True, show_boundaries: bool=True,
		show_const_names: bool=False, time: int=20 ) -> Any:
    """Retrieve astronomical object map links and imagery.

    Purpose:
        Retrieve astronomical object map links and imagery through astronomical map service. The query text determines the records or documents matched by the provider.

    Args:
        mode (str): Operation mode used to select the provider or processing workflow.
        query (str): Search text, lookup value, or provider query submitted by the caller.
        ra (float): Right ascension value.
        dec (float): Declination value.
        zoom (int): Map or chart zoom level.
        image_source (str): Imagery or survey source used to render the map or chart.
        box_color (str): Color used to draw the target box on generated map or chart output.
        show_box (bool): Whether to display box in generated output.
        show_grid (bool): Whether to display grid in generated output.
        show_lines (bool): Whether to display lines in generated output.
        show_boundaries (bool): Whether to display boundaries in generated output.
        show_const_names (bool): Whether to display const names in generated output.
        time (int): Request timeout in seconds.

    Returns:
        Any: Structured mapping produced by the operation.

    Raises:
        ValueError: If a required value is missing, blank, or outside the supported range.
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = StarMap( )
    return _instance.fetch( mode=mode, query=query, ra=ra, dec=dec, zoom=zoom,
	    image_source=image_source, box_color=box_color, show_box=show_box, show_grid=show_grid,
	    show_lines=show_lines, show_boundaries=show_boundaries,
	    show_const_names=show_const_names, time=time )

@beta_tool
def fetch_star_chart( mode: str='object_chart', query: str='', ra: float=0.0,
		dec: float=0.0, zoom: int=5, image_source: str='DSS2', box_color: str='yellow',
		show_box: bool=True, show_grid: bool=True, show_lines: bool=True, show_boundaries: bool=True,
		show_const_names: bool=False, width: int=900, height: int=450,
		magnitude: float=7.5, time: int=20 ) -> Any:
    """Retrieve static star chart and coordinate chart generation.

    Purpose:
        Retrieve static star chart and coordinate chart generation through astronomical chart service. The query text determines the records or documents matched by the provider.

    Args:
        mode (str): Operation mode used to select the provider or processing workflow.
        query (str): Search text, lookup value, or provider query submitted by the caller.
        ra (float): Right ascension value.
        dec (float): Declination value.
        zoom (int): Map or chart zoom level.
        image_source (str): Imagery or survey source used to render the map or chart.
        box_color (str): Color used to draw the target box on generated map or chart output.
        show_box (bool): Whether to display box in generated output.
        show_grid (bool): Whether to display grid in generated output.
        show_lines (bool): Whether to display lines in generated output.
        show_boundaries (bool): Whether to display boundaries in generated output.
        show_const_names (bool): Whether to display const names in generated output.
        width (int): Output image or chart width in pixels.
        height (int): Output image or chart height in pixels.
        magnitude (float): Limiting stellar magnitude used when rendering a chart.
        time (int): Request timeout in seconds.

    Returns:
        Any: Structured mapping produced by the operation.

    Raises:
        ValueError: If a required value is missing, blank, or outside the supported range.
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = StarChart( )
    return _instance.fetch( mode=mode, query=query, ra=ra, dec=dec, zoom=zoom,
	    image_source=image_source, box_color=box_color, show_box=show_box, show_grid=show_grid,
	    show_lines=show_lines, show_boundaries=show_boundaries, show_const_names=show_const_names,
	    width=width, height=height, magnitude=magnitude, time=time )

@beta_tool
def fetch_open_sky( mode: str='states_bbox', icao24: str='', airport: str='', begin: int | None=None,
		end: int | None=None, time_value: int | None=None, lamin: float | None=None,
		lomin: float | None=None, lamax: float | None=None, lomax: float | None=None,
		extended: bool=False, client_id: str=None, client_secret: str=None, time: int=20 ) -> Any:
    """Retrieve OpenSky Network aircraft, airport, and state-vector data.

    Purpose:
        Retrieve OpenSky Network aircraft, airport, and state-vector data through OpenSky Network. Use ``mode`` to select among ``arrivals_airport``, ``departures_airport``, ``flights_aircraft``, ``states_bbox``, ``track_aircraft``.

    Args:
        mode (str): Operation selector. Supported values detected in the implementation include ``arrivals_airport``, ``departures_airport``, ``flights_aircraft``, ``states_bbox``, ``track_aircraft``.
        icao24 (str): 24-bit ICAO aircraft transponder address.
        airport (str): ICAO airport identifier used to query arrivals or departures.
        begin (int | None): Beginning Unix timestamp for the requested aviation interval.
        end (int | None): Ending Unix timestamp for the requested aviation interval.
        time_value (int | None): Clock time or timestamp used by the selected provider operation.
        lamin (float | None): Bounding-box minimum latitude in decimal degrees.
        lomin (float | None): Bounding-box minimum longitude in decimal degrees.
        lamax (float | None): Bounding-box maximum latitude in decimal degrees.
        lomax (float | None): Bounding-box maximum longitude in decimal degrees.
        extended (bool): Whether extended OpenSky state-vector fields should be requested.
        client_id (str): Optional credential override used for the active request.
        client_secret (str): Optional credential override used for the active request.
        time (int): Request timeout in seconds.

    Returns:
        Any: Structured mapping produced by the operation.

    Raises:
        ValueError: If a required value is missing, blank, or outside the supported range.
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = OpenSky( )
    return _instance.fetch( mode=mode, icao24=icao24, airport=airport, begin=begin, end=end,
	    time_value=time_value, lamin=lamin, lomin=lomin, lamax=lamax, lomax=lomax,
	    extended=extended, client_id=client_id, client_secret=client_secret, time=time )

# ==========================================================================================
# CLOUD STORAGE AND REMOTE DOCUMENT LOADER TOOLS
# ==========================================================================================

@beta_tool
def load_google_drive_file( file_id: str, recursive: bool=False ) -> Any:
    """Load a Google Drive file.

    Purpose:
        Load a Google Drive file using the Google Drive loader. Boolean options control retrieval depth or supplemental content.

    Args:
        file_id (str): Provider file identifier used to load a single file.
        recursive (bool): Whether the loader should traverse nested provider or URL resources.

    Returns:
        Any: LangChain documents loaded from the requested source.

    Raises:
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = GoogleDriveLoader( )
    return _instance.load_file( file_id=file_id, recursive=recursive )

@beta_tool
def load_google_drive_folder( folder_id: str, recursive: bool=False ) -> Any:
    """Load documents from a Google Drive folder.

    Purpose:
        Load documents from a Google Drive folder using the Google Drive loader. Boolean options control retrieval depth or supplemental content.

    Args:
        folder_id (str): Provider folder identifier used to load folder contents.
        recursive (bool): Whether the loader should traverse nested provider or URL resources.

    Returns:
        Any: LangChain documents loaded from the requested source.

    Raises:
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = GoogleDriveLoader( )
    return _instance.load_folder( folder_id=folder_id, recursive=recursive )

@beta_tool
def load_onedrive( drive_id: str, folder_path: Optional[str]=None,
		object_ids: Optional[List[str]]=None, auth_with_token: bool=True ) -> Any:
    """Load documents from OneDrive.

    Purpose:
        Load documents from OneDrive using the OneDrive loader.

    Args:
        drive_id (str): OneDrive drive identifier.
        folder_path (Optional[str]): Optional folder path within the selected drive.
        object_ids (Optional[List[str]]): Optional provider object identifiers to load.
        auth_with_token (bool): Whether token-based authentication should be used.

    Returns:
        Any: LangChain documents loaded from the requested source.

    Raises:
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = OneDriveDocLoader( )
    return _instance.load( drive_id=drive_id, folder_path=folder_path, object_ids=object_ids,
	    auth_with_token=auth_with_token )

@beta_tool
def load_google_cloud_file( project_name: str, bucket: str, blob: str ) -> Any:
    """Load a Google Cloud Storage object.

    Purpose:
        Load a Google Cloud Storage object using the Google Cloud Storage loader.

    Args:
        project_name (str): Google Cloud project name used by the storage loader.
        bucket (str): Storage bucket name.
        blob (str): Cloud storage object name.

    Returns:
        Any: LangChain documents loaded from the requested source.

    Raises:
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = GoogleCloudFileLoader( )
    return _instance.load( project_name=project_name, bucket=bucket, blob=blob )

@beta_tool
def load_aws_file( bucket: str, key: str, aws_access_key_id: Optional[str]=None,
		aws_secret_access_key: Optional[str]=None, aws_session_token: Optional[str]=None,
		region_name: Optional[str]=None ) -> Any:
    """Load an Amazon S3 object.

    Purpose:
        Load an Amazon S3 object using the Amazon S3 file loader.

    Args:
        bucket (str): Storage bucket name.
        key (str): Amazon S3 object key.
        aws_access_key_id (Optional[str]): Provider identifier for the selected aws access key.
        aws_secret_access_key (Optional[str]): AWS credential or configuration value for secret access key.
        aws_session_token (Optional[str]): AWS credential or configuration value for session token.
        region_name (Optional[str]): Cloud region name used to configure the storage client.

    Returns:
        Any: LangChain documents loaded from the requested source.

    Raises:
        Error: If the implementation wraps a provider, parsing, filesystem, or processing
            failure in the project error type.
    """
    _instance = AwsFileLoader( )
    return _instance.load( bucket=bucket, key=key, aws_access_key_id=aws_access_key_id,
	    aws_secret_access_key=aws_secret_access_key, aws_session_token=aws_session_token,
	    region_name=region_name )

@beta_tool
def load_google_speech_to_text( project_id: str, file_path: str,
		config: Optional[Dict[str, Any]]=None ) -> Any:
    """Transcribe audio with Google Speech-to-Text.

    Purpose:
        Transcribe audio with Google Speech-to-Text using the Google Speech-to-Text loader.

    Args:
        project_id (str): Google Cloud project identifier used by the speech loader.
        file_path (str): Local filesystem path to the source file.
        config (Optional[Dict[str, Any]]): Optional provider configuration mapping.

    Returns:
        Any: LangChain documents loaded from the requested source.

    Raises:
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type. the project error type.
    """
    _instance = GoogleSpeechToTextLoader( )
    return _instance.load( project_id=project_id, file_path=file_path, config=config )

@beta_tool
def load_google_bucket( project_name: str, bucket: str, prefix: Optional[str]=None,
		continue_on_failure: bool=False ) -> Any:
    """Load documents from a Google Cloud Storage bucket.

    Purpose:
        Load documents from a Google Cloud Storage bucket using the Google Cloud Storage bucket loader.

    Args:
        project_name (str): Google Cloud project name used by the storage loader.
        bucket (str): Storage bucket name.
        prefix (Optional[str]): Optional object-name prefix used to restrict cloud storage results.
        continue_on_failure (bool): Whether loading should continue when an individual object fails.

    Returns:
        Any: LangChain documents loaded from the requested source.

    Raises:
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = GoogleBucketLoader( )
    return _instance.load( project_name=project_name, bucket=bucket, prefix=prefix,
	    continue_on_failure=continue_on_failure )

@beta_tool
def load_aws_bucket( bucket: str, prefix: Optional[str]=None, aws_access_key_id: Optional[str]=None,
		aws_secret_access_key: Optional[str]=None, aws_session_token: Optional[str]=None,
		region_name: Optional[str]=None, endpoint_url: Optional[str]=None ) -> Any:
    """Load documents from an Amazon S3 bucket.

    Purpose:
        Load documents from an Amazon S3 bucket using the Amazon S3 bucket loader.

    Args:
        bucket (str): Storage bucket name.
        prefix (Optional[str]): Optional object-name prefix used to restrict cloud storage results.
        aws_access_key_id (Optional[str]): Provider identifier for the selected aws access key.
        aws_secret_access_key (Optional[str]): AWS credential or configuration value for secret access key.
        aws_session_token (Optional[str]): AWS credential or configuration value for session token.
        region_name (Optional[str]): Cloud region name used to configure the storage client.
        endpoint_url (Optional[str]): Optional alternate service endpoint URL.

    Returns:
        Any: LangChain documents loaded from the requested source.

    Raises:
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure
            in the project error type.
    """
    _instance = AwsBucketLoader( )
    return _instance.load( bucket=bucket, prefix=prefix, aws_access_key_id=aws_access_key_id,
	    aws_secret_access_key=aws_secret_access_key, aws_session_token=aws_session_token,
	    region_name=region_name, endpoint_url=endpoint_url )


# ==========================================================================================
# DEMOGRAPHIC, GOVERNMENT DATA, AND LOCAL DOCUMENT TOOLS
# ==========================================================================================

@beta_tool
def fetch_census_data( mode: str='variables', year: str='2022', dataset: str='acs/acs5',
		fields: str='NAME,B01001_001E', geography_for: str='state:*', geography_in: str='',
		predicates: str='', time: int=20 ) -> Any:
    """Retrieve U.S. Census dataset and variable.

    Purpose:
        Retrieve U.S. Census dataset and variable through U.S. Census API. Use ``mode`` to select among ``data``, ``variables``.

    Args:
        mode (str): Operation selector. Supported values detected in the implementation include ``data``, ``variables``.
        year (str): Dataset or observation year requested from the provider.
        dataset (str): Provider dataset name or identifier.
        fields (str): Comma-separated or provider-specific field selection.
        geography_for (str): Census ``for`` geography clause defining the requested geography.
        geography_in (str): Optional Census ``in`` geography clause constraining the request.
        predicates (str): Additional Census query predicates appended to the request.
        time (int): Request timeout in seconds.

    Returns:
        Any: Structured mapping produced by the operation.

    Raises:
        ValueError: If a required value is missing, blank, or outside the supported range.
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = CensusData( )
    return _instance.fetch( mode=mode, year=year, dataset=dataset, fields=fields,
	    geography_for=geography_for, geography_in=geography_in, predicates=predicates, time=time )


@beta_tool
def fetch_socrata( mode: str='rows', domain: str='data.cdc.gov', dataset_id: str='',
		select: str='', where: str='', order: str='', group: str='', limit: int=25,
		offset: int=0, time: int=20 ) -> Any:
    """Retrieve Socrata dataset metadata and row.

    Purpose:
        Retrieve Socrata dataset metadata and row through Socrata. Use ``mode`` to select among ``metadata``, ``rows``. Result-count arguments bound the amount of data requested.

    Args:
        mode (str): Operation selector. Supported values detected in the implementation include ``metadata``, ``rows``.
        domain (str): Provider domain or host containing the requested dataset.
        dataset_id (str): Provider dataset identifier.
        select (str): Socrata ``$select`` expression defining returned columns or calculations.
        where (str): Socrata ``$where`` filter expression.
        order (str): Provider-supported result ordering expression.
        group (str): Socrata ``$group`` expression used to aggregate rows.
        limit (int): Maximum number of records or items to return.
        offset (int): Zero-based result offset used for pagination.
        time (int): Request timeout in seconds.

    Returns:
        Any: Structured mapping produced by the operation.

    Raises:
        ValueError: If a required value is missing, blank, or outside the supported range.
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = Socrata( )
    return _instance.fetch( mode=mode, domain=domain, dataset_id=dataset_id, select=select,
	    where=where, order=order, group=group, limit=limit, offset=offset, time=time )

@beta_tool
def fetch_united_nations( mode: str='datasets', query_path: str='', time: int=20 ) -> Any:
    """Retrieve United Nations SDMX dataset and query.

    Purpose:
        Retrieve United Nations SDMX dataset and query through United Nations SDMX service. Use ``mode`` to select among ``datasets``, ``sdmx_query``.

    Args:
        mode (str): Operation selector. Supported values detected in the implementation include ``datasets``, ``sdmx_query``.
        query_path (str): Path identifying the query resource.
        time (int): Request timeout in seconds.

    Returns:
        Any: Structured mapping produced by the operation.

    Raises:
        ValueError: If a required value is missing, blank, or outside the supported range.
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = UnitedNations( )
    return _instance.fetch( mode=mode, query_path=query_path, time=time )

@beta_tool
def fetch_world_population( mode: str='catalog', query: str='', asset_path: str='',
		page: int=1, page_size: int=25, time: int=20 ) -> Any:
    """Retrieve WorldPop catalog and raster metadata.

    Purpose:
        Retrieve WorldPop catalog and raster metadata through WorldPop. Use ``mode`` to select among ``catalog``, ``raster_metadata``, ``search``. The query text determines the records or documents matched by the provider. Result-count arguments bound the amount of data requested.

    Args:
        mode (str): Operation selector. Supported values detected in the implementation include ``catalog``, ``raster_metadata``, ``search``.
        query (str): Search text, lookup value, or provider query submitted by the caller.
        asset_path (str): Path identifying the asset resource.
        page (int): One-based result page to request.
        page_size (int): Maximum number of records requested per page.
        time (int): Request timeout in seconds.

    Returns:
        Any: Structured mapping produced by the operation.

    Raises:
        ValueError: If a required value is missing, blank, or outside the supported range.
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = WorldPopulation( )
    return _instance.fetch( mode=mode, query=query, asset_path=asset_path, page=page,
	    page_size=page_size, time=time )

@beta_tool
def load_open_city( city_id: str, dataset_id: str, limit: int=100 ) -> Any:
    """Load an Open City dataset.

    Purpose:
        Load an Open City dataset using the Open City Data loader. Result-count arguments bound the amount of data requested.

    Args:
        city_id (str): Provider identifier for the selected city.
        dataset_id (str): Provider dataset identifier.
        limit (int): Maximum number of records requested from the backing source.

    Returns:
        Any: LangChain documents loaded from the requested source.

    Raises:
        ValueError: Raised when a required value is missing, blank, or outside the supported range.
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = OpenCityLoader( )
    return _instance.load( city_id=city_id, dataset_id=dataset_id, limit=limit )

@beta_tool
def load_text( path: str, encoding: Optional[str]=None ) -> Any:
    """Load a plain-text file.

    Purpose:
        Load a plain-text file using the text loader.

    Args:
        path (str): Local file path used by the loader.
        encoding (Optional[str]): Optional file encoding passed to the backing loader.

    Returns:
        Any: LangChain documents loaded from the requested source.

    Raises:
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = TextLoader( )
    return _instance.load( path=path, encoding=encoding )

@beta_tool
def load_csv( path: str, encoding: Optional[str]='utf-8', source_column: Optional[str]=None,
		delimiter: str=',', quotechar: str='"' ) -> Any:
    """Load a CSV file.

    Purpose:
        Load a CSV file using the CSV loader.

    Args:
        path (str): Local file path used by the loader.
        encoding (Optional[str]): Optional file encoding passed to the backing loader.
        source_column (Optional[str]): Optional CSV column whose value is stored as the document source.
        delimiter (str): Field delimiter used to parse delimited text.
        quotechar (str): Quote character used to parse delimited text.

    Returns:
        Any: LangChain documents loaded from the requested source.

    Raises:
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = CsvLoader( )
    return _instance.load( path=path, encoding=encoding, source_column=source_column,
	    delimiter=delimiter, quotechar=quotechar )

@beta_tool
def read_pdf( path: str, mode: str='single' ) -> Any:
    """Read a PDF file.

    Purpose:
        Read a PDF file using the PDF reader.

    Args:
        path (str): Local file path used by the loader.
        mode (str): Operation mode used to select the provider or processing workflow.

    Returns:
        Any: LangChain documents loaded from the requested source.

    Raises:
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = PdfReader( )
    return _instance.load( path=path, mode=mode )

@beta_tool
def load_pdf( path: str, mode: str='single', extract: str='plain', include: bool=False,
		format: str='markdown-img', size: int=1000, overlap: int=150, has_tables: bool=True ) -> Any:
    """Load and extract a PDF file.

    Purpose:
        Load and extract a PDF file using the PDF loader.

    Args:
        path (str): Local file path used by the loader.
        mode (str): Operation mode used to select the provider or processing workflow.
        extract (str): PDF text-extraction strategy used by the underlying parser.
        include (bool): Whether optional embedded content should be included.
        format (str): Output or embedded-image format requested from the loader.
        size (int): Maximum chunk size used for document splitting.
        overlap (int): Number of characters or tokens repeated between adjacent chunks.
        has_tables (bool): Whether table-aware parsing or extraction should be enabled.

    Returns:
        Any: LangChain documents loaded from the requested source.

    Raises:
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = PdfLoader( size=size, overlap=overlap, has_tables=has_tables )
    return _instance.load( path=path, mode=mode, extract=extract, include=include, format=format )

@beta_tool
def load_excel( path: str, mode: str='elements', has_headers: bool=True ) -> Any:
    """Load an Excel workbook.

    Purpose:
        Load an Excel workbook using the Excel loader.

    Args:
        path (str): Local file path used by the loader.
        mode (str): Operation mode used to select the provider or processing workflow.
        has_headers (bool): Whether the first spreadsheet row should be treated as column headers.

    Returns:
        Any: LangChain documents loaded from the requested source.

    Raises:
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = ExcelLoader( )
    return _instance.load( path=path, mode=mode, has_headers=has_headers )

@beta_tool
def load_word( path: str ) -> Any:
    """Load a Word document.

    Purpose:
        Load a Word document using the Word loader.

    Args:
        path (str): Local file path used by the loader.

    Returns:
        Any: LangChain documents loaded from the requested source.

    Raises:
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = WordLoader( )
    return _instance.load( path=path )

@beta_tool
def load_markdown( path: str ) -> Any:
    """Load a Markdown document.

    Purpose:
        Load a Markdown document using the Markdown loader.

    Args:
        path (str): Local file path used by the loader.

    Returns:
        Any: LangChain documents loaded from the requested source.

    Raises:
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = MarkdownLoader( )
    return _instance.load( path=path )

@beta_tool
def load_html( path: str ) -> Any:
    """Load an HTML document.

    Purpose:
        Load an HTML document using the HTML loader.

    Args:
        path (str): Local file path used by the loader.

    Returns:
        Any: LangChain documents loaded from the requested source.

    Raises:
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = HtmlLoader( )
    return _instance.load( path=path )

@beta_tool
def load_outlook( path: str ) -> Any:
    """Load an Outlook message.

    Purpose:
        Load an Outlook message using the Outlook message loader.

    Args:
        path (str): Local file path used by the loader.

    Returns:
        Any: LangChain documents loaded from the requested source.

    Raises:
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = OutlookLoader( )
    return _instance.load( path=path )

@beta_tool
def load_spfx( library_id: str ) -> Any:
    """Load a SharePoint document library.

    Purpose:
        Load a SharePoint document library using the SharePoint loader.

    Args:
        library_id (str): SharePoint document-library identifier.

    Returns:
        Any: LangChain documents loaded from the requested source.

    Raises:
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = SpfxLoader( )
    return _instance.load( library_id=library_id )


@beta_tool
def load_spfx_folder( library_id: str, folder_id: str ) -> Any:
    """Load a SharePoint folder.

    Purpose:
        Load a SharePoint folder using the SharePoint loader.

    Args:
        library_id (str): SharePoint document-library identifier.
        folder_id (str): Provider folder identifier used to load folder contents.

    Returns:
        Any: LangChain documents loaded from the requested source.

    Raises:
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = SpfxLoader( )
    return _instance.load_folder( library_id=library_id, folder_id=folder_id )


@beta_tool
def load_powerpoint( path: str, mode: str='single' ) -> Any:
    """Load a PowerPoint presentation.

    Purpose:
        Load a PowerPoint presentation using the PowerPoint loader.

    Args:
        path (str): Local file path used by the loader.
        mode (str): Operation mode used to select the provider or processing workflow.

    Returns:
        Any: LangChain documents loaded from the requested source.

    Raises:
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = PowerPointLoader( )
    return _instance.load( path=path, mode=mode )


@beta_tool
def load_powerpoint_multiple( path: str ) -> Any:
    """Load multiple PowerPoint presentation elements.

    Purpose:
        Load multiple PowerPoint presentation elements using the PowerPoint loader.

    Args:
        path (str): Local file path used by the loader.

    Returns:
        Any: LangChain documents loaded from the requested source.

    Raises:
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = PowerPointLoader( )
    return _instance.load_multiple( path=path )


@beta_tool
def load_email( path: str, mode: str='single', attachments: bool=True ) -> Any:
    """Load an email message.

    Purpose:
        Load an email message using the email loader.

    Args:
        path (str): Local file path used by the loader.
        mode (str): Operation mode used to select the provider or processing workflow.
        attachments (bool): Whether email attachments should be included when supported.

    Returns:
        Any: LangChain documents loaded from the requested source.

    Raises:
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = EmailLoader( )
    return _instance.load( path=path, mode=mode, attachments=attachments )


@beta_tool
def load_json( filepath: str, is_text: bool=True, is_lines: bool=False ) -> Any:
    """Load JSON content.

    Purpose:
        Load JSON content using the JSON loader.

    Args:
        filepath (str): Local file path used by the loader.
        is_text (bool): Whether JSON values should be treated as text content.
        is_lines (bool): Whether the JSON source uses JSON Lines format.

    Returns:
        Any: LangChain documents loaded from the requested source.

    Raises:
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = JsonLoader( )
    return _instance.load( filepath=filepath, is_text=is_text, is_lines=is_lines )


@beta_tool
def load_xml( filepath: str ) -> Any:
    """Load an XML document.

    Purpose:
        Load an XML document using the XML loader.

    Args:
        filepath (str): Local file path used by the loader.

    Returns:
        Any: LangChain documents loaded from the requested source.

    Raises:
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = XmlLoader( )
    return _instance.load( filepath=filepath )


@beta_tool
def load_xml_tree( filepath: str ) -> Any:
    """Parse an XML document tree.

    Purpose:
        Parse an XML document tree using the XML loader.

    Args:
        filepath (str): Local file path used by the loader.

    Returns:
        Any: XML elements matching the requested XPath expression.

    Raises:
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = XmlLoader( )
    return _instance.load_tree( filepath=filepath )


@beta_tool
def load_jupyter_notebook( path: str, include_outputs: bool=False, max_output_length: int=10,
		remove_newline: bool=False, traceback: bool=False ) -> Any:
    """Load a Jupyter notebook.

    Purpose:
        Load a Jupyter notebook using the Jupyter notebook loader. Boolean options control retrieval depth or supplemental content.

    Args:
        path (str): Local file path used by the loader.
        include_outputs (bool): Whether notebook cell outputs should be included.
        max_output_length (int): Maximum notebook cell output length to retain.
        remove_newline (bool): Whether newline characters should be removed from notebook output.
        traceback (bool): Whether notebook traceback output should be included.

    Returns:
        Any: LangChain documents loaded from the requested source.

    Raises:
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = JupyterNotebookLoader( )
    return _instance.load( path=path, include_outputs=include_outputs,
	    max_output_length=max_output_length, remove_newline=remove_newline, traceback=traceback )


# ==========================================================================================
# WEATHER AND ENVIRONMENTAL DATA TOOLS
# ==========================================================================================

@beta_tool
def fetch_google_weather_current( address: str, units_system: str='METRIC', language_code: str='en',
		time: int=10 ) -> Any:
    """Retrieve google weather current data.

    Purpose:
        Retrieve google weather current data through Google Weather.

    Args:
        address (str): Street address or place description used for geocoding, validation, or routing.
        units_system (str): Measurement unit system requested from the provider.
        language_code (str): BCP-47-style language code used for provider results.
        time (int): Request timeout in seconds.

    Returns:
        Any: Structured mapping produced by the operation.

    Raises:
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = GoogleWeather( )
    return _instance.fetch_current( address=address, units_system=units_system,
	    language_code=language_code, time=time )


@beta_tool
def fetch_google_weather_hourly_forecast( address: str, hours: int=24, units_system: str='METRIC',
		language_code: str='en', time: int=10 ) -> Any:
    """Retrieve hourly forecast.

    Purpose:
        Retrieve hourly forecast through Google Weather.

    Args:
        address (str): Street address or place description used for geocoding, validation, or routing.
        hours (int): Number of hourly observations or forecast periods to request.
        units_system (str): Measurement unit system requested from the provider.
        language_code (str): BCP-47-style language code used for provider results.
        time (int): Request timeout in seconds.

    Returns:
        Any: Structured mapping produced by the operation.

    Raises:
        ValueError: If a required value is missing, blank, or outside the supported range.
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = GoogleWeather( )
    return _instance.fetch_hourly_forecast( address=address, hours=hours, units_system=units_system,
	    language_code=language_code, time=time )


@beta_tool
def fetch_google_weather_daily_forecast( address: str, days: int=5, units_system: str='METRIC',
		language_code: str='en', time: int=10 ) -> Any:
    """Retrieve daily forecast.

    Purpose:
        Retrieve daily forecast through Google Weather.

    Args:
        address (str): Street address or place description used for geocoding, validation, or routing.
        days (int): Number of calendar days included in the requested interval.
        units_system (str): Measurement unit system requested from the provider.
        language_code (str): BCP-47-style language code used for provider results.
        time (int): Request timeout in seconds.

    Returns:
        Any: Structured mapping produced by the operation.

    Raises:
        ValueError: If a required value is missing, blank, or outside the supported range.
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = GoogleWeather( )
    return _instance.fetch_daily_forecast( address=address, days=days, units_system=units_system,
	    language_code=language_code, time=time )


@beta_tool
def fetch_google_weather_hourly_history( address: str, hours: int=24, units_system: str='METRIC',
		language_code: str='en', time: int=10 ) -> Any:
    """Retrieve hourly history.

    Purpose:
        Retrieve hourly history through Google Weather.

    Args:
        address (str): Street address or place description used for geocoding, validation, or routing.
        hours (int): Number of hourly observations or forecast periods to request.
        units_system (str): Measurement unit system requested from the provider.
        language_code (str): BCP-47-style language code used for provider results.
        time (int): Request timeout in seconds.

    Returns:
        Any: Structured mapping produced by the operation.

    Raises:
        ValueError: If a required value is missing, blank, or outside the supported range.
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = GoogleWeather( )
    return _instance.fetch_hourly_history( address=address, hours=hours,
	    units_system=units_system, language_code=language_code, time=time )


@beta_tool
def fetch_google_weather_alerts( address: str, language_code: str='en', time: int=10 ) -> Any:
    """Retrieve google weather alerts data.

    Purpose:
        Retrieve google weather alerts data through Google Weather.

    Args:
        address (str): Street address or place description used for geocoding, validation, or routing.
        language_code (str): BCP-47-style language code used for provider results.
        time (int): Request timeout in seconds.

    Returns:
        Any: Structured mapping produced by the operation.

    Raises:
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = GoogleWeather( )
    return _instance.fetch_alerts( address=address, language_code=language_code, time=time )


@beta_tool
def fetch_earth_observatory( mode: str='events', status: str='open', category: str='', source: str='',
		limit: int=20, days: int=30, start_date: str='', end_date: str='', time: int=20 ) -> Any:
    """Retrieve NASA EONET events, categories, sources, and layers.

    Purpose:
        Retrieve NASA EONET events, categories, sources, and layers through NASA EONET. Date and time arguments constrain the requested interval when supplied. Result-count arguments bound the amount of data requested.

    Args:
        mode (str): Operation mode used to select the provider or processing workflow.
        status (str): Provider status filter applied to returned records.
        category (str): Optional logical category retained in tool metadata.
        source (str): Provider source identifier used to restrict or classify results.
        limit (int): Maximum number of records or items to return.
        days (int): Number of calendar days included in the requested interval.
        start_date (str): Inclusive start date for the requested time range, in the provider-supported format.
        end_date (str): Inclusive end date for the requested time range, in the provider-supported format.
        time (int): Request timeout in seconds.

    Returns:
        Any: Structured mapping produced by the operation.

    Raises:
        ValueError: If a required value is missing, blank, or outside the supported range.
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = EarthObservatory( )
    return _instance.fetch( mode=mode, status=status, category=category, source=source,
	    limit=limit, days=days, start_date=start_date, end_date=end_date, time=time )


@beta_tool
def fetch_open_weather( location: str, mode: str='current', zone: str='auto', forecast_days: int=7,
		past_days: int=0, count: int=10 ) -> Any:
    """Retrieve Open-Meteo current and forecast weather.

    Purpose:
        Retrieve Open-Meteo current and forecast weather through Open-Meteo.

    Args:
        location (str): Place name, address, or location description resolved by the provider.
        mode (str): Operation mode used to select the provider or processing workflow.
        zone (str): Timezone identifier or automatic timezone-selection mode.
        forecast_days (int): Number of forecast days to request.
        past_days (int): Number of historical days to include with the weather request.
        count (int): Maximum number of matching locations or records to consider.

    Returns:
        Any: Structured mapping produced by the operation.

    Raises:
        ValueError: If a required value is missing, blank, or outside the supported range.
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = OpenWeather( )
    return _instance.fetch( location=location, mode=mode, zone=zone, forecast_days=forecast_days,
	    past_days=past_days, count=count )


@beta_tool
def fetch_historical_weather( location: str, date: dt.date, zone: str='auto', count: int=10 ) -> Any:
    """Retrieve historical weather archive.

    Purpose:
        Retrieve historical weather archive through Open-Meteo Archive.

    Args:
        location (str): Place name, address, or location description resolved by the provider.
        date (dt.date): Date used by the provider or processing operation.
        zone (str): Timezone identifier or automatic timezone-selection mode.
        count (int): Maximum number of matching locations or records to consider.

    Returns:
        Any: Structured mapping produced by the operation.

    Raises:
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = HistoricalWeather( )
    return _instance.fetch( location=location, date=date, zone=zone, count=count )


@beta_tool
def fetch_usgs_earthquakes( mode: str='feed', feed: str='all_day.geojson', start_date: str='',
		end_date: str='', min_magnitude: float=1.0, max_magnitude: float=10.0,
		limit: int=25, order_by: str='time', event_type: str='earthquake', latitude: float | None=None,
		longitude: float | None=None, max_radius_km: float | None=None, time: int=20 ) -> Any:
    """Retrieve USGS earthquake feed and query.

    Purpose:
        Retrieve USGS earthquake feed and query through USGS Earthquake Hazards Program. Use ``mode`` to select among ``feed``, ``search``. Date and time arguments constrain the requested interval when supplied. Coordinate and bounding arguments constrain geographic scope when supported. Result-count arguments bound the amount of data requested.

    Args:
        mode (str): Operation selector. Supported values detected in the implementation include ``feed``, ``search``.
        feed (str): Predefined USGS earthquake feed name used when feed mode is selected.
        start_date (str): Inclusive start date for the requested time range, in the provider-supported format.
        end_date (str): Inclusive end date for the requested time range, in the provider-supported format.
        min_magnitude (float): Minimum earthquake magnitude to include in the result set.
        max_magnitude (float): Maximum earthquake magnitude to include in the result set.
        limit (int): Maximum number of records or items to return.
        order_by (str): Provider-supported field used to order results.
        event_type (str): USGS event type to include; ``earthquake`` is the default.
        latitude (float | None): Latitude in decimal degrees.
        longitude (float | None): Longitude in decimal degrees.
        max_radius_km (float | None): Maximum geographic search radius in kilometers.
        time (int): Request timeout in seconds.

    Returns:
        Any: Structured mapping produced by the operation.

    Raises:
        ValueError: If a required value is missing, blank, or outside the supported range.
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = USGSEarthquakes( )
    return _instance.fetch( mode=mode, feed=feed, start_date=start_date, end_date=end_date,
	    min_magnitude=min_magnitude, max_magnitude=max_magnitude, limit=limit, order_by=order_by,
	    event_type=event_type, latitude=latitude, longitude=longitude,
	    max_radius_km=max_radius_km, time=time )


@beta_tool
def fetch_usgs_water_data( mode: str='monitoring-locations', monitoring_location_id: str='',
		state_code: str='', county_code: str='', site_type: str='', parameter_code: str='',
		limit: int=25, time: int=20 ) -> Any:
    """Retrieve USGS water services records.

    Purpose:
        Retrieve USGS water services records through USGS Water Data. Use ``mode`` to select among ``latest-continuous``, ``latest-daily``, ``monitoring-locations``, ``time-series-metadata``. Result-count arguments bound the amount of data requested.

    Args:
        mode (str): Operation selector. Supported values detected in the implementation include ``latest- continuous``, ``latest-daily``, ``monitoring-locations``, ``time-series-metadata``.
        monitoring_location_id (str): USGS monitoring-location identifier used to target a specific site.
        state_code (str): State code used to restrict provider records.
        county_code (str): County code used to restrict provider records.
        site_type (str): USGS site-type code used to restrict monitoring locations.
        parameter_code (str): USGS parameter code identifying the measured property.
        limit (int): Maximum number of records or items to return.
        time (int): Request timeout in seconds.

    Returns:
        Any: Structured mapping produced by the operation.

    Raises:
        ValueError: If a required value is missing, blank, or outside the supported range.
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = USGSWaterData( )
    return _instance.fetch( mode=mode, monitoring_location_id=monitoring_location_id,
	    state_code=state_code, county_code=county_code, site_type=site_type,
	    parameter_code=parameter_code, limit=limit, time=time )

@beta_tool
def fetch_air_now( mode: str='current-zip', zip_code: str='', latitude: float | None=None,
		longitude: float | None=None, date: str='', distance: int=25, time: int=20 ) -> Any:
    """Retrieve AirNow current and forecast air quality data.

    Purpose:
        Retrieve AirNow current and forecast air quality data through AirNow. Use ``mode`` to select among ``current-latlon``, ``current-zip``, ``forecast-latlon``, ``forecast-zip``. Coordinate and bounding arguments constrain geographic scope when supported.

    Args:
        mode (str): Operation selector. Supported values detected in the implementation include ``current- latlon``, ``current-zip``, ``forecast-latlon``, ``forecast-zip``.
        zip_code (str): Provider code identifying or filtering zip.
        latitude (float | None): Latitude in decimal degrees.
        longitude (float | None): Longitude in decimal degrees.
        date (str): Date used by the provider or processing operation.
        distance (int): Maximum provider search distance, using the units defined by that service.
        time (int): Request timeout in seconds.

    Returns:
        Any: Structured mapping produced by the operation.

    Raises:
        ValueError: If a required value is missing, blank, or outside the supported range.
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = AirNow( )
    return _instance.fetch( mode=mode, zip_code=zip_code, latitude=latitude, longitude=longitude,
	    date=date, distance=distance, time=time )


@beta_tool
def fetch_climate_data( mode: str='datasets', keyword: str='', dataset: str='', start_date: str='',
		end_date: str='', stations: str='', data_types: str='', limit: int=25,
		offset: int=0, time: int=20 ) -> Any:
    """Retrieve NOAA climate dataset and data records.

    Purpose:
        Retrieve NOAA climate dataset and data records through NOAA climate services. Use ``mode`` to select among ``data``, ``datasets``. Date and time arguments constrain the requested interval when supplied. Result-count arguments bound the amount of data requested.

    Args:
        mode (str): Operation selector. Supported values detected in the implementation include ``data``, ``datasets``.
        keyword (str): Keyword used to filter provider records.
        dataset (str): Provider dataset name or identifier.
        start_date (str): Inclusive start date for the requested time range, in the provider-supported format.
        end_date (str): Inclusive end date for the requested time range, in the provider-supported format.
        stations (str): Station identifiers used to restrict climate observations.
        data_types (str): Climate data-type identifiers requested from the provider.
        limit (int): Maximum number of records or items to return.
        offset (int): Zero-based result offset used for pagination.
        time (int): Request timeout in seconds.

    Returns:
        Any: Structured mapping produced by the operation.

    Raises:
        ValueError: If a required value is missing, blank, or outside the supported range.
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = ClimateData( )
    return _instance.fetch( mode=mode, keyword=keyword, dataset=dataset, start_date=start_date,
	    end_date=end_date, stations=stations, data_types=data_types, limit=limit,
	    offset=offset, time=time )


@beta_tool
def fetch_eonet( mode: str='events', source: str='', category: str='', status: str='open',
		limit: int=25, days: int=30, start_date: str='', end_date: str='',
		bbox: str='', time: int=20 ) -> Any:
    """Retrieve NASA EONET environmental event data.

    Purpose:
        Retrieve NASA EONET environmental event data through NASA EONET. Use ``mode`` to select among ``categories``, ``events``. Date and time arguments constrain the requested interval when supplied. Coordinate and bounding arguments constrain geographic scope when supported. Result-count arguments bound the amount of data requested.

    Args:
        mode (str): Operation selector. Supported values detected in the implementation include ``categories``, ``events``.
        source (str): Provider source identifier used to restrict or classify results.
        category (str): Optional logical category retained in tool metadata.
        status (str): Provider status filter applied to returned records.
        limit (int): Maximum number of records or items to return.
        days (int): Number of calendar days included in the requested interval.
        start_date (str): Inclusive start date for the requested time range, in the provider-supported format.
        end_date (str): Inclusive end date for the requested time range, in the provider-supported format.
        bbox (str): Bounding box defining the geographic extent of the request.
        time (int): Request timeout in seconds.

    Returns:
        Any: Structured mapping produced by the operation.

    Raises:
        ValueError: If a required value is missing, blank, or outside the supported range.
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = EoNet( )
    return _instance.fetch( mode=mode, source=source, category=category, status=status,
	    limit=limit, days=days, start_date=start_date, end_date=end_date, bbox=bbox, time=time )


@beta_tool
def fetch_envirofacts( table_name: str='TRI_FACILITY', state_code: str='',
		facility_name: str='', limit: int=25, time: int=20 ) -> Any:
    """Retrieve EPA Envirofacts table and facility records.

    Purpose:
        Retrieve EPA Envirofacts table and facility records through EPA Envirofacts. Result-count arguments bound the amount of data requested.

    Args:
        table_name (str): Envirofacts table or resource name to query.
        state_code (str): State code used to restrict provider records.
        facility_name (str): Facility-name filter applied to Envirofacts records.
        limit (int): Maximum number of records or items to return.
        time (int): Request timeout in seconds.

    Returns:
        Any: Structured mapping produced by the operation.

    Raises:
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = EnviroFacts( )
    return _instance.fetch( table_name=table_name, state_code=state_code,
	    facility_name=facility_name, limit=limit, time=time )


@beta_tool
def fetch_tides_and_currents( mode: str='water-level', station_id: str='', begin_date: str='',
		end_date: str='', datum: str='MLLW', units: str='metric', time_zone: str='gmt',
		interval: str='hilo', time: int=20 ) -> Any:
    """Retrieve NOAA tides, currents, and station data.

    Purpose:
        Retrieve NOAA tides, currents, and station data through NOAA Tides & Currents. Use ``mode`` to select among ``station``, ``tide-predictions``, ``water-level``. Date and time arguments constrain the requested interval when supplied.

    Args:
        mode (str): Operation selector. Supported values detected in the implementation include ``station``, ``tide-predictions``, ``water-level``.
        station_id (str): Provider identifier for the selected station.
        begin_date (str): Beginning date for the requested interval, in the provider-supported format.
        end_date (str): Inclusive end date for the requested time range, in the provider-supported format.
        datum (str): Vertical datum used for tide or water-level measurements.
        units (str): Unit system used for returned measurements.
        time_zone (str): Timezone used for returned tide or current timestamps.
        interval (str): Provider sampling or reporting interval.
        time (int): Request timeout in seconds.

    Returns:
        Any: Structured mapping produced by the operation.

    Raises:
        ValueError: If a required value is missing, blank, or outside the supported range.
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = TidesAndCurrents( )
    return _instance.fetch( mode=mode, station_id=station_id, begin_date=begin_date,
	    end_date=end_date, datum=datum, units=units, time_zone=time_zone, interval=interval, time=time )


@beta_tool
def fetch_uv_index( mode: str='daily-zip', zip_code: str='', city: str='',
		state: str='', time: int=20 ) -> Any:
    """Retrieve EPA UV Index current and forecast data.

    Purpose:
        Retrieve EPA UV Index current and forecast data through EPA UV Index. Use ``mode`` to select among ``daily-city-state``, ``daily-zip``, ``hourly-city-state``, ``hourly-zip``.

    Args:
        mode (str): Operation selector. Supported values detected in the implementation include ``daily- city-state``, ``daily-zip``, ``hourly-city-state``, ``hourly-zip``.
        zip_code (str): Provider code identifying or filtering zip.
        city (str): City name used to locate or filter provider records.
        state (str): State name or abbreviation used to locate or filter provider records.
        time (int): Request timeout in seconds.

    Returns:
        Any: Structured mapping produced by the operation.

    Raises:
        ValueError: If a required value is missing, blank, or outside the supported range.
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = UvIndex( )
    return _instance.fetch( mode=mode, zip_code=zip_code, city=city, state=state, time=time )


@beta_tool
def fetch_purple_air( mode: str='sensors', sensor_index: int | None=None, nwlng: float | None=None,
		nwlat: float | None=None, selng: float | None=None, selat: float | None=None,
		location_type: int=0, max_age: int=0, modified_since: int=0, fields: str='',
		time: int=20 ) -> Any:
    """Retrieve PurpleAir sensor and air quality records.

    Purpose:
        Retrieve PurpleAir sensor and air quality records through PurpleAir. Use ``mode`` to select among ``sensor``, ``sensors``. Coordinate and bounding arguments constrain geographic scope when supported.

    Args:
        mode (str): Operation selector. Supported values detected in the implementation include ``sensor``, ``sensors``.
        sensor_index (int | None): PurpleAir sensor identifier.
        nwlng (float | None): Northwest bounding-box longitude in decimal degrees.
        nwlat (float | None): Northwest bounding-box latitude in decimal degrees.
        selng (float | None): Southeast bounding-box longitude in decimal degrees.
        selat (float | None): Southeast bounding-box latitude in decimal degrees.
        location_type (int): Provider type selector for location.
        max_age (int): Maximum age permitted by the operation.
        modified_since (int): Unix timestamp used to return PurpleAir sensors modified after the specified time.
        fields (str): Comma-separated or provider-specific field selection.
        time (int): Request timeout in seconds.

    Returns:
        Any: Structured mapping produced by the operation.

    Raises:
        ValueError: If a required value is missing, blank, or outside the supported range.
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = PurpleAir( )
    return _instance.fetch( mode=mode, sensor_index=sensor_index, nwlng=nwlng,
	    nwlat=nwlat, selng=selng, selat=selat, location_type=location_type,
	    max_age=max_age, modified_since=modified_since, fields=fields, time=time )


@beta_tool
def fetch_open_aq( mode: str='locations', location_id: int | None=None, parameter_id: int | None=None,
		country_id: int | None=None, coordinates: str='', radius: int=25000, providers_id: str='',
		parameters_id: str='', limit: int=25, page: int=1, time: int=20 ) -> Any:
    """Retrieve OpenAQ location, measurement, and air-quality records.

    Purpose:
        Retrieve OpenAQ location, measurement, and air-quality records through OpenAQ. Use ``mode`` to select among ``countries``, ``latest``, ``locations``, ``parameter_latest``, ``parameters``, ``providers``. Coordinate and bounding arguments constrain geographic scope when supported. Result-count arguments bound the amount of data requested.

    Args:
        mode (str): Operation selector. Supported values detected in the implementation include ``countries``, ``latest``, ``locations``, ``parameter_latest``, ``parameters``, ``providers``.
        location_id (int | None): Provider identifier for the selected location.
        parameter_id (int | None): Provider identifier for the selected parameter.
        country_id (int | None): Provider identifier for the selected country.
        coordinates (str): Latitude/longitude coordinate string used by the provider.
        radius (int): Search radius in the units specified by the operation.
        providers_id (str): Provider identifier for the selected providers.
        parameters_id (str): Provider identifier for the selected parameters.
        limit (int): Maximum number of records or items to return.
        page (int): One-based result page to request.
        time (int): Request timeout in seconds.

    Returns:
        Any: Structured mapping produced by the operation.

    Raises:
        ValueError: If a required value is missing, blank, or outside the supported range.
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = OpenAQ( )
    return _instance.fetch( mode=mode, location_id=location_id, parameter_id=parameter_id,
	    country_id=country_id, coordinates=coordinates, radius=radius, providers_id=providers_id,
	    parameters_id=parameters_id, limit=limit, page=page, time=time )


@beta_tool
def fetch_firms( mode: str='area', source: str='VIIRS_SNPP_NRT', area_coordinates: str='world',
		day_range: int=1, date: str='', sensor: str='ALL', time: int=20 ) -> Any:
    """Retrieve NASA FIRMS active fire data.

    Purpose:
        Retrieve NASA FIRMS active fire data through NASA FIRMS. Use ``mode`` to select among ``area``, ``data-availability``.

    Args:
        mode (str): Operation selector. Supported values detected in the implementation include ``area``, ``data-availability``.
        source (str): Provider source identifier used to restrict or classify results.
        area_coordinates (str): FIRMS area-of-interest coordinates or ``world`` selector.
        day_range (int): Number of days included in the FIRMS active-fire request.
        date (str): Date used by the provider or processing operation.
        sensor (str): Sensor or instrument filter applied to provider results.
        time (int): Request timeout in seconds.

    Returns:
        Any: Structured mapping produced by the operation.

    Raises:
        ValueError: If a required value is missing, blank, or outside the supported range.
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = Firms( )
    return _instance.fetch( mode=mode, source=source, area_coordinates=area_coordinates,
	    day_range=day_range, date=date, sensor=sensor, time=time )



# ==========================================================================================
# GEOSPATIAL, MAPPING, AND IMAGERY TOOLS
# ==========================================================================================

@beta_tool
def geocode_location( address: str ) -> Any:
    """Geocode location.

    Purpose:
        Geocode location using Google Maps.

    Args:
        address (str): Street address or place description used for geocoding, validation, or routing.

    Returns:
        Any: Latitude and longitude coordinate pair.

    Raises:
        ValueError: If a required value is missing, blank, or outside the supported range.
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = GoogleMaps( )
    return _instance.geocode_location( address=address )


@beta_tool
def geocode_coordinates( lat: float, long: float ) -> Any:
    """Geocode coordinates.

    Purpose:
        Geocode coordinates using Google Maps. Coordinate and bounding arguments constrain geographic scope when supported.

    Args:
        lat (float): Latitude in decimal degrees.
        long (float): Longitude in decimal degrees.

    Returns:
        Any: Text produced by the operation.

    Raises:
        ValueError: If a required value is missing, blank, or outside the supported range.
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = GoogleMaps( )
    return _instance.geocode_coordinates( lat=lat, long=long )


@beta_tool
def validate_address( address: List[str] ) -> Any:
    """Validate address.

    Purpose:
        Validate address using Google Maps.

    Args:
        address (List[str]): Street address or place description used for geocoding, validation, or routing.

    Returns:
        Any: Structured mapping produced by the operation.

    Raises:
        TypeError: If a supplied value has an unsupported type.
        ValueError: If a required value is missing, blank, or outside the supported range.
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = GoogleMaps( )
    return _instance.validate_address( address=address )


@beta_tool
def request_directions( origin: str, destination: str, mode: str='driving' ) -> Any:
    """Request directions.

    Purpose:
        Request directions using Google Maps.

    Args:
        origin (str): Starting address or place for a routing request.
        destination (str): Destination address or place for a routing request.
        mode (str): Operation mode used to select the provider or processing workflow.

    Returns:
        Any: Structured mapping produced by the operation.

    Raises:
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = GoogleMaps( )
    return _instance.request_directions( origin=origin, destination=destination, mode=mode )


@beta_tool
def fetch_global_imagery_wms_map( layer: str,
		image_date: str, bbox: Tuple[float, float, float, float],
		width: int=1200, height: int=600, projection: str='epsg4326', quality: str='best',
		image_format: str='image/png', transparent: bool=True, output_dir: str='python-examples',
		output_name: str='', time: int=20 ) -> Any:
    """Retrieve a WMS imagery map.

    Purpose:
        Retrieve a WMS imagery map through NASA Global Imagery Browse Services. Coordinate and bounding arguments constrain geographic scope when supported.

    Args:
        layer (str): Map or imagery layer identifier.
        image_date (str): Observation date used to select imagery.
        bbox (Tuple[float, float, float, float]): Bounding box defining the geographic extent of the request.
        width (int): Output image or chart width in pixels.
        height (int): Output image or chart height in pixels.
        projection (str): Coordinate reference system used for rendered imagery.
        quality (str): Imagery quality level requested from the mapping service.
        image_format (str): Output format requested for image.
        transparent (bool): Whether the generated map image should use a transparent background.
        output_dir (str): Local directory where generated imagery is written.
        output_name (str): Optional filename for generated imagery.
        time (int): Request timeout in seconds.

    Returns:
        Any: Structured mapping produced by the operation.

    Raises:
        ValueError: If a required value is missing, blank, or outside the supported range.
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = GlobalImagery( )
    return _instance.fetch_wms_map( layer=layer, image_date=image_date, bbox=bbox, width=width,
	    height=height, projection=projection, quality=quality, image_format=image_format,
	    transparent=transparent, output_dir=output_dir, output_name=output_name, time=time )


@beta_tool
def fetch_global_imagery_map_services(  ) -> Any:
    """Retrieve available imagery map services.

    Purpose:
        Retrieve available imagery map services through NASA Global Imagery Browse Services.

    Returns:
        Any: Structured mapping produced by the operation.

    Raises:
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = GlobalImagery( )
    return _instance.fetch_map_services(  )


@beta_tool
def fetch_global_imagery_mercator_map( ccrs: Any | None=None ) -> Any:
    """Render a Mercator imagery map.

    Purpose:
        Render a Mercator imagery map through NASA Global Imagery Browse Services.

    Args:
        ccrs (Any | None): Optional Cartopy coordinate reference system used to construct the map.

    Returns:
        Any: Structured mapping produced by the operation.

    Raises:
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = GlobalImagery( )
    return _instance.fetch_mercator_map( ccrs=ccrs )

@beta_tool
def fetch_google_geocoding( mode: str='forward', query: str='', latitude: float=0.0,
		longitude: float=0.0, place_id: str='', language: str='en', region: str='',
		result_type: str='', location_type: str='', time: int=10, api_key: Optional[str]=None ) -> Any:
    """Retrieve Google forward, reverse, and place geocoding.

    Purpose:
        Retrieve Google forward, reverse, and place geocoding through Google Geocoding. Use ``mode`` to select among ``forward``, ``place``, ``reverse``. The query text determines the records or documents matched by the provider. Coordinate and bounding arguments constrain geographic scope when supported. When supplied, ``api_key`` overrides the configured provider credential for this request.

    Args:
        mode (str): Operation selector. Supported values detected in the implementation include ``forward``, ``place``, ``reverse``.
        query (str): Search text, lookup value, or provider query submitted by the caller.
        latitude (float): Latitude in decimal degrees.
        longitude (float): Longitude in decimal degrees.
        place_id (str): Provider identifier for the selected place.
        language (str): Language code used for provider results or parsing.
        region (str): Provider region filter or regional bias value.
        result_type (str): Provider type selector for result.
        location_type (str): Provider type selector for location.
        time (int): Request timeout in seconds.
        api_key (Optional[str]): Optional credential override used for the active request.

    Returns:
        Any: Structured mapping produced by the operation.

    Raises:
        ValueError: If a required value is missing, blank, or outside the supported range.
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = GoogleGeocoding( )
    return _instance.fetch( mode=mode, query=query, latitude=latitude, longitude=longitude,
	    place_id=place_id, language=language, region=region, result_type=result_type,
	    location_type=location_type, time=time, api_key=api_key )


@beta_tool
def fetch_usgs_national_map( mode: str='products', dataset: str='', q: str='', bbox: str='',
		prod_formats: str='', max_items: int=25, offset: int=0, time: int=20 ) -> Any:
    """Retrieve USGS National Map datasets and products.

    Purpose:
        Retrieve USGS National Map datasets and products through USGS The National Map. Use ``mode`` to select among ``datasets``, ``products``. The query text determines the records or documents matched by the provider. Coordinate and bounding arguments constrain geographic scope when supported. Result-count arguments bound the amount of data requested.

    Args:
        mode (str): Operation selector. Supported values detected in the implementation include ``datasets``, ``products``.
        dataset (str): Provider dataset name or identifier.
        q (str): Free-text provider query used to search matching records.
        bbox (str): Bounding box defining the geographic extent of the request.
        prod_formats (str): Product-format filter applied to National Map results.
        max_items (int): Maximum number of records or items to return.
        offset (int): Zero-based result offset used for pagination.
        time (int): Request timeout in seconds.

    Returns:
        Any: Structured mapping produced by the operation.

    Raises:
        ValueError: If a required value is missing, blank, or outside the supported range.
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = USGSTheNationalMap( )
    return _instance.fetch( mode=mode, dataset=dataset, q=q, bbox=bbox, prod_formats=prod_formats,
	    max_items=max_items, offset=offset, time=time )


@beta_tool
def fetch_usgs_sciencebase( mode: str='items', q: str='', item_id: str='', max_items: int=25,
		offset: int=0, fields: str='', time: int=20 ) -> Any:
    """Retrieve USGS ScienceBase items and catalog records.

    Purpose:
        Retrieve USGS ScienceBase items and catalog records through USGS ScienceBase. Use ``mode`` to select among ``item``, ``items``. The query text determines the records or documents matched by the provider. Result-count arguments bound the amount of data requested.

    Args:
        mode (str): Operation selector. Supported values detected in the implementation include ``item``, ``items``.
        q (str): Free-text provider query used to search matching records.
        item_id (str): Provider identifier for the selected item.
        max_items (int): Maximum number of records or items to return.
        offset (int): Zero-based result offset used for pagination.
        fields (str): Comma-separated or provider-specific field selection.
        time (int): Request timeout in seconds.

    Returns:
        Any: Structured mapping produced by the operation.

    Raises:
        ValueError: If a required value is missing, blank, or outside the supported range.
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = USGSScienceBase( )
    return _instance.fetch( mode=mode, q=q, item_id=item_id, max_items=max_items, offset=offset, fields=fields, time=time )


# ==========================================================================================
# HEALTH AND BIOMEDICAL DATA TOOLS
# ==========================================================================================

@beta_tool
def fetch_health_data( mode: str='rows', domain: str='healthdata.gov', dataset_id: str='',
		select: str='', where: str='', order: str='', group: str='', limit: int=25,
		offset: int=0, time: int=20 ) -> Any:
    """Retrieve HealthData.gov Socrata metadata and rows.

    Purpose:
        Retrieve HealthData.gov Socrata metadata and rows through HealthData.gov. Use ``mode`` to select among ``metadata``, ``rows``. Result-count arguments bound the amount of data requested.

    Args:
        mode (str): Operation selector. Supported values detected in the implementation include ``metadata``, ``rows``.
        domain (str): Provider domain or host containing the requested dataset.
        dataset_id (str): Provider dataset identifier.
        select (str): Socrata ``$select`` expression defining returned columns or calculations.
        where (str): Socrata ``$where`` filter expression.
        order (str): Provider-supported result ordering expression.
        group (str): Socrata ``$group`` expression used to aggregate rows.
        limit (int): Maximum number of records or items to return.
        offset (int): Zero-based result offset used for pagination.
        time (int): Request timeout in seconds.

    Returns:
        Any: Structured mapping produced by the operation.

    Raises:
        ValueError: If a required value is missing, blank, or outside the supported range.
        Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
    """
    _instance = HealthData( )
    return _instance.fetch( mode=mode, domain=domain, dataset_id=dataset_id, select=select,
	    where=where, order=order, group=group, limit=limit, offset=offset, time=time )



@beta_tool
def fetch_global_health_data( mode: str='indicator_registry', query_path: str='',
		fmt: str='json', time: int=20 ) -> Any:
    """Retrieve WHO global health indicator and Athena data.

    Purpose:
        Retrieve WHO global health indicator and Athena data through WHO Global Health.

    Args:
        mode (str): Operation mode used to select the backing workflow.
        query_path (str): Query path value used by the operation.
        fmt (str): Fmt value used by the operation.
        time (int): Request timeout in seconds.

    Returns:
        Any: Value produced by the delegated Fonky implementation.
    """
    _instance = GlobalHealthData( )
    return _instance.fetch( mode=mode, query_path=query_path, fmt=fmt, time=time )


@beta_tool
def fetch_wonder( mode: str='metadata_template', dataset_id: str='D76',
		request_xml: str='', time: int=20 ) -> Any:
    """Retrieve CDC WONDER template and query submission.

    Purpose:
        Retrieve CDC WONDER template and query submission through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        mode (str): Operation mode used to select the backing workflow.
        dataset_id (str): Dataset id value used by the operation.
        request_xml (str): Request xml value used by the operation.
        time (int): Request timeout in seconds.

    Returns:
        Any: Value produced by the delegated Fonky implementation.
    """
    _instance = Wonder( )
    return _instance.fetch( mode=mode, dataset_id=dataset_id, request_xml=request_xml, time=time )


@beta_tool
def load_pubmed( query: str, max_docs: int=5 ) -> Any:
    """Load PubMed research documents.

    Purpose:
        Load PubMed research documents through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        query (str): Search query or natural-language request submitted to the backing operation.
        max_docs (int): Max docs value used by the operation.

    Returns:
        Any: Value produced by the delegated Fonky implementation.
    """
    _instance = PubMedSearchLoader( )
    return _instance.load( query=query, max_docs=max_docs )


# ==========================================================================================
# WEB FETCHING, CRAWLING, LOADING, AND SCRAPING TOOLS
# ==========================================================================================

@beta_tool
def fetch_web_page( url: str, time: int=10 ) -> Any:
    """Retrieve HTTP web page content and HTML extraction data.

    Purpose:
        Retrieve HTTP web page content and HTML extraction data through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        url (str): URL used by the operation.
        time (int): Request timeout in seconds.

    Returns:
        Any: Value produced by the delegated Fonky implementation.
    """
    _instance = WebFetcher( )
    return _instance.fetch( url=url, time=time )


@beta_tool
def convert_html_to_text( html: str ) -> Any:
    """Convert HTML to plain text.

    Purpose:
        Convert HTML to plain text through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        html (str): Html value used by the operation.

    Returns:
        Any: Value produced by the delegated Fonky implementation.
    """
    _instance = WebFetcher( )
    return _instance.html_to_text( html=html )


@beta_tool
def extract_web_title( html: str ) -> Any:
    """Extract a web title from supplied HTML content.

    Purpose:
        Extract a web title from supplied HTML content through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        html (str): Html value used by the operation.

    Returns:
        Any: Value produced by the delegated Fonky implementation.
    """
    _instance = WebFetcher( )
    return _instance.extract_title( html=html )


@beta_tool
def extract_web_links( base_url: str, html: str ) -> Any:
    """Extract web links from supplied HTML content.

    Purpose:
        Extract web links from supplied HTML content through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        base_url (str): Base url value used by the operation.
        html (str): Html value used by the operation.

    Returns:
        Any: Value produced by the delegated Fonky implementation.
    """
    _instance = WebFetcher( )
    return _instance.extract_links( base_url=base_url, html=html )


@beta_tool
def extract_web_structured_data( url: str, html: str,
		selected_methods: Optional[List[str]]=None ) -> Any:
    """Extract structured data from supplied HTML content.

    Purpose:
        Extract structured data from supplied HTML content through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        url (str): URL used by the operation.
        html (str): Html value used by the operation.
        selected_methods (Optional[List[str]]): Selected methods value used by the operation.

    Returns:
        Any: Value produced by the delegated Fonky implementation.
    """
    _instance = WebFetcher( )
    return _instance.extract_structured_data( url=url, html=html, selected_methods=selected_methods )


@beta_tool
def crawl_web( seed_url: str, include_title: bool=True, include_basic_text: bool=True,
		include_raw_html: bool=False, selected_methods: Optional[List[str]]=None,
		recursive: bool=False, max_depth: int=1, max_pages: int=10, same_domain_only: bool=True,
		request_timeout: int=10, delay_seconds: float=0.25, max_bytes: int=1000000,
		headers: Optional[ Dict[ str, str ] ]=None, use_playwright: bool=False ) -> Any:
    """Crawl web pages from a seed URL.

    Purpose:
        Crawl web pages from a seed URL through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        seed_url (str): Seed url value used by the operation.
        include_title (bool): Include title value used by the operation.
        include_basic_text (bool): Include basic text value used by the operation.
        include_raw_html (bool): Include raw html value used by the operation.
        selected_methods (Optional[List[str]]): Selected methods value used by the operation.
        recursive (bool): Whether nested resources should be traversed recursively.
        max_depth (int): Max depth value used by the operation.
        max_pages (int): Max pages value used by the operation.
        same_domain_only (bool): Same domain only value used by the operation.
        request_timeout (int): Request timeout value used by the operation.
        delay_seconds (float): Delay seconds value used by the operation.
        max_bytes (int): Max bytes value used by the operation.
        headers (Optional[Dict[str, str]]): Headers value used by the operation.
        use_playwright (bool): Use playwright value used by the operation.

    Returns:
        Any: Value produced by the delegated Fonky implementation.
    """
    _instance = WebCrawler( headers=headers, use_playwright=use_playwright )
    return _instance.crawl( seed_url=seed_url, include_title=include_title,
	    include_basic_text=include_basic_text, include_raw_html=include_raw_html,
	    selected_methods=selected_methods, recursive=recursive, max_depth=max_depth,
	    max_pages=max_pages, same_domain_only=same_domain_only, request_timeout=request_timeout,
	    delay_seconds=delay_seconds, max_bytes=max_bytes )


@beta_tool
def scrape_crawler_page( url: str, include_title: bool=True, include_basic_text: bool=True,
		include_raw_html: bool=False, selected_methods: Optional[List[str]]=None,
		request_timeout: int=10, max_bytes: int=1000000,
		headers: Optional[ Dict[ str, str ] ]=None, use_playwright: bool=False ) -> Any:
    """Extract a crawler page from an HTML page.

    Purpose:
        Extract a crawler page from an HTML page through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        url (str): URL used by the operation.
        include_title (bool): Include title value used by the operation.
        include_basic_text (bool): Include basic text value used by the operation.
        include_raw_html (bool): Include raw html value used by the operation.
        selected_methods (Optional[List[str]]): Selected methods value used by the operation.
        request_timeout (int): Request timeout value used by the operation.
        max_bytes (int): Max bytes value used by the operation.
        headers (Optional[Dict[str, str]]): Headers value used by the operation.
        use_playwright (bool): Use playwright value used by the operation.

    Returns:
        Any: Value produced by the delegated Fonky implementation.
    """
    _instance = WebCrawler( headers=headers, use_playwright=use_playwright )
    return _instance.scrape_page( url=url, include_title=include_title,
	    include_basic_text=include_basic_text, include_raw_html=include_raw_html,
	    selected_methods=selected_methods, request_timeout=request_timeout, max_bytes=max_bytes )


@beta_tool
def render_web_page( url: str, timeout: int=15, headers: Optional[ Dict[ str, str ] ]=None,
		use_playwright: bool=False ) -> Any:
    """Render a dynamic web page with Playwright.

    Purpose:
        Render a dynamic web page with Playwright through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        url (str): URL used by the operation.
        timeout (int): Maximum time in seconds to wait for the operation.
        headers (Optional[Dict[str, str]]): Headers value used by the operation.
        use_playwright (bool): Use playwright value used by the operation.

    Returns:
        Any: Value produced by the delegated Fonky implementation.
    """
    _instance = WebCrawler( headers=headers, use_playwright=use_playwright )
    return _instance.render_with_playwright( url=url, timeout=timeout )


@beta_tool
def load_web( urls: str | List[str], recursive: bool=False, max_depth: int=2,
		prevent_outside: bool=True, timeout: int=10, ignore: bool=True, progress: bool=True ) -> Any:
    """Load web documents.

    Purpose:
        Load web documents through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        urls (str | List[str]): Urls value used by the operation.
        recursive (bool): Whether nested resources should be traversed recursively.
        max_depth (int): Max depth value used by the operation.
        prevent_outside (bool): Prevent outside value used by the operation.
        timeout (int): Maximum time in seconds to wait for the operation.
        ignore (bool): Ignore value used by the operation.
        progress (bool): Progress value used by the operation.

    Returns:
        Any: Value produced by the delegated Fonky implementation.
    """
    _instance = WebLoader( recursive=recursive, max_depth=max_depth,
	    prevent_outside=prevent_outside, timeout=timeout, ignore=ignore, progress=progress )
    return _instance.load( urls=urls )


@beta_tool
def load_web_recursive( url: str, depth: int=2, max_time: int=10, ignore: bool=True ) -> Any:
    """Recursively load web documents.

    Purpose:
        Recursively load web documents through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        url (str): URL used by the operation.
        depth (int): Depth value used by the operation.
        max_time (int): Max time value used by the operation.
        ignore (bool): Ignore value used by the operation.

    Returns:
        Any: Value produced by the delegated Fonky implementation.
    """
    _instance = WebLoader( )
    return _instance.load_recursive( url=url, depth=depth, max_time=max_time, ignore=ignore )


@beta_tool
def load_web_pages( urls: List[str], depth: int=2, timeout: int=10, ignore: bool=True,
		progress: bool=True ) -> Any:
    """Load static web pages.

    Purpose:
        Load static web pages through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        urls (List[str]): Urls value used by the operation.
        depth (int): Depth value used by the operation.
        timeout (int): Maximum time in seconds to wait for the operation.
        ignore (bool): Ignore value used by the operation.
        progress (bool): Progress value used by the operation.

    Returns:
        Any: Value produced by the delegated Fonky implementation.
    """
    _instance = WebLoader( )
    return _instance.load_pages( urls=urls, depth=depth, timeout=timeout,
	    ignore=ignore, progress=progress )


@beta_tool
def load_github( url: str, repo: str, branch: str, filetype: str='.md' ) -> Any:
    """Load files from a GitHub repository.

    Purpose:
        Load files from a GitHub repository through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        url (str): URL used by the operation.
        repo (str): Repo value used by the operation.
        branch (str): Branch value used by the operation.
        filetype (str): Filetype value used by the operation.

    Returns:
        Any: Value produced by the delegated Fonky implementation.
    """
    _instance = GithubLoader( )
    return _instance.load( url=url, repo=repo, branch=branch, filetype=filetype )


@beta_tool
def scrape_web_page( url: str, time: int=10 ) -> Any:
    """Fetch a web page for extraction.

    Purpose:
        Fetch a web page for extraction through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        url (str): URL used by the operation.
        time (int): Request timeout in seconds.

    Returns:
        Any: Value produced by the delegated Fonky implementation.
    """
    _instance = WebExtractor( )
    return _instance.scrape( url=url, time=time )


@beta_tool
def scraper_html_to_text( html: str ) -> Any:
    """Convert scraper HTML to plain text.

    Purpose:
        Convert scraper HTML to plain text through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        html (str): Html value used by the operation.

    Returns:
        Any: Value produced by the delegated Fonky implementation.
    """
    _instance = WebExtractor( )
    return _instance.html_to_text( html=html )


@beta_tool
def scrape_paragraphs( uri: str ) -> Any:
    """Extract paragraph text from an HTML page.

    Purpose:
        Extract paragraph text from an HTML page through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        uri (str): URI used by the operation.

    Returns:
        Any: Value produced by the delegated Fonky implementation.
    """
    _instance = WebExtractor( )
    return _instance.scrape_paragraphs( uri=uri )


@beta_tool
def scrape_lists( uri: str ) -> Any:
    """Extract list-item text from an HTML page.

    Purpose:
        Extract list-item text from an HTML page through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        uri (str): URI used by the operation.

    Returns:
        Any: Value produced by the delegated Fonky implementation.
    """
    _instance = WebExtractor( )
    return _instance.scrape_lists( uri=uri )


@beta_tool
def scrape_tables( uri: str ) -> Any:
    """Extract table-cell text from an HTML page.

    Purpose:
        Extract table-cell text from an HTML page through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        uri (str): URI used by the operation.

    Returns:
        Any: Value produced by the delegated Fonky implementation.
    """
    _instance = WebExtractor( )
    return _instance.scrape_tables( uri=uri )


@beta_tool
def scrape_articles( uri: str ) -> Any:
    """Extract article text from an HTML page.

    Purpose:
        Extract article text from an HTML page through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        uri (str): URI used by the operation.

    Returns:
        Any: Value produced by the delegated Fonky implementation.
    """
    _instance = WebExtractor( )
    return _instance.scrape_articles( uri=uri )


@beta_tool
def scrape_headings( uri: str ) -> Any:
    """Extract heading text from an HTML page.

    Purpose:
        Extract heading text from an HTML page through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        uri (str): URI used by the operation.

    Returns:
        Any: Value produced by the delegated Fonky implementation.
    """
    _instance = WebExtractor( )
    return _instance.scrape_headings( uri=uri )


@beta_tool
def scrape_divisions( uri: str ) -> Any:
    """Extract division text from an HTML page.

    Purpose:
        Extract division text from an HTML page through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        uri (str): URI used by the operation.

    Returns:
        Any: Value produced by the delegated Fonky implementation.
    """
    _instance = WebExtractor( )
    return _instance.scrape_divisions( uri=uri )


@beta_tool
def scrape_sections( uri: str ) -> Any:
    """Extract section text from an HTML page.

    Purpose:
        Extract section text from an HTML page through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        uri (str): URI used by the operation.

    Returns:
        Any: Value produced by the delegated Fonky implementation.
    """
    _instance = WebExtractor( )
    return _instance.scrape_sections( uri=uri )


@beta_tool
def scrape_blockquotes( uri: str ) -> Any:
    """Extract blockquote text from an HTML page.

    Purpose:
        Extract blockquote text from an HTML page through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        uri (str): URI used by the operation.

    Returns:
        Any: Value produced by the delegated Fonky implementation.
    """
    _instance = WebExtractor( )
    return _instance.scrape_blockquotes( uri=uri )


@beta_tool
def scrape_hyperlinks( uri: str ) -> Any:
    """Extract hyperlinks from an HTML page.

    Purpose:
        Extract hyperlinks from an HTML page through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        uri (str): URI used by the operation.

    Returns:
        Any: Value produced by the delegated Fonky implementation.
    """
    _instance = WebExtractor( )
    return _instance.scrape_hyperlinks( uri=uri )


@beta_tool
def scrape_images( uri: str ) -> Any:
    """Extract image references from an HTML page.

    Purpose:
        Extract image references from an HTML page through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        uri (str): URI used by the operation.

    Returns:
        Any: Value produced by the delegated Fonky implementation.
    """
    _instance = WebExtractor( )
    return _instance.scrape_images( uri=uri )


@beta_tool
def encode_image( path: str ) -> str:
    """Encode a local image as Base64 text.

    Purpose:
        Encode a local image as Base64 text through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        path (str): Local filesystem path used by the operation.

    Returns:
        str: Value produced by the delegated Fonky implementation.
    """
    return _encode_image( path=path )

# ==========================================================================================
# TEXT PROCESSING AND NLP TOOLS
# ==========================================================================================

@beta_tool
def preprocess_load_text( filepath: str ) -> str | None:
    """Read UTF-8 text from a local file and return the raw string.

    Purpose:
        Read UTF-8 text from a local file and return the raw string through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        filepath (str): Local filesystem path used by the operation.

    Returns:
        str | None: Value produced by the delegated Fonky implementation.
    """
    instance = TextParser( )
    return instance.load_text( filepath=filepath )


@beta_tool
def preprocess_collapse_whitespace( text: str ) -> str | None:
    """Normalize spacing by lowercasing text and collapsing repeated whitespace.

    Purpose:
        Normalize spacing by lowercasing text and collapsing repeated whitespace through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        text (str): Text value processed by the operation.

    Returns:
        str | None: Value produced by the delegated Fonky implementation.
    """
    instance = TextParser( )
    return instance.collapse_whitespace( text=text )


@beta_tool
def preprocess_remove_punctuation( text: str ) -> str:
    """Strip punctuation from tokenized text.

    Purpose:
        Strip punctuation from tokenized text through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        text (str): Text value processed by the operation.

    Returns:
        str: Value produced by the delegated Fonky implementation.
    """
    instance = TextParser( )
    return instance.remove_punctuation( text=text )


@beta_tool
def preprocess_normalize_text( text: str ) -> str | None:
    """Convert text to lowercase for stable comparison and tokenization.

    Purpose:
        Convert text to lowercase for stable comparison and tokenization through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        text (str): Text value processed by the operation.

    Returns:
        str | None: Value produced by the delegated Fonky implementation.
    """
    instance = TextParser( )
    return instance.normalize_text( text=text )


@beta_tool
def preprocess_remove_errors( text: str ) -> str:
    """Filter tokens against the NLTK English words corpus.

    Purpose:
        Filter tokens against the NLTK English words corpus through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        text (str): Text value processed by the operation.

    Returns:
        str: Value produced by the delegated Fonky implementation.
    """
    instance = TextParser( )
    return instance.remove_errors( text=text )


@beta_tool
def preprocess_remove_fragments( text: str ) -> str | None:
    """Remove very short token fragments from normalized text.

    Purpose:
        Remove very short token fragments from normalized text through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        text (str): Text value processed by the operation.

    Returns:
        str | None: Value produced by the delegated Fonky implementation.
    """
    instance = TextParser( )
    return instance.remove_fragments( text=text )


@beta_tool
def preprocess_remove_symbols( text: str ) -> str | None:
    """Remove configured symbol characters from normalized text.

    Purpose:
        Remove configured symbol characters from normalized text through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        text (str): Text value processed by the operation.

    Returns:
        str | None: Value produced by the delegated Fonky implementation.
    """
    instance = TextParser( )
    return instance.remove_symbols( text=text )


@beta_tool
def preprocess_remove_html( text: str ) -> str | None:
    """Extract visible text from HTML markup.

    Purpose:
        Extract visible text from HTML markup through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        text (str): Text value processed by the operation.

    Returns:
        str | None: Value produced by the delegated Fonky implementation.
    """
    instance = TextParser( )
    return instance.remove_html( text=text )


@beta_tool
def preprocess_remove_xml( text: str ) -> str:
    """Extract inner text from XML-like markup.

    Purpose:
        Extract inner text from XML-like markup through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        text (str): Text value processed by the operation.

    Returns:
        str: Value produced by the delegated Fonky implementation.
    """
    instance = TextParser( )
    return instance.remove_xml( text=text )


@beta_tool
def preprocess_remove_markdown( text: str ) -> str | None:
    """Remove common Markdown links, image syntax, and formatting markers.

    Purpose:
        Remove common Markdown links, image syntax, and formatting markers through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        text (str): Text value processed by the operation.

    Returns:
        str | None: Value produced by the delegated Fonky implementation.
    """
    instance = TextParser( )
    return instance.remove_markdown( text=text )


@beta_tool
def preprocess_remove_stopwords( text: str ) -> str | None:
    """Remove English stop words from tokenized text.

    Purpose:
        Remove English stop words from tokenized text through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        text (str): Text value processed by the operation.

    Returns:
        str | None: Value produced by the delegated Fonky implementation.
    """
    instance = TextParser( )
    return instance.remove_stopwords( text=text )


@beta_tool
def preprocess_remove_encodings( text: str ) -> str | None:
    """Resolve HTML entities, normalize Unicode characters, and remove control characters.

    Purpose:
        Resolve HTML entities, normalize Unicode characters, and remove control characters through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        text (str): Text value processed by the operation.

    Returns:
        str | None: Value produced by the delegated Fonky implementation.
    """
    instance = TextParser( )
    return instance.remove_encodings( text=text )


@beta_tool
def preprocess_remove_headers( filepath: str, lines: int=50, headers: int=3,
		footers: int=3 ) -> str | None:
    """Detect and remove repeated page headers and footers from a text file.

    Purpose:
        Detect and remove repeated page headers and footers from a text file through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        filepath (str): Local filesystem path used by the operation.
        lines (int): Lines value used by the operation.
        headers (int): Headers value used by the operation.
        footers (int): Footers value used by the operation.

    Returns:
        str | None: Value produced by the delegated Fonky implementation.
    """
    instance = TextParser( )
    return instance.remove_headers( filepath=filepath, lines=lines,
	    headers=headers, footers=footers )


@beta_tool
def preprocess_remove_numbers( text: str ) -> str | None:
    """Remove decimal digits from text.

    Purpose:
        Remove decimal digits from text through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        text (str): Text value processed by the operation.

    Returns:
        str | None: Value produced by the delegated Fonky implementation.
    """
    instance = TextParser( )
    return instance.remove_numbers( text=text )


@beta_tool
def preprocess_remove_numerals( text: str ) -> str | None:
    """Remove Roman-numeral patterns from text.

    Purpose:
        Remove Roman-numeral patterns from text through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        text (str): Text value processed by the operation.

    Returns:
        str | None: Value produced by the delegated Fonky implementation.
    """
    instance = TextParser( )
    return instance.remove_numerals( text=text )


@beta_tool
def preprocess_remove_images( text: str ) -> str:
    """Remove Markdown image references, HTML image elements, and direct image URLs.

    Purpose:
        Remove Markdown image references, HTML image elements, and direct image URLs through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        text (str): Text value processed by the operation.

    Returns:
        str: Value produced by the delegated Fonky implementation.
    """
    instance = TextParser( )
    return instance.remove_images( text=text )


@beta_tool
def preprocess_tiktokenize( text: str, encoding: str='cl100k_base' ) -> DataFrame | None:
    """Encode text with a tiktoken tokenizer and return token identifiers as tabular data.

    Purpose:
        Encode text with a tiktoken tokenizer and return token identifiers as tabular data through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        text (str): Text value processed by the operation.
        encoding (str): Text encoding or tokenizer encoding used by the operation.

    Returns:
        DataFrame | None: Value produced by the delegated Fonky implementation.
    """
    instance = TextParser( )
    return instance.tiktokenize( text=text, encoding=encoding )


@beta_tool
def preprocess_split_sentences( text: str ) -> List[str] | None:
    """Split text into sentence strings using NLTK sentence tokenization.

    Purpose:
        Split text into sentence strings using NLTK sentence tokenization through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        text (str): Text value processed by the operation.

    Returns:
        List[str] | None: Value produced by the delegated Fonky implementation.
    """
    instance = TextParser( )
    return instance.split_sentences( text=text )


@beta_tool
def preprocess_split_pages( filepath: str, num: int=50 ) -> List[str] | None:
    """Split a text file into page-sized text blocks.

    Purpose:
        Split a text file into page-sized text blocks through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        filepath (str): Local filesystem path used by the operation.
        num (int): Num value used by the operation.

    Returns:
        List[str] | None: Value produced by the delegated Fonky implementation.
    """
    instance = TextParser( )
    return instance.split_pages( filepath=filepath, num=num )


@beta_tool
def preprocess_split_paragraphs( filepath: str ) -> DataFrame | None:
    """Read a text file and return paragraph-like text blocks as tabular data.

    Purpose:
        Read a text file and return paragraph-like text blocks as tabular data through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        filepath (str): Local filesystem path used by the operation.

    Returns:
        DataFrame | None: Value produced by the delegated Fonky implementation.
    """
    instance = TextParser( )
    return instance.split_paragraphs( filepath=filepath )


@beta_tool
def preprocess_create_frequency_distribution( tokens: List[str] ) -> DataFrame | None:
    """Build a word-frequency table from a token sequence.

    Purpose:
        Build a word-frequency table from a token sequence through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        tokens (List[str]): Token values processed by the operation.

    Returns:
        DataFrame | None: Value produced by the delegated Fonky implementation.
    """
    instance = TextParser( )
    return instance.create_frequency_distribution( tokens=tokens )


@beta_tool
def preprocess_create_vocabulary( tokens: List[str] ) -> Series | None:
    """Extract the vocabulary column from a token-frequency table.

    Purpose:
        Extract the vocabulary column from a token-frequency table through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        tokens (List[str]): Token values processed by the operation.

    Returns:
        Series | None: Value produced by the delegated Fonky implementation.
    """
    instance = TextParser( )
    return instance.create_vocabulary( tokens=tokens )


@beta_tool
def preprocess_create_wordbag( tokens: List[str] ) -> DataFrame | None:
    """Build a bag-of-words table from a token sequence.

    Purpose:
        Build a bag-of-words table from a token sequence through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        tokens (List[str]): Token values processed by the operation.

    Returns:
        DataFrame | None: Value produced by the delegated Fonky implementation.
    """
    instance = TextParser( )
    return instance.create_wordbag( tokens=tokens )


@beta_tool
def preprocess_create_vectors( tokens: List[str] ) -> DataFrame | None:
    """Create TF-IDF vectors for token values.

    Purpose:
        Create TF-IDF vectors for token values through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        tokens (List[str]): Token values processed by the operation.

    Returns:
        DataFrame | None: Value produced by the delegated Fonky implementation.
    """
    instance = TextParser( )
    return instance.create_vectors( tokens=tokens )


@beta_tool
def preprocess_clean_file( filepath: str ) -> str | None:
    """Apply the standard Fonky text-cleaning pipeline to a single file.

    Purpose:
        Apply the standard Fonky text-cleaning pipeline to a single file through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        filepath (str): Local filesystem path used by the operation.

    Returns:
        str | None: Value produced by the delegated Fonky implementation.
    """
    instance = TextParser( )
    return instance.clean_file( filepath=filepath )


@beta_tool
def preprocess_clean_files( source: str, destination: str ) -> None:
    """Apply the standard Fonky text-cleaning pipeline to every file in a directory.

    Purpose:
        Apply the standard Fonky text-cleaning pipeline to every file in a directory through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        source (str): Source value used to scope or identify the backing operation.
        destination (str): Destination used to receive generated or processed output.

    Returns:
        None: This function performs its work through the delegated implementation and does not return a value.
    """
    instance = TextParser( )
    return instance.clean_files( source=source, destination=destination )


@beta_tool
def preprocess_chunk_files( source: str, destination: str ) -> None:
    """Split text files into sentence chunks and write chunked output files.

    Purpose:
        Split text files into sentence chunks and write chunked output files through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        source (str): Source value used to scope or identify the backing operation.
        destination (str): Destination used to receive generated or processed output.

    Returns:
        None: This function performs its work through the delegated implementation and does not return a value.
    """
    instance = TextParser( )
    return instance.chunk_files( source=source, destination=destination )


@beta_tool
def preprocess_chunk_data( filepath: str, size: int=10 ) -> DataFrame | None:
    """Chunk a text file into fixed-size word groups represented as tabular data.

    Purpose:
        Chunk a text file into fixed-size word groups represented as tabular data through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        filepath (str): Local filesystem path used by the operation.
        size (int): Maximum size or group size used by the operation.

    Returns:
        DataFrame | None: Value produced by the delegated Fonky implementation.
    """
    instance = TextParser( )
    return instance.chunk_data( filepath=filepath, size=size )


@beta_tool
def preprocess_chunk_datasets( source: str, destination: str, size: int=10 ) -> DataFrame:
    """Clean and chunk a directory of text files into spreadsheet datasets.

    Purpose:
        Clean and chunk a directory of text files into spreadsheet datasets through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        source (str): Source value used to scope or identify the backing operation.
        destination (str): Destination used to receive generated or processed output.
        size (int): Maximum size or group size used by the operation.

    Returns:
        DataFrame: Value produced by the delegated Fonky implementation.
    """
    instance = TextParser( )
    return instance.chunk_datasets( source=source, destination=destination, size=size )


@beta_tool
def preprocess_convert_jsonl( source: str, destination: str, size: int=10 ) -> None:
    """Convert text files into line-oriented JSON-like chunk output.

    Purpose:
        Convert text files into line-oriented JSON-like chunk output through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        source (str): Source value used to scope or identify the backing operation.
        destination (str): Destination used to receive generated or processed output.
        size (int): Maximum size or group size used by the operation.

    Returns:
        None: This function performs its work through the delegated implementation and does not return a value.
    """
    instance = TextParser( )
    return instance.convert_jsonl( source=source, destination=destination, size=size )


@beta_tool
def preprocess_encode_sentences( tokens: List[str], model: str='all-MiniLM-L6-v2' ) -> Tuple[List[str], np.ndarray]:
    """Generate sentence-transformer embeddings for normalized token values.

    Purpose:
        Generate sentence-transformer embeddings for normalized token values through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        tokens (List[str]): Token values processed by the operation.
        model (str): Model identifier used by the operation.

    Returns:
        Tuple[List[str], np.ndarray]: Value produced by the delegated Fonky implementation.
    """
    instance = TextParser( )
    return instance.encode_sentences( tokens=tokens, model=model )


@beta_tool
def nltk_word_tokenizer( text: str ) -> List[str] | None:
    """Tokenize text into lowercased word tokens.

    Purpose:
        Tokenize text into lowercased word tokens through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        text (str): Text value processed by the operation.

    Returns:
        List[str] | None: Value produced by the delegated Fonky implementation.
    """
    instance = NltkParser( )
    return instance.word_tokenizer( text=text )


@beta_tool
def nltk_sentence_tokenizer( text: str ) -> List[str] | None:
    """Tokenize text into lowercased sentence strings.

    Purpose:
        Tokenize text into lowercased sentence strings through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        text (str): Text value processed by the operation.

    Returns:
        List[str] | None: Value produced by the delegated Fonky implementation.
    """
    instance = NltkParser( )
    return instance.sentence_tokenizer( text=text )


@beta_tool
def nltk_word_stemmer( text: str ) -> List[str] | None:
    """Stem lowercased word tokens with the configured Porter stemmer.

    Purpose:
        Stem lowercased word tokens with the configured Porter stemmer through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        text (str): Text value processed by the operation.

    Returns:
        List[str] | None: Value produced by the delegated Fonky implementation.
    """
    instance = NltkParser( )
    return instance.word_stemmer( text=text )


@beta_tool
def nltk_word_lemmatizer( text: str ) -> List[str] | None:
    """Lemmatize lowercased word tokens with the configured WordNet lemmatizer.

    Purpose:
        Lemmatize lowercased word tokens with the configured WordNet lemmatizer through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        text (str): Text value processed by the operation.

    Returns:
        List[str] | None: Value produced by the delegated Fonky implementation.
    """
    instance = NltkParser( )
    return instance.word_lemmatizer( text=text )


@beta_tool
def nltk_pos_tagger( text: str ) -> List[Tuple[str, str]] | None:
    """Assign part-of-speech tags to lowercased word tokens.

    Purpose:
        Assign part-of-speech tags to lowercased word tokens through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        text (str): Text value processed by the operation.

    Returns:
        List[Tuple[str, str]] | None: Value produced by the delegated Fonky implementation.
    """
    instance = NltkParser( )
    return instance.pos_tagger( text=text )


@beta_tool
def nltk_named_entity_recognition( text: str ) -> List[Tuple[str, str]] | None:
    """Extract named-entity text and entity labels from tagged tokens.

    Purpose:
        Extract named-entity text and entity labels from tagged tokens through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        text (str): Text value processed by the operation.

    Returns:
        List[Tuple[str, str]] | None: Value produced by the delegated Fonky implementation.
    """
    instance = NltkParser( )
    return instance.named_entity_recognition( text=text )


@beta_tool
def nltk_chunk_words( text: str, size: int=5 ) -> DataFrame | None:
    """Group word tokens into fixed-size chunks and return them as tabular data.

    Purpose:
        Group word tokens into fixed-size chunks and return them as tabular data through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        text (str): Text value processed by the operation.
        size (int): Maximum size or group size used by the operation.

    Returns:
        DataFrame | None: Value produced by the delegated Fonky implementation.
    """
    instance = NltkParser( )
    return instance.chunk_words( text=text, size=size )


@beta_tool
def nltk_chunk_sentences( text: str, size: int=15 ) -> DataFrame | None:
    """Group sentence tokens into fixed-size chunks and return them as tabular data.

    Purpose:
        Group sentence tokens into fixed-size chunks and return them as tabular data through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        text (str): Text value processed by the operation.
        size (int): Maximum size or group size used by the operation.

    Returns:
        DataFrame | None: Value produced by the delegated Fonky implementation.
    """
    instance = NltkParser( )
    return instance.chunk_sentences( text=text, size=size )


@beta_tool
def preprocess_semantic_search( query: str, tokens: List[ str ],
        model: str='all-MiniLM-L6-v2', top: int=5 ) -> List[ Tuple[ str, float ] ]:
    """Search token content by semantic similarity.

    Purpose:
        Search token content by semantic similarity through Fonky's canonical implementation so the callable can be registered directly with a Anthropic Claude client-tool workflow.

    Args:
        query (str): Search query or natural-language request submitted to the backing operation.
        tokens (List[str]): Token values processed by the operation.
        model (str): Model identifier used by the operation.
        top (int): Top value used by the operation.

    Returns:
        List[Tuple[str, float]]: Value produced by the delegated Fonky implementation.
    """
    instance = TextParser( )
    encoded = instance.encode_sentences( tokens=tokens, model=model )
    sentences, embeddings = encoded
    transformer = SentenceTransformer( model )
    return instance.semantic_search( query=query, tokens=sentences, embeddings=embeddings,
        model=transformer, top=top )


# ==========================================================================================
# PUBLIC EXPORTS
# ==========================================================================================

__all__: List[ str ] = [
	'fetch_arxiv',
	'fetch_google_drive',
	'fetch_wikipedia',
	'fetch_news',
	'fetch_cse_search',
	'fetch_gov_data',
	'fetch_congress',
	'fetch_internet_archive',
	'fetch_grokipedia',
	'load_arxiv',
	'load_wikipedia',
	'fetch_naval_observatory',
	'fetch_satellite_center',
	'fetch_nearby_objects',
	'fetch_open_science',
	'fetch_space_weather',
	'fetch_astro_catalog',
	'fetch_astro_query',
	'fetch_star_map',
	'fetch_star_chart',
	'fetch_open_sky',
	'load_google_drive_file',
	'load_google_drive_folder',
	'load_onedrive',
	'load_google_cloud_file',
	'load_aws_file',
	'load_google_speech_to_text',
	'load_google_bucket',
	'load_aws_bucket',
	'fetch_census_data',
	'fetch_socrata',
	'fetch_united_nations',
	'fetch_world_population',
	'load_open_city',
	'load_text',
	'load_csv',
	'read_pdf',
	'load_pdf',
	'load_excel',
	'load_word',
	'load_markdown',
	'load_html',
	'load_outlook',
	'load_spfx',
	'load_spfx_folder',
	'load_powerpoint',
	'load_powerpoint_multiple',
	'load_email',
	'load_json',
	'load_xml',
	'load_xml_tree',
	'load_jupyter_notebook',
	'fetch_google_weather_current',
	'fetch_google_weather_hourly_forecast',
	'fetch_google_weather_daily_forecast',
	'fetch_google_weather_hourly_history',
	'fetch_google_weather_alerts',
	'fetch_earth_observatory',
	'fetch_open_weather',
	'fetch_historical_weather',
	'fetch_usgs_earthquakes',
	'fetch_usgs_water_data',
	'fetch_air_now',
	'fetch_climate_data',
	'fetch_eonet',
	'fetch_envirofacts',
	'fetch_tides_and_currents',
	'fetch_uv_index',
	'fetch_purple_air',
	'fetch_open_aq',
	'fetch_firms',
	'geocode_location',
	'geocode_coordinates',
	'validate_address',
	'request_directions',
	'fetch_global_imagery_wms_map',
	'fetch_global_imagery_map_services',
	'fetch_global_imagery_mercator_map',
	'fetch_google_geocoding',
	'fetch_usgs_national_map',
	'fetch_usgs_sciencebase',
	'fetch_health_data',
	'fetch_global_health_data',
	'fetch_wonder',
	'load_pubmed',
	'fetch_web_page',
	'convert_html_to_text',
	'extract_web_title',
	'extract_web_links',
	'extract_web_structured_data',
	'crawl_web',
	'scrape_crawler_page',
	'render_web_page',
	'load_web',
	'load_web_recursive',
	'load_web_pages',
	'load_github',
	'scrape_web_page',
	'scraper_html_to_text',
	'scrape_paragraphs',
	'scrape_lists',
	'scrape_tables',
	'scrape_articles',
	'scrape_headings',
	'scrape_divisions',
	'scrape_sections',
	'scrape_blockquotes',
	'scrape_hyperlinks',
	'scrape_images',
	'encode_image',
	'preprocess_load_text',
	'preprocess_collapse_whitespace',
	'preprocess_remove_punctuation',
	'preprocess_normalize_text',
	'preprocess_remove_errors',
	'preprocess_remove_fragments',
	'preprocess_remove_symbols',
	'preprocess_remove_html',
	'preprocess_remove_xml',
	'preprocess_remove_markdown',
	'preprocess_remove_stopwords',
	'preprocess_remove_encodings',
	'preprocess_remove_headers',
	'preprocess_remove_numbers',
	'preprocess_remove_numerals',
	'preprocess_remove_images',
	'preprocess_tiktokenize',
	'preprocess_split_sentences',
	'preprocess_split_pages',
	'preprocess_split_paragraphs',
	'preprocess_create_frequency_distribution',
	'preprocess_create_vocabulary',
	'preprocess_create_wordbag',
	'preprocess_create_vectors',
	'preprocess_clean_file',
	'preprocess_clean_files',
	'preprocess_chunk_files',
	'preprocess_chunk_data',
	'preprocess_chunk_datasets',
	'preprocess_convert_jsonl',
	'preprocess_encode_sentences',
	'nltk_word_tokenizer',
	'nltk_sentence_tokenizer',
	'nltk_word_stemmer',
	'nltk_word_lemmatizer',
	'nltk_pos_tagger',
	'nltk_named_entity_recognition',
	'nltk_chunk_words',
	'nltk_chunk_sentences',
	'preprocess_semantic_search',
]


# ==========================================================================================
# LOCAL AUTOMATION TOOLS - SHARED STANDALONE FONKY IMPLEMENTATION
# ==========================================================================================

from .. import loco_tools as _loco_tools

local_calculate = beta_tool( _loco_tools.local_calculate )
local_percentage = beta_tool( _loco_tools.local_percentage )
local_file_list = beta_tool( _loco_tools.local_file_list )
local_file_read = beta_tool( _loco_tools.local_file_read )
local_file_exists = beta_tool( _loco_tools.local_file_exists )
local_file_info = beta_tool( _loco_tools.local_file_info )
local_file_hash = beta_tool( _loco_tools.local_file_hash )
local_sqlite_tables = beta_tool( _loco_tools.local_sqlite_tables )
local_sqlite_schema = beta_tool( _loco_tools.local_sqlite_schema )
local_sqlite_query = beta_tool( _loco_tools.local_sqlite_query )
local_excel_sheets = beta_tool( _loco_tools.local_excel_sheets )
local_excel_cell = beta_tool( _loco_tools.local_excel_cell )
local_excel_range = beta_tool( _loco_tools.local_excel_range )
local_excel_write_range = beta_tool( _loco_tools.local_excel_write_range )
local_word_read = beta_tool( _loco_tools.local_word_read )
local_word_create = beta_tool( _loco_tools.local_word_create )
local_word_append = beta_tool( _loco_tools.local_word_append )
local_word_replace = beta_tool( _loco_tools.local_word_replace )
local_slides_create = beta_tool( _loco_tools.local_slides_create )
local_slides_add_slide = beta_tool( _loco_tools.local_slides_add_slide )
local_slides_add_text = beta_tool( _loco_tools.local_slides_add_text )
local_outlook_messages = beta_tool( _loco_tools.local_outlook_messages )
local_outlook_search = beta_tool( _loco_tools.local_outlook_search )
local_outlook_read = beta_tool( _loco_tools.local_outlook_read )
local_outlook_draft = beta_tool( _loco_tools.local_outlook_draft )
local_windows_list = beta_tool( _loco_tools.local_windows_list )
local_windows_controls = beta_tool( _loco_tools.local_windows_controls )
local_system_info = beta_tool( _loco_tools.local_system_info )
local_disk_usage = beta_tool( _loco_tools.local_disk_usage )

__all__.extend( [
    'local_calculate',
    'local_percentage',
    'local_file_list',
    'local_file_read',
    'local_file_exists',
    'local_file_info',
    'local_file_hash',
    'local_sqlite_tables',
    'local_sqlite_schema',
    'local_sqlite_query',
    'local_excel_sheets',
    'local_excel_cell',
    'local_excel_range',
    'local_excel_write_range',
    'local_word_read',
    'local_word_create',
    'local_word_append',
    'local_word_replace',
    'local_slides_create',
    'local_slides_add_slide',
    'local_slides_add_text',
    'local_outlook_messages',
    'local_outlook_search',
    'local_outlook_read',
    'local_outlook_draft',
    'local_windows_list',
    'local_windows_controls',
    'local_system_info',
    'local_disk_usage',
] )
