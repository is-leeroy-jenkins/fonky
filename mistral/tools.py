'''
  ******************************************************************************************
      Assembly:                Fonky
      Filename:                tools.py
      Author:                  Terry D. Eppler
      Created:                 09-03-2026

      Last Modified By:        Terry D. Eppler
      Last Modified On:        09-03-2026
  ******************************************************************************************
  <summary>
    Provides the Mistral AI function-tool interface for Fonky.

    Purpose:
        Exposes individual Mistral AI client-side function tools over implementation classes in
        ``fetchers.py``, ``loaders.py``, ``scrapers.py``, and ``processors.py``. Each public
        operation remains an executable Fonky callable, while companion ``*_tool`` dictionaries
        provide Mistral-compatible JSON function schemas for chat and agent requests. The consuming
        application executes returned client-side tool calls and submits their results to Mistral.
  </summary>
  ******************************************************************************************
'''
from __future__ import annotations

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

def fetch_news( endpoint: str='all', query: str='', language: str='en', categories: str='',
		exclude_categories: str='', locale: str='', domains: str='', exclude_domains: str='',
		source_ids: str='', exclude_source_ids: str='', published_after: str='',
		published_before: str='', published_on: str='', sort: str='published_at',
		limit: int=10, page: int=1, include_similar: bool=True,
		headlines_per_category: int=6, time: int=10, api_key: str | None=None ) -> Any:
    """Retrieve The News API article.

        Purpose:
            Retrieve The News API article through The News API. The query text determines the records or documents matched by the provider. Date and time arguments constrain the requested interval when supplied. Result-count arguments bound the amount of data requested. Boolean options control retrieval depth or supplemental content. When supplied, ``api_key`` overrides the configured provider credential for this request.

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
            Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
        
    """
    _instance = AwsFileLoader( )
    return _instance.load( bucket=bucket, key=key, aws_access_key_id=aws_access_key_id,
	    aws_secret_access_key=aws_secret_access_key, aws_session_token=aws_session_token,
	    region_name=region_name )

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
            Error: If the implementation wraps a provider, parsing, filesystem, or processing failure in the project error type.
        
    """
    _instance = AwsBucketLoader( )
    return _instance.load( bucket=bucket, prefix=prefix, aws_access_key_id=aws_access_key_id,
	    aws_secret_access_key=aws_secret_access_key, aws_session_token=aws_session_token,
	    region_name=region_name, endpoint_url=endpoint_url )


# ==========================================================================================
# DEMOGRAPHIC, GOVERNMENT DATA, AND LOCAL DOCUMENT TOOLS
# ==========================================================================================

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


def fetch_wonder( mode: str='metadata_template', dataset_id: str='D76',
		request_xml: str='', time: int=20 ) -> Any:
    """Retrieve CDC WONDER template and query submission.

        Purpose:
            Retrieve CDC WONDER template and query submission through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

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


def load_pubmed( query: str, max_docs: int=5 ) -> Any:
    """Load PubMed research documents.

        Purpose:
            Load PubMed research documents through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

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

def fetch_web_page( url: str, time: int=10 ) -> Any:
    """Retrieve HTTP web page content and HTML extraction data.

        Purpose:
            Retrieve HTTP web page content and HTML extraction data through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            url (str): URL used by the operation.
            time (int): Request timeout in seconds.

        Returns:
            Any: Value produced by the delegated Fonky implementation.
        
    """
    _instance = WebFetcher( )
    return _instance.fetch( url=url, time=time )


def convert_html_to_text( html: str ) -> Any:
    """Convert HTML to plain text.

        Purpose:
            Convert HTML to plain text through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            html (str): Html value used by the operation.

        Returns:
            Any: Value produced by the delegated Fonky implementation.
        
    """
    _instance = WebFetcher( )
    return _instance.html_to_text( html=html )


def extract_web_title( html: str ) -> Any:
    """Extract a web title from supplied HTML content.

        Purpose:
            Extract a web title from supplied HTML content through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            html (str): Html value used by the operation.

        Returns:
            Any: Value produced by the delegated Fonky implementation.
        
    """
    _instance = WebFetcher( )
    return _instance.extract_title( html=html )


def extract_web_links( base_url: str, html: str ) -> Any:
    """Extract web links from supplied HTML content.

        Purpose:
            Extract web links from supplied HTML content through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            base_url (str): Base url value used by the operation.
            html (str): Html value used by the operation.

        Returns:
            Any: Value produced by the delegated Fonky implementation.
        
    """
    _instance = WebFetcher( )
    return _instance.extract_links( base_url=base_url, html=html )


def extract_web_structured_data( url: str, html: str,
		selected_methods: Optional[List[str]]=None ) -> Any:
    """Extract structured data from supplied HTML content.

        Purpose:
            Extract structured data from supplied HTML content through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            url (str): URL used by the operation.
            html (str): Html value used by the operation.
            selected_methods (Optional[List[str]]): Selected methods value used by the operation.

        Returns:
            Any: Value produced by the delegated Fonky implementation.
        
    """
    _instance = WebFetcher( )
    return _instance.extract_structured_data( url=url, html=html, selected_methods=selected_methods )


def crawl_web( seed_url: str, include_title: bool=True, include_basic_text: bool=True,
		include_raw_html: bool=False, selected_methods: Optional[List[str]]=None,
		recursive: bool=False, max_depth: int=1, max_pages: int=10, same_domain_only: bool=True,
		request_timeout: int=10, delay_seconds: float=0.25, max_bytes: int=1000000,
		headers: Optional[ Dict[ str, str ] ]=None, use_playwright: bool=False ) -> Any:
    """Crawl web pages from a seed URL.

        Purpose:
            Crawl web pages from a seed URL through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

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


def scrape_crawler_page( url: str, include_title: bool=True, include_basic_text: bool=True,
		include_raw_html: bool=False, selected_methods: Optional[List[str]]=None,
		request_timeout: int=10, max_bytes: int=1000000,
		headers: Optional[ Dict[ str, str ] ]=None, use_playwright: bool=False ) -> Any:
    """Extract a crawler page from an HTML page.

        Purpose:
            Extract a crawler page from an HTML page through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

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


def render_web_page( url: str, timeout: int=15, headers: Optional[ Dict[ str, str ] ]=None,
		use_playwright: bool=False ) -> Any:
    """Render a dynamic web page with Playwright.

        Purpose:
            Render a dynamic web page with Playwright through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

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


def load_web( urls: str | List[str], recursive: bool=False, max_depth: int=2,
		prevent_outside: bool=True, timeout: int=10, ignore: bool=True, progress: bool=True ) -> Any:
    """Load web documents.

        Purpose:
            Load web documents through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

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


def load_web_recursive( url: str, depth: int=2, max_time: int=10, ignore: bool=True ) -> Any:
    """Recursively load web documents.

        Purpose:
            Recursively load web documents through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

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


def load_web_pages( urls: List[str], depth: int=2, timeout: int=10, ignore: bool=True,
		progress: bool=True ) -> Any:
    """Load static web pages.

        Purpose:
            Load static web pages through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

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


def load_github( url: str, repo: str, branch: str, filetype: str='.md' ) -> Any:
    """Load files from a GitHub repository.

        Purpose:
            Load files from a GitHub repository through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

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


def scrape_web_page( url: str, time: int=10 ) -> Any:
    """Fetch a web page for extraction.

        Purpose:
            Fetch a web page for extraction through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            url (str): URL used by the operation.
            time (int): Request timeout in seconds.

        Returns:
            Any: Value produced by the delegated Fonky implementation.
        
    """
    _instance = WebExtractor( )
    return _instance.scrape( url=url, time=time )


def scraper_html_to_text( html: str ) -> Any:
    """Convert scraper HTML to plain text.

        Purpose:
            Convert scraper HTML to plain text through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            html (str): Html value used by the operation.

        Returns:
            Any: Value produced by the delegated Fonky implementation.
        
    """
    _instance = WebExtractor( )
    return _instance.html_to_text( html=html )


def scrape_paragraphs( uri: str ) -> Any:
    """Extract paragraph text from an HTML page.

        Purpose:
            Extract paragraph text from an HTML page through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            uri (str): URI used by the operation.

        Returns:
            Any: Value produced by the delegated Fonky implementation.
        
    """
    _instance = WebExtractor( )
    return _instance.scrape_paragraphs( uri=uri )


def scrape_lists( uri: str ) -> Any:
    """Extract list-item text from an HTML page.

        Purpose:
            Extract list-item text from an HTML page through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            uri (str): URI used by the operation.

        Returns:
            Any: Value produced by the delegated Fonky implementation.
        
    """
    _instance = WebExtractor( )
    return _instance.scrape_lists( uri=uri )


def scrape_tables( uri: str ) -> Any:
    """Extract table-cell text from an HTML page.

        Purpose:
            Extract table-cell text from an HTML page through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            uri (str): URI used by the operation.

        Returns:
            Any: Value produced by the delegated Fonky implementation.
        
    """
    _instance = WebExtractor( )
    return _instance.scrape_tables( uri=uri )


def scrape_articles( uri: str ) -> Any:
    """Extract article text from an HTML page.

        Purpose:
            Extract article text from an HTML page through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            uri (str): URI used by the operation.

        Returns:
            Any: Value produced by the delegated Fonky implementation.
        
    """
    _instance = WebExtractor( )
    return _instance.scrape_articles( uri=uri )


def scrape_headings( uri: str ) -> Any:
    """Extract heading text from an HTML page.

        Purpose:
            Extract heading text from an HTML page through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            uri (str): URI used by the operation.

        Returns:
            Any: Value produced by the delegated Fonky implementation.
        
    """
    _instance = WebExtractor( )
    return _instance.scrape_headings( uri=uri )


def scrape_divisions( uri: str ) -> Any:
    """Extract division text from an HTML page.

        Purpose:
            Extract division text from an HTML page through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            uri (str): URI used by the operation.

        Returns:
            Any: Value produced by the delegated Fonky implementation.
        
    """
    _instance = WebExtractor( )
    return _instance.scrape_divisions( uri=uri )


def scrape_sections( uri: str ) -> Any:
    """Extract section text from an HTML page.

        Purpose:
            Extract section text from an HTML page through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            uri (str): URI used by the operation.

        Returns:
            Any: Value produced by the delegated Fonky implementation.
        
    """
    _instance = WebExtractor( )
    return _instance.scrape_sections( uri=uri )


def scrape_blockquotes( uri: str ) -> Any:
    """Extract blockquote text from an HTML page.

        Purpose:
            Extract blockquote text from an HTML page through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            uri (str): URI used by the operation.

        Returns:
            Any: Value produced by the delegated Fonky implementation.
        
    """
    _instance = WebExtractor( )
    return _instance.scrape_blockquotes( uri=uri )


def scrape_hyperlinks( uri: str ) -> Any:
    """Extract hyperlinks from an HTML page.

        Purpose:
            Extract hyperlinks from an HTML page through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            uri (str): URI used by the operation.

        Returns:
            Any: Value produced by the delegated Fonky implementation.
        
    """
    _instance = WebExtractor( )
    return _instance.scrape_hyperlinks( uri=uri )


def scrape_images( uri: str ) -> Any:
    """Extract image references from an HTML page.

        Purpose:
            Extract image references from an HTML page through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            uri (str): URI used by the operation.

        Returns:
            Any: Value produced by the delegated Fonky implementation.
        
    """
    _instance = WebExtractor( )
    return _instance.scrape_images( uri=uri )


def encode_image( path: str ) -> str:
    """Encode a local image as Base64 text.

        Purpose:
            Encode a local image as Base64 text through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            path (str): Local filesystem path used by the operation.

        Returns:
            str: Value produced by the delegated Fonky implementation.
        
    """
    return _encode_image( path=path )

# ==========================================================================================
# TEXT PROCESSING AND NLP TOOLS
# ==========================================================================================

def preprocess_load_text( filepath: str ) -> str | None:
    """Read UTF-8 text from a local file and return the raw string.

        Purpose:
            Read UTF-8 text from a local file and return the raw string through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            filepath (str): Local filesystem path used by the operation.

        Returns:
            str | None: Value produced by the delegated Fonky implementation.
        
    """
    instance = TextParser( )
    return instance.load_text( filepath=filepath )


def preprocess_collapse_whitespace( text: str ) -> str | None:
    """Normalize spacing by lowercasing text and collapsing repeated whitespace.

        Purpose:
            Normalize spacing by lowercasing text and collapsing repeated whitespace through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            text (str): Text value processed by the operation.

        Returns:
            str | None: Value produced by the delegated Fonky implementation.
        
    """
    instance = TextParser( )
    return instance.collapse_whitespace( text=text )


def preprocess_remove_punctuation( text: str ) -> str:
    """Strip punctuation from tokenized text.

        Purpose:
            Strip punctuation from tokenized text through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            text (str): Text value processed by the operation.

        Returns:
            str: Value produced by the delegated Fonky implementation.
        
    """
    instance = TextParser( )
    return instance.remove_punctuation( text=text )


def preprocess_normalize_text( text: str ) -> str | None:
    """Convert text to lowercase for stable comparison and tokenization.

        Purpose:
            Convert text to lowercase for stable comparison and tokenization through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            text (str): Text value processed by the operation.

        Returns:
            str | None: Value produced by the delegated Fonky implementation.
        
    """
    instance = TextParser( )
    return instance.normalize_text( text=text )


def preprocess_remove_errors( text: str ) -> str:
    """Filter tokens against the NLTK English words corpus.

        Purpose:
            Filter tokens against the NLTK English words corpus through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            text (str): Text value processed by the operation.

        Returns:
            str: Value produced by the delegated Fonky implementation.
        
    """
    instance = TextParser( )
    return instance.remove_errors( text=text )


def preprocess_remove_fragments( text: str ) -> str | None:
    """Remove very short token fragments from normalized text.

        Purpose:
            Remove very short token fragments from normalized text through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            text (str): Text value processed by the operation.

        Returns:
            str | None: Value produced by the delegated Fonky implementation.
        
    """
    instance = TextParser( )
    return instance.remove_fragments( text=text )


def preprocess_remove_symbols( text: str ) -> str | None:
    """Remove configured symbol characters from normalized text.

        Purpose:
            Remove configured symbol characters from normalized text through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            text (str): Text value processed by the operation.

        Returns:
            str | None: Value produced by the delegated Fonky implementation.
        
    """
    instance = TextParser( )
    return instance.remove_symbols( text=text )


def preprocess_remove_html( text: str ) -> str | None:
    """Extract visible text from HTML markup.

        Purpose:
            Extract visible text from HTML markup through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            text (str): Text value processed by the operation.

        Returns:
            str | None: Value produced by the delegated Fonky implementation.
        
    """
    instance = TextParser( )
    return instance.remove_html( text=text )


def preprocess_remove_xml( text: str ) -> str:
    """Extract inner text from XML-like markup.

        Purpose:
            Extract inner text from XML-like markup through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            text (str): Text value processed by the operation.

        Returns:
            str: Value produced by the delegated Fonky implementation.
        
    """
    instance = TextParser( )
    return instance.remove_xml( text=text )


def preprocess_remove_markdown( text: str ) -> str | None:
    """Remove common Markdown links, image syntax, and formatting markers.

        Purpose:
            Remove common Markdown links, image syntax, and formatting markers through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            text (str): Text value processed by the operation.

        Returns:
            str | None: Value produced by the delegated Fonky implementation.
        
    """
    instance = TextParser( )
    return instance.remove_markdown( text=text )


def preprocess_remove_stopwords( text: str ) -> str | None:
    """Remove English stop words from tokenized text.

        Purpose:
            Remove English stop words from tokenized text through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            text (str): Text value processed by the operation.

        Returns:
            str | None: Value produced by the delegated Fonky implementation.
        
    """
    instance = TextParser( )
    return instance.remove_stopwords( text=text )


def preprocess_remove_encodings( text: str ) -> str | None:
    """Resolve HTML entities, normalize Unicode characters, and remove control characters.

        Purpose:
            Resolve HTML entities, normalize Unicode characters, and remove control characters through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            text (str): Text value processed by the operation.

        Returns:
            str | None: Value produced by the delegated Fonky implementation.
        
    """
    instance = TextParser( )
    return instance.remove_encodings( text=text )


def preprocess_remove_headers( filepath: str, lines: int=50, headers: int=3,
		footers: int=3 ) -> str | None:
    """Detect and remove repeated page headers and footers from a text file.

        Purpose:
            Detect and remove repeated page headers and footers from a text file through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

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


def preprocess_remove_numbers( text: str ) -> str | None:
    """Remove decimal digits from text.

        Purpose:
            Remove decimal digits from text through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            text (str): Text value processed by the operation.

        Returns:
            str | None: Value produced by the delegated Fonky implementation.
        
    """
    instance = TextParser( )
    return instance.remove_numbers( text=text )


def preprocess_remove_numerals( text: str ) -> str | None:
    """Remove Roman-numeral patterns from text.

        Purpose:
            Remove Roman-numeral patterns from text through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            text (str): Text value processed by the operation.

        Returns:
            str | None: Value produced by the delegated Fonky implementation.
        
    """
    instance = TextParser( )
    return instance.remove_numerals( text=text )


def preprocess_remove_images( text: str ) -> str:
    """Remove Markdown image references, HTML image elements, and direct image URLs.

        Purpose:
            Remove Markdown image references, HTML image elements, and direct image URLs through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            text (str): Text value processed by the operation.

        Returns:
            str: Value produced by the delegated Fonky implementation.
        
    """
    instance = TextParser( )
    return instance.remove_images( text=text )


def preprocess_tiktokenize( text: str, encoding: str='cl100k_base' ) -> DataFrame | None:
    """Encode text with a tiktoken tokenizer and return token identifiers as tabular data.

        Purpose:
            Encode text with a tiktoken tokenizer and return token identifiers as tabular data through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            text (str): Text value processed by the operation.
            encoding (str): Text encoding or tokenizer encoding used by the operation.

        Returns:
            DataFrame | None: Value produced by the delegated Fonky implementation.
        
    """
    instance = TextParser( )
    return instance.tiktokenize( text=text, encoding=encoding )


def preprocess_split_sentences( text: str ) -> List[str] | None:
    """Split text into sentence strings using NLTK sentence tokenization.

        Purpose:
            Split text into sentence strings using NLTK sentence tokenization through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            text (str): Text value processed by the operation.

        Returns:
            List[str] | None: Value produced by the delegated Fonky implementation.
        
    """
    instance = TextParser( )
    return instance.split_sentences( text=text )


def preprocess_split_pages( filepath: str, num: int=50 ) -> List[str] | None:
    """Split a text file into page-sized text blocks.

        Purpose:
            Split a text file into page-sized text blocks through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            filepath (str): Local filesystem path used by the operation.
            num (int): Num value used by the operation.

        Returns:
            List[str] | None: Value produced by the delegated Fonky implementation.
        
    """
    instance = TextParser( )
    return instance.split_pages( filepath=filepath, num=num )


def preprocess_split_paragraphs( filepath: str ) -> DataFrame | None:
    """Read a text file and return paragraph-like text blocks as tabular data.

        Purpose:
            Read a text file and return paragraph-like text blocks as tabular data through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            filepath (str): Local filesystem path used by the operation.

        Returns:
            DataFrame | None: Value produced by the delegated Fonky implementation.
        
    """
    instance = TextParser( )
    return instance.split_paragraphs( filepath=filepath )


def preprocess_create_frequency_distribution( tokens: List[str] ) -> DataFrame | None:
    """Build a word-frequency table from a token sequence.

        Purpose:
            Build a word-frequency table from a token sequence through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            tokens (List[str]): Token values processed by the operation.

        Returns:
            DataFrame | None: Value produced by the delegated Fonky implementation.
        
    """
    instance = TextParser( )
    return instance.create_frequency_distribution( tokens=tokens )


def preprocess_create_vocabulary( tokens: List[str] ) -> Series | None:
    """Extract the vocabulary column from a token-frequency table.

        Purpose:
            Extract the vocabulary column from a token-frequency table through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            tokens (List[str]): Token values processed by the operation.

        Returns:
            Series | None: Value produced by the delegated Fonky implementation.
        
    """
    instance = TextParser( )
    return instance.create_vocabulary( tokens=tokens )


def preprocess_create_wordbag( tokens: List[str] ) -> DataFrame | None:
    """Build a bag-of-words table from a token sequence.

        Purpose:
            Build a bag-of-words table from a token sequence through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            tokens (List[str]): Token values processed by the operation.

        Returns:
            DataFrame | None: Value produced by the delegated Fonky implementation.
        
    """
    instance = TextParser( )
    return instance.create_wordbag( tokens=tokens )


def preprocess_create_vectors( tokens: List[str] ) -> DataFrame | None:
    """Create TF-IDF vectors for token values.

        Purpose:
            Create TF-IDF vectors for token values through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            tokens (List[str]): Token values processed by the operation.

        Returns:
            DataFrame | None: Value produced by the delegated Fonky implementation.
        
    """
    instance = TextParser( )
    return instance.create_vectors( tokens=tokens )


def preprocess_clean_file( filepath: str ) -> str | None:
    """Apply the standard Fonky text-cleaning pipeline to a single file.

        Purpose:
            Apply the standard Fonky text-cleaning pipeline to a single file through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            filepath (str): Local filesystem path used by the operation.

        Returns:
            str | None: Value produced by the delegated Fonky implementation.
        
    """
    instance = TextParser( )
    return instance.clean_file( filepath=filepath )


def preprocess_clean_files( source: str, destination: str ) -> None:
    """Apply the standard Fonky text-cleaning pipeline to every file in a directory.

        Purpose:
            Apply the standard Fonky text-cleaning pipeline to every file in a directory through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            source (str): Source value used to scope or identify the backing operation.
            destination (str): Destination used to receive generated or processed output.

        Returns:
            None: This function performs its work through the delegated implementation and does not return a value.
        
    """
    instance = TextParser( )
    return instance.clean_files( source=source, destination=destination )


def preprocess_chunk_files( source: str, destination: str ) -> None:
    """Split text files into sentence chunks and write chunked output files.

        Purpose:
            Split text files into sentence chunks and write chunked output files through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            source (str): Source value used to scope or identify the backing operation.
            destination (str): Destination used to receive generated or processed output.

        Returns:
            None: This function performs its work through the delegated implementation and does not return a value.
        
    """
    instance = TextParser( )
    return instance.chunk_files( source=source, destination=destination )


def preprocess_chunk_data( filepath: str, size: int=10 ) -> DataFrame | None:
    """Chunk a text file into fixed-size word groups represented as tabular data.

        Purpose:
            Chunk a text file into fixed-size word groups represented as tabular data through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            filepath (str): Local filesystem path used by the operation.
            size (int): Maximum size or group size used by the operation.

        Returns:
            DataFrame | None: Value produced by the delegated Fonky implementation.
        
    """
    instance = TextParser( )
    return instance.chunk_data( filepath=filepath, size=size )


def preprocess_chunk_datasets( source: str, destination: str, size: int=10 ) -> DataFrame:
    """Clean and chunk a directory of text files into spreadsheet datasets.

        Purpose:
            Clean and chunk a directory of text files into spreadsheet datasets through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            source (str): Source value used to scope or identify the backing operation.
            destination (str): Destination used to receive generated or processed output.
            size (int): Maximum size or group size used by the operation.

        Returns:
            DataFrame: Value produced by the delegated Fonky implementation.
        
    """
    instance = TextParser( )
    return instance.chunk_datasets( source=source, destination=destination, size=size )


def preprocess_convert_jsonl( source: str, destination: str, size: int=10 ) -> None:
    """Convert text files into line-oriented JSON-like chunk output.

        Purpose:
            Convert text files into line-oriented JSON-like chunk output through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            source (str): Source value used to scope or identify the backing operation.
            destination (str): Destination used to receive generated or processed output.
            size (int): Maximum size or group size used by the operation.

        Returns:
            None: This function performs its work through the delegated implementation and does not return a value.
        
    """
    instance = TextParser( )
    return instance.convert_jsonl( source=source, destination=destination, size=size )


def preprocess_encode_sentences( tokens: List[str], model: str='all-MiniLM-L6-v2' ) -> Tuple[List[str], np.ndarray]:
    """Generate sentence-transformer embeddings for normalized token values.

        Purpose:
            Generate sentence-transformer embeddings for normalized token values through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            tokens (List[str]): Token values processed by the operation.
            model (str): Model identifier used by the operation.

        Returns:
            Tuple[List[str], np.ndarray]: Value produced by the delegated Fonky implementation.
        
    """
    instance = TextParser( )
    return instance.encode_sentences( tokens=tokens, model=model )


def nltk_word_tokenizer( text: str ) -> List[str] | None:
    """Tokenize text into lowercased word tokens.

        Purpose:
            Tokenize text into lowercased word tokens through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            text (str): Text value processed by the operation.

        Returns:
            List[str] | None: Value produced by the delegated Fonky implementation.
        
    """
    instance = NltkParser( )
    return instance.word_tokenizer( text=text )


def nltk_sentence_tokenizer( text: str ) -> List[str] | None:
    """Tokenize text into lowercased sentence strings.

        Purpose:
            Tokenize text into lowercased sentence strings through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            text (str): Text value processed by the operation.

        Returns:
            List[str] | None: Value produced by the delegated Fonky implementation.
        
    """
    instance = NltkParser( )
    return instance.sentence_tokenizer( text=text )


def nltk_word_stemmer( text: str ) -> List[str] | None:
    """Stem lowercased word tokens with the configured Porter stemmer.

        Purpose:
            Stem lowercased word tokens with the configured Porter stemmer through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            text (str): Text value processed by the operation.

        Returns:
            List[str] | None: Value produced by the delegated Fonky implementation.
        
    """
    instance = NltkParser( )
    return instance.word_stemmer( text=text )


def nltk_word_lemmatizer( text: str ) -> List[str] | None:
    """Lemmatize lowercased word tokens with the configured WordNet lemmatizer.

        Purpose:
            Lemmatize lowercased word tokens with the configured WordNet lemmatizer through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            text (str): Text value processed by the operation.

        Returns:
            List[str] | None: Value produced by the delegated Fonky implementation.
        
    """
    instance = NltkParser( )
    return instance.word_lemmatizer( text=text )


def nltk_pos_tagger( text: str ) -> List[Tuple[str, str]] | None:
    """Assign part-of-speech tags to lowercased word tokens.

        Purpose:
            Assign part-of-speech tags to lowercased word tokens through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            text (str): Text value processed by the operation.

        Returns:
            List[Tuple[str, str]] | None: Value produced by the delegated Fonky implementation.
        
    """
    instance = NltkParser( )
    return instance.pos_tagger( text=text )


def nltk_named_entity_recognition( text: str ) -> List[Tuple[str, str]] | None:
    """Extract named-entity text and entity labels from tagged tokens.

        Purpose:
            Extract named-entity text and entity labels from tagged tokens through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            text (str): Text value processed by the operation.

        Returns:
            List[Tuple[str, str]] | None: Value produced by the delegated Fonky implementation.
        
    """
    instance = NltkParser( )
    return instance.named_entity_recognition( text=text )


def nltk_chunk_words( text: str, size: int=5 ) -> DataFrame | None:
    """Group word tokens into fixed-size chunks and return them as tabular data.

        Purpose:
            Group word tokens into fixed-size chunks and return them as tabular data through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            text (str): Text value processed by the operation.
            size (int): Maximum size or group size used by the operation.

        Returns:
            DataFrame | None: Value produced by the delegated Fonky implementation.
        
    """
    instance = NltkParser( )
    return instance.chunk_words( text=text, size=size )


def nltk_chunk_sentences( text: str, size: int=15 ) -> DataFrame | None:
    """Group sentence tokens into fixed-size chunks and return them as tabular data.

        Purpose:
            Group sentence tokens into fixed-size chunks and return them as tabular data through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

        Args:
            text (str): Text value processed by the operation.
            size (int): Maximum size or group size used by the operation.

        Returns:
            DataFrame | None: Value produced by the delegated Fonky implementation.
        
    """
    instance = NltkParser( )
    return instance.chunk_sentences( text=text, size=size )


def preprocess_semantic_search( query: str, tokens: List[ str ],
        model: str='all-MiniLM-L6-v2', top: int=5 ) -> List[ Tuple[ str, float ] ]:
    """Search token content by semantic similarity.

        Purpose:
            Search token content by semantic similarity through Fonky's canonical implementation so the callable can be registered directly with a Google ADK Agent through its ``tools`` collection.

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
# XAI GROK TOOL DECLARATIONS
# ==========================================================================================

arxiv_fetch_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_arxiv',
		'description': 'Retrieve ArXiv research documents.',
		'parameters': {
			'type': 'object',
			'properties': {
				'question': {
					'type': 'string',
					'description': 'Search text, lookup value, or provider query submitted by the caller.'
				},
				'max_documents': {
					'anyOf': [
						{
							'type': 'integer'
						},
						{
							'type': 'null'
						}
					],
					'description': 'Maximum number of documents to retrieve.'
				},
				'full_documents': {
					'anyOf': [
						{
							'type': 'boolean'
						},
						{
							'type': 'null'
						}
					],
					'description': 'Whether to retrieve full document content instead of abbreviated search results.'
				},
				'include_metadata': {
					'anyOf': [
						{
							'type': 'boolean'
						},
						{
							'type': 'null'
						}
					],
					'description': 'Whether provider metadata should be included with retrieved content.'
				}
			},
			'required': [ 'question' ],
			'additionalProperties': False
		}
	}
}

google_drive_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_google_drive',
		'description': 'Retrieve Google Drive documents.',
		'parameters': {
			'type': 'object',
			'properties': {
				'question': {
					'type': 'string',
					'description': 'Search text, lookup value, or provider query submitted by the caller.'
				},
				'folder_id': {
					'type': 'string',
					'description': 'Provider folder identifier that scopes the operation.',
					'default': 'root'
				},
				'results': {
					'type': 'integer',
					'description': 'Maximum number of search results to request.',
					'default': 10
				},
				'template': {
					'type': 'string',
					'description': 'Provider query template used to construct the request.',
					'default': 'gdrive-query'
				},
				'mime_type': {
					'anyOf': [
						{
							'type': 'string'
						},
						{
							'type': 'null'
						}
					],
					'description': 'Optional MIME type used to restrict matching files.'
				},
				'mode': {
					'type': 'string',
					'description': 'Operation mode used to select the provider or processing workflow.',
					'default': 'documents'
				}
			},
			'required': [ 'question' ],
			'additionalProperties': False
		}
	}
}

wikipedia_fetch_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_wikipedia',
		'description': 'Retrieve Wikipedia documents.',
		'parameters': {
			'type': 'object',
			'properties': {
				'question': {
					'type': 'string',
					'description': 'Search text, lookup value, or provider query submitted by the caller.'
				},
				'language': {
					'anyOf': [
						{
							'type': 'string'
						},
						{
							'type': 'null'
						}
					],
					'description': 'Language code used for provider results or parsing.'
				},
				'max_documents': {
					'anyOf': [
						{
							'type': 'integer'
						},
						{
							'type': 'null'
						}
					],
					'description': 'Maximum number of documents to retrieve.'
				},
				'include_metadata': {
					'anyOf': [
						{
							'type': 'boolean'
						},
						{
							'type': 'null'
						}
					],
					'description': 'Whether provider metadata should be included with retrieved content.'
				}
			},
			'required': [ 'question' ],
			'additionalProperties': False
		}
	}
}

news_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_news',
		'description': 'Retrieve The News API article.',
		'parameters': {
			'type': 'object',
			'properties': {
				'endpoint': {
					'type': 'string',
					'description': 'Provider endpoint or endpoint family to request.',
					'default': 'all'
				},
				'query': {
					'type': 'string',
					'description': 'Search text, lookup value, or provider query submitted by the caller.',
					'default': ''
				},
				'language': {
					'type': 'string',
					'description': 'Language code used for provider results or parsing.',
					'default': 'en'
				},
				'categories': {
					'type': 'string',
					'description': 'Comma-separated news categories used to include matching articles.',
					'default': ''
				},
				'exclude_categories': {
					'type': 'string',
					'description': 'Filter value used to exclude categories from provider results.',
					'default': ''
				},
				'locale': {
					'type': 'string',
					'description': 'Locale filter applied to news results.',
					'default': ''
				},
				'domains': {
					'type': 'string',
					'description': 'Comma-separated source domains used to include matching news articles.',
					'default': ''
				},
				'exclude_domains': {
					'type': 'string',
					'description': 'Filter value used to exclude domains from provider results.',
					'default': ''
				},
				'source_ids': {
					'type': 'string',
					'description': 'Provider identifiers for the selected source.',
					'default': ''
				},
				'exclude_source_ids': {
					'type': 'string',
					'description': 'Filter value used to exclude source ids from provider results.',
					'default': ''
				},
				'published_after': {
					'type': 'string',
					'description': 'Earliest publication timestamp accepted by the news query.',
					'default': ''
				},
				'published_before': {
					'type': 'string',
					'description': 'Latest publication timestamp accepted by the news query.',
					'default': ''
				},
				'published_on': {
					'type': 'string',
					'description': 'Specific publication date used to restrict news results.',
					'default': ''
				},
				'sort': {
					'type': 'string',
					'description': 'Provider-supported result ordering expression.',
					'default': 'published_at'
				},
				'limit': {
					'type': 'integer',
					'description': 'Maximum number of records or items to return.',
					'default': 10
				},
				'page': {
					'type': 'integer',
					'description': 'One-based result page to request.',
					'default': 1
				},
				'include_similar': {
					'type': 'boolean',
					'description': 'Whether to include similar in the result.',
					'default': True
				},
				'headlines_per_category': {
					'type': 'integer',
					'description': 'Maximum number of headlines returned for each category in headline mode.',
					'default': 6
				},
				'time': {
					'type': 'integer',
					'description': 'Request timeout in seconds.',
					'default': 10
				},
				'api_key': {
					'anyOf': [
						{
							'type': 'string'
						},
						{
							'type': 'null'
						}
					],
					'description': 'Optional credential override used for the active request.'
				}
			},
			'required': [ ],
			'additionalProperties': False
		}
	}
}

cse_search_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_cse_search',
		'description': 'Retrieve Google Programmable Search Engine results.',
		'parameters': {
			'type': 'object',
			'properties': {
				'keywords': {
					'type': 'string',
					'description': 'Search text, lookup value, or provider query submitted by the caller.'
				},
				'results': {
					'type': 'integer',
					'description': 'Maximum number of search results to request.',
					'default': 10
				},
				'start': {
					'type': 'integer',
					'description': 'Starting result position used for pagination.',
					'default': 1
				},
				'exact_terms': {
					'type': 'string',
					'description': 'Phrase that must appear exactly in Google Custom Search results.',
					'default': ''
				},
				'exclude_terms': {
					'type': 'string',
					'description': 'Terms that must not appear in Google Custom Search results.',
					'default': ''
				},
				'file_type': {
					'type': 'string',
					'description': 'File-extension filter applied to Google Custom Search results.',
					'default': ''
				},
				'date_restrict': {
					'type': 'string',
					'description': 'Google Custom Search date restriction expression.',
					'default': ''
				},
				'gl': {
					'type': 'string',
					'description': 'Google country-code boost applied to search results.',
					'default': ''
				},
				'lr': {
					'type': 'string',
					'description': 'Google language restriction expression.',
					'default': ''
				},
				'safe': {
					'type': 'string',
					'description': 'Google SafeSearch setting.',
					'default': 'off'
				},
				'search_type': {
					'type': 'string',
					'description': 'Google Custom Search result type; use the provider-supported image-search value when requesting images.',
					'default': ''
				},
				'site_search': {
					'type': 'string',
					'description': 'Domain or site used to restrict Google Custom Search results.',
					'default': ''
				},
				'site_search_filter': {
					'type': 'string',
					'description': 'Whether ``site_search`` is included or excluded by Google Custom Search.',
					'default': ''
				},
				'sort': {
					'type': 'string',
					'description': 'Provider-supported result ordering expression.',
					'default': ''
				},
				'img_size': {
					'type': 'string',
					'description': 'Image-size filter used for Google image search.',
					'default': ''
				},
				'img_type': {
					'type': 'string',
					'description': 'Google image type filter.',
					'default': ''
				},
				'img_color_type': {
					'type': 'string',
					'description': 'Google image color-type filter.',
					'default': ''
				},
				'img_dominant_color': {
					'type': 'string',
					'description': 'Dominant-color filter used for Google image search.',
					'default': ''
				},
				'time': {
					'type': 'integer',
					'description': 'Request timeout in seconds.',
					'default': 10
				},
				'api_key': {
					'anyOf': [
						{
							'type': 'string'
						},
						{
							'type': 'null'
						}
					],
					'description': 'Optional credential override used for the active request.'
				},
				'cse_id': {
					'anyOf': [
						{
							'type': 'string'
						},
						{
							'type': 'null'
						}
					],
					'description': 'Google Programmable Search Engine identifier.'
				}
			},
			'required': [ 'keywords' ],
			'additionalProperties': False
		}
	}
}

gov_data_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_gov_data',
		'description': 'Retrieve Data.gov package and collection.',
		'parameters': {
			'type': 'object',
			'properties': {
				'mode': {
					'type': 'string',
					'description': 'Operation selector. Supported values detected in the implementation include ``collection``, ``package_summary``, ``search``.',
					'default': 'search'
				},
				'query': {
					'type': 'string',
					'description': 'Search text, lookup value, or provider query submitted by the caller.',
					'default': ''
				},
				'page_size': {
					'type': 'integer',
					'description': 'Maximum number of records requested per page.',
					'default': 10
				},
				'offset_mark': {
					'type': 'string',
					'description': 'Provider continuation marker used for paginated Data.gov search results.',
					'default': '*'
				},
				'sort_field': {
					'type': 'string',
					'description': 'Provider field used to order search results.',
					'default': 'score'
				},
				'sort_order': {
					'type': 'string',
					'description': 'Sort direction applied to the provider search.',
					'default': 'DESC'
				},
				'package_id': {
					'type': 'string',
					'description': 'Provider identifier for the selected package.',
					'default': ''
				},
				'collection': {
					'type': 'string',
					'description': 'Provider collection identifier used to restrict results.',
					'default': ''
				},
				'start_date': {
					'type': 'string',
					'description': 'Inclusive start date for the requested time range, in the provider-supported format.',
					'default': ''
				},
				'time': {
					'type': 'integer',
					'description': 'Request timeout in seconds.',
					'default': 20
				}
			},
			'required': [ ],
			'additionalProperties': False
		}
	}
}

congress_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_congress',
		'description': 'Retrieve Congress.gov legislative data.',
		'parameters': {
			'type': 'object',
			'properties': {
				'mode': {
					'type': 'string',
					'description': 'Operation selector. Supported values detected in the implementation include ``bill_detail``, ``bills``, ``congresses``, ``law_detail``, ``laws``, ``report_detail``, ``reports``.',
					'default': 'congresses'
				},
				'congress': {
					'type': 'integer',
					'description': 'Congress number used to scope legislative records.',
					'default': 0
				},
				'bill_type': {
					'type': 'string',
					'description': 'Provider type selector for bill.',
					'default': ''
				},
				'bill_number': {
					'type': 'integer',
					'description': 'Legislative bill number used with the selected Congress and bill type.',
					'default': 0
				},
				'law_type': {
					'type': 'string',
					'description': 'Provider type selector for law.',
					'default': ''
				},
				'law_number': {
					'type': 'integer',
					'description': 'Public or private law number used with the selected law type.',
					'default': 0
				},
				'report_type': {
					'type': 'string',
					'description': 'Provider type selector for report.',
					'default': ''
				},
				'report_number': {
					'type': 'integer',
					'description': 'Committee report number used with the selected Congress and report type.',
					'default': 0
				},
				'offset': {
					'type': 'integer',
					'description': 'Zero-based result offset used for pagination.',
					'default': 0
				},
				'limit': {
					'type': 'integer',
					'description': 'Maximum number of records or items to return.',
					'default': 20
				},
				'sort': {
					'type': 'string',
					'description': 'Provider-supported result ordering expression.',
					'default': 'updateDate+desc'
				},
				'from_date_time': {
					'type': 'string',
					'description': 'Earliest provider update timestamp to include.',
					'default': ''
				},
				'to_date_time': {
					'type': 'string',
					'description': 'Latest provider update timestamp to include.',
					'default': ''
				},
				'conference': {
					'type': 'boolean',
					'description': 'Whether to restrict committee reports to conference reports.',
					'default': False
				},
				'time': {
					'type': 'integer',
					'description': 'Request timeout in seconds.',
					'default': 20
				}
			},
			'required': [ ],
			'additionalProperties': False
		}
	}
}

internet_archive_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_internet_archive',
		'description': 'Retrieve Internet Archive search and metadata.',
		'parameters': {
			'type': 'object',
			'properties': {
				'keywords': {
					'type': 'string',
					'description': 'Search text, lookup value, or provider query submitted by the caller.'
				},
				'fields': {
					'anyOf': [
						{
							'type': 'array',
							'items': {
								'type': 'string'
							}
						},
						{
							'type': 'null'
						}
					],
					'description': 'Comma-separated or provider-specific field selection.'
				},
				'rows': {
					'type': 'integer',
					'description': 'Maximum number of rows to request.',
					'default': 10
				},
				'page': {
					'type': 'integer',
					'description': 'One-based result page to request.',
					'default': 1
				},
				'sort': {
					'type': 'string',
					'description': 'Provider-supported result ordering expression.',
					'default': 'downloads desc'
				},
				'media_type': {
					'type': 'string',
					'description': 'Provider type selector for media.',
					'default': ''
				},
				'collection': {
					'type': 'string',
					'description': 'Provider collection identifier used to restrict results.',
					'default': ''
				},
				'time': {
					'type': 'integer',
					'description': 'Request timeout in seconds.',
					'default': 20
				}
			},
			'required': [ 'keywords' ],
			'additionalProperties': False
		}
	}
}

grokipedia_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_grokipedia',
		'description': 'Retrieve Grokipedia search and page.',
		'parameters': {
			'type': 'object',
			'properties': {
				'mode': {
					'type': 'string',
					'description': 'Operation mode used to select the provider or processing workflow.',
					'default': 'search'
				},
				'query': {
					'type': 'string',
					'description': 'Search text, lookup value, or provider query submitted by the caller.',
					'default': ''
				},
				'page': {
					'type': 'string',
					'description': 'One-based result page to request.',
					'default': ''
				},
				'limit': {
					'type': 'integer',
					'description': 'Maximum number of records or items to return.',
					'default': 12
				},
				'offset': {
					'type': 'integer',
					'description': 'Zero-based result offset used for pagination.',
					'default': 0
				},
				'include_content': {
					'type': 'boolean',
					'description': 'Whether to include content in the result.',
					'default': True
				}
			},
			'required': [ ],
			'additionalProperties': False
		}
	}
}

arxiv_load_tool = {
	'type': 'function',
	'function': {
		'name': 'load_arxiv',
		'description': 'Load ArXiv research documents.',
		'parameters': {
			'type': 'object',
			'properties': {
				'question': {
					'type': 'string',
					'description': 'Search query or prompt submitted to the backing loader.'
				}
			},
			'required': [ 'question' ],
			'additionalProperties': False
		}
	}
}

wikipedia_load_tool = {
	'type': 'function',
	'function': {
		'name': 'load_wikipedia',
		'description': 'Load Wikipedia articles.',
		'parameters': {
			'type': 'object',
			'properties': {
				'question': {
					'type': 'string',
					'description': 'Search query or prompt submitted to the backing loader.'
				}
			},
			'required': [ 'question' ],
			'additionalProperties': False
		}
	}
}

naval_observatory_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_naval_observatory',
		'description': 'Retrieve U.S. Naval Observatory celestial-navigation data.',
		'parameters': {
			'type': 'object',
			'properties': {
				'mode': {
					'type': 'string',
					'description': 'Operation mode used to select the provider or processing workflow.',
					'default': 'celnav'
				},
				'date_value': {
					'type': 'string',
					'description': 'Calendar date used by the selected provider operation.',
					'default': ''
				},
				'time_value': {
					'type': 'string',
					'description': 'Clock time or timestamp used by the selected provider operation.',
					'default': ''
				},
				'latitude': {
					'type': 'number',
					'description': 'Latitude in decimal degrees.',
					'default': 0.0
				},
				'longitude': {
					'type': 'number',
					'description': 'Longitude in decimal degrees.',
					'default': 0.0
				},
				'location_label': {
					'type': 'string',
					'description': 'Human-readable label associated with the supplied coordinates.',
					'default': ''
				},
				'time': {
					'type': 'integer',
					'description': 'Request timeout in seconds.',
					'default': 20
				}
			},
			'required': [ ],
			'additionalProperties': False
		}
	}
}

satellite_center_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_satellite_center',
		'description': 'Retrieve SSC satellite observatory, ground-station, and location data.',
		'parameters': {
			'type': 'object',
			'properties': {
				'mode': {
					'type': 'string',
					'description': 'Operation mode used to select the provider or processing workflow.',
					'default': 'observatories'
				},
				'query': {
					'type': 'string',
					'description': 'Search text, lookup value, or provider query submitted by the caller.',
					'default': ''
				},
				'start_time': {
					'type': 'string',
					'description': 'Beginning timestamp for the requested provider interval.',
					'default': ''
				},
				'end_time': {
					'type': 'string',
					'description': 'Ending timestamp for the requested provider interval.',
					'default': ''
				},
				'coordinate_systems': {
					'type': 'string',
					'description': 'Coordinate system or comma-separated coordinate systems requested from the satellite service.',
					'default': 'gse'
				},
				'resolution_factor': {
					'type': 'integer',
					'description': 'Sampling resolution factor applied to returned satellite location data.',
					'default': 1
				},
				'time': {
					'type': 'integer',
					'description': 'Request timeout in seconds.',
					'default': 20
				}
			},
			'required': [ ],
			'additionalProperties': False
		}
	}
}

nearby_objects_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_nearby_objects',
		'description': 'Retrieve JPL SSD and CNEOS near-Earth object data.',
		'parameters': {
			'type': 'object',
			'properties': {
				'mode': {
					'type': 'string',
					'description': 'Operation mode used to select the provider or processing workflow.',
					'default': 'close_approaches'
				},
				'start_date': {
					'type': 'string',
					'description': 'Inclusive start date for the requested time range, in the provider-supported format.',
					'default': ''
				},
				'end_date': {
					'type': 'string',
					'description': 'Inclusive end date for the requested time range, in the provider-supported format.',
					'default': ''
				},
				'query': {
					'type': 'string',
					'description': 'Search text, lookup value, or provider query submitted by the caller.',
					'default': ''
				},
				'query_type': {
					'type': 'string',
					'description': 'Provider type selector for query.',
					'default': 'sstr'
				},
				'dist_max': {
					'type': 'string',
					'description': 'Maximum close-approach distance expression accepted by the JPL service.',
					'default': '10LD'
				},
				'body': {
					'type': 'string',
					'description': 'Solar-system body used as the reference object.',
					'default': 'Earth'
				},
				'sort': {
					'type': 'string',
					'description': 'Provider-supported result ordering expression.',
					'default': 'date'
				},
				'limit': {
					'type': 'integer',
					'description': 'Maximum number of records or items to return.',
					'default': 20
				},
				'dv': {
					'type': 'number',
					'description': 'Delta-v threshold or mission constraint used by the near-Earth object query.',
					'default': 6.0
				},
				'dur': {
					'type': 'integer',
					'description': 'Mission duration constraint, in days, used by the near-Earth object query.',
					'default': 360
				},
				'stay': {
					'type': 'integer',
					'description': 'Target stay-duration constraint, in days, used by the near-Earth object query.',
					'default': 8
				},
				'launch': {
					'type': 'string',
					'description': 'Launch-year or launch-window expression used by the near-Earth object query.',
					'default': '2020-2045'
				},
				'h': {
					'type': 'number',
					'description': 'Absolute-magnitude threshold used by the near-Earth object query.',
					'default': 26.0
				},
				'occ': {
					'type': 'integer',
					'description': 'Opportunity-count or occurrence constraint used by the mission query.',
					'default': 7
				},
				'include_physical': {
					'type': 'boolean',
					'description': 'Whether to include physical in the result.',
					'default': True
				},
				'include_close_approaches': {
					'type': 'boolean',
					'description': 'Whether to include close approaches in the result.',
					'default': True
				},
				'ca_body': {
					'type': 'string',
					'description': 'Reference body used for close-approach data.',
					'default': 'Earth'
				},
				'include_discovery': {
					'type': 'boolean',
					'description': 'Whether to include discovery in the result.',
					'default': True
				},
				'time': {
					'type': 'integer',
					'description': 'Request timeout in seconds.',
					'default': 20
				}
			},
			'required': [ ],
			'additionalProperties': False
		}
	}
}

open_science_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_open_science',
		'description': 'Retrieve NASA Open Science Data Repository resources.',
		'parameters': {
			'type': 'object',
			'properties': {
				'mode': {
					'type': 'string',
					'description': 'Operation mode used to select the provider or processing workflow.',
					'default': 'dataset'
				},
				'query': {
					'type': 'string',
					'description': 'Search text, lookup value, or provider query submitted by the caller.',
					'default': ''
				},
				'accession': {
					'type': 'string',
					'description': 'Dataset accession identifier used to retrieve a specific Open Science resource.',
					'default': ''
				},
				'format_value': {
					'type': 'string',
					'description': 'Provider output format.',
					'default': 'json'
				},
				'time': {
					'type': 'integer',
					'description': 'Request timeout in seconds.',
					'default': 20
				}
			},
			'required': [ ],
			'additionalProperties': False
		}
	}
}

space_weather_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_space_weather',
		'description': 'Retrieve NASA DONKI space weather endpoints.',
		'parameters': {
			'type': 'object',
			'properties': {
				'mode': {
					'type': 'string',
					'description': 'Operation mode used to select the provider or processing workflow.',
					'default': 'cme'
				},
				'start_date': {
					'type': 'string',
					'description': 'Inclusive start date for the requested time range, in the provider-supported format.',
					'default': ''
				},
				'end_date': {
					'type': 'string',
					'description': 'Inclusive end date for the requested time range, in the provider-supported format.',
					'default': ''
				},
				'time': {
					'type': 'integer',
					'description': 'Request timeout in seconds.',
					'default': 20
				},
				'location': {
					'type': 'string',
					'description': 'Place name, address, or location description resolved by the provider.',
					'default': 'ALL'
				},
				'catalog': {
					'type': 'string',
					'description': 'Provider catalog filter.',
					'default': 'ALL'
				},
				'notification_type': {
					'type': 'string',
					'description': 'Provider type selector for notification.',
					'default': 'all'
				},
				'most_accurate_only': {
					'type': 'boolean',
					'description': 'Whether to restrict results to the provider-designated most accurate analyses.',
					'default': True
				},
				'complete_entry_only': {
					'type': 'boolean',
					'description': 'Whether to restrict results to complete provider entries.',
					'default': True
				},
				'speed': {
					'type': 'integer',
					'description': 'Minimum or target speed constraint used by the space-weather query.',
					'default': 0
				},
				'half_angle': {
					'type': 'integer',
					'description': 'Half-angle constraint used by the space-weather query.',
					'default': 0
				},
				'keyword': {
					'type': 'string',
					'description': 'Keyword used to filter provider records.',
					'default': ''
				},
				'api_key': {
					'anyOf': [
						{
							'type': 'string'
						},
						{
							'type': 'null'
						}
					],
					'description': 'Optional credential override used for the active request.'
				}
			},
			'required': [ ],
			'additionalProperties': False
		}
	}
}

astro_catalog_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_astro_catalog',
		'description': 'Retrieve Open Astronomy Catalog queries.',
		'parameters': {
			'type': 'object',
			'properties': {
				'mode': {
					'type': 'string',
					'description': 'Operation mode used to select the provider or processing workflow.',
					'default': 'object_query'
				},
				'query': {
					'type': 'string',
					'description': 'Search text, lookup value, or provider query submitted by the caller.',
					'default': ''
				},
				'quantity': {
					'type': 'string',
					'description': 'Provider quantity or field requested from the catalog.',
					'default': ''
				},
				'attributes': {
					'type': 'string',
					'description': 'Provider attributes requested for matching catalog records.',
					'default': ''
				},
				'arguments': {
					'type': 'string',
					'description': 'Keyword arguments passed to the bound callable.',
					'default': ''
				},
				'ra': {
					'type': 'string',
					'description': 'Right ascension value.',
					'default': ''
				},
				'dec': {
					'type': 'string',
					'description': 'Declination value.',
					'default': ''
				},
				'radius': {
					'type': 'integer',
					'description': 'Search radius in the units specified by the operation.',
					'default': 2
				},
				'data_format': {
					'type': 'string',
					'description': 'Provider output data format.',
					'default': 'json'
				},
				'time': {
					'type': 'integer',
					'description': 'Request timeout in seconds.',
					'default': 20
				}
			},
			'required': [ ],
			'additionalProperties': False
		}
	}
}

astro_query_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_astro_query',
		'description': 'Retrieve Simbad and astronomy object search operations.',
		'parameters': {
			'type': 'object',
			'properties': {
				'mode': {
					'type': 'string',
					'description': 'Operation mode used to select the provider or processing workflow.',
					'default': 'object_search'
				},
				'query': {
					'type': 'string',
					'description': 'Search text, lookup value, or provider query submitted by the caller.',
					'default': ''
				},
				'ra': {
					'type': 'string',
					'description': 'Right ascension value.',
					'default': ''
				},
				'dec': {
					'type': 'string',
					'description': 'Declination value.',
					'default': ''
				},
				'radius': {
					'type': 'number',
					'description': 'Search radius in the units specified by the operation.',
					'default': 0.5
				},
				'radius_unit': {
					'type': 'string',
					'description': 'Unit applied to the search radius.',
					'default': 'deg'
				},
				'row_limit': {
					'type': 'integer',
					'description': 'Maximum number of rows returned by the astronomy query.',
					'default': 100
				}
			},
			'required': [ ],
			'additionalProperties': False
		}
	}
}

star_map_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_star_map',
		'description': 'Retrieve astronomical object map links and imagery.',
		'parameters': {
			'type': 'object',
			'properties': {
				'mode': {
					'type': 'string',
					'description': 'Operation mode used to select the provider or processing workflow.',
					'default': 'object_link'
				},
				'query': {
					'type': 'string',
					'description': 'Search text, lookup value, or provider query submitted by the caller.',
					'default': ''
				},
				'ra': {
					'type': 'number',
					'description': 'Right ascension value.',
					'default': 0.0
				},
				'dec': {
					'type': 'number',
					'description': 'Declination value.',
					'default': 0.0
				},
				'zoom': {
					'type': 'integer',
					'description': 'Map or chart zoom level.',
					'default': 5
				},
				'image_source': {
					'type': 'string',
					'description': 'Imagery or survey source used to render the map or chart.',
					'default': 'DSS2'
				},
				'box_color': {
					'type': 'string',
					'description': 'Color used to draw the target box on generated map or chart output.',
					'default': 'yellow'
				},
				'show_box': {
					'type': 'boolean',
					'description': 'Whether to display box in generated output.',
					'default': True
				},
				'show_grid': {
					'type': 'boolean',
					'description': 'Whether to display grid in generated output.',
					'default': True
				},
				'show_lines': {
					'type': 'boolean',
					'description': 'Whether to display lines in generated output.',
					'default': True
				},
				'show_boundaries': {
					'type': 'boolean',
					'description': 'Whether to display boundaries in generated output.',
					'default': True
				},
				'show_const_names': {
					'type': 'boolean',
					'description': 'Whether to display const names in generated output.',
					'default': False
				},
				'time': {
					'type': 'integer',
					'description': 'Request timeout in seconds.',
					'default': 20
				}
			},
			'required': [ ],
			'additionalProperties': False
		}
	}
}

star_chart_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_star_chart',
		'description': 'Retrieve static star chart and coordinate chart generation.',
		'parameters': {
			'type': 'object',
			'properties': {
				'mode': {
					'type': 'string',
					'description': 'Operation mode used to select the provider or processing workflow.',
					'default': 'object_chart'
				},
				'query': {
					'type': 'string',
					'description': 'Search text, lookup value, or provider query submitted by the caller.',
					'default': ''
				},
				'ra': {
					'type': 'number',
					'description': 'Right ascension value.',
					'default': 0.0
				},
				'dec': {
					'type': 'number',
					'description': 'Declination value.',
					'default': 0.0
				},
				'zoom': {
					'type': 'integer',
					'description': 'Map or chart zoom level.',
					'default': 5
				},
				'image_source': {
					'type': 'string',
					'description': 'Imagery or survey source used to render the map or chart.',
					'default': 'DSS2'
				},
				'box_color': {
					'type': 'string',
					'description': 'Color used to draw the target box on generated map or chart output.',
					'default': 'yellow'
				},
				'show_box': {
					'type': 'boolean',
					'description': 'Whether to display box in generated output.',
					'default': True
				},
				'show_grid': {
					'type': 'boolean',
					'description': 'Whether to display grid in generated output.',
					'default': True
				},
				'show_lines': {
					'type': 'boolean',
					'description': 'Whether to display lines in generated output.',
					'default': True
				},
				'show_boundaries': {
					'type': 'boolean',
					'description': 'Whether to display boundaries in generated output.',
					'default': True
				},
				'show_const_names': {
					'type': 'boolean',
					'description': 'Whether to display const names in generated output.',
					'default': False
				},
				'width': {
					'type': 'integer',
					'description': 'Output image or chart width in pixels.',
					'default': 900
				},
				'height': {
					'type': 'integer',
					'description': 'Output image or chart height in pixels.',
					'default': 450
				},
				'magnitude': {
					'type': 'number',
					'description': 'Limiting stellar magnitude used when rendering a chart.',
					'default': 7.5
				},
				'time': {
					'type': 'integer',
					'description': 'Request timeout in seconds.',
					'default': 20
				}
			},
			'required': [ ],
			'additionalProperties': False
		}
	}
}

open_sky_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_open_sky',
		'description': 'Retrieve OpenSky Network aircraft, airport, and state-vector data.',
		'parameters': {
			'type': 'object',
			'properties': {
				'mode': {
					'type': 'string',
					'description': 'Operation selector. Supported values detected in the implementation include ``arrivals_airport``, ``departures_airport``, ``flights_aircraft``, ``states_bbox``, ``track_aircraft``.',
					'default': 'states_bbox'
				},
				'icao24': {
					'type': 'string',
					'description': '24-bit ICAO aircraft transponder address.',
					'default': ''
				},
				'airport': {
					'type': 'string',
					'description': 'ICAO airport identifier used to query arrivals or departures.',
					'default': ''
				},
				'begin': {
					'anyOf': [
						{
							'type': 'integer'
						},
						{
							'type': 'null'
						}
					],
					'description': 'Beginning Unix timestamp for the requested aviation interval.'
				},
				'end': {
					'anyOf': [
						{
							'type': 'integer'
						},
						{
							'type': 'null'
						}
					],
					'description': 'Ending Unix timestamp for the requested aviation interval.'
				},
				'time_value': {
					'anyOf': [
						{
							'type': 'integer'
						},
						{
							'type': 'null'
						}
					],
					'description': 'Clock time or timestamp used by the selected provider operation.'
				},
				'lamin': {
					'anyOf': [
						{
							'type': 'number'
						},
						{
							'type': 'null'
						}
					],
					'description': 'Bounding-box minimum latitude in decimal degrees.'
				},
				'lomin': {
					'anyOf': [
						{
							'type': 'number'
						},
						{
							'type': 'null'
						}
					],
					'description': 'Bounding-box minimum longitude in decimal degrees.'
				},
				'lamax': {
					'anyOf': [
						{
							'type': 'number'
						},
						{
							'type': 'null'
						}
					],
					'description': 'Bounding-box maximum latitude in decimal degrees.'
				},
				'lomax': {
					'anyOf': [
						{
							'type': 'number'
						},
						{
							'type': 'null'
						}
					],
					'description': 'Bounding-box maximum longitude in decimal degrees.'
				},
				'extended': {
					'type': 'boolean',
					'description': 'Whether extended OpenSky state-vector fields should be requested.',
					'default': False
				},
				'client_id': {
					'type': 'string',
					'description': 'Optional credential override used for the active request.'
				},
				'client_secret': {
					'type': 'string',
					'description': 'Optional credential override used for the active request.'
				},
				'time': {
					'type': 'integer',
					'description': 'Request timeout in seconds.',
					'default': 20
				}
			},
			'required': [ ],
			'additionalProperties': False
		}
	}
}

google_drive_file_tool = {
	'type': 'function',
	'function': {
		'name': 'load_google_drive_file',
		'description': 'Load a Google Drive file.',
		'parameters': {
			'type': 'object',
			'properties': {
				'file_id': {
					'type': 'string',
					'description': 'Provider file identifier used to load a single file.'
				},
				'recursive': {
					'type': 'boolean',
					'description': 'Whether the loader should traverse nested provider or URL resources.',
					'default': False
				}
			},
			'required': [ 'file_id' ],
			'additionalProperties': False
		}
	}
}

google_drive_folder_tool = {
	'type': 'function',
	'function': {
		'name': 'load_google_drive_folder',
		'description': 'Load documents from a Google Drive folder.',
		'parameters': {
			'type': 'object',
			'properties': {
				'folder_id': {
					'type': 'string',
					'description': 'Provider folder identifier used to load folder contents.'
				},
				'recursive': {
					'type': 'boolean',
					'description': 'Whether the loader should traverse nested provider or URL resources.',
					'default': False
				}
			},
			'required': [ 'folder_id' ],
			'additionalProperties': False
		}
	}
}

onedrive_tool = {
	'type': 'function',
	'function': {
		'name': 'load_onedrive',
		'description': 'Load documents from OneDrive.',
		'parameters': {
			'type': 'object',
			'properties': {
				'drive_id': {
					'type': 'string',
					'description': 'OneDrive drive identifier.'
				},
				'folder_path': {
					'anyOf': [
						{
							'type': 'string'
						},
						{
							'type': 'null'
						}
					],
					'description': 'Optional folder path within the selected drive.'
				},
				'object_ids': {
					'anyOf': [
						{
							'type': 'array',
							'items': {
								'type': 'string'
							}
						},
						{
							'type': 'null'
						}
					],
					'description': 'Optional provider object identifiers to load.'
				},
				'auth_with_token': {
					'type': 'boolean',
					'description': 'Whether token-based authentication should be used.',
					'default': True
				}
			},
			'required': [ 'drive_id' ],
			'additionalProperties': False
		}
	}
}

google_cloud_file_tool = {
	'type': 'function',
	'function': {
		'name': 'load_google_cloud_file',
		'description': 'Load a Google Cloud Storage object.',
		'parameters': {
			'type': 'object',
			'properties': {
				'project_name': {
					'type': 'string',
					'description': 'Google Cloud project name used by the storage loader.'
				},
				'bucket': {
					'type': 'string',
					'description': 'Storage bucket name.'
				},
				'blob': {
					'type': 'string',
					'description': 'Cloud storage object name.'
				}
			},
			'required': [ 'project_name', 'bucket', 'blob' ],
			'additionalProperties': False
		}
	}
}

aws_file_tool = {
	'type': 'function',
	'function': {
		'name': 'load_aws_file',
		'description': 'Load an Amazon S3 object.',
		'parameters': {
			'type': 'object',
			'properties': {
				'bucket': {
					'type': 'string',
					'description': 'Storage bucket name.'
				},
				'key': {
					'type': 'string',
					'description': 'Amazon S3 object key.'
				},
				'aws_access_key_id': {
					'anyOf': [
						{
							'type': 'string'
						},
						{
							'type': 'null'
						}
					],
					'description': 'Provider identifier for the selected aws access key.'
				},
				'aws_secret_access_key': {
					'anyOf': [
						{
							'type': 'string'
						},
						{
							'type': 'null'
						}
					],
					'description': 'AWS credential or configuration value for secret access key.'
				},
				'aws_session_token': {
					'anyOf': [
						{
							'type': 'string'
						},
						{
							'type': 'null'
						}
					],
					'description': 'AWS credential or configuration value for session token.'
				},
				'region_name': {
					'anyOf': [
						{
							'type': 'string'
						},
						{
							'type': 'null'
						}
					],
					'description': 'Cloud region name used to configure the storage client.'
				}
			},
			'required': [ 'bucket', 'key' ],
			'additionalProperties': False
		}
	}
}

google_speech_to_text_tool = {
	'type': 'function',
	'function': {
		'name': 'load_google_speech_to_text',
		'description': 'Transcribe audio with Google Speech-to-Text.',
		'parameters': {
			'type': 'object',
			'properties': {
				'project_id': {
					'type': 'string',
					'description': 'Google Cloud project identifier used by the speech loader.'
				},
				'file_path': {
					'type': 'string',
					'description': 'Local filesystem path to the source file.'
				},
				'config': {
					'anyOf': [
						{
							'type': 'object',
							'additionalProperties': True
						},
						{
							'type': 'null'
						}
					],
					'description': 'Optional provider configuration mapping.'
				}
			},
			'required': [ 'project_id', 'file_path' ],
			'additionalProperties': False
		}
	}
}

google_bucket_tool = {
	'type': 'function',
	'function': {
		'name': 'load_google_bucket',
		'description': 'Load documents from a Google Cloud Storage bucket.',
		'parameters': {
			'type': 'object',
			'properties': {
				'project_name': {
					'type': 'string',
					'description': 'Google Cloud project name used by the storage loader.'
				},
				'bucket': {
					'type': 'string',
					'description': 'Storage bucket name.'
				},
				'prefix': {
					'anyOf': [
						{
							'type': 'string'
						},
						{
							'type': 'null'
						}
					],
					'description': 'Optional object-name prefix used to restrict cloud storage results.'
				},
				'continue_on_failure': {
					'type': 'boolean',
					'description': 'Whether loading should continue when an individual object fails.',
					'default': False
				}
			},
			'required': [ 'project_name', 'bucket' ],
			'additionalProperties': False
		}
	}
}

aws_bucket_tool = {
	'type': 'function',
	'function': {
		'name': 'load_aws_bucket',
		'description': 'Load documents from an Amazon S3 bucket.',
		'parameters': {
			'type': 'object',
			'properties': {
				'bucket': {
					'type': 'string',
					'description': 'Storage bucket name.'
				},
				'prefix': {
					'anyOf': [
						{
							'type': 'string'
						},
						{
							'type': 'null'
						}
					],
					'description': 'Optional object-name prefix used to restrict cloud storage results.'
				},
				'aws_access_key_id': {
					'anyOf': [
						{
							'type': 'string'
						},
						{
							'type': 'null'
						}
					],
					'description': 'Provider identifier for the selected aws access key.'
				},
				'aws_secret_access_key': {
					'anyOf': [
						{
							'type': 'string'
						},
						{
							'type': 'null'
						}
					],
					'description': 'AWS credential or configuration value for secret access key.'
				},
				'aws_session_token': {
					'anyOf': [
						{
							'type': 'string'
						},
						{
							'type': 'null'
						}
					],
					'description': 'AWS credential or configuration value for session token.'
				},
				'region_name': {
					'anyOf': [
						{
							'type': 'string'
						},
						{
							'type': 'null'
						}
					],
					'description': 'Cloud region name used to configure the storage client.'
				},
				'endpoint_url': {
					'anyOf': [
						{
							'type': 'string'
						},
						{
							'type': 'null'
						}
					],
					'description': 'Optional alternate service endpoint URL.'
				}
			},
			'required': [ 'bucket' ],
			'additionalProperties': False
		}
	}
}

census_data_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_census_data',
		'description': 'Retrieve U.S. Census dataset and variable.',
		'parameters': {
			'type': 'object',
			'properties': {
				'mode': {
					'type': 'string',
					'description': 'Operation selector. Supported values detected in the implementation include ``data``, ``variables``.',
					'default': 'variables'
				},
				'year': {
					'type': 'string',
					'description': 'Dataset or observation year requested from the provider.',
					'default': '2022'
				},
				'dataset': {
					'type': 'string',
					'description': 'Provider dataset name or identifier.',
					'default': 'acs/acs5'
				},
				'fields': {
					'type': 'string',
					'description': 'Comma-separated or provider-specific field selection.',
					'default': 'NAME,B01001_001E'
				},
				'geography_for': {
					'type': 'string',
					'description': 'Census ``for`` geography clause defining the requested geography.',
					'default': 'state:*'
				},
				'geography_in': {
					'type': 'string',
					'description': 'Optional Census ``in`` geography clause constraining the request.',
					'default': ''
				},
				'predicates': {
					'type': 'string',
					'description': 'Additional Census query predicates appended to the request.',
					'default': ''
				},
				'time': {
					'type': 'integer',
					'description': 'Request timeout in seconds.',
					'default': 20
				}
			},
			'required': [ ],
			'additionalProperties': False
		}
	}
}

socrata_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_socrata',
		'description': 'Retrieve Socrata dataset metadata and row.',
		'parameters': {
			'type': 'object',
			'properties': {
				'mode': {
					'type': 'string',
					'description': 'Operation selector. Supported values detected in the implementation include ``metadata``, ``rows``.',
					'default': 'rows'
				},
				'domain': {
					'type': 'string',
					'description': 'Provider domain or host containing the requested dataset.',
					'default': 'data.cdc.gov'
				},
				'dataset_id': {
					'type': 'string',
					'description': 'Provider dataset identifier.',
					'default': ''
				},
				'select': {
					'type': 'string',
					'description': 'Socrata ``$select`` expression defining returned columns or calculations.',
					'default': ''
				},
				'where': {
					'type': 'string',
					'description': 'Socrata ``$where`` filter expression.',
					'default': ''
				},
				'order': {
					'type': 'string',
					'description': 'Provider-supported result ordering expression.',
					'default': ''
				},
				'group': {
					'type': 'string',
					'description': 'Socrata ``$group`` expression used to aggregate rows.',
					'default': ''
				},
				'limit': {
					'type': 'integer',
					'description': 'Maximum number of records or items to return.',
					'default': 25
				},
				'offset': {
					'type': 'integer',
					'description': 'Zero-based result offset used for pagination.',
					'default': 0
				},
				'time': {
					'type': 'integer',
					'description': 'Request timeout in seconds.',
					'default': 20
				}
			},
			'required': [ ],
			'additionalProperties': False
		}
	}
}

united_nations_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_united_nations',
		'description': 'Retrieve United Nations SDMX dataset and query.',
		'parameters': {
			'type': 'object',
			'properties': {
				'mode': {
					'type': 'string',
					'description': 'Operation selector. Supported values detected in the implementation include ``datasets``, ``sdmx_query``.',
					'default': 'datasets'
				},
				'query_path': {
					'type': 'string',
					'description': 'Path identifying the query resource.',
					'default': ''
				},
				'time': {
					'type': 'integer',
					'description': 'Request timeout in seconds.',
					'default': 20
				}
			},
			'required': [ ],
			'additionalProperties': False
		}
	}
}

world_population_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_world_population',
		'description': 'Retrieve WorldPop catalog and raster metadata.',
		'parameters': {
			'type': 'object',
			'properties': {
				'mode': {
					'type': 'string',
					'description': 'Operation selector. Supported values detected in the implementation include ``catalog``, ``raster_metadata``, ``search``.',
					'default': 'catalog'
				},
				'query': {
					'type': 'string',
					'description': 'Search text, lookup value, or provider query submitted by the caller.',
					'default': ''
				},
				'asset_path': {
					'type': 'string',
					'description': 'Path identifying the asset resource.',
					'default': ''
				},
				'page': {
					'type': 'integer',
					'description': 'One-based result page to request.',
					'default': 1
				},
				'page_size': {
					'type': 'integer',
					'description': 'Maximum number of records requested per page.',
					'default': 25
				},
				'time': {
					'type': 'integer',
					'description': 'Request timeout in seconds.',
					'default': 20
				}
			},
			'required': [ ],
			'additionalProperties': False
		}
	}
}

open_city_tool = {
	'type': 'function',
	'function': {
		'name': 'load_open_city',
		'description': 'Load an Open City dataset.',
		'parameters': {
			'type': 'object',
			'properties': {
				'city_id': {
					'type': 'string',
					'description': 'Provider identifier for the selected city.'
				},
				'dataset_id': {
					'type': 'string',
					'description': 'Provider dataset identifier.'
				},
				'limit': {
					'type': 'integer',
					'description': 'Maximum number of records requested from the backing source.',
					'default': 100
				}
			},
			'required': [ 'city_id', 'dataset_id' ],
			'additionalProperties': False
		}
	}
}

text_tool = {
	'type': 'function',
	'function': {
		'name': 'load_text',
		'description': 'Load a plain-text file.',
		'parameters': {
			'type': 'object',
			'properties': {
				'path': {
					'type': 'string',
					'description': 'Local file path used by the loader.'
				},
				'encoding': {
					'anyOf': [
						{
							'type': 'string'
						},
						{
							'type': 'null'
						}
					],
					'description': 'Optional file encoding passed to the backing loader.'
				}
			},
			'required': [ 'path' ],
			'additionalProperties': False
		}
	}
}

csv_tool = {
	'type': 'function',
	'function': {
		'name': 'load_csv',
		'description': 'Load a CSV file.',
		'parameters': {
			'type': 'object',
			'properties': {
				'path': {
					'type': 'string',
					'description': 'Local file path used by the loader.'
				},
				'encoding': {
					'anyOf': [
						{
							'type': 'string'
						},
						{
							'type': 'null'
						}
					],
					'description': 'Optional file encoding passed to the backing loader.',
					'default': 'utf-8'
				},
				'source_column': {
					'anyOf': [
						{
							'type': 'string'
						},
						{
							'type': 'null'
						}
					],
					'description': 'Optional CSV column whose value is stored as the document source.'
				},
				'delimiter': {
					'type': 'string',
					'description': 'Field delimiter used to parse delimited text.',
					'default': ','
				},
				'quotechar': {
					'type': 'string',
					'description': 'Quote character used to parse delimited text.',
					'default': '"'
				}
			},
			'required': [ 'path' ],
			'additionalProperties': False
		}
	}
}

pdf_read_tool = {
	'type': 'function',
	'function': {
		'name': 'read_pdf',
		'description': 'Read a PDF file.',
		'parameters': {
			'type': 'object',
			'properties': {
				'path': {
					'type': 'string',
					'description': 'Local file path used by the loader.'
				},
				'mode': {
					'type': 'string',
					'description': 'Operation mode used to select the provider or processing workflow.',
					'default': 'single'
				}
			},
			'required': [ 'path' ],
			'additionalProperties': False
		}
	}
}

pdf_load_tool = {
	'type': 'function',
	'function': {
		'name': 'load_pdf',
		'description': 'Load and extract a PDF file.',
		'parameters': {
			'type': 'object',
			'properties': {
				'path': {
					'type': 'string',
					'description': 'Local file path used by the loader.'
				},
				'mode': {
					'type': 'string',
					'description': 'Operation mode used to select the provider or processing workflow.',
					'default': 'single'
				},
				'extract': {
					'type': 'string',
					'description': 'PDF text-extraction strategy used by the underlying parser.',
					'default': 'plain'
				},
				'include': {
					'type': 'boolean',
					'description': 'Whether optional embedded content should be included.',
					'default': False
				},
				'format': {
					'type': 'string',
					'description': 'Output or embedded-image format requested from the loader.',
					'default': 'markdown-img'
				},
				'size': {
					'type': 'integer',
					'description': 'Maximum chunk size used for document splitting.',
					'default': 1000
				},
				'overlap': {
					'type': 'integer',
					'description': 'Number of characters or tokens repeated between adjacent chunks.',
					'default': 150
				},
				'has_tables': {
					'type': 'boolean',
					'description': 'Whether table-aware parsing or extraction should be enabled.',
					'default': True
				}
			},
			'required': [ 'path' ],
			'additionalProperties': False
		}
	}
}

excel_tool = {
	'type': 'function',
	'function': {
		'name': 'load_excel',
		'description': 'Load an Excel workbook.',
		'parameters': {
			'type': 'object',
			'properties': {
				'path': {
					'type': 'string',
					'description': 'Local file path used by the loader.'
				},
				'mode': {
					'type': 'string',
					'description': 'Operation mode used to select the provider or processing workflow.',
					'default': 'elements'
				},
				'has_headers': {
					'type': 'boolean',
					'description': 'Whether the first spreadsheet row should be treated as column headers.',
					'default': True
				}
			},
			'required': [ 'path' ],
			'additionalProperties': False
		}
	}
}

word_tool = {
	'type': 'function',
	'function': {
		'name': 'load_word',
		'description': 'Load a Word document.',
		'parameters': {
			'type': 'object',
			'properties': {
				'path': {
					'type': 'string',
					'description': 'Local file path used by the loader.'
				}
			},
			'required': [ 'path' ],
			'additionalProperties': False
		}
	}
}

markdown_tool = {
	'type': 'function',
	'function': {
		'name': 'load_markdown',
		'description': 'Load a Markdown document.',
		'parameters': {
			'type': 'object',
			'properties': {
				'path': {
					'type': 'string',
					'description': 'Local file path used by the loader.'
				}
			},
			'required': [ 'path' ],
			'additionalProperties': False
		}
	}
}

html_tool = {
	'type': 'function',
	'function': {
		'name': 'load_html',
		'description': 'Load an HTML document.',
		'parameters': {
			'type': 'object',
			'properties': {
				'path': {
					'type': 'string',
					'description': 'Local file path used by the loader.'
				}
			},
			'required': [ 'path' ],
			'additionalProperties': False
		}
	}
}

outlook_tool = {
	'type': 'function',
	'function': {
		'name': 'load_outlook',
		'description': 'Load an Outlook message.',
		'parameters': {
			'type': 'object',
			'properties': {
				'path': {
					'type': 'string',
					'description': 'Local file path used by the loader.'
				}
			},
			'required': [ 'path' ],
			'additionalProperties': False
		}
	}
}

spfx_tool = {
	'type': 'function',
	'function': {
		'name': 'load_spfx',
		'description': 'Load a SharePoint document library.',
		'parameters': {
			'type': 'object',
			'properties': {
				'library_id': {
					'type': 'string',
					'description': 'SharePoint document-library identifier.'
				}
			},
			'required': [ 'library_id' ],
			'additionalProperties': False
		}
	}
}

spfx_folder_tool = {
	'type': 'function',
	'function': {
		'name': 'load_spfx_folder',
		'description': 'Load a SharePoint folder.',
		'parameters': {
			'type': 'object',
			'properties': {
				'library_id': {
					'type': 'string',
					'description': 'SharePoint document-library identifier.'
				},
				'folder_id': {
					'type': 'string',
					'description': 'Provider folder identifier used to load folder contents.'
				}
			},
			'required': [ 'library_id', 'folder_id' ],
			'additionalProperties': False
		}
	}
}

powerpoint_tool = {
	'type': 'function',
	'function': {
		'name': 'load_powerpoint',
		'description': 'Load a PowerPoint presentation.',
		'parameters': {
			'type': 'object',
			'properties': {
				'path': {
					'type': 'string',
					'description': 'Local file path used by the loader.'
				},
				'mode': {
					'type': 'string',
					'description': 'Operation mode used to select the provider or processing workflow.',
					'default': 'single'
				}
			},
			'required': [ 'path' ],
			'additionalProperties': False
		}
	}
}

powerpoint_multiple_tool = {
	'type': 'function',
	'function': {
		'name': 'load_powerpoint_multiple',
		'description': 'Load multiple PowerPoint presentation elements.',
		'parameters': {
			'type': 'object',
			'properties': {
				'path': {
					'type': 'string',
					'description': 'Local file path used by the loader.'
				}
			},
			'required': [ 'path' ],
			'additionalProperties': False
		}
	}
}

email_tool = {
	'type': 'function',
	'function': {
		'name': 'load_email',
		'description': 'Load an email message.',
		'parameters': {
			'type': 'object',
			'properties': {
				'path': {
					'type': 'string',
					'description': 'Local file path used by the loader.'
				},
				'mode': {
					'type': 'string',
					'description': 'Operation mode used to select the provider or processing workflow.',
					'default': 'single'
				},
				'attachments': {
					'type': 'boolean',
					'description': 'Whether email attachments should be included when supported.',
					'default': True
				}
			},
			'required': [ 'path' ],
			'additionalProperties': False
		}
	}
}

json_tool = {
	'type': 'function',
	'function': {
		'name': 'load_json',
		'description': 'Load JSON content.',
		'parameters': {
			'type': 'object',
			'properties': {
				'filepath': {
					'type': 'string',
					'description': 'Local file path used by the loader.'
				},
				'is_text': {
					'type': 'boolean',
					'description': 'Whether JSON values should be treated as text content.',
					'default': True
				},
				'is_lines': {
					'type': 'boolean',
					'description': 'Whether the JSON source uses JSON Lines format.',
					'default': False
				}
			},
			'required': [ 'filepath' ],
			'additionalProperties': False
		}
	}
}

xml_tool = {
	'type': 'function',
	'function': {
		'name': 'load_xml',
		'description': 'Load an XML document.',
		'parameters': {
			'type': 'object',
			'properties': {
				'filepath': {
					'type': 'string',
					'description': 'Local file path used by the loader.'
				}
			},
			'required': [ 'filepath' ],
			'additionalProperties': False
		}
	}
}

xml_tree_tool = {
	'type': 'function',
	'function': {
		'name': 'load_xml_tree',
		'description': 'Parse an XML document tree.',
		'parameters': {
			'type': 'object',
			'properties': {
				'filepath': {
					'type': 'string',
					'description': 'Local file path used by the loader.'
				}
			},
			'required': [ 'filepath' ],
			'additionalProperties': False
		}
	}
}

jupyter_notebook_tool = {
	'type': 'function',
	'function': {
		'name': 'load_jupyter_notebook',
		'description': 'Load a Jupyter notebook.',
		'parameters': {
			'type': 'object',
			'properties': {
				'path': {
					'type': 'string',
					'description': 'Local file path used by the loader.'
				},
				'include_outputs': {
					'type': 'boolean',
					'description': 'Whether notebook cell outputs should be included.',
					'default': False
				},
				'max_output_length': {
					'type': 'integer',
					'description': 'Maximum notebook cell output length to retain.',
					'default': 10
				},
				'remove_newline': {
					'type': 'boolean',
					'description': 'Whether newline characters should be removed from notebook output.',
					'default': False
				},
				'traceback': {
					'type': 'boolean',
					'description': 'Whether notebook traceback output should be included.',
					'default': False
				}
			},
			'required': [ 'path' ],
			'additionalProperties': False
		}
	}
}

google_weather_current_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_google_weather_current',
		'description': 'Retrieve google weather current data.',
		'parameters': {
			'type': 'object',
			'properties': {
				'address': {
					'type': 'string',
					'description': 'Street address or place description used for geocoding, validation, or routing.'
				},
				'units_system': {
					'type': 'string',
					'description': 'Measurement unit system requested from the provider.',
					'default': 'METRIC'
				},
				'language_code': {
					'type': 'string',
					'description': 'BCP-47-style language code used for provider results.',
					'default': 'en'
				},
				'time': {
					'type': 'integer',
					'description': 'Request timeout in seconds.',
					'default': 10
				}
			},
			'required': [ 'address' ],
			'additionalProperties': False
		}
	}
}

google_weather_hourly_forecast_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_google_weather_hourly_forecast',
		'description': 'Retrieve hourly forecast.',
		'parameters': {
			'type': 'object',
			'properties': {
				'address': {
					'type': 'string',
					'description': 'Street address or place description used for geocoding, validation, or routing.'
				},
				'hours': {
					'type': 'integer',
					'description': 'Number of hourly observations or forecast periods to request.',
					'default': 24
				},
				'units_system': {
					'type': 'string',
					'description': 'Measurement unit system requested from the provider.',
					'default': 'METRIC'
				},
				'language_code': {
					'type': 'string',
					'description': 'BCP-47-style language code used for provider results.',
					'default': 'en'
				},
				'time': {
					'type': 'integer',
					'description': 'Request timeout in seconds.',
					'default': 10
				}
			},
			'required': [ 'address' ],
			'additionalProperties': False
		}
	}
}

google_weather_daily_forecast_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_google_weather_daily_forecast',
		'description': 'Retrieve daily forecast.',
		'parameters': {
			'type': 'object',
			'properties': {
				'address': {
					'type': 'string',
					'description': 'Street address or place description used for geocoding, validation, or routing.'
				},
				'days': {
					'type': 'integer',
					'description': 'Number of calendar days included in the requested interval.',
					'default': 5
				},
				'units_system': {
					'type': 'string',
					'description': 'Measurement unit system requested from the provider.',
					'default': 'METRIC'
				},
				'language_code': {
					'type': 'string',
					'description': 'BCP-47-style language code used for provider results.',
					'default': 'en'
				},
				'time': {
					'type': 'integer',
					'description': 'Request timeout in seconds.',
					'default': 10
				}
			},
			'required': [ 'address' ],
			'additionalProperties': False
		}
	}
}

google_weather_hourly_history_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_google_weather_hourly_history',
		'description': 'Retrieve hourly history.',
		'parameters': {
			'type': 'object',
			'properties': {
				'address': {
					'type': 'string',
					'description': 'Street address or place description used for geocoding, validation, or routing.'
				},
				'hours': {
					'type': 'integer',
					'description': 'Number of hourly observations or forecast periods to request.',
					'default': 24
				},
				'units_system': {
					'type': 'string',
					'description': 'Measurement unit system requested from the provider.',
					'default': 'METRIC'
				},
				'language_code': {
					'type': 'string',
					'description': 'BCP-47-style language code used for provider results.',
					'default': 'en'
				},
				'time': {
					'type': 'integer',
					'description': 'Request timeout in seconds.',
					'default': 10
				}
			},
			'required': [ 'address' ],
			'additionalProperties': False
		}
	}
}

google_weather_alerts_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_google_weather_alerts',
		'description': 'Retrieve google weather alerts data.',
		'parameters': {
			'type': 'object',
			'properties': {
				'address': {
					'type': 'string',
					'description': 'Street address or place description used for geocoding, validation, or routing.'
				},
				'language_code': {
					'type': 'string',
					'description': 'BCP-47-style language code used for provider results.',
					'default': 'en'
				},
				'time': {
					'type': 'integer',
					'description': 'Request timeout in seconds.',
					'default': 10
				}
			},
			'required': [ 'address' ],
			'additionalProperties': False
		}
	}
}

earth_observatory_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_earth_observatory',
		'description': 'Retrieve NASA EONET events, categories, sources, and layers.',
		'parameters': {
			'type': 'object',
			'properties': {
				'mode': {
					'type': 'string',
					'description': 'Operation mode used to select the provider or processing workflow.',
					'default': 'events'
				},
				'status': {
					'type': 'string',
					'description': 'Provider status filter applied to returned records.',
					'default': 'open'
				},
				'category': {
					'type': 'string',
					'description': 'Optional logical category retained in tool metadata.',
					'default': ''
				},
				'source': {
					'type': 'string',
					'description': 'Provider source identifier used to restrict or classify results.',
					'default': ''
				},
				'limit': {
					'type': 'integer',
					'description': 'Maximum number of records or items to return.',
					'default': 20
				},
				'days': {
					'type': 'integer',
					'description': 'Number of calendar days included in the requested interval.',
					'default': 30
				},
				'start_date': {
					'type': 'string',
					'description': 'Inclusive start date for the requested time range, in the provider-supported format.',
					'default': ''
				},
				'end_date': {
					'type': 'string',
					'description': 'Inclusive end date for the requested time range, in the provider-supported format.',
					'default': ''
				},
				'time': {
					'type': 'integer',
					'description': 'Request timeout in seconds.',
					'default': 20
				}
			},
			'required': [ ],
			'additionalProperties': False
		}
	}
}

open_weather_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_open_weather',
		'description': 'Retrieve Open-Meteo current and forecast weather.',
		'parameters': {
			'type': 'object',
			'properties': {
				'location': {
					'type': 'string',
					'description': 'Place name, address, or location description resolved by the provider.'
				},
				'mode': {
					'type': 'string',
					'description': 'Operation mode used to select the provider or processing workflow.',
					'default': 'current'
				},
				'zone': {
					'type': 'string',
					'description': 'Timezone identifier or automatic timezone-selection mode.',
					'default': 'auto'
				},
				'forecast_days': {
					'type': 'integer',
					'description': 'Number of forecast days to request.',
					'default': 7
				},
				'past_days': {
					'type': 'integer',
					'description': 'Number of historical days to include with the weather request.',
					'default': 0
				},
				'count': {
					'type': 'integer',
					'description': 'Maximum number of matching locations or records to consider.',
					'default': 10
				}
			},
			'required': [ 'location' ],
			'additionalProperties': False
		}
	}
}

historical_weather_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_historical_weather',
		'description': 'Retrieve historical weather archive.',
		'parameters': {
			'type': 'object',
			'properties': {
				'location': {
					'type': 'string',
					'description': 'Place name, address, or location description resolved by the provider.'
				},
				'date': {
					'type': 'object',
					'description': 'Date used by the provider or processing operation.'
				},
				'zone': {
					'type': 'string',
					'description': 'Timezone identifier or automatic timezone-selection mode.',
					'default': 'auto'
				},
				'count': {
					'type': 'integer',
					'description': 'Maximum number of matching locations or records to consider.',
					'default': 10
				}
			},
			'required': [ 'location', 'date' ],
			'additionalProperties': False
		}
	}
}

usgs_earthquakes_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_usgs_earthquakes',
		'description': 'Retrieve USGS earthquake feed and query.',
		'parameters': {
			'type': 'object',
			'properties': {
				'mode': {
					'type': 'string',
					'description': 'Operation selector. Supported values detected in the implementation include ``feed``, ``search``.',
					'default': 'feed'
				},
				'feed': {
					'type': 'string',
					'description': 'Predefined USGS earthquake feed name used when feed mode is selected.',
					'default': 'all_day.geojson'
				},
				'start_date': {
					'type': 'string',
					'description': 'Inclusive start date for the requested time range, in the provider-supported format.',
					'default': ''
				},
				'end_date': {
					'type': 'string',
					'description': 'Inclusive end date for the requested time range, in the provider-supported format.',
					'default': ''
				},
				'min_magnitude': {
					'type': 'number',
					'description': 'Minimum earthquake magnitude to include in the result set.',
					'default': 1.0
				},
				'max_magnitude': {
					'type': 'number',
					'description': 'Maximum earthquake magnitude to include in the result set.',
					'default': 10.0
				},
				'limit': {
					'type': 'integer',
					'description': 'Maximum number of records or items to return.',
					'default': 25
				},
				'order_by': {
					'type': 'string',
					'description': 'Provider-supported field used to order results.',
					'default': 'time'
				},
				'event_type': {
					'type': 'string',
					'description': 'USGS event type to include; ``earthquake`` is the default.',
					'default': 'earthquake'
				},
				'latitude': {
					'anyOf': [
						{
							'type': 'number'
						},
						{
							'type': 'null'
						}
					],
					'description': 'Latitude in decimal degrees.'
				},
				'longitude': {
					'anyOf': [
						{
							'type': 'number'
						},
						{
							'type': 'null'
						}
					],
					'description': 'Longitude in decimal degrees.'
				},
				'max_radius_km': {
					'anyOf': [
						{
							'type': 'number'
						},
						{
							'type': 'null'
						}
					],
					'description': 'Maximum geographic search radius in kilometers.'
				},
				'time': {
					'type': 'integer',
					'description': 'Request timeout in seconds.',
					'default': 20
				}
			},
			'required': [ ],
			'additionalProperties': False
		}
	}
}

usgs_water_data_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_usgs_water_data',
		'description': 'Retrieve USGS water services records.',
		'parameters': {
			'type': 'object',
			'properties': {
				'mode': {
					'type': 'string',
					'description': 'Operation selector. Supported values detected in the implementation include ``latest- continuous``, ``latest-daily``, ``monitoring-locations``, ``time-series-metadata``.',
					'default': 'monitoring-locations'
				},
				'monitoring_location_id': {
					'type': 'string',
					'description': 'USGS monitoring-location identifier used to target a specific site.',
					'default': ''
				},
				'state_code': {
					'type': 'string',
					'description': 'State code used to restrict provider records.',
					'default': ''
				},
				'county_code': {
					'type': 'string',
					'description': 'County code used to restrict provider records.',
					'default': ''
				},
				'site_type': {
					'type': 'string',
					'description': 'USGS site-type code used to restrict monitoring locations.',
					'default': ''
				},
				'parameter_code': {
					'type': 'string',
					'description': 'USGS parameter code identifying the measured property.',
					'default': ''
				},
				'limit': {
					'type': 'integer',
					'description': 'Maximum number of records or items to return.',
					'default': 25
				},
				'time': {
					'type': 'integer',
					'description': 'Request timeout in seconds.',
					'default': 20
				}
			},
			'required': [ ],
			'additionalProperties': False
		}
	}
}

air_now_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_air_now',
		'description': 'Retrieve AirNow current and forecast air quality data.',
		'parameters': {
			'type': 'object',
			'properties': {
				'mode': {
					'type': 'string',
					'description': 'Operation selector. Supported values detected in the implementation include ``current- latlon``, ``current-zip``, ``forecast-latlon``, ``forecast-zip``.',
					'default': 'current-zip'
				},
				'zip_code': {
					'type': 'string',
					'description': 'Provider code identifying or filtering zip.',
					'default': ''
				},
				'latitude': {
					'anyOf': [
						{
							'type': 'number'
						},
						{
							'type': 'null'
						}
					],
					'description': 'Latitude in decimal degrees.'
				},
				'longitude': {
					'anyOf': [
						{
							'type': 'number'
						},
						{
							'type': 'null'
						}
					],
					'description': 'Longitude in decimal degrees.'
				},
				'date': {
					'type': 'string',
					'description': 'Date used by the provider or processing operation.',
					'default': ''
				},
				'distance': {
					'type': 'integer',
					'description': 'Maximum provider search distance, using the units defined by that service.',
					'default': 25
				},
				'time': {
					'type': 'integer',
					'description': 'Request timeout in seconds.',
					'default': 20
				}
			},
			'required': [ ],
			'additionalProperties': False
		}
	}
}

climate_data_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_climate_data',
		'description': 'Retrieve NOAA climate dataset and data records.',
		'parameters': {
			'type': 'object',
			'properties': {
				'mode': {
					'type': 'string',
					'description': 'Operation selector. Supported values detected in the implementation include ``data``, ``datasets``.',
					'default': 'datasets'
				},
				'keyword': {
					'type': 'string',
					'description': 'Keyword used to filter provider records.',
					'default': ''
				},
				'dataset': {
					'type': 'string',
					'description': 'Provider dataset name or identifier.',
					'default': ''
				},
				'start_date': {
					'type': 'string',
					'description': 'Inclusive start date for the requested time range, in the provider-supported format.',
					'default': ''
				},
				'end_date': {
					'type': 'string',
					'description': 'Inclusive end date for the requested time range, in the provider-supported format.',
					'default': ''
				},
				'stations': {
					'type': 'string',
					'description': 'Station identifiers used to restrict climate observations.',
					'default': ''
				},
				'data_types': {
					'type': 'string',
					'description': 'Climate data-type identifiers requested from the provider.',
					'default': ''
				},
				'limit': {
					'type': 'integer',
					'description': 'Maximum number of records or items to return.',
					'default': 25
				},
				'offset': {
					'type': 'integer',
					'description': 'Zero-based result offset used for pagination.',
					'default': 0
				},
				'time': {
					'type': 'integer',
					'description': 'Request timeout in seconds.',
					'default': 20
				}
			},
			'required': [ ],
			'additionalProperties': False
		}
	}
}

eonet_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_eonet',
		'description': 'Retrieve NASA EONET environmental event data.',
		'parameters': {
			'type': 'object',
			'properties': {
				'mode': {
					'type': 'string',
					'description': 'Operation selector. Supported values detected in the implementation include ``categories``, ``events``.',
					'default': 'events'
				},
				'source': {
					'type': 'string',
					'description': 'Provider source identifier used to restrict or classify results.',
					'default': ''
				},
				'category': {
					'type': 'string',
					'description': 'Optional logical category retained in tool metadata.',
					'default': ''
				},
				'status': {
					'type': 'string',
					'description': 'Provider status filter applied to returned records.',
					'default': 'open'
				},
				'limit': {
					'type': 'integer',
					'description': 'Maximum number of records or items to return.',
					'default': 25
				},
				'days': {
					'type': 'integer',
					'description': 'Number of calendar days included in the requested interval.',
					'default': 30
				},
				'start_date': {
					'type': 'string',
					'description': 'Inclusive start date for the requested time range, in the provider-supported format.',
					'default': ''
				},
				'end_date': {
					'type': 'string',
					'description': 'Inclusive end date for the requested time range, in the provider-supported format.',
					'default': ''
				},
				'bbox': {
					'type': 'string',
					'description': 'Bounding box defining the geographic extent of the request.',
					'default': ''
				},
				'time': {
					'type': 'integer',
					'description': 'Request timeout in seconds.',
					'default': 20
				}
			},
			'required': [ ],
			'additionalProperties': False
		}
	}
}

envirofacts_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_envirofacts',
		'description': 'Retrieve EPA Envirofacts table and facility records.',
		'parameters': {
			'type': 'object',
			'properties': {
				'table_name': {
					'type': 'string',
					'description': 'Envirofacts table or resource name to query.',
					'default': 'TRI_FACILITY'
				},
				'state_code': {
					'type': 'string',
					'description': 'State code used to restrict provider records.',
					'default': ''
				},
				'facility_name': {
					'type': 'string',
					'description': 'Facility-name filter applied to Envirofacts records.',
					'default': ''
				},
				'limit': {
					'type': 'integer',
					'description': 'Maximum number of records or items to return.',
					'default': 25
				},
				'time': {
					'type': 'integer',
					'description': 'Request timeout in seconds.',
					'default': 20
				}
			},
			'required': [ ],
			'additionalProperties': False
		}
	}
}

tides_and_currents_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_tides_and_currents',
		'description': 'Retrieve NOAA tides, currents, and station data.',
		'parameters': {
			'type': 'object',
			'properties': {
				'mode': {
					'type': 'string',
					'description': 'Operation selector. Supported values detected in the implementation include ``station``, ``tide-predictions``, ``water-level``.',
					'default': 'water-level'
				},
				'station_id': {
					'type': 'string',
					'description': 'Provider identifier for the selected station.',
					'default': ''
				},
				'begin_date': {
					'type': 'string',
					'description': 'Beginning date for the requested interval, in the provider-supported format.',
					'default': ''
				},
				'end_date': {
					'type': 'string',
					'description': 'Inclusive end date for the requested time range, in the provider-supported format.',
					'default': ''
				},
				'datum': {
					'type': 'string',
					'description': 'Vertical datum used for tide or water-level measurements.',
					'default': 'MLLW'
				},
				'units': {
					'type': 'string',
					'description': 'Unit system used for returned measurements.',
					'default': 'metric'
				},
				'time_zone': {
					'type': 'string',
					'description': 'Timezone used for returned tide or current timestamps.',
					'default': 'gmt'
				},
				'interval': {
					'type': 'string',
					'description': 'Provider sampling or reporting interval.',
					'default': 'hilo'
				},
				'time': {
					'type': 'integer',
					'description': 'Request timeout in seconds.',
					'default': 20
				}
			},
			'required': [ ],
			'additionalProperties': False
		}
	}
}

uv_index_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_uv_index',
		'description': 'Retrieve EPA UV Index current and forecast data.',
		'parameters': {
			'type': 'object',
			'properties': {
				'mode': {
					'type': 'string',
					'description': 'Operation selector. Supported values detected in the implementation include ``daily- city-state``, ``daily-zip``, ``hourly-city-state``, ``hourly-zip``.',
					'default': 'daily-zip'
				},
				'zip_code': {
					'type': 'string',
					'description': 'Provider code identifying or filtering zip.',
					'default': ''
				},
				'city': {
					'type': 'string',
					'description': 'City name used to locate or filter provider records.',
					'default': ''
				},
				'state': {
					'type': 'string',
					'description': 'State name or abbreviation used to locate or filter provider records.',
					'default': ''
				},
				'time': {
					'type': 'integer',
					'description': 'Request timeout in seconds.',
					'default': 20
				}
			},
			'required': [ ],
			'additionalProperties': False
		}
	}
}

purple_air_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_purple_air',
		'description': 'Retrieve PurpleAir sensor and air quality records.',
		'parameters': {
			'type': 'object',
			'properties': {
				'mode': {
					'type': 'string',
					'description': 'Operation selector. Supported values detected in the implementation include ``sensor``, ``sensors``.',
					'default': 'sensors'
				},
				'sensor_index': {
					'anyOf': [
						{
							'type': 'integer'
						},
						{
							'type': 'null'
						}
					],
					'description': 'PurpleAir sensor identifier.'
				},
				'nwlng': {
					'anyOf': [
						{
							'type': 'number'
						},
						{
							'type': 'null'
						}
					],
					'description': 'Northwest bounding-box longitude in decimal degrees.'
				},
				'nwlat': {
					'anyOf': [
						{
							'type': 'number'
						},
						{
							'type': 'null'
						}
					],
					'description': 'Northwest bounding-box latitude in decimal degrees.'
				},
				'selng': {
					'anyOf': [
						{
							'type': 'number'
						},
						{
							'type': 'null'
						}
					],
					'description': 'Southeast bounding-box longitude in decimal degrees.'
				},
				'selat': {
					'anyOf': [
						{
							'type': 'number'
						},
						{
							'type': 'null'
						}
					],
					'description': 'Southeast bounding-box latitude in decimal degrees.'
				},
				'location_type': {
					'type': 'integer',
					'description': 'Provider type selector for location.',
					'default': 0
				},
				'max_age': {
					'type': 'integer',
					'description': 'Maximum age permitted by the operation.',
					'default': 0
				},
				'modified_since': {
					'type': 'integer',
					'description': 'Unix timestamp used to return PurpleAir sensors modified after the specified time.',
					'default': 0
				},
				'fields': {
					'type': 'string',
					'description': 'Comma-separated or provider-specific field selection.',
					'default': ''
				},
				'time': {
					'type': 'integer',
					'description': 'Request timeout in seconds.',
					'default': 20
				}
			},
			'required': [ ],
			'additionalProperties': False
		}
	}
}

open_aq_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_open_aq',
		'description': 'Retrieve OpenAQ location, measurement, and air-quality records.',
		'parameters': {
			'type': 'object',
			'properties': {
				'mode': {
					'type': 'string',
					'description': 'Operation selector. Supported values detected in the implementation include ``countries``, ``latest``, ``locations``, ``parameter_latest``, ``parameters``, ``providers``.',
					'default': 'locations'
				},
				'location_id': {
					'anyOf': [
						{
							'type': 'integer'
						},
						{
							'type': 'null'
						}
					],
					'description': 'Provider identifier for the selected location.'
				},
				'parameter_id': {
					'anyOf': [
						{
							'type': 'integer'
						},
						{
							'type': 'null'
						}
					],
					'description': 'Provider identifier for the selected parameter.'
				},
				'country_id': {
					'anyOf': [
						{
							'type': 'integer'
						},
						{
							'type': 'null'
						}
					],
					'description': 'Provider identifier for the selected country.'
				},
				'coordinates': {
					'type': 'string',
					'description': 'Latitude/longitude coordinate string used by the provider.',
					'default': ''
				},
				'radius': {
					'type': 'integer',
					'description': 'Search radius in the units specified by the operation.',
					'default': 25000
				},
				'providers_id': {
					'type': 'string',
					'description': 'Provider identifier for the selected providers.',
					'default': ''
				},
				'parameters_id': {
					'type': 'string',
					'description': 'Provider identifier for the selected parameters.',
					'default': ''
				},
				'limit': {
					'type': 'integer',
					'description': 'Maximum number of records or items to return.',
					'default': 25
				},
				'page': {
					'type': 'integer',
					'description': 'One-based result page to request.',
					'default': 1
				},
				'time': {
					'type': 'integer',
					'description': 'Request timeout in seconds.',
					'default': 20
				}
			},
			'required': [ ],
			'additionalProperties': False
		}
	}
}

firms_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_firms',
		'description': 'Retrieve NASA FIRMS active fire data.',
		'parameters': {
			'type': 'object',
			'properties': {
				'mode': {
					'type': 'string',
					'description': 'Operation selector. Supported values detected in the implementation include ``area``, ``data-availability``.',
					'default': 'area'
				},
				'source': {
					'type': 'string',
					'description': 'Provider source identifier used to restrict or classify results.',
					'default': 'VIIRS_SNPP_NRT'
				},
				'area_coordinates': {
					'type': 'string',
					'description': 'FIRMS area-of-interest coordinates or ``world`` selector.',
					'default': 'world'
				},
				'day_range': {
					'type': 'integer',
					'description': 'Number of days included in the FIRMS active-fire request.',
					'default': 1
				},
				'date': {
					'type': 'string',
					'description': 'Date used by the provider or processing operation.',
					'default': ''
				},
				'sensor': {
					'type': 'string',
					'description': 'Sensor or instrument filter applied to provider results.',
					'default': 'ALL'
				},
				'time': {
					'type': 'integer',
					'description': 'Request timeout in seconds.',
					'default': 20
				}
			},
			'required': [ ],
			'additionalProperties': False
		}
	}
}

location_tool = {
	'type': 'function',
	'function': {
		'name': 'geocode_location',
		'description': 'Geocode location.',
		'parameters': {
			'type': 'object',
			'properties': {
				'address': {
					'type': 'string',
					'description': 'Street address or place description used for geocoding, validation, or routing.'
				}
			},
			'required': [ 'address' ],
			'additionalProperties': False
		}
	}
}

coordinates_tool = {
	'type': 'function',
	'function': {
		'name': 'geocode_coordinates',
		'description': 'Geocode coordinates.',
		'parameters': {
			'type': 'object',
			'properties': {
				'lat': {
					'type': 'number',
					'description': 'Latitude in decimal degrees.'
				},
				'long': {
					'type': 'number',
					'description': 'Longitude in decimal degrees.'
				}
			},
			'required': [ 'lat', 'long' ],
			'additionalProperties': False
		}
	}
}

address_tool = {
	'type': 'function',
	'function': {
		'name': 'validate_address',
		'description': 'Validate address.',
		'parameters': {
			'type': 'object',
			'properties': {
				'address': {
					'type': 'array',
					'items': {
						'type': 'string'
					},
					'description': 'Street address or place description used for geocoding, validation, or routing.'
				}
			},
			'required': [ 'address' ],
			'additionalProperties': False
		}
	}
}

directions_tool = {
	'type': 'function',
	'function': {
		'name': 'request_directions',
		'description': 'Request directions.',
		'parameters': {
			'type': 'object',
			'properties': {
				'origin': {
					'type': 'string',
					'description': 'Starting address or place for a routing request.'
				},
				'destination': {
					'type': 'string',
					'description': 'Destination address or place for a routing request.'
				},
				'mode': {
					'type': 'string',
					'description': 'Operation mode used to select the provider or processing workflow.',
					'default': 'driving'
				}
			},
			'required': [ 'origin', 'destination' ],
			'additionalProperties': False
		}
	}
}

global_imagery_wms_map_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_global_imagery_wms_map',
		'description': 'Retrieve a WMS imagery map.',
		'parameters': {
			'type': 'object',
			'properties': {
				'layer': {
					'type': 'string',
					'description': 'Map or imagery layer identifier.'
				},
				'image_date': {
					'type': 'string',
					'description': 'Observation date used to select imagery.'
				},
				'bbox': {
					'type': 'array',
					'description': 'Bounding box defining the geographic extent of the request.'
				},
				'width': {
					'type': 'integer',
					'description': 'Output image or chart width in pixels.',
					'default': 1200
				},
				'height': {
					'type': 'integer',
					'description': 'Output image or chart height in pixels.',
					'default': 600
				},
				'projection': {
					'type': 'string',
					'description': 'Coordinate reference system used for rendered imagery.',
					'default': 'epsg4326'
				},
				'quality': {
					'type': 'string',
					'description': 'Imagery quality level requested from the mapping service.',
					'default': 'best'
				},
				'image_format': {
					'type': 'string',
					'description': 'Output format requested for image.',
					'default': 'image/png'
				},
				'transparent': {
					'type': 'boolean',
					'description': 'Whether the generated map image should use a transparent background.',
					'default': True
				},
				'output_dir': {
					'type': 'string',
					'description': 'Local directory where generated imagery is written.',
					'default': 'python-examples'
				},
				'output_name': {
					'type': 'string',
					'description': 'Optional filename for generated imagery.',
					'default': ''
				},
				'time': {
					'type': 'integer',
					'description': 'Request timeout in seconds.',
					'default': 20
				}
			},
			'required': [ 'layer', 'image_date', 'bbox' ],
			'additionalProperties': False
		}
	}
}

global_imagery_map_services_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_global_imagery_map_services',
		'description': 'Retrieve available imagery map services.',
		'parameters': {
			'type': 'object',
			'properties': { },
			'required': [ ],
			'additionalProperties': False
		}
	}
}

global_imagery_mercator_map_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_global_imagery_mercator_map',
		'description': 'Render a Mercator imagery map.',
		'parameters': {
			'type': 'object',
			'properties': {
				'ccrs': {
					'anyOf': [
						{
							'type': 'object'
						},
						{
							'type': 'null'
						}
					],
					'description': 'Optional Cartopy coordinate reference system used to construct the map.'
				}
			},
			'required': [ ],
			'additionalProperties': False
		}
	}
}

google_geocoding_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_google_geocoding',
		'description': 'Retrieve Google forward, reverse, and place geocoding.',
		'parameters': {
			'type': 'object',
			'properties': {
				'mode': {
					'type': 'string',
					'description': 'Operation selector. Supported values detected in the implementation include ``forward``, ``place``, ``reverse``.',
					'default': 'forward'
				},
				'query': {
					'type': 'string',
					'description': 'Search text, lookup value, or provider query submitted by the caller.',
					'default': ''
				},
				'latitude': {
					'type': 'number',
					'description': 'Latitude in decimal degrees.',
					'default': 0.0
				},
				'longitude': {
					'type': 'number',
					'description': 'Longitude in decimal degrees.',
					'default': 0.0
				},
				'place_id': {
					'type': 'string',
					'description': 'Provider identifier for the selected place.',
					'default': ''
				},
				'language': {
					'type': 'string',
					'description': 'Language code used for provider results or parsing.',
					'default': 'en'
				},
				'region': {
					'type': 'string',
					'description': 'Provider region filter or regional bias value.',
					'default': ''
				},
				'result_type': {
					'type': 'string',
					'description': 'Provider type selector for result.',
					'default': ''
				},
				'location_type': {
					'type': 'string',
					'description': 'Provider type selector for location.',
					'default': ''
				},
				'time': {
					'type': 'integer',
					'description': 'Request timeout in seconds.',
					'default': 10
				},
				'api_key': {
					'anyOf': [
						{
							'type': 'string'
						},
						{
							'type': 'null'
						}
					],
					'description': 'Optional credential override used for the active request.'
				}
			},
			'required': [ ],
			'additionalProperties': False
		}
	}
}

usgs_national_map_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_usgs_national_map',
		'description': 'Retrieve USGS National Map datasets and products.',
		'parameters': {
			'type': 'object',
			'properties': {
				'mode': {
					'type': 'string',
					'description': 'Operation selector. Supported values detected in the implementation include ``datasets``, ``products``.',
					'default': 'products'
				},
				'dataset': {
					'type': 'string',
					'description': 'Provider dataset name or identifier.',
					'default': ''
				},
				'q': {
					'type': 'string',
					'description': 'Free-text provider query used to search matching records.',
					'default': ''
				},
				'bbox': {
					'type': 'string',
					'description': 'Bounding box defining the geographic extent of the request.',
					'default': ''
				},
				'prod_formats': {
					'type': 'string',
					'description': 'Product-format filter applied to National Map results.',
					'default': ''
				},
				'max_items': {
					'type': 'integer',
					'description': 'Maximum number of records or items to return.',
					'default': 25
				},
				'offset': {
					'type': 'integer',
					'description': 'Zero-based result offset used for pagination.',
					'default': 0
				},
				'time': {
					'type': 'integer',
					'description': 'Request timeout in seconds.',
					'default': 20
				}
			},
			'required': [ ],
			'additionalProperties': False
		}
	}
}

usgs_sciencebase_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_usgs_sciencebase',
		'description': 'Retrieve USGS ScienceBase items and catalog records.',
		'parameters': {
			'type': 'object',
			'properties': {
				'mode': {
					'type': 'string',
					'description': 'Operation selector. Supported values detected in the implementation include ``item``, ``items``.',
					'default': 'items'
				},
				'q': {
					'type': 'string',
					'description': 'Free-text provider query used to search matching records.',
					'default': ''
				},
				'item_id': {
					'type': 'string',
					'description': 'Provider identifier for the selected item.',
					'default': ''
				},
				'max_items': {
					'type': 'integer',
					'description': 'Maximum number of records or items to return.',
					'default': 25
				},
				'offset': {
					'type': 'integer',
					'description': 'Zero-based result offset used for pagination.',
					'default': 0
				},
				'fields': {
					'type': 'string',
					'description': 'Comma-separated or provider-specific field selection.',
					'default': ''
				},
				'time': {
					'type': 'integer',
					'description': 'Request timeout in seconds.',
					'default': 20
				}
			},
			'required': [ ],
			'additionalProperties': False
		}
	}
}

health_data_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_health_data',
		'description': 'Retrieve HealthData.gov Socrata metadata and rows.',
		'parameters': {
			'type': 'object',
			'properties': {
				'mode': {
					'type': 'string',
					'description': 'Operation selector. Supported values detected in the implementation include ``metadata``, ``rows``.',
					'default': 'rows'
				},
				'domain': {
					'type': 'string',
					'description': 'Provider domain or host containing the requested dataset.',
					'default': 'healthdata.gov'
				},
				'dataset_id': {
					'type': 'string',
					'description': 'Provider dataset identifier.',
					'default': ''
				},
				'select': {
					'type': 'string',
					'description': 'Socrata ``$select`` expression defining returned columns or calculations.',
					'default': ''
				},
				'where': {
					'type': 'string',
					'description': 'Socrata ``$where`` filter expression.',
					'default': ''
				},
				'order': {
					'type': 'string',
					'description': 'Provider-supported result ordering expression.',
					'default': ''
				},
				'group': {
					'type': 'string',
					'description': 'Socrata ``$group`` expression used to aggregate rows.',
					'default': ''
				},
				'limit': {
					'type': 'integer',
					'description': 'Maximum number of records or items to return.',
					'default': 25
				},
				'offset': {
					'type': 'integer',
					'description': 'Zero-based result offset used for pagination.',
					'default': 0
				},
				'time': {
					'type': 'integer',
					'description': 'Request timeout in seconds.',
					'default': 20
				}
			},
			'required': [ ],
			'additionalProperties': False
		}
	}
}

global_health_data_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_global_health_data',
		'description': 'Retrieve WHO global health indicator and Athena data.',
		'parameters': {
			'type': 'object',
			'properties': {
				'mode': {
					'type': 'string',
					'description': 'Operation mode used to select the backing workflow.',
					'default': 'indicator_registry'
				},
				'query_path': {
					'type': 'string',
					'description': 'Query path value used by the operation.',
					'default': ''
				},
				'fmt': {
					'type': 'string',
					'description': 'Fmt value used by the operation.',
					'default': 'json'
				},
				'time': {
					'type': 'integer',
					'description': 'Request timeout in seconds.',
					'default': 20
				}
			},
			'required': [ ],
			'additionalProperties': False
		}
	}
}

wonder_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_wonder',
		'description': 'Retrieve CDC WONDER template and query submission.',
		'parameters': {
			'type': 'object',
			'properties': {
				'mode': {
					'type': 'string',
					'description': 'Operation mode used to select the backing workflow.',
					'default': 'metadata_template'
				},
				'dataset_id': {
					'type': 'string',
					'description': 'Dataset id value used by the operation.',
					'default': 'D76'
				},
				'request_xml': {
					'type': 'string',
					'description': 'Request xml value used by the operation.',
					'default': ''
				},
				'time': {
					'type': 'integer',
					'description': 'Request timeout in seconds.',
					'default': 20
				}
			},
			'required': [ ],
			'additionalProperties': False
		}
	}
}

pubmed_tool = {
	'type': 'function',
	'function': {
		'name': 'load_pubmed',
		'description': 'Load PubMed research documents.',
		'parameters': {
			'type': 'object',
			'properties': {
				'query': {
					'type': 'string',
					'description': 'Search query or natural-language request submitted to the backing operation.'
				},
				'max_docs': {
					'type': 'integer',
					'description': 'Max docs value used by the operation.',
					'default': 5
				}
			},
			'required': [ 'query' ],
			'additionalProperties': False
		}
	}
}

web_page_fetch_tool = {
	'type': 'function',
	'function': {
		'name': 'fetch_web_page',
		'description': 'Retrieve HTTP web page content and HTML extraction data.',
		'parameters': {
			'type': 'object',
			'properties': {
				'url': {
					'type': 'string',
					'description': 'URL used by the operation.'
				},
				'time': {
					'type': 'integer',
					'description': 'Request timeout in seconds.',
					'default': 10
				}
			},
			'required': [ 'url' ],
			'additionalProperties': False
		}
	}
}

html_to_text_convert_tool = {
	'type': 'function',
	'function': {
		'name': 'convert_html_to_text',
		'description': 'Convert HTML to plain text.',
		'parameters': {
			'type': 'object',
			'properties': {
				'html': {
					'type': 'string',
					'description': 'Html value used by the operation.'
				}
			},
			'required': [ 'html' ],
			'additionalProperties': False
		}
	}
}

web_title_tool = {
	'type': 'function',
	'function': {
		'name': 'extract_web_title',
		'description': 'Extract a web title from supplied HTML content.',
		'parameters': {
			'type': 'object',
			'properties': {
				'html': {
					'type': 'string',
					'description': 'Html value used by the operation.'
				}
			},
			'required': [ 'html' ],
			'additionalProperties': False
		}
	}
}

web_links_tool = {
	'type': 'function',
	'function': {
		'name': 'extract_web_links',
		'description': 'Extract web links from supplied HTML content.',
		'parameters': {
			'type': 'object',
			'properties': {
				'base_url': {
					'type': 'string',
					'description': 'Base url value used by the operation.'
				},
				'html': {
					'type': 'string',
					'description': 'Html value used by the operation.'
				}
			},
			'required': [ 'base_url', 'html' ],
			'additionalProperties': False
		}
	}
}

web_structured_data_tool = {
	'type': 'function',
	'function': {
		'name': 'extract_web_structured_data',
		'description': 'Extract structured data from supplied HTML content.',
		'parameters': {
			'type': 'object',
			'properties': {
				'url': {
					'type': 'string',
					'description': 'URL used by the operation.'
				},
				'html': {
					'type': 'string',
					'description': 'Html value used by the operation.'
				},
				'selected_methods': {
					'anyOf': [
						{
							'type': 'array',
							'items': {
								'type': 'string'
							}
						},
						{
							'type': 'null'
						}
					],
					'description': 'Selected methods value used by the operation.'
				}
			},
			'required': [ 'url', 'html' ],
			'additionalProperties': False
		}
	}
}

web_crawl_tool = {
	'type': 'function',
	'function': {
		'name': 'crawl_web',
		'description': 'Crawl web pages from a seed URL.',
		'parameters': {
			'type': 'object',
			'properties': {
				'seed_url': {
					'type': 'string',
					'description': 'Seed url value used by the operation.'
				},
				'include_title': {
					'type': 'boolean',
					'description': 'Include title value used by the operation.',
					'default': True
				},
				'include_basic_text': {
					'type': 'boolean',
					'description': 'Include basic text value used by the operation.',
					'default': True
				},
				'include_raw_html': {
					'type': 'boolean',
					'description': 'Include raw html value used by the operation.',
					'default': False
				},
				'selected_methods': {
					'anyOf': [
						{
							'type': 'array',
							'items': {
								'type': 'string'
							}
						},
						{
							'type': 'null'
						}
					],
					'description': 'Selected methods value used by the operation.'
				},
				'recursive': {
					'type': 'boolean',
					'description': 'Whether nested resources should be traversed recursively.',
					'default': False
				},
				'max_depth': {
					'type': 'integer',
					'description': 'Max depth value used by the operation.',
					'default': 1
				},
				'max_pages': {
					'type': 'integer',
					'description': 'Max pages value used by the operation.',
					'default': 10
				},
				'same_domain_only': {
					'type': 'boolean',
					'description': 'Same domain only value used by the operation.',
					'default': True
				},
				'request_timeout': {
					'type': 'integer',
					'description': 'Request timeout value used by the operation.',
					'default': 10
				},
				'delay_seconds': {
					'type': 'number',
					'description': 'Delay seconds value used by the operation.',
					'default': 0.25
				},
				'max_bytes': {
					'type': 'integer',
					'description': 'Max bytes value used by the operation.',
					'default': 1000000
				},
				'headers': {
					'anyOf': [
						{
							'type': 'object',
							'additionalProperties': True
						},
						{
							'type': 'null'
						}
					],
					'description': 'Headers value used by the operation.'
				},
				'use_playwright': {
					'type': 'boolean',
					'description': 'Use playwright value used by the operation.',
					'default': False
				}
			},
			'required': [ 'seed_url' ],
			'additionalProperties': False
		}
	}
}

crawler_page_tool = {
	'type': 'function',
	'function': {
		'name': 'scrape_crawler_page',
		'description': 'Extract a crawler page from an HTML page.',
		'parameters': {
			'type': 'object',
			'properties': {
				'url': {
					'type': 'string',
					'description': 'URL used by the operation.'
				},
				'include_title': {
					'type': 'boolean',
					'description': 'Include title value used by the operation.',
					'default': True
				},
				'include_basic_text': {
					'type': 'boolean',
					'description': 'Include basic text value used by the operation.',
					'default': True
				},
				'include_raw_html': {
					'type': 'boolean',
					'description': 'Include raw html value used by the operation.',
					'default': False
				},
				'selected_methods': {
					'anyOf': [
						{
							'type': 'array',
							'items': {
								'type': 'string'
							}
						},
						{
							'type': 'null'
						}
					],
					'description': 'Selected methods value used by the operation.'
				},
				'request_timeout': {
					'type': 'integer',
					'description': 'Request timeout value used by the operation.',
					'default': 10
				},
				'max_bytes': {
					'type': 'integer',
					'description': 'Max bytes value used by the operation.',
					'default': 1000000
				},
				'headers': {
					'anyOf': [
						{
							'type': 'object',
							'additionalProperties': True
						},
						{
							'type': 'null'
						}
					],
					'description': 'Headers value used by the operation.'
				},
				'use_playwright': {
					'type': 'boolean',
					'description': 'Use playwright value used by the operation.',
					'default': False
				}
			},
			'required': [ 'url' ],
			'additionalProperties': False
		}
	}
}

web_page_render_tool = {
	'type': 'function',
	'function': {
		'name': 'render_web_page',
		'description': 'Render a dynamic web page with Playwright.',
		'parameters': {
			'type': 'object',
			'properties': {
				'url': {
					'type': 'string',
					'description': 'URL used by the operation.'
				},
				'timeout': {
					'type': 'integer',
					'description': 'Maximum time in seconds to wait for the operation.',
					'default': 15
				},
				'headers': {
					'anyOf': [
						{
							'type': 'object',
							'additionalProperties': True
						},
						{
							'type': 'null'
						}
					],
					'description': 'Headers value used by the operation.'
				},
				'use_playwright': {
					'type': 'boolean',
					'description': 'Use playwright value used by the operation.',
					'default': False
				}
			},
			'required': [ 'url' ],
			'additionalProperties': False
		}
	}
}

web_load_tool = {
	'type': 'function',
	'function': {
		'name': 'load_web',
		'description': 'Load web documents.',
		'parameters': {
			'type': 'object',
			'properties': {
				'urls': {
					'anyOf': [
						{
							'type': 'string'
						},
						{
							'type': 'array',
							'items': {
								'type': 'string'
							}
						}
					],
					'description': 'Urls value used by the operation.'
				},
				'recursive': {
					'type': 'boolean',
					'description': 'Whether nested resources should be traversed recursively.',
					'default': False
				},
				'max_depth': {
					'type': 'integer',
					'description': 'Max depth value used by the operation.',
					'default': 2
				},
				'prevent_outside': {
					'type': 'boolean',
					'description': 'Prevent outside value used by the operation.',
					'default': True
				},
				'timeout': {
					'type': 'integer',
					'description': 'Maximum time in seconds to wait for the operation.',
					'default': 10
				},
				'ignore': {
					'type': 'boolean',
					'description': 'Ignore value used by the operation.',
					'default': True
				},
				'progress': {
					'type': 'boolean',
					'description': 'Progress value used by the operation.',
					'default': True
				}
			},
			'required': [ 'urls' ],
			'additionalProperties': False
		}
	}
}

web_recursive_tool = {
	'type': 'function',
	'function': {
		'name': 'load_web_recursive',
		'description': 'Recursively load web documents.',
		'parameters': {
			'type': 'object',
			'properties': {
				'url': {
					'type': 'string',
					'description': 'URL used by the operation.'
				},
				'depth': {
					'type': 'integer',
					'description': 'Depth value used by the operation.',
					'default': 2
				},
				'max_time': {
					'type': 'integer',
					'description': 'Max time value used by the operation.',
					'default': 10
				},
				'ignore': {
					'type': 'boolean',
					'description': 'Ignore value used by the operation.',
					'default': True
				}
			},
			'required': [ 'url' ],
			'additionalProperties': False
		}
	}
}

web_pages_tool = {
	'type': 'function',
	'function': {
		'name': 'load_web_pages',
		'description': 'Load static web pages.',
		'parameters': {
			'type': 'object',
			'properties': {
				'urls': {
					'type': 'array',
					'items': {
						'type': 'string'
					},
					'description': 'Urls value used by the operation.'
				},
				'depth': {
					'type': 'integer',
					'description': 'Depth value used by the operation.',
					'default': 2
				},
				'timeout': {
					'type': 'integer',
					'description': 'Maximum time in seconds to wait for the operation.',
					'default': 10
				},
				'ignore': {
					'type': 'boolean',
					'description': 'Ignore value used by the operation.',
					'default': True
				},
				'progress': {
					'type': 'boolean',
					'description': 'Progress value used by the operation.',
					'default': True
				}
			},
			'required': [ 'urls' ],
			'additionalProperties': False
		}
	}
}

github_tool = {
	'type': 'function',
	'function': {
		'name': 'load_github',
		'description': 'Load files from a GitHub repository.',
		'parameters': {
			'type': 'object',
			'properties': {
				'url': {
					'type': 'string',
					'description': 'URL used by the operation.'
				},
				'repo': {
					'type': 'string',
					'description': 'Repo value used by the operation.'
				},
				'branch': {
					'type': 'string',
					'description': 'Branch value used by the operation.'
				},
				'filetype': {
					'type': 'string',
					'description': 'Filetype value used by the operation.',
					'default': '.md'
				}
			},
			'required': [ 'url', 'repo', 'branch' ],
			'additionalProperties': False
		}
	}
}

web_page_scrape_tool = {
	'type': 'function',
	'function': {
		'name': 'scrape_web_page',
		'description': 'Fetch a web page for extraction.',
		'parameters': {
			'type': 'object',
			'properties': {
				'url': {
					'type': 'string',
					'description': 'URL used by the operation.'
				},
				'time': {
					'type': 'integer',
					'description': 'Request timeout in seconds.',
					'default': 10
				}
			},
			'required': [ 'url' ],
			'additionalProperties': False
		}
	}
}

html_to_text_scraper_tool = {
	'type': 'function',
	'function': {
		'name': 'scraper_html_to_text',
		'description': 'Convert scraper HTML to plain text.',
		'parameters': {
			'type': 'object',
			'properties': {
				'html': {
					'type': 'string',
					'description': 'Html value used by the operation.'
				}
			},
			'required': [ 'html' ],
			'additionalProperties': False
		}
	}
}

paragraphs_tool = {
	'type': 'function',
	'function': {
		'name': 'scrape_paragraphs',
		'description': 'Extract paragraph text from an HTML page.',
		'parameters': {
			'type': 'object',
			'properties': {
				'uri': {
					'type': 'string',
					'description': 'URI used by the operation.'
				}
			},
			'required': [ 'uri' ],
			'additionalProperties': False
		}
	}
}

lists_tool = {
	'type': 'function',
	'function': {
		'name': 'scrape_lists',
		'description': 'Extract list-item text from an HTML page.',
		'parameters': {
			'type': 'object',
			'properties': {
				'uri': {
					'type': 'string',
					'description': 'URI used by the operation.'
				}
			},
			'required': [ 'uri' ],
			'additionalProperties': False
		}
	}
}

tables_tool = {
	'type': 'function',
	'function': {
		'name': 'scrape_tables',
		'description': 'Extract table-cell text from an HTML page.',
		'parameters': {
			'type': 'object',
			'properties': {
				'uri': {
					'type': 'string',
					'description': 'URI used by the operation.'
				}
			},
			'required': [ 'uri' ],
			'additionalProperties': False
		}
	}
}

articles_tool = {
	'type': 'function',
	'function': {
		'name': 'scrape_articles',
		'description': 'Extract article text from an HTML page.',
		'parameters': {
			'type': 'object',
			'properties': {
				'uri': {
					'type': 'string',
					'description': 'URI used by the operation.'
				}
			},
			'required': [ 'uri' ],
			'additionalProperties': False
		}
	}
}

headings_tool = {
	'type': 'function',
	'function': {
		'name': 'scrape_headings',
		'description': 'Extract heading text from an HTML page.',
		'parameters': {
			'type': 'object',
			'properties': {
				'uri': {
					'type': 'string',
					'description': 'URI used by the operation.'
				}
			},
			'required': [ 'uri' ],
			'additionalProperties': False
		}
	}
}

divisions_tool = {
	'type': 'function',
	'function': {
		'name': 'scrape_divisions',
		'description': 'Extract division text from an HTML page.',
		'parameters': {
			'type': 'object',
			'properties': {
				'uri': {
					'type': 'string',
					'description': 'URI used by the operation.'
				}
			},
			'required': [ 'uri' ],
			'additionalProperties': False
		}
	}
}

sections_tool = {
	'type': 'function',
	'function': {
		'name': 'scrape_sections',
		'description': 'Extract section text from an HTML page.',
		'parameters': {
			'type': 'object',
			'properties': {
				'uri': {
					'type': 'string',
					'description': 'URI used by the operation.'
				}
			},
			'required': [ 'uri' ],
			'additionalProperties': False
		}
	}
}

blockquotes_tool = {
	'type': 'function',
	'function': {
		'name': 'scrape_blockquotes',
		'description': 'Extract blockquote text from an HTML page.',
		'parameters': {
			'type': 'object',
			'properties': {
				'uri': {
					'type': 'string',
					'description': 'URI used by the operation.'
				}
			},
			'required': [ 'uri' ],
			'additionalProperties': False
		}
	}
}

hyperlinks_tool = {
	'type': 'function',
	'function': {
		'name': 'scrape_hyperlinks',
		'description': 'Extract hyperlinks from an HTML page.',
		'parameters': {
			'type': 'object',
			'properties': {
				'uri': {
					'type': 'string',
					'description': 'URI used by the operation.'
				}
			},
			'required': [ 'uri' ],
			'additionalProperties': False
		}
	}
}

images_tool = {
	'type': 'function',
	'function': {
		'name': 'scrape_images',
		'description': 'Extract image references from an HTML page.',
		'parameters': {
			'type': 'object',
			'properties': {
				'uri': {
					'type': 'string',
					'description': 'URI used by the operation.'
				}
			},
			'required': [ 'uri' ],
			'additionalProperties': False
		}
	}
}

image_tool = {
	'type': 'function',
	'function': {
		'name': 'encode_image',
		'description': 'Encode a local image as Base64 text.',
		'parameters': {
			'type': 'object',
			'properties': {
				'path': {
					'type': 'string',
					'description': 'Local filesystem path used by the operation.'
				}
			},
			'required': [ 'path' ],
			'additionalProperties': False
		}
	}
}

load_text_tool = {
	'type': 'function',
	'function': {
		'name': 'preprocess_load_text',
		'description': 'Read UTF-8 text from a local file and return the raw string.',
		'parameters': {
			'type': 'object',
			'properties': {
				'filepath': {
					'type': 'string',
					'description': 'Local filesystem path used by the operation.'
				}
			},
			'required': [ 'filepath' ],
			'additionalProperties': False
		}
	}
}

collapse_whitespace_tool = {
	'type': 'function',
	'function': {
		'name': 'preprocess_collapse_whitespace',
		'description': 'Normalize spacing by lowercasing text and collapsing repeated whitespace.',
		'parameters': {
			'type': 'object',
			'properties': {
				'text': {
					'type': 'string',
					'description': 'Text value processed by the operation.'
				}
			},
			'required': [ 'text' ],
			'additionalProperties': False
		}
	}
}

remove_punctuation_tool = {
	'type': 'function',
	'function': {
		'name': 'preprocess_remove_punctuation',
		'description': 'Strip punctuation from tokenized text.',
		'parameters': {
			'type': 'object',
			'properties': {
				'text': {
					'type': 'string',
					'description': 'Text value processed by the operation.'
				}
			},
			'required': [ 'text' ],
			'additionalProperties': False
		}
	}
}

normalize_text_tool = {
	'type': 'function',
	'function': {
		'name': 'preprocess_normalize_text',
		'description': 'Convert text to lowercase for stable comparison and tokenization.',
		'parameters': {
			'type': 'object',
			'properties': {
				'text': {
					'type': 'string',
					'description': 'Text value processed by the operation.'
				}
			},
			'required': [ 'text' ],
			'additionalProperties': False
		}
	}
}

remove_errors_tool = {
	'type': 'function',
	'function': {
		'name': 'preprocess_remove_errors',
		'description': 'Filter tokens against the NLTK English words corpus.',
		'parameters': {
			'type': 'object',
			'properties': {
				'text': {
					'type': 'string',
					'description': 'Text value processed by the operation.'
				}
			},
			'required': [ 'text' ],
			'additionalProperties': False
		}
	}
}

remove_fragments_tool = {
	'type': 'function',
	'function': {
		'name': 'preprocess_remove_fragments',
		'description': 'Remove very short token fragments from normalized text.',
		'parameters': {
			'type': 'object',
			'properties': {
				'text': {
					'type': 'string',
					'description': 'Text value processed by the operation.'
				}
			},
			'required': [ 'text' ],
			'additionalProperties': False
		}
	}
}

remove_symbols_tool = {
	'type': 'function',
	'function': {
		'name': 'preprocess_remove_symbols',
		'description': 'Remove configured symbol characters from normalized text.',
		'parameters': {
			'type': 'object',
			'properties': {
				'text': {
					'type': 'string',
					'description': 'Text value processed by the operation.'
				}
			},
			'required': [ 'text' ],
			'additionalProperties': False
		}
	}
}

remove_html_tool = {
	'type': 'function',
	'function': {
		'name': 'preprocess_remove_html',
		'description': 'Extract visible text from HTML markup.',
		'parameters': {
			'type': 'object',
			'properties': {
				'text': {
					'type': 'string',
					'description': 'Text value processed by the operation.'
				}
			},
			'required': [ 'text' ],
			'additionalProperties': False
		}
	}
}

remove_xml_tool = {
	'type': 'function',
	'function': {
		'name': 'preprocess_remove_xml',
		'description': 'Extract inner text from XML-like markup.',
		'parameters': {
			'type': 'object',
			'properties': {
				'text': {
					'type': 'string',
					'description': 'Text value processed by the operation.'
				}
			},
			'required': [ 'text' ],
			'additionalProperties': False
		}
	}
}

remove_markdown_tool = {
	'type': 'function',
	'function': {
		'name': 'preprocess_remove_markdown',
		'description': 'Remove common Markdown links, image syntax, and formatting markers.',
		'parameters': {
			'type': 'object',
			'properties': {
				'text': {
					'type': 'string',
					'description': 'Text value processed by the operation.'
				}
			},
			'required': [ 'text' ],
			'additionalProperties': False
		}
	}
}

remove_stopwords_tool = {
	'type': 'function',
	'function': {
		'name': 'preprocess_remove_stopwords',
		'description': 'Remove English stop words from tokenized text.',
		'parameters': {
			'type': 'object',
			'properties': {
				'text': {
					'type': 'string',
					'description': 'Text value processed by the operation.'
				}
			},
			'required': [ 'text' ],
			'additionalProperties': False
		}
	}
}

remove_encodings_tool = {
	'type': 'function',
	'function': {
		'name': 'preprocess_remove_encodings',
		'description': 'Resolve HTML entities, normalize Unicode characters, and remove control characters.',
		'parameters': {
			'type': 'object',
			'properties': {
				'text': {
					'type': 'string',
					'description': 'Text value processed by the operation.'
				}
			},
			'required': [ 'text' ],
			'additionalProperties': False
		}
	}
}

remove_headers_tool = {
	'type': 'function',
	'function': {
		'name': 'preprocess_remove_headers',
		'description': 'Detect and remove repeated page headers and footers from a text file.',
		'parameters': {
			'type': 'object',
			'properties': {
				'filepath': {
					'type': 'string',
					'description': 'Local filesystem path used by the operation.'
				},
				'lines': {
					'type': 'integer',
					'description': 'Lines value used by the operation.',
					'default': 50
				},
				'headers': {
					'type': 'integer',
					'description': 'Headers value used by the operation.',
					'default': 3
				},
				'footers': {
					'type': 'integer',
					'description': 'Footers value used by the operation.',
					'default': 3
				}
			},
			'required': [ 'filepath' ],
			'additionalProperties': False
		}
	}
}

remove_numbers_tool = {
	'type': 'function',
	'function': {
		'name': 'preprocess_remove_numbers',
		'description': 'Remove decimal digits from text.',
		'parameters': {
			'type': 'object',
			'properties': {
				'text': {
					'type': 'string',
					'description': 'Text value processed by the operation.'
				}
			},
			'required': [ 'text' ],
			'additionalProperties': False
		}
	}
}

remove_numerals_tool = {
	'type': 'function',
	'function': {
		'name': 'preprocess_remove_numerals',
		'description': 'Remove Roman-numeral patterns from text.',
		'parameters': {
			'type': 'object',
			'properties': {
				'text': {
					'type': 'string',
					'description': 'Text value processed by the operation.'
				}
			},
			'required': [ 'text' ],
			'additionalProperties': False
		}
	}
}

remove_images_tool = {
	'type': 'function',
	'function': {
		'name': 'preprocess_remove_images',
		'description': 'Remove Markdown image references, HTML image elements, and direct image URLs.',
		'parameters': {
			'type': 'object',
			'properties': {
				'text': {
					'type': 'string',
					'description': 'Text value processed by the operation.'
				}
			},
			'required': [ 'text' ],
			'additionalProperties': False
		}
	}
}

tiktokenize_tool = {
	'type': 'function',
	'function': {
		'name': 'preprocess_tiktokenize',
		'description': 'Encode text with a tiktoken tokenizer and return token identifiers as tabular data.',
		'parameters': {
			'type': 'object',
			'properties': {
				'text': {
					'type': 'string',
					'description': 'Text value processed by the operation.'
				},
				'encoding': {
					'type': 'string',
					'description': 'Text encoding or tokenizer encoding used by the operation.',
					'default': 'cl100k_base'
				}
			},
			'required': [ 'text' ],
			'additionalProperties': False
		}
	}
}

split_sentences_tool = {
	'type': 'function',
	'function': {
		'name': 'preprocess_split_sentences',
		'description': 'Split text into sentence strings using NLTK sentence tokenization.',
		'parameters': {
			'type': 'object',
			'properties': {
				'text': {
					'type': 'string',
					'description': 'Text value processed by the operation.'
				}
			},
			'required': [ 'text' ],
			'additionalProperties': False
		}
	}
}

split_pages_tool = {
	'type': 'function',
	'function': {
		'name': 'preprocess_split_pages',
		'description': 'Split a text file into page-sized text blocks.',
		'parameters': {
			'type': 'object',
			'properties': {
				'filepath': {
					'type': 'string',
					'description': 'Local filesystem path used by the operation.'
				},
				'num': {
					'type': 'integer',
					'description': 'Num value used by the operation.',
					'default': 50
				}
			},
			'required': [ 'filepath' ],
			'additionalProperties': False
		}
	}
}

split_paragraphs_tool = {
	'type': 'function',
	'function': {
		'name': 'preprocess_split_paragraphs',
		'description': 'Read a text file and return paragraph-like text blocks as tabular data.',
		'parameters': {
			'type': 'object',
			'properties': {
				'filepath': {
					'type': 'string',
					'description': 'Local filesystem path used by the operation.'
				}
			},
			'required': [ 'filepath' ],
			'additionalProperties': False
		}
	}
}

create_frequency_distribution_tool = {
	'type': 'function',
	'function': {
		'name': 'preprocess_create_frequency_distribution',
		'description': 'Build a word-frequency table from a token sequence.',
		'parameters': {
			'type': 'object',
			'properties': {
				'tokens': {
					'type': 'array',
					'items': {
						'type': 'string'
					},
					'description': 'Token values processed by the operation.'
				}
			},
			'required': [ 'tokens' ],
			'additionalProperties': False
		}
	}
}

create_vocabulary_tool = {
	'type': 'function',
	'function': {
		'name': 'preprocess_create_vocabulary',
		'description': 'Extract the vocabulary column from a token-frequency table.',
		'parameters': {
			'type': 'object',
			'properties': {
				'tokens': {
					'type': 'array',
					'items': {
						'type': 'string'
					},
					'description': 'Token values processed by the operation.'
				}
			},
			'required': [ 'tokens' ],
			'additionalProperties': False
		}
	}
}

create_wordbag_tool = {
	'type': 'function',
	'function': {
		'name': 'preprocess_create_wordbag',
		'description': 'Build a bag-of-words table from a token sequence.',
		'parameters': {
			'type': 'object',
			'properties': {
				'tokens': {
					'type': 'array',
					'items': {
						'type': 'string'
					},
					'description': 'Token values processed by the operation.'
				}
			},
			'required': [ 'tokens' ],
			'additionalProperties': False
		}
	}
}

create_vectors_tool = {
	'type': 'function',
	'function': {
		'name': 'preprocess_create_vectors',
		'description': 'Create TF-IDF vectors for token values.',
		'parameters': {
			'type': 'object',
			'properties': {
				'tokens': {
					'type': 'array',
					'items': {
						'type': 'string'
					},
					'description': 'Token values processed by the operation.'
				}
			},
			'required': [ 'tokens' ],
			'additionalProperties': False
		}
	}
}

clean_file_tool = {
	'type': 'function',
	'function': {
		'name': 'preprocess_clean_file',
		'description': 'Apply the standard Fonky text-cleaning pipeline to a single file.',
		'parameters': {
			'type': 'object',
			'properties': {
				'filepath': {
					'type': 'string',
					'description': 'Local filesystem path used by the operation.'
				}
			},
			'required': [ 'filepath' ],
			'additionalProperties': False
		}
	}
}

clean_files_tool = {
	'type': 'function',
	'function': {
		'name': 'preprocess_clean_files',
		'description': 'Apply the standard Fonky text-cleaning pipeline to every file in a directory.',
		'parameters': {
			'type': 'object',
			'properties': {
				'source': {
					'type': 'string',
					'description': 'Source value used to scope or identify the backing operation.'
				},
				'destination': {
					'type': 'string',
					'description': 'Destination used to receive generated or processed output.'
				}
			},
			'required': [ 'source', 'destination' ],
			'additionalProperties': False
		}
	}
}

chunk_files_tool = {
	'type': 'function',
	'function': {
		'name': 'preprocess_chunk_files',
		'description': 'Split text files into sentence chunks and write chunked output files.',
		'parameters': {
			'type': 'object',
			'properties': {
				'source': {
					'type': 'string',
					'description': 'Source value used to scope or identify the backing operation.'
				},
				'destination': {
					'type': 'string',
					'description': 'Destination used to receive generated or processed output.'
				}
			},
			'required': [ 'source', 'destination' ],
			'additionalProperties': False
		}
	}
}

chunk_data_tool = {
	'type': 'function',
	'function': {
		'name': 'preprocess_chunk_data',
		'description': 'Chunk a text file into fixed-size word groups represented as tabular data.',
		'parameters': {
			'type': 'object',
			'properties': {
				'filepath': {
					'type': 'string',
					'description': 'Local filesystem path used by the operation.'
				},
				'size': {
					'type': 'integer',
					'description': 'Maximum size or group size used by the operation.',
					'default': 10
				}
			},
			'required': [ 'filepath' ],
			'additionalProperties': False
		}
	}
}

chunk_datasets_tool = {
	'type': 'function',
	'function': {
		'name': 'preprocess_chunk_datasets',
		'description': 'Clean and chunk a directory of text files into spreadsheet datasets.',
		'parameters': {
			'type': 'object',
			'properties': {
				'source': {
					'type': 'string',
					'description': 'Source value used to scope or identify the backing operation.'
				},
				'destination': {
					'type': 'string',
					'description': 'Destination used to receive generated or processed output.'
				},
				'size': {
					'type': 'integer',
					'description': 'Maximum size or group size used by the operation.',
					'default': 10
				}
			},
			'required': [ 'source', 'destination' ],
			'additionalProperties': False
		}
	}
}

convert_jsonl_tool = {
	'type': 'function',
	'function': {
		'name': 'preprocess_convert_jsonl',
		'description': 'Convert text files into line-oriented JSON-like chunk output.',
		'parameters': {
			'type': 'object',
			'properties': {
				'source': {
					'type': 'string',
					'description': 'Source value used to scope or identify the backing operation.'
				},
				'destination': {
					'type': 'string',
					'description': 'Destination used to receive generated or processed output.'
				},
				'size': {
					'type': 'integer',
					'description': 'Maximum size or group size used by the operation.',
					'default': 10
				}
			},
			'required': [ 'source', 'destination' ],
			'additionalProperties': False
		}
	}
}

encode_sentences_tool = {
	'type': 'function',
	'function': {
		'name': 'preprocess_encode_sentences',
		'description': 'Generate sentence-transformer embeddings for normalized token values.',
		'parameters': {
			'type': 'object',
			'properties': {
				'tokens': {
					'type': 'array',
					'items': {
						'type': 'string'
					},
					'description': 'Token values processed by the operation.'
				},
				'model': {
					'type': 'string',
					'description': 'Model identifier used by the operation.',
					'default': 'all-MiniLM-L6-v2'
				}
			},
			'required': [ 'tokens' ],
			'additionalProperties': False
		}
	}
}

word_tokenizer_tool = {
	'type': 'function',
	'function': {
		'name': 'nltk_word_tokenizer',
		'description': 'Tokenize text into lowercased word tokens.',
		'parameters': {
			'type': 'object',
			'properties': {
				'text': {
					'type': 'string',
					'description': 'Text value processed by the operation.'
				}
			},
			'required': [ 'text' ],
			'additionalProperties': False
		}
	}
}

sentence_tokenizer_tool = {
	'type': 'function',
	'function': {
		'name': 'nltk_sentence_tokenizer',
		'description': 'Tokenize text into lowercased sentence strings.',
		'parameters': {
			'type': 'object',
			'properties': {
				'text': {
					'type': 'string',
					'description': 'Text value processed by the operation.'
				}
			},
			'required': [ 'text' ],
			'additionalProperties': False
		}
	}
}

word_stemmer_tool = {
	'type': 'function',
	'function': {
		'name': 'nltk_word_stemmer',
		'description': 'Stem lowercased word tokens with the configured Porter stemmer.',
		'parameters': {
			'type': 'object',
			'properties': {
				'text': {
					'type': 'string',
					'description': 'Text value processed by the operation.'
				}
			},
			'required': [ 'text' ],
			'additionalProperties': False
		}
	}
}

word_lemmatizer_tool = {
	'type': 'function',
	'function': {
		'name': 'nltk_word_lemmatizer',
		'description': 'Lemmatize lowercased word tokens with the configured WordNet lemmatizer.',
		'parameters': {
			'type': 'object',
			'properties': {
				'text': {
					'type': 'string',
					'description': 'Text value processed by the operation.'
				}
			},
			'required': [ 'text' ],
			'additionalProperties': False
		}
	}
}

pos_tagger_tool = {
	'type': 'function',
	'function': {
		'name': 'nltk_pos_tagger',
		'description': 'Assign part-of-speech tags to lowercased word tokens.',
		'parameters': {
			'type': 'object',
			'properties': {
				'text': {
					'type': 'string',
					'description': 'Text value processed by the operation.'
				}
			},
			'required': [ 'text' ],
			'additionalProperties': False
		}
	}
}

named_entity_recognition_tool = {
	'type': 'function',
	'function': {
		'name': 'nltk_named_entity_recognition',
		'description': 'Extract named-entity text and entity labels from tagged tokens.',
		'parameters': {
			'type': 'object',
			'properties': {
				'text': {
					'type': 'string',
					'description': 'Text value processed by the operation.'
				}
			},
			'required': [ 'text' ],
			'additionalProperties': False
		}
	}
}

chunk_words_tool = {
	'type': 'function',
	'function': {
		'name': 'nltk_chunk_words',
		'description': 'Group word tokens into fixed-size chunks and return them as tabular data.',
		'parameters': {
			'type': 'object',
			'properties': {
				'text': {
					'type': 'string',
					'description': 'Text value processed by the operation.'
				},
				'size': {
					'type': 'integer',
					'description': 'Maximum size or group size used by the operation.',
					'default': 5
				}
			},
			'required': [ 'text' ],
			'additionalProperties': False
		}
	}
}

chunk_sentences_tool = {
	'type': 'function',
	'function': {
		'name': 'nltk_chunk_sentences',
		'description': 'Group sentence tokens into fixed-size chunks and return them as tabular data.',
		'parameters': {
			'type': 'object',
			'properties': {
				'text': {
					'type': 'string',
					'description': 'Text value processed by the operation.'
				},
				'size': {
					'type': 'integer',
					'description': 'Maximum size or group size used by the operation.',
					'default': 15
				}
			},
			'required': [ 'text' ],
			'additionalProperties': False
		}
	}
}

semantic_search_tool = {
	'type': 'function',
	'function': {
		'name': 'preprocess_semantic_search',
		'description': 'Search token content by semantic similarity.',
		'parameters': {
			'type': 'object',
			'properties': {
				'query': {
					'type': 'string',
					'description': 'Search query or natural-language request submitted to the backing operation.'
				},
				'tokens': {
					'type': 'array',
					'items': {
						'type': 'string'
					},
					'description': 'Token values processed by the operation.'
				},
				'model': {
					'type': 'string',
					'description': 'Model identifier used by the operation.',
					'default': 'all-MiniLM-L6-v2'
				},
				'top': {
					'type': 'integer',
					'description': 'Top value used by the operation.',
					'default': 5
				}
			},
			'required': [ 'query', 'tokens' ],
			'additionalProperties': False
		}
	}
}

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
	'arxiv_fetch_tool',
	'google_drive_tool',
	'wikipedia_fetch_tool',
	'news_tool',
	'cse_search_tool',
	'gov_data_tool',
	'congress_tool',
	'internet_archive_tool',
	'grokipedia_tool',
	'arxiv_load_tool',
	'wikipedia_load_tool',
	'naval_observatory_tool',
	'satellite_center_tool',
	'nearby_objects_tool',
	'open_science_tool',
	'space_weather_tool',
	'astro_catalog_tool',
	'astro_query_tool',
	'star_map_tool',
	'star_chart_tool',
	'open_sky_tool',
	'google_drive_file_tool',
	'google_drive_folder_tool',
	'onedrive_tool',
	'google_cloud_file_tool',
	'aws_file_tool',
	'google_speech_to_text_tool',
	'google_bucket_tool',
	'aws_bucket_tool',
	'census_data_tool',
	'socrata_tool',
	'united_nations_tool',
	'world_population_tool',
	'open_city_tool',
	'text_tool',
	'csv_tool',
	'pdf_read_tool',
	'pdf_load_tool',
	'excel_tool',
	'word_tool',
	'markdown_tool',
	'html_tool',
	'outlook_tool',
	'spfx_tool',
	'spfx_folder_tool',
	'powerpoint_tool',
	'powerpoint_multiple_tool',
	'email_tool',
	'json_tool',
	'xml_tool',
	'xml_tree_tool',
	'jupyter_notebook_tool',
	'google_weather_current_tool',
	'google_weather_hourly_forecast_tool',
	'google_weather_daily_forecast_tool',
	'google_weather_hourly_history_tool',
	'google_weather_alerts_tool',
	'earth_observatory_tool',
	'open_weather_tool',
	'historical_weather_tool',
	'usgs_earthquakes_tool',
	'usgs_water_data_tool',
	'air_now_tool',
	'climate_data_tool',
	'eonet_tool',
	'envirofacts_tool',
	'tides_and_currents_tool',
	'uv_index_tool',
	'purple_air_tool',
	'open_aq_tool',
	'firms_tool',
	'location_tool',
	'coordinates_tool',
	'address_tool',
	'directions_tool',
	'global_imagery_wms_map_tool',
	'global_imagery_map_services_tool',
	'global_imagery_mercator_map_tool',
	'google_geocoding_tool',
	'usgs_national_map_tool',
	'usgs_sciencebase_tool',
	'health_data_tool',
	'global_health_data_tool',
	'wonder_tool',
	'pubmed_tool',
	'web_page_fetch_tool',
	'html_to_text_convert_tool',
	'web_title_tool',
	'web_links_tool',
	'web_structured_data_tool',
	'web_crawl_tool',
	'crawler_page_tool',
	'web_page_render_tool',
	'web_load_tool',
	'web_recursive_tool',
	'web_pages_tool',
	'github_tool',
	'web_page_scrape_tool',
	'html_to_text_scraper_tool',
	'paragraphs_tool',
	'lists_tool',
	'tables_tool',
	'articles_tool',
	'headings_tool',
	'divisions_tool',
	'sections_tool',
	'blockquotes_tool',
	'hyperlinks_tool',
	'images_tool',
	'image_tool',
	'load_text_tool',
	'collapse_whitespace_tool',
	'remove_punctuation_tool',
	'normalize_text_tool',
	'remove_errors_tool',
	'remove_fragments_tool',
	'remove_symbols_tool',
	'remove_html_tool',
	'remove_xml_tool',
	'remove_markdown_tool',
	'remove_stopwords_tool',
	'remove_encodings_tool',
	'remove_headers_tool',
	'remove_numbers_tool',
	'remove_numerals_tool',
	'remove_images_tool',
	'tiktokenize_tool',
	'split_sentences_tool',
	'split_pages_tool',
	'split_paragraphs_tool',
	'create_frequency_distribution_tool',
	'create_vocabulary_tool',
	'create_wordbag_tool',
	'create_vectors_tool',
	'clean_file_tool',
	'clean_files_tool',
	'chunk_files_tool',
	'chunk_data_tool',
	'chunk_datasets_tool',
	'convert_jsonl_tool',
	'encode_sentences_tool',
	'word_tokenizer_tool',
	'sentence_tokenizer_tool',
	'word_stemmer_tool',
	'word_lemmatizer_tool',
	'pos_tagger_tool',
	'named_entity_recognition_tool',
	'chunk_words_tool',
	'chunk_sentences_tool',
	'semantic_search_tool',
]


# ==========================================================================================
# LOCAL AUTOMATION TOOLS - SHARED STANDALONE FONKY IMPLEMENTATION
# ==========================================================================================

from .. import loco_tools as _loco_tools

local_calculate = _loco_tools.local_calculate
local_percentage = _loco_tools.local_percentage
local_file_list = _loco_tools.local_file_list
local_file_read = _loco_tools.local_file_read
local_file_exists = _loco_tools.local_file_exists
local_file_info = _loco_tools.local_file_info
local_file_hash = _loco_tools.local_file_hash
local_sqlite_tables = _loco_tools.local_sqlite_tables
local_sqlite_schema = _loco_tools.local_sqlite_schema
local_sqlite_query = _loco_tools.local_sqlite_query
local_excel_sheets = _loco_tools.local_excel_sheets
local_excel_cell = _loco_tools.local_excel_cell
local_excel_range = _loco_tools.local_excel_range
local_excel_write_range = _loco_tools.local_excel_write_range
local_word_read = _loco_tools.local_word_read
local_word_create = _loco_tools.local_word_create
local_word_append = _loco_tools.local_word_append
local_word_replace = _loco_tools.local_word_replace
local_slides_create = _loco_tools.local_slides_create
local_slides_add_slide = _loco_tools.local_slides_add_slide
local_slides_add_text = _loco_tools.local_slides_add_text
local_outlook_messages = _loco_tools.local_outlook_messages
local_outlook_search = _loco_tools.local_outlook_search
local_outlook_read = _loco_tools.local_outlook_read
local_outlook_draft = _loco_tools.local_outlook_draft
local_windows_list = _loco_tools.local_windows_list
local_windows_controls = _loco_tools.local_windows_controls
local_system_info = _loco_tools.local_system_info
local_disk_usage = _loco_tools.local_disk_usage

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


# Provider-native function-call definitions, executed through the local Python callables.
local_tool_definitions = [
    { 'type': 'function', 'function': { 'name': operation.__name__,
        'description': operation.__doc__.splitlines( )[ 0 ],
        'parameters': _loco_tools.local_tool_schema( operation ) } }
    for operation in (
    _loco_tools.local_calculate,
    _loco_tools.local_percentage,
    _loco_tools.local_file_list,
    _loco_tools.local_file_read,
    _loco_tools.local_file_exists,
    _loco_tools.local_file_info,
    _loco_tools.local_file_hash,
    _loco_tools.local_sqlite_tables,
    _loco_tools.local_sqlite_schema,
    _loco_tools.local_sqlite_query,
    _loco_tools.local_excel_sheets,
    _loco_tools.local_excel_cell,
    _loco_tools.local_excel_range,
    _loco_tools.local_excel_write_range,
    _loco_tools.local_word_read,
    _loco_tools.local_word_create,
    _loco_tools.local_word_append,
    _loco_tools.local_word_replace,
    _loco_tools.local_slides_create,
    _loco_tools.local_slides_add_slide,
    _loco_tools.local_slides_add_text,
    _loco_tools.local_outlook_messages,
    _loco_tools.local_outlook_search,
    _loco_tools.local_outlook_read,
    _loco_tools.local_outlook_draft,
    _loco_tools.local_windows_list,
    _loco_tools.local_windows_controls,
    _loco_tools.local_system_info,
    _loco_tools.local_disk_usage,
    _loco_tools.local_tool_schema,
    )
]
__all__.append( 'local_tool_definitions' )
