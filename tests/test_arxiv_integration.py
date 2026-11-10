"""
    ******************************************************************************************
        Assembly:                Fonky
        Filename:                test_arxiv_integration.py
    ******************************************************************************************

    Purpose:
        Test Fonky's existing ArXiv fetcher without importing unrelated optional
        providers; optionally run one live arXiv query using its real retriever.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest


class FakeError( Exception ):
    """Preserve Fonky's Error metadata contract in isolated tests."""


def throw_if( name: str, value: object ) -> None:
    """Enforce required arguments for the isolated production class."""
    if value is None or isinstance( value, str ) and not value.strip( ):
        raise ValueError( name )


def load_fetcher( retriever: Any ) -> Any:
    """Compile the unmodified ArXiv class against an injectable retriever.

    Purpose:
        Isolates the exact production class for mock and live integration testing,
        without installing every other Fonky provider dependency.

    Args:
        retriever: LangChain ArxivRetriever constructor or mocked equivalent.

    Returns:
        Any: The original ArXiv class from fetchers.py.
    """
    path = Path( __file__ ).resolve( ).parents[ 1 ] / 'fetchers.py'
    module = ast.parse( path.read_text( encoding='utf-8' ) )
    node = next( item for item in module.body
        if isinstance( item, ast.ClassDef ) and item.name == 'ArXiv' )
    tree = ast.Module( body=[
        ast.ImportFrom( module='__future__',
            names=[ ast.alias( name='annotations' ) ], level=0 ),
        node ], type_ignores=[ ] )
    ast.fix_missing_locations( tree )
    namespace: dict[ str, Any ] = {
        'Fetcher': type( 'Fetcher', (object,), { } ),
        'ArxivRetriever': retriever, 'Document': Any,
        'Error': FakeError, 'throw_if': throw_if,
    }
    exec( compile( tree, str( path ), 'exec' ), namespace )
    return namespace[ 'ArXiv' ]


def test_query_and_defaults( ) -> None:
    """Preserve the existing retrieval signature and default behavior."""
    retriever = Mock( )
    retriever.return_value.invoke.return_value = [ 'one', 'two' ]
    catalog = load_fetcher( retriever )( )
    documents = catalog.fetch( question='  quantum computing  ' )

    assert documents == [ 'one', 'two' ]
    retriever.assert_called_once_with(
        load_max_docs=5, get_full_documents=False,
        load_all_available_meta=False )
    retriever.return_value.invoke.assert_called_once_with( 'quantum computing' )


def test_caller_overrides_and_bounds( ) -> None:
    """Verify explicit retrieval overrides remain supported."""
    retriever = Mock( )
    catalog = load_fetcher( retriever )(
        max_documents=10, full_documents=False, include_metadata=False )
    catalog.fetch( 'test', max_documents=350,
        full_documents=True, include_metadata=True )

    retriever.assert_called_once_with(
        load_max_docs=300, get_full_documents=True,
        load_all_available_meta=True )


def test_blank_query_does_not_call_provider( ) -> None:
    """Avoid hitting the API with an invalid search query."""
    retriever = Mock( )
    catalog = load_fetcher( retriever )( )

    with pytest.raises( FakeError ):
        catalog.fetch( question='  ' )
    retriever.assert_not_called( )


@pytest.mark.skipif(
    os.getenv( 'FONKY_ARXIV_LIVE', '' ) != '1',
    reason='Live external integration is opt-in.' )
def test_live_arxiv_summary_retrieval( ) -> None:
    """Run one live publication lookup through the actual LangChain retriever."""
    from langchain_community.retrievers import ArxivRetriever
    catalog = load_fetcher( ArxivRetriever )( )
    documents = catalog.fetch( question='1706.03762' )

    assert documents
    assert isinstance( documents[ 0 ].page_content, str )
    assert documents[ 0 ].page_content.strip( )
    assert documents[ 0 ].metadata
