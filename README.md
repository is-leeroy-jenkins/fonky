###### fonky

![](https://github.com/is-leeroy-jenkins/Fonky/blob/main/resources/images/fonky-project.png)

<p align="left">
  <a href="#-purpose">Purpose</a> &nbsp;|&nbsp;
  <a href="#%EF%B8%8F-architecture">Architecture</a> &nbsp;|&nbsp;
  <a href="#-package-structure">Structure</a> &nbsp;|&nbsp;
  <a href="#%EF%B8%8F-installation">Installation</a> &nbsp;|&nbsp;
  <a href="#-provider-integrations">Integrations</a> &nbsp;|&nbsp;
  <a href="https://github.com/is-leeroy-jenkins/Fonky/blob/main/resources/Tools.md#-tool-index">Tool Index</a> &nbsp;|&nbsp;
  <a href="https://github.com/is-leeroy-jenkins/Fonky/blob/main/resources/User-Guide.md">User Guide</a> &nbsp;|&nbsp;
  <a href="https://github.com/is-leeroy-jenkins/Fonky/blob/main/resources/Configuration.md#configuration">Configuration</a> &nbsp;|&nbsp;
</p>

___

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-0078FC?style=for-the-badge&logo=github)](https://is-leeroy-jenkins.github.io/fonky/)


## 🎯 Purpose

Fonky is a Python library that provides a unified collection of reusable tools for AI, data acquisition, document processing, web access, geospatial analysis, environmental data, and other common application workflows. It helps solve the problem of repeatedly implementing and maintaining provider-specific integrations by encapsulating existing loaders, fetchers, scrapers, preprocessors, and related utilities behind consistent, easy-to-call interfaces. Fonky can be imported directly into Python applications, notebooks, automation pipelines, or AI-agent frameworks, allowing developers to invoke individual tools as ordinary functions or expose them through provider-specific integrations such as GPT, Claude, Gemini, Grok, Mistral, and LangChain without duplicating the underlying implementation.

## 🛠️ Architecture

![Fonky provider-native architecture](https://github.com/is-leeroy-jenkins/Fonky/blob/main/resources/images/fonky-architecture.png)

## 🔁 Workflow

![Fonky provider tool execution workflow](https://github.com/is-leeroy-jenkins/Fonky/blob/main/resources/images/fonky-workflow.png)

## 📦 Package Structure

```text
fonky/
├── __init__.py
├── boogr.py
├── loco.py                  # Standalone local Office / OS / SQLite automation
├── loco_tools.py            # 29 one-call AI tool functions over loco.py
├── config.py
├── fetchers.py
├── loaders.py
├── models.py
├── processors.py
├── scrapers.py
├── gpt/
│   ├── __init__.py
│   └── tools.py
├── claude/
│   ├── __init__.py
│   └── tools.py
├── gemini/
│   ├── __init__.py
│   └── tools.py
├── grok/
│   ├── __init__.py
│   └── tools.py
├── mistral/
│   ├── __init__.py
│   └── tools.py
└── langchain/
    ├── __init__.py
    └── tools.py
```

## ⚙️ Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel
python -m pip install fonky
python -m pip check
```

For an editable development installation from a repository checkout:

```powershell
python -m pip install -e .
```

### Playwright

```powershell
python -m playwright install chromium
```


## 🔑 Configuration and API Keys

[API Key Set-up Instructions](https://github.com/is-leeroy-jenkins/Fonky/blob/main/resources/Configuration.md)

### Environment Variables

| `config.py` constant        | Environment variable         | Service / setting                                      |
|-----------------------------|------------------------------|--------------------------------------------------------|
| `AIRNOW_API_KEY`            | `AIRNOW_API_KEY`             | AirNow                                                 |
| `CLAUDE_API_KEY`            | `CLAUDE_API_KEY`             | Anthropic Claude                                       |
| `CONGRESS_API_KEY`          | `CONGRESS_API_KEY`           | Congress.gov / congressional data                      |
| `CHROMA_API_KEY`            | `CHROMA_API_KEY`             | Chroma                                                 |
| `CHROMA_TENET_ID`           | `CHROMA_TENET_ID`            | Chroma tenant identifier                               |
| `GEOAPIFY_API_KEY`          | `GEOAPIFY_API_KEY`           | Geoapify                                               |
| `GEOCODING_API_KEY`         | `GEOCODING_API_KEY`          | Geocoding service                                      |
| `GEMINI_API_KEY`            | `GEMINI_API_KEY`             | Google Gemini                                          |
| `GOOGLE_API_KEY`            | `GOOGLE_API_KEY`             | Google APIs / Programmable Search                      |
| `GOOGLE_CSE_ID`             | `GOOGLE_CSE_ID`              | Google Programmable Search Engine ID                   |
| `GOOGLE_CLOUD_PROJECT_ID`   | `GOOGLE_CLOUD_PROJECT_ID`    | Google Cloud project                                   |
| `GOOGLE_CLOUD_LOCATION`     | `GOOGLE_CLOUD_LOCATION`      | Google Cloud location                                  |
| `GOVINFO_API_KEY`           | `GOVINFO_API_KEY`            | GovInfo                                                |
| `GOOGLE_GENAI_USE_VERTEXAI` | `GOOGLE_GENAI_USE_VERTEXAI`  | Google GenAI Vertex AI mode                            |
| `GOOGLE_WEATHER_API_KEY`    | `GOOGLE_WEATHER_API_KEY`     | Google Weather                                         |
| `GOOGLE_ACCOUNT_FILE`       | `GOOGLE_ACCOUNT_CREDENTIALS` | Google service-account credentials file                |
| `GOOGLE_DRIVE_TOKEN_PATH`   | `GOOGLE_DRIVE_TOKEN_PATH`    | Google Drive OAuth token path                          |
| `GOOGLE_DRIVE_FOLDER_ID`    | `GOOGLE_DRIVE_FOLDER_ID`     | Default Google Drive folder                            |
| `HUGGINGFACE_API_KEY`       | `HUGGINGFACE_API_KEY`        | Hugging Face                                           |
| `IPINFO_API_KEY`            | `IPINFO_API_KEY`             | IPinfo                                                 |
| `OPENAI_API_KEY`            | `OPENAI_API_KEY`             | OpenAI                                                 |
| `PINECONE_API_KEY`          | `PINECONE_API_KEY`           | Pinecone                                               |
| `LANGSMITH_API_KEY`         | `LANGSMITH_API_KEY`          | LangSmith                                              |
| `LLAMAINDEX_API_KEY`        | `LLAMAINDEX_API_KEY`         | LlamaIndex                                             |
| `LLAMACLOUD_API_KEY`        | `LLAMACLOUD_API_KEY`         | LlamaCloud                                             |
| `MISTRAL_API_KEY`           | `MISTRAL_API_KEY`            | Mistral                                                |
| `NASA_API_KEY`              | `NASA_API_KEY`               | NASA APIs                                              |
| `NASA_EARTHDATA_TOKEN`      | `NASA_EARTHDATA_TOKEN`       | NASA Earthdata                                         |
| `NEWS_API_KEY`              | `NEWSAPI_API_KEY`            | NewsAPI                                                |
| `THENEWS_API_KEY`           | `THENEWSAPI_API_KEY`         | TheNewsAPI                                             |
| `WEATHERAPI_API_KEY`        | `WEATHERAPI_API_KEY`         | WeatherAPI                                             |
| `XAI_API_KEY`               | `XAI_API_KEY`                | xAI                                                    |
| `O365_CLIENT_ID`            | `O365_CLIENT_ID`             | Microsoft 365 OAuth client ID                          |
| `O365_CLIENT_SECRET`        | `O365_CLIENT_SECRET`         | Microsoft 365 OAuth client secret                      |
| `OPENAQ_API_KEY`            | `OPENAQ_API_KEY`             | OpenAQ                                                 |
| `OPENSKY_API_CLIENT_ID`     | `OPENSKY_API_CLIENT_ID`      | OpenSky API client ID                                  |
| `OPENSKY_API_CREDENTIALS`   | `OPENSKY_API_CREDENTIALS`    | OpenSky API credentials                                |
| `OPENSKY_API_CLIENT_SECRET` | `OPENSKY_API_CLIENT_ID`      | OpenSky API client secret binding in current config.py |
| `CENSUS_API_KEY`            | `CENSUS_API_KEY`             | U.S. Census                                            |
| `SOCRATA_API_KEY`           | `SOCRATA_API_KEY`            | Socrata                                                |
| `HEALTHDATA_API_KEY`        | `HEALTHDATA_API_KEY`         | HealthData.gov                                         |
| `USGS_WATERDATA_API_KEY`    | `USGS_API_KEY`               | USGS                                                   |
| `DATA_GOV_API_KEY`          | `DATAGOV_API_KEY`            | Data.gov                                               |
| `FIRMS_MAP_KEY`             | `FIRMS_MAP_KEY`              | NASA FIRMS                                             |
| `PURPLEAIR_API_KEY`         | `PURPLEAIR_API_KEY`          | PurpleAir                                              |
| `SKY_MAP_TOKEN`             | `SKY_MAP_TOKEN`              | Sky Map                                                |


## 🤖 Provider Integrations

| Provider | Fonky module | Native tool contract | Tool-result boundary |
|---|---|---|---|
| OpenAI Agents SDK | `fonky.gpt.tools` | `@function_tool` objects | OpenAI Agents runtime |
| Anthropic Claude | `fonky.claude.tools` | `@beta_tool` objects | Local execution; return a string or supported Anthropic content block |
| Google ADK | `fonky.gemini.tools` | Plain typed callables | Google ADK runtime |
| xAI Grok | `fonky.grok.tools` | Callable plus explicit `*_tool` schema | Local execution and xAI tool-result submission |
| Mistral AI | `fonky.mistral.tools` | Callable plus JSON `*_tool` declaration | Local execution; serialize content with the matching `tool_call_id` |
| LangChain | `fonky.langchain.tools` | `@tool(parse_docstring=True)` objects | LangChain runtime |


### OpenAI Agents SDK

```python
from agents import Agent, Runner

from fonky.gpt.tools import fetch_arxiv
from fonky.gpt.tools import fetch_wikipedia

agent = Agent(
    name='Research Assistant',
    instructions='Use the supplied Fonky tools when required.',
    tools=[
        fetch_arxiv,
        fetch_wikipedia,
    ] )

result = Runner.run_sync(
    agent,
    'Research retrieval augmented generation.' )

print( result.final_output )
```

### Anthropic Claude

Fonky exposes a direct Anthropic integration through `fonky.claude.tools`. Each public Claude tool is decorated with Anthropic's `@beta_tool` and delegates directly to the canonical Fonky implementation in `fetchers.py`, `loaders.py`, `scrapers.py`, or `processors.py`.

```python
from anthropic import Anthropic

from fonky.claude.tools import fetch_arxiv
from fonky.claude.tools import fetch_wikipedia

client = Anthropic()

tools = [
    fetch_arxiv.to_dict(),
    fetch_wikipedia.to_dict(),
]

response = client.beta.messages.create(
    model='claude-sonnet-4-6',
    max_tokens=4096,
    tools=tools,
    messages=[
        {
            'role': 'user',
            'content': 'Research retrieval augmented generation.',
        },
    ] )

print( response )
```

The Claude adapter does not depend on `fonky.gpt` or unwrap another provider's tools. It exposes the same Fonky operations as native Anthropic `beta_tool` objects while preserving the underlying implementation signatures, defaults, documentation, and behavior.

> **Structured tool results:** Anthropic's automatic Tool Runner expects tool results to be strings or supported Anthropic content blocks. Fonky tools that return dictionaries, DataFrames, NumPy arrays, document collections, or other structured Python values retain those native return types. Applications using those tools in an Anthropic tool-result loop should serialize the returned value before sending it back to Claude.

### Google ADK

```python
from google.adk.agents import Agent

from fonky.gemini.tools import fetch_arxiv
from fonky.gemini.tools import fetch_wikipedia

agent = Agent(
    name='research_assistant',
    model='gemini-3.7-flash',
    instruction='Use the supplied Fonky tools when required.',
    tools=[
        fetch_arxiv,
        fetch_wikipedia,
    ] )
```

### xAI Grok

```python
from fonky.grok.tools import cse_search_tool
from fonky.grok.tools import fetch_cse_search

tools = [
    cse_search_tool,
]

# Pass ``tools`` to the xAI chat request.
# When Grok requests ``fetch_cse_search``, execute the callable locally:
result = fetch_cse_search(
    keywords='federal appropriations law',
    results=5 )
```

### Mistral AI

Fonky exposes executable wrappers and Mistral-compatible JSON function declarations through
`fonky.mistral.tools`. Each declaration is paired with a callable that delegates directly to the
canonical Fonky implementation.

```python
from mistralai.client import Mistral

from fonky.config import MISTRAL_API_KEY
from fonky.mistral.tools import cse_search_tool
from fonky.mistral.tools import fetch_cse_search

client = Mistral(
    api_key=MISTRAL_API_KEY )

tools = [
    cse_search_tool,
]

response = client.chat.complete(
    model='mistral-medium-latest',
    messages=[
        {
            'role': 'user',
            'content': 'Find sources about federal appropriations law.',
        },
    ],
    tools=tools )

result = fetch_cse_search(
    keywords='federal appropriations law',
    results=5 )

print( response )
print( result )
```

When Mistral returns a tool call, the application executes the matching Fonky callable locally and
returns a serialized tool-result message. Fonky preserves canonical return types, so dictionaries,
DataFrames, NumPy arrays, and document collections must be serialized by the calling workflow.

### LangChain

```python
from fonky.langchain.tools import fetch_arxiv
from fonky.langchain.tools import fetch_wikipedia

tools = [
    fetch_arxiv,
    fetch_wikipedia,
]
```


## 📚 Documentation

- [Tools Reference](https://github.com/is-leeroy-jenkins/fonky/blob/main/resources/Tools.md)
- [User Guide](https://github.com/is-leeroy-jenkins/fonky/blob/main/resources/User-Guide.md)
- [MkDocs Site](https://is-leeroy-jenkins.github.io/fonky/)

## 📝 License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/is-leeroy-jenkins/Fonky/blob/main/LICENSE.txt)


## 🖥️ Local Automation (No Cloud API Key)

Fonky includes nine local classes through `fonky.loco`: `Outlook`, `Word`, `PowerPoint`, `Excel`, `Calculator`, `File`, `Data`, `Windows`, and `Shell`. They provide file-backed Office operations, SQLite queries, arithmetic, file management, Windows automation, and allowlisted shell commands. Live Microsoft Office automation requires Windows, classic Outlook for mail, and installed Office applications.

```python
from fonky.loco import Calculator, Excel

calculator = Calculator()
print(calculator.evaluate_expression('(12 + 8) / 4'))

excel = Excel(backend='file')
excel.create_workbook()
excel.write_cell('Sheet', 'A1', 'Budget')
excel.save_workbook('budget.xlsx')
excel.close_workbook()
```

Windows extras: `python -m pip install 'fonky[local]'` (available after publication). [Local automation guide](docs/user-guide/Local-Automation.md) describes the backends, permissions, testing, and limitations.


### Local AI Tools: GPT, Claude, Gemini, Grok, Mistral, and LangChain

Fonky now exposes **29 approved local operations** without modifying the standalone
`loco.py` automation classes. Each provider's existing `tools.py` module exports
matching names such as `local_calculate`, `local_file_read`,
`local_excel_cell`, and `local_sqlite_query`. Shared implementation and host
permissions live in `fonky.loco_tools`, not in an additional agent framework.

Configure a trusted local workspace before file-backed tool calls:

```powershell
$env:FONKY_LOCAL_ROOT = "C:\\Work\\Fonky"
```

File-backed Office document writes are disabled until the host sets
`FONKY_LOCAL_WRITE_ENABLED=1`. Local Outlook and Windows UI inspection are disabled
until the host sets `FONKY_LOCAL_OUTLOOK_ENABLED=1` or
`FONKY_LOCAL_DESKTOP_ENABLED=1`, respectively. Models cannot set these flags
through the exposed tool parameters. Local file and database paths must remain
within the configured workspace.

```python
from agents import Agent
from fonky.gpt.tools import local_calculate, local_file_read

agent = Agent(
    name='Fonky Local',
    instructions='Use the permitted local tools.',
    tools=[local_calculate, local_file_read],
)
```

For Anthropic, Google ADK, and LangChain, import the same tool names from their
respective `fonky.<provider>.tools` modules. Grok and Mistral additionally export
`local_tool_definitions` for native JSON-schema function calling, with matching
local Python callables for execution.

See the [Local Automation Guide](docs/user-guide/Local-Automation.md) and
[Local AI Tool API Reference](docs/api/loco-tools.md) for the complete tool list,
limits, permissions, and examples.
