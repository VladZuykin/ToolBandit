"""Agent-side abstractions for estimating and invoking real tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import math
import time
from typing import Any, Callable, Optional
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import numpy as np


@dataclass(frozen=True)
class RuntimeEstimate:
    estimated_cost: float
    estimated_latency: float


@dataclass(frozen=True)
class ToolExecution:
    output: Any
    observed_cost: float
    observed_latency: float
    status_code: Optional[int] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None


class ToolAdapter(ABC):
    """Contract implemented by every real tool integration."""

    tool_name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]

    @abstractmethod
    def estimate(self, arguments: dict[str, Any]) -> RuntimeEstimate:
        """Estimate resources before the call."""

    @abstractmethod
    def invoke(self, arguments: dict[str, Any]) -> ToolExecution:
        """Execute the tool and return actual measured resources."""

    @abstractmethod
    def catalog_entry(self) -> dict[str, Any]:
        """Return semantic metadata and real pricing configuration."""


class LatencyTracker:
    """Online mean/p95 latency estimator with a hard timeout cold start."""

    def __init__(self, timeout: float, *, max_samples: int = 500) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.timeout = float(timeout)
        self.max_samples = max_samples
        self.samples: list[float] = []

    def observe(self, latency: float) -> None:
        self.samples.append(float(latency))
        if len(self.samples) > self.max_samples:
            del self.samples[: len(self.samples) - self.max_samples]

    def estimate(self) -> float:
        if not self.samples:
            # The enforceable timeout is used until a real observation exists.
            return self.timeout
        return float(np.mean(self.samples))


JsonTransport = Callable[[str, float, str], tuple[int, Any]]


class OpenMeteoWeatherAdapter(ToolAdapter):
    """Real current-weather adapter using Open-Meteo's public API."""

    tool_name = "open_meteo_current_weather"
    description = (
        "Get current weather for geographic latitude and longitude, including "
        "temperature, apparent temperature, precipitation, wind, and weather code."
    )
    input_schema = {
        "type": "object",
        "required": ["latitude", "longitude"],
        "properties": {
            "latitude": {"type": "number", "minimum": -90, "maximum": 90},
            "longitude": {"type": "number", "minimum": -180, "maximum": 180},
        },
    }
    output_schema = {
        "type": "object",
        "required": ["latitude", "longitude", "current", "current_units"],
    }
    endpoint = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, *, timeout: float = 3.0,
                 transport: Optional[JsonTransport] = None) -> None:
        self.timeout = float(timeout)
        self.latency = LatencyTracker(timeout)
        self.transport = transport or self._http_get

    @staticmethod
    def _number(arguments: dict[str, Any], name: str, low: float, high: float) -> float:
        value = float(arguments[name])
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{name} must be in [{low}, {high}]")
        return value

    def _url(self, arguments: dict[str, Any]) -> str:
        latitude = self._number(arguments, "latitude", -90, 90)
        longitude = self._number(arguments, "longitude", -180, 180)
        query = urlencode({
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "temperature_2m,apparent_temperature,precipitation,"
                "weather_code,wind_speed_10m"
            ),
            "timezone": "auto",
        })
        return f"{self.endpoint}?{query}"

    @staticmethod
    def _http_get(url: str, timeout: float, user_agent: str) -> tuple[int, Any]:
        request = Request(url, headers={"User-Agent": user_agent}, method="GET")
        with urlopen(request, timeout=timeout) as response:
            return int(response.status), json.loads(response.read().decode("utf-8"))

    def estimate(self, arguments: dict[str, Any]) -> RuntimeEstimate:
        self._url(arguments)  # validate before ranking
        mean = self.latency.estimate()
        return RuntimeEstimate(
            estimated_cost=0.0,  # Open-Meteo public endpoint has no per-call fee.
            estimated_latency=mean,
        )

    def invoke(self, arguments: dict[str, Any]) -> ToolExecution:
        url = self._url(arguments)
        started = time.perf_counter()
        try:
            status, output = self.transport(url, self.timeout, "ToolBandit/1.0")
            latency = time.perf_counter() - started
            self.latency.observe(latency)
            return ToolExecution(
                output=output,
                observed_cost=0.0,
                observed_latency=latency,
                status_code=status,
            )
        except Exception as error:
            latency = time.perf_counter() - started
            self.latency.observe(latency)
            return ToolExecution(
                output=None,
                observed_cost=0.0,
                observed_latency=latency,
                error_code=type(error).__name__,
                error_message=str(error),
            )

    def catalog_entry(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "cost": 0.0,
        }


class PublicJsonApiAdapter(ToolAdapter):
    """Small production-shaped adapter for a free, read-only JSON endpoint."""

    def __init__(
        self, *, tool_name: str, description: str,
        input_schema: dict[str, Any], output_schema: dict[str, Any],
        url_builder: Callable[[dict[str, Any]], str], timeout: float = 8.0,
        transport: Optional[JsonTransport] = None,
        output_transform: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        self.tool_name = tool_name
        self.description = description
        self.input_schema = input_schema
        self.output_schema = output_schema
        self._url_builder = url_builder
        self.timeout = float(timeout)
        self.latency = LatencyTracker(timeout)
        self.transport = transport or OpenMeteoWeatherAdapter._http_get
        self.output_transform = output_transform

    def estimate(self, arguments: dict[str, Any]) -> RuntimeEstimate:
        self._url_builder(arguments)
        return RuntimeEstimate(0.0, self.latency.estimate())

    def invoke(self, arguments: dict[str, Any]) -> ToolExecution:
        url = self._url_builder(arguments)
        started = time.perf_counter()
        try:
            status, output = self.transport(url, self.timeout, "ToolBandit/1.0 (API showcase)")
            if self.output_transform is not None:
                output = self.output_transform(output)
            elapsed = time.perf_counter() - started
            self.latency.observe(elapsed)
            return ToolExecution(output, 0.0, elapsed, status_code=status)
        except Exception as error:
            elapsed = time.perf_counter() - started
            self.latency.observe(elapsed)
            return ToolExecution(
                None, 0.0, elapsed, error_code=type(error).__name__,
                error_message=str(error),
            )

    def catalog_entry(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "cost": 0.0,
            "latency": self.latency.estimate() if self.latency.samples else None,
        }


@dataclass(frozen=True)
class PublicApiExample:
    adapter: ToolAdapter
    query: str
    arguments: dict[str, Any]
    intent: str
    expected_contract: str


def public_api_examples(*, timeout: float = 8.0) -> list[PublicApiExample]:
    """Ten real read-only APIs used by the manual end-to-end showcase."""

    obj = {"type": "object"}
    text_arg = lambda name: {"type": "object", "required": [name],
                             "properties": {name: {"type": "string"}}}
    def adapter(name, description, schema, builder, output_transform=None):
        return PublicJsonApiAdapter(
            tool_name=name, description=description, input_schema=schema,
            output_schema=obj, url_builder=builder, timeout=timeout,
            output_transform=output_transform,
        )

    def pokemon_summary(data):
        """Keep the semantic result; omit the enormous sprites/moves payload."""
        return {
            "id": data.get("id"),
            "name": data.get("name"),
            "height": data.get("height"),
            "weight": data.get("weight"),
            "types": [item.get("type", {}).get("name") for item in data.get("types", [])],
            "stats": {
                item.get("stat", {}).get("name"): item.get("base_stat")
                for item in data.get("stats", [])
            },
            "abilities": [
                item.get("ability", {}).get("name") for item in data.get("abilities", [])
            ],
        }

    weather = OpenMeteoWeatherAdapter(timeout=timeout)
    return [
        PublicApiExample(weather, "What is the current weather in Moscow?",
                         {"latitude": 55.7558, "longitude": 37.6173},
                         "Get current weather at Moscow coordinates",
                         "Return current weather with a numeric temperature and observation time."),
        PublicApiExample(adapter(
            "open_meteo_geocoding", "Find geographic coordinates and country for a city name.",
            text_arg("name"), lambda a: "https://geocoding-api.open-meteo.com/v1/search?" +
            urlencode({"name": a["name"], "count": 3, "language": "en", "format": "json"})),
            "Find coordinates of Berlin", {"name": "Berlin"}, "Geocode Berlin",
            "Return at least one Berlin result containing latitude and longitude."),
        PublicApiExample(adapter(
            "open_library_book_search", "Search books, authors and publication metadata by title.",
            text_arg("title"), lambda a: "https://openlibrary.org/search.json?" +
            urlencode({"title": a["title"], "fields": "key,title,author_name,first_publish_year", "limit": 3})),
            "Find the book The Hobbit and its author", {"title": "The Hobbit"}, "Search for The Hobbit",
            "Return a book titled The Hobbit and an author name."),
        PublicApiExample(adapter(
            "timeapi_timezone", "Get current local date and time for an IANA timezone.",
            text_arg("timezone"), lambda a: "https://timeapi.io/api/time/current/zone?" +
            urlencode({"timeZone": a["timezone"]})),
            "What time is it now in Tokyo?", {"timezone": "Asia/Tokyo"}, "Get current time in Tokyo",
            "Return the Asia/Tokyo timezone and current local datetime."),
        PublicApiExample(adapter(
            "frankfurter_exchange_rate", "Get current foreign exchange rate between two currency codes.",
            {"type": "object", "required": ["from", "to"]},
            lambda a: "https://api.frankfurter.app/latest?" + urlencode({"from": a["from"], "to": a["to"]})),
            "What is the latest USD to EUR exchange rate?", {"from": "USD", "to": "EUR"},
            "Get latest USD to EUR rate", "Return a numeric EUR exchange rate with USD as base."),
        PublicApiExample(adapter(
            "pokeapi_pokemon", "Look up Pokemon species statistics, types, abilities and measurements by name.",
            text_arg("name"), lambda a: "https://pokeapi.co/api/v2/pokemon/" + quote(a["name"].lower(), safe=""),
            pokemon_summary),
            "Give me Pikachu's Pokemon stats and types", {"name": "pikachu"}, "Look up Pikachu",
            "Return Pikachu with numeric stats and at least one Pokemon type."),
        PublicApiExample(adapter(
            "github_repository", "Get public GitHub repository metadata such as stars, language and description.",
            {"type": "object", "required": ["owner", "repo"]},
            lambda a: "https://api.github.com/repos/{}/{}".format(quote(a["owner"], safe=""), quote(a["repo"], safe=""))),
            "Show public metadata for the python/cpython GitHub repository", {"owner": "python", "repo": "cpython"},
            "Get python/cpython repository metadata", "Return full_name python/cpython plus numeric stars and primary language."),
        PublicApiExample(adapter(
            "hacker_news_item", "Get a Hacker News story, comment or poll by numeric item ID.",
            {"type": "object", "required": ["item_id"]},
            lambda a: "https://hacker-news.firebaseio.com/v0/item/{}.json".format(int(a["item_id"]))),
            "Fetch Hacker News item 8863", {"item_id": 8863}, "Get Hacker News item 8863",
            "Return Hacker News item id 8863 with its type and author."),
        PublicApiExample(adapter(
            "nominatim_place_search", "Search OpenStreetMap places and return coordinates and display name.",
            text_arg("query"), lambda a: "https://nominatim.openstreetmap.org/search?" +
            urlencode({"q": a["query"], "format": "jsonv2", "limit": 2})),
            "Find Red Square in Moscow on OpenStreetMap", {"query": "Red Square, Moscow"},
            "Geocode Red Square in Moscow", "Return a Red Square result with latitude, longitude and display name."),
        PublicApiExample(adapter(
            "ipify_public_ip", "Return the caller's public internet IP address.",
            {"type": "object", "properties": {}}, lambda a: "https://api.ipify.org?format=json"),
            "What is this machine's public IP address?", {}, "Get public IP address",
            "Return a syntactically valid IPv4 or IPv6 address in the ip field."),
    ]
