"""A bounded current-weather capability backed by Open-Meteo.

Only the fixed Open-Meteo endpoints below are reachable.  User input is
validated as a location query and is never interpreted as a URL or command.
"""
from __future__ import annotations

import re
from datetime import UTC, datetime

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.brain.actions import ActionResult, ActionStatus, CapabilityResult
from app.brain.conversation_capabilities import (
    CapabilityDescriptor,
    ConversationAction,
)
from app.brain.schemas import TaskAssessment, TaskType

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_TIMEOUT_SECONDS = 8.0
_LOCATION_PATTERN = re.compile(
    r"(?:weather|temperature|forecast)(?:\s+(?:like|is|be))?\s+(?:in|for|at)\s+(.+?)(?:[?.!]|$)",
    re.IGNORECASE,
)
_LOCATION_SUFFIX = re.compile(r"\s+\b(?:right\s+now|today|please)\b\s*$", re.IGNORECASE)
_US_STATE_NAMES = frozenset({
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut",
    "delaware", "florida", "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa",
    "kansas", "kentucky", "louisiana", "maine", "maryland", "massachusetts", "michigan",
    "minnesota", "mississippi", "missouri", "montana", "nebraska", "nevada", "new hampshire",
    "new jersey", "new mexico", "new york", "north carolina", "north dakota", "ohio",
    "oklahoma", "oregon", "pennsylvania", "rhode island", "south carolina", "south dakota",
    "tennessee", "texas", "utah", "vermont", "virginia", "washington", "west virginia",
    "wisconsin", "wyoming",
})


class WeatherParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    location: str = Field(min_length=2, max_length=160)


class WeatherLocation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1, max_length=120)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    country: str | None = Field(default=None, max_length=120)
    admin1: str | None = Field(default=None, max_length=120)


class GeocodingPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    results: list[WeatherLocation] = Field(default_factory=list, max_length=10)


class WeatherCurrent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    time: str = Field(min_length=1, max_length=64)
    temperature_2m: float
    apparent_temperature: float
    precipitation: float = Field(ge=0)
    weather_code: int = Field(ge=0, le=99)
    wind_speed_10m: float = Field(ge=0)


class ForecastPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    timezone: str = Field(min_length=1, max_length=100)
    current: WeatherCurrent


class WeatherReport(BaseModel):
    """Safe normalized output used for conversation grounding."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    location: str = Field(min_length=1, max_length=300)
    temperature_f: float
    apparent_temperature_f: float
    precipitation_inches: float = Field(ge=0)
    condition: str = Field(min_length=1, max_length=80)
    wind_mph: float = Field(ge=0)
    observed_at: str = Field(min_length=1, max_length=64)
    timezone: str = Field(min_length=1, max_length=100)
    provider: str = Field(default="Open-Meteo", min_length=1, max_length=80)
    retrieved_at: str = Field(min_length=1, max_length=64)


class WeatherCapabilityError(RuntimeError):
    """A safe internal signal that becomes a typed action failure."""


class WeatherCapability:
    name = "weather"
    # This stable public operation name is part of the registered capability
    # contract.  The provider may change; the cognitive action identifier may
    # not silently drift with it.
    operations = ("current_weather",)
    descriptor = CapabilityDescriptor(
        name=name,
        operations=operations,
        supported_task_types=(TaskType.RETRIEVAL,),
    )

    def __init__(self, *, transport: httpx.AsyncBaseTransport | None = None, timeout: float = WEATHER_TIMEOUT_SECONDS) -> None:
        if timeout <= 0 or timeout > 30:
            raise ValueError("Weather timeout must be greater than zero and no more than 30 seconds")
        self._transport = transport
        self._timeout = timeout

    def supports(self, operation: str) -> bool:
        return operation in self.operations

    def parameter_model(self, operation: str) -> type[BaseModel] | None:
        return WeatherParameters if self.supports(operation) else None

    def build_conversation_action(
        self,
        *,
        user_input: str,
        assessment: TaskAssessment,
    ) -> ConversationAction | None:
        if assessment.task_type not in self.descriptor.supported_task_types or not assessment.requires_tools:
            return None
        match = _LOCATION_PATTERN.search(user_input.strip())
        if match is None:
            return None
        location = _normalize_location_query(_LOCATION_SUFFIX.sub("", match.group(1)).strip(" ,"))
        try:
            parameters = WeatherParameters(location=location)
        except ValidationError:
            return None
        return ConversationAction(operation="current_weather", parameters=parameters.model_dump())

    async def execute(self, operation: str, parameters: BaseModel) -> CapabilityResult:
        if operation != "current_weather" or not isinstance(parameters, WeatherParameters):
            raise WeatherCapabilityError("Unsupported weather operation")
        timeout = httpx.Timeout(self._timeout)
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=False,
                transport=self._transport,
            ) as client:
                geocoding = await client.get(
                    GEOCODING_URL,
                    params={"name": parameters.location, "count": 1, "language": "en", "format": "json"},
                )
                geocoding.raise_for_status()
                locations = GeocodingPayload.model_validate(geocoding.json()).results
                if not locations:
                    raise WeatherCapabilityError("No matching location")
                location = locations[0]
                forecast = await client.get(
                    FORECAST_URL,
                    params={
                        "latitude": location.latitude,
                        "longitude": location.longitude,
                        "current": "temperature_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m",
                        "temperature_unit": "fahrenheit",
                        "wind_speed_unit": "mph",
                        "precipitation_unit": "inch",
                        "timezone": "auto",
                    },
                )
                forecast.raise_for_status()
        except httpx.TimeoutException as exc:
            # ActionExecutor maps a native timeout to the stable timed_out
            # action status instead of hiding it as an undifferentiated failure.
            raise TimeoutError("Weather provider timed out") from exc
        payload = ForecastPayload.model_validate(forecast.json())
        place = ", ".join(part for part in (location.name, location.admin1, location.country) if part)
        report = WeatherReport(
            location=place,
            temperature_f=payload.current.temperature_2m,
            apparent_temperature_f=payload.current.apparent_temperature,
            precipitation_inches=payload.current.precipitation,
            condition=_condition_name(payload.current.weather_code),
            wind_mph=payload.current.wind_speed_10m,
            observed_at=payload.current.time,
            timezone=payload.timezone,
            retrieved_at=datetime.now(UTC).isoformat(),
        )
        return CapabilityResult(output=report.model_dump())

    def format_conversation_result(self, result: ActionResult) -> str | None:
        if result.status is not ActionStatus.COMPLETED:
            return None
        try:
            report = WeatherReport.model_validate(result.output)
        except ValidationError:
            return None
        return (
            f"It’s currently {report.temperature_f:.0f}°F and {report.condition.lower()} in {report.location}. "
            f"It feels like {report.apparent_temperature_f:.0f}°F, with winds around {report.wind_mph:.0f} mph."
        )


def _condition_name(code: int) -> str:
    conditions: dict[int, str] = {
        0: "Clear sky", 1: "Mostly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Foggy", 48: "Icy fog", 51: "Light drizzle", 53: "Drizzle",
        55: "Heavy drizzle", 61: "Light rain", 63: "Rain", 65: "Heavy rain",
        71: "Light snow", 73: "Snow", 75: "Heavy snow", 80: "Rain showers",
        81: "Rain showers", 82: "Heavy rain showers", 95: "Thunderstorm",
        96: "Thunderstorm with hail", 99: "Thunderstorm with hail",
    }
    return conditions.get(code, "Unspecified conditions")


def _normalize_location_query(location: str) -> str:
    """Preserve an explicit place while making a spoken US state geocodable."""

    if "," in location:
        return location
    city, separator, state = location.rpartition(" ")
    if separator and city and state.lower() in _US_STATE_NAMES:
        return f"{city}, {state}"
    return location
