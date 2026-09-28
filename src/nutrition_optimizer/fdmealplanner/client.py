"""Public, anonymous FDMealPlanner guest client.

Only the routes used by the public MealPlanner web application are included.
The bearer token is held in memory for the lifetime of a client instance and
is never written to disk or included in exception messages.
"""

from __future__ import annotations

import base64
import calendar
import json
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from typing import Any
from urllib.parse import quote, urlsplit

from curl_cffi import requests

from ..meal_identity import MEAL_PERIOD_IDS as DINING_PERIOD_IDS
from .models import FDApplicationConfiguration, FDMealsPayload

DEFAULT_FRONTEND_ORIGIN = "https://www.fdmealplanner.com"
DEFAULT_APPLICATION_DATA_URL = (
    "https://applicationdata.fdmealplanner.com/api/v1/"
    "application-data-webapi/initial-application-data"
)
DEFAULT_ANONYMOUS_TOKEN_ORIGIN = "https://users.fdmealplanner.com"
DEFAULT_LOCATION_SEARCH_URL = (
    "https://locations.fdmealplanner.com/api/v2/"
    "location-data-webapi/search-location"
)
DEFAULT_TENANT_ID = 7
DEFAULT_ACCOUNT_ID = 10033
DEFAULT_LOCATION_ID = 10112
# These are public guest-application values shipped to every MealPlanner web
# visitor.  They are not a staff credential.  The live tenant API hostname is
# deliberately not listed here: it is discovered from API_CONFIGURATION.
DEFAULT_PUBLIC_CLIENT_KEY = (
    "D4qSnj2SJXF2EEWw6tcxKG8oTvhtZ72moLq93YSARSvbUbBBgbDQ2DPngDFM3lh5"
)
DEFAULT_PUBLIC_CLIENT_SECRET_KEY = (
    "FuzuaYdy4XUJh9UPW9Wh9dQ9mMuAHDfDfQeWfBJPJbmqR3MLpIkjiLY2IhnfAqLl"
)
DEFAULT_DEVICE_TYPE = "MealPlanner-Web"


class FDMealPlannerError(RuntimeError):
    """Base error for public FDMealPlanner access and parsing."""


class FDMealPlannerConfigurationError(FDMealPlannerError):
    """The public bootstrap did not contain usable routing configuration."""


class FDMealPlannerProtocolError(FDMealPlannerError):
    """A public response had an unexpected shape or could not be decoded."""


class FDMealPlannerHTTPError(FDMealPlannerError):
    """A public request returned a non-success HTTP status."""

    def __init__(self, status_code: int, url: str) -> None:
        super().__init__(f"FDMealPlanner request failed with HTTP {status_code}: {url}")
        self.status_code = status_code
        self.url = url


class FDMealPlannerDependencyError(FDMealPlannerError):
    """A runtime dependency required for the public token bootstrap is absent."""


def _is_https_url(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    parsed = urlsplit(value)
    return parsed.scheme.lower() == "https" and bool(parsed.netloc)


def _tenant_url(value: Any, tenant_id: int, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FDMealPlannerConfigurationError(f"missing {field_name} in API_CONFIGURATION")
    url = value.strip().replace("{tenantId}", str(tenant_id))
    if not _is_https_url(url):
        raise FDMealPlannerConfigurationError(f"{field_name} must be an HTTPS URL")
    return url


def _configuration_entries(value: Any) -> list[Mapping[str, Any]]:
    """Normalize the currently shipped list-shaped API_CONFIGURATION value."""

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError) as exc:
            raise FDMealPlannerConfigurationError("API_CONFIGURATION is not valid JSON") from exc

    if isinstance(value, Mapping):
        if "tenantId" in value:
            return [value]
        for key in ("tenants", "tenantConfigurations", "data", "result"):
            nested = value.get(key)
            if isinstance(nested, (list, Mapping, str)):
                entries = _configuration_entries(nested)
                if entries:
                    return entries
        return []

    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [entry for entry in value if isinstance(entry, Mapping)]
    return []


def parse_initial_application_data(
    payload: Mapping[str, Any],
    *,
    tenant_id: int = DEFAULT_TENANT_ID,
    location_search_url: str = DEFAULT_LOCATION_SEARCH_URL,
) -> FDApplicationConfiguration:
    """Extract tenant routing from public ``initial-application-data`` JSON."""

    if not isinstance(payload, Mapping):
        raise FDMealPlannerConfigurationError("initial application data must be an object")
    data = payload.get("data")
    if not isinstance(data, Mapping):
        raise FDMealPlannerConfigurationError("initial application data has no data object")
    configuration = data.get("configuration")
    if not isinstance(configuration, Sequence) or isinstance(configuration, (str, bytes)):
        raise FDMealPlannerConfigurationError("initial application data has no configuration list")

    settings: dict[str, Any] = {}
    for entry in configuration:
        if not isinstance(entry, Mapping):
            continue
        key = entry.get("key")
        if isinstance(key, str):
            settings[key.upper()] = entry.get("value")

    security_prefix = settings.get("APP_SEC_PREFIX")
    if not isinstance(security_prefix, str) or not security_prefix.strip():
        raise FDMealPlannerConfigurationError("missing APP_SEC_PREFIX")

    entries = _configuration_entries(settings.get("API_CONFIGURATION"))
    tenant_entry = next(
        (
            entry
            for entry in entries
            if str(entry.get("tenantId", "")).strip() == str(tenant_id)
        ),
        None,
    )
    if tenant_entry is None:
        raise FDMealPlannerConfigurationError(f"tenant {tenant_id} is absent from API_CONFIGURATION")
    urls = tenant_entry.get("URL") or tenant_entry.get("url")
    if not isinstance(urls, Mapping):
        raise FDMealPlannerConfigurationError(f"tenant {tenant_id} has no URL configuration")

    menu_url = _tenant_url(urls.get("MENU_PLANNER_DATA_URL"), tenant_id, "MENU_PLANNER_DATA_URL")
    meal_period_url = _tenant_url(
        urls.get("MEAL_PERIOD_API_URL"), tenant_id, "MEAL_PERIOD_API_URL"
    )
    configured_location_url = (
        settings.get("LOCATION_SEARCH_URL")
        or settings.get("SEARCH_LOCATION_URL")
        or location_search_url
    )
    if not _is_https_url(configured_location_url):
        raise FDMealPlannerConfigurationError("location search URL must be an HTTPS URL")
    return FDApplicationConfiguration(
        tenant_id=int(tenant_id),
        security_prefix=security_prefix.strip(),
        menu_url=menu_url,
        meal_period_url=meal_period_url,
        location_search_url=configured_location_url,
    )


def _encrypt_guest_message(security_prefix: str, message: str) -> str:
    """Encrypt public guest bootstrap material with the app's RSA public key."""

    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import padding
    except ImportError as exc:  # pragma: no cover - exercised in dependency failures
        raise FDMealPlannerDependencyError(
            "cryptography is required for the FDMealPlanner guest bootstrap"
        ) from exc

    try:
        encoded_key = security_prefix.encode("ascii")
        try:
            key_bytes = base64.b64decode(encoded_key, validate=True)
        except ValueError:
            key_bytes = encoded_key
        if key_bytes.startswith(b"-----BEGIN"):
            public_key = serialization.load_pem_public_key(key_bytes)
        else:
            public_key = serialization.load_der_public_key(key_bytes)
        encrypted = public_key.encrypt(message.encode("utf-8"), padding.PKCS1v15())
    except Exception as exc:
        raise FDMealPlannerConfigurationError("APP_SEC_PREFIX is not a usable RSA public key") from exc
    return base64.b64encode(encrypted).decode("ascii")


def _response_status(response: Any) -> int:
    try:
        return int(response.status_code)
    except (AttributeError, TypeError, ValueError) as exc:
        raise FDMealPlannerProtocolError("HTTP response has no valid status code") from exc


def _response_json(response: Any, *, url: str) -> Mapping[str, Any]:
    status_code = _response_status(response)
    if status_code < 200 or status_code >= 300:
        raise FDMealPlannerHTTPError(status_code, url)
    try:
        payload = response.json()
    except Exception as exc:
        raise FDMealPlannerProtocolError(f"FDMealPlanner returned invalid JSON: {url}") from exc
    if not isinstance(payload, Mapping):
        raise FDMealPlannerProtocolError(f"FDMealPlanner JSON root is not an object: {url}")
    return payload


def _extract_list(payload: Mapping[str, Any], *, label: str) -> tuple[Mapping[str, Any], ...]:
    value: Any = payload.get("data")
    if isinstance(value, Mapping):
        value = value.get("result") or value.get("data")
    if value is None:
        value = payload.get("result")
    if not isinstance(value, list):
        raise FDMealPlannerProtocolError(f"FD {label} response has no list data")
    result: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise FDMealPlannerProtocolError(f"FD {label} data[{index}] is not an object")
        result.append(item)
    return tuple(result)


class FDMealPlannerClient:
    """Small anonymous client for the public FDMealPlanner guest application."""

    def __init__(
        self,
        *,
        http: Any | None = None,
        application_data_url: str = DEFAULT_APPLICATION_DATA_URL,
        anonymous_token_origin: str = DEFAULT_ANONYMOUS_TOKEN_ORIGIN,
        location_search_url: str = DEFAULT_LOCATION_SEARCH_URL,
        public_client_key: str = DEFAULT_PUBLIC_CLIENT_KEY,
        public_client_secret_key: str = DEFAULT_PUBLIC_CLIENT_SECRET_KEY,
        device_type: str = DEFAULT_DEVICE_TYPE,
        timeout: float = 30.0,
        clock: Callable[[], int] | None = None,
        encryptor: Callable[[str, str], str] | None = None,
    ) -> None:
        if not _is_https_url(application_data_url):
            raise ValueError("application_data_url must be an HTTPS URL")
        if not _is_https_url(anonymous_token_origin):
            raise ValueError("anonymous_token_origin must be an HTTPS URL")
        if not _is_https_url(location_search_url):
            raise ValueError("location_search_url must be an HTTPS URL")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self._http = http if http is not None else requests.Session(impersonate="chrome")
        self._application_data_url = application_data_url
        self._anonymous_token_origin = anonymous_token_origin.rstrip("/")
        self._location_search_url = location_search_url
        self._public_client_key = public_client_key
        self._public_client_secret_key = public_client_secret_key
        self._device_type = device_type
        self._timeout = timeout
        self._clock = clock or (lambda: int(time.time() * 1000))
        self._encryptor = encryptor or _encrypt_guest_message
        self._configuration: FDApplicationConfiguration | None = None
        self._access_token: str | None = None

    def bootstrap(self, *, tenant_id: int = DEFAULT_TENANT_ID) -> FDApplicationConfiguration:
        """Load public routing and obtain a fresh anonymous guest bearer token."""

        initial_payload = self._request_json(
            "GET",
            self._application_data_url,
            headers={"Accept": "application/json"},
        )
        configuration = parse_initial_application_data(
            initial_payload,
            tenant_id=tenant_id,
            location_search_url=self._location_search_url,
        )
        self._configuration = configuration
        self._access_token = self._obtain_anonymous_token(configuration)
        return configuration

    def _obtain_anonymous_token(self, configuration: FDApplicationConfiguration) -> str:
        request_id = str(int(self._clock()))
        inner_payload = {
            "key": request_id,
            "clientKey": self._public_client_key,
            "clientSecretKey": self._public_client_secret_key,
            "deviceType": self._device_type,
        }
        inner_json = json.dumps(inner_payload, separators=(",", ":"), ensure_ascii=False)
        encrypted_message = self._encryptor(configuration.security_prefix, inner_json)
        body = {
            "requestId": request_id,
            "messageJson": encrypted_message,
            "isSecureData": True,
            "isAnonymousUser": True,
        }
        token_url = (
            f"{self._anonymous_token_origin}/api/v1/token-data/"
            f"{quote(self._public_client_key, safe='')}/token"
        )
        payload = self._request_json(
            "POST",
            token_url,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            json_body=body,
        )
        data = payload.get("data")
        if not isinstance(data, Mapping):
            data = payload
        token = data.get("accessToken")
        if not isinstance(token, str) or not token.strip():
            raise FDMealPlannerProtocolError("anonymous token response has no access token")
        return token.strip()

    def _request_json(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        json_body: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        request_headers = dict(headers or {})
        try:
            if method == "GET":
                response = self._http.get(
                    url,
                    params=dict(params or {}),
                    headers=request_headers,
                    timeout=self._timeout,
                )
            elif method == "POST":
                response = self._http.post(
                    url,
                    json=dict(json_body or {}),
                    headers=request_headers,
                    timeout=self._timeout,
                )
            else:
                raise ValueError(f"unsupported HTTP method: {method}")
        except FDMealPlannerError:
            raise
        except Exception as exc:
            raise FDMealPlannerError(f"FDMealPlanner request could not be sent: {url}") from exc
        return _response_json(response, url=url)

    def _authorized_json(
        self,
        *,
        tenant_id: int,
        url: str,
        params: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        for attempt in range(2):
            if self._configuration is None or self._configuration.tenant_id != tenant_id or not self._access_token:
                self.bootstrap(tenant_id=tenant_id)
            assert self._access_token is not None
            headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {self._access_token}",
            }
            try:
                response = self._http.get(
                    url,
                    params=dict(params),
                    headers=headers,
                    timeout=self._timeout,
                )
            except Exception as exc:
                raise FDMealPlannerError(f"FDMealPlanner request could not be sent: {url}") from exc
            if _response_status(response) == 401 and attempt == 0:
                self._access_token = None
                self._configuration = None
                continue
            return _response_json(response, url=url)
        raise FDMealPlannerHTTPError(401, url)

    def search_locations(self, *, search_text: str | None = None) -> tuple[Mapping[str, Any], ...]:
        """Search the public location directory without requiring a user login."""

        params: dict[str, Any] = {}
        if search_text is not None:
            params["searchText"] = search_text
        payload = self._request_json(
            "GET",
            self._location_search_url,
            params=params,
            headers={"Accept": "application/json"},
        )
        return _extract_list(payload, label="location")

    def fetch_meal_periods(
        self,
        *,
        tenant_id: int = DEFAULT_TENANT_ID,
        account_id: int = DEFAULT_ACCOUNT_ID,
        location_id: int = DEFAULT_LOCATION_ID,
    ) -> tuple[Mapping[str, Any], ...]:
        """Fetch the public meal-period directory for one account/location."""

        if self._configuration is None or self._configuration.tenant_id != tenant_id:
            self.bootstrap(tenant_id=tenant_id)
        assert self._configuration is not None
        payload = self._authorized_json(
            tenant_id=tenant_id,
            url=self._configuration.meal_period_url,
            params={"accountId": account_id, "locationId": location_id},
        )
        return _extract_list(payload, label="meal-period")

    def fetch_meals(
        self,
        *,
        tenant_id: int,
        account_id: int,
        location_id: int,
        meal_period_id: int,
        year: int,
        month: int,
        start_date: date | None = None,
        end_date: date | None = None,
        time_offset: int = 0,
    ) -> FDMealsPayload:
        """Fetch one explicitly bounded month/meal-period payload."""

        if not 1 <= month <= 12:
            raise ValueError("month must be between 1 and 12")
        if not 1 <= meal_period_id:
            raise ValueError("meal_period_id must be positive")
        first_day = date(year, month, 1) if start_date is None else start_date
        last_day = date(year, month, calendar.monthrange(year, month)[1]) if end_date is None else end_date
        if first_day > last_day:
            raise ValueError("start_date must not be after end_date")
        if first_day.year != year or first_day.month != month or last_day.year != year or last_day.month != month:
            raise ValueError("date bounds must remain within the requested month")
        if self._configuration is None or self._configuration.tenant_id != tenant_id:
            self.bootstrap(tenant_id=tenant_id)
        assert self._configuration is not None
        params = {
            "menuId": 0,
            "accountId": account_id,
            "locationId": location_id,
            "mealPeriodId": meal_period_id,
            "tenantId": tenant_id,
            "monthId": f"{month:02d}",
            "startDate": first_day.strftime("%Y/%m/%d"),
            "endDate": last_day.strftime("%Y/%m/%d"),
            "timeOffset": time_offset,
        }
        payload = self._authorized_json(
            tenant_id=tenant_id,
            url=self._configuration.menu_url,
            params=params,
        )
        return FDMealsPayload.from_payload(payload)

    def fetch_phelps_month(
        self,
        *,
        year: int,
        month: int,
        meal_period_id: int,
        start_date: date | None = None,
        end_date: date | None = None,
        time_offset: int = 0,
    ) -> FDMealsPayload:
        """Fetch one bounded Phelps month using the verified public IDs."""

        return self.fetch_meals(
            tenant_id=DEFAULT_TENANT_ID,
            account_id=DEFAULT_ACCOUNT_ID,
            location_id=DEFAULT_LOCATION_ID,
            meal_period_id=meal_period_id,
            year=year,
            month=month,
            start_date=start_date,
            end_date=end_date,
            time_offset=time_offset,
        )
