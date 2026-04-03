"""Shared Pydantic models used across all tool domains."""

from pydantic import BaseModel, Field


class DateRangeFilter(BaseModel):
    """Date range filter for Ariba reporting views."""

    created_date_from: str | None = Field(default=None, description="Start date, e.g. 2025-01-01")
    created_date_to: str | None = Field(default=None, description="End date, e.g. 2025-01-31")

    def to_dict(self) -> dict | None:
        d = {}
        if self.created_date_from:
            d["createdDateFrom"] = self.created_date_from
        if self.created_date_to:
            d["createdDateTo"] = self.created_date_to
        return d or None
