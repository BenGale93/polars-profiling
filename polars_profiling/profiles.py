"""Module containing the standard profiles and profilers"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Generic, Protocol, TypeVar

import polars as pl
import polars.selectors as cs
from jinja2 import Template
from polars.type_aliases import SelectorType


class BaseProfile(Protocol):
    def to_html(self) -> str:
        pass


T_co = TypeVar("T_co", bound=BaseProfile, covariant=True)


class BaseProfiler(ABC, Generic[T_co]):
    @abstractmethod
    def summary_expression(self) -> list[pl.Expr]:
        pass

    @abstractmethod
    def dtype_filter(self) -> SelectorType:
        pass

    @abstractmethod
    def result_constructor(self, *args: Any) -> T_co:
        pass

    def summarise(self, df: pl.DataFrame) -> dict[str, T_co]:
        sub_df = df.select(self.dtype_filter())

        results = sub_df.select(self.summary_expression()).row(0)

        n_cols = len(sub_df.columns)
        described = [results[n::n_cols] for n in range(n_cols)]

        return {
            col: self.result_constructor(*data)
            for col, data in zip(sub_df.columns, described, strict=True)
        }


NUMERIC_TEMPLATE = Template(
    """
<table>
    <tbody>
        <tr><th>Distinct</th><td>{{profile.distinct}}</td></tr>
        <tr><th>Missing</th><td>{{profile.null_count}}</td></tr>
        <tr><th>Infinite</th><td>{{profile.infinite}}</td></tr>
        <tr><th>Mean</th><td>{{profile.mean}}</td></tr>
        <tr><th>Min</th><td>{{profile.minimum}}</td></tr>
        <tr><th>Max</th><td>{{profile.maximum}}</td></tr>
        <tr><th>Zeros</th><td>{{profile.zeros}}</td></tr>
        <tr><th>Negative</th><td>{{profile.negative}}</td></tr>
    </tbody>
</table>
    """
)


@dataclass(slots=True)
class NumericProfile:
    null_count: int
    distinct: float
    infinite: int
    mean: float
    minimum: float
    maximum: float
    zeros: int
    negative: int
    percentiles: dict[float, float]
    range: float | None = None  # noqa
    iqr: float | None = None

    def __post_init__(self) -> None:
        self.range = self.maximum - self.minimum
        p_75 = self.percentiles.get(0.75, None)
        p_25 = self.percentiles.get(0.25, None)
        self.iqr = None if p_75 is None or p_25 is None else (p_75 - p_25)

    def to_html(self) -> str:
        return NUMERIC_TEMPLATE.render(profile=self)


class NumericProfiler(BaseProfiler[NumericProfile]):
    def __init__(self, percentiles: list[float] | None = None) -> None:
        self.percentiles = percentiles if percentiles is not None else [0.05, 0.25, 0.5, 0.75, 0.95]

    def summary_expression(self) -> list[pl.Expr]:
        return [
            pl.all().null_count().prefix("null_count:"),
            pl.all().unique().len().prefix("unique:"),
            pl.all().is_infinite().sum().prefix("infinite:"),
            pl.all().mean().prefix("mean:"),
            pl.all().min().prefix("min:"),
            pl.all().max().prefix("max:"),
            pl.all().eq(0).sum().prefix("zero:"),
            pl.all().lt(0).sum().prefix("negative:"),
            *[pl.all().quantile(p).prefix(f"{p}:") for p in self.percentiles],
        ]

    def dtype_filter(self) -> SelectorType:
        return cs.numeric()

    def result_constructor(self, *args: Any) -> NumericProfile:
        percentiles = dict(zip(self.percentiles, args[8:], strict=True))
        return NumericProfile(*args[:8], percentiles=percentiles)  # type: ignore


TEMPORAL_TEMPLATE = Template(
    """
<table>
    <tbody>
        <tr><th>Distinct</th><td>{{profile.distinct}}</td></tr>
        <tr><th>Missing</th><td>{{profile.null_count}}</td></tr>
        <tr><th>Min</th><td>{{profile.minimum}}</td></tr>
        <tr><th>Max</th><td>{{profile.maximum}}</td></tr>
    </tbody>
</table>
    """
)


@dataclass(slots=True)
class BasicTemporalProfile:
    null_count: int
    distinct: float
    minimum: datetime
    maximum: datetime

    def to_html(self) -> str:
        return TEMPORAL_TEMPLATE.render(profile=self)


class BasicTemporalProfiler(BaseProfiler[BasicTemporalProfile]):
    def summary_expression(self) -> list[pl.Expr]:
        return [
            pl.all().null_count().prefix("null_count:"),
            pl.all().unique().len().prefix("unique"),
            pl.all().min().prefix("min:"),
            pl.all().max().prefix("max:"),
        ]

    def dtype_filter(self) -> SelectorType:
        return cs.temporal()

    def result_constructor(self, *args: Any) -> BasicTemporalProfile:
        return BasicTemporalProfile(*args)
