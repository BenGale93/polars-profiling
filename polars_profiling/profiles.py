"""Module containing the standard profiles and profilers"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generic, Protocol, TypeVar

import numpy as np
import polars as pl
import polars.selectors as cs
from polars.type_aliases import SelectorType

from polars_profiling import templates


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
    range: float = field(init=False)  # noqa

    def __post_init__(self) -> None:
        self.range = self.maximum - self.minimum

    def to_html(self) -> str:
        return templates.render("numeric.html", profile=self, round=round)


class NumericProfiler(BaseProfiler[NumericProfile]):
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
        ]

    def dtype_filter(self) -> SelectorType:
        return cs.numeric()

    def result_constructor(self, *args: Any) -> NumericProfile:
        return NumericProfile(*args)


@dataclass(slots=True)
class QuantileProfile:
    percentiles: dict[float, float]
    iqr: float = field(init=False)

    def __post_init__(self) -> None:
        p_75 = self.percentiles.get(0.75, None)
        p_25 = self.percentiles.get(0.25, None)
        self.iqr = np.nan if p_75 is None or p_25 is None else (p_75 - p_25)

    def to_html(self) -> str:
        percentiles = {k: round(v, 3) for k, v in self.percentiles.items()}
        iqr = round(self.iqr, 3)
        return templates.render("quantile.html", percentiles=percentiles, iqr=iqr)


class QuantileProfiler(BaseProfiler[QuantileProfile]):
    def __init__(self, percentiles: list[float] | None = None) -> None:
        self.percentiles = percentiles if percentiles is not None else [0.05, 0.25, 0.5, 0.75, 0.95]

    def summary_expression(self) -> list[pl.Expr]:
        return [pl.all().quantile(p).prefix(f"{p}:") for p in self.percentiles]

    def dtype_filter(self) -> SelectorType:
        return cs.numeric()

    def result_constructor(self, *args: Any) -> QuantileProfile:
        percentiles = dict(zip(self.percentiles, args, strict=True))
        return QuantileProfile(percentiles=percentiles)


@dataclass(slots=True)
class BasicTemporalProfile:
    null_count: int
    distinct: float
    minimum: datetime
    maximum: datetime

    def to_html(self) -> str:
        return templates.render("temporal.html", profile=self)


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
