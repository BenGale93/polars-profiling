"""A library for profiling your datasets using Polars."""
from collections import defaultdict
from dataclasses import dataclass

import polars as pl

from polars_profiling import profiles, templates


@dataclass
class ProfileManager:
    name: str
    dtype: str
    profiles: list[profiles.BaseProfile]

    def to_html(self) -> str:
        return templates.render("manager.html", manager=self)


PROFILER_TYPES: list[type[profiles.BaseProfiler[profiles.BaseProfile]]] = [
    profiles.NumericProfiler,
    profiles.BasicTemporalProfiler,
    profiles.QuantileProfiler,
    profiles.StatsProfiler,
]


def run_column_profiles(df: pl.DataFrame) -> dict[str, ProfileManager]:
    profile_results = [p().summarise(df) for p in PROFILER_TYPES]
    profile_map = defaultdict(list)
    for profile_result in profile_results:
        for col, profile_ in profile_result.items():
            profile_map[col].append(profile_)

    return {
        name: ProfileManager(name, str(df.select(name).dtypes[0]), p)
        for name, p in profile_map.items()
    }


@dataclass
class TableSummary:
    variables: int
    observations: int
    duplicates: int
    variable_types: dict[str, int]

    def to_html(self) -> str:
        return templates.render("dataset.html", table=self)


def get_table_summary(df: pl.DataFrame) -> TableSummary:
    observations, variables = df.shape
    duplicates = int(df.is_duplicated().sum())
    variable_types = {
        str(row[0]): int(row[1]) for row in pl.Series(df.dtypes).value_counts().iter_rows()
    }

    return TableSummary(variables, observations, duplicates, variable_types)


@dataclass
class BaseDescription:
    table: TableSummary
    variables: dict[str, ProfileManager]

    def to_html(self) -> str:
        return templates.render("base.html", base=self)


def run_profile(df: pl.DataFrame) -> BaseDescription:
    table = get_table_summary(df)
    variables = run_column_profiles(df)

    return BaseDescription(table=table, variables=variables)
