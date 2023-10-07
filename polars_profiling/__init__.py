"""A library for profiling your datasets using Polars."""
from dataclasses import dataclass
from functools import reduce

import polars as pl
from jinja2 import Template

from polars_profiling import profiles

PROFILE_MANAGER_TEMPLATE = Template(
    """
<div>
<h3>{{manager.name}}</h3>
<p>{{manager.dtype}}</p>
{{manager.profile.to_html()}}
</div>
    """
)


@dataclass
class ProfileManager:
    name: str
    dtype: str
    profile: profiles.BaseProfile

    def to_html(self) -> str:
        return PROFILE_MANAGER_TEMPLATE.render(manager=self)


PROFILER_TYPES: list[type[profiles.BaseProfiler[profiles.BaseProfile]]] = [
    profiles.NumericProfiler,
    profiles.BasicTemporalProfiler,
]


def run_column_profiles(df: pl.DataFrame) -> dict[str, ProfileManager]:
    list_of_dicts = [p().summarise(df) for p in PROFILER_TYPES]
    profile_map = reduce(lambda d1, d2: d1 | d2, list_of_dicts)

    return {
        name: ProfileManager(name, str(df.select(name).dtypes[0]), p)
        for name, p in profile_map.items()
    }


TABLE_SUMMARY = Template(
    """
<h3>Dataset statistics</h3>
<table>
    <tbody>
        <tr><th>Variable</th><td>{{table.variables}}</td></tr>
        <tr><th>Observations</th><td>{{table.observations}}</td></tr>
        <tr><th>Duplicates</th><td>{{table.duplicates}}</td></tr>
    </tbody>
</table>
    """
)


@dataclass
class TableSummary:
    variables: int
    observations: int
    duplicates: int
    variable_types: dict[str, int]

    def to_html(self) -> str:
        return TABLE_SUMMARY.render(table=self)


def get_table_summary(df: pl.DataFrame) -> TableSummary:
    observations, variables = df.shape
    duplicates = int(df.is_duplicated().sum())
    variable_types = {
        str(row[0]): int(row[1]) for row in pl.Series(df.dtypes).value_counts().iter_rows()
    }

    return TableSummary(variables, observations, duplicates, variable_types)


BASE_DESCRIPTION = Template(
    """
<body>
{{base.table.to_html()}}

<br>

{% for profile in base.variables.values() %}
{{profile.to_html()}}
{% endfor %}
</body>
    """
)


@dataclass
class BaseDescription:
    table: TableSummary
    variables: dict[str, ProfileManager]

    def to_html(self) -> str:
        return BASE_DESCRIPTION.render(base=self)


def run_profile(df: pl.DataFrame) -> BaseDescription:
    table = get_table_summary(df)
    variables = run_column_profiles(df)

    return BaseDescription(table=table, variables=variables)
