from typing import Any

import jinja2

package_loader = jinja2.PackageLoader("polars_profiling", "templates")

jinja2_env = jinja2.Environment(lstrip_blocks=True, trim_blocks=True, loader=package_loader)


def render(template_name: str, **kwargs: Any) -> str:
    return jinja2_env.get_template(template_name).render(**kwargs)
