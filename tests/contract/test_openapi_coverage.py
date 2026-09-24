"""Keep the published Core route index and YAML in step with Axum routes."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
METHODS = r"get|post|put|delete"


def axum_routes(source: str) -> set[tuple[str, str]]:
    router = source.split("pub fn app(state: AppState) -> Router {", 1)[1].split(
        ".layer(middleware", 1
    )[0]
    routes: set[tuple[str, str]] = set()
    for match in re.finditer(rf'\.route\(\s*"([^"]+)"\s*,\s*({METHODS})\(', router):
        path, method = match.groups()
        routes.add((path, method))
        continuation = router[match.end() :].split(".route(", 1)[0]
        routes.update((path, extra) for extra in re.findall(rf"\.({METHODS})\(", continuation))
    return routes


def yaml_routes(source: str) -> set[tuple[str, str]]:
    paths = source.split("\npaths:\n", 1)[1].split("\ncomponents:\n", 1)[0]
    routes: set[tuple[str, str]] = set()
    path = ""
    for line in paths.splitlines():
        route = re.fullmatch(r"  (/[^:]+):", line)
        if route:
            path = route.group(1)
        else:
            method = re.fullmatch(rf"    ({METHODS}):", line)
            if method:
                routes.add((path, method.group(1)))
    return routes


def json_index_routes(source: str) -> set[tuple[str, str]]:
    index = source.split("async fn openapi() -> Json<Value> {", 1)[1].split(
        "#[cfg(test)]", 1
    )[0]
    routes: set[tuple[str, str]] = set()
    for path, operations in re.findall(r'"(/[^\"]+)":\s*\{([^\n]*)', index):
        routes.update(
            (path, method)
            for method in re.findall(rf'"({METHODS})"\s*:', operations)
        )
    return routes


class OpenApiCoverageTests(unittest.TestCase):
    def test_core_documents_every_axum_route(self) -> None:
        rust = (ROOT / "backend/src/lib.rs").read_text()
        yaml = (ROOT / "docs/openapi/core.openapi.yaml").read_text()
        actual = axum_routes(rust)
        self.assertEqual(actual, yaml_routes(yaml))
        self.assertEqual(actual, json_index_routes(rust))


if __name__ == "__main__":
    unittest.main()
