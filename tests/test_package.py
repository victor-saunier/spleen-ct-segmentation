import spleenseg


def test_package_is_installed_and_has_a_version() -> None:
    # With the src/ layout this import only works if the package is really installed,
    # so this catches a broken pyproject.toml or environment early.
    assert isinstance(spleenseg.__version__, str)
    assert spleenseg.__version__
