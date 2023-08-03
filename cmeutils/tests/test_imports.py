# ignoring unused imports for testing purposes
# flake8: noqa: F401
def test_import():
    try:
        import cmeutils
        from cmeutils import gsd_utils
    except ImportError:
        assert False
