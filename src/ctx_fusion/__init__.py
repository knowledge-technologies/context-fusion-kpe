try:
    from ctx_fusion._version import GIT_COMMIT
except ImportError:
    GIT_COMMIT = "unknown"


def main() -> None:
    print("Hello from ctx-fusion!")
