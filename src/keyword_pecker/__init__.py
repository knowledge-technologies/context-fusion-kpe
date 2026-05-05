try:
    from keyword_pecker._version import GIT_COMMIT
except ImportError:
    GIT_COMMIT = "unknown"


def main() -> None:
    print("Hello from keyword-pecker!")
