import pathlib
import re
import subprocess
import sys


FILE_PATTERN = re.compile(
    r"^(run_.*\.py|src/.*\.py|tests/.*\.(py|json|txt)|"
    r"inference_data_examples/.*\.(py|json|txt))$"
)


def get_files() -> list[str]:
    return [
        file.as_posix()
        for file in pathlib.Path.cwd().rglob("*")
        if file.is_file()
        and ".git" not in file.parts
        and FILE_PATTERN.fullmatch(file.relative_to(pathlib.Path.cwd()).as_posix())
    ]


def main() -> int:
    files = get_files()

    if not files:
        return 0

    print(files)

    command = sys.argv[1:]

    print(command)

    result = subprocess.run(
        command + files,
        check=False,
    )

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())