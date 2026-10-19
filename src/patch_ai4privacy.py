from pathlib import Path
import ai4privacy


def main():
    # Find the installed ai4privacy package
    package_dir = Path(ai4privacy.__file__).resolve().parent

    # Locate protect.py relative to the package
    path = package_dir / "protect" / "protect.py"

    print(f"Found protect.py at:")
    print(f"  {path}")

    if not path.exists():
        raise RuntimeError(
            f"Could not find:\n{path}\n"
            "The installed ai4privacy package may have a different structure."
        )

    text = path.read_text()

    old = 'placeholder = f"[PII_{i+1}]"'
    new = 'placeholder = "[MASK]"'

    if new in text:
        print("Already patched.")
        return

    if old not in text:
        raise RuntimeError(
            f"Expected line was not found in:\n{path}\n"
            "The installed ai4privacy version may be different."
        )

    # Backup original file
    backup = path.with_suffix(".py.bak")
    backup.write_text(text)

    # Apply modification
    path.write_text(text.replace(old, new, 1))

    print("Successfully patched ai4privacy.")
    print(f"Backup created at:")
    print(f"  {backup}")


if __name__ == "__main__":
    main()