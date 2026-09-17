"""Module execution and the console command share exactly the same CLI."""
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
