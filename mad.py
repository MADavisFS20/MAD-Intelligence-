import sys

try:
    from mad.cli import main
except ModuleNotFoundError as e:
    if e.name == "torch":
        sys.exit("PyTorch is not installed. In Termux run: pkg install python-torch\n"
                 "(see README.md, Setup). Then retry.")
    raise

if __name__ == "__main__":
    main()
