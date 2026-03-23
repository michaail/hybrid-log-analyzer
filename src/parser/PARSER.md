# Parser Structure

## High-Level Structure

The script defines:

### `DrainTrainer` class

The `DrainTrainer` class:

- Initializes a Drain3 `TemplateMiner`
- Reads a log file and "trains" (learns templates)
- Stores mappings from templates to raw lines for validation
- Runs several heuristic validation checks
- Exports learned templates to JSON for manual inspection

### `main()` function

The `main()` function:

- Parses command-line arguments
- Constructs a `DrainTrainer`
- Runs `fit_file()`, `validate()`, and `export_templates()` in order

## Drain3

Drain3 is an online implementation of the Drain parser: it incrementally builds a parse tree and assigns each log message to one of a set of learned templates.
