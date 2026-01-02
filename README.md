# Proberbiaml

## Usage:

1. Download the proverbi `pdf` file and place it in the repository's folder. Name it `proverbi.pdf`.

2. Create the python virtual environment and install the dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

3. Run the first script to generate the `csv` file with the scraped content:

```bash
python parse_proverbi.py proverbi.pdf --skip-pages 64 --out proverbi.csv
```

4. Run the second script to generate the database with embedded sentences and indexes:

```bash
python proverbi_semantic.py build --csv proverbi.csv --recreate --include-title
```

5. Enjoy! Describe a situation you want a `proverbio` for, and run the following command to find the closest 3 `proverbios`:

```bash
python proverbi_semantic.py search --query "Mi piacciono i proverbi e ora ci faccio un gioco al riguardo" --top-k 3
```
