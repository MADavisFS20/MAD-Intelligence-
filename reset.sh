set -e
cd ~/MI
echo "DELETES: checkpoints/, tokenizer/mad_bpe.json, config.json,"
echo "         data/processed/*, data/instructions/auto_from_code.jsonl, __pycache__"
echo "KEEPS:   data/raw, other data/instructions files, mad/ code"
read -p "Type WIPE to continue: " a
[ "$a" = "WIPE" ] || { echo "Aborted."; exit 1; }
rm -rf checkpoints tokenizer/mad_bpe.json config.json data/processed data/instructions/auto_from_code.jsonl
find . -name __pycache__ -type d -prune -exec rm -rf {} +
mkdir -p checkpoints tokenizer data/processed
echo "Done."
