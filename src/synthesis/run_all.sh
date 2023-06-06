# Extract from config
FWD_FILE=$(cat $1 | \
     python -c 'import json,sys;obj=json.load(sys.stdin); print(obj["align_fwd_file"]);')
REV_FILE=$(cat $1 | \
python -c 'import json,sys;obj=json.load(sys.stdin); print(obj["align_rev_file"]);')
RESULT=$(cat $1 | \
     python -c 'import json,sys;obj=json.load(sys.stdin); print(obj["align_sym_file"]);')
TRANSLITERATE=$(cat $1 | \
     python -c 'import json,sys;obj=json.load(sys.stdin); print(obj["transliterate"]);')


# Run translation
python3 src/synthesis/translate.py $1

# Run transliteration if specified
if [ "$TRANSLITERATE" = "True" ]
then
    python3 src/synthesis/transliterate.py $1
fi

# Run alignment
python3 src/synthesis/align.py $1

# Run symmetrization
$2/fast_align/build/atools -c grow-diag-final-and -i $FWD_FILE -j $REV_FILE >$RESULT

# Run retrieval
python3 src/synthesis/retrieve.py $1