#!/bin/bash

KEY_SIZE=
OUTPUT_FILE=

print_usage() {
    cat <<EOF
Usage: $0 --size=<AES key size> --out=<output file>

Options:
  --size=SIZE      AES key size in bits. Allowed values: 128, 192, 256
  --out=FILE       Path to write the generated AES key

Example:
  $0 --size=256 --out=mykey.hex
EOF
    exit 1
}

for arg in "$@"; do
    case $arg in
        --size=*)
            KEY_SIZE="${arg#*=}"
            shift
            ;;
        --out=*)
            OUTPUT_FILE="${arg#*=}"
            shift
            ;;
    esac
done

case "$KEY_SIZE" in
    128|192|256)
        ;;
    *)
        echo "Invalid AES key size: $key_size-bit. Allowed sizes: 128, 192, 256."
        exit 1
        ;;
esac

if [[ -z "$OUTPUT_FILE" ]]; then
    print_usage
fi

num_bytes=$(($KEY_SIZE / 8))
KEY=$(openssl rand -hex "$num_bytes")

IV=$(openssl rand -hex 16) # Used for AES-CBC encryption

printf "%s" "$KEY" > "$OUTPUT_FILE"
echo wrote "$OUTPUT_FILE"

printf "%s" "$IV" > "$OUTPUT_FILE.iv"
echo wrote "$OUTPUT_FILE.iv"

