#!/bin/bash

KEY_FILE=
IV_FILE=

print_usage() {
    cat <<EOF
Usage: $0 --key=<AES key path> --iv=<IV path>

Options:
  --key=KEY     Path to AES key file
  --iv=IV       Path to IV file

Example:
  $0 --key=mykey.hex --iv=mykey.hex.iv
EOF
    exit 1
}

for arg in "$@"; do
    case $arg in
        --key=*)
            KEY_FILE="${arg#*=}"
            shift
            ;;
        --iv=*)
            IV_FILE="${arg#*=}"
            shift
            ;;
    esac
done

if [[ -z "$KEY_FILE" ]]; then
    print_usage
fi

if [[ ! -f "$KEY_FILE" || ( ! -z "$IV_FILE" && ! -f "$IV_FILE" ) ]]; then
    echo key or iv are not valid files
    exit 1
fi

key=$(cat "$KEY_FILE")

if [[ ! -z "$IV_FILE" ]]; then
    iv=$(cat "$IV_FILE")
else
    iv=$(openssl rand -hex 16)
fi

key_size=$(( ${#key} / 2 * 8  ))
case "$key_size" in
    128|192|256)
        ;;
    *)
        echo "Invalid AES key size: $key_size-bit. Allowed sizes: 128, 192, 256."
        exit 1
        ;;
esac

openssl enc "-aes-$key_size-cbc" -K "$key" -iv "$iv" -a -A
