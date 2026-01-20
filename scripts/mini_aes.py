import argparse
import sys
import numpy as np

from sage.all import * # type: ignore
from sage.crypto.block_cipher.miniaes import MiniAES

maes = MiniAES()
K = FiniteField(Integer(16), "x")
MS = MatrixSpace(K, Integer(2), Integer(2)) # type:ignore
bin = BinaryStrings()

def generate_key(args):
    key = maes.random_key()
    print(key)
    key_bin = maes.GF_to_binary(key)

    with open(args.out, "w+") as f:
        f.write(key_bin.__str__())

    print(f"wrote key at: {args.out}")

def encrypt(args):
    def char_to_int(char):
        number = int(char)
        if number < 0 or number > 15: raise ValueError(f"Invalid integer value: {number}")
        return Integer(number)
    
    with open(args.key, "r") as f:
        key_bin = bin(f.read())
    key = MS(maes.binary_to_GF(key_bin))

    txt = sys.stdin.read()
    try:
        numbers = [char_to_int(char) for char in txt.split(",")]
    except ValueError as e:
        print("Expected input to be a sequence of numbers [0, 15]:", e, file=sys.stderr)
        sys.exit(1)

    plain_text_array = np.array(maes.integer_to_GF(numbers)).reshape((-1, 4))
    ciphered_text = ""

    plain_text = MS(plain_text_array[0, :].tolist())
    ciphered_text += ",".join([c.__str__() for c in maes.GF_to_integer(maes.encrypt(plain_text, key))])
    sys.stdout.write(ciphered_text)
    
    for i in range(1, len(plain_text_array)):
        sys.stdout.write(",")
        plain_text = MS(plain_text_array[i, :].tolist())
        ciphered_text += ",".join([c.__str__() for c in maes.GF_to_integer(maes.encrypt(plain_text, key))])
    
    sys.stdout.write(ciphered_text)

def decrypt(args):
    print("Not implemented")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(prog="tool")
    subparsers = parser.add_subparsers(required=True)

    generate_key_parser = subparsers.add_parser("new_key")
    generate_key_parser.add_argument("--out", type=str, default="./data/keys/mini_aes.bin")
    generate_key_parser.set_defaults(func=generate_key)

    encrypt_parser = subparsers.add_parser("encrypt")
    encrypt_parser.add_argument("--key", type=str, help="Mini-AES key path")
    encrypt_parser.add_argument("--iv", type=str, help="Mini-AES IV path")
    encrypt_parser.set_defaults(func=encrypt)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
