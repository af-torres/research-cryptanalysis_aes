import argparse
import sys

from sage.all import * # type: ignore
from sage.crypto.block_cipher.miniaes import MiniAES

maes = MiniAES()
MS = MatrixSpace(FiniteField(Integer(16), "x"), Integer(2), Integer(2)) # type: ignore

def generate_key(args):
    key = maes.random_key()
    key_bin = maes.GF_to_binary(key)

    with open(args.out, "w+") as f:
        f.write(key_bin.__str__())

    print(f"wrote key at: {args.out}")

def encrypt(args):
    txt = sys.stdin.read()
    

def decrypt(args):
    pass

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