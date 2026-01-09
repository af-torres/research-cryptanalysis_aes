import secrets
from typing import List

class MiniAES5:
    """Mini AES implementation working on 5-bit words"""
    
    def __init__(self):
        # S-box for 5-bit values (32 entries) - must be a permutation
        self.sbox = [
            14, 4, 13, 1, 2, 15, 11, 8,
            3, 10, 6, 12, 5, 9, 0, 7,
            16, 20, 24, 28, 17, 21, 25, 29,
            18, 22, 26, 30, 19, 23, 27, 31
        ]
        
        # Inverse S-box
        self.inv_sbox = [0] * 32
        for i, v in enumerate(self.sbox):
            self.inv_sbox[v] = i
        
        # Verify S-box is a valid permutation
        assert len(set(self.sbox)) == 32, "S-box must be a permutation"
        
        # Round constants for key schedule
        self.rcon = [0x01, 0x02, 0x04, 0x08, 0x10]
    
    def key_expansion(self, key: List[int], rounds: int = 3) -> List[int]:
        """Expand the key for multiple rounds"""
        w = key[:]
        
        for i in range(4, 4 * (rounds + 1)):
            temp = w[i - 1]
            if i % 4 == 0:
                # Apply S-box and XOR with round constant
                temp = self.sbox[temp] ^ self.rcon[(i // 4) - 1]
            w.append(w[i - 4] ^ temp)
        
        return w
    
    def _sub_bytes(self, state: List[int]) -> List[int]:
        """Apply S-box substitution"""
        return [self.sbox[b & 0x1F] for b in state]
    
    def _inv_sub_bytes(self, state: List[int]) -> List[int]:
        """Apply inverse S-box substitution"""
        return [self.inv_sbox[b & 0x1F] for b in state]
    
    def _shift_rows(self, state: List[int]) -> List[int]:
        """Shift rows - swap middle two elements (self-inverse for 2x2)"""
        return [state[0], state[2], state[1], state[3]]
    
    def _add_round_key(self, state: List[int], round_key: List[int]) -> List[int]:
        """XOR state with round key"""
        return [(s ^ k) & 0x1F for s, k in zip(state, round_key)]
    
    def encrypt_block(self, block: List[int], round_keys: List[int]) -> List[int]:
        """Encrypt a single 4-word block"""
        state = [b & 0x1F for b in block]
        rounds = len(round_keys) // 4 - 1
        
        # Initial round key addition
        state = self._add_round_key(state, round_keys[0:4])
        
        # Main rounds (without mix columns for simplicity)
        for round_num in range(1, rounds + 1):
            state = self._sub_bytes(state)
            state = self._shift_rows(state)
            state = self._add_round_key(state, round_keys[round_num*4:(round_num+1)*4])
        
        return state
    
    def decrypt_block(self, block: List[int], round_keys: List[int]) -> List[int]:
        """Decrypt a single 4-word block"""
        state = [b & 0x1F for b in block]
        rounds = len(round_keys) // 4 - 1
        
        # Process rounds in reverse
        for round_num in range(rounds, 0, -1):
            state = self._add_round_key(state, round_keys[round_num*4:(round_num+1)*4])
            state = self._shift_rows(state)  # self-inverse
            state = self._inv_sub_bytes(state)
        
        # Initial round key
        state = self._add_round_key(state, round_keys[0:4])
        
        return state

def generate_key() -> List[int]:
    """Generate a random 20-bit key as 4 words of 5 bits each"""
    return [secrets.randbelow(32) for _ in range(4)]

def _pad_pkcs7(data: bytes, block_size: int) -> bytes:
    """Add PKCS7 padding"""
    padding_len = block_size - (len(data) % block_size)
    if padding_len == 0:
        padding_len = block_size
    return data + bytes([padding_len] * padding_len)


def _unpad_pkcs7(data: bytes) -> bytes:
    """Remove PKCS7 padding"""
    if len(data) == 0:
        return data
    padding_len = data[-1]
    if padding_len > len(data) or padding_len == 0:
        return data
    # Verify padding
    for i in range(1, padding_len + 1):
        if data[-i] != padding_len:
            return data
    return data[:-padding_len]


def encrypt(data: bytes, key: List[int]) -> bytes:
    """Encrypt a sequence of bytes"""
    if len(data) == 0:
        return bytes()
    
    aes = MiniAES5()
    round_keys = aes.key_expansion(key)
    
    # Pad the data to ensure we have full blocks
    # Each block is 4 words of 5 bits = 20 bits = 2.5 bytes
    # So we work with blocks of 5 bytes (= 8 words = 2 blocks of 4 words)
    padded_data = _pad_pkcs7(data, 5)
    
    result = []
    
    # Process 5 bytes at a time (8 words, 2 blocks)
    for i in range(0, len(padded_data), 5):
        chunk = padded_data[i:i+5]
        
        # Convert 5 bytes to 8 words of 5 bits
        bits = 0
        for byte in chunk:
            bits = (bits << 8) | byte
        
        words = []
        for j in range(8):
            word = (bits >> (40 - (j+1)*5)) & 0x1F
            words.append(word)
        
        # Encrypt two blocks
        encrypted = []
        encrypted.extend(aes.encrypt_block(words[0:4], round_keys))
        encrypted.extend(aes.encrypt_block(words[4:8], round_keys))
        
        # Convert 8 words back to 5 bytes
        bits_out = 0
        for word in encrypted:
            bits_out = (bits_out << 5) | (word & 0x1F)
        
        for j in range(5):
            byte = (bits_out >> (40 - (j+1)*8)) & 0xFF
            result.append(byte)
    
    return bytes(result)


def decrypt(encrypted_data: bytes, key: List[int]) -> bytes:
    """Decrypt a sequence of bytes"""
    if len(encrypted_data) == 0:
        return bytes()
    
    aes = MiniAES5()
    round_keys = aes.key_expansion(key)
    
    result = []
    
    # Process 5 bytes at a time
    for i in range(0, len(encrypted_data), 5):
        chunk = encrypted_data[i:i+5]
        if len(chunk) < 5:
            break
        
        # Convert 5 bytes to 8 words of 5 bits
        bits = 0
        for byte in chunk:
            bits = (bits << 8) | byte
        
        words = []
        for j in range(8):
            word = (bits >> (40 - (j+1)*5)) & 0x1F
            words.append(word)
        
        # Decrypt two blocks
        decrypted = []
        decrypted.extend(aes.decrypt_block(words[0:4], round_keys))
        decrypted.extend(aes.decrypt_block(words[4:8], round_keys))
        
        # Convert 8 words back to 5 bytes
        bits_out = 0
        for word in decrypted:
            bits_out = (bits_out << 5) | (word & 0x1F)
        
        for j in range(5):
            byte = (bits_out >> (40 - (j+1)*8)) & 0xFF
            result.append(byte)
    
    # Remove padding
    return _unpad_pkcs7(bytes(result))


# Example usage
if __name__ == "__main__":
    print("=== Mini AES-5 Test ===\n")
    
    # Generate a key
    key = generate_key()
    print(f"Key (5-bit words): {key}")
    print(f"Key (hex): {sum(k << (15 - i*5) for i, k in enumerate(key)):05x}")
    
    # Test various messages
    test_messages = [
        b"Hello, Mini AES!",
        b"Hi",
        b"Test",
        b"A longer message to test!",
        b"X",
        b"12345"
    ]
    
    all_passed = True
    for message in test_messages:
        print(f"\n--- Testing: {message} ---")
        
        # Encrypt
        ciphertext = encrypt(message, key)
        print(f"Encrypted ({len(ciphertext)} bytes): {ciphertext.hex()}")
        
        # Decrypt
        decrypted = decrypt(ciphertext, key)
        print(f"Decrypted: {decrypted}")
        
        # Verify
        if message == decrypted:
            print("✓ Success!")
        else:
            print(f"✗ FAILED! Expected {message}, got {decrypted}")
            all_passed = False
    
    print("\n" + "="*40)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")