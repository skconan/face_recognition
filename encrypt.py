import json
from dotenv import dotenv_values
from base64 import b64encode, b64decode

from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad


config = dotenv_values(".env")

AES_KEY = bytes(config.get("AES_KEY", "").encode("utf-8"))
BLOCK_SIZE = AES.block_size


def encrypt_data(data: dict) -> str:
    cipher = AES.new(AES_KEY, AES.MODE_CBC)
    iv = cipher.iv
    json_data = json.dumps(data).encode("utf-8")
    padded_data = pad(json_data, BLOCK_SIZE)
    encrypted_bytes = cipher.encrypt(padded_data)
    return b64encode(iv + encrypted_bytes).decode("utf-8")


def decrypt_data(encrypted_data: str) -> dict:
    encrypted_bytes = b64decode(encrypted_data)
    iv = encrypted_bytes[:BLOCK_SIZE]
    cipher = AES.new(AES_KEY, AES.MODE_CBC, iv)
    decrypted_padded_data = cipher.decrypt(encrypted_bytes[BLOCK_SIZE:])
    decrypted_data = unpad(decrypted_padded_data, BLOCK_SIZE)
    return json.loads(decrypted_data.decode("utf-8"))


def test_aes_encryption_decryption():
    """Test AES encryption and decryption."""
    test_data = {"message": "Test message", "name": "Supakit", "id": 12345}

    print("Original Data:")
    print(test_data)

    encrypted_data = encrypt_data(test_data)
    print("\nEncrypted Data (Base64):")
    print(encrypted_data)

    decrypted_data = decrypt_data(encrypted_data)
    print("\nDecrypted Data:")
    print(decrypted_data)

    assert (
        test_data == decrypted_data
    ), "Decrypted data does not match the original data!"


if __name__ == "__main__":
    # test_aes_encryption_decryption()
    data = input("Enter data to decrypt: ")
    decrypted_data = decrypt_data(data)
    print(f"Decrypted data: {decrypted_data}")
