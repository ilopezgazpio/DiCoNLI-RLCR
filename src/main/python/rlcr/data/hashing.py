import hashlib


def string_to_short_id(string):
    # Use SHA-256 hash function
    hash_object = hashlib.sha256(string.encode())
    # Convert the hash to a hexadecimal string
    hex_dig = hash_object.hexdigest()
    # Convert the hexadecimal string to an integer
    hash_int = int(hex_dig, 16)
    # Truncate to 8 digits by taking modulo 10^8
    hash_id = hash_int % 10**10
    return hash_id


def hash_dataset(example, key):
    return {"id": string_to_short_id(example[key])}
