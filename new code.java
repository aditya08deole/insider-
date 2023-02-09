import hashlib
import random

def generate_anonymous_address(address):
    """
    Generates an anonymous address for an airdrop recipient.
    """
    # Hash the original address to create a unique identifier
    hashed_address = hashlib.sha256(address.encode()).hexdigest()

    # Generate a random number to use as a salt
    salt = str(random.randint(0, 100000))