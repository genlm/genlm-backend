"""Token class representing a vocabulary token with its ID and byte representation"""


class Token:
    """Represents a vocabulary token with both its ID and byte representation.

    Args:
        token_id (int): The unique identifier for this token in the vocabulary.
        byte_string (bytes): The byte representation of this token.

    Attributes:
        token_id (int): The unique identifier for this token in the vocabulary.
        byte_string (bytes): The byte representation of this token.
    """

    __slots__ = ("token_id", "byte_string")

    def __init__(self, token_id: int, byte_string: bytes):
        """Initialize a Token instance.

        Args:
            token_id (int): The unique identifier for this token.
            byte_string (bytes): The byte representation of this token.
        """
        if not isinstance(token_id, int):
            raise TypeError(f"token_id must be an int, got {type(token_id)}")
        if not isinstance(byte_string, bytes):
            raise TypeError(f"byte_string must be bytes, got {type(byte_string)}")

        self.token_id = token_id
        self.byte_string = byte_string

    def __repr__(self):
        """Return a string representation of the token."""
        return f"Token(token_id={self.token_id}, byte_string={self.byte_string!r})"

    def __eq__(self, other):
        """Check equality based on token_id only.

        compare only token_id because multiple tokens can have the same
        byte_string. The token_id is the unique identifier.
        """
        if not isinstance(other, Token):
            return False
        return self.token_id == other.token_id

    def __hash__(self):
        """Hash based on token_id only (consistent with __eq__)."""
        return hash(self.token_id)

    def __len__(self):
        """Return the length of the byte string."""
        return len(self.byte_string)

    def __getitem__(self, key):
        """Allow indexing into the byte string for iteration compatibility."""
        return self.byte_string[key]

    def __iter__(self):
        """Allow iteration over the bytes in the byte string."""
        return iter(self.byte_string)

    def decode(self, encoding="utf-8", errors="strict"):
        """Decode the byte string to a string.

        This method provides compatibility with bytes.decode() interface.

        Args:
            encoding (str): The encoding to use. Defaults to 'utf-8'.
            errors (str): Error handling strategy. Defaults to 'strict'.

        Returns:
            (str): The decoded string.
        """
        return self.byte_string.decode(encoding, errors)
