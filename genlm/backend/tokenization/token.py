"""Token class representing a vocabulary token with its ID and byte representation.

Token subclasses bytes for backwards compatibility so that ``b"".join(tokens)``
works. Equality and hashing between Token objects are based on token_id (not
byte content), because multiple tokens can share the same byte string.
"""


class Token(bytes):
    """A vocabulary token carrying both a token ID and its byte representation.

    Subclasses ``bytes`` so that existing code using byte operations (``b"".join``,
    ``len``, indexing, ``.decode()``) continues to work. Equality and hashing
    between Token objects use ``token_id``, not byte content.

    Args:
        token_id (int): The unique identifier for this token in the vocabulary.
        byte_string (bytes): The byte representation of this token.
    """

    def __new__(cls, token_id: int, byte_string: bytes):
        if not isinstance(token_id, int):
            raise TypeError(f"token_id must be an int, got {type(token_id)}")
        if not isinstance(byte_string, bytes):
            raise TypeError(f"byte_string must be bytes, got {type(byte_string)}")
        obj = super().__new__(cls, byte_string)
        obj.token_id = token_id
        return obj

    @property
    def byte_string(self):
        """The byte representation of this token (as plain bytes)."""
        return bytes(self)

    def __repr__(self):
        return f"Token(token_id={self.token_id}, byte_string={bytes(self)!r})"

    # -- Equality / hashing: by token_id between Tokens, by content vs bytes --

    def __eq__(self, other):
        if isinstance(other, Token):
            return self.token_id == other.token_id
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, Token):
            return self.token_id != other.token_id
        return NotImplemented

    def __hash__(self):
        return hash(self.token_id)

    # -- Ordering: by token_id --

    def __lt__(self, other):
        if not isinstance(other, Token):
            return NotImplemented
        return self.token_id < other.token_id

    def __le__(self, other):
        if not isinstance(other, Token):
            return NotImplemented
        return self.token_id <= other.token_id

    def __gt__(self, other):
        if not isinstance(other, Token):
            return NotImplemented
        return self.token_id > other.token_id

    def __ge__(self, other):
        if not isinstance(other, Token):
            return NotImplemented
        return self.token_id >= other.token_id

    # -- Pickle / deepcopy support --

    def __reduce__(self):
        return (Token, (self.token_id, bytes(self)))
