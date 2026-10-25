# Hardware Configuration Constants
PAGE_BITS = 6
BLOCK_BITS = 10

PAGE_MASK = (1 << PAGE_BITS) - 1   # 0x3F (binary: 00111111)
BLOCK_MASK = (1 << BLOCK_BITS) - 1 # 0x3FF (binary: 1111111111)


# SLC Small Block (Legacy)
SLC_SMALL = {
    "name": "SLC-Small-Block",
    "page_size": 1024, # Bytes
    "pages_per_block": 2**PAGE_BITS,
    "total_blocks": 2**BLOCK_BITS,       # <-- explicit, not derived
}


def encode_address(block: int, page: int) -> int:
    """
    Encodes a Block ID and Page ID into a single 16-bit physical address.
    """
    # Safety checks to prevent bit overflow
    if not (0 <= block <= BLOCK_MASK):
        raise ValueError(f"Block {block} exceeds maximum allowed value of {BLOCK_MASK}")
    if not (0 <= page <= PAGE_MASK):
        raise ValueError(f"Page {page} exceeds maximum allowed value of {PAGE_MASK}")
        
    # Shift block to the left by 6 bits and combine with page
    physical_address = (block << PAGE_BITS) | page
    return physical_address


def decode_address(physical_address: int) -> tuple[int, int]:
    """
    Decodes a 16-bit physical address back into (Block ID, Page ID).
    """
    # Extract page from the lowest 6 bits
    page = physical_address & PAGE_MASK
    
    # Shift right by 6 bits to extract the block
    block = (physical_address >> PAGE_BITS) & BLOCK_MASK
    
    return block, page


# --- Verification Example ---
if __name__ == "__main__":
    # Let's say we want to store data in Block 512, Page 3
    target_block = 512
    target_page = 3
    
    # 1. Encode
    encoded_ppa = encode_address(target_block, target_page)
    print(f"Encoded Address (Integer): {encoded_ppa}")
    print(f"Encoded Address (Hex):     {hex(encoded_ppa)}")
    print(f"Encoded Address (Binary):  {bin(encoded_ppa)[2:].zfill(16)}")
    print("-" * 40)
    
    # 2. Decode
    decoded_block, decoded_page = decode_address(encoded_ppa)
    print(f"Decoded -> Block: {decoded_block}, Page: {decoded_page}")
    
    # Verify match
    assert (target_block, target_page) == (decoded_block, decoded_page), "Mismatch detected!"
