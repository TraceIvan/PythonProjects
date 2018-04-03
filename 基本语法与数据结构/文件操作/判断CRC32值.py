import zlib
print(zlib.crc32(b'1234'))
print(zlib.crc32(b'111'))
print(zlib.crc32(b'SDIBT'))
import binascii
print(binascii.crc32('SDIBT'.encode()))
import os
print(zlib.crc32(os.path.abspath('file2.txt').encode()))