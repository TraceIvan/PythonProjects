import hashlib
import _md5
md5value=hashlib.md5()
md5value.update('12345'.encode())
md5value=md5value.hexdigest()
print(md5value)
md5value=_md5.md5()
md5value.update('12345'.encode())
md5value=md5value.hexdigest()
print(md5value)