from setuptools import setup, find_packages

setup(
    name="src",
    version="0.0.1",
    author="Sayyed Sabir Ali",
    author_email="sabirali969091@gmail.com",
    packages=find_packages()
)

# name="src" (Ye tumhare package ka naam hai. Jab koi pip install karega, ye naam use hoga)
# version="0.0.1" (Tumhare package ka version)
# author, author_email (Tumhara naam aur contact)
# packages=find_packages()
# Ye automatically tumhare repo ke andar jo Python packages (folders with __init__.py) hain, unko dhoond kar include karega.