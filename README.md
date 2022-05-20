# LFGFX (Looking for GFX)
lfgfx is a tool to help reverse engineers analyze blobs of graphics data from N64 games.

<img width="670" alt="image" src="https://user-images.githubusercontent.com/2985314/169339874-bba46522-477f-4e7f-a049-603f5249b229.png">

### Currently supports the following graphics objects:
* display list
* vtx
* texture image
* palette

with more on the way!

### Requirements
`pip install -r requirements.txt`

## Usage
```
./lfgfx.py ~/repos/papermario/assets/us/37ADD0.bin 0x09000000
```
