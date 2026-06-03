"""Pixel-scan the Kontakt header row to find exact FILE/LIBRARY positions."""
import json
from PIL import ImageGrab

with open('dashboard~/calibration.json') as f:
    cal = json.load(f)['kontakt']
kx, ky = cal['x'], cal['y']
header_y = cal['header_y']

img = ImageGrab.grab(all_screens=True)
crop = img.crop((kx, ky + header_y - 3, kx + cal['w'], ky + header_y + 4))
scaled = crop.resize((crop.width * 5, crop.height * 5), resample=0)
scaled.save('dashboard~/_header_precise.png')

print(f'Header row y={header_y} (abs y={ky+header_y}), window width={cal["w"]}')
print('Scanning bright clusters (brightness > 140):')

in_cluster = False
cluster_start = 0
cluster_pixels = []
for x in range(0, cal['w']):
    px = img.getpixel((kx + x, ky + header_y))
    b = (px[0] + px[1] + px[2]) / 3
    if b > 140:
        if not in_cluster:
            in_cluster = True
            cluster_start = x
            cluster_pixels = [x]
        else:
            cluster_pixels.append(x)
    else:
        if in_cluster:
            # Check if this is a real end or just a 1-2px gap
            gap = x - cluster_pixels[-1]
            if gap > 3:
                cx = (cluster_start + cluster_pixels[-1]) // 2
                # Get representative color
                px_rep = img.getpixel((kx + cx, ky + header_y))
                print(f'  x={cluster_start:3d}-{cluster_pixels[-1]:3d} center={cx:3d} abs=({kx+cx},{ky+header_y}) '
                      f'rgb=({px_rep[0]:3d},{px_rep[1]:3d},{px_rep[2]:3d})')
                in_cluster = False
                cluster_pixels = []

if in_cluster:
    cx = (cluster_start + cluster_pixels[-1]) // 2
    print(f'  x={cluster_start:3d}-{cluster_pixels[-1]:3d} center={cx:3d} (to end of window)')

print()
print('Saved _header_precise.png (5x scale)')
