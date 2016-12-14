# Batch thumbnail generation script using PIL

import sys
import os
import os.path
from PIL import Image
from PIL import ImageOps
from PIL import ImageDraw
from PIL import ImageEnhance

# use these as a number to identify the images after transformation image.
k = 101  # flipped image
s = 201  # mirror images start 201 suffix to 300
h = 301  # flip and rotated image will start from 301 to 400
# Loop through all provided images and add numbers based on transforms

# transpose


t = 401

# Dim image

d = 501

# rotate
r1 = 601
r2 = 701
r3 = 801
r4 = 901
r5 = 1
r6 = 2
r7 = 3
r8 = 4
r9 = 5
r10 = 6

# Enhancer
e3 = 9
e4 = 10
e5 = 11
e6 = 12
e7 = 13
e8 = 14
e9 = 15

# List files
dir_path = 'F:\\Development\\pcml\\training\\images_ext'
files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]

for filepath in files:

    try:
        # Attempt to open an image file
        image = Image.open(filepath)
    except (IOError):
        # Report error, and then skip to the next argument
        print("Problem opening", filepath, ":", IOError)
        continue

    # Mirror Image
    imageMirror = ImageOps.mirror(image)

    # transpose
    imgageTranspose = image.transpose(Image.TRANSPOSE)

    # gray

    imageGray = image.convert("L")

    # rotation

    imgr1 = image.rotate(25, resample=0, expand=0)
    imgr2 = image.rotate(50, resample=0, expand=0)
    imgr3 = image.rotate(75, resample=0, expand=0)
    imgr4 = image.rotate(100, resample=0, expand=0)
    imgr5 = image.rotate(120, resample=0, expand=0)
    imgr6 = image.rotate(140, resample=0, expand=0)
    imgr7 = image.rotate(160, resample=0, expand=0)
    imgr8 = image.rotate(45, resample=0, expand=0)
    imgr9 = image.rotate(75, resample=0, expand=0)
    imgr10 = image.rotate(60, resample=0, expand=0)

    # Enhancer
    enh = ImageEnhance.Contrast(image)
    enh3 = enh.enhance(1.3)  # enhancer3
    enh4 = enh.enhance(1.4)
    enh5 = enh.enhance(1.5)
    enh6 = enh.enhance(1.6)
    enh7 = enh.enhance(1.7)
    enh8 = enh.enhance(1.8)
    enh9 = enh.enhance(1.9)

    # Flip Image
    imageFlip = ImageOps.flip(image)
    # Flip left and right and rotate image
    mirror = image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90)
    sz = max(image.size + mirror.size)
    result = Image.new(image.mode, (sz, sz))
    result.paste(image, (0, 0) + image.size)

    # now paste the mirrored image, but with a triangular binary mask
    mask = Image.new('1', mirror.size)
    draw = ImageDraw.Draw(mask)
    draw.polygon([0, 0, 0, sz, sz, sz], outline='white', fill='white')
    result.paste(mirror, (0, 0) + mirror.size, mask)

    # clean up and save the result
    del mirror, mask, draw

    # Split our original filename into name and extension
    (name, extension) = os.path.splitext(filepath)

    # Save the thumbnail as "(original_name)_number.png"
    imageMirror.save(name + '_' + str(k) + '.png')
    imageFlip.save(name + '_' + str(s) + '.png')
    result.save(name + '_' + str(h) + '.png')
    imgageTranspose.save(name + '_' + str(t) + '.png')
    # imageGray.save(name + '_' + str(d) + '.png')
    # rotation
    imgr1.save(name + '_' + str(r1) + '.png')
    imgr2.save(name + '_' + str(r2) + '.png')
    imgr3.save(name + '_' + str(r3) + '.png')
    imgr4.save(name + '_' + str(r4) + '.png')
    imgr5.save(name + '_' + str(r5) + '.png')
    imgr6.save(name + '_' + str(r6) + '.png')
    imgr7.save(name + '_' + str(r7) + '.png')
    imgr8.save(name + '_' + str(r8) + '.png')
    imgr9.save(name + '_' + str(r9) + '.png')
    imgr10.save(name + '_' + str(r10) + '.png')
    # contrast
    enh3.save(name + '_' + str(e3) + '.png')
    enh4.save(name + '_' + str(e4) + '.png')
    enh5.save(name + '_' + str(e5) + '.png')
    enh6.save(name + '_' + str(e6) + '.png')
    enh7.save(name + '_' + str(e7) + '.png')
    enh8.save(name + '_' + str(e8) + '.png')
    enh9.save(name + '_' + str(e9) + '.png')

    # increment the number count to xrm
    k = k + 1
    s = s + 1
    h = h + 1
    d = d + 1
    t = t + 1
    r1 = r1 + 1
    r2 = r2 + 1
    r3 = r3 + 1
    r4 = r4 + 1
    r5 = r5 + 1
    r6 = r6 + 1
    r7 = r7 + 1
    r8 = r8 + 1
    r9 = r9 + 1
    r10 = r10 + 1
    e3 = e3 + 1
    e4 = e4 + 1
    e5 = e5 + 1
    e6 = e6 + 1
    e7 = e7 + 1
    e8 = e8 + 1
    e9 = e9 + 1


