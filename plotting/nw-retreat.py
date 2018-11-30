import PIL
from PIL import ImageDraw
from PIL import ImageFont
import tempfile
import os

# HD resolution: 1920x1080
hd_res = [1920, 1080]

font = ImageFont.truetype("/Library/Fonts/Arial Bold.ttf", 40)

pism_logo = PIL.Image.open("data/colorbars/pism-logo.png")
speed_colorbar = PIL.Image.open("data/colorbars/speed_blue_red_nonlin_0_1500_horizontal.png")
time_colorbar = PIL.Image.open("data/colorbars/bath_112_horizontal.png")
overview_map = PIL.Image.open("data/colorbars/overview.png")

# Start Year 2015
offset = 7

def text(draw, text, x, y, color):
    "Draw text 'text', centered at (x,y), using color 'color' (R,G,B)."
    size = draw.textsize(text, font=font)
    draw.text((x - size[0] / 2, y - size[1] / 2),
              text,
              color, font=font)


def size(old_size, desired_width):
    "Compute the new size of an image given its old size and the desired width. Preserves the aspect ratio."
    ratio = float(desired_width) / old_size[0]
    # and the actual panel size
    return int(old_size[0] * ratio), int(old_size[1] * ratio)

def size_height(old_size, desired_height):
    "Compute the new size of an image given its old size and the desired height. Preserves the aspect ratio."
    ratio = float(desired_height) / old_size[1]
    # and the actual panel size
    return int(old_size[0] * ratio), int(old_size[1] * ratio)


def generate_frame(index, output_filename):
    "generate one frame and save it to the file output_filename"
    # load panels
    rcp85_filename = "data/nw-600m/frame{:04d}.png".format(index + offset)

    ts_filename = "data/profiles/rcp85_Upernavik_Isstrom_S_{:04d}.png".format(index + offset)

    rcp85 = PIL.Image.open(rcp85_filename)

    # set the panel width
    # and the actual panel size
    panel_height = hd_res[1]
    panel_size = size_height(rcp85.size, panel_height)
    panel_width = panel_size[0]
    # height of the header (white strip above the panels), in pixels
    header = 50
    # size of the border around the panels, in pixels
    border = 5

    # resize input images
    rcp85 = rcp85.resize(panel_size, resample=1)

    # open the ts plot
    ts = PIL.Image.open(ts_filename)
    ts = ts.resize(size(ts.size, hd_res[0] - panel_width - border), resample=1)
    ts_width = ts.size[0]
    ts_height = ts.size[1]

    # resize colorbars
    overview = overview_map.resize(size(overview_map.size, 150), resample=1)
    pism = pism_logo.resize(size(pism_logo.size, 280), resample=1)
    speed = speed_colorbar.resize(size(time_colorbar.size, 500), resample=1)
    time = time_colorbar.resize(size(time_colorbar.size, 500), resample=1)

    bar_height = time.size[1]

    # set the size of the resulting image
    canvas_size = (hd_res[0], hd_res[1])
    img_width = canvas_size[0]

    # create the output image
    img = PIL.Image.new("RGB", canvas_size, color=(255, 255, 255))

    # paste individual panels into the output image
    img.paste(rcp85, (hd_res[0] - panel_size[0], 0))
    img.paste(ts,  (0, 260))
    img.paste(overview, (hd_res[0] - overview.size[0], 0))
    img.paste(pism, (10, hd_res[1] - 100))
    img.paste(time, (400, hd_res[1] - 100), mask=time.split()[3])
    img.paste(speed, (1150, hd_res[1] - 100), mask=speed.split()[3])

    # add text
    draw = PIL.ImageDraw.Draw(img)

    text(draw,
         "Year {:04d} CE".format(2008 + index + offset),
         200,
         40,
         (0, 0, 0))

    text(draw,
         u"Upernavik Isstr\u00F8m S".format(2008 + index + offset),
         600,
         250,
         (0, 0, 0))
    
    img.save(output_filename)


# max index to process
N = 300

for k in range(N):
    print('Generating frame {}'.format(k))
    generate_frame(k, "output/nw_g600m_rcp85_%04d.png" % k)

print("")
