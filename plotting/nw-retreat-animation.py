#/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import PIL
from PIL import ImageDraw
from PIL import ImageFont
import tempfile
import os

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.description = "Generating scripts for warming experiments."
parser.add_argument(
    "--rcp", dest="rcp", help="""RCP scenario. default=45.""", default=45
)
options = parser.parse_args()
rcp = options.rcp
rcp_dict = {"26": "RCP 2.6", "45": "RCP 4.5", "85": "RCP 8.5", "CTRL": "CTRL"}

print(rcp)
# max index to process
N = 335

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

def box(draw, pos, color):
    "Draw text 'rectangle', spanning pos =  [(x0,y0), (x1, y1)], using color 'color' (R,G,B)."
    draw.rectangle(pos, color)


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


def generate_frame(rcp, index, output_filename):
    "generate one frame and save it to the file output_filename"
    # load panels
    rcp_filename = "data/nw-600m/frame{:04d}.png".format(index + offset)

    ts_filename = "data/profiles/rcp{}_Upernavik_Isstrom_S_{:04d}.png".format(rcp, index + offset)

    rcp_img = PIL.Image.open(rcp_filename)

    # set the panel width
    # and the actual panel size
    panel_height = hd_res[1]
    panel_size = size_height(rcp_img.size, panel_height)
    panel_width = panel_size[0]
    # height of the header (white strip above the panels), in pixels
    header = 50
    # size of the border around the panels, in pixels
    border = 5

    # resize input images
    rcp_img = rcp_img.resize(panel_size, resample=1)

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

    # add text and box
    draw = PIL.ImageDraw.Draw(img)

    # paste individual panels into the output image
    img.paste(rcp, (hd_res[0] - panel_size[0], 0))
    img.paste(ts,  (0, 220))
    img.paste(overview, (hd_res[0] - overview.size[0], 0))
    img.paste(pism, (10, hd_res[1] - 100))
    img.paste(time, (400, hd_res[1] - 100), mask=time.split()[3])
    box(draw, [(1150, hd_res[1] - 100), (1150 + speed.size[0], hd_res[1] - 100 + speed.size[1])], (255, 255, 255))
    img.paste(speed, (1150, hd_res[1] - 100), mask=speed.split()[3])

    text(draw,
         "Year {:04d} CE".format(2008 + index + offset),
         200,
         40,
         (0, 0, 0))

    text(draw,
         "{}".format(rcp_dict[str(rcp)]),
         800,
         40,
         "#990002")

    text(draw,
         u"Upernavik Isstr\u00F8m S".format(2008 + index + offset),
         600,
         220,
         (0, 0, 0))
    
    img.save(output_filename)


def generate_nw_frame(rcp, index, output_filename):
    "generate one frame and save it to the file output_filename"
    # load panels
    rcp_filename = "data/nw-600m-rcp{}/frame{:04d}.png".format(rcp, index + offset)
    rcp_img = PIL.Image.open(rcp_filename)

    ts_filename = "data/discharge/d_contrib_discharge_contrib_rcp{}_{:04d}.png".format(rcp, index + offset)

    # set the panel width
    # and the actual panel size
    panel_height = hd_res[1]
    panel_size = size_height(rcp_img.size, panel_height)
    panel_width = panel_size[0]
    # height of the header (white strip above the panels), in pixels
    header = 50
    # size of the border around the panels, in pixels
    border = 5

    # resize input images
    rcp_img = rcp_img.resize(panel_size, resample=1)

    # open the ts plot
    ts = PIL.Image.open(ts_filename)
    ts = ts.resize(size(ts.size, hd_res[0] - panel_width - border), resample=1)
    ts_width = ts.size[0]
    ts_height = ts.size[1]

    # resize colorbars
    overview = overview_map.resize(size(overview_map.size, 150), resample=1)
    pism = pism_logo.resize(size(pism_logo.size, 240), resample=1)
    speed = speed_colorbar.resize(size(speed_colorbar.size, 440), resample=1)

    bar_height = speed.size[1]

    # set the size of the resulting image
    canvas_size = (hd_res[0], hd_res[1])
    img_width = canvas_size[0]

    # create the output image
    img = PIL.Image.new("RGB", canvas_size, color=(255, 255, 255))

    # add text and box
    draw = PIL.ImageDraw.Draw(img)

    # paste individual panels into the output image
    img.paste(rcp_img, (hd_res[0] - panel_width, 0))
    img.paste(ts,  (0, 400))
    img.paste(overview, (hd_res[0] - overview.size[0], 0))
    img.paste(pism, (10, hd_res[1] - 100))
    box(draw, [(400, hd_res[1] - 110), (400 + speed.size[0], hd_res[1] - 110 + speed.size[1])], (255, 255, 255))
    img.paste(speed, (400, hd_res[1] - 110), mask=speed.split()[3])

    text(draw,
         "Year {:04d} CE".format(2008 + index + offset),
         500,
         40,
         (0, 0, 0))
    text(draw,
         "{}".format(rcp_dict[str(rcp)]),
         100,
         40,
         "#990002")

    text(draw,
         "Contribution of outlet glaciers to discharge",
         430,
         380,
         "#000000")

    img.save(output_filename)


def generate_upernavik_frame(rcp, index, output_filename):
    "generate one frame and save it to the file output_filename"

    # height of the header (white strip above the panels), in pixels
    header = 50
    # size of the border around the panels, in pixels
    border = 5

    # load panels

    ts_filename = "data/profiles/rcp{}_Upernavik_Isstrom_S_{:04d}.png".format(rcp, index + offset)

    # open the ts plot
    ts = PIL.Image.open(ts_filename)

    # set the panel width
    # and the actual panel size
    panel_height = hd_res[1]
    panel_size = size_height(ts.size, panel_height)
    panel_width = panel_size[0]

    ts = ts.resize(size_height(ts.size, hd_res[1] - 200), resample=1)
    ts_width = ts.size[0]
    ts_height = ts.size[1]

    # resize colorbars
    overview = overview_map.resize(size(overview_map.size, 150), resample=1)
    pism = pism_logo.resize(size(pism_logo.size, 280), resample=1)

    # set the size of the resulting image
    canvas_size = (hd_res[0], hd_res[1])
    img_width = canvas_size[0]

    # create the output image
    img = PIL.Image.new("RGB", canvas_size, color=(255, 255, 255))

    # add text and box
    draw = PIL.ImageDraw.Draw(img)

    # paste individual panels into the output image
    img.paste(ts, (200, 0))
    img.paste(pism, (10, hd_res[1]))

    text(draw,
         "Year {:04d} CE".format(2008 + index + offset),
         200,
         40,
         (0, 0, 0))

    text(draw,
         "{}".format(rcp_dict[str(rcp)]),
         1700,
         40,
         "#990002")

    text(draw,
         u"Upernavik Isstr\u00F8m S".format(2008 + index + offset),
         900,
         100,
         (0, 0, 0))
    
    img.save(output_filename)


for k in range(401):
    print('Generating frame {}'.format(k))
    generate_nw_frame(rcp, k, "output/nw_g600m_rcp{}_{:04d}.png".format(rcp, k))
    generate_upernavik_frame(rcp, k, "output/upernavik_g600m_rcp{}_{:04d}.png".format(rcp, k))
    #generate_frame(rcp, k, "output/nw_g600m_rcp_%04d.png" % k)

print("")
