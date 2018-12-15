import PIL
from PIL import ImageDraw
from PIL import ImageFont
import tempfile
import os

# HD resolution: 1920x1080

font = ImageFont.truetype("/Library/Fonts/Arial Bold.ttf", 40)

topo_colorbar = PIL.Image.open("data/colorbars/greenland-topography_horizontal.png")
speed_colorbar = PIL.Image.open("data/colorbars/speed_blue_red_nonlin_0_1500_horizontal.png")


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


def generate_frame(index, output_filename):
    "generate one frame and save it to the file output_filename"
    # load panels
    rcp26_filename = "data/rcp26/frame%04d.png" % index
    rcp45_filename = "data/rcp45/frame%04d.png" % index
    rcp85_filename = "data/rcp85/frame%04d.png" % index

    ts_filename = "data/ts_plots/dgmsl_limnsw_%04d.png" % index

    rcp26 = PIL.Image.open(rcp26_filename)
    rcp45 = PIL.Image.open(rcp45_filename)
    rcp85 = PIL.Image.open(rcp85_filename)

    # set the panel width
    panel_width = 400
    # and the actual panel size
    panel_size = size(rcp26.size, panel_width)
    panel_height = panel_size[1]

    # height of the header (white strip above the panels), in pixels
    header = 50
    # size of the border around the panels, in pixels
    border = 5

    # resize input images
    rcp26 = rcp26.resize(panel_size, resample=1)
    rcp45 = rcp45.resize(panel_size, resample=1)
    rcp85 = rcp85.resize(panel_size, resample=1)

    # open the ts plot
    ts = PIL.Image.open(ts_filename)
    ts = ts.resize(size(ts.size, panel_width * 3), resample=1)
    ts_height = ts.size[1]

    # resize colorbars
    topo = topo_colorbar.resize(size(topo_colorbar.size, panel_width), resample=1)
    speed = speed_colorbar.resize(size(topo_colorbar.size, panel_width), resample=1)

    bar_height = topo.size[1]

    # set the size of the resulting image
    canvas_size = (panel_width * 3 + 4 * border,
                   header + panel_height + bar_height + ts_height + 2)
    img_width = canvas_size[0]

    # create the output image
    img = PIL.Image.new("RGB", canvas_size, color=(255, 255, 255))

    # paste individual panels into the output image
    img.paste(rcp26, (border, header))
    img.paste(topo, (border, header + panel_height + border), mask=topo.split()[3])
    img.paste(rcp45, (2*border + panel_width, header))
    img.paste(rcp85, (3*border + 2*panel_width, header))
    img.paste(speed, (3*border + 2*panel_width, header + panel_height + border), mask=speed.split()[3])

    img.paste(ts, (int(img_width / 2 - panel_width * 3 / 2),
                   int(header + panel_height + bar_height)))

    # add text
    draw = PIL.ImageDraw.Draw(img)

    text(draw,
         "RCP 2.6",
         img_width / 6,
         header / 2,
         '#003466')

    text(draw,
         "RCP 4.5",
         img_width / 2,
         header / 2,
         '#5492CD')

    text(draw,
         "RCP 8.5",
         5 * img_width / 6,
         header / 2,
         '#990002')

    text(draw,
         "Year %04d CE" % (2008 + index),
         img_width / 2,
         header + panel_height + border + bar_height / 2,
         (0, 0, 0))

    img.save(output_filename)


# max index to process
N = 1000

for k in range(N):
    print('Generating frame {}'.format(k))
    generate_frame(k, "output/gris_g900m_rcps_%04d.png" % k)

print("")
