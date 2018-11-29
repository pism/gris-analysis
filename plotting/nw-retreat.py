import PIL
from PIL import ImageDraw
from PIL import ImageFont
import tempfile
import os

# HD resolution: 1920x1080

font = ImageFont.truetype("/Library/Fonts/Arial Bold.ttf", 40)

time_colorbar = PIL.Image.open("data/colorbars/bath_112_horizontal.png")
speed_colorbar = PIL.Image.open("data/colorbars/speed_blue_red_nonlin_0_1500_horizontal.png")

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


def generate_frame(index, output_filename):
    "generate one frame and save it to the file output_filename"
    # load panels
    rcp85_filename = "data/nw-600m/frame{:04d}.png".format(index + offset)

    ts_filename = "data/profiles/rcp85_Upernavik_Isstrom_S_{:04d}.png".format(index + offset)

    rcp85 = PIL.Image.open(rcp85_filename)

    # set the panel width
    panel_width = 400
    # and the actual panel size
    panel_size = size(rcp85.size, panel_width)
    panel_height = panel_size[1]

    # height of the header (white strip above the panels), in pixels
    header = 50
    # size of the border around the panels, in pixels
    border = 5

    # resize input images
    rcp85 = rcp85.resize(panel_size, resample=1)

    # open the ts plot
    ts = PIL.Image.open(ts_filename)
    ts = ts.resize(size(ts.size, panel_width * 3), resample=1)
    ts_height = ts.size[1]

    # resize colorbars
    time = time_colorbar.resize(size(time_colorbar.size, panel_width), resample=1)
    speed = speed_colorbar.resize(size(time_colorbar.size, panel_width), resample=1)

    bar_height = time.size[1]

    # set the size of the resulting image
    canvas_size = (panel_width * 3 + 4 * border,
                   header + panel_height + bar_height + ts_height + 2)
    img_width = canvas_size[0]

    # create the output image
    img = PIL.Image.new("RGB", canvas_size, color=(255, 255, 255))

    # paste individual panels into the output image
    img.paste(rcp85, (border, header))
    img.paste(time, (border, header + panel_height + border), mask=time.split()[3])
    img.paste(speed, (3*border + 2*panel_width, header + panel_height + border), mask=speed.split()[3])

    img.paste(ts, (int(img_width / 2 - panel_width * 3 / 2),
                   int(header + panel_height + bar_height)))

    # add text
    draw = PIL.ImageDraw.Draw(img)

    text(draw,
         "Year %04d CE" % (2008 + index + offset),
         img_width / 2,
         header + panel_height + border + bar_height / 2,
         (0, 0, 0))

    img.save(output_filename)


# max index to process
N = 300

for k in range(N):
    print('Generating frame {}'.format(k))
    generate_frame(k, "output/nw_g600m_rcp85_%04d.png" % k)

print("")
