import json
import argparse
import os.path
from PIL import Image, ImageDraw, ImageOps








if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='render a NailedIt json file')
    parser.add_argument('filename', help='NailedIt json file')
    parser.add_argument('-res', '--resolution', metavar=("X","Y"), type=int, nargs=2, default=[1920,1080], help='output resolution in the format Width Height, e.g. -res 1920 1080')
    parser.add_argument('-max', type=int, help="stop after max lines")
    parser.add_argument('-out', help="specify an output filename, defaults to filename_out.png")
    parser.add_argument('-mode', choices=("final", "incremental"), default="final", help="rendering mode. final outputs the end result, while incremental saves a file after every nth line")
    parser.add_argument('-se', '--save_every', metavar="n", type=int, default=10, help="in incremental mode saves a file every Nth line")
    parser.add_argument('--show', action="store_true", help="display the final image")
    parser.add_argument('-oversample', metavar="times", type=int, default=1, choices=(1, 2, 3, 4, 5), help="image oversampling factor")
    parser.add_argument('-wo', '--widthoverride', metavar="new_width", type=int, help="thread width override (width before oversampling)" )
    parser.add_argument('-trail', '--trail_length', metavar="len", type=int, default=0, help="draw a trail in incremental mode if len > 0")
    parser.add_argument('-fo', '--force_invert', action="store_true", help="invert image")
    args = parser.parse_args()

    # open the file
    infile = open(args.filename, "r")
    js = json.load(infile)
    infile.close()

    outpath = args.out
    if not outpath:
        sp = os.path.splitext(args.filename)
        outpath = sp[0]+"_out.png"

    params = js["2:parameters:"]
    nails  = js["3:nails"]
    thread = js["4:thread"]

    if args.force_invert:
        params["img_invert"] = 1

    img_w = args.resolution[0] * args.oversample
    img_h = args.resolution[1] * args.oversample
    sf = (img_w / float(params["proc_width"]), img_h / float(params["proc_height"]))

    background_color = (params["backgroundColor"],)*3
    thread_color = (params["threadColor"][0],)*3
    thread_width = args.oversample if not args.widthoverride else args.widthoverride
    trail_len = args.trail_length
    trail_color = (255,0,0)
    if "img_invert" in params and params["img_invert"] > 0:
        trail_color = (255-trail_color[0], 255-trail_color[1], 255-trail_color[2])

    if args.max:
        thread = thread[:args.max]
    section_size = args.save_every if (args.mode == 'incremental' and args.save_every) else len(thread)


    image = Image.new("RGB", (img_w, img_h))
    draw = ImageDraw.Draw(image)
    imgno=0
    while 1:
        ln = [nails[pid] for pid in thread[:section_size*(imgno+1)]]
        ln = [(l[0]*sf[0], l[1]*sf[1]) for l in ln]     # scale up to image size
        #ln = [item for sublist in ln for item in sublist]

        image.paste(background_color, (0,0)+image.size)
        #draw.line(ln, fill=thread_color, width=thread_width)
        last = ln[0]
        for i, p in enumerate(ln[1:]):
            col = thread_color
            if trail_len and i > len(ln)-trail_len and len(ln) < len(thread):
                f = float(i - (len(ln) - trail_len))/trail_len
                fi = 1.0-f
                col = ( int(col[0]*fi+trail_color[0]*f), int(col[1]*fi+trail_color[1]*f), int(col[2]*fi+trail_color[2]*f) )
            draw.line((last[0], last[1], p[0], p[1]), width=thread_width, fill=col)
            last = p

        if args.oversample > 1:
            out_image = image.resize((args.resolution[0], args.resolution[1]), resample=Image.BICUBIC)
        else:
            out_image = image

        if "img_invert" in params and params["img_invert"] > 0:
            out_image = ImageOps.invert(out_image)

        if args.mode == 'incremental':
            sp = os.path.splitext(outpath)
            out = "{:}.{:04d}.png".format(sp[0], imgno)
        else:
            out = outpath

        out_image.save(out)
        print "saved", out
        print "lines", len(ln)

        if section_size*(imgno+1) >= len(thread):
            break
        imgno += 1


    if args.show:
        out_image.show()
