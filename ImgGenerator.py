import itertools
import math
import os
from random import Random
import json

from PIL import Image, ImageDraw
import numpy

PIXEL_COLOR = 0
EXT = "png"
DefaultRenderDir = os.path.join(".", "content")


class RandomImageOp:


    def __init__(self, width, height, prefix="rnd"):
        self._image = Image.new("L", (width, height), 255)
        self._prefix = prefix

    def save(self, uid, ext="bmp"):
        self._image.save(uid, ext)

    @property
    def file_name(self):
        return self._prefix

    def _reducer(self, normalize, generate_ax):
        triggers = []
        row_sum = 0
        last_colour = None
        over_range = generate_ax() if callable(generate_ax) else generate_ax
        for x, y in over_range:
            pix = self._image.getpixel((x, y))
            if pix == PIXEL_COLOR:
                if pix == last_colour:
                    triggers[-1] += 1 #triggers[-1] = triggers[-1] + 1
                else:
                    triggers.append(1)
                row_sum += 1
            last_colour = pix
        if normalize:
            row_max = max(triggers, default=1)
            triggers = [x / row_max for x in triggers]
        return triggers

    def top_reduce(self, normalize=False):
        reduce = []
        max_len = 0
        for x in range(0, self._image.width):
            triggers = self._reducer(normalize,
                                     zip(itertools.repeat(x), range(0, self._image.height))
                                     )
            reduce.append(triggers)
            max_len = max(len(triggers), max_len)
        return {"reduce": reduce, "max_len": max_len}

    def left_reduce(self, normalize=False):
        reduce = []
        max_len = 0
        for y in range(0, self._image.height):
            triggers = self._reducer(normalize,
                                     zip(range(0, self._image.width), itertools.repeat(y))
                                     )
            reduce.append(triggers)
            max_len = max(len(triggers), max_len)
        return {"reduce": reduce, "max_len": max_len}

    @property
    def image(self):
        return self._image

    @staticmethod
    def _gen_rnd(width, height, prefix, method):
        r = RandomImageOp(width, height, prefix)
        # scan by row
        for y in range(0, height):
            for x in range(0, width):
                off = int(round(method()))
                if 0 <= off < width:
                    r._image.putpixel((off, y), PIXEL_COLOR)
        return r

    @staticmethod
    def normal_rnd(width, height):
        rnd = Random()
        return RandomImageOp._gen_rnd(width, height, "gaus", lambda: rnd.normalvariate(width / 2.0, width / 4.0))

    @staticmethod
    def log_rnd(width, height):
        rnd = Random()
        return RandomImageOp._gen_rnd(width, height, "log",
                                      lambda: rnd.lognormvariate(math.log(width / 2.0), math.log(width / 4.0)))

    @staticmethod
    def weibull_rnd(width, height):
        rnd = Random()
        return RandomImageOp._gen_rnd(width, height, "weibul",
                                      lambda: rnd.weibullvariate(width, math.log(width)))

    @staticmethod
    def linear_rnd(width, height):
        rnd = Random()
        image = RandomImageOp(width, height, "lin")
        draw = ImageDraw.Draw(image._image)
        for i in range(1, int(math.sqrt(width * height) / 1.3)):
            draw.line((rnd.randint(0, width), rnd.randint(0, height), rnd.randint(0, width), rnd.randint(0, height)),
                      fill=0)
        del draw
        return image
    arc_degrees = [0, 30, 60, 90, 57, 120, 150, 320, 210]
    @staticmethod
    def fig_rnd(width, height):
        rnd = Random()
        image = RandomImageOp(width, height, "lin")
        draw = ImageDraw.Draw(image._image)
        for i in range(1, int(math.sqrt(width * height) / 2)):
            #draw.line((rnd.randint(0, width), rnd.randint(0, height), rnd.randint(0, width), rnd.randint(0, height)),
            #          fill=0)
            start = rnd.choice(RandomImageOp.arc_degrees)
            end = rnd.choice(RandomImageOp.arc_degrees)
            draw.pieslice(
                (rnd.randint(0, width), rnd.randint(0, height), rnd.randint(0, width), rnd.randint(0, height)),
                start=start, end=end,
                fill=0
            )
        del draw
        return image

    @staticmethod
    def from_numpy(arr):
        assert len(arr.shape) == 3

        for i in enumerate:
            it = numpy.nditer(arr, flags=['multi_index'])
            while not it.finished:
                print("%d <%s>" % (it[0], it.multi_index), end=' ')
                it.iternext()


def rand_image(width, height, seed=13):
    rnd = Random(seed)
    align_w, align_h = width // 8 or 1, height
    while True:
        data = bytearray(align_w * align_h)
        for i in range(0, align_h):
            for j in range(0, align_w):
                data[i * align_w + j] = rnd.randint(0, 255)
        yield Image.frombuffer("L", (align_w, height), bytes(data), "raw")


class HTMLTemplate:

    def __init__(self, fname="rando.js"):
        self._fname = fname
        self._store = None

    def __enter__(self):
        dest = os.path.join(DefaultRenderDir, self._fname)
        self._store = open(dest, "w")
        self._store.write("var data=[\n")
        return self

    def __exit__(self, type, value, tb):
        self._store.write("\n];")
        self._store.close()

    def save(self, img, uid):
        img_fname = os.path.join(DefaultRenderDir, "{0}_{1}.{2}".format(img.file_name, uid, EXT))
        img.save(img_fname, EXT)
        outdata = {"image": img_fname,
                   "top_reduce": img.top_reduce(),
                   "left_reduce": img.left_reduce()
                   }

        self._store.write(json.dumps(outdata))
        self._store.write(", ")


W = 128
H = 128
PREDEFINED_IMG_RANDOMS = (
#        RandomImageOp.normal_rnd,
#        RandomImageOp.weibull_rnd,
#       RandomImageOp.log_rnd,
#    RandomImageOp.linear_rnd,
    RandomImageOp.fig_rnd,
)


def main():
    # vv = rand_image(W, H)
    output = HTMLTemplate()
    with output:
        for i in range(0, 10):
            filename = "_{0}.{1}".format(i, EXT)
            print("Generating:" + filename)
            # img = next(vv)
            # img.save(filename, "bmp")
            for renderer in PREDEFINED_IMG_RANDOMS:
                output.save(renderer(W, H), i)
    print("Open file: '{0}' to display result".format(os.path.join(DefaultRenderDir, "rando.html")))

def produce_numpy_sample(w=W, h=H):
    def make_numpy_arr(reduce_dict, transpose):
        # arrays from reduce may be non-complete, pad each left with zeros (align to right)
        reduce = reduce_dict["reduce"]
        max_dim = w // 2 if transpose else h // 2  # reduce_dict["max_len"]
        result = numpy.asarray([[0] * (max_dim - len(lit)) + lit for lit in reduce], numpy.float32)
        if transpose:
            result.transpose()
        return result

    while True:
        for img_factory in PREDEFINED_IMG_RANDOMS:
            img = img_factory(W, H)
            top_d = img.top_reduce(True)
            left_d = img.left_reduce(True)
            yield (make_numpy_arr(top_d, False),
                   make_numpy_arr(left_d, True),
                   numpy.asarray(img.image).astype('float32') )


def gen_numpy_2chanel_sample(w=W, h=H):
    def make_numpy_arr(reduce_dict, transpose):
        # arrays from reduce may be non-complete, pad each left with zeros (align to right)
        reduce = reduce_dict["reduce"]
        max_dim = w // 2 if transpose else h // 2  # reduce_dict["max_len"]
        result = numpy.asarray([[0] * (max_dim - len(lit)) + lit for lit in reduce], numpy.float32)
        if transpose:
            result.transpose()
        return result

    while True:
        for img_factory in PREDEFINED_IMG_RANDOMS:
            img = img_factory(w, h)
            top_d = img.top_reduce(False)
            left_d = img.left_reduce(False)
            yield numpy.stack([
                make_numpy_arr(top_d, True),
                make_numpy_arr(left_d, False)
                ]).reshape((h, w//2, 2)), numpy.asarray(img.image) #.astype('float32') / 255.


if __name__ == "__main__":
    main()
