import PIL
from PIL import Image


class ImageConsumer:

    class Cell:
        def __init__(self, parent):
            self._parent = parent
            self._annotation = ""
            self._image = None

        def __enter__(self):
            return self
        def __exit__(self, excpt_type, excpt_val, excpt_tb):
            self._parent._entries.append(self)

        def annotate(self, text):
            self._annotation = text

        def image(self, image):
            self._image = image

        def himage_list(self, img_list, pad_x=1):
            if not img_list:
                return
            w,h = img_list[0].size
            tub = Image.new("RGB", ((w+pad_x) * len(img_list), h))
            for i, img in enumerate(img_list):
                tub.paste(img, ((w+pad_x)*i, 0))
            self.image(tub)

    def __init__(self):
        self._entries = []

    def step(self):
        return ImageConsumer.Cell(self)

    def as_html(self, file_name):
        with open(file_name, "w") as fw:
            fw.write(
                """<html>
                <div>""")

            for i, cell in enumerate(self._entries):
                img_f_name = f"img_{i:03}.png"
                cell._image.save(img_f_name)
                fw.write(f"<div>{cell._annotation}</div><img src='{img_f_name}'/>\n")

            fw.write(
                """</div>
                </html>"""
            )