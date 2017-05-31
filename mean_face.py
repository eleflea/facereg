import os

from PIL import Image

WIDTH = 90
HEIGHT = 120


def resize_img(direct):
    files = os.listdir(direct)
    for file in files:
        img = Image.open(os.path.join(direct, file))
        img = img.resize((WIDTH, HEIGHT))
        img.save(os.path.join(direct, file))


def mean_face(direct):
    def mean(*kw):
        return sum(kw) / len(kw)

    img_map = []
    files = os.listdir(direct)
    for file in files:
        img = Image.open(os.path.join(direct, file))
        img = img.convert('L')
        img_map.append(list(img.getdata()))
    face = list(map(mean, *img_map))
    mean_img = Image.new('L', (WIDTH, HEIGHT))
    mean_img.putdata(face)
    mean_img.show()
    return face


if __name__ == '__main__':
    # resize_img("G:\\T_T\\sth\\face\\mean_face")
    mean_face("G:\\T_T\\sth\\face\\mean_face")
