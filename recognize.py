import math

from PIL import Image

HEIGHT = 64
WIDTH = 64
DEAL_MAP = [(1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1),
            (0, 1), (1, 1), (1, 0)]


def reader(path):
    im = Image.open(path)
    im = im.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
    im = im.convert('L')
    return im


def map_lbp(im):
    lbp_list = []
    for x in range(1, im.size[0] - 1):
        for y in range(1, im.size[1] - 1):
            center_value = im.getpixel((x, y))
            local_value = [im.getpixel((x + d[0], y + d[1])) for d in DEAL_MAP]
            lbp_value = 0
            for pos in range(len(DEAL_MAP)):
                lbp_value += (center_value > local_value[pos]) << (7 - pos)
            lbp_list.append(lbp_value)

    return lbp_list


def show_lbp_map(lbp_list):
    im_lbp = Image.new('L', (WIDTH - 2, HEIGHT - 2))
    im_lbp.putdata(lbp_list)
    im_lbp.show()


def isdirected(binary):
    binary = '0' * (10 - len(binary)) + binary[2:]
    time = 0
    char = binary[0]
    for c in binary[1:]:
        if char != c:
            char = c
            time += 1
    if time <= 2:
        return True
    else:
        return False


def vector_histogram(v):
    other = 0
    vector_hist = [-1 for _ in range(256)]
    for num in range(256):
        if isdirected(bin(num)):
            vector_hist[num] = 0
    for num in v:
        if isdirected(bin(num)):
            vector_hist[num] += 1
        else:
            other += 1
    vector_hist = [v for v in vector_hist if v != -1]
    vector_hist.append(other)
    return vector_hist


def vector_cos(v1, v2):
    dot = sum(map(lambda x, y: x * y, v1, v2))
    sq_v1 = sum(map(lambda x: x**2, v1))
    sq_v2 = sum(map(lambda x: x**2, v2))
    return dot / (math.sqrt(sq_v1) * math.sqrt(sq_v2))


def distance(v1, v2):
    return math.sqrt(sum(map(lambda x, y: (x - y)**2, v1, v2)))


def chi_dist(v1, v2):
    def di(x, y):
        if x == y:
            return 0
        return (x - y)**2 / (x + y)
    return sum(map(di, v1, v2))


def im_grid(im, w, h):
    im_parts = []
    for wi in range(w):
        for hi in range(h):
            im_parts.append(im.crop(
                (wi * im.size[0] / w, hi * im.size[1] / h, (wi + 1) * im.size[0] / w, (hi + 1) * im.size[0] / h)))

    return im_parts


def compare(v, path, fun=vector_cos):
    face = Image.new('L', (WIDTH, HEIGHT))
    face.putdata(v)
    parts = im_grid(face, 8, 8)
    v1 = []
    for p in parts:
        v1 += vector_histogram(map_lbp(p))
    img = reader(path)
    parts = im_grid(img, 8, 8)
    v2 = []
    for p in parts:
        v2 += vector_histogram(map_lbp(p))

    return fun(v1, v2)


if __name__ == '__main__':
    img = reader('G:\\T_T\\sth\\face\\xl2.jpg')
    parts = im_grid(img, 4, 4)
    v1 = []
    for p in parts:
        v1 += vector_histogram(map_lbp(p))
    lbp_list = map_lbp(img)
    show_lbp_map(lbp_list)
    im_lbp = Image.new('L', (WIDTH - 2, HEIGHT - 2))
    im_lbp.putdata(lbp_list)
    lbp_list = map_lbp(im_lbp)
    im_lbp = Image.new('L', (WIDTH - 4, HEIGHT - 4))
    im_lbp.putdata(lbp_list)
    im_lbp.show()
    # v1 = vector_histogram(lbp_list)
    img2 = reader('G:\\T_T\\sth\\face\\xgq3.jpg')
    parts = im_grid(img2, 4, 4)
    v2 = []
    for p in parts:
        v2 += vector_histogram(map_lbp(p))
    # lbp_list2 = map_lbp(img2)
    # v2 = vector_histogram(lbp_list2)
    # print(v1, v2)
    print(vector_cos(v1, v2))
    print(distance(v1, v2))
    # print(vector_cos(lbp_list, lbp_list2))
    # img.show()
    # img2.show()
    # show_lbp_map(lbp_list)
    # show_lbp_map(lbp_list2)
