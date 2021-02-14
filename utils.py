import skimage#数字图片处理包
import skimage.io
import skimage.transform
import numpy as np


# synset = [l.strip() for l in open('synset.txt').readlines()]


# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path):#对于路径中的图片像素归一化并截取中心最大方块无变形地变换为固定尺寸的方块。
    # load image
    img = skimage.io.imread(path) #读取路径中图片
    img = img / 255.0 #像素值归一化
    assert (0 <= img).all() and (img <= 1.0).all() #assert确保其后的语句为真，若为假则抛出异常
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2]) #找出图片长宽中的最小值，0位是纵向高度，1位是横向宽度
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]#以长宽中的短边为边长，将图片裁剪为正方形
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))#将裁剪好的正方形变换为其他尺寸的正方形，实现无变形的变换
    return resized_img


# returns the top1 string
def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]#读取文件中每一行字符串并去除头尾指定的字符，比如\n

    # print prob
    pred = np.argsort(prob)[::-1]#将数组prob中的一组概率值的索引按照其对应数字从大到小排列

    # Get top1 label
    top1 = synset[pred[0]]#按照最大概率值对应的索引号寻找对应的名称
    print(("Top1: ", top1, prob[pred[0]]))
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print(("Top5: ", top5))
    return top1


def load_image2(path, height=None, width=None):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    if height is not None and width is not None:
        ny = height
        nx = width
    elif height is not None:
        ny = height
        nx = img.shape[1] * ny / img.shape[0]
    elif width is not None:
        nx = width
        ny = img.shape[0] * nx / img.shape[1]
    else:
        ny = img.shape[0]
        nx = img.shape[1]
    return skimage.transform.resize(img, (ny, nx))


def test():
    img = skimage.io.imread("./others/hair.jpg")
    ny = 300
    nx = img.shape[1] * ny / img.shape[0]
    img = skimage.transform.resize(img, (ny, nx))
    img=(img*255).astype(np.uint8)
    skimage.io.imsave("./test_data/test/output.jpg", img)


if __name__ == "__main__":
    test()
