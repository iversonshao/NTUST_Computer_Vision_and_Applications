import cv2
import numpy as np

#create a mask by cropping the original image with the given points
def cropimg(src_ori, point):
    mask = np.zeros((src_ori.shape), dtype=np.uint8)
    cv2.fillPoly(mask, [point.astype(np.int32)], (255,)*src_ori.shape[2])
    return cv2.bitwise_and(src_ori, mask), mask

#calculate the homography matrix
def cal_homography_matrix(src, dst):
    homo_matrix, _ = cv2.findHomography(src, dst)
    return homo_matrix

#1to2 2to1 use homo matrix
def swap(ori, p1, p2, homo_matrix):
    ori_shape = (ori.shape[1], ori.shape[0])
    #x' = Hx x = H^-1x'
    swap_p1 = cv2.warpPerspective(p1, homo_matrix, ori_shape)
    swap_p2 = cv2.warpPerspective(p2, np.linalg.inv(homo_matrix), ori_shape)
    return swap_p1, swap_p2

#fill the masked area
def fill(ori, homo_matrix, p1, p2, mask1, mask2):

    background = cv2.bitwise_and(ori, cv2.bitwise_not(cv2.bitwise_or(mask1, mask2)))
    swap_p1, swap_p2 = swap(ori, p1, p2, homo_matrix)
    swap_img = cv2.bitwise_or(swap_p1, swap_p2)
    fill_img = cv2.bitwise_or(background, swap_img)
    return fill_img

if __name__ == '__main__':
    ori_img = cv2.imread('Homework#2/Swap_ArtGallery.jpg')
    ori_img = cv2.resize(ori_img,(960,540))
    cv2.imshow('ori_img', ori_img)
    cv2.waitKey(0)

    woman = np.array([[110,20],[114,416], [393,367], [397,54]], dtype=np.float32)
    cat = np.array([[614,73],[610,196],[718,200],[723,68]], dtype=np.float32)

    hm = cal_homography_matrix(woman, cat)
    imageA, maskA = cropimg(ori_img, woman)
    imageB, maskB = cropimg(ori_img, cat)
    result = fill(ori_img, hm, imageA, imageB, maskA, maskB)
    cv2.imshow('result',result)
    cv2.waitKey(0)
    cv2.imwrite("M11215075.jpg", result)
    cv2.destroyAllWindows()
