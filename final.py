import matplotlib.pyplot as plt
import matplotlib.image as mp_img
import numpy as np
import cv2
import math
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)   
    
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, a=0.8, b=1., y=0.):
    return cv2.addWeighted(initial_img, a, img, b, y)

def video_convert(input_name, channel, out_suf = ''):
    white_output = 'test_videos_output' + out_suf + '/' + input_name + '.mp4'
    clip1 = VideoFileClip("test_videos/" + input_name + ".mp4")
    white_clip = clip1.fl_image(channel)
    white_clip.write_videofile(white_output, audio=False)

def select_white_yellow(img):
    converted = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # white color mask
    lower = np.uint8([  0, 200,   0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted, lower, upper)
    # yellow color mask
    lower = np.uint8([ 10,   0, 100])
    upper = np.uint8([ 40, 255, 255])
    yellow_mask = cv2.inRange(converted, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(img, img, mask = mask)

def eighth_channel (img):
    white_yellow_image = select_white_yellow(img)
    gray_image = grayscale(white_yellow_image) #applying grayscale transform
    gauss_image = gaussian_blur(gray_image, 15) #use gaussian blur to make it easier for canny to detect edges
    canny_image = canny(gauss_image, 50, 150) #use canny transform for finding edges of image
    interest_reg = np.array([[[img.shape[1]/2, img.shape[0]/2], [img.shape[1], img.shape[0]], [0, img.shape[0]]]], dtype=np.int32)
    interest_reg_1 = np.array([[[img.shape[1]/2, img.shape[0]/2+25], [img.shape[1]-5, img.shape[0]], [5, img.shape[0]]]], dtype=np.int32)
    region_image_0 = region_of_interest(canny_image, interest_reg_1) #set rest of the picture to black
    lines_image = hough_lines(region_image_0, 1, np.pi/180, 20, 20, 300) #use hough_lines for getting an image of lines
    lined_image = weighted_img(lines_image, img) #adding transparent lines image to initial image
    return lined_image #showing the output

#video_convert('solidWhiteRight', eighth_channel, "_eighth")
#video_convert('solidYellowLeft', eighth_channel, "_eighth")
#video_convert('challenge', eighth_channel, "_eighth")

###other good methods that can be used
###rest of file can be commented

def ninth_channel (img):
    white_yellow_image = select_white_yellow(img)
    gray_image = grayscale(white_yellow_image) #applying grayscale transform
    interest_reg = np.array([[[img.shape[1]/2, img.shape[0]/2], [img.shape[1], img.shape[0]], [0, img.shape[0]]]], dtype=np.int32)
    region_image_0 = region_of_interest(gray_image, interest_reg) #set rest of the picture to black
    canny_image = canny(region_image_0, 100, 110) #use canny transform for finding edges of image
    lines_image = hough_lines(canny_image, 1, np.pi/180, 90, 1, 100) #use hough_lines for getting an image of lines
    interest_reg_1 = np.array([[[img.shape[1]/2, img.shape[0]/2+25], [img.shape[1]-5, img.shape[0]], [5, img.shape[0]]]], dtype=np.int32)
    region_image_1 = region_of_interest(lines_image, interest_reg_1) #set rest of the picture to black
    lined_image = weighted_img(region_image_1, img) #adding transparent lines image to initial image
    return lined_image #showing the output

#video_convert('solidWhiteRight', ninth_channel, "_ninth")
#video_convert('solidYellowLeft', ninth_channel, "_ninth")
#video_convert('challenge', ninth_channel, "_ninth")

def white_hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines, [255, 255, 255])
    return line_img

def video_convert_fl(input_name, channel, out_suf = '', is_sub = False, sub_time = 5):
    white_output = 'test_videos_output' + out_suf + '/' + input_name + '.mp4'
    clip1 = VideoFileClip("test_videos/" + input_name + ".mp4")
    if is_sub:
        clip1 = clip1.subclip(0,sub_time)
    white_clip = clip1.fl(channel)
    white_clip.write_videofile(white_output, audio=False)

def tenth_channel (img, get_white=False):
    white_yellow_image = select_white_yellow(img)
    gray_image = grayscale(white_yellow_image) #applying grayscale transform
    gauss_image = gaussian_blur(gray_image, 15) #use gaussian blur to make it easier for canny to detect edges
    canny_image = canny(gauss_image, 50, 150) #use canny transform for finding edges of image
    interest_reg = np.array([[[img.shape[1]/2, img.shape[0]/2], [img.shape[1], img.shape[0]], [0, img.shape[0]]]], dtype=np.int32)
    interest_reg_1 = np.array([[[img.shape[1]/2, img.shape[0]/2+25], [img.shape[1]-5, img.shape[0]], [5, img.shape[0]]]], dtype=np.int32)
    region_image_0 = region_of_interest(canny_image, interest_reg_1) #set rest of the picture to black
    if get_white:
        return white_hough_lines(region_image_0, 1, np.pi/180, 20, 20, 300) #use hough_lines for getting an image of lines
    else:
        lines_image = hough_lines(region_image_0, 1, np.pi/180, 20, 20, 300) #use hough_lines for getting an image of lines
    lined_image = weighted_img(lines_image, img) #adding transparent lines image to initial image
    return lined_image

def tenth_channel_fl(get_frame, t):
    res = get_frame(t)
    frames = 2
    for i in range(frames):
        res = weighted_img(tenth_channel(get_frame(t-i), True), res, 0.3)
    return tenth_channel(weighted_img(res, get_frame(t)))

video_convert_fl('solidWhiteRight', tenth_channel_fl, "_tenth")
video_convert_fl('solidYellowLeft', tenth_channel_fl, "_tenth")
video_convert_fl('challenge', tenth_channel_fl, "_tenth")
