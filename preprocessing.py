import cv2

vidcap = cv2.VideoCapture('/Users/yongyangnie/Desktop/speedchallenge/data/train.mp4')
success,image = vidcap.read()
count = 0

while success:
    cv2.imwrite("/Users/yongyangnie/Desktop/speedchallenge/data/frames/train/frame%d.jpg" % count, image)     # save frame as JPEG file      
    success,image = vidcap.read()
    count += 1

    if count % 1000 == 0:
    	print('Read a new frame: ', success)
    	