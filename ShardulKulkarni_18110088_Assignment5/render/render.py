

import cv2
import numpy
import matplotlib.pyplot as plt
import os
def scaleAndShow(im, name = 'window', height = None, waitKey = 1):
    def callback(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, y, im[y, x])
    
    cv2.namedWindow(name)
    cv2.setMouseCallback(name,callback)
    if height is not None:
        width = int(im.shape[1]*height/im.shape[0])
        im = cv2.resize(im, (width, height), interpolation= cv2.INTER_NEAREST)
    cv2.imshow(name, im)
    if cv2.waitKey(waitKey) == ord('q'):
        exit()


class Renderer():


    def __init__(self, height = 600, width = 600, recordLocation = None ):
        shape = (height, width, 3)
        self.height = height
        self.width = width
        self.writer = None
        if recordLocation is not None:
            self.writer = cv2.VideoWriter(recordLocation, cv2.VideoWriter_fourcc(*'XVID'), 25, (width, height))


        self.origImage = numpy.ones(shape, dtype=numpy.uint8) * 255


    def putText(self, image, info = {}):
        for i, (k, v) in enumerate(info.items()):
            cv2.putText(image, k + ' : ' + str(v), (10, 20 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        return image
    
    def draw(self, image):
        raise NotImplementedError

    def getInfo(self):
        raise NotImplementedError

    def render(self, height = 600, pause = 10):
        image = self.origImage.copy()
        image = self.draw(image)

        image = self.putText(image, self.getInfo())

        if self.writer is not None:
            self.writer.write(image)
        scaleAndShow(image, height= height, waitKey = pause)
        return image

if __name__ == "__main__":

    
    def render(image):
        cv2.line(image, (0, 0), (100, 100), (0, 128, 0), 1)
        return image

    r = Renderer(600, 600, render)    

    for i in range(100):
        r.render(info = {'hello' : '00'})


        