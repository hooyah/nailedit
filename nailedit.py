from PySide import QtGui, QtCore
from PIL import Image, ImageDraw, ImageFilter
from PIL import ImageEnhance
from PIL import ImageQt
from PointCloud import PointCloud
import random
import numpy


class Viewer(QtGui.QMainWindow):

    def __init__(self, parameters):
        super(Viewer, self).__init__()

        self.parameters = parameters

        self.imageLabel = QtGui.QLabel()
        self.imageLabel.setBackgroundRole(QtGui.QPalette.Base)
        self.imageLabel.setSizePolicy(QtGui.QSizePolicy.Ignored,
                QtGui.QSizePolicy.Ignored)
        self.imageLabel.setScaledContents(True)

        self.scrollArea = QtGui.QScrollArea()
        self.scrollArea.setBackgroundRole(QtGui.QPalette.Dark)
        self.scrollArea.setWidget(self.imageLabel)
        self.setCentralWidget(self.scrollArea)

        self.setWindowTitle("NailedIt")
        self.resize(parameters["proc_width"]+5, parameters["proc_height"]+5)

        self.mode ="points"
        self.residual = 0

        self.iterationCounter = 0
        self.imgCounter = 0
        self.outPath = "Q:\\Projects\\code\\nailedit\\render\\img_{:04d}.jpg"
        self.save_image = False


        self.timer = QtCore.QTimer(self)
        self.connect(self.timer, QtCore.SIGNAL("timeout()"), self.workPoints)
        self.timer.setSingleShot(True)
        self.timer.start(0)



    def showImage(self, pil_image):

        self.qim = ImageQt.ImageQt(pil_image)   # don't let python clean up the data
        self.imageLabel.setPixmap(QtGui.QPixmap.fromImage(self.qim))
        self.imageLabel.adjustSize()


    def workPoints(self):

        targetImage = self.parameters["TargetImage"]
        #if not "CurrentImage" in self.parameters:
        self.parameters["CurrentImage"] = Image.new("RGB", targetImage.size, parameters["backgroundColor"])

        currentImage = self.parameters["CurrentImage"]
        pnts = parameters["PointCloud"]

        pnts.relax(currentImage)
        draw_points(currentImage, pnts)

        self.showImage(currentImage)
        self.iterationCounter += 1
        if self.iterationCounter == 50:
            self.disconnect(self.timer, QtCore.SIGNAL("timeout()"), self.workPoints)
            self.connect(self.timer, QtCore.SIGNAL("timeout()"), self.March)
            self.iterationCounter = 0
            self.parameters.pop("CurrentImage")
            print self.parameters.keys()
        self.timer.start(10)


    def March(self):

        targetImage = self.parameters["TargetImage"]
        np_targetArray = numpy.array(targetImage).astype("float32")

        beauty_image = self.parameters["BeautyImage"] if "BeautyImage" in self.parameters else Image.new("RGB", targetImage.size, parameters["backgroundColor"])

        if not "CurrentImage" in self.parameters:
            self.parameters["CurrentImage"] = Image.new("RGB", targetImage.size, parameters["backgroundColor"])
        currentImage = self.parameters["CurrentImage"]

        pnts = parameters["PointCloud"]
        current_point_idx = parameters["currentPoint"]
        last_point_idx = parameters["lastPoint"]

        # find next best point

        # get most reasonable neightbors
        neighbours_idx = pnts.findNeighbours(current_point_idx, 0.25)

        # check how good the neighbours are
        candidates = []
        #base_diff = self.residual #image_diff(currentImage, np_targetArray)
        for neighbour in neighbours_idx:
            if neighbour != last_point_idx:
                new_img = draw_thread(currentImage.copy(), pnt1=pnts.p[current_point_idx], pnt2=pnts.p[neighbour], color=parameters["threadColor"])
                cur_diff = image_diff(new_img, np_targetArray)      # what is the difference to the target
                quality = cur_diff #- base_diff                      # how much better did this line make the result
                length = pnts.p[current_point_idx].dist(pnts.p[neighbour])
                candidates.append((quality, neighbour, cur_diff))

        # fish out the best match
        candidates.sort()
        bestMatch = candidates[0][1]
        print "iteration", self.iterationCounter, "residual", candidates[0][2], "improvement", candidates[0][2]-self.residual
        self.residual = candidates[0][2]

        img = draw_thread(currentImage.copy(), pnts.p[current_point_idx], pnts.p[bestMatch], parameters["threadColor"])
        self.parameters["CurrentImage"] = img
        self.parameters["lastPoint"] = current_point_idx
        self.parameters["currentPoint"] = bestMatch

        # pretty render
        beauty_image = draw_thread(beauty_image, pnts.p[current_point_idx], pnts.p[bestMatch], (255,255,255,255))
        beauty_image = Image.blend(beauty_image, img, 0.1)
        draw_points(beauty_image, pnts)
        self.parameters["BeautyImage"] = beauty_image
        self.showImage(beauty_image)

        if self.save_image and self.iterationCounter%4==0:
            beauty_image.save(self.outPath.format(self.imgCounter))
            self.imgCounter += 1

        self.timer.start(10)
        self.iterationCounter += 1



def Enhance(image, width, height):

    img = image.resize((width, height))
    enh = ImageEnhance.Contrast(img)
    img = enh.enhance(1.25)
    bt  = ImageEnhance.Brightness(img)
    img = bt.enhance(0.8)

    return img.convert("L")


def Scatter(resX, resY):

    pc = PointCloud()
    pc.addGrid(resX, resY)
    pc.addRandom(int(resX * resY * 0.3))

    return pc

def draw_points(image, pnts):

    w, h = image.width-1, image.height-1
    draw = ImageDraw.Draw(image, mode="RGBA")
    for p in pnts.p:
        #draw.rectangle([p.x*w-2, p.y*h-2, p.x*w+2, p.y*h+2], fill=(255,255,255,127), outline=(255,255,255,255))
        draw.point((p.x*w, p.y*h), (255,255,0,255))

def draw_thread(image, pnt1, pnt2, color):

    w, h = image.width-1, image.height-1
    draw = ImageDraw.Draw(image, mode="RGBA")
    draw.line([pnt1.x*w, pnt1.y*h, pnt2.x*w, pnt2.y*h], width=2, fill=color)
    return image


def image_diff(image, targetArray):

    #image = image.filter(ImageFilter.GaussianBlur(3))

    np_image = numpy.array(image)[:,:,1].astype("float32")
    return numpy.sum(numpy.absolute(np_image-targetArray))










if __name__ == '__main__':
    import sys

    app = QtGui.QApplication(sys.argv)

    parameters = {
        "proc_width":500,
        "proc_height":500,
        "grid_resX":25,
        "grid_resY":25,
        "backgroundColor":(0,0,0),
        "threadColor":(255, 255, 255, 20),
        "currentPoint" : 0,
        "lastPoint": -1
    }



    # 1 load image
    img = Image.open("einstein.jpeg")

    # 1.1 enhance/conform image
    img = Enhance(img, parameters["proc_width"], parameters["proc_height"])

    # 1.2 analyse image
    #   1.3 find edges amd corners

    # 2 scatter points
    parameters["PointCloud"] = Scatter(parameters["grid_resX"], parameters["grid_resY"])

    parameters["TargetImage"] = img


    imageViewer = Viewer(parameters)
    imageViewer.show()
    sys.exit(app.exec_())



"""
bruteForce_max : first test
t2: thicker stroke, dimmed target, length limit, more transparent stroke, getting stuck
t3: same as t2, thinner lines
t4: thicker lines again, more transparent, moves around easier for much longer before it gets stuck
t5: randomized points 7.5k iterations
t6: 30x30, bit more resolution, point brick pattern (still randomized, pretty bad actually), stuck after 8k iters
t7: normalized, smaller screen, 25x25 points
t8: truely stacked points now normalized
t9: thinner lines, unnormalized
t10: better scattering 20% random points, 2p linewidth

ideas: scipy draw line directly into numpy array, skip pillow conversions
        do error calculation of blurred picture
        using max num connections for nail selection based on brighness
        find out how what causes endless loops
        add nail collision detection
"""