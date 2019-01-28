from PySide import QtGui, QtCore
from PIL import Image, ImageDraw, ImageFilter
from PIL import ImageEnhance, ImageChops
from PIL import ImageQt
from PointCloud import PointCloud
import random
import numpy
from scipy import ndimage
from multiprocessing import Pool as ThreadPool
import time


class Viewer(QtGui.QMainWindow):

    def __init__(self, parameters):
        super(Viewer, self).__init__()

        self.parameters = parameters

        self.multiWidget = QtGui.QWidget()

        self.imageLabel = QtGui.QLabel()
        self.imageLabel.setBackgroundRole(QtGui.QPalette.Base)
        self.imageLabel.setSizePolicy(QtGui.QSizePolicy.Ignored,
                QtGui.QSizePolicy.Ignored)
        self.imageLabel.setScaledContents(False)

        self.imageLabel2 = QtGui.QLabel()
        self.imageLabel2.setBackgroundRole(QtGui.QPalette.Base)
        self.imageLabel2.setSizePolicy(QtGui.QSizePolicy.Ignored,
                QtGui.QSizePolicy.Ignored)
        self.imageLabel2.setScaledContents(False)

        self.bl = QtGui.QVBoxLayout(self.multiWidget)
        self.bl.addWidget(self.imageLabel)
        self.bl.addWidget(self.imageLabel2)

        self.scrollArea = QtGui.QScrollArea()
        self.scrollArea.setBackgroundRole(QtGui.QPalette.Dark)
        self.scrollArea.setWidget(self.multiWidget)
        self.setCentralWidget(self.scrollArea)

        self.scrollArea.setLayout(self.bl)


        self.setWindowTitle("NailedIt")
        self.resize(parameters["proc_width"]+5, parameters["proc_height"]*2+10)

        self.mode ="points"
        self.avg_improvement = (255**2)*2*parameters["proc_width"]

        self.string_length = 0
        self.iterationCounter = 0
        self.imgCounter = 0
        self.outPath = "Q:\\Projects\\code\\nailedit\\render\\img_{:04d}.jpg"
        self.save_image = False


        self.targetImage = self.parameters["TargetImage"]
        self.np_targetArray = numpy.array(self.targetImage).astype("float32")/255
        self.parameters["CurrentImage"] = Image.new("RGB", self.targetImage.size, parameters["backgroundColor"])
        self.residual = image_diff(self.parameters["CurrentImage"], self.np_targetArray)

        self.threadpool = ThreadPool()
        self.lastTime = time.time()


        self.timer = QtCore.QTimer(self)
        self.connect(self.timer, QtCore.SIGNAL("timeout()"), self.workImage)
        self.timer.setSingleShot(True)
        self.timer.start(0)


    def showImage(self, pil_image, slot=0):

        if slot == 0:
            self.qim = ImageQt.ImageQt(pil_image)   # don't let python clean up the data
            self.imageLabel.setPixmap(QtGui.QPixmap.fromImage(self.qim))
            self.imageLabel.adjustSize()
        elif slot == 1:
            self.qim2 = ImageQt.ImageQt(pil_image)   # don't let python clean up the data
            self.imageLabel2.setPixmap(QtGui.QPixmap.fromImage(self.qim2))
            self.imageLabel2.adjustSize()


    def workImage(self):

        if not "DetailImage" in parameters:

            targetImage = self.parameters["TargetImage"]
            img = targetImage#.filter(ImageFilter.GaussianBlur(2))
            np_img = numpy.array(img).astype("float32")
            #gradx, grady = numpy.gradient(np_img)
            #gradmag = numpy.sqrt(gradx**2 + grady**2)
            gradmag = ndimage.gaussian_gradient_magnitude(np_img, 3)
            gradmag = gradmag / gradmag.max() * 255
            img = Image.fromarray(gradmag.astype("uint8"))
            #img = img.filter(ImageFilter.GaussianBlur(5))
            img = self.parameters["DetailImage"] = img

            self.showImage(targetImage)
            self.showImage(Image.merge("RGB", (img,img,img)), slot=1)
            self.timer.start(3000)

        else:
            self.disconnect(self.timer, QtCore.SIGNAL("timeout()"), self.workImage)
            self.connect(self.timer, QtCore.SIGNAL("timeout()"), self.workPoints)
            self.timer.start(10)

    def workPoints(self):

        targetImage = self.parameters["TargetImage"]
        currentImage = self.parameters["CurrentImage"]
        currentImage.paste(parameters["backgroundColor"], (0,0)+currentImage.size)
        pnts = parameters["PointCloud"]

        pnts.relax(currentImage, 50, self.parameters["DetailImage"])
        draw_points(currentImage, pnts)

        self.showImage(currentImage)
        self.iterationCounter += 1
        if self.iterationCounter == 50:
            self.disconnect(self.timer, QtCore.SIGNAL("timeout()"), self.workPoints)
            self.connect(self.timer, QtCore.SIGNAL("timeout()"), self.March)
            self.iterationCounter = 0
            currentImage.paste(parameters["backgroundColor"], (0, 0) + currentImage.size)
            print self.parameters.keys()
        self.timer.start(10)



    def March(self):


        beauty_image = self.parameters["BeautyImage"] if "BeautyImage" in self.parameters else Image.new("RGB", self.targetImage.size, parameters["backgroundColor"])
        currentImage = self.parameters["CurrentImage"]

        pnts = parameters["PointCloud"]
        current_point_idx = parameters["currentPoint"]
        last_point_idx = parameters["lastPoint"]

        # find next best point

        # get most reasonable neightbors
        neighbours_idx = pnts.findNeighbours(current_point_idx, parameters["proc_width"]*0.25)

        # check how good the neighbours are
        #candidates = []
        #base_diff = self.residual #image_diff(currentImage, np_targetArray)
        #for neighbour in neighbours_idx:
        #    if neighbour != last_point_idx:
        #        new_img = draw_thread(currentImage.copy(), pnt1=pnts.p[current_point_idx], pnt2=pnts.p[neighbour], color=parameters["threadColor"])
        #        cur_diff = image_diff(new_img, self.np_targetArray)      # what is the difference to the target
        #        quality = cur_diff - base_diff                      # how much better did this line make the result
        #        length = pnts.p[current_point_idx].dist(pnts.p[neighbour])
        #        candidates.append((quality, neighbour, cur_diff, length))
        col = parameters["threadColor"]
        params = [(currentImage, pnts.p[current_point_idx], pnts.p[neighbour], self.np_targetArray, neighbour, col, self.residual) for neighbour in neighbours_idx if neighbour != last_point_idx]
        candidates = self.threadpool.map(check_quality, params)


        # fish out the best match
        candidates.sort()
        #candidates.sort(reverse=True)
        # pick the first candidate that is negative if availably
        #for c in candidates:
        #    bestMatch = c[1]
        #    if c[0] < 0.0:
        #        break
        bestMatch = candidates[0]

        improvement = self.residual - candidates[0][2]
        self.residual = candidates[0][2]
        self.avg_improvement = self.avg_improvement*.9 + improvement * .1
        self.string_length += candidates[0][3] * self.parameters["ppi"]
        print "iteration", self.iterationCounter, "residual", candidates[0][2], "improvement", improvement, "avg", self.avg_improvement, "string {:.1f}m".format(self.string_length)
        print candidates[:5]

        img = draw_thread(currentImage.copy(), pnts.p[current_point_idx], pnts.p[bestMatch[1]], parameters["threadColor"])
        self.parameters["CurrentImage"] = img
        self.parameters["lastPoint"] = current_point_idx
        self.parameters["currentPoint"] = bestMatch[1]

        pnts.cool(0.1)
        pnts.p[bestMatch[1]].heat = 1.0

        # pretty render
        beauty_image = draw_thread(beauty_image, pnts.p[current_point_idx], pnts.p[bestMatch[1]], (255,120,120,255))
        beauty_image = Image.blend(beauty_image, img, 0.1)
        draw_points(beauty_image, pnts)
        self.parameters["BeautyImage"] = beauty_image
        self.showImage(beauty_image)

        if self.save_image and self.iterationCounter%4==0:
            beauty_image.save(self.outPath.format(self.imgCounter))
            self.imgCounter += 1

        # render a difference image
        if self.iterationCounter % 10 == 0:
            redlut   = tuple(((127-i)*2) if i <= 127 else 0 for i in xrange(256))
            greenlut = tuple(0 if i <= 127 else ((i-127)*2) for i in xrange(256))
            bluelut  = tuple([0]*256)

            difImage = ImageChops.subtract(self.targetImage, img.convert("L"), 2, 127)
            difImage = Image.merge("RGB", (difImage, difImage, difImage))
            difImage = difImage.point((redlut + greenlut + bluelut))
            self.showImage(difImage, slot=1)

            now = time.time()
            print  now-self.lastTime, "s/10 iterations"
            self.lastTime = now

        self.iterationCounter += 1

        if abs(self.avg_improvement) <= 0.0001:
            print "no more improvement"
            self.timer.start(1000)
        else:
            self.timer.start(10)




def check_quality(params):

    img = params[0]
    p1 = params[1]
    p2 = params[2]
    trg = params[3]
    ind = params[4]
    col = params[5]
    prevResidual = params[6]

    new_img = draw_thread(img.copy(), pnt1=p1, pnt2=p2, color=col)
    cur_diff = image_diff(new_img, trg)    # what is the difference to the target
    quality = (cur_diff - prevResidual)    # how much better did this line make the result
    quality += quality * 3 * (1.0-p2.heat) # attenuate by previously visited
    length = p1.dist(p2)
    return (quality, ind, cur_diff, length)


def Enhance(image, width, height):

    img = image.resize((width, height))
    enh = ImageEnhance.Contrast(img)
    img = enh.enhance(1.25)
    bt  = ImageEnhance.Brightness(img)
    img = bt.enhance(0.95)

    return img.convert("L")


def Scatter(resX, resY, dimx, dimy):

    pc = PointCloud(dimx, dimy)
    pc.addGrid(resX, resY)
    pc.addRandom(int(resX * resY * 0.3))

    return pc

def draw_points(image, pnts):

    draw = ImageDraw.Draw(image, mode="RGBA")
    for p in pnts.p:
        #draw.rectangle([p.x*w-2, p.y*h-2, p.x*w+2, p.y*h+2], fill=(255,255,255,127), outline=(255,255,255,255))
        draw.point((p.x, p.y), (255,int(255*(1.0-p.heat)),0,255))


def draw_thread(image, pnt1, pnt2, color):

    draw = ImageDraw.Draw(image, mode="RGBA")
    draw.line([pnt1.x, pnt1.y, pnt2.x, pnt2.y], width=2, fill=color)
    return image


def image_diff(image, targetArray):

    #image = image.filter(ImageFilter.GaussianBlur(2))

    lum = image.getchannel("R")
    #np_image = numpy.array(image)[:,:,1].astype("float32")
    np_image = numpy.array(lum).astype("float32")
    np_image = numpy.multiply(np_image, 1.0/255, out=np_image)
    #error = np_image-targetArray
    error = numpy.subtract(np_image, targetArray, out=np_image)

    #nul = numpy.zeros(error.shape)
    #better = numpy.clip(error, -2000000000, 0)
    #worse  = numpy.multiply(numpy.clip(error, 0, 2000000000, out=error), 2, out=error)

    #error = numpy.add(better, worse, out=error)
    error = numpy.multiply(error, error, out=error)

    #error = (better+worse)**2 # error squred
    return numpy.sum(error)










if __name__ == '__main__':
    import sys

    app = QtGui.QApplication(sys.argv)

    parameters = {
        "proc_width":500,
        "proc_height":500,
        "ppi": 0.3/500, # pixels per meter
        "grid_resX":25,
        "grid_resY":25,
        "backgroundColor":(0,0,0),
        "threadColor":(255, 255, 255, 40),
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
    parameters["PointCloud"] = Scatter(parameters["grid_resX"], parameters["grid_resY"], parameters["proc_width"]-1, parameters["proc_height"]-1)

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
t11: error squared, brighter image, normalized, 20k iters
    t11 but not normalized (identical) -> meh
t12: 5 times higher penalty for error
t13: 2 times higher penalty 16k stuck
t14: twice the opacity, removed penalty
t15: point scatter affected by DetailImage (gradient magnidute)
t16: increased Detail pull a bit, added heat

ideas: scipy draw line directly into numpy array, skip pillow conversions
        do error calculation of blurred picture (preliminary tests have shown no improvements, 10x slower)
        using max num connections for nail selection based on brighness
        find out how what causes endless loops
        add nail collision detection
        increase nail density in areas with higher frequency detail
        errors should have a magnitude more weight then improvements
"""