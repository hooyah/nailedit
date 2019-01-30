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
import json, numbers


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


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
        self.imageLabel.setStyleSheet("border: 0px")
        self.imageLabel.setContentsMargins(0, 0, 0, 0)

        self.imageLabel2 = QtGui.QLabel()
        self.imageLabel2.setBackgroundRole(QtGui.QPalette.Base)
        #self.imageLabel2.setSizePolicy(QtGui.QSizePolicy.Ignored,
        #        QtGui.QSizePolicy.Ignored)
        self.imageLabel2.setScaledContents(False)
        self.imageLabel.setStyleSheet("border: 0px")
        self.imageLabel2.setContentsMargins(0, 0, 0, 0)

        self.bl = QtGui.QVBoxLayout(self.multiWidget)
        self.bl.addWidget(self.imageLabel)
        self.bl.addWidget(self.imageLabel2)

        self.scrollArea = QtGui.QScrollArea()
        self.scrollArea.setBackgroundRole(QtGui.QPalette.Dark)
        self.scrollArea.setWidget(self.multiWidget)
        self.setCentralWidget(self.scrollArea)

        self.scrollArea.setLayout(self.bl)


        self.mode ="ProcessImage"
        self.setWindowTitle("NailedIt - "+self.mode)
        self.resize(parameters["proc_width"]+50, parameters["proc_height"]*2+50)

        self.avg_improvement = (255**2)*2*parameters["proc_width"]

        self.string_path = []
        self.string_length = 0
        self.iterationCounter = 0
        self.imgCounter = 0
        self.outPath = "Q:\\Projects\\code\\nailedit\\render\\img_{:04d}.jpg"
        self.save_image = False


        self.targetImage = self.parameters["TargetImage"]
        self.np_targetArray = numpy.array(self.targetImage).astype("float32")/255.0
        self.parameters["CurrentImage"] = Image.new("RGB", self.targetImage.size, parameters["backgroundColor"])
        self.residual = image_diff(self.parameters["CurrentImage"], self.np_targetArray)

        self.threadpool = ThreadPool()
        self.lastTime = time.time()


        self.timer = QtCore.QTimer(self)
        self.connect(self.timer, QtCore.SIGNAL("timeout()"), self.workImage)
        self.timer.setSingleShot(True)
        self.timer.start(0)


    def save_as(self, filename):

        imagename = filename.replace(".json", ".png")
        data = { "1:summary": {
                    "number of nails" : len(self.parameters["PointCloud"].p),
                    "thread length" : self.string_length,
                    "result image" : imagename,
                    "num_segments" : self.iterationCounter
                 },
                 "2:parameters:" : dict(kv for kv in self.parameters.iteritems() if is_jsonable(kv[1])),
                 "3:nails" : [ (p.x,p.y) for p in self.parameters["PointCloud"].p ],
                 "4:thread": self.string_path
               }
        with open(filename, "w") as f:
            json.dump(data, f, indent=4, sort_keys=True)
            f.close()
            self.parameters["CurrentImage"].save(imagename)
            print "done writing", filename



    def closeEvent(self, event):

        self.timer.stop()
        if self.mode == "Threading" and QtGui.QMessageBox.question(self, "Quit", "Save it?", QtGui.QMessageBox.Yes, QtGui.QMessageBox.No) == QtGui.QMessageBox.Yes:
            filename = QtGui.QFileDialog.getSaveFileName(self, "Save as", "./", "Nailedit (*.json)")
            if filename:
                self.save_as(filename[0])


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

        targetImage = self.parameters["TargetImage"]
        if not "DetailImage" in self.parameters:

            self.setWindowTitle("NailedIt - Detail Image")
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
            self.timer.start(2000)

        elif not "EdgesImage" in self.parameters:

            self.setWindowTitle("NailedIt - Edges Image")
            img = targetImage#.filter(ImageFilter.GaussianBlur(2))
            np_img = numpy.array(img).astype("float32")
            gradmag = ndimage.gaussian_gradient_magnitude(np_img, 1)
            gradmag = gradmag / gradmag.max() * 255
            img = Image.fromarray(gradmag.astype("uint8"))
            img = self.parameters["EdgesImage"] = img

            self.showImage(Image.merge("RGB", (img,img,img)), slot=1)
            self.timer.start(4000)


        else:
            self.disconnect(self.timer, QtCore.SIGNAL("timeout()"), self.workImage)
            self.connect(self.timer, QtCore.SIGNAL("timeout()"), self.workPoints)
            self.timer.start(10)
            self.mode ="ProcessPoints"
            self.setWindowTitle("NailedIt - "+self.mode)

    def workPoints(self):

        targetImage = self.parameters["TargetImage"]
        currentImage = self.parameters["CurrentImage"]
        currentImage.paste(self.parameters["backgroundColor"], (0,0)+currentImage.size)

        minDist = self.parameters["nailDistMin"]
        maxDist = self.parameters["nailDistMax"]
        img_w = self.parameters["proc_width"]
        img_h = self.parameters["proc_height"]

        if "PointCloud" in self.parameters:
            pnts = self.parameters["PointCloud"]
        else:
            pc = PointCloud(img_w, img_h)
            pc.scatterOnMask(self.parameters["EdgesImage"], (img_w*img_h)/(minDist**2), int(minDist), threshold=0.2)
            pc.heat(1.0)

            img = self.parameters["EdgesImage"]
            img = draw_points(Image.merge("RGB", (img,img,img)), pc, 3)
            self.showImage(img, slot=1)

            gridDistX = img_w / maxDist
            gridDistY = img_h / maxDist
            pc.addGrid(gridDistX, gridDistY)
            pc.addRandom(int(gridDistX * gridDistY * 0.3))
            self.parameters["PointCloud"] = pc

            self.timer.start(1000)
            return



        pnts.relax(currentImage, 50, self.parameters["DetailImage"], minDist, maxDist)
        draw_points(currentImage, pnts)

        self.showImage(currentImage)
        self.iterationCounter += 1
        if self.iterationCounter == 50:
            start = self.parameters["start_at"]
            self.parameters["currentPoint"] = pnts.closestPoint(float(start[0])*img_w, float(start[1]*img_h))
            self.string_path.append(self.parameters["currentPoint"])

            self.disconnect(self.timer, QtCore.SIGNAL("timeout()"), self.workPoints)
            self.connect(self.timer, QtCore.SIGNAL("timeout()"), self.March)
            self.iterationCounter = 0
            currentImage.paste(self.parameters["backgroundColor"], (0, 0) + currentImage.size)
            self.mode = "Threading"
            self.setWindowTitle("NailedIt - " + self.mode)
        self.timer.start(10)



    def March(self):


        beauty_image = self.parameters["BeautyImage"] if "BeautyImage" in self.parameters else Image.new("RGB", self.targetImage.size, self.parameters["backgroundColor"])
        currentImage = self.parameters["CurrentImage"]

        pnts = self.parameters["PointCloud"]
        current_point_idx = self.parameters["currentPoint"]
        last_point_idx = self.parameters["lastPoint"]

        # find next best point

        # get most reasonable neightbors
        neighbours_idx = pnts.findNeighbours(current_point_idx, self.parameters["proc_width"]*0.25)

        # check how good the neighbours are
        col = self.parameters["threadColor"]

        # clamp current_image with target to acurately detect overshoot
        cur_np = numpy.array(currentImage.getchannel("R"))
        max_v = self.np_targetArray.max()*255
        #cur_np = numpy.minimum(cur_np, (self.np_targetArray*255).astype("uint8"), out=cur_np)
        cur_np = numpy.clip(cur_np, 0, max_v, out=cur_np)
        currentImage = Image.merge("RGB", (Image.fromarray(cur_np), Image.fromarray(cur_np), Image.fromarray(cur_np)))

        params = [(currentImage, pnts.p[current_point_idx], pnts.p[neighbour], self.np_targetArray, neighbour, col, self.residual) for neighbour in neighbours_idx if neighbour != last_point_idx]
        candidates = self.threadpool.map(check_quality, params)


        # fish out the best match
        candidates = [c for c in candidates if c[0] != 0]   # c[0] == 0 indicates color clipping due to
        candidates.sort()
        #candidates.sort(reverse=True)
        bestMatch = candidates[0]

        improvement = bestMatch[0]#self.residual - candidates[0][2]
        self.residual = bestMatch[2]
        self.avg_improvement = self.avg_improvement*.9 + improvement * .1
        self.string_length += bestMatch[3] * self.parameters["ppi"]
        self.string_path.append(bestMatch[1])
        print "iteration", self.iterationCounter, "residual", bestMatch[2], "improvement", improvement, "avg", self.avg_improvement, "string {:.1f}m".format(self.string_length)
        #print candidates[:5]

        img = draw_thread(currentImage.copy(), pnts.p[current_point_idx], pnts.p[bestMatch[1]], self.parameters["threadColor"])
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

    length = p1.dist(p2)
    b_len = max(int(abs(p1.x-p2.x)), int(abs(p1.y-p2.y))) # bresenham num pixels

    new_img = draw_thread(img.copy(), pnt1=p1, pnt2=p2, color=col)
    cur_diff = image_diff(new_img, trg)    # what is the difference to the target
    quality = (cur_diff - prevResidual)/b_len    # how much better did this line make the result
    quality += abs(quality) * 10 * p2.heat # attenuate by previously visited
    return (quality, ind, cur_diff, length)


def Enhance(image, width, height):

    img = image.resize((width, height))
    enh = ImageEnhance.Contrast(img)
    img = enh.enhance(1.25)
    bt  = ImageEnhance.Brightness(img)
    img = bt.enhance(0.80)

    return img.convert("L")


"""def Scatter(resX, resY, dimx, dimy):

    pc = PointCloud(dimx, dimy)
    pc.addGrid(resX, resY)
    pc.addRandom(int(resX * resY * 0.3))

    return pc"""

def draw_points(image, pnts, size=1):

    w = int(size-1)/2
    draw = ImageDraw.Draw(image, mode="RGBA")
    if w < 1 :
        for p in pnts.p:
            draw.point((p.x, p.y), (255,int(255*(1.0-p.heat)),0,255))
    else:
        for p in pnts.p:
            draw.rectangle([p.x-w, p.y-w, p.x+w, p.y+w], fill=(255,int(255*(1.0-p.heat)),0,127), outline=(255,255,255,255))

    return image


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
    better = numpy.clip(error, -2000000000, 0)
    worse  = numpy.multiply(numpy.clip(error, 0, 2000000000, out=error), 5, out=error)
    error = numpy.add(better, worse, out=error)
    error = numpy.multiply(error, error, out=error)
    #error = numpy.abs(error, out=error)

    #error = (better+worse)**2 # error squred
    return numpy.sum(error)










if __name__ == '__main__':
    import sys

    app = QtGui.QApplication(sys.argv)

    ppm = 0.3/500 # pixels per meter
    params = {
        "proc_width":500,
        "proc_height":500,
        "ppi": ppm,
        #"grid_resX":25,
        #"grid_resY":25,
        "nailDistMin": 6.0 / 1000.0 / ppm,
        "nailDistMax": 16.0 / 1000.0 / ppm,
        "backgroundColor":(0,0,0),
        "threadColor":(255, 255, 255, 20),
        "currentPoint" : 0,
        "lastPoint": -1,
        "start_at": (0.5,0)
    }



    # 1 load image
    params["inputImagePath"] = "einstein.jpeg"
    img = Image.open(params["inputImagePath"])

    # 1.1 enhance/conform image
    img = Enhance(img, params["proc_width"], params["proc_height"])

    # 1.2 analyse image
    #   1.3 find edges amd corners

    # 2 scatter points
    #parameters["PointCloud"] = Scatter(parameters["grid_resX"], parameters["grid_resY"], parameters["proc_width"]-1, parameters["proc_height"]-1)

    params["TargetImage"] = img


    imageViewer = Viewer(params)
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
t16: increased Detail pull a bit, added heat, complete overhaul
t17: points on edges
t18: retry penalty, smaller min dist, white point lowered
t19: least squared, more edge points, less opacity thread, darker target

ideas: scipy draw line directly into numpy array, skip pillow conversions
        do error calculation of blurred picture (preliminary tests have shown no improvements, 10x slower)
        using max num connections for nail selection based on brighness
        find out how what causes endless loops
        add nail collision detection
        increase nail density in areas with higher frequency detail
        errors should have a magnitude more weight then improvements
"""