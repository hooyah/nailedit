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

        self.segmentCount = {}
        self.string_path = []
        self.string_length = 0
        self.iterationCounter = 0
        self.imgCounter = 0
        self.outPath = "Q:\\Projects\\code\\nailedit\\render\\img_{:04d}.jpg"
        self.save_image = False


        self.targetImage = self.parameters["TargetImage"]
        self.np_targetArray = PIL_to_array(self.targetImage)
        self.parameters["CurrentImage"] = numpy.array(Image.new("L", self.targetImage.size, parameters["backgroundColor"]), dtype="float32")/255
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
            array_to_PIL_rgb(self.parameters["CurrentImage"]).save(imagename)
            print "done writing", filename



    def closeEvent(self, event):

        counts = [(c[1], c[0]) for c in self.segmentCount.iteritems()]
        counts.sort(reverse=True)
        print counts[:100]

        self.timer.stop()
        if self.mode == "Threading" and QtGui.QMessageBox.question(self, "Quit", "Save it?", QtGui.QMessageBox.Yes, QtGui.QMessageBox.No) == QtGui.QMessageBox.Yes:
            filename = QtGui.QFileDialog.getSaveFileName(self, "Save as", "./", "Nailedit (*.json)")
            if filename:
                self.save_as(filename[0])


    def showImage(self, image, slot=0):

        if isinstance(image, numpy.ndarray):
            image = array_to_PIL_rgb(image)

        if slot == 0:
            self.qim = ImageQt.ImageQt(image)   # don't let python clean up the data
            self.imageLabel.setPixmap(QtGui.QPixmap.fromImage(self.qim))
            self.imageLabel.adjustSize()
        elif slot == 1:
            self.qim2 = ImageQt.ImageQt(image)   # don't let python clean up the data
            self.imageLabel2.setPixmap(QtGui.QPixmap.fromImage(self.qim2))
            self.imageLabel2.adjustSize()


    def workImage(self):

        targetImage = self.parameters["TargetImage"]
        if not "DetailImage" in self.parameters:

            self.setWindowTitle("NailedIt - Detail Image")
            gradmag = ndimage.gaussian_gradient_magnitude(self.np_targetArray, 3)
            gradmag = gradmag / gradmag.max()
            self.parameters["DetailImage"] = gradmag

            self.showImage(targetImage)
            self.showImage(gradmag, slot=1)
            self.timer.start(1000)

        elif not "EdgesImage" in self.parameters:

            if "edgesImagePath" in self.parameters:
                img = Image.open(self.parameters["edgesImagePath"])
                img = img.resize((self.parameters["proc_width"], self.parameters["proc_height"]))
                self.parameters["EdgesImage"] = numpy.array(img.getchannel("R"), dtype='float32')/255
            else:
                self.setWindowTitle("NailedIt - Edges Image")
                gradmag = ndimage.gaussian_gradient_magnitude(self.np_targetArray, 1.5)
                gradmag = gradmag / gradmag.max()
                self.parameters["EdgesImage"] = gradmag

            self.showImage(self.parameters["EdgesImage"], slot=1)
            self.timer.start(1000)

        else:
            npt = ndimage.filters.gaussian_filter(self.np_targetArray, 4)
            self.blurredTarget = npt

            self.disconnect(self.timer, QtCore.SIGNAL("timeout()"), self.workImage)
            self.connect(self.timer, QtCore.SIGNAL("timeout()"), self.workPoints)
            self.timer.start(10)
            self.mode ="ProcessPoints"
            self.setWindowTitle("NailedIt - "+self.mode)

    def workPoints(self):

        targetImage = self.parameters["TargetImage"]
        currentImage = self.parameters["CurrentImage"]
        currentImage[:] = self.parameters["backgroundColor"]

        minDist = self.parameters["nailDistMin"]
        maxDist = self.parameters["nailDistMax"]
        img_w = self.parameters["proc_width"]
        img_h = self.parameters["proc_height"]

        if "PointCloud" in self.parameters:
            pnts = self.parameters["PointCloud"]
        else:
            pc = PointCloud(img_w, img_h)
            pc.scatterOnMask(self.parameters["EdgesImage"], (img_w*img_h)/(minDist**2), minDist, threshold=0.25)
            pc.heat(1.0)

            img = array_to_PIL_rgb(self.parameters["EdgesImage"])
            img = draw_points(img, pc, 3)
            self.showImage(img, slot=1)

            gridDistX = img_w / maxDist
            gridDistY = img_h / maxDist
            pc.addGrid(gridDistX, gridDistY)
            pc.addRandom(int(gridDistX * gridDistY * 0.3))
            self.parameters["PointCloud"] = pc

            #pc.maskPoints(targetImage, 0.03)

            self.timer.start(10)
            return


        img = array_to_PIL_rgb(currentImage)
        pnts.relax(img, 10, self.parameters["DetailImage"], minDist, maxDist)
        draw_points(img, pnts)

        self.showImage(img)
        self.iterationCounter += 1


        if self.iterationCounter == 50:     # debugging

            foo = Image.new("RGB", targetImage.size)

            draw = ImageDraw.Draw(foo, "RGB")

            problems = [0]*len(pnts.p)
            for me,p in enumerate(pnts.p) :
                if not p.heat > 0:
                    cps = pnts.closestPoints(p.x, p.y, minDist, me)
                    if len(cps):
                        problems[me] = len(cps)

            numOffenders = 0
            for me, p in enumerate(pnts.p):
                bad = problems[me] > 0
                if bad:
                    numOffenders += 1
                draw.rectangle((p.x-1, p.y-1, p.x+1, p.y+1), (255, 0, 0) if bad else (255,255,0))

            if numOffenders:
                # remove the offending point with the most neighbours
                problems = [ (prob, id) for id,prob in enumerate(problems)]
                problems.sort(reverse=True)
                del pnts.p[problems[0][1]]
                self.iterationCounter -= 1

            else:
                # last ditch check including all points (even the ones on edges)
                for me,p in enumerate(pnts.p) :
                    cps = pnts.closestPoints(p.x, p.y, minDist, me)
                    cps = [cp for cp in cps if p.dist(pnts.p[cp]) < minDist]
                    if len(cps):
                        numOffenders +=1
                print "point cleanup done. number of minDists:", numOffenders



            self.showImage(foo)
            self.timer.start(10)


        elif self.iterationCounter == 51:
            start = self.parameters["start_at"]
            self.parameters["currentPoint"] = pnts.closestPoint(float(start[0])*img_w, float(start[1]*img_h))[0]
            self.string_path.append(self.parameters["currentPoint"])
            pnts.heat(0)
            for p in pnts.p:
                p.neighbors = None

            self.disconnect(self.timer, QtCore.SIGNAL("timeout()"), self.workPoints)
            self.connect(self.timer, QtCore.SIGNAL("timeout()"), self.March)
            self.iterationCounter = 0
            currentImage[:] = self.parameters["backgroundColor"]
            self.mode = "Threading"
            self.setWindowTitle("NailedIt - " + self.mode)
            self.timer.start(10)

        else:
            self.timer.start(10)



    def March(self):


        beauty_image = self.parameters["BeautyImage"] if "BeautyImage" in self.parameters else Image.new("RGB", self.targetImage.size, self.parameters["backgroundColor"])
        currentImage = self.parameters["CurrentImage"]

        pnts = self.parameters["PointCloud"]
        current_point_idx = self.parameters["currentPoint"]
        last_point_idx = self.parameters["lastPoint"]

        # find next best point

        # get most reasonable neightbors
        neighbours_idx = pnts.p[current_point_idx].neighbors
        if neighbours_idx == None:
            neighbours_idx = pnts.findNeighbours(current_point_idx, self.parameters["proc_width"]*0.25)
            remove_nail_collisions(pnts, current_point_idx, neighbours_idx, self.parameters["nailDiameter"]/2)
            pnts.p[current_point_idx].neighbors = neighbours_idx

        remove_saturated_segments(current_point_idx, neighbours_idx, self.segmentCount, self.parameters["maxSegmentConnect"])

        # check how good the neighbours are
        col = self.parameters["threadColor"]

        # clamp current_image with target to accurately detect overshoot
        max_v = self.np_targetArray.max()
        currentImage = numpy.clip(currentImage, 0, max_v, out=currentImage)

        params = [(currentImage, pnts.p[current_point_idx], pnts.p[neighbour], self.np_targetArray, neighbour, col, self.residual, self.blurredTarget) for neighbour in neighbours_idx if neighbour != last_point_idx]
        candidates = self.threadpool.map(check_quality, params)
        #candidates = [check_quality(p) for p in params]


        # fish out the best match
        candidates.sort()
        #candidates.sort(reverse=True)
        bestMatch = candidates[0]

        improvement = bestMatch[0]#self.residual - candidates[0][2]
        self.residual = bestMatch[2] # (has to be recalculated if changing target data below)
        self.avg_improvement = self.avg_improvement*.9 + improvement * .1
        self.string_length += bestMatch[3] * self.parameters["ppi"]
        self.string_path.append(bestMatch[1])


        currentImage = draw_thread(currentImage.copy(), pnts.p[current_point_idx], pnts.p[bestMatch[1]], self.parameters["threadColor"])
        self.parameters["CurrentImage"] = currentImage
        self.parameters["lastPoint"] = current_point_idx
        self.parameters["currentPoint"] = bestMatch[1]

        seg = (min(current_point_idx, bestMatch[1]), max(current_point_idx, bestMatch[1]))
        if seg in self.segmentCount:
            self.segmentCount[seg] += 1
        else:
            self.segmentCount[seg] = 1

        pnts.cool(0.1)
        pnts.p[bestMatch[1]].heat = 1.0

        # remove areas that are currentImage>=target from the target
        #cur_np = numpy.array(currentImage.getchannel("R"), dtype="float32")
        #cur_np = numpy.multiply(cur_np, 1.0/255.0, out=cur_np)
        #rem = (cur_np < self.np_targetArray).astype(float)
        #numpy.multiply(self.np_targetArray, rem, out=self.np_targetArray)
        #self.residual = image_diff(currentImage, self.np_targetArray, cur_np)

        print "iteration", self.iterationCounter, "residual", bestMatch[2], "improvement", improvement, "avg", self.avg_improvement, "string {:.1f}m".format(self.string_length), "n",bestMatch[1]
        #print candidates[:5]


        # pretty render
        beauty_image = draw_thread_rgb(beauty_image, pnts.p[current_point_idx], pnts.p[bestMatch[1]], (255,120,120,255))
        beauty_image = Image.blend(beauty_image, array_to_PIL_rgb(currentImage), 0.1)
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


            #difImage = ImageChops.subtract(self.targetImage, currentImage.getchannel("R"), 2, 127)
            #difImage = Image.merge("RGB", (difImage, difImage, difImage))
            df = self.np_targetArray - currentImage
            numpy.multiply(df, 0.5, out=df)
            numpy.add(df, 0.5, out=df)
            difImage = array_to_PIL_rgb(df)
            difImage = difImage.point((redlut + greenlut + bluelut))
            self.showImage(difImage, slot=1)

            now = time.time()
            print  now-self.lastTime, "s/10 iterations"
            self.lastTime = now

        self.iterationCounter += 1

        if self.iterationCounter >= self.parameters["maxIterations"]:
            self.close()

        if abs(self.avg_improvement) <= 0.0001:
            print "no more improvement"
            self.timer.start(10)
        else:
            self.timer.start(10)




def check_quality(params):

    #return (random.uniform(-100,100),params[4],random.uniform(-100,100), 10)
    img = params[0]
    p1 = params[1]
    p2 = params[2]
    trg = params[3]
    ind = params[4]
    col = params[5]
    prevResidual = params[6]
    blurredTarget = params[7]

    length = p1.dist(p2)
    b_len = max(int(abs(p1.x-p2.x)+1), int(abs(p1.y-p2.y))+1) # bresenham num pixels drawn

    new_img = draw_thread(img, pnt1=p1, pnt2=p2, color=col)
    cur_diff = image_diff(new_img, trg)    # what is the difference to the target

    #blurredImg = ndimage.filters.gaussian_filter(new_img, 4)
    #cur_diff += image_diff(blurredImg, blurredTarget)    # what is the difference to the target
    #cur_diff *= 0.5

    #quality = (cur_diff - prevResidual)/(b_len**2)    # how much better did this line make the result
    quality = (cur_diff - prevResidual) / (b_len)
    #quality = (cur_diff - prevResidual) / length
    quality += abs(quality) * 10 * p2.heat # attenuate by previously visited
    return (quality, ind, cur_diff, length)


def Enhance(image, width, height):

    img = image.resize((width, height))
    enh = ImageEnhance.Contrast(img)
    #img = enh.enhance(1.25)
    bt  = ImageEnhance.Brightness(img)
    img = bt.enhance(0.80)

    return img.convert("L")





def remove_saturated_segments(fromIdx, neighbours, segCounts, maxCount):

    modified=False
    rem = set()
    for n in neighbours:
        seg = (min(fromIdx,n),max(fromIdx,n))
        if seg in segCounts and segCounts[seg] >= maxCount:
            rem.add(n)
            modified = True

    for r in rem:
        neighbours.remove(r)

    return modified


def remove_nail_collisions(pc, pt_id1, neighbors, maxDist):

    occl = set()
    for n in neighbors:
        occl.update( detect_points_on_line(pc, pt_id1, n, neighbors, maxDist) )

    for o in occl:
        neighbors.remove(o)


def detect_points_on_line(pc, ind1, ind2, neighbors, maxDist):
    """ returns all points of neighbors that are closer then maxDist from the line """

    occluded = set()

    a = list(neighbors)
    #a.remove(ind2)
    A = numpy.array([pc.p[i].asTupple() for i in a])
    B = numpy.repeat((pc.p[ind1].asTupple(),), len(a), axis=0)
    C = numpy.repeat((pc.p[ind2].asTupple(),), len(a), axis=0)

    lenBC = pc.p[ind1].dist(pc.p[ind2])
    #print "len", lenBC, ind1, ind2
    # project A onto BC (all the points onto the line
    CB = (C - B)
    D = CB / lenBC #/ numpy.sqrt((CB**2).sum(-1))[..., numpy.newaxis]   # normaized vector BC
    V = A - B
    t = (V*D).sum(-1)[...,numpy.newaxis] # dot product element wise
    P = B + D * t
    AP = (A - P)
    distSqr = (AP**2).sum(-1)[..., numpy.newaxis]
    onRay = distSqr <= maxDist * maxDist
    onLine = [(t[i][0], a[i]) for i in xrange(len(a)) if onRay[i][0] and t[i][0] >= 0.0]
    onLine.sort()
    #print [(a[i], onRay[i][0], t[i][0], distSqr[i][0]) for i in xrange(len(a))]
    #print onLine
    if len(onLine) > 0:
        for i in onLine[1:]:
            occluded.add(i[1])
        #occluded.add(ind2)

    return occluded


def array_to_PIL_rgb(imgArray):
    ar = imgArray*255
    ar = numpy.clip(ar, 0, 255, out=ar)
    img = Image.fromarray(ar.astype("uint8"))
    img = Image.merge("RGB", (img,img,img))
    return img

def PIL_to_array(pil_image):
    if pil_image.mode == "RGB":
        ret = numpy.array(pil_image.getchannel("R"), dtype="float32")
    else:
        ret = numpy.array(pil_image, dtype="float32")
    numpy.multiply(ret, 1.0/255, out=ret)
    return ret


def draw_points(pil_image, pnts, size=1):

    w = int(size-1)/2
    draw = ImageDraw.Draw(pil_image, mode="RGBA")
    if w < 1 :
        for p in pnts.p:

            if p.ignore:
                col = (0, 100, 255, 255)
            else:
                col = (255, int(255 * (1.0 - p.heat)), 0, 255)

            draw.point((p.x, p.y), col)
    else:
        for p in pnts.p:

            if p.ignore:
                col = (0, 100, 255, 120)
            else:
                col = (255, int(255 * (1.0 - p.heat)), 0, 120)
            draw.rectangle([p.x-w, p.y-w, p.x+w, p.y+w], fill=col, outline=(255,255,0,255))

    return pil_image


def draw_thread_rgb(image, pnt1, pnt2, color):

    img = image.copy()
    draw = ImageDraw.Draw(img, mode="RGBA")
    draw.line([pnt1.x, pnt1.y, pnt2.x, pnt2.y], width=2, fill=color)
    return img

def draw_thread(imageArray, pnt1, pnt2, color):

    img = array_to_PIL_rgb(imageArray)
    draw = ImageDraw.Draw(img, mode="RGBA")
    draw.line([pnt1.x, pnt1.y, pnt2.x, pnt2.y], width=2, fill=(color[0], color[0], color[0], color[1]))
    return PIL_to_array(img)


def image_diff(imageArray, targetArray):

    error = numpy.subtract(imageArray, targetArray)

    better = numpy.clip(error, -2000000000, 0)
    #worse  = numpy.multiply(numpy.clip(error, 0, 2000000000, out=error), 5, out=error)
    worse = numpy.multiply(numpy.clip(error, 0, 2000000000, out=error), 4, out=error)
    error = numpy.add(better, worse, out=error)
    #error = numpy.multiply(error, error, out=error) # error**2
    error = numpy.abs(error, out=error)

    return numpy.sum(error)










if __name__ == '__main__':
    import sys

    app = QtGui.QApplication(sys.argv)

    mpp = 0.3/500 # meters per pixel
    params = {
        "proc_width":500,       # image size
        "proc_height":500,      #
        "ppi": mpp,             # scale factor to real world, meters per pixel
        #"grid_resX":25,
        #"grid_resY":25,
        "nailDistMin": 6.0 / 1000.0 / mpp,      # minimum nail distance
        "nailDistMax": 16.0 / 1000.0 / mpp,     # maximum nail distance
        "nailDiameter": 1.5 / 1000.0 / mpp,       # diameter of the nail for avoidance
        "backgroundColor":0,                # canvas color
        "threadColor":(255, 20),                # string color
        "currentPoint" : 0,
        "lastPoint": -1,
        "start_at": (0.5,0),                     # closest point to start in rel coords [0..1]
        "inputImagePath": "einstein2.png",
        "edgesImagePath": "einstein_edges.png",   # optional
        "maxSegmentConnect": 5,             # max number of times two nails can be connected
        "maxIterations":14500
    }



    # 1 load image
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
t20: removed done pixels from target (set to black...questionable)
t21: masked outside pixels, normalized to l**2
t22: error not squared
t23: added blurred quality weight
t24: normalized with l not squared
t25: painfully added nail collision detection... this better be good (good speedup!!)
t26: fixed clamping, trying real len to normalize instead of b_len
t27: reduced penalty from *5 to *2
t28: added custom edge image, adjusted target image manually to improve area around eyes
t29: penalty *4, segment limit
t30: removed all! minDist nails


ideas: scipy draw line directly into numpy array, skip pillow conversions
        do error calculation of blurred picture (preliminary tests have shown no improvements, 10x slower) maybe try adding the blurred test to the non blurred test, so halftones in surrounding area add to the sharp comparison
        using max num connections for nail selection based on brighness
        find out how what causes endless loops
        add nail collision detection
        increase nail density in areas with higher frequency detail
        errors should have a magnitude more weight then improvements
        try binary quality, -1, 1 for good and bad pixels instead of distance
"""