from PySide import QtGui, QtCore
from PIL import Image, ImageQt, ImageDraw
import os, json
import numpy
import svgwrite
#import PointCloud
from PointCloud import Point2, PointCloud, intersect_line
from gcode import Mach3 as gc
import re

class Viewer(QtGui.QMainWindow):

    def __init__(self, parameters, scale=Point2(1.0,1.0)):
        super(Viewer, self).__init__()

        #self.layout = QtGui.QBoxLayout(QtGui.QBoxLayout.TopToBottom, None)
        self.layout = QtGui.QVBoxLayout()
        self.setLayout(self.layout)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.parameters = parameters


        self.multiWidget = QtGui.QScrollArea(self)
        self.multiWidget.setSizePolicy(QtGui.QSizePolicy.Expanding,
                QtGui.QSizePolicy.Expanding)
        self.bl = QtGui.QVBoxLayout(self.multiWidget)

        #self.bn = QtGui.QPushButton(self, text="Hello")
        #self.bn.setSizePolicy(QtGui.QSizePolicy.Expanding,
        #        QtGui.QSizePolicy.Expanding)
        #self.layout.addWidget(self.bn)

        self.imageLabel = QtGui.QLabel(self.multiWidget)
        self.imageLabel.setSizePolicy(QtGui.QSizePolicy.Expanding,
                QtGui.QSizePolicy.Expanding)
        self.imageLabel.setScaledContents(True)
        self.bl.addWidget(self.imageLabel)
        #self.imageLabel.setStyleSheet("border: 0px")
        #self.imageLabel.setContentsMargins(0, 0, 0, 0)
        #self.imageLabel.setText("nothing loaded")

        self.imageLabel2 = QtGui.QLabel(self.multiWidget)
        self.imageLabel2.setSizePolicy(QtGui.QSizePolicy.Expanding,
                QtGui.QSizePolicy.Minimum)
        self.imageLabel2.setScaledContents(False)
        #self.imageLabel2.setStyleSheet("border: 0px")
        #self.imageLabel2.setContentsMargins(0, 0, 0, 0)
        self.bl.addWidget(self.imageLabel2)

        self.imageLabel3 = QtGui.QLabel(self.multiWidget)
        #self.imageLabel3.setSizePolicy(QtGui.QSizePolicy.Ignored,
        #        QtGui.QSizePolicy.Ignored)
        #self.imageLabel3.setScaledContents(False)
        #self.imageLabel3.setStyleSheet("border: 0px")
        #self.imageLabel3.setContentsMargins(0, 0, 0, 0)
        self.bl.addWidget(self.imageLabel3)

        self.cutoffSlider = QtGui.QSlider(self.multiWidget, orientation=QtCore.Qt.Horizontal)
        self.cutoffSlider.sliderReleased.connect(self.updateNailsImage)
        self.bl.addWidget(self.cutoffSlider)


        self.setCentralWidget(self.multiWidget)
        #self.setCentralWidget(self.imageLabel)

        self.setWindowTitle("NailedIt - analyser")
        self.resize(600, 600)

        self.scaleFactor = scale
        self.deviation = []
        self.debug_cnt = 1

        self.timer = QtCore.QTimer(self)
        self.connect(self.timer, QtCore.SIGNAL("timeout()"), self.debug)
        self.timer.setSingleShot(True)
        #self.timer.start(0)


        #   menu
        fileMenu = self.menuBar().addMenu("File")
        fileMenu.addAction("Open", self.openFile)
        fileMenu.addSeparator()
        fileMenu.addAction("Export Nails SVG", self.saveNailsSVG)
        fileMenu.addAction("Generate Gcode", self.generateGcode)

        optionsMenu = self.menuBar().addMenu("Options")
        self.showNailsAction = QtGui.QAction("show nails", optionsMenu, checkable=True, triggered=self.updateNailsImage)
        optionsMenu.addAction(self.showNailsAction)
        self.showOverlaps = QtGui.QAction("calculate overlaps", optionsMenu, checkable=True, triggered=self.updateNailsImage)
        optionsMenu.addAction(self.showOverlaps)
        self.reversePaths = QtGui.QAction("reverse path", optionsMenu, checkable=True, triggered=self.reverseTriggered)
        optionsMenu.addAction(self.reversePaths)


        #self.layout.addWidget(self.menu)
        #self.layout.addWidget(self.scrollArea)



        self.showImage(Image.new("RGB", (500,500)))
        #if "image" in self.parameters:
        #    self.showImage(self.parameters["image"])


    def reverseTriggered(self):

        self.deviation = []
        self.updateNailsImage()


    def showImage(self, image, slot=0):

        if slot == 0:
            self.qim = ImageQt.ImageQt(image)   # don't let python clean up the data
            self.imageLabel.setPixmap(QtGui.QPixmap.fromImage(self.qim))
            self.imageLabel.adjustSize()
        elif slot == 1:
            self.qim2 = ImageQt.ImageQt(image)   # don't let python clean up the data
            self.imageLabel2.setPixmap(QtGui.QPixmap.fromImage(self.qim2))
            self.imageLabel2.adjustSize()
        elif slot == 2:
            self.qim3 = ImageQt.ImageQt(image)   # don't let python clean up the data
            self.imageLabel3.setPixmap(QtGui.QPixmap.fromImage(self.qim3))
            self.imageLabel3.adjustSize()



    def openFile(self):

        filename = QtGui.QFileDialog.getOpenFileName(self, "Open Nailedit File", "./", "Nailedit (*.json)")[0]
        if filename:
            js = load_nailedit(filename)
            if js:
                self.parameters["Nails"] = js
                print "num nails:", len(js["3:nails"]), "string length:", js['1:summary']["thread length"]
                self.parameters["filename"] = filename
                self.setWindowTitle("NailedIt - analyser (%s)"%os.path.basename(filename))
                self.cutoffSlider.setMaximum(len(js["4:thread"]))
                self.cutoffSlider.setSliderPosition(self.cutoffSlider.maximum())
                self.deviation = []
                self.updateNailsImage()


    def updateDeviation(self):

        if "Nails" in self.parameters and self.deviation:
            nails = self.parameters["Nails"]
            w = nails['2:parameters:']["proc_width"]
            h = nails['2:parameters:']["proc_height"]
            img = Image.new("RGB", (w, 100), (255,255,255))

            draw = ImageDraw.Draw(img, "RGB")
            for x in xrange(w):

                d = int(float(x)/w * len(self.deviation))
                v = float(self.deviation[d])/self.deviation[0] * 100

                draw.line((x, 100, x, 100-v), fill=(255,0,0), width=1)

            self.showImage(img, slot=1)


    def debug(self):
        #################################################################################################
        if "Nails" in self.parameters:

            js = self.parameters["Nails"]
            nails = js["3:nails"]
            path =  js["4:thread"]

            img = self.parameters["image"].copy()

            inters = 0
            up_to = self.debug_cnt
            p4 = Point2(nails[path[up_to]][0], nails[path[up_to]][1])
            p3 = Point2(nails[path[up_to-1]][0], nails[path[up_to-1]][1])

            draw = ImageDraw.Draw(img, "RGBA")

            if up_to > 1:
                p1 = Point2(nails[path[0]][0], nails[path[0]][1])
                for c in path[1:up_to]:
                    p2 = Point2(nails[c][0], nails[c][1])
                    s, t = intersect_line(p1, p2, p3, p4)
                    if 0 < s < 1 and 0 < t < 1:
                        inters += 1
                        draw.line((p1.x, p1.y, p2.x, p2.y), (0,255,0,255))
                    p1 = p2

                draw.line((p3.x, p3.y, p4.x, p4.y), (255,0,0, 255), width=2)

            self.debug_cnt += 1
            self.showImage(img)
            print "intersects", inters
        self.timer.start(1000 if self.debug_cnt % 10 == 0 else 50)

    def updateNailsImage(self):

        if "Nails" in self.parameters:
            nails = self.parameters["Nails"]
            w = nails['2:parameters:']["proc_width"]
            h = nails['2:parameters:']["proc_height"]
            img = self.parameters["image"] = Image.new("RGB", (w, h))

            trg = Image.open(nails['2:parameters:']["inputImagePath"])
            if trg:
                trg = trg.resize(img.size)

            ret = draw_nails(nails, img, showNails=self.showNailsAction.isChecked(), targetImage=trg if not self.deviation else None, lastString=self.cutoffSlider.value(), reversed=self.reversePaths.isChecked())
            if ret:
                self.deviation = ret
            self.showImage(self.parameters["image"])
            self.updateDeviation()

            if self.showOverlaps.isChecked():
                ovl = self.parameters["overlap"] = Image.new("RGB", (w, h))
                draw_overlap(nails, ovl, lastString=self.cutoffSlider.value(), reversed=self.reversePaths.isChecked())
                self.showImage(ovl, slot=2)

    def saveNailsSVG(self):

        if "Nails" in self.parameters:
            filename = QtGui.QFileDialog.getSaveFileName(self, "Export Nails", "./", "SVG (*.svg)")[0]
            if filename:
                save_nails_SVG(self.parameters["Nails"], filename, self.scaleFactor)


    def generateGcode(self):

        if not "Nails" in self.parameters:
            return

        filename = QtGui.QFileDialog.getSaveFileName(self, "Generate Gcode", "./", "gcode (*.tap)")[0]
        if not filename:
            return

        js = self.parameters["Nails"]
        nails = js["3:nails"]
        path =  js["4:thread"]
        w = js["2:parameters:"]["proc_width"]
        h = js["2:parameters:"]["proc_height"]
        sf = Point2(1,1) * self.scaleFactor * 1000.0 * js["2:parameters:"]["ppi"] # pixels to millimeters
        minNailDist = 2.8 # keep this distance to the nails

        origin = Point2(0,0)
        pc = PointCloud(1,1)
        pc.addFromList(nails)
        cp = pc.closestPoint(origin.x, origin.y)
        print "origin", nails[cp[0]]

        engine = gc()
        code = engine.generateStringPath(os.path.basename(self.parameters["filename"]), nails, path[:self.cutoffSlider.value()], minNailDist, scaleFactor=sf, origin=Point2(0,0), startPosition=Point2(0.5, -0.1/6)*w)
        #code = engine.generateStringPath(os.path.basename(self.parameters["filename"]), nails, path[:20], minNailDist, scaleFactor=sf, origin=Point2(0,0), startPosition=Point2(0.5, -0.1)*w)
        if code:
            with open(filename, "w") as f:
                f.write(code)
                f.close()
                print "written gcode to", filename

            img = self.drawGcode(self.parameters["image"], code, Point2(1.0/sf.x,1.0/sf.y), Point2(0,0))
            self.showImage(img)


    def drawGcode(self, img, code, scaleFact, origin):

        drw = ImageDraw.Draw(img)
        mpx,mpy,mpz = 0,0,0
        lines = code.split('\n')
        for line in lines:
            x = re.search(r"(Y)([0-9.-]+)", line)
            x = float(x.group(2)) if x else mpx
            y = re.search(r"(X)([0-9.-]+)", line)
            y = float(y.group(2)) if y else mpy
            z = re.search(r"(Z)([0-9.-]+)", line)
            z = float(z.group(2)) if z else mpz
            i = re.search(r"(J)([0-9.-]+)", line)
            i = float(i.group(2)) if i else None
            j = re.search(r"(I)([0-9.-]+)", line)
            j = float(j.group(2)) if j else None


            if line.startswith("G0 "):
                drw.line((mpx * scaleFact.x + origin.x, mpy * scaleFact.y + origin.y,
                            x * scaleFact.x + origin.x,   y * scaleFact.y + origin.y), (0,0,255))
                mpx, mpy, mpz = x, y, z
            elif line.startswith("G1 "):
                drw.line((mpx * scaleFact.x + origin.x, mpy * scaleFact.y + origin.y,
                            x * scaleFact.x + origin.x,   y * scaleFact.y + origin.y), (255,50,0))
                mpx, mpy, mpz = x, y, z
            elif line.startswith("G2 ") or line.startswith("G3 "):
                r = Point2(Point2(i,j).length() * scaleFact.x, Point2(i,j).length() * scaleFact.y)
                drw.arc(((mpx+i) * scaleFact.x + origin.x - abs(r.x), (mpy+j) * scaleFact.y + origin.y - abs(r.x),
                         (mpx+i) * scaleFact.x + origin.x + abs(r.x), (mpy+j) * scaleFact.y + origin.y + abs(r.x)), 0, 360, (255,0,0))
                mpx, mpy, mpz = x, y, z

        return img


def load_nailedit(filepath):

    print 'loading "%s"'%filepath
    with open(filepath, 'r') as f:

        js = json.load(f)
        f.close()
        print 'done'
        return js



def save_nails_SVG(nails, filename, scale):

    svg = svgwrite.Drawing(filename, profile='tiny')
    pnts = nails["3:nails"]
    r = 1
    ptmm = nails["2:parameters:"]["ppi"] * 1000
    if "nailDiameter" in nails["2:parameters:"]:
        r = nails["2:parameters:"]["nailDiameter"] * ptmm

    for p in pnts:
        svg.add(svg.circle((p[0]*ptmm*scale.x*svgwrite.mm, p[1]*ptmm*scale.y*svgwrite.mm), r*svgwrite.mm))
    svg.save()
    print "saved as ", filename
    print "sc:", scale, "dim:", nails["2:parameters:"]["proc_width"]*ptmm*scale, "x", nails["2:parameters:"]["proc_height"]*ptmm*scale


def draw_nails(nails, img, showNails=True, lastString=10 ** 7, targetImage=None, reversed=False):

    pnts = nails["3:nails"]
    path = nails["4:thread"] if not reversed else nails["4:thread"][::-1]
    params = nails["2:parameters:"]

    backgroundCol = params["backgroundColor"]
    if not isinstance(backgroundCol, list):
        backgroundCol = (backgroundCol, backgroundCol, backgroundCol)
    else:
        backgroundCol = tuple(backgroundCol)

    stringCol = params["threadColor"]
    if len(stringCol) == 2:
        stringCol = [stringCol[0], stringCol[0], stringCol[0], stringCol[1]]
    print stringCol


    # over sampling
    mmpp = params["ppi"]/0.001 #mm per pixel
    threadThickness = 0.3 # mm
    oversampling = 1
    w = params["proc_width"] * mmpp
    h = params["proc_width"] * mmpp
    iw = int(w / threadThickness * oversampling)
    ih = int(h / threadThickness * oversampling)
    img_hi = Image.new("RGB", (int(iw), int(ih)))
    scl = mmpp / threadThickness * oversampling
    stringCol[3] = 255
    width = oversampling
    print "buffer img %dx%d"%(iw,ih), "thread:",


    draw = ImageDraw.Draw(img_hi, "RGBA")
    calcDeviation = targetImage != None

    if calcDeviation:
        lastString = 10 ** 7
        target_np = numpy.array(targetImage.getchannel("R"), dtype="float32")
        dev = []

    current_p = path[0]
    for i, next_p in enumerate(path[1:lastString]):
        draw.line((pnts[current_p][0]*scl, pnts[current_p][1]*scl, pnts[next_p][0]*scl, pnts[next_p][1]*scl), fill=tuple(stringCol), width=oversampling)

        if calcDeviation and not i % 50:
            current_np = numpy.array(img_hi.resize(targetImage.size, resample=Image.BICUBIC).getchannel("R"), dtype="float32")
            deviation = numpy.subtract(target_np, current_np, out=current_np)
            dev.append( numpy.sum(deviation) / 255 )

        if i % 1000 == 0:
            print 'drawing',i,"/",len(path)

        current_p = next_p

    # resize to target image size
    img.paste(img_hi.resize(img.size, resample=Image.BICUBIC))

    if showNails:
        draw = ImageDraw.Draw(img, "RGBA")
        for pt in pnts:
            draw.rectangle([pt[0] - 2, pt[1] - 2, pt[0] + 2, pt[1] + 2], fill=(255,120,0,100), outline=(255, 255, 0, 255))

    if calcDeviation:
        return dev



def draw_overlap(nails, img, lastString=10 ** 7, reversed=False):

    pnts = nails["3:nails"]
    path = nails["4:thread"] if not reversed else nails["4:thread"][::-1]
    params = nails["2:parameters:"]


    img_line = Image.new("F", img.size)
    draw = ImageDraw.Draw(img_line)
    overlap_np = numpy.array(img_line)

    nailConnects = [0]*len(pnts)
    current_p = path[0]
    nailConnects[current_p] += 1
    for i, next_p in enumerate(path[1:lastString]):
        img_line.paste(0, (0,0)+img_line.size)
        draw.line((pnts[current_p][0], pnts[current_p][1], pnts[next_p][0], pnts[next_p][1]), fill=1.0, width=1)
        draw.rectangle((pnts[current_p][0]-1, pnts[current_p][1]-1, pnts[current_p][0]+1, pnts[current_p][1]+1), fill=0.0)
        draw.rectangle((pnts[next_p][0]-1, pnts[next_p][1]-1, pnts[next_p][0]+1, pnts[next_p][1]+1), fill=0.0)

        current_np = numpy.array(img_line, dtype="float32")
        numpy.add(current_np, overlap_np, out=overlap_np)

        if i % 1000 == 0:
            print 'overlap',i,"/",len(path)

        nailConnects[next_p] += 1
        current_p = next_p

    # resize to target image size
    print "max overlaps", overlap_np.max(), "max nail connects", max(nailConnects)
    overlap_np = (overlap_np / overlap_np.max()) * 255.0
    overlap = Image.fromarray(overlap_np.astype("uint8"))
    overlap = Image.merge("RGB", (overlap, overlap, overlap))

    redlut = tuple(max(0, (i-85)*3) if i < 170 else 255 for i in xrange(256))
    greenlut = tuple(max(0, (i-170)*3) for i in xrange(256))
    bluelut = tuple( i*3 if i < 85 else max(0,255-i*3) for i in xrange(256) )

    overlap = overlap.point((redlut + greenlut + bluelut))
    img.paste(overlap)




if __name__ == '__main__':
    import sys

    filepath = "Q:\\Projects\\code\\nailedit\\t26.json"
    #nails = load_nailedit(filepath)

    #img = Image.new("RGB", (nails['2:parameters:']["proc_width"], nails["2:parameters:"]["proc_height"]))
    #draw_nails(nails, img)

    app = QtGui.QApplication(sys.argv)

    params = {
        #"image": img
    }

    imageViewer = Viewer(params, 29.0/30.0)
    imageViewer.show()
    sys.exit(app.exec_())
