from PySide import QtGui, QtCore
from PIL import Image, ImageQt, ImageDraw
import json
import numpy


class Viewer(QtGui.QMainWindow):

    def __init__(self, parameters):
        super(Viewer, self).__init__()

        self.parameters = parameters

        self.imageLabel = QtGui.QLabel()
        self.imageLabel.setBackgroundRole(QtGui.QPalette.Base)
        self.imageLabel.setSizePolicy(QtGui.QSizePolicy.Ignored,
                QtGui.QSizePolicy.Ignored)
        self.imageLabel.setScaledContents(False)
        self.imageLabel.setStyleSheet("border: 0px")
        self.imageLabel.setContentsMargins(0, 0, 0, 0)

        self.scrollArea = QtGui.QScrollArea()
        self.scrollArea.setBackgroundRole(QtGui.QPalette.Dark)
        self.scrollArea.setWidget(self.imageLabel)
        self.setCentralWidget(self.scrollArea)

        self.setWindowTitle("NailedIt - analyser")
        self.resize(600, 600)
        self.showImage(self.parameters["image"])

    def showImage(self, image, slot=0):

        if slot == 0:
            self.qim = ImageQt.ImageQt(image)   # don't let python clean up the data
            self.imageLabel.setPixmap(QtGui.QPixmap.fromImage(self.qim))
            self.imageLabel.adjustSize()




def load_nailedit(filepath):

    print 'loading "%s"'%filepath
    with open(filepath, 'r') as f:

        js = json.load(f)
        f.close()
        print 'done'
        return js


def draw_nails(nails, img):

    pnts = nails["3:nails"]
    path = nails["4:thread"]
    params = nails["2:parameters:"]

    backgroundCol = params["backgroundColor"]
    if not isinstance(backgroundCol, list):
        backgroundCol = (backgroundCol, backgroundCol, backgroundCol)

    stringCol = params["threadColor"]
    if len(stringCol) == 2:
        stringCol = (stringCol[0], stringCol[0], stringCol[0], stringCol[1])

    draw = ImageDraw.Draw(img, "RGBA")

    current_p = path[0]
    for i, next_p in enumerate(path[1:]):
        draw.line((pnts[current_p][0], pnts[current_p][1], pnts[next_p][0], pnts[next_p][1]), fill=stringCol, width=1)
        current_p = next_p
        if i % 1000 == 0:
            print 'drawing',i,"/",len(path)


if __name__ == '__main__':
    import sys

    filepath = "Q:\\Projects\\code\\nailedit\\t26.json"
    nails = load_nailedit(filepath)

    img = Image.new("RGB", (nails['2:parameters:']["proc_width"], nails["2:parameters:"]["proc_height"]))
    draw_nails(nails, img)

    app = QtGui.QApplication(sys.argv)

    params = {
        "image": img
    }

    imageViewer = Viewer(params)
    imageViewer.show()
    sys.exit(app.exec_())
