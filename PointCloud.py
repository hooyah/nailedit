import math
import random
import numpy as np
from scipy.spatial import Delaunay, ConvexHull
from PIL import ImageDraw

class Point2():

    def __init__(self, x=0.0, y=0.0, heat=0.0):
        self.x = x
        self.y = y
        self.heat = heat

    def __repr__(self):
        return "({:.3f}, {:.3f})".format(self.x, self.y)

    def dist(self, other):
        return math.sqrt((self.x-other.x)**2 + (self.y-other.y)**2)

    def clamped(self, minx, maxx, miny, maxy):
        return Point2( max(minx, min(maxx, self.x)), max(miny, min(maxy, self.y)), self.heat )

    def __mul__(self, other):
        return Point2(self.x*other, self.y*other, self.heat)

    def __add__(self, other):
        return Point2(self.x+other.x, self.y+other.y, self.heat)

    def __sub__(self, other):
        return Point2(self.x-other.x, self.y-other.y, self.heat)


def remap(val, from_min, from_max, to_min, to_max):
    return (((val - from_min) * (to_max - to_min)) / (from_max - from_min)) + to_min


class PointCloud():

    def __init__(self, dimx, dimy):
        self.p = []
        self.width = dimx
        self.height = dimy


    def addGrid(self, w, h, offset=0.5):

        # jittered
        #return [Point2(float(x)/(w-1)+random.uniform(-.5,.5)*(1.0/(w-1)),float(y)/(h-1)+random.uniform(-.5,.5)*(1.0/(h-1))) for x in xrange(w) for y in xrange(h)]

        # offset
        pt = [Point2(float(x) / (w - 1) + ((offset / (w - 1)) if y % 2 else 0), float(y) / (h - 1)) for x in xrange(w) for y in xrange(h)]
        self.p += [Point2(p.x*self.width, p.y*self.height) for p in pt if p.x <= 1.0]

    def addRandom(self, num):

        random.seed(1234)
        self.p += [Point2(random.uniform(0, self.width), random.uniform(0,self.height)) for n in xrange(num)]

    def cool(self, f=0.1):

        for pnt in self.p:
            pnt.heat = pnt.heat * (1.0-f)


    def relax(self, image=None, iterations=50, detail_img=None):

        npp = np.array([[pnt.x,pnt.y] for pnt in self.p])
        tri = Delaunay(npp)

        # mask the autside border
        mask = set()
        for t_ind, ns in enumerate(tri.neighbors):
            for n_ind, n in enumerate(ns):
                if n == -1:
                    for i in [0,1,2]:
                        if i != n_ind:
                            mask.add(tri.simplices[t_ind][i])

        if image:
            drawn = set()
            draw = ImageDraw.Draw(image)
            for t in tri.simplices:
                pp = (tri.points[t[0]], tri.points[t[1]], tri.points[t[2]])
                for i,j in [(0,1),(1,2),(2,0)]:
                    pair = (min(t[i], t[j]), max(t[i], t[j]))
                    if not pair in drawn:
                        draw.line([pp[i][0], pp[i][1], pp[j][0],pp[j][1]], (180,150,0))
                        drawn.add(pair)


        # average vertex positions
        """
        for it in range(iterations):
            for i, pnt in enumerate(self.p):
                if i not in mask:
                    neighbours = tri.vertex_neighbor_vertices[1][tri.vertex_neighbor_vertices[0][i]:tri.vertex_neighbor_vertices[0][i+1]]
                    # put it smack in the middle
                    mid = np.mean(tri.points[neighbours], axis=0)
                    self.p[i] = Point2(mid[0], mid[1])
            for i in xrange(len(tri.points)):
                tri.points[i][0] = self.p[i].x
                tri.points[i][1] = self.p[i].y
        """
        # try to average edge length
        numEdges = 0
        targetLength = 0.0
        edgedone = set()
        for t in tri.simplices:
            for i,j in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]:
                pair = (min(i, j), max(i, j))
                if pair not in edgedone:
                    edgedone.add(pair)
                    targetLength += self.p[i].dist(self.p[j])
                    numEdges += 1
        targetLength /= numEdges
        print "targetLen", targetLength

        targetLength *= 1.1 # make it squeeze into the corners
        ease = 0.25 # only move it this much of the desired distance
        edgedone = set()
        for t in tri.simplices:
            for i,j in [(t[0],t[1]),(t[1],t[2]),(t[2],t[0])]:
                pair = (min(i, j), max(i, j))
                if not pair in edgedone:
                    l = self.p[i].dist(self.p[j])
                    f = (targetLength/l)*ease + (1.0-ease)
                    # scale edge around midpoint
                    mp = (self.p[i] + self.p[j]) * 0.5

                    # scale lenght by detail image
                    if detail_img:
                        det = detail_img.getpixel((mp.x, mp.y))
                        det = remap(det, 0, 255, 1.0, 0.6)
                        f *= det

                    self.p[i] = (self.p[i] - mp) * f + mp
                    self.p[j] = (self.p[j] - mp) * f + mp
                    self.p[i] = self.p[i].clamped(0.0, self.width, 0.0, self.height)
                    self.p[j] = self.p[j].clamped(0.0, self.width, 0.0, self.height)
                    edgedone.add(pair)

        #print len(self.p), len(tri.points), np.max(tri.simplices)



    def findNeighbours(self, pnt, max_radius):

        #grid
        # for now just return everything but the given point
        #ret = range(len(pnts))
        #del ret[pnt]
        ret = [i for i in xrange(len(self.p)) if (i != pnt and self.p[pnt].dist(self.p[i]) < max_radius)]

        return ret
