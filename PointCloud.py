import math
import random
import numpy as np
from scipy.spatial import Delaunay, cKDTree
from PIL import ImageDraw

class Point2(object):

    def __init__(self, x=0.0, y=0.0, heat=0.0, ignore=False, numConnects=0):
        self.x = x
        self.y = y
        self.heat = heat
        self.ignore = ignore
        self.numConnects = numConnects

    def __repr__(self):
        return "({:.3f}, {:.3f})".format(self.x, self.y)

    def length(self):
        return math.sqrt(self.x**2 + self.y**2)

    def dist(self, other):
        return math.sqrt((self.x-other.x)**2 + (self.y-other.y)**2)

    def dist2(self, other):
        return (self.x-other.x)**2 + (self.y-other.y)**2

    def clamped(self, minx, maxx, miny, maxy):
        return Point2( max(minx, min(maxx, self.x)), max(miny, min(maxy, self.y)), self.heat, self.ignore, self.numConnects )

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def cross25D(self, other):
        """ returns the z component of the cross product with the two vectors assumed to lay in the xy plane with z=0 """

        return self.x * other.y - other.x * self.y


    def asTupple(self):
        return (self.x, self.y)

    def __mul__(self, other):
        if isinstance(other, Point2):
            return Point2(self.x * other.x, self.y * other.x, self.heat, self.ignore, self.numConnects)
        else:
            return Point2(self.x*other, self.y*other, self.heat, self.ignore, self.numConnects)

    def __div__(self, other):
        return Point2(self.x/other, self.y/other, self.heat, self.ignore, self.numConnects)

    def __add__(self, other):
        return Point2(self.x+other.x, self.y+other.y, self.heat, self.ignore, self.numConnects)

    def __sub__(self, other):
        return Point2(self.x-other.x, self.y-other.y, self.heat, self.ignore, self.numConnects)



class Circle2(object):

    def __init__(self, x, y, r):

        self.p = Point2(x,y)
        self.r = r

    def tangentP(self, vect):
        """ given a vector, returns the two points of tangency"""
        norm = Point2(vect.y, -vect.x)/vect.length()
        return [ self.p+norm*self.r, self.p-norm*self.r ]

    def normals(self, vect):
        """ given a vector, returns two vectors pointing to the two points of tangency"""
        norm = Point2(vect.y, -vect.x) / vect.length() * self.r
        return [norm, norm*-1]

    def intersectRay(self, p1, p2):
        a = (p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2
        b = 2 * ((p2.x - p1.x) * (p1.x - self.p.x) + (p2.y - p1.y) * (p1.y - self.p.y))
        c = (p1.x - self.p.x) ** 2 + (p1.y - self.p.y) ** 2 - self.r ** 2
        bsq = b**2 - 4*a*c

        #print "intersect",self.p.x, self.p.y, self.r
        #print a, b, c, bsq
        if bsq >=0 and a != 0.0:
            bsq = math.sqrt(bsq)
            t = [ (-b - bsq)/(2*a), (-b + bsq)/(2*a) ]
            rp1 = (p2-p1)*t[0]+p1
            rp2 = (p2-p1)*t[1]+p1
            #print "ip:", t[0], t[1], rp1, rp2
            if rp1.dist2(rp2) < 1e-4:
                return [rp1]
            else:
                return [rp1,rp2]
        return []


def project(a, b, c):
    """ project a on bc"""

    d = (c - b) / c.dist(b)
    v = a - b
    t = v.dot(d)
    p = b + d * t
    return p


def project_param(a, b, c):
    """ project a on bc, returns t as  p = (1-t)*a + t*b """

    t = (a - b).dot(c - b) / float(b.dist2(c))
    return t

def intersect_line(a1, a2, b1, b2):
    """
    :param a1: Point2
    :param a2: Point2
    :param b1: Point2
    :param b2: Point2
    :return: s, t, i1=a1 + t*a2,  i2=b1 + s*b2
    """

    s1_x = float(a2.x - a1.x)
    s1_y = float(a2.y - a1.y)
    s2_x = float(b2.x - b1.x)
    s2_y = float(b2.y - b1.y)

    dr = (-s2_x * s1_y + s1_x * s2_y)
    if dr != 0:
        s =  (s2_x * (a1.y - b1.y) - s2_y * (a1.x - b1.x)) / dr
        t = (-s1_y * (a1.x - b1.x) + s1_x * (a1.y - b1.y)) / dr
    else:
        s = 1e6
        t = 1e6

    return s, t



def remap(val, from_min, from_max, to_min, to_max):
    return (((val - from_min) * (to_max - to_min)) / (from_max - from_min)) + to_min


class PointCloud(object):

    def __init__(self, dimx, dimy):
        self.p = []
        self.width = dimx
        self.height = dimy
        self.kd = None


    def addGrid(self, w, h, offset=0.5):

        # jittered
        #return [Point2(float(x)/(w-1)+random.uniform(-.5,.5)*(1.0/(w-1)),float(y)/(h-1)+random.uniform(-.5,.5)*(1.0/(h-1))) for x in xrange(w) for y in xrange(h)]

        # offset
        pt = [Point2(float(x) / (w - 1) + ((offset / (w - 1)) if y % 2 else 0), float(y) / (h - 1)) for x in xrange(int(w)) for y in xrange(int(h))]
        self.p += [Point2(p.x*(self.width-0.01), p.y*(self.height-0.01)) for p in pt if p.x <= 1.0]

    def addRandom(self, num):

        random.seed(1234)
        self.p += [Point2(random.uniform(0, float(self.width)-0.01), random.uniform(0,float(self.height)-0.01)) for n in xrange(num)]

    def addFromList(self, coordList):

        self.p += [Point2(l[0],l[1]) for l in coordList]


    def translate(self, x, y):

        offs = Point2(x, y)
        for pt in self.p:
            pt += offs

    def scale(self, sx, sy):

        for pt in self.p:
            pt.x *= sx
            pt.y *= sy


    def cool(self, f=0.1):

        for pnt in self.p:
            pnt.heat = pnt.heat * (1.0-f)

    def heat(self, temp):

        for pnt in self.p:
            pnt.heat = temp


    def maskPoints(self, maskImg, thresh):

        for pt in self.p:
            msk = maskImg.getpixel((pt.x, pt.y))/255.0
            if msk <= thresh:
                pt.ignore = True



    def scatterOnMask(self, maskImg, numPoints, minDist, threshold = 0.2):

        print 'scattering',numPoints,'points'
        random.seed(4826)
        # brute force
        num = 0
        fail = 0
        #while num < numPoints and fail < numPoints*10:
        #    pt = Point2(random.uniform(1, self.width-1.01), random.uniform(1,self.height-1.01))
        #    msk = maskImg[int(pt.y)][int(pt.x)]
        #    if msk >= threshold :
        #        if len(self.p) and self.p[self.closestPoint(pt.x, pt.y)].dist(pt) < minDist:
        #            fail += 1
        #            continue
        #        num += 1
        #        self.p.append(pt)

        np.random.seed(64726)
        f = maskImg.flatten()
        interesting = np.where(f >= threshold)[0]
        np.random.shuffle(interesting)
        for i in interesting:
            pt = Point2(float(i % maskImg.shape[1]), float(i / maskImg.shape[1]))
            if len(self.p)==0 or self.closestPoint(pt.x, pt.y)[1] >= minDist:
                self.p.append(pt)
                num += 1
                if num >= numPoints:
                    break
            else:
                fail += 1
                if fail >= numPoints*10:
                    break

        print "successfully scattered", num, "of", numPoints, "points"



    def relax(self, image, iterations, detail_img, minDist, maxDist):

        npp = np.array([[pnt.x,pnt.y] for pnt in self.p])
        tri = Delaunay(npp)

        msk = [pt.heat for pt in self.p]
        # mask the autside border
        for t_ind, ns in enumerate(tri.neighbors):
            for n_ind, n in enumerate(ns):
                if n == -1:
                    for i in [0,1,2]:
                        if i != n_ind:
                            msk[tri.simplices[t_ind][i]] = 1.0

        # draw mesh
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
        print "targetLen", targetLength, "min", minDist, "max", maxDist

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
                    if not isinstance(detail_img, type(None)):
                        #det = detail_img.getpixel((mp.x, mp.y))
                        det = detail_img[int(mp.y)][int(mp.x)]
                        det = remap(det, 0., 1., maxDist/l, minDist/l)
                        f *= 1.0 - det / iterations

                    if msk[i] <= 0.0 and not self.p[i].ignore:
                        self.p[i] = (self.p[i] - mp) * f + mp
                        self.p[i] = self.p[i].clamped(0.0, self.width-0.01, 0.0, self.height-0.01)
                    if msk[j] <= 0.0 and not self.p[j].ignore:
                        self.p[j] = (self.p[j] - mp) * f + mp
                        self.p[j] = self.p[j].clamped(0.0, self.width-0.01, 0.0, self.height-0.01)
                    edgedone.add(pair)

        #print len(self.p), len(tri.points), np.max(tri.simplices)


    def closestPoint(self, x, y, thatsNot=-1):

        #if not self.kd:
        #    self.npp = np.array([(pt.x, pt.y) for pt in self.p])
        #    self.kd = cKDTree(self.npp)

        to = Point2(x, y)
        dst = [(pnt.dist2(to), i) for i,pnt in enumerate(self.p) if not pnt.ignore and i != thatsNot]
        dst.sort()
        return dst[0][1], math.sqrt(dst[0][0])

    def closestPoints(self, x, y, radius, thatsNot=-1):

        radius = radius*radius
        to = Point2(x, y)
        dst = [(pnt.dist2(to), i) for i,pnt in enumerate(self.p) if not pnt.ignore and i != thatsNot]
        ret = [d[1] for d in dst if d[0] <= radius]
        return ret


    def findNeighbours(self, pntInd, max_radius):

        #grid
        # for now just return everything but the given point
        #ret = range(len(pnts))
        #del ret[pnt]
        r = max_radius**2
        ret = [i for i in xrange(len(self.p)) if (i != pntInd and not self.p[i].ignore and self.p[pntInd].dist2(self.p[i]) < r)]

        random.seed(73674)
        random.shuffle(ret)
        return ret


    def findPointsNearRay(self, p1, p2, maxDist):
        """ returns all points that are closer then maxDist from the line """

        A = np.array([pt.asTupple() for pt in self.p])
        B = np.repeat((p1.asTupple(),), len(A), axis=0)
        C = np.repeat((p2.asTupple(),), len(A), axis=0)

        lenBC = p1.dist(p2)

        # project A onto BC (all the points onto the line)
        CB = C - B
        D = CB / lenBC
        V = A - B
        t = (V*D).sum(-1)[...,np.newaxis] # dot product element wise
        P = B + D * t
        AP = (A - P)
        distSqr = (AP**2).sum(-1)[..., np.newaxis]

        maxDist = maxDist**2
        onLine = [(t[i][0]/lenBC, i) for i in xrange(len(A)) if distSqr[i][0] <= maxDist]

        return onLine
