from PointCloud import *
import math

class Mach3:


    def __init__(self, nails, connects, scaleFactor, origin):

        self.pc = None
        self.connects = None

        self.origin = None
        self.scale  = None

        self.safeZ = 15.0
        self.feedRate = 2500
        self.drillFeed = 1000
        self.mposX = 0
        self.mposY = 0
        self.mposZ = 0
        self.feed = 0

        self.startZ = 0.6
        self.threadThickness = 0.25
        self.maxZ = 7.0
        self.ramp_angle = 15 # degrees
        self.z_hop = 3

        self.nailThreadHeights = None
        self.ramp_A_p = None
        self.ramp_A_z = 0
        self.ramp_B_p = None
        self.ramp_B_z = 0

        self.epsilon = 0.1      # moves smaller than this will be omitted (currently only z)


        self.pc = PointCloud(100,100)
        self.pc.addFromList(nails)
        self.connects = connects
        self.connectHeight = [0.0]*len(connects)
        self.pc.translate(-origin.x, -origin.y)
        self.pc.scale(scaleFactor.x, scaleFactor.y)

        self.program = None
        self.resetProgram()



    def resetProgram(self):

        self.program = []


    def estrip(self, multilineString):

        s = multilineString.split('\n')
        ss = [st.strip() for st in s]
        return '\n'.join(ss)


    def addStartup(self):

        ret = """
            G90 G94 G91.1 G40 G49 G17
            G21
            """.format(saveZ=self.safeZ)
        self.program.append(self.estrip(ret))


    def addComment(self, comment, prepend=False):

        if prepend:
            self.program.insert(0, "(%s)"%comment)
        else:
            self.program.append("(%s)"%comment)

    def spindleOn(self, onoff=True):
        self.program.append("M3" if onoff else "M5")

    def dwell(self, seconds):
        self.program.append("G4 P{:.1f}".format(seconds))

    def addPause(self):
        self.program.append("M1")

    def addEnd(self):
        self.program.append("\nM30")



    def cannedDrillCycle(self, points, startz, depthz, feed):
        """ flipped X/Y !"""

        p = points[0]
        self.program.append("G98 G81 X{x:.2f} Y{y:.2f} Z{z:.2f} R{r:.2f} F{f:.1f}".format(x=p.y, y=p.x, z=depthz, r=startz, f=feed))
        for p in points[1:]:
            self.program.append("X{x:.2f} Y{y:.2f}".format(x=p.y, y=p.x))




    def moveTo_ramped(self, x=None, y=None, feed=None, rapid=False):
        """ flipped X/Y"""

       # if self.z_hop == 0:
       #     self.moveTo(x, y, self.ramp_B_z, feed, rapid)
       #     return

        if x is None:
            x = self.mposX
        if y is None:
            y = self.mposY

        #print "moveto_ramped:", x, y,

        # calculate ramped_z, subdivide at ramp start/end if necessary
        from_p = Point2(self.mposX, self.mposY)
        to_p   = Point2(x, y)
        from_t = max(0., min(1., project_param(from_p, self.ramp_A_p, self.ramp_B_p)))
        to_t   = max(0., min(1., project_param(to_p, self.ramp_A_p, self.ramp_B_p)))
        ramp_up_t = self.ramp_t()
        ramp_down_t = 1.0 - ramp_up_t

        # note: from_t to_t are not parameters of this line but of the ramp!, should be close enough though, let's see
        if from_t < ramp_up_t < to_t:
            intermediate_p = from_p * (1.-ramp_up_t) + to_p * ramp_up_t
            z = self.ramped_z(intermediate_p)
            self.moveTo(intermediate_p.x, intermediate_p.y, z, feed, rapid)
            #print "intermediate1", ramp_up_t, z,
        if from_t < ramp_down_t < to_t and ramp_down_t != ramp_up_t:
            intermediate_p = from_p * (1.-ramp_down_t) + to_p * ramp_down_t
            z = self.ramped_z(intermediate_p)
            self.moveTo(intermediate_p.x, intermediate_p.y, z, feed, rapid)
            #print "intermediate2", ramp_down_t, z,

        z = self.ramped_z(to_p)
        #print "done at z", z
        self.moveTo(to_p.x, to_p.y, z, feed, rapid)


    def moveTo(self, x=None, y=None, z=None, feed=None, rapid=False):
        """ flipped X/Y !"""

        if x is None:
            x = self.mposX
        if y is None:
            y = self.mposY
        if z is None:
            z = self.mposZ
        if feed is None:
            feed = self.feed

        if abs(x-self.mposX) <= self.epsilon and abs(y-self.mposY) <= self.epsilon and abs(z-self.mposZ) <= self.epsilon:
            return

        mv = 'G0' if rapid else 'G1'
        if not rapid and self.feed != feed:
            mv += ' F{:.1f}'.format(self.feedRate)
            self.feed = feed
        if abs(x-self.mposX) > self.epsilon:
            mv += ' Y{:.2f}'.format(x)
            self.mposX = x
        if abs(y-self.mposY) > self.epsilon:
            mv += ' X{:.2f}'.format(y)
            self.mposY = y
        if abs(z - self.mposZ) > self.epsilon:
            mv += ' Z{:.2f}'.format(z)
            self.mposZ = z

        self.program.append(mv)


    def arcTo_ramped(self, x, y, pivotx, pivoty, ccw=False):

        z = self.ramped_z(Point2(x, y))
        self.arcTo(x, y, z, pivotx, pivoty, ccw)

    def arcTo(self, x, y, z, pivotx, pivoty, ccw=False):
        """ flipped X/Y !"""

        if z is None:
            z = self.mposZ


        if Point2(x,y).dist(Point2(self.mposX, self.mposY)) < 0.5: # straight move instead
            mv = 'G1 Y{x:.2f} X{y:.2f}'.format(x=x, y=y)
        else:
            mv = 'G{move} Y{x:.2f} X{y:.2f} J{i:.2f} I{j:.2f}'.format(move=2 if ccw else 3, x=x, y=y, i=pivotx-self.mposX, j=pivoty-self.mposY)
        self.mposX = x
        self.mposY = y

        if abs(z-self.mposZ) > self.epsilon:
            mv += ' Z{:.2f}'.format(z)
            self.mposZ = z

        self.program.append(mv)




    def moveTo_withAvoid(self, x, y, minDist, ignoreNails=[]):
        """ move to x,y keeping distance to all nails on the way """

        minDist = float(minDist)
        p1 = Point2(self.mposX, self.mposY)
        p2 = Point2(x, y)

        # find all points too close to the line
        closePoints = self.pc.findPointsNearRay(p1, p2, minDist)
        closePoints = [cp for cp in closePoints if 0 < cp[0] < 1 and cp[1] not in ignoreNails]
        closePoints.sort()

        #print "collisions", closePoints
        for cp in closePoints:
            # insert an arc around the offending point
            circ = Circle2(self.pc.p[cp[1]].x, self.pc.p[cp[1]].y, minDist)
            intersects = circ.intersectRay(p1, p2)

            if len(intersects) == 2 and intersects[0].dist(intersects[1]) > 0.5:
                if (p1.dist(circ.p) - minDist) < -0.01:
                    print "----- MINDIST VIOLATION -----", p1.dist(circ.p), "<", minDist
                cw = (p2-p1).cross25D(circ.p-p1)
                self.moveTo_ramped(intersects[0].x, intersects[0].y)
                self.arcTo_ramped(intersects[1].x, intersects[1].y, circ.p.x, circ.p.y, ccw = cw >= 0)
                p1 = intersects[1]

        self.moveTo_ramped(p2.x, p2.y)


    def calc_num_intersects(self, from_c, to_c):

        inters = 0
        con = self.connects
        p4 = self.pc.p[con[to_c]]
        p3 = self.pc.p[con[to_c - 1]]
        h1 = 0
        h2 = 0
        maxh = self.startZ

        if to_c > from_c + 1:
            p1 = self.pc.p[con[from_c]]
            h1 = self.connectHeight[from_c]
            for i in xrange(from_c+1,to_c):
                p2 = self.pc.p[con[i]]
                h2 = self.connectHeight[i]
                s,t = intersect_line(p1,p2,p3,p4)
                if 0 < s < 1 and 0 < t < 1:
                    inters += 1
                    ih = h1 * (1-s) + h2 * s
                    maxh = max(maxh, ih)

                p1 = p2

        return inters, maxh

    def update_z_ramp(self, p1, z1, p2, z2):

        self.ramp_A_p = p1
        self.ramp_A_z = z1
        self.ramp_B_p = p2
        self.ramp_B_z = z2


    def ramp_t(self):
        """ ramp positions on the line. ramp up [0,t] ramp down [1.0-t, 1.0]"""

        # ramp angle
        tn = math.tan(self.ramp_angle/180.0*math.pi)
        ramp_len = self.z_hop / tn
        t = ramp_len / self.ramp_A_p.dist(self.ramp_B_p)
        return min(0.5, t)


    def ramped_z(self, p):

        # ramp angle
        tn = math.tan(self.ramp_angle/180.0*math.pi)

        # where are we on the ramp
        t = max(0., min(1., project_param(p, self.ramp_A_p, self.ramp_B_p)))

        if t <= 0.5: # ramping up
            ramp_height = min(self.z_hop, tn * p.dist(self.ramp_A_p))
        else: # ramping down
            ramp_height = min(self.z_hop, tn * p.dist(self.ramp_B_p))

        z = self.ramp_A_z * (1.0-t) + self.ramp_B_z * t + ramp_height
        return z


    def generateStringPath(self, name, startPosition, minNailDistance):



        #startPosition -= self.origin
        #startPosition.x *= self.scale.x
        #startPosition.y *= self.scale.y

        self.nailThreadHeights = [0.0] * len(self.pc.p)
        self.resetProgram()

        self.addStartup()
        self.moveTo(z=self.safeZ, rapid=True)
        self.moveTo(x=startPosition.x, y=startPosition.y, rapid=True)
        self.moveTo(z=self.startZ)
        self.addPause()


        percentDone = 0
        max_height = 0
        currentZ = self.startZ
        lastNail = -1
        string_length = 0
        current_loop_start = 0
        last_pause_at = 0

        for pathId, nailId in enumerate(self.connects[:-1]):
        #for i, nail in enumerate(connects[:20]):

            #move to current nail
            cur_pos  = self.pc.p[nailId]
            next_pos = self.pc.p[self.connects[pathId+1]]

            #z = currentZ
            # first check if this next segment intersects the loop
            num_i, max_h = self.calc_num_intersects(0, pathId)
            currentZ = max_h
            #if num_i > 0:
            #    #yep, so the string may raise up
            #    currentZ += self.threadThickness
                # start a new loop, because everything that came before will be below new strings
            #    current_loop_start = pathId
            # next check if the string on the nail is already at current height
            if self.nailThreadHeights[nailId] >= currentZ:
                #yep, nail already visited by this loop, raise it
                currentZ = self.nailThreadHeights[nailId] + self.threadThickness
            self.connectHeight[pathId] = currentZ
            self.nailThreadHeights[nailId] = currentZ

            if currentZ - last_pause_at > self.maxZ or pathId % 10 == 0:
                print "adding pause", currentZ, pathId
                self.addPause()
                last_pause_at = currentZ
            #currentZ = min(currentZ, self.maxZ)


            string_length += Point2(cur_pos.x, cur_pos.y).dist(Point2(self.mposX, self.mposY))

            # move to the circle around the current nail, depending on where the next nail is
            # calculate the tangent on the 'left' or 'right side of the circle
            v_from = Point2(cur_pos.x, cur_pos.y) - Point2(self.mposX, self.mposY)
            v_next = Point2(next_pos.x, next_pos.y) - Point2(cur_pos.x, cur_pos.y)
            circumference = Circle2(cur_pos.x, cur_pos.y, minNailDistance)

            # entry point
            norms = circumference.normals(v_from)
            tangs = circumference.tangentP(v_next)
            clock = v_from.cross25D(v_next)

            if clock >=0:
                p1 = circumference.p + norms[0]   # circle entry
                p2 = tangs[0]
            else:
                p1 = circumference.p + norms[1]
                p2 = tangs[1]

            self.update_z_ramp(Point2(self.mposX, self.mposY), self.mposZ, p1, currentZ)
            self.moveTo_withAvoid(x=p1.x, y=p1.y, minDist=minNailDistance, ignoreNails=[nailId, lastNail])
            self.arcTo(p2.x, p2.y, self.mposZ, cur_pos.x, cur_pos.y, ccw=clock >= 0)

            lastNail = nailId

            perc = int(float(pathId) / len(self.connects) * 100)
            if perc != percentDone:
                print "generating gcode: %d%%"%perc, "max_intersects", max_height
                percentDone = perc


        self.addEnd()

        self.addComment(" String: %d m "%int(string_length/1000), prepend=True)
        self.addComment(" Num connects: %d, maxNailVisits: %d " % (len(self.connects), max(self.nailThreadHeights)), prepend=True)
        self.addComment(name, prepend=True)

        return "\n".join(self.program)




    def generateDrillPattern(self, name, depth):

        pc = self.pc.copy()
        print "nail extend", pc.bbox()

        self.resetProgram()
        self.addStartup()
        self.moveTo(z=self.safeZ, rapid=True)
        self.spindleOn()
        self.dwell(4.0)

        pnts = list()
        current = Point2(0,0)
        while len(pc.p):
            np,d = pc.closestPoint(current.x, current.y)
            pnts.append(pc.p[np])
            current = pc.p[np]
            pc.remove(np)

        self.cannedDrillCycle(pnts, self.startZ, depth, self.drillFeed )

        self.addComment("num holes: %d"%len(pnts), prepend=True)
        self.addComment(name, prepend=True)

        return "\n".join(self.program)


