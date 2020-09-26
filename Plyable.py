#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 11:55:43 2020

@author: elizabethmeiklejohn
"""

"""Complete collection of Plyable functions as of 08.11.20. Does not include example code."""

import numpy as np
import trimesh
import math
import random
import time

"""TRANSFORMING MESHES"""

def pullVertex (mesh, vertex, dist, normals):
    """Moves the selected vertex the specified distance in its normal direction"""
    """Arg vertex refers to the index number of the vertex."""
    """Always specify normals as mesh.vertex_normals. This was added so the fn doesn't have to call it more than 1x."""
    a, b, c = normals[vertex,0], normals[vertex,1], normals[vertex,2]
    mesh.vertices[vertex] += np.array([a*dist, b*dist, c*dist])

def getDist (mesh, i, j):
    """Calculates distance between two mesh vertices as described by index numbers."""
    a,b,c, = mesh.vertices[i,0], mesh.vertices[i,1], mesh.vertices[i,2]
    d,e,f = mesh.vertices[j,0], mesh.vertices[j,1], mesh.vertices[j,2]
    dist = math.sqrt((a-d)**2 + (b-e)**2 + (c-f)**2)
    return dist

def scale (mesh, vertex, origin, x=1, y=1, z=1):
    """Scales a vertex in x, y and/or z relative to an origin."""
    """Modifies the original mesh in place."""
    """Vertex must be an index number in mesh.vertices"""
    """Origin must be a list [float, float float] or equivalent ie. mesh.vertices[n] or mesh.centroid"""
    if origin.all == 0:
        mesh.vertices[vertex] *= [x,y,z]
    else:
        a,b,c, = mesh.vertices[vertex,0], mesh.vertices[vertex,1], mesh.vertices[vertex,2]
        d,e,f = origin[0], origin[1], origin[2]
        mesh.vertices[vertex] = [x*(a-d)+d, y*(b-e)+e, z*(c-f)+f]
        
def randomOffset(mesh, axis, min, max):
    """Offsets each vertex in the specified direction by a random amount in the range (-max, max)."""
    """Input must be a trimesh, not a numpy array"""
    if 'x' in axis or 'X' in axis:
        for i in range(mesh.vertices.shape[0]):
            mesh.vertices[i,0] += random.uniform((float(min)), float(max))
    if 'y' in axis or 'Y' in axis:
        for i in range(mesh.vertices.shape[0]):
            mesh.vertices[i,1] += random.uniform((float(min)), float(max))
    if 'z' in axis or 'Z' in axis:
        for i in range(mesh.vertices.shape[0]):
            mesh.vertices[i,2] += random.uniform((float(min)), float(max))
    if 'normal' in axis:
        normals = mesh.vertex_normals
        for i in range(mesh.vertices.shape[0]):
            pullVertex(mesh, i, random.uniform((float(min)), float(max)), normals)      
    return mesh
        
def makeFalloffList (dist, thickness, shape):
    """Creates list of distances to move each point. Makes thickness value odd."""
    if thickness % 2 == 0:
        thickness -= 1
    xSteps = list(range(0, thickness))
    ySteps = []
    mid = thickness//2
    if shape == 'triangle':
        for x in xSteps:
            ySteps.append(dist * min(x, 2*mid-x)/mid)
    elif shape == 'circle':
        for x in xSteps:
            ySteps.append(dist/mid * math.sqrt(mid**2 - (x-mid)**2))
    elif shape == 'sine':
        for x in xSteps:
            ySteps.append(dist/2 * (math.sin((math.pi*x /mid) - (math.pi/2))+1))
    elif shape == 'peak':
        for x in xSteps:
            ySteps.append(-(dist/mid * math.sqrt(mid**2 - (min(x, 2*mid-x))**2))+dist)
    else:
        for x in xSteps:
            ySteps.append(dist)
    return ySteps

def neighborlyMove(mesh, dist, skip=1, thickness=1, falloff='flat'):
    """Starts with a random vertex in mesh, moves in z and then skips designated # of neighbors."""
    """Does not spread outwards from node (like a bullseye) but selects next pt at random."""
    startTime = time.perf_counter()
    """Falloff list is truncated to its second half because NM starts at the center, not the edge."""
    falloffList = makeFalloffList(dist, 2*thickness-1, falloff)[thickness-1:] 
    checked = np.zeros((mesh.vertices.shape[0],1))
    vertexList = np.hstack((mesh.vertices, checked))
    thicknessCounter = 0
    normalsList = mesh.vertex_normals
    while vertexList.sum(axis=0)[-1] < vertexList.shape[0]:
        """While not every vertex has been checked (value of 1 in "checked" column)"""
        start = []
        start.append(random.randint(0, mesh.vertices.shape[0]-1))
        """Start with 1 seed instead of a list of seeds - guarantees space between."""
        while thicknessCounter < thickness:
            toTry = []
            counter = 0
            for i in start:
                if vertexList[i,-1] == 0:
                    pullVertex(mesh, i, falloffList[thicknessCounter], normalsList)
                    vertexList[i,-1] += 1
                    for j in mesh.vertex_neighbors[i]:
                        if j not in toTry:
                            toTry.append(j)
            start = toTry
            thicknessCounter += 1
        while counter < skip:
            counter += 1
            thicknessCounter = 0
            nextTry = []
            for j in toTry:
                if vertexList[j,-1] == 0:
                    vertexList[j,-1] += 1
                    for k in mesh.vertex_neighbors[j]:
                        if k not in nextTry and vertexList[k,-1] == 0:
                            nextTry.append(k)
            toTry = nextTry
    endTime = time.perf_counter()
    print('Finished in ' + str(endTime - startTime) + ' seconds.')
    return mesh

def rippleMove(mesh, dist, skip, thickness=1, seeds=1, falloff='flat'):
    """Starts with a random vertex in mesh, moves along normal and then skips designated # of neighbors."""
    """Uses last set of skipped neighbors as next set of start points"""
    startTime = time.perf_counter()
    if thickness % 2 == 0:
        thickness -= 1
    falloffList = makeFalloffList(dist, thickness, falloff)    
    checked = np.zeros((mesh.vertices.shape[0],1))
    vertexList = np.hstack((mesh.vertices, checked))
    start = []
    thicknessCounter = 0
    normalsList = mesh.vertex_normals
    for i in range(seeds):
        start.append(random.randint(0, mesh.vertices.shape[0]-1))
    while vertexList.sum(axis=0)[-1] < vertexList.shape[0]:
        """While not every vertex has been checked (value of 1 in "checked" column)"""
        while thicknessCounter < thickness:
            toTry = []
            counter = 0
            for i in start:
                if vertexList[i,-1] == 0:
                    pullVertex(mesh, i, falloffList[thicknessCounter], normalsList)
                    vertexList[i,-1] += 1
                    for j in mesh.vertex_neighbors[i]:
                        if j not in toTry:
                            toTry.append(j)
            start = toTry
            thicknessCounter += 1
        while counter < skip:
            counter += 1
            thicknessCounter = 0
            nextTry = []
            for j in toTry:
                if vertexList[j,-1] == 0:
                    vertexList[j,-1] += 1
                    for k in mesh.vertex_neighbors[j]:
                        if k not in nextTry and vertexList[k,-1] == 0:
                            nextTry.append(k)
            toTry = nextTry
        start = toTry
    endTime = time.perf_counter()
    print('Finished in ' + str(endTime - startTime) + ' seconds.')
    return mesh

"""WRITING MESHES FROM SCRATCH"""

def makeVertexGrid(m,n):
    """Create a grid of vertices m wide and n high, all with z=0."""
    """Version from 05.04.20.py - EDITED based on .ai doc"""
    x = np.tile(np.arange(m), n)
    y = np.repeat(np.arange(n), m)
    z = np.zeros(n*m)
    grid = np.column_stack((x,y,z))
    return grid

def makeFaceGrid(m,n):
    """Create a list of faces (subdivided squares from bottom left to top right)""" 
    """for a grid a vertices wide and b vertices high, specified in XYZ order."""
    """Version from 05.18.20.py"""
    grid = np.zeros((2*(m-1)*(n-1), 3))
    counter = 0
    for i in range(m*(n-1)):
        if i%m == m-1:
            pass
        else:
            grid[counter] = [i,i+1,i+m+1]
            grid[counter+1] = [i,i+m+1,i+m]
            counter +=2
    return grid

def makePlane (m,n):
    """Create a plane of n*m vertices, all with z=0."""
    return trimesh.Trimesh(vertices=makeVertexGrid(m,n), faces=makeFaceGrid(m,n))

def makePrism (m,n,p):
    """Creates a rectangular prism of size m*n*p vertices in XYZ directions."""
    if m == n and n == p:
        """Cube alert. Create 3 1D arrays"""
        fast = np.tile(np.arange(n), n)
        slow = np.repeat(np.arange(n), n)
        still = np.zeros(n**2)
        """Define 6 vertex grids"""
        bottom = np.column_stack((fast, slow, still))
        top = bottom + np.array([0,0,n-1])
        front = np.column_stack((fast, still, slow))
        back= front + np.array([0, n-1, 0])
        left= np.column_stack((still, fast, slow))
        right=left + np.array([n-1, 0, 0])
    else:
        """Create 9 1D arrays"""
        XbyY = np.tile(np.arange(m), n)
        XbyZ = np.tile(np.arange(m), p)
        YbyZ = np.tile(np.arange(n), p)
        YbyX = np.repeat(np.arange(n), m)
        ZbyX = np.repeat(np.arange(p), m)
        ZbyY = np.repeat(np.arange(p), n)
        minX = np.zeros(n*p)
        minY = np.zeros(m*p)
        minZ = np.zeros(m*n)
        """Define 6 vertex grids"""
        bottom = np.column_stack((XbyY, YbyX, minZ))
        top = bottom + np.array([0,0,p-1])
        front = np.column_stack((XbyZ, minY, ZbyX))
        back = front + np.array([0,n-1, 0])
        left = np.column_stack((minX, YbyZ, ZbyY))
        right = left + np.array([m-1, 0,0])
    """Create triangular face lists for each side of the prism."""
    topFaces = makeFaceGrid(m,n)
    bottomFaces = topFaces.copy()[:,::-1]
    frontFaces = makeFaceGrid(m, p)
    backFaces = frontFaces.copy()[:,::-1]
    rightFaces = makeFaceGrid(n,p)
    leftFaces = rightFaces.copy()[:,::-1]
    """Add offset values to face lists"""
    topFaces += m*n
    frontFaces += 2*m*n
    backFaces += m*(2*n+p)
    leftFaces += 2*m*(n+p)
    rightFaces += 2*m*(n+p) + n*p
    """Stack vertex lists. Order is important for face definition!"""
    vertexList = np.vstack((bottom, top, front, back, left, right))
    """Stack face lists. Keep same order as vertexList and appropriate offsets!"""
    faceList = np.vstack((bottomFaces, topFaces, frontFaces, backFaces, leftFaces, rightFaces))
    return trimesh.Trimesh(vertices=vertexList, faces=faceList)

def makeSphere(n):
    cube = makePrism(n,n,n)
    sphere = cube.copy()
    origin = cube.centroid
    r = (n-1)/2
    """This is equal to the radius of the sphere, and used in below equations"""
    o = 1.853
    m = 2.15
    h = math.sqrt(r**2+(m-o)**2)
    """Hypotenuse of the right triangle of width r and height=the jump from o to m"""
    q = h**2/(2*(m-o))
    """Radius of the circle containing (0,o) and (r,m)"""
    coeffList = []
    scaleArray = np.zeros(((n+1)//2,(n+1)//2))
    """Dims of scaleArray are # of rings x # of steps in each ring"""
    for i in range((n+1)//2):
        """Generate a list of coefficients from 1.853 to 2.804"""
        coeffList.append(-1*math.sqrt(q**2-i**2)+q+o)
        """Arc between (0,o) and (r,m)"""
    for i in range((n+1)//2):
        for j in range((n+1)//2):
            """Populate scaleArray"""
            coeff = coeffList[i]
            scaleArray[i,j] = (math.sqrt((coeff*r)**2-j**2)-r*(coeff-1))/r
    for i in range(cube.vertices.shape[0]):
        x = abs(int(origin[0]-cube.vertices[i,0]))
        """which x-ring does this vertex belong to?"""
        y = abs(int(origin[1]-cube.vertices[i,1]))
        """which y-ring does this vertex belong to?"""
        z = abs(int(origin[2]-cube.vertices[i,2]))
        """which z- ring does this vertex belong to?"""
        minRing = min(x,y,z)
        a = scaleArray[minRing, x]
        b = scaleArray[minRing, y]
        c = scaleArray[minRing, z]
        scale(sphere, i, origin, b*c, a*c, a*b)
    return sphere

"""TRANSFORMING MESHES BASED ON COLOR"""

def textureToColor (mesh):
    """Applies vertex colors to an newly created Trimesh based on imported textured mesh."""
    """Troubleshoot by checking texturedMesh.visual.uv and texturedMesh.visual.to_color().kind"""
    """If results of above are False and None, import textured mesh to Meshmixer (or similar) and re-export."""
    mesh2 = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
    colorList=mesh.visual.to_color().vertex_colors
    try:
        mesh2.visual.vertex_colors[:] = colorList[:]
        """Exception will be raised if mesh2 has fewer vertices than mesh (common due to merging of duplicates)"""
    except ValueError:
        counter=0
        for i in range(mesh2.vertices.shape[0]):
            """If vertices are not identical, it means the vertex in mesh is a dup that was removed. Try next vertex in mesh."""
            while not np.array_equal(mesh2.vertices[i], mesh.vertices[i+counter]):
                counter+=1
            """Each vertex receives the color of its match in mesh2 (which is offset by the # of dups found thus far.)"""
            mesh2.visual.vertex_colors[i] = colorList[i+counter]
    return mesh2

def RGBAtoCMYK(mesh):
    """Converts a mesh from RGBA to CMYK. This is a commonly accepted conversion formula."""
    """Assumes input is a Trimesh with vertex color list intact."""
    """Returns a mesh with CMYK values replacing RGBA values in mesh.visual.vertex_color list."""
    for i in mesh.visual.vertex_colors:
        r,g,b = i[0:3]
        r1,g1,b1 = r/255, g/255, b/255
        k = 1- max(r1, g1, b1)
        if k == 1:
            c = 0
            m = 0
            y = 0
        else:
            c = (1-r1-k)/(1-k)
            m = (1-g1-k)/(1-k)
            y = (1-b1-k)/(1-k)
        i[:]=100*c,100*m,100*y,100*k
        """color array type is uint8 - cmyk vals are expressed from 0-100."""
    return mesh

def CMYKtoCMYKW(mesh):
    """Converts a mesh from CMYK to CMYKW. This formula was written for Plyable."""
    """c+m+y+k+w = 100."""
    column={'c':0,'m':1,'y':2,'k':3}
    """Where to place color values in np array, based on which color they belong to."""
    for i in mesh.visual.vertex_colors:
        cmyk=[('c',i[0]),('m',i[1]),('y',i[2]),('k',i[3])]
        cmyk.sort(key = lambda pair: pair[1], reverse=True)
        np4 = cmyk[3][1]
        """how much 4color NP coverage is there? Equal to coverage of least prevalent color."""
        np3 = cmyk[2][1] - cmyk[3][1]
        np2 = cmyk[1][1] - cmyk[2][1]
        np1 = cmyk[0][1] - cmyk[1][1]
        
        i[column[cmyk[0][0]]]=np4/4 + np3/3 + np2/2 + np1
        """using the name of most-prevalent color, place its value in correct column of np array."""
        i[column[cmyk[1][0]]]=np4/4 + np3/3 + np2/2
        i[column[cmyk[2][0]]]=np4/4 + np3/3
        i[column[cmyk[3][0]]]=np4/4
        """w value is not stored but is equal to 1-i[0]-i[1]-i[2]-i[3]"""
    return mesh

def colorReduce(mesh, color1, color2, color3=None, color4=None):
    """Note: mess with this fn to create interesting effects. W was glitching because of 1-sum instead of 100-sum, K is TBD."""
    """Converts a mesh from CMYKW to 2-4 colors. Only possible 4color combo on Objet is CMY'K'."""
    """Order of colors specified doesn't matter!"""
    """Zeroes out colors not specified, and does not reallocate their coverage to other colors."""
    if ((color1=='w' and color2=='k') or (color1=='k' and color2=='w')) and color3==None and color4==None:
        """Exception for converting color to grayscale in a value-preserving way."""
        coeff = 1
        """May need to decrease this value, to compensate for black being darker than other pigments."""
        for i in mesh.visual.vertex_colors:
            k = np.sum(i)*coeff
            i[:] = 0,0,0,k
    else:
        column={'c':0,'m':1,'y':2,'k':3, 'w':4}
        palette=[]
        """Palette is a list of column numbers for each specified color (since these are not given in any order)."""
        for h in color1, color2, color3, color4:
            try:
                palette.append(column[h])
            except KeyError:
                """error will be raised if any color is None"""
                pass
        for i in mesh.visual.vertex_colors:
            total=0
            """this will be used to scale the specified colors so that they add up to 100."""
            for j in palette:
                try:
                    total += i[j]
                except IndexError:
                    """error will be raised if any color is 'w'"""
                    w = 100-np.sum(i)
                    """can adjust 100 to other numbers to create color error"""
                    total += w
            if total==0:
                """If this vertex doesn't include any of the specified colors, assign it the average of all specified colors"""
                for k in palette:
                    i[k] += 1
                    total += 1
                if 4 in palette:
                    """if w is in palette - ensure that listed vals DON'T add up to 100, ie. white is present!"""
                    total += 1
            for l in range(4):
                if l not in palette:
                    i[l] = 0
                    """Zeroes out values for colors not specified."""
            i[:] = 100*i[0]/total, 100*i[1]/total, 100*i[2]/total, 100*i[3]/total
    return mesh
    
def CMYKWtoCMYK(mesh):
    """converts a mesh back to real CMYK."""
    column={'c':0,'m':1,'y':2,'k':3, 'w':4}
    for i in mesh.visual.vertex_colors:
        cmyk=[('c',i[0]),('m',i[1]),('y',i[2]),('k',i[3])]
        cmyk.sort(key = lambda pair: pair[1], reverse=True)
        """Sorting color name/val pairs by prevalence, highest first."""
        color4 = 4*cmyk[3][1]
        """How many "squares" does the least prevalent color cover? In CMYK world we count these as wholes instead of portions."""
        """We know that the least prevalent color only appears in NP4 areas, hence *4."""
        color3 = (cmyk[2][1]-cmyk[3][1])*3 + color4
        """Color3 has all the coverage of color4 plus the NP3 areas in which it appears."""
        color2 = (cmyk[1][1]-cmyk[2][1])*2 + color3
        """Color2 has all the coverage of color3  plus the NP2 areas in which it appears."""
        color1 = (cmyk[0][1]-cmyk[1][1]) + color2
        """Color1 has all the overage of color2 plus the NP1 areas in which it appears."""
        i[column[cmyk[0][0]]]=color1
        """using the name of most-prevalent color, place its value in correct column of np array."""
        i[column[cmyk[1][0]]]=color2
        i[column[cmyk[2][0]]]=color3
        i[column[cmyk[3][0]]]=color4
    return mesh

def CMYKtoRGBA(mesh):
    """converts a mesh back to RGBA, ready for export."""
    for i in mesh.visual.vertex_colors:
        c,m,y,k = i[:]
        r = 255 * (1-c/100) * (1-k/100)
        g = 255 * (1-m/100) * (1-k/100)
        b = 255 * (1-y/100) * (1-k/100)
        i[:] = r,g,b,255
    return mesh

def multiColor (mesh, maxDist, color1, color2, color3=None, color4=None):
    """Creates 2-4 mesh bodies to be printed as an assembly."""
    """Colors can be 'c','m','y','k' or'w'."""
    """maxDist is in mm, 2-3mm is optimal for max opacity on Objet. Testing TBD"""
    reColor = {'c':[0,255,255,255], 'm':[255,0,255,255], 'y':[255,255,0,255], 'k':[0,0,0,255], 'w':[255,255,255,255]}
    """For recoloring export mesh bodies."""
    reducedMesh = colorReduce(CMYKtoCMYKW(RGBAtoCMYK(mesh)), color1, color2, color3, color4)
    """Conversion: RGBA>CMYK>CMYKW>reduced palette."""
    column={'c':0,'m':1,'y':2,'k':3, 'w':4}
    """lookup color value in np array based on name"""
    stack = {'w':1,'y':2,'m':3,'c':4,'k':5, None:6}
    """order that colors should be stacked in for best visuals. use this to rearrange input colors in case they are specified out of order."""
    colors=[color1, color2, color3, color4]
    colors.sort(key = lambda color: stack[color])
    """Creates list of inputs sorted according to stack list"""
    outer = reducedMesh.copy()
    outer.visual.vertex_colors[:]=reColor[colors[0]]
    shells=[outer]
    """add each subsequent shell (Trimesh) to this list"""
    normals=mesh.vertex_normals
    vertex_colors=reducedMesh.visual.vertex_colors
    size=mesh.vertices.shape[0]
    """shorthand for # of vertices"""
    innerFaces=mesh.faces[:,::-1] + size
    for i in range(3):
        """don't need to do offset based on last color, it will just be a solid body"""
        if colors[i+1]==None:
            pass
        else:
            thisMesh=shells[-1].copy()
            for j in range(size):
                try:
                    scale = vertex_colors[j,column[colors[i]]]
                except IndexError:
                    """error will be raised if any color is 'w'"""
                    scale = 100 - np.sum(vertex_colors[j])
                    """W is equal to 100 minus all other color values for that vertex"""
                pullVertex(thisMesh, j, -1*maxDist*(scale/100 + .01), normals)
                """pullVertex modifies in-place, so thisMesh is now changed"""
                thisMesh.visual.vertex_colors[j] = reColor[colors[i+1]]
                """only needed if exporting shell"""
            shells.append(thisMesh)
    for i in range(len(shells)):
        shells[i].export('shell' + str(i+1) + str(colors[i] + '.ply'))   
        try:
            body = trimesh.Trimesh(vertices = np.vstack((shells[i].vertices, shells[i+1].vertices)), faces = np.vstack((mesh.faces, innerFaces)))
        except IndexError:
            """raised when i = last item in shells"""
            body = shells[i]
        body.visual.vertex_colors[:] = reColor[colors[i]]
        body.export('body' + str(i+1) + str(colors[i] + str(maxDist) + 'dist.ply'))   
    render = CMYKtoRGBA(CMYKWtoCMYK(reducedMesh))
    render.export('render' + str(colors[0]) + str(colors[1]) + str(colors[2]) + str(colors[3]) + '.ply')