#pragma once

#include "MeshData.h"

#include <vector>
#include <fstream>
#include <iostream>
#include <string>
using namespace std;

const float PI_2_FLOAT = 6.2831853f;
const float PI_FLOAT = 3.1415927f;

class Mesh
{
public:
	vector<Face*> faces;
	vector<Edge*> edges;
	vector<Vertex*> verts;

	void loadOff(string path, bool quadToTri);
	void saveOff(string path);

	void addFace(size_t* vertices, size_t sideCount);
	void addVertex(const Point& p);

	void sortCounterClockwise();

protected:
	Edge* addEdge(Vertex* v1, Vertex* v2);
};