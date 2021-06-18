#pragma once

#include "Point.h"
#include "Mesh.h"
#include "DooSabin.h"
#include "Butterfly.h"

#include <sstream>

class MyRunner
{
public:
	void run_body_mesh();
	size_t getIterationInput();

private:
	void true_run_body_mesh(string, string, size_t, int);
};