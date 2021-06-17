#pragma once

#include "Point.h"
#include "Mesh.h"
#include "Loop.h"
#include "CatmullClark.h"
#include "DooSabin.h"
#include "Butterfly.h"

#include <sstream>

class MyRunner
{
public:
	void run();
	void run_body_mesh();
	void getFilePathInput(string *);
	BaseScheme *getSchemeInput();
	BaseScheme *getSchemeInput_body(int *);
	size_t getIterationInput();

private:
	void true_run_body_mesh(string, string, size_t, int);
};