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
	void getFilePathInput(string*);
	BaseScheme* getSchemeInput();
	size_t getIterationInput();
};
