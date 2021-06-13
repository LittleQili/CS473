#pragma once

#include "Point.h"
#include "Mesh.h"
#include "Loop.h"
#include "CatmullClark.h"
#include "DooSabin.h"
#include "PetersReiff.h"
#include "Butterfly.h"
#include "KobbeltRoot3.h"

#include <sstream>

class MyRunner
{
public:
	void run();
	void getFilePathInput(string*);
	BaseScheme* getSchemeInput();
	size_t getIterationInput();
};
