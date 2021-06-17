#pragma once

#include "MyRunner.h"


int main(int, char** argv)
{
	MyRunner tryrun;
	tryrun.run_body_mesh();

	system("pause");
	return 0;
}