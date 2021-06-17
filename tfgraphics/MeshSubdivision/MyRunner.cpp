#include "MyRunner.h"

void MyRunner::run()
{
	string meshpath;

	Mesh *mesh = new Mesh();
	BaseScheme *scheme = getSchemeInput();
	getFilePathInput(&meshpath);
	cout << "meshpath:" << meshpath << endl;
	mesh->loadOff(meshpath, scheme->needsTriangulation());
	// mesh->loadOff(meshpath, false);
	Mesh *schemeMesh = scheme->run(mesh);
	size_t iterations = getIterationInput() - 1;

	for (size_t i = 0; i < iterations; i++)
		schemeMesh = scheme->run(schemeMesh);

	string outfilename;
	cout << "please enter the new .off file name (no need .off): ";
	cin >> outfilename;
	schemeMesh->saveOff(string("Meshes\\output\\") + outfilename + string(".off"));
}

void MyRunner::run_body_mesh()
{
	string inmeshpath[8], outmeshpath[8];
	int schenum;
	cout << "Enter a Subdivision Scheme:\n"
		 << "1: Loop\n2: Butterfly\n3: Doo-Sabin\n";
	cin >> schenum;
	size_t iterations = getIterationInput() - 1;
	for (int i = 0; i < 8; ++i)
	{
		inmeshpath[i] = string("Meshes/body/Dancer_test_sequence") + to_string(i) + string(".off");
		outmeshpath[i] = string("Meshes\\output\\Dancer_test_sequence") + to_string(i);
		true_run_body_mesh(inmeshpath[i], outmeshpath[i], iterations, schenum);
	}
}

void MyRunner::true_run_body_mesh(string infile, string outfile, size_t itnum, int schenum)
{
	// int schenum;
	string schestring[3] = {string("loop"), string("butterfly"), string("DooSabin")};
	Mesh *mesh = new Mesh();
	BaseScheme *scheme;
	switch (schenum)
	{
	case 1:
		scheme = new Loop();
		break;
	case 2:
		scheme = new Butterfly();
		break;
	case 3:
		scheme = new DooSabin();
		break;
	default:
		cerr << "Invalid Scheme" << endl;
		exit(0);
		break;
	}
	cout << "meshpath:" << infile << endl;
	mesh->loadOff(infile, false);
	cout << "Begin Running Iteration: " << 1 << endl;
	Mesh *schemeMesh = scheme->run(mesh);
	cout << "Iteration " << 1 << "finished, saving file..." << endl;
	schemeMesh->saveOff(outfile + string("_it") + to_string(0) + "_" + schestring[schenum - 1] + ".off");

	for (size_t i = 0; i < itnum; i++)
	{
		cout << "Begin Running Iteration: " << i + 2 << endl;
		schemeMesh = scheme->run(schemeMesh);
		cout << "Iteration " << i + 2 << "finished, saving file..." << endl;
		schemeMesh->saveOff(outfile + string("_it") + to_string(i + 1) + "_" + schestring[schenum - 1] + ".off");
	}
}

void MyRunner::getFilePathInput(string *meshpath)
{
	std::stringstream ss;
	int in;

	cout << "Enter a Mesh Category Number:\n1: Geometry\n2: Letters\n3: Numbers\n";
	cin >> in;
	ss << "Meshes/" << (in == 1 ? "Geometry" : (in == 2 ? "Letters" : "Numbers")) << "/";

	cout << "Enter a Mesh Number:\n";
	if (in == 1)
	{
		cout << "1: boxcube\n2: boxtorus\n3: cube\n4: helix2\n5: octtorus\n";
		cin >> in;

		if (in == 1)
			ss << "boxcube.off";
		else if (in == 2)
			ss << "boxtorus.off";
		else if (in == 3)
			ss << "cube.off";
		else if (in == 4)
			ss << "helix2.off";
		else if (in == 5)
			ss << "octtorus.off";
	}
	else if (in == 2)
	{
		cout << "1: D\n2: L\n3: M\n4: S\n5: T\n6: V\n7: W\n8: X\n";
		cin >> in;

		if (in == 1)
			ss << "D.off";
		else if (in == 2)
			ss << "L.off";
		else if (in == 3)
			ss << "M.off";
		else if (in == 4)
			ss << "S.off";
		else if (in == 5)
			ss << "T.off";
		else if (in == 6)
			ss << "V.off";
		else if (in == 7)
			ss << "W.off";
		else
			ss << "X.off";
	}
	else
	{
		cout << "1: 0\n2: 5\n3: 6\n4: 7\n5: 8\n6: 9\n";
		cin >> in;

		if (in == 1)
			ss << "0.off";
		else if (in == 2)
			ss << "5.off";
		else if (in == 3)
			ss << "6.off";
		else if (in == 4)
			ss << "7.off";
		else if (in == 5)
			ss << "8.off";
		else if (in == 5)
			ss << "9.off";
	}

	string s = ss.str();
	cout << "Opening: " << s << endl;
	*meshpath = s;
}

BaseScheme *MyRunner::getSchemeInput()
{
	int in;
	cout << "Enter a Subdivision Scheme:\n"
		 << "1: Catmull-Clark\n2: Loop\n3: Butterfly\n4: Doo-Sabin\n";
	cin >> in;

	if (in == 1)
		return new CatmullClark();
	else if (in == 2)
		return new Loop();
	else if (in == 3)
		return new Butterfly();
	else if (in == 4)
		return new DooSabin();
}

BaseScheme *MyRunner::getSchemeInput_body(int *sche)
{
	int in;
	cout << "Enter a Subdivision Scheme:\n"
		 << "1: Loop\n2: Butterfly\n3: Doo-Sabin\n";
	cin >> in;

	sche[0] = in;

	if (in == 1)
		return new Loop();
	else if (in == 2)
		return new Butterfly();
	else if (in == 3)
		return new DooSabin();
}

size_t MyRunner::getIterationInput()
{
	int in;
	cout << "Enter Scheme Iteration Count: ";
	cin >> in;
	return (size_t)in;
}
