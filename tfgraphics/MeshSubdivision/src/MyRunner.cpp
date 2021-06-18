#include "MyRunner.h"

void MyRunner::run_body_mesh()
{
	string inmeshpath[8], outmeshpath[8];
	int schenum;
	cout << "Enter a Subdivision Scheme:\n"
		 << "1: Butterfly\n2: Doo-Sabin\n";
	cin >> schenum;
	size_t iterations = getIterationInput() - 1;
	for (int i = 0; i < 8; ++i)
	{
		inmeshpath[i] = string("Meshes/body/Dancer_test_sequence") + to_string(i) + string(".off");
		outmeshpath[i] = string("Meshes\\output\\Dancer_test_sequence") + to_string(i);
		// true_run_body_mesh(inmeshpath[i], outmeshpath[i], iterations, schenum);
	}
	true_run_body_mesh(inmeshpath[1], outmeshpath[1], iterations, schenum);
}

void MyRunner::true_run_body_mesh(string infile, string outfile, size_t itnum, int schenum)
{
	// int schenum;
	string schestring[2] = {string("butterfly"), string("DooSabin")};
	Mesh *mesh = new Mesh();
	BaseScheme *scheme;
	switch (schenum)
	{
	case 1:
		scheme = new Butterfly();
		break;
	case 2:
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

size_t MyRunner::getIterationInput()
{
	int in;
	cout << "Enter Scheme Iteration Count: ";
	cin >> in;
	return (size_t)in;
}
