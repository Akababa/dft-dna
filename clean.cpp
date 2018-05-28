#include<bits/stdc++.h>

using namespace std;

string getone(auto &fi) {
	string tt;
	string s;
	while (fi >> s) {
		if (s.size() == 0) continue;
		if (s[0] == '>') break;
		tt += s;
	}
	// cout<<tt<<endl;
	return tt;
}

string filename = "Protists.fasta";

int main() {
	ifstream fi;
	fi.open(filename);
	vector<string> fastas;
	while (fi) {
		auto aa = getone(fi);
		if (aa.size())
			fastas.push_back(aa);
	}
	ofstream fo{"1" + filename};
	bool lol[256]; memset(lol, 0, sizeof lol);
	for (auto& bb : fastas) {
		fo << bb << '\n';
		for (char c : bb) {
			lol[c] = true;
		}
	}
	for (int i = 0; i < 256; i++) {
		if (lol[i]) {
			bool ok= (i == 'A' || i == 'C' || i == 'G' || i == 'T');
			if(!ok){
				cout<<(char)i<<endl;
			}
			assert(ok);
		}
	}
	cout << fastas.size() << endl;
}