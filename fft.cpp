// #define NDEBUG
#include<bits/stdc++.h>
#include "fffft.h"
typedef std::complex<double> Complex;
typedef std::valarray<Complex> CArray;
using namespace std;

int N;
vector<vector<string>> seqs;
Complex vals[256];

int highestPowerof2(int n) {
	int res = 0;
	for (int i = n; i >= 1; i--)
	{
		// If i is a power of 2
		if ((i & (i - 1)) == 0)
		{
			res = i;
			break;
		}
	}
	return res;
}

void setvals(Complex A, Complex C, Complex G, Complex T) {
	vals['A'] = A;
	vals['C'] = C;
	vals['G'] = G;
	vals['T'] = T;
}

CArray convert(const string &aa) {
	CArray ddd (highestPowerof2(aa.size() - 1) * 2);
	for (int i = 0; i < (int)aa.size(); i++) {
		ddd[i] = vals[aa[i]];
	}
	return ddd;
}

valarray<double> getMagnitudeArray(const string &aa) {
	CArray data = convert(aa);
	fft(data);
	valarray<double> mags(data.size());
	for (int i = 0; i < data.size(); i++) {
		mags[i] = abs(data[i]);
	}
	return mags;
}

void standardScale(valarray<double> &v) {
	double mean = v.sum() / v.size();
	v -= mean;

	double var = 0.0;
	std::for_each (std::begin(v), std::end(v), [&](const double d) {
		var += d * d;
	});

	double stdev = sqrt(var / v.size());

	v /= stdev;
}

int main() {
	setvals(1, 0, 0, 0);
	ifstream fi{"cleaned/1Primates.fasta"};
	fi >> N;
	int sizes[N];
	for (int i = 0; i < N; i++) {
		fi >> sizes[i];
	}
	for (int i = 0; i < N; i++) {
		seqs.emplace_back();
		for (int j = 0; j < sizes[i]; j++) {
			string a; fi >> a;
			seqs[i].push_back(a);
		}
	}

	ofstream fo{"features/2Primates.out"};
	fo << N << '\n';

	for (int i = 0; i < N; i++) {
		fo << seqs[i].size() << '\n';
		for(int j=0;j<seqs[i].size();j++){
			auto aaaa = getMagnitudeArray(seqs[i][j]);
			standardScale(aaaa);
			fo<<'[';
			for(int k=0;k<aaaa.size();k++){
				fo<<aaaa[k]<<',';
			}
			fo<<"],";
		}
	}

	// for (int i = 0; i < aaaa.size(); i++) {
	// 	cout << aaaa[i] << " ";
	// }
	// cout << endl;
	// cout << aaaa.sum() << endl;


	// CArray data = convert(seqs[0][0]);
	// // for (int i = 0; i < data.size(); ++i) {
	// // 	std::cout << (data[i]) << " ";
	// // }
	// // cout << endl;
	// // const Complex test[] = { 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0};
	// // data={test, 7};
	// // forward fft

	// // for(int i=0;i<1000;i++){
	// // 	CArray dada=data;
	// // 	fft(dada);
	// // }
	// // return 0;
	// fft(data);

	// // cout << "fft" << endl;
	// // for (int i = 0; i < data.size(); ++i) {
	// // 	cout << abs(data[i]) << " ";
	// // }
	// // cout << endl;

	// // inverse fft
	// ifft(data);

	// // cout << endl << "ifft" << endl;
	// for (int i = 0; i < data.size(); ++i) {
	// 	// cout << abs(data[i]) << " ";
	// 	bool ok = (abs(data[i]) > 0.5) == (i < seqs[0][0].size() && seqs[0][0][i] == 'A');
	// 	if (!ok) {
	// 		cout << i << endl;
	// 	}
	// 	assert(ok);
	// }
	// // cout << endl;
}