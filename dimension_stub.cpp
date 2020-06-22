#include <iostream>
#include "dimension.h"

using namespace std;

void test_dimension3() {
  int w=5,h=4,d=3;
  Dimension3 d3(w,h,d);
  for (int k=0; k<d; k++) {
    for (int j=0; j<h; j++) {
      for (int i=0; i<w; i++) {
        cout << d3.at_depth(i,j,k) << " ";
      }
      cout << endl;
    }
    cout << endl;
  }
}

void test_dimension4() {
  int w=5,h=4,d=3,c=3;
  Dimension4 d4(w,h,c,d);
  for (int k=0; k<d; k++) {
    for (int j=0; j<h; j++) {
      for (int i=0; i<w; i++) {
        cout << d4.copy_at_depth(i,j,k,0) << " ";
      }
      cout << endl;
    }
    cout << endl;
  }
}

int main() {
  test_dimension4();
}
