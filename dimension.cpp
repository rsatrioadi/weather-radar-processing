#include "dimension.h"

Dimension3::Dimension3(int w, int h, int d):
  width(w), height(h), depth(d) {

}

int Dimension3::at_depth(int x, int y, int depth){
  return y * width + x + depth*width*height;
}

Dimension4::Dimension4(int w, int h, int c, int d):
  width(w), height(h), copies(c), depth(d) {

}

int Dimension4::copy_at_depth(int x, int y, int copy, int depth){
  return y * width + x + copy*width*height + depth*width*height*copies;
}
