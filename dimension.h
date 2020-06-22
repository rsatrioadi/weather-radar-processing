#ifndef DIMENSION_H
#define DIMENSION_H

class Dimension3 {
  public:
    const int width, height, depth;
    int at_depth(int x, int y, int depth);
    Dimension3(int w, int h, int d);
};

class Dimension4 {
  public:
    const int width, height, copies, depth;
    int copy_at_depth(int x, int y, int copy, int depth);
    Dimension4(int w, int h, int c, int d);
};

#endif /* DIMENSION_H */
