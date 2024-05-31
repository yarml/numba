#include <arpa/inet.h>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <numba.hh>

using namespace std;

vector<pair<int, NumbaMatrix>> loadLabeledImages(char const *imagesPath,
                                                 char const *labelsPath) {
  vector<pair<int, NumbaMatrix>> labeledImages;
  ifstream images(imagesPath, ios::binary);
  ifstream labels(labelsPath, ios::binary);
  assert(images.is_open());
  assert(labels.is_open());

  uint32_t imagesMagic, labelsMagic;

  images.read((char *)&imagesMagic, sizeof(imagesMagic));
  labels.read((char *)&labelsMagic, sizeof(labelsMagic));

  imagesMagic = ntohl(imagesMagic);
  labelsMagic = ntohl(labelsMagic);

  assert(imagesMagic == 0x803);
  assert(labelsMagic == 0x801);

  uint32_t nImages1, nImages2;

  images.read((char *)&nImages1, sizeof(nImages1));
  labels.read((char *)&nImages2, sizeof(nImages2));

  nImages1 = ntohl(nImages1);
  nImages2 = ntohl(nImages2);

  assert(nImages1 == nImages2);

  uint32_t nImages = nImages1;

  uint32_t nRows, nCols;

  images.read((char *)&nRows, sizeof(nRows));
  images.read((char *)&nCols, sizeof(nCols));

  nRows = ntohl(nRows);
  nCols = ntohl(nCols);

  assert(nRows == 28 && nCols == 28);

  for (size_t i = 0; i < nImages; ++i) {
    NumbaMatrix image(nRows, nCols);
    for (size_t j = 0; j < nRows; ++j) {
      for (size_t k = 0; k < nCols; ++k) {
        uint8_t pixel;
        images.read((char *)&pixel, sizeof(pixel));
        image(k, j) = pixel / 255.0;
      }
    }
    uint8_t label;
    labels.read((char *)&label, sizeof(label));
    labeledImages.push_back(make_pair(label, move(image)));
  }

  return move(labeledImages);
}