#pragma once

#include <Matrix.hh>
#include <utility>
#include <vector>

std::vector<std::pair<int, NumbaMatrix>> loadLabeledImages(char const *imagesPath,
                                         char const *labelsPath);
